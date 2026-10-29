"""Resumable, authenticated shard transfer over the existing Mummy/aiortc channel.

The receiver requests only hashes it does not already possess. Completed shard
files survive a disconnected WebRTC session; partial .part files do not. There
is a bounded in-flight request window and at most one streaming incoming shard.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path

from .core import PatchError
from .p2p import CHUNK, WINDOW, PatchNode
from .shards import (MAX_SHARD_MANIFEST, _blob_path, _check_hash, _check_object,
                     _atomic_write, has_object, manifest_bytes, parse_manifest,
                     read_manifest, shard_map, verify_shards)

log = logging.getLogger('fedpara.shards')
V = 2
CONTROL_LIMIT = 8192
REQUEST_WINDOW = 4


def ctrl(kind, **kwargs) -> str:
    result = json.dumps({'v':V,'type':kind,**kwargs},separators=(',',':'))
    if len(result) > CONTROL_LIMIT:
        raise PatchError('oversized shard control message')
    return result


async def _wait_buffer(channel, *, stall_seconds: float = 45.0):
    """Use common event-driven SCTP backpressure on both shard and patch streams."""
    from .p2p import wait_send_window
    return await wait_send_window(channel,stall_seconds=stall_seconds)


async def send_manifest(channel, payload: bytes, signer=None, on_progress=None):
    if not 0 < len(payload) <= MAX_SHARD_MANIFEST:
        raise PatchError('bad outgoing manifest')
    digest = hashlib.sha256(payload).hexdigest()
    start = {'size':len(payload),'sha256':digest}
    if signer is not None:
        start['sig'] = signer(len(payload),digest)
    channel.send(ctrl('manifest-start',**start))
    for off in range(0,len(payload),CHUNK):
        await _wait_buffer(channel)
        channel.send(payload[off:off+CHUNK])
        if on_progress: on_progress()
    await _wait_buffer(channel)
    channel.send(ctrl('manifest-end',sha256=digest))


async def send_shard(channel, store: Path, digest: str, size: int, on_progress=None):
    path = _blob_path(store,digest)
    # Detect poisoned/corrupt local store before emitting even a header.
    # Hashing a 64 MiB shard can block SCTP keepalives if executed on the
    # asyncio thread. Authentication/integrity stays mandatory, but runs offloop.
    if not await asyncio.to_thread(_check_object,path,digest,size):
        raise PatchError('requested local shard absent/corrupted')
    channel.send(ctrl('shard-start',sha256=digest,size=size))
    with path.open('rb') as f:
        while data := f.read(CHUNK):
            await _wait_buffer(channel)
            channel.send(data)
            if on_progress: on_progress()
    await _wait_buffer(channel)
    channel.send(ctrl('shard-end',sha256=digest))


class ShardSession:
    """Synchronous ordered DataChannel receiver state machine; fake-channel testable."""
    def __init__(self, store: Path, recv_dir: Path, *, local_manifest: Path | None = None,
                 max_state_bytes: int = 32*1024**3, verifier=None,
                 require_model_match: bool = True, request_remote: bool = True,
                 defer_verify: bool = False):
        self.store, self.recv_dir = Path(store),Path(recv_dir)
        self.defer_verify = bool(defer_verify)
        self.max_state_bytes,self.verifier = max_state_bytes,verifier
        self.require_model_match = require_model_match
        self.request_remote = request_remote
        self.local_data = (Path(local_manifest).read_bytes() if local_manifest else None)
        self.local_meta = (parse_manifest(self.local_data,max_state_bytes=max_state_bytes)
                           if self.local_data is not None else None)
        self.local_hash = hashlib.sha256(self.local_data).hexdigest() if self.local_data else None
        self.local_objects = shard_map(self.local_meta) if self.local_meta else {}
        self.remote_meta = None
        self.remote_hash = None
        self.remote_data = None
        self.received_manifest: Path | None = None
        self.remote_iterator = None
        self.outstanding: dict[str,int] = {}
        self.remote_complete = False
        self.acked_local = False
        self.reading = None
        self.head = None
        self.raw = None
        self.file = None
        self.part_path = None
        self.sha = None
        self.written = 0
        self.served = set()
        self.requested_shards: list[str] = []
        self.requested_payload_bytes = 0
        self.received_payload_bytes = 0
        self.reused_objects = 0
        self.received_objects = 0
        self.recv_dir.mkdir(parents=True,exist_ok=True)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.part_path is not None:
            self.part_path.unlink(missing_ok=True)
            self.part_path = None
        self.reading = None

    def _fill_requests(self):
        if self.remote_iterator is None:
            return self._finish_if_ready()
        missing = []
        while len(self.outstanding) < REQUEST_WINDOW:
            try:
                h, size = next(self.remote_iterator)
            except StopIteration:
                self.remote_iterator = None
                break
            if has_object(self.store,h,size):
                self.reused_objects += 1
                continue
            self.outstanding[h] = size
            self.requested_payload_bytes += size
            missing.append(h)
        if missing:
            return [ctrl('need',manifest=self.remote_hash,hashes=missing)]
        return self._finish_if_ready()

    def _finish_if_ready(self):
        if self.remote_meta is None or self.outstanding or self.remote_iterator is not None:
            return []
        if not self.remote_complete:
            # LiveSnapshotLink runs the final hash verification in a worker
            # thread: multi-GB synchronous hashing would stall all aiortc
            # callbacks, keepalives, and peer event processing.
            if self.defer_verify:
                return []
            verify_shards(self.remote_meta,self.store)
            path = self.recv_dir / f'{self.remote_hash}.fmanifest'
            _atomic_write(path,self.remote_data)
            self.received_manifest = path
            self.remote_complete = True
        return [ctrl('complete',manifest=self.remote_hash)]

    def feed(self, message: bytes | str) -> tuple[list[str],list[str]]:
        """Returns (control replies to send immediately, shard hashes to send)."""
        if isinstance(message,bytes):
            if self.reading is None or not 0 < len(message) <= CHUNK:
                raise PatchError('unsolicited shard binary data')
            self.written += len(message)
            if self.written > self.head['size']:
                raise PatchError('incoming transfer larger than declared')
            self.sha.update(message)
            if self.reading == 'manifest':
                self.raw.extend(message)
            else:
                self.file.write(message)
            return [],[]
        if not isinstance(message,str) or len(message) > CONTROL_LIMIT:
            raise PatchError('bad shard control frame')
        try:
            msg = json.loads(message)
        except ValueError as exc:
            raise PatchError('invalid control JSON') from exc
        if not isinstance(msg,dict) or type(msg.get('v')) is not int or msg['v'] != V:
            raise PatchError('incorrect shard protocol version')
        kind = msg.get('type')
        if kind == 'manifest-start':
            if self.reading is not None or self.remote_meta is not None:
                raise PatchError('duplicate/overlapping manifest')
            size, digest = msg.get('size'),_check_hash(msg.get('sha256'))
            if type(size) is not int or not 0 < size <= MAX_SHARD_MANIFEST:
                raise PatchError('oversized manifest')
            if self.verifier is not None:
                self.verifier(msg)   # authenticate manifest bytes BEFORE trusting tensor metadata
            self.head = {'size':size,'sha256':digest}
            self.reading = 'manifest'
            self.raw = bytearray()
            self.written = 0
            self.sha = hashlib.sha256()
            return [],[]
        if kind == 'manifest-end':
            if self.reading != 'manifest' or msg.get('sha256') != self.head['sha256']:
                raise PatchError('invalid manifest end')
            if self.written != self.head['size'] or self.sha.hexdigest() != self.head['sha256']:
                raise PatchError('manifest hash/size mismatch')
            self.remote_data = bytes(self.raw)
            self.remote_meta = parse_manifest(self.remote_data,max_state_bytes=self.max_state_bytes)
            self.remote_hash = self.sha.hexdigest()
            if self.local_meta is not None and self.require_model_match:
                for key in ('model_id','revision'):
                    if self.local_meta[key] != self.remote_meta[key]:
                        raise PatchError('incompatible peer state: '+key)
                if self.local_meta['peer_id'] == self.remote_meta['peer_id']:
                    raise PatchError('same peer_id on both P2P endpoints')
            self.reading = None
            self.raw = None
            self.head = None
            if not self.request_remote:
                # A send-only node may inspect the peer announcement but must
                # never fetch remote model bytes or falsely ACK receipt.
                return [],[]
            self.remote_iterator = iter(shard_map(self.remote_meta).items())
            return self._fill_requests(),[]
        if kind == 'need':
            if self.local_meta is None or msg.get('manifest') != self.local_hash:
                raise PatchError('request for unknown local manifest')
            hashes = msg.get('hashes')
            if not isinstance(hashes,list) or not 1 <= len(hashes) <= REQUEST_WINDOW:
                raise PatchError('invalid shard request window')
            if len(set(hashes)) != len(hashes):
                raise PatchError('duplicate hashes in shard request')
            for h in hashes:
                if not isinstance(h,str) or h not in self.local_objects:
                    raise PatchError('request for foreign shard')
                if h in self.served:
                    raise PatchError('duplicate shard transfer request')
            self.served.update(hashes)
            self.requested_shards.extend(hashes)
            return [],hashes
        if kind == 'complete':
            if self.local_hash is None or msg.get('manifest') != self.local_hash:
                raise PatchError('unexpected shard completion ACK')
            self.acked_local = True
            return [],[]
        if kind == 'shard-start':
            if self.reading is not None or self.remote_meta is None or self.remote_complete:
                raise PatchError('unexpected shard start')
            digest, size = _check_hash(msg.get('sha256')),msg.get('size')
            if digest not in self.outstanding or type(size) is not int or self.outstanding[digest] != size:
                raise PatchError('unrequested shard or incorrect length')
            directory = _blob_path(self.store,digest).parent
            directory.mkdir(parents=True,exist_ok=True)
            fd, filename = tempfile.mkstemp(prefix='.incoming-',suffix='.part',dir=directory)
            self.part_path = Path(filename)
            self.file = os.fdopen(fd,'wb')
            self.reading = 'shard'
            self.head = {'size':size,'sha256':digest}
            self.written = 0
            self.sha = hashlib.sha256()
            return [],[]
        if kind == 'shard-end':
            if self.reading != 'shard' or msg.get('sha256') != self.head['sha256']:
                raise PatchError('invalid shard end')
            if self.written != self.head['size'] or self.sha.hexdigest() != self.head['sha256']:
                raise PatchError('corrupted/incomplete incoming shard')
            self.file.close()
            self.file = None
            dest = _blob_path(self.store,self.head['sha256'])
            os.replace(self.part_path,dest)
            self.part_path = None
            del self.outstanding[self.head['sha256']]
            self.received_objects += 1
            self.received_payload_bytes += self.written
            self.reading = self.head = None
            self.sha = None
            return self._fill_requests(),[]
        raise PatchError('unrecognized shard control frame')


class ShardNode(PatchNode):
    def __init__(self, signal_url: str, room: str, token: str, *,
                 store: Path, recv_dir: Path, local_manifest: Path | None = None,
                 max_state_bytes: int = 32*1024**3, signer=None, verifier=None,
                 request_remote: bool = True):
        super().__init__(signal_url,room,token,max_bytes=max_state_bytes,
                         signer=signer,verifier=verifier)
        self.session = ShardSession(store,recv_dir,local_manifest=local_manifest,
                                    max_state_bytes=max_state_bytes,verifier=verifier,
                                    request_remote=request_remote)
        self.requests = asyncio.Queue(maxsize=REQUEST_WINDOW * 2)
        self.local_acked = asyncio.Event()
        self.failed = asyncio.Event()

    def _attach_channel(self,ch):
        self._channel = ch

        @ch.on('open')
        def on_open():
            self.connected.set()
            self._send_task = asyncio.create_task(self._sender())
            def on_sender_done(t):
                if not t.cancelled():
                    exc = t.exception()
                    if exc is not None:
                        self.last_error = exc
                        self.failed.set()
                        ch.close()
            self._send_task.add_done_callback(on_sender_done)

        @ch.on('close')
        def on_close():
            self.connected.clear()
            self.session.close()
            if not self.failed.is_set() and not self.stopped.is_set():
                self.last_error = ConnectionError('P2P channel closed before shard exchange completed')
                self.failed.set()
            if self._send_task:
                self._send_task.cancel()

        @ch.on('message')
        def on_message(msg):
            try:
                replies, asks = self.session.feed(msg)
                for reply in replies:
                    ch.send(reply)
                for digest in asks:
                    self.requests.put_nowait(digest)
                if self.session.received_manifest is not None and not self.incoming.full():
                    self.incoming.put_nowait(self.session.received_manifest)
                    # Ensure exactly one inbound completion event.
                    self.session.received_manifest = None
                if self.session.acked_local:
                    self.local_acked.set()
            except (PatchError, asyncio.QueueFull, OSError, ValueError) as exc:
                self.last_error = exc
                self.session.close()
                log.error('Shard transfer rejected: %s',exc)
                self.failed.set()
                ch.close()

    async def _sender(self):
        if self.session.local_data is not None:
            await send_manifest(self._channel,self.session.local_data,self.signer)
        while True:
            digest = await self.requests.get()
            await send_shard(self._channel,self.session.store,digest,
                             self.session.local_objects[digest])


async def exchange_shards(node: ShardNode, *, receive: bool = True,
                          timeout: float = 3600) -> Path | None:
    """One connection, optionally send and/or receive a sharded model manifest."""
    if node.session.local_data is None and not receive:
        raise ValueError('exchange would neither send nor receive')
    task = asyncio.create_task(node.run())
    try:
        futures = []
        if receive:
            futures.append(asyncio.create_task(node.incoming.get()))
        if node.session.local_data is not None:
            futures.append(asyncio.create_task(node.local_acked.wait()))
        all_jobs = asyncio.gather(*futures)
        failure = asyncio.create_task(node.failed.wait())
        done, pending = await asyncio.wait((all_jobs,task,failure),timeout=timeout,
                                            return_when=asyncio.FIRST_COMPLETED)
        if failure in done and node.failed.is_set():
            raise node.last_error or ConnectionError('shard protocol terminated')
        if task in done:
            await task
            raise ConnectionError('WebRTC disconnected before shard sync completed')
        if all_jobs not in done:
            raise TimeoutError('shard sync timed out')
        results = all_jobs.result()
        # Completion ACK must flush before this one-shot peer closes transport.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15.0
        while (node._channel is not None and node._channel.readyState == 'open'
               and node._channel.bufferedAmount > 0 and loop.time() < deadline):
            await asyncio.sleep(.05)
        await asyncio.sleep(.2)
        return next((x for x in results if isinstance(x,Path)),None)
    finally:
        for future in locals().get('futures',[]):
            if not future.done(): future.cancel()
        if 'all_jobs' in locals() and not all_jobs.done(): all_jobs.cancel()
        if 'failure' in locals() and not failure.done(): failure.cancel()
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
