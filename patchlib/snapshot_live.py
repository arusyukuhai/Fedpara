"""Signed, resumable *full model* sync on a second DataChannel of an Elastic link.

The data stream is never forwarded by Mummy. Snapshot manifests are authenticated
with the room peer's Ed25519 key (established on the elastic control channel).
Blob hashes are checked by ShardSession. An event-inventory tip is separately
signed on that control channel before any snapshot may be installed.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

from .core import PatchError, _canonical_json
from .shards import read_manifest, shard_map, manifest_bytes, verify_shards, _atomic_write
from .shardp2p import ShardSession, send_manifest, send_shard, ctrl

log = logging.getLogger("fedpara.snapshot")


def journal_tip(journal) -> tuple[str, int]:
    """Order independent across peer journals (SQLite insertion order differs)."""
    cursor = journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq')
    hash_ = hashlib.sha256()
    count = 0
    for origin, seq, digest in cursor:
        hash_.update(_canonical_json([origin,seq,digest])+b'\n')
        count += 1
    return hash_.hexdigest(),count


def snapshot_inventory(journal, cutoffs=None):
    """Content-addressed journal vector clock for a consistent snapshot cut.

    A receiver may possess *later* events: only rows at or before each
    origin's signed cut-off are hashed, so those later events can be replayed
    over the received state without double-applying committed earlier ones.
    """
    from .snapshot_boundary import validate_cutoffs
    if cutoffs is None:
        cutoffs = {origin:seq for origin,seq in journal.db.execute(
            'SELECT origin,MAX(seq) FROM events GROUP BY origin')}
    validate_cutoffs(cutoffs)
    digest=hashlib.sha256()
    count=0
    for origin,seq,value in journal.db.execute(
            'SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        if seq<=cutoffs.get(origin,0):
            digest.update(_canonical_json([origin,seq,value])+b'\n')
            count+=1
    return digest.hexdigest(),count,cutoffs


class LiveSnapshotLink:
    def __init__(self, store: Path, recv_dir: Path, max_state_bytes: int, verifier_ready):
        self.store=Path(store)
        self.recv_dir=Path(recv_dir)
        self.max_state_bytes=max_state_bytes
        self.verifier_ready=verifier_ready
        self.session=self._new_session()
        self.outbox=asyncio.Queue(maxsize=32)
        self.completed=asyncio.Queue(maxsize=2)
        self.channel=None
        self.worker=None
        self.verify_task=None
        self.connected=asyncio.Event()
        self.error=None
        self.last_activity=time.monotonic()
        self.last_received=self.last_activity

    def _new_session(self):
        # Keep content-addressed, SHA-256-verified files on disk, but never
        # carry one SCTP connection's partial manifest/shard state into another.
        return ShardSession(self.store,self.recv_dir,
            max_state_bytes=self.max_state_bytes,
            verifier=self._verify,request_remote=True,defer_verify=True)

    def mark_failed(self, exc):
        if self.error is None:
            self.error=exc
            log.error('snapshot transport failed: %s: %s',type(exc).__name__,exc,
                      exc_info=exc if exc.__traceback__ else False)
        channel=self.channel
        if channel is not None and channel.readyState!='closed':
            channel.close()

    def stalled(self, timeout: float) -> bool:
        """A connection may be active while no snapshot bytes move."""
        return self.connected.is_set() and time.monotonic()-self.last_activity > timeout

    def _verify(self, signed_header):
        verifier = self.verifier_ready()
        if verifier is None:
            raise PatchError('snapshot manifest before peer identity authentication')
        verifier(signed_header)

    def attach(self,ch):
        # A reconnection MUST reset in-flight frames. Retaining the old
        # ShardSession causes the new manifest-start to be rejected as a
        # duplicate and permanently strands a new participant.
        previous=self.channel
        if previous is not None and previous is not ch:
            if self.worker: self.worker.cancel()
            if self.verify_task: self.verify_task.cancel()
            self.verify_task=None
            self.session.close()
            self.session=self._new_session()
            self.outbox=asyncio.Queue(maxsize=32)
            self.connected.clear()
            self.worker=None
            self.error=None
            if previous.readyState=='open':
                previous.close()
        self.channel=ch
        self.last_activity=time.monotonic()

        @ch.on('open')
        def opened():
            if self.channel is not ch or self.connected.is_set():
                return
            self.last_activity=time.monotonic()
            self.connected.set()
            log.info('full-model state DataChannel ready label=%s id=%s',
                     ch.label if hasattr(ch,'label') else 'fedpara-state',
                     getattr(ch,'id',None))
            self.worker=asyncio.create_task(self._sender(ch))

        @ch.on('close')
        def closed():
            if self.channel is not ch: return
            self.connected.clear()
            self.session.close()
            if self.worker: self.worker.cancel()
            if self.verify_task: self.verify_task.cancel()
            self.verify_task=None
            self.worker=None

        @ch.on('message')
        def message_received(data):
            if self.channel is not ch: return
            try:
                replies,asks=self.session.feed(data)
                self.last_activity=time.monotonic()
                self.last_received=self.last_activity
                for rep in replies:
                    self.outbox.put_nowait(('frame',rep))
                for digest in asks:
                    self.outbox.put_nowait(('shard',digest))
                if self.session.received_manifest:
                    self.completed.put_nowait(self.session.received_manifest)
                    self.session.received_manifest=None
                # Never hash multi-GB checkpoints synchronously from an
                # aiortc on(message) callback. Do it in a cancellable task
                # while SCTP/DTLS timers and signed heartbeats keep running.
                session=self.session
                if (session.remote_meta is not None and
                        session.remote_iterator is None and not session.outstanding and
                        session.reading is None and not session.remote_complete and
                        (self.verify_task is None or self.verify_task.done())):
                    self.verify_task=asyncio.create_task(self._finish_receiver(ch,session))
            except Exception as exc:
                # aiortc event callbacks must not leak exceptions into the
                # event dispatcher: an AttributeError/NoneType previously left
                # the joiner waiting forever, with no actionable error.
                self.mark_failed(exc)

        # aiortc sets incoming remote-created DataChannels to 'open' BEFORE
        # RTCPeerConnection emits its 'datachannel' callback. In that case the
        # open event has *already happened*: merely registering on('open')
        # leaves connected false for the lifetime of an otherwise healthy SCTP
        # stream. The explicit state check also handles both event orders.
        if ch.readyState == 'open':
            opened()
        else:
            log.info('full-model state DataChannel attached; awaiting open state=%s',
                     ch.readyState)

    async def _finish_receiver(self,channel,session):
        """Verify a complete snapshot outside the signaling/event-loop thread."""
        try:
            meta=session.remote_meta
            data=session.remote_data
            digest=session.remote_hash
            if meta is None or data is None or digest is None:
                raise PatchError('snapshot receiver has no complete manifest')
            dest=self.recv_dir / f'{digest}.fmanifest'
            def verify_and_commit():
                verify_shards(meta,self.store)
                _atomic_write(dest,data)
            await asyncio.to_thread(verify_and_commit)
            if self.channel is not channel or self.session is not session:
                return  # Do not commit an obsolete, disconnected transfer.
            session.received_manifest=dest
            session.remote_complete=True
            self.completed.put_nowait(dest)
            session.received_manifest=None
            self.outbox.put_nowait(('frame',ctrl('complete',manifest=digest)))
            self.last_activity=time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.channel is channel and self.session is session:
                self.mark_failed(exc)

    async def _sender(self,channel):
        def progressed():
            if self.channel is channel:
                self.last_activity=time.monotonic()
        try:
            while True:
                kind,data=await self.outbox.get()
                progressed()
                if kind=='frame':
                    if channel.readyState!='open':raise ConnectionError('snapshot channel closed')
                    channel.send(data)
                elif kind=='manifest':
                    await send_manifest(channel,data,self.signer,on_progress=progressed)
                elif kind=='shard':
                    await send_shard(channel,self.store,data,self.session.local_objects[data],
                                     on_progress=progressed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.channel is channel:
                self.mark_failed(exc)

    def offer(self, path: Path, signer):
        """A new immutable manifest; sender reuses its content-addressed store."""
        if not Path(path).is_file():raise FileNotFoundError(path)
        meta=read_manifest(path,max_state_bytes=self.max_state_bytes)
        raw=manifest_bytes(meta)
        self.session.local_data=raw
        self.session.local_meta=meta
        self.session.local_hash=hashlib.sha256(raw).hexdigest()
        self.session.local_objects=shard_map(meta)
        self.session.served.clear()
        self.signer=signer
        self.outbox.put_nowait(('manifest',raw))
        self.last_activity=time.monotonic()
        return self.session.local_hash

    def reset_receiver(self):
        """Restart a finished download; verified content-addressed blobs persist."""
        if self.session.reading is not None or self.session.outstanding:
            return False
        s=self.session
        s.remote_meta=s.remote_hash=s.remote_data=s.remote_iterator=None
        s.remote_complete=False
        s.acked_local=False
        s.received_manifest=None
        s.reused_objects=s.received_objects=0
        return True

    def on_authenticated(self, signer):
        self.signer=signer
