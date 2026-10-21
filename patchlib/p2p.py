"""P2P transfer of .fpatch archives over aiortc DataChannel.

Mummy is only used for SDP / ICE signaling. Patch bytes go through DTLS/SCTP
end-to-end (direct ICE or TURN relay). aiortc currently uses complete SDP
rather than genuine outgoing Trickle ICE; browser peers can trickle to it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import MAX_PATCH, PatchError, decode_patch
from .state import inspect_state

log = logging.getLogger('fedpara.p2p')
CHUNK = 16 * 1024
WINDOW = 256 * 1024


class FrameReceiver:
    """At most one incoming patch at a time; bounded by max_bytes."""
    def __init__(self, max_bytes: int = MAX_PATCH, verifier=None):
        self.max_bytes = max_bytes
        self.verifier = verifier
        self.head: dict | None = None
        self.data = bytearray()

    def feed(self, message: str | bytes) -> bytes | None:
        if isinstance(message, str):
            try:
                msg = json.loads(message)
            except (ValueError, TypeError) as exc:
                raise PatchError('invalid frame JSON') from exc
            if not isinstance(msg, dict) or msg.get('v') != 1:
                raise PatchError('invalid frame version')
            if msg.get('type') == 'start':
                if self.head is not None:
                    raise PatchError('overlapping patch transmissions')
                size, digest = msg.get('size'), msg.get('sha256')
                if not isinstance(size, int) or not 0 < size <= self.max_bytes:
                    raise PatchError('invalid patch size')
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise PatchError('invalid checksum')
                if self.verifier is not None:
                    self.verifier(msg)
                self.head = msg
                self.data = bytearray()
            elif msg.get('type') == 'end':
                if self.head is None or msg.get('id') != self.head.get('id'):
                    raise PatchError('invalid patch end marker')
                if len(self.data) != self.head['size']:
                    raise PatchError('truncated patch')
                if hashlib.sha256(self.data).hexdigest() != self.head['sha256']:
                    raise PatchError('checksum mismatch')
                result = bytes(self.data)
                self.head, self.data = None, bytearray()
                return result
            else:
                raise PatchError('unknown frame type')
        elif isinstance(message, bytes):
            if self.head is None or not message or len(message) > CHUNK:
                raise PatchError('unexpected binary patch frame')
            if len(self.data) + len(message) > self.head['size']:
                raise PatchError('patch exceeds announced size')
            self.data.extend(message)
        else:
            raise PatchError('invalid message data type')
        return None


async def transmit(channel, data: bytes, max_bytes: int = MAX_PATCH, signer=None):
    if not 0 < len(data) <= max_bytes:
        raise PatchError('patch too large')
    patch_id = secrets.token_hex(12)
    digest = hashlib.sha256(data).hexdigest()
    head = {'v':1, 'type':'start', 'id':patch_id, 'size':len(data), 'sha256':digest}
    if signer is not None: head['sig'] = signer(len(data), digest)
    channel.send(json.dumps(head))
    # SCTP messages are modestly sized for compatibility with browsers.
    for start in range(0, len(data), CHUNK):
        while channel.readyState == 'open' and channel.bufferedAmount > WINDOW:
            await asyncio.sleep(0.015)
        if channel.readyState != 'open':
            raise ConnectionError('DataChannel closed during patch transfer')
        channel.send(data[start:start + CHUNK])
    while channel.readyState == 'open' and channel.bufferedAmount > WINDOW:
        await asyncio.sleep(0.015)
    if channel.readyState != 'open':
        raise ConnectionError('DataChannel closed before end marker')
    channel.send(json.dumps({'v':1,'type':'end','id':patch_id}))


async def transmit_file(channel, path: Path, max_bytes: int, signer=None):
    """Stream a multi-GB snapshot from disk (two read passes, bounded memory)."""
    path = Path(path)
    size = path.stat().st_size
    if not 0 < size <= max_bytes:
        raise PatchError('state archive is too large')
    checksum = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024*1024), b''):
            checksum.update(chunk)
    head = {'v':1,'type':'start','id':secrets.token_hex(12), 'size':size,
            'sha256':checksum.hexdigest()}
    if signer is not None: head['sig'] = signer(size, head['sha256'])
    channel.send(json.dumps(head))
    with path.open('rb') as fp:
        while chunk := fp.read(CHUNK):
            while channel.readyState == 'open' and channel.bufferedAmount > WINDOW:
                await asyncio.sleep(.015)
            if channel.readyState != 'open':
                raise ConnectionError('DataChannel closed during state transfer')
            channel.send(chunk)
    while channel.readyState == 'open' and channel.bufferedAmount > WINDOW:
        await asyncio.sleep(.015)
    if channel.readyState != 'open':
        raise ConnectionError('DataChannel closed before state end marker')
    channel.send(json.dumps({'v':1,'type':'end','id':head['id']}))


class DiskFrameReceiver:
    """Receive complete .fstate archives to disk, never buffer model in RAM."""
    def __init__(self, dest: Path, max_bytes: int, verifier=None):
        self.dest, self.max_bytes, self.verifier = Path(dest), max_bytes, verifier
        self.dest.mkdir(parents=True, exist_ok=True)
        self.head = None
        self.handle = None
        self.path = None
        self.sha = hashlib.sha256()
        self.written = 0

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)
            self.path = None
        self.head = None

    def feed(self, message: str | bytes) -> Path | None:
        try:
            if isinstance(message, str):
                if len(message) > 4096:
                    raise PatchError('oversized transfer control frame')
                frame = json.loads(message)
                if not isinstance(frame,dict) or frame.get('v') != 1:
                    raise PatchError('invalid transfer frame')
                if frame.get('type') == 'start':
                    if self.head is not None: raise PatchError('overlapping state transfers')
                    size, digest = frame.get('size'),frame.get('sha256')
                    if type(size) is not int or not 0 < size <= self.max_bytes:
                        raise PatchError('invalid state size')
                    if not isinstance(digest,str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise PatchError('invalid state digest')
                    if self.verifier is not None: self.verifier(frame)
                    fd, filename = tempfile.mkstemp(prefix='.incoming-', suffix='.part', dir=self.dest)
                    self.path = Path(filename)
                    self.handle = os.fdopen(fd,'wb')
                    self.head = frame
                    self.sha = hashlib.sha256(); self.written = 0
                elif frame.get('type') == 'end':
                    if self.head is None or frame.get('id') != self.head.get('id'):
                        raise PatchError('bad state end marker')
                    self.handle.close(); self.handle = None
                    if self.written != self.head['size'] or self.sha.hexdigest() != self.head['sha256']:
                        raise PatchError('state transfer checksum/size mismatch')
                    meta = inspect_state(self.path, max_bytes=self.max_bytes)
                    path = self.dest / f"{meta['peer_id']}_{meta['seq']}_{self.head['sha256'][:16]}.fstate"
                    os.replace(self.path,path)
                    self.path = None; self.head = None
                    return path
                else:
                    raise PatchError('unknown state frame')
            elif isinstance(message, bytes):
                if self.head is None or not message or len(message) > CHUNK:
                    raise PatchError('unexpected state data')
                self.written += len(message)
                if self.written > self.head['size']:
                    raise PatchError('state exceeds declared size')
                self.handle.write(message)
                self.sha.update(message)
            else:
                raise PatchError('unexpected data type')
            return None
        except Exception:
            self.close()
            raise


class PatchNode:
    def __init__(self, signal_url: str, room: str, token: str = '', *,
                 max_bytes: int = MAX_PATCH, turn_only: bool = False,
                 state_dir: Path | None = None, signer=None, verifier=None):
        if not (16 <= len(room) <= 64) or not all(c.isalnum() and c.isascii() or c in '_-' for c in room):
            raise ValueError('room must be 16-64 ASCII alphanumeric/_/- characters')
        if not signal_url.startswith(('wss://','ws://')):
            raise ValueError('signaling URL must be ws:// or wss://')
        self.url, self.room, self.token = signal_url, room, token
        self.max_bytes = max_bytes
        self.turn_only = turn_only
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.signer, self.verifier = signer, verifier
        self.outgoing: asyncio.Queue[bytes | Path] = asyncio.Queue(maxsize=1)
        self.incoming: asyncio.Queue[bytes | Path] = asyncio.Queue(maxsize=8)
        self.connected = asyncio.Event()
        self.stopped = asyncio.Event()
        self.last_error: Exception | None = None
        self._channel = None
        self._send_task = None
        self.sent = asyncio.Event()

    def publish(self, data: bytes) -> None:
        """Keep only newest complete snapshot; avoids unbounded send queues."""
        if len(data) > self.max_bytes:
            raise PatchError('patch exceeds configured limit')
        if self.outgoing.full():
            try: self.outgoing.get_nowait()
            except asyncio.QueueEmpty: pass
        self.outgoing.put_nowait(data)

    def publish_file(self, path: Path) -> None:
        if self.state_dir is None:
            raise ValueError('publish_file requires state transfer mode')
        path = Path(path)
        inspect_state(path, max_bytes=self.max_bytes)
        if path.stat().st_size > self.max_bytes:
            raise PatchError('state archive exceeds transfer cap')
        if self.outgoing.full():
            try: self.outgoing.get_nowait()
            except asyncio.QueueEmpty: pass
        self.outgoing.put_nowait(path)

    async def _pump(self):
        while True:
            data = await self.outgoing.get()
            self.sent.clear()
            if isinstance(data, Path):
                await transmit_file(self._channel, data, max_bytes=self.max_bytes, signer=self.signer)
                log.info('P2P state archive sent: %s', data)
            else:
                await transmit(self._channel, data, max_bytes=self.max_bytes, signer=self.signer)
                log.info('P2P patch sent (%s bytes)', len(data))
            self.sent.set()

    def _attach_channel(self, ch):
        self._channel = ch
        framer = (DiskFrameReceiver(self.state_dir,self.max_bytes,self.verifier)
                  if self.state_dir is not None
                  else FrameReceiver(self.max_bytes,self.verifier))

        @ch.on('open')
        def on_open():
            self.connected.set()
            log.info('WebRTC DataChannel open')
            self._send_task = asyncio.create_task(self._pump())

        @ch.on('close')
        def on_close():
            if isinstance(framer, DiskFrameReceiver): framer.close()
            self.connected.clear()
            if self._send_task:
                self._send_task.cancel()

        @ch.on('message')
        def on_message(message):
            try:
                result = framer.feed(message)
                if result is not None:
                    # Validate the outer archive before admitting it.
                    if self.state_dir is None:
                        decode_patch(result, self.max_bytes)
                    self.incoming.put_nowait(result)
                    log.info('P2P %s received', 'state' if self.state_dir else 'patch')
            except (PatchError, asyncio.QueueFull, ValueError, OSError) as e:
                self.last_error = e
                log.warning('Rejected incoming patch: %s', e)


    async def run(self):
        # Lazy dependencies: standalone math/test code doesn't need aiortc.
        import websockets
        from aiortc import (RTCPeerConnection, RTCSessionDescription,
                            RTCConfiguration, RTCIceServer)
        from aiortc.sdp import candidate_from_sdp

        pc = None
        try:
            async with websockets.connect(self.url, max_size=200_000, ping_interval=20) as ws:
                await ws.send(json.dumps({'type':'join','mode':'p2p','room':self.room,'token':self.token}))
                first = json.loads(await ws.recv())
                if first.get('type') != 'joined':
                    raise ConnectionError('join failed: '+str(first))
                servers = []
                for raw in first.get('iceServers', []):
                    urls = raw.get('urls', [])
                    if isinstance(urls, str): urls = [urls]
                    servers.append(RTCIceServer(urls=urls, username=raw.get('username'),
                                                credential=raw.get('credential')))
                pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
                pending = []
                role = first['role']

                def attach_channel(ch):
                    self._attach_channel(ch)

                @pc.on('datachannel')
                def datachannel(ch):
                    if ch.label == 'fedpara-patch':
                        attach_channel(ch)
                    else:
                        ch.close()

                @pc.on('connectionstatechange')
                async def connection_change():
                    log.info('ICE/DTLS connection state: %s', pc.connectionState)

                async def signal(message):
                    await ws.send(json.dumps(message))

                async def offer():
                    await pc.setLocalDescription(await pc.createOffer())
                    # aiortc gathers all candidates into localDescription.sdp;
                    # this is compatible with Mummy and browser Trickle ICE.
                    await signal({'type':'offer','sdp':pc.localDescription.sdp})

                if role == 'initiator':
                    attach_channel(pc.createDataChannel('fedpara-patch', ordered=True))

                peer_left = False
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        kind = msg.get('type')
                        if kind == 'peer-joined' and role == 'initiator':
                            await offer()
                        elif kind == 'offer' and role == 'answerer':
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg['sdp'], type='offer'))
                            for cand in pending: await pc.addIceCandidate(cand)
                            pending.clear()
                            await pc.setLocalDescription(await pc.createAnswer())
                            await signal({'type':'answer', 'sdp':pc.localDescription.sdp})
                        elif kind == 'answer' and role == 'initiator':
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg['sdp'], type='answer'))
                            for cand in pending: await pc.addIceCandidate(cand)
                            pending.clear()
                        elif kind == 'candidate':
                            if msg.get('candidate') is None:
                                candidate = None
                            else:
                                info = msg['candidate']
                                line = info.get('candidate', '')
                                if not line: continue
                                candidate = candidate_from_sdp(line.removeprefix('candidate:'))
                                candidate.sdpMid = info.get('sdpMid')
                                candidate.sdpMLineIndex = info.get('sdpMLineIndex')
                            if pc.remoteDescription is None:
                                pending.append(candidate)
                            else:
                                await pc.addIceCandidate(candidate)
                        elif kind == 'peer-left':
                            log.warning('Remote peer left the room')
                            peer_left = True
                            break
                        elif kind == 'error':
                            raise ConnectionError(msg.get('message', 'signaling error'))
                except (websockets.exceptions.ConnectionClosed, OSError):
                    if not self.connected.is_set():
                        raise
                # Once ICE/DTLS is established, signaling is no longer required.
                # Continue over the already-open SCTP DataChannel in emergency.
                if not peer_left and self.connected.is_set():
                    log.warning("Signaling unavailable; retaining live WebRTC DataChannel")
                    while self.connected.is_set():
                        await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = exc
            raise
        finally:
            self.connected.clear()
            self.stopped.set()
            if self._send_task:
                self._send_task.cancel()
            if pc is not None:
                await pc.close()


async def exchange_file(node: PatchNode, source: Path | None, output_dir: Path, timeout: float = 120):
    """One-shot transfer; both peers optionally send their local snapshot."""
    if source is not None:
        data = source.read_bytes()
        decode_patch(data, node.max_bytes)
        node.publish(data)
    task = asyncio.create_task(node.run())
    try:
        receive = asyncio.create_task(node.incoming.get())
        done, _ = await asyncio.wait((receive, task), timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            raise ConnectionError('WebRTC connection closed before receiving patch')
        if receive not in done:
            receive.cancel()
            raise TimeoutError('Timed out waiting for a P2P patch')
        result = receive.result()
        meta, _ = decode_patch(result, node.max_bytes)
        if source is not None:
            mine, _ = decode_patch(source.read_bytes(), node.max_bytes)
            for field in ('model_id', 'revision', 'layout', 'kind', 'rank', 'alpha'):
                if mine.get(field) != meta.get(field):
                    raise PatchError('received patch incompatible: '+field)
            if mine.get('peer_id') == meta.get('peer_id'):
                raise PatchError('both peers use the same peer-id')
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{meta['peer_id']}_{meta['seq']}.fpatch"
        dest.write_bytes(result)
        log.info('Saved validated remote patch: %s', dest)
        if source is not None:
            await asyncio.wait_for(node.sent.wait(), min(30.0, timeout))
            # Finish draining aiortc's SCTP send buffer before closing.
            deadline = asyncio.get_running_loop().time() + 10.0
            while node._channel is not None and node._channel.readyState == 'open' and node._channel.bufferedAmount > 0 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.03)
        return dest
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


async def exchange_state(node: PatchNode, source: Path | None, timeout: float = 600):
    """P2P state transfer; node must be configured with state_dir."""
    if node.state_dir is None:
        raise ValueError('state node requires --recv-dir')
    source_meta = None
    if source is not None:
        source_meta = inspect_state(source, max_bytes=node.max_bytes)
        node.publish_file(source)
    task = asyncio.create_task(node.run())
    try:
        receive = asyncio.create_task(node.incoming.get())
        done,_ = await asyncio.wait((receive,task),timeout=timeout,return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            raise ConnectionError('peer disconnected before state arrived')
        if receive not in done:
            receive.cancel()
            raise TimeoutError('timed out waiting for model state')
        result = receive.result()
        meta = inspect_state(result,max_bytes=node.max_bytes)
        if source_meta is not None:
            if any(source_meta[x] != meta[x] for x in ('model_id','revision')):
                result.unlink(missing_ok=True)
                raise PatchError('incoming state is for a different model/revision')
            if source_meta['peer_id'] == meta['peer_id']:
                result.unlink(missing_ok=True)
                raise PatchError('incoming state has same peer_id')
            await asyncio.wait_for(node.sent.wait(), min(timeout,120.0))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30.0
            while node._channel is not None and node._channel.readyState == 'open' and node._channel.bufferedAmount > 0 and loop.time() < deadline:
                await asyncio.sleep(.05)
        return result
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
