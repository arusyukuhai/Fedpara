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
import time
import secrets
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import MAX_PATCH, PatchError, decode_patch
from .state import inspect_state
from .transport_keepalive import signaling_keepalive

log = logging.getLogger('fedpara.p2p')
CHUNK = 16 * 1024
WINDOW = 256 * 1024


async def wait_send_window(channel, *, stall_seconds: float = 45.0):
    """Event-driven SCTP flow control, with bounded fallback polling.

    Older aiortc / fake channels may not emit bufferedamountlow; in that case
    poll at 2ms instead of the former 15ms. Do not busy-spin, retain a hard
    no-progress deadline, and unregister transient callbacks on cancellation.
    """
    previous=None
    deadline=time.monotonic()+stall_seconds
    while channel.readyState == 'open' and channel.bufferedAmount > WINDOW:
        remaining=channel.bufferedAmount
        if previous is None or remaining < previous:
            deadline=time.monotonic()+stall_seconds
        if time.monotonic()>=deadline:
            raise TimeoutError('SCTP send buffer stalled; reconnect transport')
        previous=remaining
        # aiortc's bufferedamountlow signal avoids 15ms fixed sleeps during
        # normal high-bandwidth transfers. The fallback also works on test
        # channels without a pyee event emitter.
        signal=asyncio.Event()
        callback=signal.set
        subscribed=False
        try:
            if hasattr(channel,'once') and hasattr(channel,'remove_listener'):
                try:
                    threshold = min(WINDOW // 2, remaining - 1)
                    channel.bufferedAmountLowThreshold=threshold
                    channel.once('bufferedamountlow',callback)
                    subscribed=True
                except (AttributeError,ValueError,RuntimeError):
                    subscribed=False
            if subscribed:
                if channel.readyState!='open' or channel.bufferedAmount<=WINDOW:
                    continue
                try:
                    await asyncio.wait_for(signal.wait(),timeout=.25)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(.002)
        finally:
            if subscribed:
                try: channel.remove_listener('bufferedamountlow',callback)
                except (AttributeError,ValueError,RuntimeError): pass
    if channel.readyState!='open':
        raise ConnectionError('DataChannel closed during transfer')


class FrameReceiver:
    """At most one incoming patch at a time; bounded by max_bytes."""
    def __init__(self, max_bytes: int = MAX_PATCH, verifier=None):
        self.max_bytes = max_bytes
        self.verifier = verifier
        self.head: dict | None = None
        self.data = bytearray()
        self._digest = hashlib.sha256()

    def feed(self, message: str | bytes) -> bytes | None:
        if isinstance(message, str):
            # A data frame must not smuggle an unbounded JSON allocation into
            # aiortc's synchronous callback; legitimate START/END fit easily.
            if len(message) > 4096:
                raise PatchError('oversized patch transfer control frame')
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
                if type(size) is not int or not 0 < size <= self.max_bytes:
                    raise PatchError('invalid patch size')
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise PatchError('invalid checksum')
                if self.verifier is not None:
                    self.verifier(msg)
                self.head = msg
                self.data = bytearray()
                self._digest = hashlib.sha256()
            elif msg.get('type') == 'end':
                if self.head is None or msg.get('id') != self.head.get('id'):
                    raise PatchError('invalid patch end marker')
                if len(self.data) != self.head['size']:
                    raise PatchError('truncated patch')
                if self._digest.hexdigest() != self.head['sha256']:
                    raise PatchError('checksum mismatch')
                result = bytes(self.data)
                self.head, self.data = None, bytearray()
                self._digest = hashlib.sha256()
                return result
            else:
                raise PatchError('unknown frame type')
        elif isinstance(message, bytes):
            if self.head is None or not message or len(message) > CHUNK:
                raise PatchError('unexpected binary patch frame')
            if len(self.data) + len(message) > self.head['size']:
                raise PatchError('patch exceeds announced size')
            self.data.extend(message)
            self._digest.update(message)
        else:
            raise PatchError('invalid message data type')
        return None


async def transmit(channel, data: bytes, max_bytes: int = MAX_PATCH, signer=None):
    if not 0 < len(data) <= max_bytes:
        raise PatchError('patch too large')
    patch_id = secrets.token_hex(12)
    # Large signed updates must not hash dozens of MiB synchronously on the
    # WebRTC event loop. Small controls can avoid a thread-hop.
    if len(data) >= 1024 * 1024:
        digest = await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
    else:
        digest = hashlib.sha256(data).hexdigest()
    head = {'v':1, 'type':'start', 'id':patch_id, 'size':len(data), 'sha256':digest}
    if signer is not None: head['sig'] = signer(len(data), digest)
    channel.send(json.dumps(head))
    # SCTP messages are modestly sized for compatibility with browsers.
    for start in range(0, len(data), CHUNK):
        await wait_send_window(channel)
        channel.send(data[start:start + CHUNK])
    await wait_send_window(channel)
    channel.send(json.dumps({'v':1,'type':'end','id':patch_id}))


async def transmit_file(channel, path: Path, max_bytes: int, signer=None):
    """Stream a multi-GB snapshot from disk (two read passes, bounded memory)."""
    path = Path(path)
    size = path.stat().st_size
    if not 0 < size <= max_bytes:
        raise PatchError('state archive is too large')
    def digest_file():
        checksum = hashlib.sha256()
        with path.open('rb') as fp:
            for chunk in iter(lambda: fp.read(1024*1024), b''):
                checksum.update(chunk)
        return checksum.hexdigest()
    digest=await asyncio.to_thread(digest_file)
    head = {'v':1,'type':'start','id':secrets.token_hex(12), 'size':size,
            'sha256':digest}
    if signer is not None: head['sig'] = signer(size, head['sha256'])
    channel.send(json.dumps(head))
    with path.open('rb') as fp:
        while chunk := fp.read(CHUNK):
            await wait_send_window(channel)
            channel.send(chunk)
    await wait_send_window(channel)
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
        # A closed SCTP channel must wake the signaling loop immediately.
        # Otherwise ws.recv() can keep the old peer task alive for 120 seconds.
        self.link_closed = asyncio.Event()
        self.stopped = asyncio.Event()
        self.protocol_failure = asyncio.Event()
        self.last_error: Exception | None = None
        self._channel = None
        self._framer = None
        self._send_task = None
        self.sent = asyncio.Event()
        # Observable finite-state negotiation. A slow ICE gather is not a dead link.
        self.phase = 'created'
        self.phase_since = time.monotonic()
        self.role = None
        self.rtc_state = 'new'
        self.ice_state = 'new'

    def set_phase(self, phase: str):
        if phase != self.phase:
            self.phase = phase
            self.phase_since = time.monotonic()
            log.info('P2P room=%s phase=%s role=%s ice=%s rtc=%s',
                     self.room, phase, self.role, self.ice_state, self.rtc_state)

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
            try:
                if isinstance(data, Path):
                    await transmit_file(self._channel, data, max_bytes=self.max_bytes, signer=self.signer)
                    log.info('P2P state archive sent: %s', data)
                else:
                    await transmit(self._channel, data, max_bytes=self.max_bytes, signer=self.signer)
                    log.info('P2P patch sent (%s bytes)', len(data))
                self.sent.set()
            except (asyncio.CancelledError, ConnectionError, TimeoutError):
                # A transport may disappear while a signed update is in the
                # SCTP buffer. Put it back for the next attachment if space is
                # available; journal frontier repair is the durable fallback.
                if not self.outgoing.full():
                    self.outgoing.put_nowait(data)
                raise

    def _attach_channel(self, ch):
        if self._channel is not ch:
            self.connected.clear()
            if self._send_task is not None:
                self._send_task.cancel()
                self._send_task=None
            # The retired DataChannel may never emit its close callback once
            # replaced. Do not leak an unfinished .incoming-*.part snapshot.
            if isinstance(self._framer, DiskFrameReceiver):
                self._framer.close()
        self._channel = ch
        self.link_closed.clear()
        opened_once = False
        framer = (DiskFrameReceiver(self.state_dir,self.max_bytes,self.verifier)
                  if self.state_dir is not None
                  else FrameReceiver(self.max_bytes,self.verifier))
        self._framer = framer

        @ch.on('open')
        def on_open():
            nonlocal opened_once
            if self._channel is not ch or opened_once:
                return
            opened_once=True
            self.connected.set()
            log.info('WebRTC DataChannel open')
            self._send_task = asyncio.create_task(self._pump())

        @ch.on('close')
        def on_close():
            if self._channel is not ch:
                return
            if isinstance(framer, DiskFrameReceiver): framer.close()
            self.connected.clear()
            self.link_closed.set()
            if self.last_error is None:
                self.last_error=ConnectionError('patch DataChannel closed')
            if self._send_task:
                self._send_task.cancel()

        @ch.on('message')
        def on_message(message):
            # A callback queued by an obsolete channel can run after the new
            # one opened. It must not deliver stale frames into a fresh link.
            if self._channel is not ch or ch.readyState != 'open':
                return
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

        # Remote-created aiortc channels are often already open here; no
        # second 'open' event is emitted after we install listeners.
        if ch.readyState == 'open':
            on_open()

    async def run(self):
        # Lazy dependencies: standalone math/test code doesn't need aiortc.
        import websockets
        from aiortc import (RTCPeerConnection, RTCSessionDescription,
                            RTCConfiguration, RTCIceServer)
        from aiortc.sdp import candidate_from_sdp

        pc = None
        try:
            self.set_phase('signaling-connect')
            async with websockets.connect(self.url, max_size=200_000,
                                          open_timeout=10, close_timeout=4,
                                          **signaling_keepalive(self.url)) as ws:
                self.set_phase('signaling-join')
                await ws.send(json.dumps({'type':'join','mode':'p2p','room':self.room,'token':self.token}))
                first = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if first.get('type') != 'joined':
                    raise ConnectionError('join failed: '+str(first))
                servers = []
                for raw in first.get('iceServers', []):
                    urls = raw.get('urls', [])
                    if isinstance(urls, str): urls = [urls]
                    servers.append(RTCIceServer(urls=urls, username=raw.get('username'),
                                                credential=raw.get('credential')))
                pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
                from .negotiation_guard import NegotiationGuard
                role = first['role']
                negotiation=NegotiationGuard(role)
                self.role = role
                self.set_phase('waiting-offer' if role == 'answerer' else 'waiting-peer')

                @pc.on('iceconnectionstatechange')
                async def ice_change():
                    self.ice_state = pc.iceConnectionState
                    log.info('P2P room=%s ICE=%s role=%s', self.room, self.ice_state, role)

                def attach_channel(ch):
                    self._attach_channel(ch)

                @pc.on('datachannel')
                def datachannel(ch):
                    if ch.label == 'fedpara-patch':
                        attach_channel(ch)
                    elif ch.label == 'fedpara-state' and hasattr(self, '_attach_extra_channel'):
                        self._attach_extra_channel(ch)
                    else:
                        ch.close()

                @pc.on('connectionstatechange')
                async def connection_change():
                    self.rtc_state = pc.connectionState
                    log.info('P2P room=%s ICE/DTLS=%s role=%s', self.room, self.rtc_state, role)
                    if pc.connectionState == 'connected':
                        self.set_phase('rtc-connected')
                    if pc.connectionState == 'failed':
                        # The recv loop wakes within one second and owns
                        # connection teardown. Never re-close the WebSocket
                        # from a late callback on an already-closed aiortc pc.
                        self.last_error=ConnectionError('ICE/DTLS failed')
                        self.link_closed.set()

                async def signal(message):
                    await ws.send(json.dumps(message))

                offer_sent = False
                async def offer():
                    nonlocal offer_sent
                    if offer_sent:
                        return
                    offer_sent = True
                    self.set_phase('gathering-offer')
                    await pc.setLocalDescription(await pc.createOffer())
                    # aiortc gathers all candidates into localDescription.sdp;
                    # this is compatible with Mummy and browser Trickle ICE.
                    self.set_phase('offer-sent')
                    await signal({'type':'offer','sdp':pc.localDescription.sdp})

                if role == 'initiator':
                    attach_channel(pc.createDataChannel('fedpara-patch', ordered=True))
                    if hasattr(self, '_attach_extra_channel'):
                        self._attach_extra_channel(pc.createDataChannel('fedpara-state', ordered=True))

                peer_left = False
                last_signal_ping=time.monotonic()
                try:
                    while True:
                        if self.link_closed.is_set():
                            raise ConnectionError('P2P DataChannel closed; retrying transport')
                        if self.protocol_failure.is_set():
                            raise ConnectionError('signed peer handshake failed: '+str(self.last_error))
                        if time.monotonic()-last_signal_ping>=30.0:
                            # Application-level ping avoids relying on pong
                            # timeouts delayed by competing GPU/CPU workers.
                            await ws.send('{"type":"ping"}')
                            last_signal_ping=time.monotonic()
                        try:
                            raw = await asyncio.wait_for(ws.recv(),timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        kind = msg.get('type')
                        if kind == 'peer-joined' and role == 'initiator':
                            await offer()
                        elif kind == 'offer' and role == 'answerer':
                            if not negotiation.accept_offer(pc.signalingState):
                                log.debug('ignored duplicate or stale SDP offer room=%s state=%s',self.room,pc.signalingState)
                                continue
                            self.set_phase('answering-offer')
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg['sdp'], type='offer'))
                            for cand in negotiation.release(): await pc.addIceCandidate(cand)
                            await pc.setLocalDescription(await pc.createAnswer())
                            self.set_phase('answer-sent')
                            await signal({'type':'answer', 'sdp':pc.localDescription.sdp})
                        elif kind == 'answer' and role == 'initiator':
                            if not negotiation.accept_answer(pc.signalingState):
                                log.debug('ignored duplicate or stale SDP answer room=%s state=%s',self.room,pc.signalingState)
                                continue
                            self.set_phase('processing-answer')
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg['sdp'], type='answer'))
                            for cand in negotiation.release(): await pc.addIceCandidate(cand)
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
                            key=json.dumps(msg.get('candidate'),sort_keys=True,separators=(',',':'))
                            decision=negotiation.push_candidate(key,candidate,pc.remoteDescription is not None)
                            if decision is True:
                                await pc.addIceCandidate(candidate)
                        elif kind == 'peer-left':
                            # Losing signaling does not necessarily terminate an
                            # already established direct transport; preserve it.
                            log.warning('P2P signaling peer-left room=%s channel_open=%s',
                                        self.room, self.connected.is_set())
                            peer_left = True
                            break
                        elif kind == 'error':
                            raise ConnectionError(msg.get('message', 'signaling error'))
                except (websockets.exceptions.ConnectionClosed, OSError):
                    if not self.connected.is_set():
                        raise
                # Once ICE/DTLS is established, signaling is no longer required.
                # Continue over the already-open SCTP DataChannel in emergency.
                if self.connected.is_set():
                    log.warning("Signaling unavailable; retaining live WebRTC DataChannel")
                    while self.connected.is_set():
                        if self.protocol_failure.is_set():
                            raise ConnectionError('signed peer handshake failed: '+str(self.last_error))
                        if self.link_closed.is_set():
                            raise ConnectionError('P2P DataChannel closed after signaling outage')
                        # Wake immediately on DataChannel closure instead of
                        # polling every 500ms (critical in stop/resume races).
                        try:
                            await asyncio.wait_for(self.link_closed.wait(),timeout=1.0)
                        except asyncio.TimeoutError:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Keep the original authenticated-control error, not the later
            # generic "signaling aborted" wrapper, for GUI diagnostics.
            if not self.protocol_failure.is_set() or self.last_error is None:
                self.last_error = exc
            log.error('WebRTC link failure room=%s role=%s phase=%s ice=%s rtc=%s: %s: %s',
                      self.room,self.role,self.phase,self.ice_state,self.rtc_state,
                      type(exc).__name__,exc)
            self.set_phase('error')
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
