"""Room-id-only discovery and automatic cryptographic identities.

A room ID is a bearer secret, not proof of a participant's real-world identity.
The signaling server relays only membership/SDP/ICE; this module does not relay
weights or patches via that server.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path

from .core import PatchError
from .identity import _message

ROOM = re.compile(r'^[A-Za-z0-9_-]{24,64}$')
IDENT = re.compile(r'^p[0-9a-f]{32}$')


def validate_room(room: str) -> str:
    if not isinstance(room, str) or not ROOM.fullmatch(room):
        raise ValueError('room-id must be 24-64 ASCII [A-Za-z0-9_-] characters; use a randomly generated value')
    return room


def create_room_id() -> str:
    return secrets.token_urlsafe(24)  # 192 random bits


def pair_room(room: str, a: str, b: str) -> str:
    validate_room(room)
    if not isinstance(a,str) or not IDENT.fullmatch(a) or not isinstance(b,str) or not IDENT.fullmatch(b) or a == b:
        raise ValueError('invalid pair IDs')
    return hashlib.sha256(b'fedpara/pair/v1\0'+room.encode()+b'\0'+min(a,b).encode()+b'\0'+max(a,b).encode()).hexdigest()


def id_from_raw_key(raw: bytes) -> str:
    if not isinstance(raw,bytes) or len(raw)!=32:
        raise PatchError('expected 32-byte Ed25519 public key')
    return 'p'+hashlib.sha256(b'fedpara/peer/v1\0'+raw).hexdigest()[:32]


def raw_to_verifier(raw: bytes):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    pub=Ed25519PublicKey.from_public_bytes(raw)
    def verify(auth):
        try:
            sig=base64.b64decode(auth['sig'],validate=True)
            pub.verify(sig,_message(auth['size'],auth['sha256']))
        except (KeyError, TypeError, ValueError, InvalidSignature) as e:
            raise PatchError('invalid room peer signature') from e
    return verify


def identity_from_public_b64(value: str, peer: str):
    if not isinstance(value,str) or len(value)!=44 or not IDENT.fullmatch(peer):
        raise PatchError('invalid embedded peer identity')
    try: raw=base64.b64decode(value,validate=True)
    except ValueError as e: raise PatchError('malformed embedded key') from e
    if id_from_raw_key(raw)!=peer: raise PatchError('identity and public key disagree')
    return raw_to_verifier(raw)


def auto_identity(path: Path):
    """Create one persistent local signing key; never transmit the private key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        key=Ed25519PrivateKey.generate()
        encoded=key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,serialization.NoEncryption())
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as fh:
            fh.write(encoded);fh.flush();os.fsync(fh.fileno())
    if path.is_symlink() or not stat.S_ISREG(os.stat(path).st_mode) or (os.stat(path).st_mode & 0o077):
        raise PermissionError('identity PEM must be a private 0600 regular file')
    key=serialization.load_pem_private_key(path.read_bytes(),password=None)
    if not isinstance(key,Ed25519PrivateKey):raise PatchError('identity is not Ed25519')
    raw=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    peer=id_from_raw_key(raw)
    pub64=base64.b64encode(raw).decode('ascii')
    def sign(size,digest):
        return base64.b64encode(key.sign(_message(size,digest))).decode('ascii')
    return peer, sign, pub64, raw_to_verifier(raw)


class RoomDiscovery:
    """Lightweight signaling-only membership; auto reconnects on dropped WS."""
    def __init__(self, ws_url: str, room: str, peer_id: str, token: str = '', *, max_peers: int = 32,
                 manifest=None, backup_urls=()):
        from .room_manifest import validate_manifest
        self.url=ws_url
        self.urls=tuple(dict.fromkeys([ws_url,*backup_urls]))
        if any(not u.startswith(('ws://','wss://')) for u in self.urls):
            raise ValueError('invalid signaling URL')
        self.active_url=ws_url
        self.manifest=validate_manifest(manifest) if manifest is not None else None
        self.remote_manifest=None
        self.emergency=False
        self.room=validate_room(room)
        if not IDENT.fullmatch(peer_id): raise ValueError('peer ID must be key-derived')
        self.peer_id=peer_id
        self.token=token
        self.max_peers=max_peers
        self.members: set[str]=set()
        self.online=False
        self.updated=asyncio.Event()

    def accept(self, message: dict):
        """Parse server-authored membership events, ignore unrelated WS messages."""
        if not isinstance(message,dict): raise PatchError('invalid lobby message')
        kind=message.get('type')
        if kind=='group-members':
            from .room_manifest import validate_manifest, manifest_digest
            supplied=message.get('manifest')
            if supplied is not None:
                validate_manifest(supplied)
                if self.manifest is not None and manifest_digest(supplied)!=manifest_digest(self.manifest):
                    raise PatchError('room manifest differs from pinned local descriptor')
                self.remote_manifest=supplied
            elif self.manifest is not None:
                raise PatchError('room did not confirm immutable model/dataset manifest')
            peers=message.get('peers')
            if not isinstance(peers,list) or len(peers)>self.max_peers or any(not isinstance(p,str) or not IDENT.fullmatch(p) for p in peers):
                raise PatchError('invalid lobby roster')
            if len(set(peers))!=len(peers):raise PatchError('duplicate IDs in lobby')
            self.members=set(peers)-{self.peer_id}
        elif kind in ('group-peer-joined','group-peer-left'):
            peer=message.get('peer')
            if not isinstance(peer,str) or not IDENT.fullmatch(peer):raise PatchError('invalid lobby peer')
            if peer==self.peer_id:return
            if kind=='group-peer-joined':
                if len(self.members)>=self.max_peers-1: raise PatchError('too many lobby members')
                self.members.add(peer)
            else:self.members.discard(peer)
        elif kind=='error':raise ConnectionError('lobby rejected: '+str(message.get('message','error')))
        else:return
        self.updated.set()

    async def run(self, stop: asyncio.Event):
        import websockets
        from websockets.exceptions import ConnectionClosed
        backoff=1.
        while not stop.is_set():
            for url in self.urls:
              if stop.is_set(): break
              try:
                async with websockets.connect(url,max_size=65536,ping_interval=20,open_timeout=8) as ws:
                    self.active_url=url
                    payload={'type':'group-join','room':self.room,'peer':self.peer_id,'token':self.token}
                    if self.manifest is not None: payload['manifest']=self.manifest
                    await ws.send(json.dumps(payload))
                    first=json.loads(await asyncio.wait_for(ws.recv(),15))
                    if first.get('type')!='group-members':
                        raise ConnectionError('lobby join failed: '+str(first))
                    self.accept(first)
                    self.online=True;self.emergency=False;backoff=1.
                    while not stop.is_set():
                        try: data=await asyncio.wait_for(ws.recv(),2)
                        except asyncio.TimeoutError:continue
                        self.accept(json.loads(data))
              except (OSError,ConnectionError,asyncio.TimeoutError,ValueError,ConnectionClosed,PatchError) as exc:
                import logging
                logging.getLogger('fedpara.roomid').warning('discovery endpoint %s unavailable: %s',url,exc)
              finally:
                self.online=False
                # KEEP last-known peers. Existing authenticated P2P DataChannels
                # continue to function even if every signaling service dies.
                self.emergency=True
                self.updated.set()
            if not stop.is_set():
                try:await asyncio.wait_for(stop.wait(),backoff)
                except asyncio.TimeoutError:pass
                backoff=min(60.,backoff*2)


async def fetch_room_manifest(ws_url: str, room: str, token: str='', *, timeout: float=10):
    """Read-only room metadata discovery over a short signaling connection.

    No data, source code or model weights transit signaling. Returned JSON is
    validated, *not* trusted as a signed publisher statement; save and pin it.
    """
    import websockets
    from .room_manifest import validate_manifest
    validate_room(room)
    if not isinstance(ws_url,str) or not ws_url.startswith(('ws://','wss://')):
        raise ValueError('invalid ws URL')
    if ws_url.startswith('ws://'):
        from urllib.parse import urlsplit
        if urlsplit(ws_url).hostname not in ('localhost','127.0.0.1','::1'):
            raise PermissionError('remote signaling must use encrypted wss://')
    # Ephemeral unused identity. No signature or persistent account is created.
    peer='p'+secrets.token_hex(16)
    async with websockets.connect(ws_url,max_size=65536,open_timeout=timeout) as ws:
        await ws.send(json.dumps({'type':'group-join','room':room,'peer':peer,'token':token}))
        reply=json.loads(await asyncio.wait_for(ws.recv(),timeout=timeout))
        if reply.get('type') != 'group-members':
            raise ConnectionError('room is unavailable: '+str(reply.get('message','join rejected')))
        if 'manifest' not in reply:
            raise ValueError('room has no registered model/dataset collection yet')
        return validate_manifest(reply['manifest'])
