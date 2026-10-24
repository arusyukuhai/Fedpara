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
import logging
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
    def __init__(self, ws_url: str, room: str, peer_id: str, token: str = '', *, max_peers: int = 96,
                 manifest=None, backup_urls=(), publish=False, publish_token=None):
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
        self.publish=bool(publish)
        self.publish_token=publish_token if publish_token is not None else os.environ.get("ROOM_PUBLISH_TOKEN", "")
        self.max_peers=max_peers
        self.members: set[str]=set()
        # Authoritative first roster is sticky. A second member may join
        # before the trainer loop observes the initial empty roster.
        # Losing this evidence deadlocks two simultaneous room launches.
        self.first_roster_was_empty: bool | None = None
        self.room_existing = False
        self.online=False
        self.fatal_error: Exception | None = None
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
            if self.first_roster_was_empty is None:
                self.room_existing = message.get('existing_room') is True
                self.first_roster_was_empty = not (set(peers)-{self.peer_id})
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
        logging.getLogger('fedpara.roomid').info(
            'room presence: peer=%s type=%s living=%d',self.peer_id,kind,1+len(self.members))
        self.updated.set()

    async def run(self, stop: asyncio.Event):
        import websockets
        import time
        from .transport_keepalive import signaling_keepalive
        from websockets.exceptions import ConnectionClosed
        backoff=1.
        while not stop.is_set():
            for url in self.urls:
              if stop.is_set(): break
              try:
                connected_at=None
                async with websockets.connect(url,max_size=65536,open_timeout=8,
                                              **signaling_keepalive(url)) as ws:
                    self.active_url=url
                    payload={'type':'group-join','room':self.room,'peer':self.peer_id,'token':self.token}
                    if self.manifest is not None: payload['manifest']=self.manifest
                    if self.publish:
                        payload['publish']=True
                        if self.publish_token:payload['publish_token']=self.publish_token
                    await ws.send(json.dumps(payload))
                    first=json.loads(await asyncio.wait_for(ws.recv(),15))
                    if first.get('type')!='group-members':
                        if first.get('type') == 'error':
                            reason=str(first.get('message','error'))
                            # The server may briefly refuse joins while a
                            # previous socket is still registered. Capacity
                            # and duplicate-presence errors are RETRYABLE;
                            # bad credentials/configuration are not.
                            if reason == 'Group full, identity already online, or server busy':
                                raise ConnectionError('temporary lobby rejection: '+reason)
                            raise PatchError('lobby rejected room: '+reason)
                        raise ConnectionError('lobby join failed: '+str(first))
                    self.accept(first)
                    self.online=True;self.emergency=False
                    connected_at=time.monotonic()
                    last_ping=connected_at
                    while not stop.is_set():
                        if time.monotonic()-last_ping>=30.0:
                            await ws.send('{"type":"ping"}')
                            last_ping=time.monotonic()
                        try: data=await asyncio.wait_for(ws.recv(),2)
                        except asyncio.TimeoutError:continue
                        self.accept(json.loads(data))
              except PatchError as exc:
                # A pinned manifest mismatch or invalid server roster is
                # permanent; repeated joins cannot fix it. Fail visibly.
                self.fatal_error=exc
                logging.getLogger('fedpara.roomid').error(
                    'discovery rejected room permanently: %s',exc)
                return
              except (OSError,ConnectionError,asyncio.TimeoutError,ValueError,ConnectionClosed) as exc:
                logging.getLogger('fedpara.roomid').warning('discovery endpoint %s unavailable: %s',url,exc)
              finally:
                if connected_at is not None and time.monotonic()-connected_at>=60.0:
                    backoff=1.0  # Only a sustained healthy session resets it.
                self.online=False
                # KEEP last-known peers. Existing authenticated P2P DataChannels
                # continue to function even if every signaling service dies.
                self.emergency=True
                self.updated.set()
            if not stop.is_set():
                # Stable, peer-specific jitter prevents every GUI instance
                # from retrying on precisely the same clock boundary.
                skew=int.from_bytes(hashlib.sha256(self.peer_id.encode('ascii')).digest()[:2],
                                    'big')/65535.0
                try:await asyncio.wait_for(stop.wait(),backoff+0.25*skew)
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
    # Never use a temporary group-join for a metadata lookup: it marks the room
    # as previously inhabited on Mummy and can permanently prevent the first
    # real participant from becoming its snapshot-serving genesis founder.
    async with websockets.connect(ws_url,max_size=65536,open_timeout=timeout) as ws:
        await ws.send(json.dumps({'type':'group-describe','room':room,'token':token}))
        reply=json.loads(await asyncio.wait_for(ws.recv(),timeout=timeout))
        if reply.get('type') != 'room-description':
            reason=str(reply.get('message','unexpected signaling reply'))
            if reason == 'Unknown message type':
                raise RuntimeError('Mummy signaling server is outdated: update/restart '
                                   'src/main.nim for read-only group-describe before loading a room')
            raise ConnectionError('room metadata is unavailable: '+reason)
        if reply.get('room') != room:
            raise ValueError('room-description does not match requested room')
        return validate_manifest(reply['manifest'])


def public_room_endpoint(ws_url: str) -> str:
    """Translate only the user-chosen signaling origin to its directory API."""
    from urllib.parse import urlsplit, urlunsplit
    u=urlsplit(ws_url)
    if u.scheme not in ('ws','wss') or not u.hostname or u.username or u.password or u.fragment:
        raise ValueError('invalid signaling endpoint')
    if u.scheme == 'ws' and u.hostname not in ('127.0.0.1','localhost','::1'):
        raise PermissionError('remote directory requires WSS')
    return urlunsplit(('https' if u.scheme=='wss' else 'http',u.netloc,'/api/rooms','',''))


def list_public_rooms(ws_url: str, *, timeout=5.0) -> list[dict]:
    """Bounded, read-only public registry; no code execution or data downloads."""
    import urllib.request
    url=public_room_endpoint(ws_url)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise ConnectionError('directory responded with status '+str(response.status))
        if response.headers.get('Content-Length') and int(response.headers['Content-Length']) > 256*1024:
            raise ValueError('room directory response too large')
        raw=response.read(256*1024+1)
    if len(raw)>256*1024:raise ValueError('room directory response too large')
    document=json.loads(raw)
    entries=document.get('rooms') if isinstance(document,dict) else None
    if not isinstance(entries,list) or len(entries)>256:raise ValueError('invalid room directory')
    result=[]
    for entry in entries:
        if not isinstance(entry,dict) or not isinstance(entry.get('room_id'),str):
            raise ValueError('invalid room listing entry')
        validate_room(entry['room_id'])
        if (not isinstance(entry.get('model'),str) or len(entry['model'])>200 or
            not isinstance(entry.get('online'),int) or not 0<=entry['online']<=96 or
            not isinstance(entry.get('datasets'),list) or len(entry['datasets'])>12):
            raise ValueError('invalid room listing data')
        if any(not isinstance(x,str) or len(x)>200 for x in entry['datasets']):
            raise ValueError('invalid room dataset summary')
        result.append(entry)
    return sorted(result,key=lambda r:(-r['online'],r['room_id']))


async def register_room_manifest(ws_url: str, room: str, manifest: dict, *,
                                 publish: bool=False, token: str='', timeout=8) -> None:
    """Persist a room descriptor on Mummy without joining as a fake trainer."""
    import websockets
    from .room_manifest import validate_manifest
    validate_room(room)
    validate_manifest(manifest)
    public_room_endpoint(ws_url)  # also enforces TLS for remote hosts
    payload={'type':'room-register','room':room,'manifest':manifest,'publish':bool(publish)}
    if token: payload['token']=token
    if publish and os.environ.get('ROOM_PUBLISH_TOKEN'):
        payload['publish_token']=os.environ['ROOM_PUBLISH_TOKEN']
    async with websockets.connect(ws_url,max_size=65536,open_timeout=timeout) as ws:
        await ws.send(json.dumps(payload,separators=(',',':')))
        reply=json.loads(await asyncio.wait_for(ws.recv(),timeout))
    if reply.get('type')!='room-registered' or reply.get('room')!=room:
        raise ConnectionError('server registration refused: '+str(reply.get('message','unknown reason')))
