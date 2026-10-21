"""Signed elastic anti-entropy over Mummy signaling + aiortc DataChannels."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .core import PEER_ID, PatchError, _canonical_json
from .elastic import ElasticEngine, EventJournal
from .identity import signer_from_pem, verifier_from_pem
from .livefed import parse_links, parse_peer_keys
from .p2p import FrameReceiver, PatchNode, WINDOW

log = logging.getLogger('fedpara.elastic')
CTRL_VERSION = 3
CTRL_MAX = 16*1024
INVENTORY_PAGE = 64
REQUEST_PAGE = 32


def encode_control(msg: dict, signer=None) -> str:
    body = {'v':CTRL_VERSION, **msg}
    raw = _canonical_json(body)
    header = {'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
              'sig':signer(len(raw),hashlib.sha256(raw).hexdigest()) if signer else None}
    wire = json.dumps({**body, '_auth':header},separators=(',',':'))
    if len(wire) > CTRL_MAX:
        raise PatchError('oversized elastic control')
    return wire


def decode_control(wire: str, verifier=None, unsafe=False) -> dict:
    if not isinstance(wire,str) or len(wire)>CTRL_MAX:
        raise PatchError('invalid elastic control size')
    try: msg=json.loads(wire)
    except ValueError as e: raise PatchError('bad elastic control JSON') from e
    if not isinstance(msg,dict) or msg.get('v')!=CTRL_VERSION or '_auth' not in msg:
        raise PatchError('not an elastic control frame')
    auth=msg.pop('_auth')
    if not isinstance(auth,dict) or set(auth)!={'size','sha256','sig'}:
        raise PatchError('invalid control auth')
    payload = _canonical_json(msg)
    if type(auth['size']) is not int or auth['size']!=len(payload) or hashlib.sha256(payload).hexdigest()!=auth['sha256']:
        raise PatchError('corrupt elastic control frame')
    if not unsafe:
        if verifier is None: raise PatchError('peer public key missing')
        verifier(auth)
    return msg


class ElasticNode(PatchNode):
    def __init__(self, *args, origin: str, remote: str, genesis: str, unsafe=False,
                 public_key: str | None = None, **kwargs):
        super().__init__(*args,**kwargs)
        self.origin,self.remote,self.genesis,self.unsafe=origin,remote,genesis,unsafe
        self.expert_bridge=None
        self.public_key=public_key
        self.outgoing = asyncio.Queue(maxsize=3)
        self.controls = asyncio.Queue(maxsize=128)
        self.counter = 0
        self.remote_counter = -1
        self.ready = False
        self.verified_at = 0.0
        self.pending: set[tuple[str,int]] = set()
        self.inventory_done = False
        self.inventory_cursor = 0
        self.last_inventory = 0.0
        self.next_inventory_at = 0.0
        self.connected_since = 0.0

    def control(self, kind, **data) -> bool:
        ch=self._channel
        if ch is None or ch.readyState!='open' or ch.bufferedAmount > WINDOW:
            return False
        self.counter+=1
        try:
            ch.send(encode_control({'type':kind,'peer':self.origin,'genesis':self.genesis,
                                    'counter':self.counter,**data},self.signer))
        except (ConnectionError, OSError) as exc:
            self.last_error=exc
            self.connected.clear()
            return False
        return True

    def _attach_channel(self, ch):
        self._channel=ch
        framer=FrameReceiver(self.max_bytes,self.verifier)

        @ch.on('open')
        def on_open():
            self.connected.set()
            self.connected_since=time.monotonic()
            self._send_task=asyncio.create_task(self._pump())
            self.control('hello', **({'pub':self.public_key} if self.public_key else {}))

        @ch.on('close')
        def on_close():
            self.connected.clear()
            self.ready=False
            if self._send_task: self._send_task.cancel()

        @ch.on('message')
        def on_message(message):
            try:
                if isinstance(message,str) and message.startswith('FP-EXPERT1:'):
                    if self.verifier is None or not self.connected.is_set():
                        raise PatchError('expert frame before authenticated hello')
                    if self.expert_bridge is None:
                        raise PatchError('no expert handler configured')
                    self.expert_bridge.accept(message)
                elif isinstance(message,str) and '"_auth"' in message:
                    # Auto mode permits exactly one unauthenticated *claim*:
                    # a hello containing a key which must verify its own
                    # key-derived peer ID and the signature over this hello.
                    if self.public_key is not None and self.verifier is None:
                        proposed=json.loads(message)
                        if proposed.get('type')!='hello' or proposed.get('peer')!=self.remote:
                            raise PatchError('signed hello must precede traffic')
                        from .roomid import identity_from_public_b64
                        verify=identity_from_public_b64(proposed.get('pub'),self.remote)
                        decoded=decode_control(message,verify,False)
                        self.verifier=verify
                        framer.verifier=verify
                    else:
                        decoded=decode_control(message,self.verifier,self.unsafe)
                    if (decoded.get('peer')!=self.remote or
                        decoded.get('genesis')!=self.genesis or
                        type(decoded.get('counter')) is not int or
                        decoded['counter']<=self.remote_counter):
                        raise PatchError('wrong control peer/genesis or replayed counter')
                    self.remote_counter=decoded['counter']
                    self.verified_at=time.monotonic()
                    self.controls.put_nowait(decoded)
                else:
                    if self.public_key is not None and self.verifier is None:
                        raise PatchError('data received before authenticated hello')
                    event=framer.feed(message)
                    if event is not None:
                        self.incoming.put_nowait(event)
            except (PatchError,ValueError,asyncio.QueueFull,TypeError,RuntimeError) as e:
                self.last_error=e
                log.warning('rejecting peer frame from %s: %s',self.remote,e)
                ch.close()


@dataclass
class PeerSlot:
    peer: str
    room: str
    key_path: Path|None
    node: ElasticNode|None = None
    task: asyncio.Task|None = None
    retry_after: float = 0.0
    retries: int = 0
    disabled: bool = False


def read_roster(path: Path, myself: str) -> dict[str, tuple[str,Path|None]]:
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict) or len(data)>255:
        raise ValueError('roster file must be object of <=255 peers')
    result={}
    for peer,rec in data.items():
        if not PEER_ID.fullmatch(peer) or peer==myself or not isinstance(rec,dict):
            raise ValueError('invalid roster peer')
        if set(rec)-{'room','public_key'}:
            raise ValueError('unknown roster peer key')
        room=rec.get('room')
        if not isinstance(room,str) or not 16<=len(room)<=64 or not re.fullmatch(r'[A-Za-z0-9_-]+',room):
            raise ValueError('room must be 16-64 ASCII characters')
        key=rec.get('public_key')
        if key is not None and not isinstance(key,str):
            raise ValueError('public_key must be path or omitted in unsafe test mode')
        result[peer]=(room, Path(key) if key else None)
    return result


async def train_elastic(args, *, load_model:Callable, training_batches:Callable, run_step:Callable):
    """Train regardless of absent peers; signed event gossip eventually reconciles.

    This is asynchronous additive model-delta averaging, not synchronous FedAvg.
    No consensus under network partition: peers can temporarily diverge.
    """
    if args.trust_remote_code or args.factory_file or args.allow_legacy_bin:
        raise PermissionError('Network-enabled training forbids untrusted Python/pickle. Train in offline sandbox.')
    if args.resume: raise ValueError('resume local .fpatch not compatible with elastic updates')
    if not args.state_base: raise ValueError('elastic needs immutable --state-base as common genesis')
    if args.grad_accum<1 or args.local_steps<1 or args.rounds<0:
        raise ValueError('bad training step settings')
    if (args.max_journal_gb < 1 or args.peer_timeout <= args.heartbeat_interval or
        args.sync_timeout < 0 or args.inventory_interval <= 0 or
        args.inventory_cycle <= 0 or args.checkpoint_every < 0):
        raise ValueError('invalid elastic liveness, journal or checkpoint settings')
    if args.peers is not None:
        raise ValueError('--peers is only supported with --mode synchronous; elastic derives denominator from verified live members')
    room_only=bool(getattr(args,'room_id',None))
    if room_only:
        from .roomid import validate_room, auto_identity, RoomDiscovery, pair_room
        validate_room(args.room_id)
        peer_id, room_signer, room_pub, room_verify = auto_identity(
            Path(str(args.journal)+'.identity.pem'))
        if args.peer_id is not None and args.peer_id!=peer_id:
            raise ValueError('room mode uses a generated cryptographic peer ID; omit --peer-id')
        args.peer_id=peer_id
        if args.unsafe_unsigned or args.signing_key or args.self_public_key or args.link or args.roster_file or args.trusted_peer_key:
            raise ValueError('room mode is automatic; remove legacy roster/key options')
    if not room_only and not args.unsafe_unsigned and not args.signing_key:
        raise PermissionError('legacy elastic mode requires signed events and pinned public keys')
    for signal_url in [args.ws,*getattr(args,'ws_backup',[])]:
        if signal_url.startswith('ws://') and not args.unsafe_unsigned:
            host=signal_url.split('/')[2].split(':')[0]
            if host not in ('127.0.0.1','localhost','[::1]'):
                raise PermissionError('internet signaling must use WSS')
    fixed_links=parse_links(args.link,args.peer_id) if args.link else {}
    fixed_keys=parse_peer_keys(args.trusted_peer_key)
    roster={p:(room,fixed_keys.get(p)) for p,room in fixed_links.items()}
    if args.roster_file:
        roster.update(read_roster(args.roster_file,args.peer_id))
    if not args.unsafe_unsigned:
        if any(key is None or not key.is_file() for _,key in roster.values()):
            raise PermissionError('each remote peer needs pinned Ed25519 public key')
    if args.unsafe_unsigned: log.warning('UNSIGNED elastic mode is INSECURE; local testing only')
    # Device-specific decisions are finalized BEFORE allocating the model.
    # Architecture, LoHA rank and initial model layout remain shared invariants.
    controller=None
    if getattr(args,'auto_tune',False):
        from .adaptive import detect_device, initial_policy, AdaptiveController
        profile=detect_device()
        policy=initial_policy(profile,requested_local_steps=args.local_steps,
                              max_batch=max(1,args.batch_size))
        controller=AdaptiveController(profile,policy)
        if args.device == 'cpu' and policy.device != 'cpu':args.device=policy.device
        args.batch_size=policy.batch_size
        args.grad_accum=max(args.grad_accum,policy.grad_accum)
        args.gradient_checkpointing |= policy.gradient_checkpointing
        args.wire_dtype=policy.wire_dtype
        args.local_steps=policy.local_steps
        print('adaptive local policy:',json.dumps(policy.json()),flush=True)

    room_manifest=None
    if room_only:
        from .room_manifest import load_manifest, make_manifest
        has_local_data=(any(getattr(args,field,None) for field in ('batches_dir','text_file','jsonl'))
                        or not hasattr(args,'batches_dir'))
        if not getattr(args,'room_manifest',None) and not has_local_data:
            from .roomid import fetch_room_manifest
            room_manifest=await fetch_room_manifest(args.ws,args.room_id,args.token)
            # Keep a pinned local copy: restart must not silently accept a changed room.
            local_pin=Path(str(args.journal)+'.room-manifest.json')
            if local_pin.exists():
                from .room_manifest import manifest_digest
                if manifest_digest(load_manifest(local_pin)) != manifest_digest(room_manifest):
                    raise PatchError('Room descriptor changed since last session; inspect before replacing pinned copy')
            else:
                local_pin.parent.mkdir(parents=True,exist_ok=True)
                local_pin.write_text(json.dumps(room_manifest,ensure_ascii=False,indent=2),encoding='utf8')
            args.room_manifest=local_pin
        if room_manifest is None:
            room_manifest=(load_manifest(args.room_manifest) if getattr(args,'room_manifest',None)
                           else make_manifest(getattr(args,'model',getattr(args,'model_id',None) or 'unknown'),
                                              getattr(args,'revision','main'),'private-local',
                                              architecture=getattr(args,'hf_task','causal-lm'),
                                              dataset_description='Private federated samples are kept on peers'))
        if ((getattr(args,'model',None) is not None and room_manifest['model']['id'] != args.model)
                or (getattr(args,'revision',None) is not None and room_manifest['model']['revision']!=args.revision)):
            raise PatchError('room manifest model ID/revision differs from local selected model')
        if room_manifest['code']['sha256'] and not getattr(args,'room_manifest',None):
            raise PatchError('custom code must be pinned by local manifest')
    signer=room_signer if room_only else (signer_from_pem(Path(args.signing_key)) if args.signing_key else None)
    # Trust keys retained for peers which permanently left: their signed history
    # must still authenticate when gossiped by another connected peer.
    trusted={}
    key_bytes={}
    registry_path=Path(str(args.journal)+'.trusted.json')
    if registry_path.is_file():
        registry=json.loads(registry_path.read_text(encoding='utf-8'))
        if not isinstance(registry,dict) or len(registry)>257:
            raise PatchError('corrupt pinned identity registry')
        for peer,encoded in registry.items():
            if not PEER_ID.fullmatch(peer) or not isinstance(encoded,str):
                raise PatchError('bad archived public key')
            data=base64.b64decode(encoded,validate=True)
            # Obtain verifier from bytes without trusting a temp file.
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            from .identity import _message
            pub=load_pem_public_key(data)
            if not isinstance(pub,Ed25519PublicKey):raise PatchError('non-Ed25519 key in archived registry')
            def make_verifier(k):
                def verify(msg):
                    try:
                        signature=base64.b64decode(msg['sig'],validate=True)
                        k.verify(signature,_message(msg['size'],msg['sha256']))
                    except (ValueError,TypeError,KeyError,InvalidSignature) as exc:
                        raise PatchError('invalid archived origin signature') from exc
                return verify
            trusted[peer]=make_verifier(pub)
            key_bytes[peer]=data

    def store_pin(peer, data, verifier):
        if peer in key_bytes and key_bytes[peer]!=data:
            raise PatchError(f'public key for {peer} changed; manual key migration required')
        if peer not in key_bytes:
            key_bytes[peer]=data
            trusted[peer]=verifier
            registry_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=registry_path.with_name(registry_path.name+'.tmp')
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            try:
                with os.fdopen(fd,'w',encoding='utf-8') as fh:
                    json.dump({p:base64.b64encode(b).decode('ascii') for p,b in key_bytes.items()},fh)
                    fh.flush();os.fsync(fh.fileno())
                os.replace(tmp,registry_path)
            finally:
                tmp.unlink(missing_ok=True)
    for peer,(_,key) in roster.items():
        if key: store_pin(peer,key.read_bytes(),verifier_from_pem(key))
    if signer and not room_only:
        if not args.self_public_key:
            raise PermissionError('required --self-public-key for relay/replay verification')
        path=Path(args.self_public_key)
        store_pin(args.peer_id,path.read_bytes(),verifier_from_pem(path))
    model,adapter=load_model(args)
    genesis_input={'model_id':adapter.model_id,'revision':adapter.revision,
                   'layout':adapter.layout,'kind':adapter.kind,'rank':adapter.rank,
                   'alpha':adapter.alpha}
    if getattr(args,'moe_plan',None):
        plan=json.loads(Path(args.moe_plan).read_text(encoding='utf-8'))
        genesis_input['moe_plan_sha256']=hashlib.sha256(_canonical_json(plan)).hexdigest()
    if room_only:
        from .room_manifest import manifest_digest
        genesis_input['room_manifest_sha256']=manifest_digest(room_manifest)
        # Prevent joining a same-name room with different genesis checkpoint.
        digest=hashlib.sha256()
        with Path(args.state_base).open('rb') as genesis_file:
            for part in iter(lambda:genesis_file.read(1024*1024),b''):
                digest.update(part)
        genesis_input['checkpoint_sha256']=digest.hexdigest()
        genesis_input['room_fingerprint']=hashlib.sha256(args.room_id.encode('ascii')).hexdigest()
    genesis=hashlib.sha256(_canonical_json(genesis_input)).hexdigest()
    journal=EventJournal(args.journal,genesis,max_bytes=args.max_journal_gb*1024**3)
    engine=ElasticEngine(adapter,args.peer_id,genesis,journal,signer=signer,trusted=trusted,
                          unsafe=args.unsafe_unsigned,max_delta_rms=args.max_delta_rms,
                          event_public=room_pub if room_only else None)
    engine.replay()
    print(f'elastic genesis={genesis} recovered={journal.count()} events; log={args.journal}',flush=True)
    slots:dict[str,PeerSlot]={}
    room_stop=asyncio.Event() if room_only else None
    lobby=RoomDiscovery(args.ws,args.room_id,args.peer_id,args.token,
                         manifest=room_manifest,backup_urls=getattr(args,'ws_backup',[])) if room_only else None
    lobby_task=asyncio.create_task(lobby.run(room_stop)) if room_only else None
    def new_optimizer():
        return torch.optim.AdamW(list(adapter.trainable_parameters()),lr=args.lr,weight_decay=args.weight_decay)
    optimizer=new_optimizer()
    optimizer.zero_grad(set_to_none=True)
    batch_iter=training_batches(args)
    loop=asyncio.get_running_loop()
    local_steps=0; emitted=0; recent_loss=float('nan')
    last_control=0.0; last_config=0.0; first_boot=loop.time()
    next_checkpoint=(journal.count()//args.checkpoint_every+1)*args.checkpoint_every if args.checkpoint_every else None

    def update_roster():
        desired=dict(roster)
        if room_only:
            # Symmetric ring: at most four direct connections per node.
            # Gossip carries immutable updates beyond immediate neighbors.
            # Never drop already-active peer links when signaling goes down.
            members=sorted(lobby.members | {args.peer_id})
            index=members.index(args.peer_id)
            close_peers={members[(index+offset)%len(members)] for offset in (-2,-1,1,2)}-{args.peer_id}
            desired={peer:(pair_room(args.room_id,args.peer_id,peer),None) for peer in close_peers}

        if args.roster_file:
            desired.update(read_roster(args.roster_file,args.peer_id))
        for peer, (room,key) in desired.items():
            raw=key.read_bytes() if key else None
            if peer in key_bytes and raw is not None and key_bytes[peer]!=raw:
                raise PatchError('peer public key rotation requires explicit, audited migration')
            if key: store_pin(peer,raw,verifier_from_pem(key))
            if not room_only and not args.unsafe_unsigned and not key:
                raise PermissionError(f'peer {peer} lacks pinned key')
            old=slots.get(peer)
            if old and (old.room!=room or old.key_path!=key):
                raise PatchError('roster entry room/key changed while running; restart safely')
            if old is None:
                slots[peer]=PeerSlot(peer,room,key)
                print(f'peer registered: {peer}',flush=True)
        for peer,slot in slots.items():
            slot.disabled=peer not in desired

    def living_count():
        if room_only:
            # Group membership is maintained by the signaling service.
            # In ring gossip topology only up to four neighbors connect
            # directly, so neighbor count is NOT the FedAvg-style divisor.
            if lobby.online:return 1+len(lobby.members)
            # Outage: use ONLY recent authenticated DataChannel heartbeats.
            now=time.monotonic()
            return 1+sum(1 for s in slots.values() if not s.disabled and s.node and
                         s.node.connected.is_set() and s.node.ready and
                         now-s.node.verified_at < args.peer_timeout)
        now=time.monotonic()
        return 1+sum(1 for s in slots.values() if not s.disabled and s.node is not None
                     and s.node.connected.is_set() and s.node.ready
                     and now-s.node.verified_at < args.peer_timeout)

    async def fanout(wire: bytes):
        for slot in slots.values():
            node=slot.node
            if node and node.connected.is_set() and not slot.disabled and not node.outgoing.full():
                await node.outgoing.put(wire)
        # A full queue doesn't lose data: journal inventory announces it later.

    async def seal_if_needed():
        nonlocal local_steps,emitted,optimizer
        if not local_steps: return
        # Only unfinished accumulation needs one final optimizer step.
        if pending[0]:
            with adapter.lock:
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            pending[0]=0
        wire=engine.seal_local(living_count(),dtype=args.wire_dtype,
                               max_wire_bytes=args.max_patch_mb*1024**2)
        emitted+=1;local_steps=0
        optimizer=new_optimizer();optimizer.zero_grad(set_to_none=True)
        nonlocal_optimizer[0]=optimizer
        await fanout(wire)
        print(f'elastic local event={emitted}, living={living_count()}, bytes={len(wire)}',flush=True)

    pending=[0]

    async def stop_slot(slot:PeerSlot):
        if slot.task:
            slot.task.cancel()
            await asyncio.gather(slot.task,return_exceptions=True)
        slot.task=None;slot.node=None

    async def start_slots(now):
        for slot in list(slots.values()):
            if slot.disabled:
                if slot.task: await stop_slot(slot)
                continue
            if slot.task and slot.node is not None and slot.node.connected.is_set() and (
                time.monotonic()-max(slot.node.verified_at,slot.node.connected_since) > args.peer_timeout):
                log.warning('peer %s heartbeat expired; reconnecting without blocking training',slot.peer)
                await stop_slot(slot)
                slot.retries=min(slot.retries+1,6)
                slot.retry_after=now+min(60.,2.**slot.retries)
            if slot.task and slot.task.done():
                if not slot.task.cancelled():
                    error=slot.task.exception()
                    if error: log.warning('peer %s disconnected: %s',slot.peer,error)
                await stop_slot(slot)
                slot.retries=min(slot.retries+1,6)
                slot.retry_after=now+min(60.,2.**slot.retries)
            if slot.task is None and now>=slot.retry_after:
                node=ElasticNode((lobby.active_url if room_only and lobby.online else args.ws),slot.room,args.token,origin=args.peer_id,
                    remote=slot.peer,genesis=genesis,unsafe=args.unsafe_unsigned,
                    max_bytes=args.max_patch_mb*1024**2, signer=signer,
                    verifier=trusted.get(slot.peer),
                    public_key=room_pub if room_only else None)
                slot.node=node
                slot.task=asyncio.create_task(node.run(),name='elastic-'+slot.peer)
                log.info('connecting %s',slot.peer)

    async def read_peer_events():
        for slot in list(slots.values()):
            node=slot.node
            if node is None or slot.disabled: continue
            for _ in range(32):
                if node.controls.empty():break
                msg=node.controls.get_nowait()
                typ=msg.get('type')
                if typ == 'hello':
                    node.control('hello-ack')
                    node.inventory_cursor=0
                elif typ == 'inventory':
                    entries=msg.get('events')
                    if not isinstance(entries,list) or len(entries)>INVENTORY_PAGE:
                        raise PatchError('malformed inventory')
                    missing=[]
                    for item in entries:
                        if not (isinstance(item,list) and len(item)==3 and
                                isinstance(item[0],str) and PEER_ID.fullmatch(item[0]) and
                                type(item[1]) is int and 1<=item[1]<2**63 and
                                isinstance(item[2],str) and len(item[2])==64):
                            raise PatchError('malformed event identity in inventory')
                        original=journal.has(item[0],item[1])
                        if original and original[0]!=item[2]:
                            raise PatchError('conflicting event in inventory')
                        if original is None and (item[0],item[1]) not in node.pending:
                            missing.append([item[0],item[1]])
                            node.pending.add((item[0],item[1]))
                    for i in range(0,len(missing),REQUEST_PAGE):
                        node.control('need',events=missing[i:i+REQUEST_PAGE])
                    if msg.get('done') is True: node.inventory_done=True
                    if node.inventory_done and not node.pending:node.ready=True
                elif typ == 'need':
                    entries=msg.get('events')
                    if not isinstance(entries,list) or not 1<=len(entries)<=REQUEST_PAGE:
                        raise PatchError('bad need request')
                    for ident in entries:
                        if not (isinstance(ident,list) and len(ident)==2 and
                                isinstance(ident[0],str) and PEER_ID.fullmatch(ident[0]) and
                                type(ident[1]) is int and 1<=ident[1]<2**63):
                            raise PatchError('bad requested event ID')
                        wire=journal.get(ident[0],ident[1])
                        if wire is not None and node.connected.is_set():
                            if node.outgoing.full(): break
                            node.outgoing.put_nowait(wire)
                elif typ in ('hello-ack','heartbeat'):
                    if controller is not None:
                        controller.observe(slot.peer,msg.get('cap'))
                else:
                    raise PatchError('unsupported elastic control')
            for _ in range(16):
                if node.incoming.empty():break
                wire=node.incoming.get_nowait()
                if not isinstance(wire,bytes):raise PatchError('nonbinary elastic event')
                # Preserve local work against current base before accepting an
                # external update. Replacing factors without sealing loses steps.
                from .elastic import unpack_event
                origin,meta,_,_=unpack_event(wire,trusted,unsafe=args.unsafe_unsigned)
                old=journal.has(origin,meta['seq'])
                if old:
                    engine.ingest(wire)  # verify retransmission; don't interrupt training
                    node.pending.discard((origin,meta['seq']))
                    continue
                if origin!=args.peer_id and local_steps:
                    await seal_if_needed()
                try:
                    if engine.ingest(wire):
                        # Apply normalized delta, recreate AdamW after every remote event.
                        nonlocal_optimizer[0]=new_optimizer()
                        nonlocal_optimizer[0].zero_grad(set_to_none=True)
                        await fanout(wire)  # relay immutable origin-signed event
                finally:
                    # Any one source can fill a need requested from another relay.
                    for ps in slots.values():
                        if ps.node is not None:
                            ps.node.pending.discard((origin,meta['seq']))
                            if ps.node.inventory_done and not ps.node.pending:
                                ps.node.ready=True

    async def maybe_checkpoint():
        nonlocal next_checkpoint,optimizer
        if next_checkpoint is None or journal.count()<next_checkpoint:return
        # Make a bounded, durable recovery point of the *committed* base.
        await seal_if_needed()
        from .shards import export_state_sharded
        root=Path(args.checkpoint_dir)
        root.mkdir(parents=True,exist_ok=True)
        count=journal.count()
        manifest=root/f'elastic-{count:09d}.fmanifest'
        export_state_sharded(model,root/'objects-store',manifest,
            model_id=adapter.model_id,revision=adapter.revision,
            peer_id=args.peer_id,seq=count,adapter=adapter,
            shard_bytes=args.shard_mb*1024**2,max_state_bytes=args.max_state_gb*1024**3)
        # Signed checkpoint descriptor binds manifest+genesis to a complete
        # log inventory. It does not automatically authorize an epoch reset.
        history_hash=hashlib.sha256();offset=0
        while True:
            page,done=journal.inventory(offset,96)
            for ident in page:
                history_hash.update(_canonical_json(ident)+b'\n')
            offset+=len(page)
            if done:break
        payload={'protocol':'fedpara-elastic-checkpoint-v1',
                 'origin':args.peer_id,'genesis':genesis,'event_count':offset,
                 'event_inventory_sha256':history_hash.hexdigest(),
                 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
        contents=_canonical_json(payload)
        if signer:
            payload['signature']=signer(len(contents),hashlib.sha256(contents).hexdigest())
        target=manifest.with_suffix('.checkpoint.json')
        tmp=target.with_suffix('.part')
        with tmp.open('wb') as handle:
            handle.write(_canonical_json(payload));handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,target)
        print(f'elastic recovery checkpoint: {manifest} ({count} journal events)',flush=True)
        next_checkpoint=(journal.count()//args.checkpoint_every+1)*args.checkpoint_every

    # Use one mutable reference when updating optimizer in nested receive handler.
    nonlocal_optimizer=[optimizer]
    try:
        update_roster()
        while args.rounds == 0 or emitted < args.rounds:
            now=loop.time()
            if now-last_config >= 2.0:
                update_roster();last_config=now
            await start_slots(now)
            if now-last_control >= args.heartbeat_interval:
                for slot in slots.values():
                    if slot.node and not slot.disabled and slot.node.connected.is_set():
                        slot.node.control('heartbeat')
                last_control=now
            for slot in slots.values():
                node=slot.node
                if node is None or slot.disabled or not node.connected.is_set():continue
                if now >= node.next_inventory_at and now-node.last_inventory>=args.inventory_interval:
                    ids,done=journal.inventory(node.inventory_cursor,INVENTORY_PAGE)
                    if node.control('inventory',events=ids,done=done):
                        node.inventory_cursor+=len(ids)
                        node.last_inventory=now
                        if done:
                            node.inventory_cursor=0
                            node.next_inventory_at=now+args.inventory_cycle
            # Flush network events before training the next minibatch.
            await read_peer_events()
            if controller is not None:
                # Only mutate safe runtime pacing. Patch rank/layout never change.
                args.local_steps=controller.tune(timeout=args.peer_timeout).local_steps
            await maybe_checkpoint()
            optimizer=nonlocal_optimizer[0]
            # Wait briefly for catch-up when a known peer is reachable; do NOT
            # wait forever for offline peers (permanent departures).
            catchup=any(s.node and s.node.connected.is_set() and not s.node.ready
                        and time.monotonic()-s.node.connected_since < args.sync_timeout
                        for s in slots.values() if not s.disabled)
            if now-first_boot<args.boot_grace or catchup:
                await asyncio.sleep(.03)
                continue
            loss=await asyncio.to_thread(run_step,model,adapter,optimizer,next(batch_iter),args)
            local_steps+=1; pending[0]+=1; recent_loss=loss
            if pending[0]>=args.grad_accum:
                with adapter.lock:
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                pending[0]=0
            if local_steps>=args.local_steps:
                await seal_if_needed()
                nonlocal_optimizer[0]=optimizer
                await maybe_checkpoint()
            if local_steps==1 or local_steps%args.log_steps==0:
                print(f'elastic steps={local_steps} emitted={emitted} loss={recent_loss:.5g} active={living_count()}',flush=True)
            await asyncio.sleep(0)
        if local_steps:
            await seal_if_needed()
        if args.shard_state_manifest and args.shard_store:
            from .shards import export_state_sharded
            export_state_sharded(model,args.shard_store,args.shard_state_manifest,
                model_id=adapter.model_id,revision=adapter.revision,peer_id=args.peer_id,
                seq=journal.count(),adapter=adapter,shard_bytes=args.shard_mb*1024**2,
                max_state_bytes=args.max_state_gb*1024**3)
        if args.full_state_output:
            from .state import export_state
            export_state(model,args.full_state_output,model_id=adapter.model_id,
                revision=adapter.revision,peer_id=args.peer_id,seq=journal.count(),
                adapter=adapter,max_bytes=args.max_state_gb*1024**3)
    finally:
        if room_only:
            room_stop.set()
            lobby_task.cancel()
            await asyncio.gather(lobby_task,return_exceptions=True)
        for slot in slots.values(): await stop_slot(slot)
        journal.close()
