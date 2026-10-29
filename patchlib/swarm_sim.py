"""Multi-process (2–16) real-transport Fedpara swarm fault injector.

The trainer is the ordinary train-live path; no mocked WebRTC, gradients,
state transfers or gossip are used. The optional embedded *signaling-only*
server uses real TCP websockets, while aiortc uses real local UDP/DTLS/SCTP.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import sys
import time
from typing import Any
import zipfile

from .room_manifest import make_manifest, validate_manifest, manifest_digest
from .roomid import create_room_id
from .topology import neighbors


class SimulationError(RuntimeError):
    pass


class LocalSignaling:
    """Bounded, ephemeral Mummy-compatible group/pair websocket test server.

    For fidelity against Nim itself, prefer --ws with a running Mummy server.
    The embedded server NEVER transports model data and listens on loopback.
    """
    def __init__(self, *, port: int = 0, limit: int = 16):
        self.port = port
        self.limit = limit
        self.server = None
        self.groups: dict[str, dict[Any, str]] = defaultdict(dict)
        self.manifests: dict[str, dict] = {}
        self.revisions: dict[str, int] = defaultdict(int)
        self.degrees: dict[str, int] = {}
        self.pairs: dict[str, list[Any]] = defaultdict(list)
        self.lock = asyncio.Lock()
        self.stats = defaultdict(int)

    async def __aenter__(self):
        import websockets
        self.server = await websockets.serve(self.handler, '127.0.0.1', self.port,
            max_size=200_000, ping_interval=20, ping_timeout=60)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_):
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self):
        return f'ws://127.0.0.1:{self.port}/ws'

    async def handler(self, ws):
        affiliation = None
        room = None
        try:
            async for data in ws:
                try:
                    msg = json.loads(data)
                    if not isinstance(msg, dict):
                        continue
                except (ValueError, TypeError):
                    continue
                kind = msg.get('type')
                if affiliation is None:
                    room = msg.get('room')
                    if not isinstance(room, str) or not 24 <= len(room) <= 64 or not re.fullmatch(r'[\w-]+', room, flags=re.ASCII):
                        await ws.send(json.dumps({'type':'error','message':'invalid room'}))
                        return
                    if kind == 'group-join':
                        peer = msg.get('peer')
                        if not isinstance(peer,str) or not re.fullmatch(r'p[0-9a-f]{32}',peer):
                            await ws.send(json.dumps({'type':'error','message':'invalid peer'}))
                            return
                        if msg.get('topology_v2') is not True:
                            await ws.send(json.dumps({'type':'error',
                                'message':'Unsupported client: bounded-v2 topology is required; upgrade Fedpara'}))
                            return
                        degree=msg.get('topology_degree',8)
                        if type(degree)!=int or degree not in (2,4,6,8):
                            await ws.send(json.dumps({'type':'error','message':'Invalid topology degree'}))
                            return
                        manifest = msg.get('manifest')
                        if manifest is not None:
                            try: manifest = validate_manifest(manifest)
                            except Exception:
                                await ws.send(json.dumps({'type':'error','message':'invalid manifest'}))
                                return
                        async with self.lock:
                            existing = room in self.groups and bool(self.groups[room])
                            old_manifest = self.manifests.get(room)
                            previous=dict(self.groups[room])
                            old_ids=sorted(previous.values())
                            if ((old_manifest is None and manifest is None) or
                                (old_manifest is not None and manifest is not None and manifest_digest(old_manifest) != manifest_digest(manifest)) or
                                (existing and degree!=self.degrees[room]) or
                                len(previous)>=self.limit or peer in previous.values()):
                                err = True
                            else:
                                err = False
                                if old_manifest is None:
                                    self.manifests[room] = manifest
                                self.groups[room][ws] = peer
                                self.degrees[room]=degree
                                affiliation = 'group'
                                self.stats['group_joins'] += 1
                                self.revisions[room]+=1
                                rev=self.revisions[room]
                                after=sorted(old_ids+[peer])
                        if err:
                            await ws.send(json.dumps({'type':'error','message':'group rejected'}))
                            return
                        await ws.send(json.dumps({'type':'group-members','topology':'bounded-v2',
                            'degree':degree,'revision':rev,'online':len(after),
                            'peers':sorted(neighbors(peer,after,max_degree=degree)),
                            'existing_room':existing, 'manifest':self.manifests[room]}))
                        for old,old_peer in previous.items():
                            if neighbors(old_peer,old_ids,max_degree=degree) != neighbors(old_peer,after,max_degree=degree):
                                try:
                                    await old.send(json.dumps({'type':'group-neighbors',
                                        'topology':'bounded-v2','revision':rev,'online':len(after),
                                        'peers':sorted(neighbors(old_peer,after,max_degree=degree))}))
                                except Exception:pass
                        continue
                    elif kind == 'join' and msg.get('mode') == 'p2p':
                        async with self.lock:
                            links = self.pairs[room]
                            if len(links) >= 2:
                                err=True
                            else:
                                err=False
                                role='initiator' if not links else 'answerer'
                                prior=links[0] if links else None
                                links.append(ws)
                                affiliation='p2p'
                                self.stats['pair_joins'] += 1
                        if err:
                            await ws.send(json.dumps({'type':'error','message':'pair full'}))
                            return
                        await ws.send(json.dumps({'type':'joined','role':role,
                                                'room':room,'iceServers':[]}))
                        if prior is not None:
                            try:await prior.send(json.dumps({'type':'peer-joined'}))
                            except Exception: pass
                        continue
                    await ws.send(json.dumps({'type':'error','message':'unknown mode'}))
                    return
                elif affiliation == 'p2p' and kind in ('offer','answer','candidate'):
                    async with self.lock:
                        other = next((x for x in self.pairs[room] if x is not ws),None)
                    if other is not None:
                        # Preserve Mummy's tiny forwarding surface: SDP and ICE only.
                        payload = {'type':kind}
                        if kind in ('offer','answer'):
                            if not isinstance(msg.get('sdp'),str) or len(msg['sdp'])>128000:
                                continue
                            payload['sdp'] = msg['sdp']
                        else:
                            payload['candidate'] = msg.get('candidate')
                        try:
                            await other.send(json.dumps(payload))
                            self.stats['signals_forwarded'] += 1
                        except Exception: pass
        finally:
            if affiliation == 'group':
                async with self.lock:
                    previous=dict(self.groups[room])
                    p = self.groups[room].pop(ws,None)
                    remaining=dict(self.groups[room])
                    degree=self.degrees.get(room,8)
                    if p:
                        self.revisions[room]+=1
                        rev=self.revisions[room]
                    else:
                        rev=self.revisions[room]
                    if not remaining:self.degrees.pop(room,None)
                if p:
                    before=sorted(previous.values())
                    after=sorted(remaining.values())
                    for old,old_peer in remaining.items():
                        if neighbors(old_peer,before,max_degree=degree) != neighbors(old_peer,after,max_degree=degree):
                            try:
                                await old.send(json.dumps({'type':'group-neighbors',
                                    'topology':'bounded-v2','revision':rev,'online':len(after),
                                    'peers':sorted(neighbors(old_peer,after,max_degree=degree))}))
                            except Exception:pass
            elif affiliation == 'p2p':
                async with self.lock:
                    if ws in self.pairs[room]:self.pairs[room].remove(ws)
                    others=list(self.pairs[room])
                for other in others:
                    try:await other.send(json.dumps({'type':'peer-left'}))
                    except Exception:pass


def prepare_fixture(root: Path, *, vocab: int = 64, length: int = 12):
    """Create an offline packaged HF causal LM and multiple real tensor batches.

    Uses no remote code, downloads or datasets. Deterministic genesis is shared.
    """
    import torch
    from safetensors.torch import save_file
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise SimulationError('transformers missing: pip install transformers safetensors aiortc') from exc
    model_dir = root / 'fixture' / 'model'
    batches_dir = root / 'fixture' / 'batches'
    state = root / 'fixture' / 'genesis.fstate'
    model_dir.mkdir(parents=True,exist_ok=True)
    batches_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(19071)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_positions=32, n_ctx=32,
                            n_embd=16,n_layer=1,n_head=2, bos_token_id=1,eos_token_id=2))
    from .cli import save_model
    save_model(model,model_dir)
    # Do not tokenize inside the workers (and therefore avoid Hub I/O).
    gen=torch.Generator().manual_seed(19123)
    for i in range(24):
        ids = torch.randint(3,vocab,(1,length),generator=gen,dtype=torch.long)
        save_file({'input_ids':ids, 'attention_mask':torch.ones_like(ids),
                   'labels':ids.clone()},batches_dir/f'batch-{i:03d}.safetensors')
    from .state import export_state
    export_state(model,state,model_id='fixture/model',revision='main',
                 peer_id='p'+'0'*32)
    model=None
    manifest=make_manifest('fixture/model','main','offline synthetic tensor batches',
                           architecture='causal-lm',description='Swarm E2E tiny GPT-2',
                           dataset_description='Deterministic local safetensors batches')
    (root/'fixture'/'room.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def journal_inventory(path:Path):
    if not path.exists():return {}
    # Open read-only: do not interfere with 8 WAL writers.
    db=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True,timeout=1)
    try:
        return {(p,int(s)):d for p,s,d in db.execute('SELECT origin,seq,digest FROM events')}
    finally:db.close()


def inventory_report(paths:list[Path]):
    inventories = [journal_inventory(p) for p in paths]
    union={}
    conflict=False
    for inv in inventories:
        for key,value in inv.items():
            if key in union and union[key] != value:
                conflict=True
            union[key]=value
    missing=[len(union.keys()-inv.keys()) for inv in inventories]
    mismatches=[sum(1 for k,v in inv.items() if union.get(k)!=v) for inv in inventories]
    return {'unique_events':len(union),'per_peer_events':[len(x) for x in inventories],
            'per_peer_missing':missing,'per_peer_conflicting':mismatches,
            'all_equal':not conflict and not any(missing) and not any(mismatches)}



def detailed_gossip_report(paths: list[Path]):
    """Fault-localizing inventory summary, without copying raw private patches.

    Counts describe missing *signed events*, not loss in the signaling server.
    They include events still available on other peers after a SIGKILL.
    """
    from collections import Counter
    inventories=[journal_inventory(path) for path in paths]
    union=set().union(*(set(inv) for inv in inventories))
    by_origin=Counter(origin for origin,_ in union)
    details=[]
    for i,inv in enumerate(inventories):
        missing=union-set(inv)
        origin_counts=Counter(origin for origin,_ in missing)
        details.append({'peer':i,'event_count':len(inv),'missing':len(missing),
                        'missing_by_origin':dict(sorted(origin_counts.items()))})
    return {'unique_events':len(union),'union_by_origin':dict(sorted(by_origin.items())),
            'peers':details}


def compare_fstates(paths:list[Path]):
    """Compare actual floating point tensors, not ZIP hashes (which include peer IDs)."""
    import numpy as np
    from .state import verify_state
    if not all(p.exists() for p in paths):
        return {'ok':False,'error':'missing full-state export',
                'missing':[str(p) for p in paths if not p.exists()]}
    metas=[verify_state(p) for p in paths]
    specs0 = metas[0]['tensors']
    max_abs=0.; max_rel=0.; worst=''; mismatched=[]
    for other in metas[1:]:
        if [(s['name'],s['shape'],s['dtype']) for s in other['tensors']] != [(s['name'],s['shape'],s['dtype']) for s in specs0]:
            return {'ok':False,'error':'tensor schema mismatch'}
    # Read one tensor at a time; never materialize all model states at once.
    with zipfile.ZipFile(paths[0]) as ref:
        for spec in specs0:
            if not spec['dtype'].startswith(('torch.float','torch.bfloat')):
                for p in paths[1:]:
                    with zipfile.ZipFile(p) as z:
                        if z.read(spec['entry']) != ref.read(spec['entry']):
                            mismatched.append(spec['name'])
                continue
            dtype=spec['dtype']
            if dtype == 'torch.bfloat16':
                # Convert bfloat16 bits to float32 by exact exponent/mantissa placement.
                def values(raw):
                    return (np.frombuffer(raw,dtype='<u2').astype('<u4') << 16).view('<f4')
            else:
                dtype_map={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}
                if dtype not in dtype_map:
                    return {'ok':False,'error':f'unsupported state comparison dtype {dtype}'}
                def values(raw, dt=dtype_map[dtype]):return np.frombuffer(raw,dtype=dt).astype('float64')
            a=values(ref.read(spec['entry'])).astype('float64')
            for path in paths[1:]:
                with zipfile.ZipFile(path) as z:
                    b=values(z.read(spec['entry'])).astype('float64')
                if a.shape!=b.shape:
                    return {'ok':False,'error':'shape mismatch'}
                diff=np.abs(a-b)
                value=float(diff.max(initial=0.))
                rel=float((diff/(np.maximum(np.abs(a),1e-6))).max(initial=0.))
                if value>max_abs: max_abs=value;worst=spec['name']
                max_rel=max(max_rel,rel)
    return {'ok':max_abs<=1e-3 and not mismatched,
            'max_abs':max_abs,'max_rel':max_rel,'worst_tensor':worst,
            'nonfloat_mismatches':sorted(set(mismatched))}


@dataclass
class Worker:
    number:int
    root:Path
    process: Any=None
    monitor: Any=None
    log_file: Any=None
    starts:int=0
    joined:bool=False
    snapshot:bool=False
    signed_hello:int=0
    # These counters are reset on each subprocess start.  For a SIGKILL
    # recovery, the previous run's Hello must not masquerade as a new link.
    run_signed_hello:int=0
    rejoin_ready:bool=False
    emitted:int=0
    last_phase:str='unstarted'
    errors:list[str]=field(default_factory=list)
    exits:list[int]=field(default_factory=list)
    log_tail:list[str]=field(default_factory=list)
    max_active:int=0
    max_living:int=0
    losses:list[float]=field(default_factory=list)
    disconnects:int=0
    epoch_digest:str|None=None

    @property
    def journal(self):return self.root/'events.sqlite3'
    @property
    def full_state(self):return self.root/'final.fstate'


class SwarmRunner:
    def __init__(self,args):
        self.a=args
        self.root=Path(args.output).resolve()
        self.workers=[Worker(i,self.root/f'peer-{i:02d}') for i in range(args.peers)]
        self.rng=random.Random(args.seed)
        self.events=[]
        self.started=time.monotonic()
        self.signaling=None
        self.signal_server=None
        self.pause_file=self.root/'pause.flag'
        self.closed=False

    def note(self,event,**kwargs):
        info={'time_s':round(time.monotonic()-self.started,3),'event':event,**kwargs}
        self.events.append(info)
        print(f'[swarm {info["time_s"]:7.2f}s] {event} '+ ' '.join(f'{k}={v}' for k,v in kwargs.items()),flush=True)

    async def monitor(self, worker:Worker, proc, fp):
        while line:=await proc.stdout.readline():
            msg=line.decode('utf-8','replace').rstrip()
            fp.write(msg+'\n');fp.flush()
            worker.log_tail=(worker.log_tail+[msg])[-50:]
            if 'REJOIN READY' in msg:
                worker.rejoin_ready=True
                worker.joined=True
            elif ('JOIN READY' in msg or 'ROOM FOUNDER READY' in msg
                  or 'ROOM GENESIS READY' in msg):
                worker.joined=True
            if 'SNAPSHOT READY' in msg:worker.snapshot=True
            if 'EPOCH V2 CERTIFIED' in msg:
                m=re.search(r'digest=([a-f0-9]{16})',msg)
                if m:worker.epoch_digest=m.group(1)
            if 'SIGNED HELLO OK' in msg:
                worker.signed_hello+=1
                worker.run_signed_hello+=1
            if ('SNAPSHOT ERROR' in msg or 'PEER BLOCKED' in msg or
                'Traceback' in msg or re.search(r'^[A-Za-z_][\w.]*(?:Error|Exception):',msg)):
                worker.errors.append(msg[:500])
                worker.errors=worker.errors[-80:]
            if 'elastic local event=' in msg:
                worker.emitted=max(worker.emitted,int(re.search(r'event=(\d+)',msg).group(1)))
            active=re.search(r'\bactive=(\d+)\b',msg)
            living=re.search(r'\bliving=(\d+)\b',msg)
            if active:worker.max_active=max(worker.max_active,int(active.group(1)))
            if living:worker.max_living=max(worker.max_living,int(living.group(1)))
            value=re.search(r'loss=([0-9.eE+\-]+)',msg)
            if value:
                try:worker.losses=(worker.losses+[float(value.group(1))])[-500:]
                except ValueError:pass
            if 'disconnected:' in msg or 'datachannel-closed' in msg:
                worker.disconnects+=1
            if 'JOIN WAIT' in msg:worker.last_phase='join-wait'
            if 'REJOIN READY' in msg:worker.last_phase='rejoin-ready'
            elif 'JOIN READY' in msg:worker.last_phase='ready'
            if 'SNAPSHOT' in msg:worker.last_phase='snapshot'
        worker.exits.append(await proc.wait())
        fp.close()

    async def spawn(self,i:int):
        w=self.workers[i]
        if w.process is not None and w.process.returncode is None:
            raise SimulationError(f'worker {i} already running')
        w.root.mkdir(parents=True,exist_ok=True)
        argv=[sys.executable,'-u','-m','patchlib.cli','train-live',
              '--mode','elastic','--room-id',self.room_id,'--ws',self.signaling,
              '--room-manifest','fixture/room.json','--model','fixture/model',
              '--state-base','fixture/genesis.fstate','--batches-dir','fixture/batches',
              '--no-auto-tune','--device','cpu','--kind',self.a.kind,'--rank',str(self.a.rank),
              '--local-steps',str(self.a.local_steps),'--batch-size','1','--grad-accum','1',
              '--log-steps','1','--rounds','0','--boot-grace','0.3',
              '--inventory-interval','0.3','--inventory-cycle','3',
              '--heartbeat-interval','1','--peer-timeout','25',
              '--sync-timeout','8','--durable-every','1',
              '--max-neighbors',str(getattr(self.a,'max_neighbors',8)),'--swarm-max-events',str(getattr(self.a,'max_events_per_peer',0)),
              '--journal',str(w.journal),
              '--swarm-pause-file',str(self.pause_file),
              '--full-state-output',str(w.full_state)]
        if getattr(self.a,'epoch_v2_probe',False):
            argv.extend(['--epoch-v2-probe-committee',str(self.root/'epoch-committee.json')])
        env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        w.starts+=1
        w.joined=False
        w.rejoin_ready=False
        w.run_signed_hello=0
        fp=(w.root/f'run-{w.starts:02d}.log').open('w',encoding='utf-8',buffering=1)
        proc=await asyncio.create_subprocess_exec(*argv,cwd=self.root,env=env,
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        w.process=proc
        w.monitor=asyncio.create_task(self.monitor(w,proc,fp))
        self.note('peer-start',peer=i,pid=proc.pid,starts=w.starts)

    async def stop(self,i:int,*,crash:bool=False):
        w=self.workers[i]
        p=w.process
        if p is None or p.returncode is not None:return
        p.send_signal(signal.SIGKILL if crash else signal.SIGTERM)
        try:await asyncio.wait_for(p.wait(),timeout=25)
        except asyncio.TimeoutError:
            p.kill();await p.wait()
        if w.monitor:await w.monitor
        self.note('peer-stop',peer=i,crash=crash,code=p.returncode)

    async def wait_until(self, condition,seconds,description):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if condition():return True
            await asyncio.sleep(.2)
        self.note('timeout',description=description)
        return False

    async def wait_for_rejoin(self, i: int, seconds: float) -> bool:
        """Require durable restoration AND a new authenticated P2P link.

        Merely seeing the old signed Hello count or a process PID is not
        evidence that the restarted process has re-entered the swarm.
        """
        w=self.workers[i]
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if w.process is None or w.process.returncode is not None:
                self.note('rejoin-process-exited',peer=i,
                          code=None if w.process is None else w.process.returncode,
                          phase=w.last_phase)
                self.report_rejoin_failure(i)
                return False
            if w.rejoin_ready and w.run_signed_hello>0:
                self.note('peer-rejoined',peer=i,
                          authenticated_links=w.run_signed_hello,
                          phase=w.last_phase)
                return True
            await asyncio.sleep(.2)
        self.note('timeout',description='peer-rejoined',peer=i)
        self.report_rejoin_failure(i)
        return False

    def report_rejoin_failure(self, i: int) -> None:
        w=self.workers[i]
        details={
            'peer':i,'starts':w.starts,
            'process_alive':bool(w.process is not None and w.process.returncode is None),
            'rejoin_ready':w.rejoin_ready,
            'authenticated_links_this_run':w.run_signed_hello,
            'last_phase':w.last_phase,
            'errors':w.errors[-8:],
            'log_file':str(w.root/f'run-{w.starts:02d}.log'),
            'last_logs':w.log_tail[-35:],
        }
        (w.root/'rejoin-diagnostic.json').write_text(json.dumps(details,indent=2))
        self.note('rejoin-diagnostic',peer=i,phase=w.last_phase,
                  authenticated=w.run_signed_hello,
                  ready=w.rejoin_ready,report=w.root/'rejoin-diagnostic.json')

    async def wait_for_settle(self,seconds):
        until=time.monotonic()+seconds
        stable_since=None
        previous=None
        while time.monotonic()<until:
            if any(w.process is None or w.process.returncode is not None for w in self.workers):
                return False
            report=inventory_report([w.journal for w in self.workers])
            key=tuple(report['per_peer_events'])
            if report['all_equal'] and report['unique_events']>0:
                if key==previous:
                    if stable_since and time.monotonic()-stable_since>=3:
                        return True
                else:stable_since=time.monotonic()
            else:stable_since=None
            previous=key
            await asyncio.sleep(1)
        return False

    def report_startup_failure(self, group: list[int], *, wave: int) -> Path:
        """Preserve useful diagnostics when mass bootstrap does not complete.

        Logs contain no model tensors, but can contain peer IDs and public room
        metadata; keep the diagnostic in the local run directory only.
        """
        details=[]
        for worker in self.workers:
            if worker.starts == 0:
                continue
            recent=worker.log_tail[-90:]
            signal_lines=[line for line in recent if any(mark in line for mark in
                ('JOIN WAIT','SNAPSHOT ','SNAPSHOT SOURCE','SIGNED HELLO',
                 'PEER BLOCKED','ERROR','Traceback','disconnected','ConnectionError'))]
            details.append({
                'peer':worker.number,
                'in_failed_wave':worker.number in group,
                'process_alive':bool(worker.process is not None and worker.process.returncode is None),
                'joined':worker.joined,
                'snapshot_verified':worker.snapshot,
                'signed_hello_count':worker.signed_hello,
                'max_active':worker.max_active,
                'last_phase':worker.last_phase,
                'errors':worker.errors[-12:],
                'recent_relevant_logs':signal_lines[-24:],
                'log_file':str(worker.root / f'run-{worker.starts:02d}.log')})
        target=self.root/'startup-diagnostic.json'
        target.write_text(json.dumps({
            'failure':'initial-snapshot-bootstrap', 'wave':wave,
            'wave_peer_numbers':group,
            'join_completed':sum(w.joined for w in self.workers),
            'snapshot_completed':sum(w.snapshot for w in self.workers),
            'workers':details},indent=2),encoding='utf-8')
        self.note('startup-diagnostic',wave=wave,
                  joined=sum(w.joined for w in self.workers),
                  snapshots=sum(w.snapshot for w in self.workers),report=target)
        return target

    async def join_newcomers(self):
        """Keep snapshot bootstrap bounded while testing a larger live swarm.

        The source is immutable while the initial full-state fanout completes.
        This is a harness barrier, NOT a production room admission restriction.
        """
        a=self.a
        total=a.peers-1
        configured=getattr(a,'bootstrap_batch_size',0)
        batch_size=(min(4,total) if a.peers>8 else total) if configured==0 else configured
        for wave,offset in enumerate(range(1,a.peers,batch_size),start=1):
            batch=list(range(offset,min(offset+batch_size,a.peers)))
            self.note('bootstrap-wave-start',wave=wave,
                      peers=','.join(map(str,batch)),batch_size=len(batch))
            for i in batch:
                await self.spawn(i)
                await asyncio.sleep(a.join_interval)
            deadline=time.monotonic()+a.startup_timeout
            last_progress=0.0
            while time.monotonic()<deadline:
                if all(self.workers[i].joined and self.workers[i].snapshot for i in batch):
                    self.note('bootstrap-wave-complete',wave=wave,
                              peers=','.join(map(str,batch)),
                              verified_snapshots=sum(w.snapshot for w in self.workers))
                    break
                for i in batch:
                    process=self.workers[i].process
                    if process is None or process.returncode is not None:
                        diag=self.report_startup_failure(batch,wave=wave)
                        raise SimulationError(f'newcomer {i} exited during snapshot bootstrap; inspect {diag}')
                if time.monotonic()-last_progress>=15:
                    self.note('bootstrap-wave-progress',wave=wave,
                              ready=sum(self.workers[i].joined for i in batch),
                              snapshots=sum(self.workers[i].snapshot for i in batch),
                              expected=len(batch))
                    last_progress=time.monotonic()
                await asyncio.sleep(.25)
            else:
                diag=self.report_startup_failure(batch,wave=wave)
                raise SimulationError('at least one newcomer could not verify full-model snapshot; '
                                      f'inspect {diag}')

    async def run(self):
        a=self.a
        if not 2 <= a.peers <= 16:
            raise SimulationError('swarm-sim supports 2..16 real peer processes')
        if getattr(a,'max_neighbors',8) not in (2,4,6,8):
            raise SimulationError('--max-neighbors must be 2, 4, 6, or 8')
        if not 1 <= a.faults < a.peers:
            raise SimulationError('faults must be in range 1..peers-1')
        if self.root.exists() and any(self.root.iterdir()):
            raise SimulationError('output directory must be new or empty; never overwrite a prior trial')
        self.root.mkdir(parents=True,exist_ok=True)
        self.room_id=create_room_id()
        self.pause_file.unlink(missing_ok=True)
        # The 16-peer integration test needs stable snapshot exporters. While
        # bootstrapping, workers exchange full-model shards and signatures but
        # do not emit new optimizer events. Training starts together AFTER all
        # newcomers have a verified full model. Eight-peer legacy mode remains
        # unchanged unless --bootstrap-batch-size is explicitly supplied.
        bootstrap_barrier = a.peers > 8 or getattr(a,'bootstrap_batch_size',0)>0
        if bootstrap_barrier:
            self.pause_file.write_text('bootstrap pause; snapshots only; no optimization\n')
            self.note('bootstrap-training-paused',peers=a.peers)
        await asyncio.to_thread(prepare_fixture,self.root)
        if getattr(a,'epoch_v2_probe',False):
            # Generate the exact durable identities that train-live will reuse.
            # Do NOT select a mutable online roster for consensus.
            from .roomid import auto_identity
            committee=[auto_identity(Path(str(w.journal)+'.identity.pem'))[0] for w in self.workers]
            (self.root/'epoch-committee.json').write_text(json.dumps(sorted(committee),indent=2))
        (self.root/'scenario.json').write_text(json.dumps({
            'room':self.room_id,'seed':a.seed,'peers':a.peers,'faults':a.faults,
            'kind':a.kind,'rank':a.rank,'local_steps':a.local_steps,
            'max_events_per_peer':a.max_events_per_peer,
            'max_neighbors':getattr(a,'max_neighbors',8),
            'bootstrap_batch_size':getattr(a,'bootstrap_batch_size',0)},indent=2))
        try:
            if a.embedded_signaling:
                self.signal_server=await LocalSignaling(port=a.port).__aenter__()
                self.signaling=self.signal_server.url
            else:
                self.signaling=a.ws
                from urllib.parse import urlsplit
                parsed=urlsplit(self.signaling)
                if parsed.scheme == 'ws' and parsed.hostname not in ('127.0.0.1','localhost','::1'):
                    raise SimulationError('remote Mummy signaling must use TLS (wss://)')
                if parsed.scheme not in ('ws','wss'):
                    raise SimulationError('--ws requires ws:// (loopback only) or wss://')
                import websockets
                # No silent fallback to simulated networking.
                async with websockets.connect(self.signaling,open_timeout=4):pass
            self.note('transport',endpoint=self.signaling,mode='aiortc-real-WebRTC')
            await self.spawn(0)
            if not await self.wait_until(lambda:self.workers[0].joined,a.startup_timeout,'founder-ready'):
                raise SimulationError('founder failed to enter room; inspect peer-00/run-01.log')
            await self.join_newcomers()
            self.note('all-joined',snapshots=sum(w.snapshot for w in self.workers))
            if bootstrap_barrier:
                self.pause_file.unlink(missing_ok=True)
                self.note('bootstrap-training-resumed',peers=a.peers)
            await asyncio.sleep(a.warmup)
            choices=self.rng.sample(list(range(1,a.peers)),k=a.faults)
            for i in choices:
                await self.stop(i,crash=True)
                self.note('fault-injected',peer=i,mode='SIGKILL')
                await asyncio.sleep(a.down_time)
                await self.spawn(i)
                if not await self.wait_for_rejoin(i,a.startup_timeout):
                    raise SimulationError(f'peer {i} did not resume; inspect '
                        f'{self.workers[i].root}/rejoin-diagnostic.json and '
                        f'run-{self.workers[i].starts:02d}.log')
                await asyncio.sleep(a.between_faults)
            self.pause_file.write_text('stop optimization; keep WebRTC and gossip active\n')
            self.note('quiesce',reason='drain signed update inventory and validate model')
            settled=await self.wait_for_settle(a.settle_timeout)
            self.note('gossip-converged' if settled else 'gossip-timeout')
            epoch_probe_ok=None
            if getattr(a,'epoch_v2_probe',False) and settled:
                epoch_probe_ok=await self.wait_until(
                    lambda: all(w.epoch_digest is not None for w in self.workers) and
                    len({w.epoch_digest for w in self.workers})==1,
                    a.epoch_v2_timeout,'epoch-v2-consensus')
                self.note('epoch-v2-certified' if epoch_probe_ok else 'epoch-v2-timeout',
                          committee=len(self.workers),digest=self.workers[0].epoch_digest)
            elif getattr(a,'epoch_v2_probe',False):
                epoch_probe_ok=False
            for i in range(a.peers):await self.stop(i,crash=False)
            event_report=inventory_report([w.journal for w in self.workers])
            # Save actionable diagnostics even on ordinary FAIL, not only
            # exceptional exits. report.json alone hides the missing origins.
            diagnostics=detailed_gossip_report([w.journal for w in self.workers])
            if not event_report['all_equal']:
                (self.root/'gossip-diagnostic.json').write_text(
                    json.dumps(diagnostics,indent=2),encoding='utf-8')
            state_report=compare_fstates([w.full_state for w in self.workers])
            join_ok=all(w.joined for w in self.workers)
            snapshot_ok=all(w.snapshot for w in self.workers[1:])
            link_ok=all(w.signed_hello > 0 for w in self.workers)
            churn_ok=all(self.workers[i].starts >= 2 for i in choices)
            from .topology import topology_metrics
            expected=topology_metrics([f'p{i:032x}' for i in range(a.peers)],
                                      max_degree=getattr(a,'max_neighbors',8))
            cap=getattr(a,'max_neighbors',8)
            max_active=max((w.max_active for w in self.workers),default=0)
            # active includes local self; this checks real sampled WebRTC links.
            within_cap=all(w.max_active <= cap+1 for w in self.workers)
            has_link=all(w.max_active >= 2 for w in self.workers)
            overlay_ok=(expected['undirected'] and expected['max_degree']<=cap and
                        within_cap and has_link)
            overlay_report={
                'configured_max_neighbors':cap,
                'expected_links':expected['links'],
                'expected_max_neighbors':expected['max_degree'],
                'symmetric':expected['undirected'],
                'per_peer_observed_max_direct':[max(0,w.max_active-1) for w in self.workers],
                'observed_max_direct_neighbors':max(0,max_active-1),
                'observed_cap_respected':within_cap,
                'all_peers_observed_connected':has_link,
                'note':'Actual observed counts are sampled from process status logs, not packet captures.'}
            self.note('overlay-verified' if overlay_ok else 'overlay-failed',
                      expected_links=expected['links'],
                      observed_max_direct=max(0,max_active-1),
                      cap=cap)
            all_ok=bool(join_ok and snapshot_ok and link_ok and churn_ok and
                        overlay_ok and settled and event_report['all_equal'] and
                        state_report['ok'] and
                        (epoch_probe_ok is None or epoch_probe_ok))
            result={'status':'PASS' if all_ok else 'FAIL', 'scenario':'random-churn',
                    'peers':a.peers,'seed':a.seed,'failed_and_rejoined':choices,
                    'real_transport':True,'embedded_signaling':a.embedded_signaling,
                    'duration_seconds':round(time.monotonic()-self.started,2),
                    'gossip':event_report,'model':state_report,'overlay':overlay_report,
                    'epoch_v2_probe':{'enabled':bool(getattr(a,'epoch_v2_probe',False)),
                                      'certified':epoch_probe_ok,
                                      'digests':[w.epoch_digest for w in self.workers],
                                      'note':'Signed no-op epoch consensus over real WebRTC; v1 training remains unchanged'},
                    'checks':{'initial_join':join_ok,'full_snapshot_for_new_peers':snapshot_ok,
                              'signed_hello_on_all_peers':link_ok,'churn_restarts':churn_ok,
                              'bounded_overlay_verified':overlay_ok,
                              'gossip_converged':settled,'same_update_inventory':event_report['all_equal'],
                              'effective_weights_within_tolerance':state_report['ok']},
                    'workers':[{'peer':w.number,'starts':w.starts,'joined':w.joined,
                        'snapshot_received':w.snapshot,'signed_hello_count':w.signed_hello,
                        'max_active':w.max_active,'max_living':w.max_living,
                        'disconnects_logged':w.disconnects,
                        'loss_first':w.losses[0] if w.losses else None,
                        'loss_last':w.losses[-1] if w.losses else None,
                        'local_events':w.emitted,'exit_codes':w.exits,
                        'errors':w.errors[-5:], 'last_phase':w.last_phase} for w in self.workers],
                    'events':self.events}
            # Optional controlled counterfactual: reaggregate a bounded sample
            # of the actual signed V1 trace against the immutable genesis.
            # This is diagnostic ONLY; V2 did not train or exchange proposals.
            if getattr(a,'v2_analysis',False):
                if event_report['all_equal'] and state_report['ok']:
                    try:
                        from .elastic_v2 import compare_legacy_swarm_trace
                        analysis=await asyncio.to_thread(compare_legacy_swarm_trace,self.root,
                            cap_per_peer=getattr(a,'v2_cap_per_peer',8))
                        result['elastic_v2_counterfactual']=analysis
                        (self.root/'elastic-v2-counterfactual.json').write_text(
                            json.dumps(analysis,indent=2),encoding='utf-8')
                        self.note('elastic-v2-counterfactual',loss=analysis['heldout_loss'],
                                  baseline=analysis['v1_final_heldout_loss'])
                    except Exception as exc:
                        result['elastic_v2_counterfactual']={'status':'ERROR',
                            'message':f'{type(exc).__name__}: {exc}'}
                        self.note('elastic-v2-analysis-error',message=str(exc))
                else:
                    result['elastic_v2_counterfactual']={'status':'SKIPPED',
                        'reason':'v1 swarm did not fully converge; no shared signed event set'}
            (self.root/'report.json').write_text(json.dumps(result,indent=2))
            self.note('result',status=result['status'],events=event_report['unique_events'],
                      max_abs=state_report.get('max_abs'),report=self.root/'report.json')
            return result
        except BaseException as exc:
            # Preserve evidence when a cold-join or recovery check times out.
            diag={'status':'FAIL','exception':f'{type(exc).__name__}: {exc}',
                  'events':self.events,
                  'workers':[{'peer':w.number,'pid':w.process.pid if w.process else None,
                    'starts':w.starts,'join_ready':w.joined,
                    'snapshot_ready':w.snapshot,'last_phase':w.last_phase,
                    'last_logs':w.log_tail[-15:]} for w in self.workers]}
            (self.root/'failure-report.json').write_text(json.dumps(diag,indent=2))
            raise
        finally:
            for i in range(a.peers):
                try:await self.stop(i)
                except Exception:pass
            if self.signal_server is not None:
                await self.signal_server.__aexit__(None,None,None)


def add_parser(sub):
    p=sub.add_parser('swarm-sim',help='2–16 real aiortc peers, random churn, gossip and weight verification')
    p.add_argument('--peers',type=int,default=8,help='Real peer processes (2..16)')
    p.add_argument('--max-neighbors',type=int,default=8,choices=(2,4,6,8),
                   help='Hard cap on direct WebRTC peers per process (default: 8)')
    p.add_argument('--allow-over-8',action='store_true',
                   help='Deprecated compatibility flag; no longer required for 16 peers')
    p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--faults',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--embedded-signaling',action='store_true',help='Use loopback websocket signaling; still real aiortc/ICE/DTLS/UDP')
    p.add_argument('--port',type=int,default=0,help='Embedded signaling loopback TCP port (0=auto)')
    p.add_argument('--ws',default='ws://127.0.0.1:8080/ws',help='Use real Mummy signaling (recommended)')
    p.add_argument('--bootstrap-batch-size',type=int,default=0,
                   help='Newcomers per verified startup wave (0=auto: 4 for >8 peers, legacy concurrent for <=8)')
    p.add_argument('--join-interval',type=float,default=1.5)
    p.add_argument('--startup-timeout',type=float,default=300)
    p.add_argument('--warmup',type=float,default=15)
    p.add_argument('--down-time',type=float,default=8)
    p.add_argument('--between-faults',type=float,default=10)
    p.add_argument('--settle-timeout',type=float,default=240)
    p.add_argument('--local-steps',type=int,default=4)
    p.add_argument('--epoch-v2-probe',action='store_true',help='Run a signed no-op Paxos epoch consensus barrier over real WebRTC after V1 Gossip settles (does NOT perform V2 training)')
    p.add_argument('--epoch-v2-timeout',type=float,default=90,help='Wait for every peer to obtain identical signed Epoch v2 certificate')
    p.add_argument('--v2-analysis',action='store_true',
                   help='After V1 E2E, analyze signed V1 events using experimental V2 formula (counterfactual, not V2 training)')
    p.add_argument('--v2-cap-per-peer',type=int,default=8,
                   help='Bounded signed proposals per peer in V2 counterfactual (1..64)')
    p.add_argument('--max-events-per-peer',type=int,default=0,
                   help='Cap each peer event generation in this convergence test (0=unlimited stress mode)')
    p.add_argument('--kind',choices=('lora','loha'),default='loha')
    p.add_argument('--rank',type=int,default=2)
    return p


def cli(args):
    for name in ('join_interval','startup_timeout','warmup','down_time','between_faults','settle_timeout','epoch_v2_timeout','max_events_per_peer'):
        if getattr(args,name)<0:raise SystemExit(f'--{name.replace("_","-")} must be nonnegative')
    if not 0 <= args.bootstrap_batch_size <= args.peers-1:
        raise SystemExit('--bootstrap-batch-size must be 0..peers-1')
    if not 1 <= args.v2_cap_per_peer <= 64:
        raise SystemExit('--v2-cap-per-peer must be in 1..64')
    import importlib.util
    needed=['torch','safetensors','transformers','websockets','aiortc','cryptography']
    missing=[m for m in needed if importlib.util.find_spec(m) is None]
    if missing:raise SystemExit('swarm-sim requires real dependencies: '+', '.join(missing)+
                                '; pip install -r requirements-swarm-sim.txt')
    try:
        result=asyncio.run(SwarmRunner(args).run())
    except (SimulationError, OSError, RuntimeError) as exc:
        print(f'SWARM FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(0 if result['status']=='PASS' else 1)
