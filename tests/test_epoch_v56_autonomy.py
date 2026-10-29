import asyncio
import hashlib
import sqlite3
import pytest
import torch

from patchlib.core import PatchError,_canonical_json
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.epoch_certificate_v46 import CompactEpochLedger, proposal_v2
from patchlib.epoch_autonomous_v56 import AutonomousPagedEpoch
from patchlib.frontier_commitment_v44 import build_frontier_pages
from patchlib.online_generation import GenerationLedger, AutomaticGenerationControl
from patchlib.live_generation_transport import GenerationWebRTCBridge
from patchlib.shards import export_state_sharded, read_manifest, semantic_shard_sha256
from tests.test_v44_low_cost_generation import DummyNode


def make_network(tmp_path,origins=4096):
    names=['a','b','c'];signers={};verifiers={}
    for name in names:
        prv=tmp_path/f'{name}.pem';pub=tmp_path/f'{name}.pub'
        generate_keypair(prv,pub)
        signers[name]=signer_from_pem(prv);verifiers[name]=verifier_from_pem(pub)
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT)')
    hashes=[(f'p{i:05d}',1,hashlib.sha256(f'{i}'.encode()).hexdigest()) for i in range(origins)]
    db.executemany('INSERT INTO events VALUES(?,?,?)',hashes)
    inv=hashlib.sha256()
    for r in db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        inv.update(_canonical_json(list(r))+b'\n')
    h,pages=build_frontier_pages({p:1 for p,_,_ in hashes})
    path=tmp_path/'snapshot.fmanifest';store=tmp_path/'snapshot'
    export_state_sharded(torch.nn.Linear(2,3),store,path,model_id='tiny',revision='v1',peer_id='a',seq=1,shard_bytes=65536)
    p=proposal_v2(room='room',epoch=1,parent='0'*64,
        manifest=hashlib.sha256(path.read_bytes()).hexdigest(),
        semantic=semantic_shard_sha256(read_manifest(path)),inventory=inv.hexdigest(),
        frontier_header=h,replicas=names)
    ledgers={};new={};old={};bridges={}
    for name in names:
        ledgers[name]=CompactEpochLedger(tmp_path/f'{name}.v2',room='room',peer=name,
            committee=names,signer=signers[name],trusted=verifiers,genesis='0'*64)
        old[name]=GenerationLedger(tmp_path/f'{name}.v1',room='room',peer=name,
            committee=names,signer=signers[name],trusted=verifiers,genesis='0'*64)
        new[name]=AutonomousPagedEpoch(ledgers[name],tmp_path/f'{name}.stage',
            local_checkpoint=lambda _,p=path,s=store,d=db:(p,s,d),
            local_export_verified=lambda _:True)
        bridges[name]=GenerationWebRTCBridge(AutomaticGenerationControl(old[name],lambda _:False,lambda _:False),
            interval=.05,compact_ledger=ledgers[name],compact_control=new[name])
    return p,pages,db,old,new,ledgers,bridges


def deliver(bridges,a,b,now):
    node=DummyNode();bridges[a].tick([(b,node)],now=now)
    for kind,data in node.wires:
        bridges[b].receive(a,kind,data)
    return node.wires


def test_4096_origin_autonomous_paged_proposal_vote_gossip(tmp_path):
    p,pages,db,old,new,ledgers,bridges=make_network(tmp_path)
    try:
        assert p['frontier_header']['pages']==128
        new['a'].offer(p,pages)
        assert ledgers['a'].latest() is None
        for i in range(340):
            t=i+1.
            deliver(bridges,'a','b',t)
            deliver(bridges,'b','a',t)
            if ledgers['a'].latest() is not None and ledgers['b'].latest() is not None:
                break
        assert ledgers['a'].latest()['proposal']==p
        assert ledgers['b'].latest()['proposal']==p
        # A long-disconnected third peer needs only the signed chain, not all pages.
        deliver(bridges,'a','c',400)
        deliver(bridges,'c','a',401)
        deliver(bridges,'a','c',402)
        assert ledgers['c'].certified_epoch(1)['proposal']==p
    finally:
        for v in new.values():v.close()
        for v in ledgers.values():v.close()
        for v in old.values():v.close()
        db.close()


def test_compact_page_spoof_and_conflicts_rejected(tmp_path):
    p,pages,db,old,new,ledgers,bridges=make_network(tmp_path,96)
    try:
        d=new['a'].offer(p,pages)
        endorsed=new['a'].announcements()[0]
        bridges['b'].receive('a','generation-wire',{'payload':endorsed})
        bad=dict(pages[0]);bad['entries']=[['z',1]]
        with pytest.raises(PatchError):
            new['b'].receive('a',{'kind':'compact-page','data':{'digest':d,'page':bad}})
        with pytest.raises(PatchError):
            new['b'].receive('a',{'kind':'compact-page-need','data':{'digest':d,'index':99999}})
    finally:
        for v in new.values():v.close()
        for v in ledgers.values():v.close()
        for v in old.values():v.close()
        db.close()
