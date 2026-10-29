import hashlib
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch

from patchlib.core import PatchError,_canonical_json
from patchlib.epoch_export_v57 import export_at_step_boundary,verify_receipt
from patchlib.epoch_compact_trainer_v57 import CertifiedCompactTrainerLedger
from patchlib.epoch_autonomous_v56 import AutonomousPagedEpoch
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.epoch_certificate_v46 import CompactEpochLedger
from patchlib.online_generation import _digest
from tests.test_epoch_v56_autonomy import make_network,deliver


def test_compact_cert_only_is_not_a_trainer_cut(tmp_path):
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=4096)
    try:
        controls['a'].offer(p,pages)
        for i in range(400):
            deliver(bridges,'a','b',i+1.0)
            deliver(bridges,'b','a',i+1.0)
            if ledgers['a'].latest() and ledgers['b'].latest():break
        assert ledgers['a'].latest()
        adapter=CertifiedCompactTrainerLedger(ledgers['a'],controls['a'])
        cp=adapter.certified_epoch(1)['proposal']
        assert cp['cut']==p['inventory']
        assert cp['cutoffs']['p04095']==1
        assert cp['protocol']=='kagami-local-compact-trainer-projection-v57'
        assert cp['compact_proposal_digest']==_digest(p)
        # Recipient C received only the SIGNED certificate, not Merkle pages.
        deliver(bridges,'a','c',500)
        deliver(bridges,'c','a',501)
        deliver(bridges,'a','c',502)
        assert ledgers['c'].latest()
        late=CertifiedCompactTrainerLedger(ledgers['c'],controls['c'])
        assert late.certified_epoch(1) is None
        assert controls['c'].missing_pages(1)[0]['data']['index']==0
        # Even a forged page insertion cannot replace a committed Merkle leaf.
        with pytest.raises(PatchError):
            controls['c'].receive('a',{'kind':'compact-page','data':
                {'digest':_digest(p),'page':{'index':0,'entries':[['evil',1]],'siblings':[]}}})
        # Download all original pages (bounded message-by-message transfer).
        digest=_digest(p)
        for page in pages:controls['c'].receive('a',{'kind':'compact-page',
                                'data':{'digest':digest,'page':page}})
        assert late.certified_epoch(1)['proposal']==cp
        assert controls['c'].purge_committed()==0
        assert late.certified_epoch(1)['proposal']==cp
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()


def test_exporter_signed_cut_is_bound_to_actual_local_manifest(tmp_path):
    keys={};signers={};trusted={}
    for peer in ('a','b','c'):
        prv=tmp_path/(peer+'.key');pub=tmp_path/(peer+'.pub')
        generate_keypair(prv,pub)
        signers[peer]=signer_from_pem(prv)
        trusted[peer]=verifier_from_pem(pub)
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT, PRIMARY KEY(origin,seq))')
    db.executemany('INSERT INTO events VALUES(?,?,?)',[
        ('a',1,hashlib.sha256(b'event1').hexdigest()),
        ('b',1,hashlib.sha256(b'event2').hexdigest())])
    class Source:
        def __init__(self,db):self.db=db
        def count(self):return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    source=Source(db)
    model=torch.nn.Linear(4,2)
    adapter=SimpleNamespace(model_id='tiny',revision='r',groups=[],lock=threading.RLock())
    p,pages,receipt=export_at_step_boundary(journal=source,model=model,adapter=adapter,
        store=tmp_path/'objects',incoming=tmp_path/'incoming',room='room',epoch=1,
        parent='0'*64,replicas=['a','b','c'],signer=signers['a'],peer='a',
        shard_bytes=65536,max_state_bytes=1024**2)
    assert verify_receipt(receipt,p,trusted=trusted,committee=['a','b','c'])
    assert (tmp_path/'incoming'/(p['manifest']+'.fmanifest')).exists()
    with pytest.raises(PatchError):verify_receipt(receipt,{**p,'inventory':'f'*64},trusted=trusted,committee=['a','b','c'])
    # A hole is forbidden even if a max-frontier would have suggested coverage.
    db.execute('INSERT INTO events VALUES(?,?,?)',('a',3,'a'*64))
    with pytest.raises(PatchError,match='noncontiguous'):
        export_at_step_boundary(journal=source,model=model,adapter=adapter,
            store=tmp_path/'objects',incoming=tmp_path/'incoming',room='room',epoch=1,
            parent='0'*64,replicas=['a','b','c'],signer=signers['a'],peer='a',
            shard_bytes=65536,max_state_bytes=1024**2)
    db.close()


def test_receipt_cannot_be_forged_by_an_unpinned_peer(tmp_path):
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=2)
    try:
        with pytest.raises(PatchError):
            controls['b'].receive_export_receipt({'payload':{},'peer':'evil','signature':'bad'},p)
        assert controls['b'].has_export_receipt(p) is False
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()

@pytest.mark.asyncio
async def test_authenticated_v2_projection_reaches_step_boundary_service(tmp_path):
    from patchlib.epoch_hot_swap_v45 import TrainerGeneration,TrainerStepRouter
    from patchlib.live_epoch_switch_v48 import (LiveEpochSwitchService,PreparedEpoch,
                                                 certified_genesis)
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=400)
    try:
        controls['a'].offer(p,pages)
        for i in range(100):
            deliver(bridges,'a','b',i+1.0)
            deliver(bridges,'b','a',i+1.0)
            if ledgers['a'].latest():break
        ledger=CertifiedCompactTrainerLedger(ledgers['a'],controls['a'])
        cp=ledger.certified_epoch(1)['proposal']
        genesis=certified_genesis(cp)
        prior=TrainerGeneration(0,'0'*64,'0'*64,object(),object(),object(),
                                SimpleNamespace(genesis='0'*64))
        after=TrainerGeneration(1,genesis,p['manifest'],object(),object(),object(),
                                SimpleNamespace(genesis=genesis))
        phases=[]
        async def shadow(candidate):
            phases.append('prepare')
            assert candidate==cp
            return PreparedEpoch(1,_digest(cp),p['manifest'],after,
                tmp_path/'placeholder',tmp_path/'objects')
        async def flush(_):phases.append('flush')
        async def carry(*_):phases.append('carry')
        async def attest(*_):phases.append('attest');return True
        async def transport(*_):phases.append('transport');return True
        service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(prior),
            root=tmp_path/'activation',build_shadow=shadow,validate_shards=False,
            verify_local=lambda *_:True,flush_unpublished=flush,carry_suffix=carry,
            transport_handoff=transport,attest_journal=attest)
        assert await service.activate_next()
        assert service.router.current is after
        assert phases==['prepare','flush','carry','attest','transport']
        assert not await service.activate_next()
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()


def test_deferring_votes_keeps_p2p_callbacks_nonblocking(tmp_path):
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=64)
    try:
        # The production owner now defers expensive full-shard verification.
        c=AutonomousPagedEpoch(ledgers['a'],tmp_path/'deferred.sqlite3',
            local_checkpoint=lambda _: (_ for _ in ()).throw(AssertionError('blocking I/O')),
            local_export_verified=lambda _:True,defer_votes=True)
        try:
            c.offer(p,pages) if False else None
            # A trusted signed candidate may be stored and forwarded without
            # a blocking vote (actual vote happens in a worker).
            controls['a'].offer(p,pages)
            c.receive('a',controls['a'].announcements()[0])
            for page in pages:c.receive('a',{'kind':'compact-page','data':{'digest':_digest(p),'page':page}})
            assert ledgers['a'].latest() is None
        finally:c.close()
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()

def test_compact_snapshot_transfer_is_requested_only_for_committed_manifest(tmp_path):
    from patchlib.live_generation_transport import validate_control
    from tests.test_v44_low_cost_generation import DummyNode
    p,pages,db,old,controls,ledgers,bridges=make_network(tmp_path,origins=64)
    try:
        controls['a'].offer(p,pages)
        for i in range(100):
            deliver(bridges,'a','b',i+1.)
            deliver(bridges,'b','a',i+1.)
            if ledgers['a'].latest() and ledgers['b'].latest():break
        assert ledgers['a'].latest() is not None
        bridges['a'].snapshot_available=lambda m:m==p['manifest']
        bridges['b'].snapshot_available=lambda m:False
        avail=DummyNode();bridges['a'].tick([('b',avail)],now=1000.)
        assert any(t=='generation-compact-have' and d['manifest']==p['manifest'] for t,d in avail.wires)
        for kind,fields in avail.wires:
            if kind=='generation-compact-have':
                bridges['b'].receive('a',kind,fields)
        request=DummyNode();bridges['b'].tick([('a',request)],now=1001.)
        assert any(t=='generation-compact-snapshot-request' and d['manifest']==p['manifest']
                   for t,d in request.wires)
        with pytest.raises(PatchError):
            validate_control('generation-compact-snapshot-request',
                             {'epoch':1,'manifest':'not-a-digest'})
    finally:
        for x in controls.values():x.close()
        for x in ledgers.values():x.close()
        for x in old.values():x.close()
        db.close()
