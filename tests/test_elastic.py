import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from patchlib.core import PatchError, UniversalAdapter
from patchlib.elastic import ElasticEngine, EventJournal, unpack_event
from patchlib.elasticnet import encode_control, decode_control, read_roster
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from test_livefed import Tiny


def make_pair(tmp_path, kind='loha'):
    torch.manual_seed(40)
    model=Tiny()
    genesis=hashlib.sha256(b'base checkpoint').hexdigest()
    private={}; public={}; signer={}; verify={}
    for p in ('alice','bob','carol'):
        private[p]=tmp_path/f'{p}.key';public[p]=tmp_path/f'{p}.pub'
        generate_keypair(private[p],public[p])
        signer[p]=signer_from_pem(private[p]);verify[p]=verifier_from_pem(public[p])
    engines={}
    for p in ('alice','bob','carol'):
        adapter=UniversalAdapter(deepcopy(model),'one-base','fstate-sha256:seed',kind=kind,rank=3)
        journal=EventJournal(tmp_path/f'{p}.db',genesis,max_bytes=1_000_000)
        engines[p]=ElasticEngine(adapter,p,genesis,journal,signer=signer[p],trusted=verify)
    return engines,signer,verify


def perturb(engine, val):
    with torch.no_grad():
        for g in engine.adapter.groups:
            for i,t in enumerate(g.overlay.local.parts):
                t.fill_(val + i * .002)


def originals(engine):
    out=[]
    for g in engine.adapter.groups:
        module,_,attr=g.aliases[0].rpartition('.')
        out.append(engine.adapter.model.get_submodule(module).parametrizations[attr].original.clone().detach())
    return out


@pytest.mark.parametrize('kind',['lora','loha'])
def test_permanent_departure_and_rejoin_consistent_additive_updates(tmp_path,kind):
    eng, _, _ = make_pair(tmp_path,kind)
    alice,bob,carol=(eng[x] for x in ('alice','bob','carol'))
    initial=originals(alice)
    # All start from same genesis. Carol disappears indefinitely.
    perturb(alice,.1); a1=alice.seal_local(2)
    perturb(bob,.12); b1=bob.seal_local(2)
    bob.ingest(a1); alice.ingest(b1)
    # No quorum wait: live peer count becomes 1 and work keeps progressing.
    perturb(alice,.15); a2=alice.seal_local(1)
    perturb(bob,.16); b2=bob.seal_local(1)
    bob.ingest(a2);alice.ingest(b2)
    # Carol joins much later by obtaining the ledger's missing events.
    for wire in (b2,a1,a2,b1):
        assert carol.ingest(wire)
    assert not carol.ingest(a1)
    for a,b in zip(originals(alice),originals(carol)):
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    for a,b in zip(originals(alice),originals(bob)):
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    # Reconstruct after a crash, without trusting or executing saved code.
    restored=UniversalAdapter(Tiny(),'one-base','fstate-sha256:seed',kind=kind,rank=3)
    # Model must start from precisely the same genesis weights.
    # Tiny() random initialization differs, so restore before replay below.
    for g,orig in zip(restored.groups,initial):
        m,a=g.aliases[0].rpartition('.')[0],g.aliases[0].rpartition('.')[-1]
        restored.model.get_submodule(m).parametrizations[a].original.data.copy_(orig)
    restart=ElasticEngine(restored,'carol',carol.genesis,carol.journal,
                          signer=carol.signer,trusted=carol.trusted)
    restart.replay()
    for a,b in zip(originals(alice),originals(restart)):
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    assert restart.journal.count()==4


def test_replay_equivocation_genesis_and_signature(tmp_path):
    engines,_,keys=make_pair(tmp_path)
    a,b=engines['alice'],engines['bob']
    perturb(a,.1)
    e=a.seal_local(2)
    assert b.ingest(e)
    assert not b.ingest(e)
    corrupt=e[:-1]+bytes([e[-1]^7])
    with pytest.raises(PatchError):b.ingest(corrupt)
    head,m,_,_=unpack_event(e,keys)
    assert head=='alice' and m['denominator']==2
    with pytest.raises(PatchError,match='genesis'):
        EventJournal(tmp_path/'bob.db',hashlib.sha256(b'other').hexdigest())
    with pytest.raises(PatchError,match='untrusted'):
        unpack_event(e,{})


def test_log_hard_cap_and_inventory(tmp_path):
    eng,_,_=make_pair(tmp_path)
    a=eng['alice']
    perturb(a,.1); e=a.seal_local(1)
    rows, done=a.journal.inventory(0,64)
    assert len(rows)==1 and done
    assert rows[0][:2]==['alice',1]
    with pytest.raises(PatchError,match='full'):
        a.journal.max_bytes = len(e)+10
        perturb(a,.12)
        a.seal_local(1)
    assert a.journal.count()==1


def test_control_signature_tampering_and_counter(tmp_path):
    _, signers, keys=make_pair(tmp_path)
    wire=encode_control({'type':'heartbeat','peer':'alice','genesis':'a'*64,'counter':6},signers['alice'])
    assert decode_control(wire,keys['alice'])['counter']==6
    with pytest.raises(PatchError):decode_control(wire.replace('alice','alice!'),keys['alice'])
    with pytest.raises(PatchError):decode_control(wire,keys['bob'])


def test_roster_file_late_join(tmp_path):
    roster=tmp_path/'roster.json'
    roster.write_text('{"bob":{"room":"abcdefghijklmnop","public_key":"bob.pem"}}')
    assert read_roster(roster,'alice')['bob'][0]=='abcdefghijklmnop'
    roster.write_text('{"alice":{"room":"abcdefghijklmnop"}}')
    with pytest.raises(ValueError):read_roster(roster,'alice')


@pytest.mark.parametrize("reinit_every",[0,1,20])
def test_real_train_loop_with_permanently_absent_peer(tmp_path,monkeypatch,reinit_every):
    import asyncio
    from types import SimpleNamespace
    from patchlib import elasticnet
    # Simulate a permanently unavailable remote; no actual RTC dependencies.
    async def always_disconnected(self):
        raise ConnectionError('offline forever')
    monkeypatch.setattr(elasticnet.ElasticNode,'run',always_disconnected)
    model=Tiny()
    initial=deepcopy(model)
    def load(_):
        return model,UniversalAdapter(model,'one-base','fstate-sha256:seed',rank=3)
    def batches(_):
        while True: yield None
    def step(m,adapt,opt,batch,args):
        with adapt.lock:
            loss=m.forward()
            loss.backward()
            return loss.item()
    argv=SimpleNamespace(
        trust_remote_code=False,factory_file=None,allow_legacy_bin=False,resume=None,
        state_base='initial.fstate',grad_accum=1,local_steps=2,rounds=3,
        reinit_every=reinit_every,
        signing_key=None,unsafe_unsigned=True,ws='ws://127.0.0.1:8080/ws',
        link=['bob=pair_room_123456789'],peer_id='alice',trusted_peer_key=[],
        roster_file=None,model_id='one-base',journal=tmp_path/'online.db',
        self_public_key=None,max_journal_gb=1,max_delta_rms=1000.,
        lr=.0001,weight_decay=0.,device='cpu',wire_dtype='float16',
        token='',max_patch_mb=5,peer_timeout=.2,heartbeat_interval=.05,
        boot_grace=0.,sync_timeout=.1,inventory_interval=.05,
        inventory_cycle=.1,shard_state_manifest=None,shard_store=None,
        full_state_output=None,log_steps=100,peers=None,checkpoint_every=2,checkpoint_dir=tmp_path/'ck',shard_mb=4,max_state_gb=1)
    asyncio.run(elasticnet.train_elastic(argv,load_model=load,training_batches=batches,run_step=step))
    j=EventJournal(argv.journal,hashlib.sha256(
        __import__('patchlib.core',fromlist=['_canonical_json'])._canonical_json({
        'model_id':'one-base','revision':'fstate-sha256:seed',
        'layout':UniversalAdapter(initial,'one-base','fstate-sha256:seed',rank=3).layout,
        'kind':'loha','rank':3,'alpha':3.0,
        'optimizer_protocol':'continual-elastic-fedsum-relora-v4',
        'reinit_every':reinit_every})).hexdigest())
    assert j.count()==3
    assert len(list((tmp_path/"ck").glob("*.fmanifest")))>=1
    assert len(list((tmp_path/"ck").glob("*.checkpoint.json")))>=1
    for wire in j.events():
        _,meta,_,_=unpack_event(wire,{},unsafe=True)
        assert meta['denominator']==1
        assert meta['reinit_after'] == bool(reinit_every and meta['seq'] % reinit_every == 0)
    j.close()


def test_authenticated_fake_datachannel_gossip(tmp_path):
    import asyncio
    from patchlib.elasticnet import ElasticNode
    class Channel:
        readyState='connecting';bufferedAmount=0
        def __init__(self):
            self.events={};self.other=None
        def on(self,name):
            def reg(fn):self.events[name]=fn;return fn
            return reg
        def send(self,payload):self.other.events['message'](payload)
        def close(self):
            self.readyState='closed'
            if 'close' in self.events:self.events['close']()
    async def main():
        engines,signers,verifiers=make_pair(tmp_path)
        a=ElasticNode('ws://localhost/ws','pair_room_123456789',origin='alice',remote='bob',genesis=engines['alice'].genesis,
                      signer=signers['alice'],verifier=verifiers['bob'])
        b=ElasticNode('ws://localhost/ws','pair_room_123456789',origin='bob',remote='alice',genesis=engines['bob'].genesis,
                      signer=signers['bob'],verifier=verifiers['alice'])
        ac=Channel();bc=Channel();ac.other=bc;bc.other=ac
        a._attach_channel(ac);b._attach_channel(bc)
        ac.readyState=bc.readyState='open'
        ac.events['open']();bc.events['open']()
        await asyncio.sleep(.01)
        assert a.controls.get_nowait()['type']=='hello'
        assert b.controls.get_nowait()['type']=='hello'
        perturb(engines['alice'],.09)
        wire=engines['alice'].seal_local(2)
        await a.outgoing.put(wire)
        got=await asyncio.wait_for(b.incoming.get(),3)
        assert engines['bob'].ingest(got)
        ac.close();bc.close()
        await asyncio.sleep(.01)
    asyncio.run(main())


def test_journal_cannot_change_local_peer_identity(tmp_path):
    eng,_,_=make_pair(tmp_path)
    with pytest.raises(PatchError,match='different local peer'):
        ElasticEngine(eng['alice'].adapter,'other',eng['alice'].genesis,eng['alice'].journal,unsafe=True)
