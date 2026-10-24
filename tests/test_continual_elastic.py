"""Continual optimizer/LoHa semantics and historical replay regression checks."""
import asyncio
from copy import deepcopy
import hashlib

import pytest
import torch

from patchlib.core import PatchError, UniversalAdapter, decode_patch
from patchlib.continual import ContinualElasticEngine, PROTOCOL
from patchlib.elastic import EventJournal, pack_event, unpack_event
from patchlib.localprogress import restore_working, save_working, _tip
from test_livefed import Tiny


def make_engine(tmp_path, peer, model=None, rank=3):
    model = model if model is not None else Tiny()
    adapter = UniversalAdapter(model,'tiny','base',kind='loha',rank=rank)
    genesis=hashlib.sha256(b'continual-genesis-test').hexdigest()
    journal=EventJournal(tmp_path / (peer+'.sqlite3'),genesis)
    eng=ContinualElasticEngine(adapter,peer,genesis,journal,unsafe=True)
    return eng


def bases(eng):
    vals=[]
    for g in eng.adapter.groups:
        module,_,attr=g.aliases[0].rpartition('.')
        vals.append(eng.adapter.model.get_submodule(module).parametrizations[attr].original.detach().clone())
    return vals


def raw_factor_state(eng):
    return [p.detach().clone() for p in eng.adapter.trainable_parameters()]


def effective_delta_after(eng):
    return [g.overlay.local.delta().detach() - g.overlay.anchor.delta().detach() for g in eng.adapter.groups]


def loss_step(eng, optimizer):
    with eng.adapter.lock:
        loss=eng.adapter.model.forward()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_continual_keeps_factors_and_no_double_application(tmp_path):
    torch.manual_seed(99)
    eng=make_engine(tmp_path,'alice')
    opt=torch.optim.AdamW(list(eng.adapter.trainable_parameters()),lr=1e-3)
    for _ in range(3):
        loss_step(eng,opt)
    first=raw_factor_state(eng)
    state_steps=[float(opt.state[p]['step']) for p in eng.adapter.trainable_parameters()]
    before=bases(eng)
    event=eng.seal_local(1)
    after=bases(eng)
    assert any(not torch.equal(x,y) for x,y in zip(before,after))
    assert all(torch.allclose(x,torch.zeros_like(x),atol=1e-7) for x in effective_delta_after(eng))
    assert all(torch.equal(x,y) for x,y in zip(first,raw_factor_state(eng)))
    assert not eng.ingest(event)
    assert all(torch.equal(x,y) for x,y in zip(after,bases(eng)))
    for _ in range(2):
        loss_step(eng,opt)
    assert all(float(opt.state[p]['step'])==step+2 for p,step in zip(eng.adapter.trainable_parameters(),state_steps))
    eng.seal_local(1)
    assert all(torch.allclose(x,torch.zeros_like(x),atol=1e-7) for x in effective_delta_after(eng))
    eng.journal.close()


def test_continual_remote_replay_and_dedup(tmp_path):
    torch.manual_seed(3)
    template=Tiny()
    a=make_engine(tmp_path,'a',deepcopy(template)); b=make_engine(tmp_path,'b',deepcopy(template))
    c=make_engine(tmp_path,'c',deepcopy(template))
    oa=torch.optim.AdamW(list(a.adapter.trainable_parameters()),lr=0.003)
    ob=torch.optim.AdamW(list(b.adapter.trainable_parameters()),lr=0.003)
    loss_step(a,oa); wa=asyncio.run(a.seal_local_responsive(1))
    loss_step(b,ob); wb=b.seal_local(1)
    # Remote delivery must not mutate trainable factors and optimizer moments.
    saved=raw_factor_state(a)
    moments=[oa.state[p]['exp_avg'].clone() for p in a.adapter.trainable_parameters()]
    assert a.ingest(wb)
    assert b.ingest(wa)
    assert c.ingest(wb)
    assert c.ingest(wa)
    assert not a.ingest(wb)
    for i in range(len(saved)):
        assert torch.equal(saved[i],raw_factor_state(a)[i])
    for old,p in zip(moments,a.adapter.trainable_parameters()):
        assert torch.equal(old,oa.state[p]['exp_avg'])
    for x,y,z in zip(bases(a),bases(b),bases(c)):
        assert torch.allclose(x,y,rtol=1e-5,atol=1e-6)
        assert torch.allclose(x,z,rtol=1e-5,atol=1e-6)
    # A new event must carry v3 anchor/after factors, never v1 patch bodies.
    _,meta,weights,_=unpack_event(wa,{},unsafe=True)
    assert meta['elastic_protocol']==PROTOCOL
    assert any('.b' in k for k in weights)
    # Crash/restart: base+signed journal and working LoHa/AdamW resume.
    save_working(a.journal,a.adapter,oa,0,journal_tip=_tip(a.journal))
    torch.manual_seed(3)
    reboot=make_engine(tmp_path,'a2',deepcopy(template))
    for wire in a.journal.events():
        assert reboot.ingest(wire)
    for x,y in zip(bases(a),bases(reboot)):
        assert torch.allclose(x,y,atol=1e-6)
    a.journal.close();b.journal.close();c.journal.close();reboot.journal.close()


def test_continual_restore_after_crash_and_reject_legacy(tmp_path):
    torch.manual_seed(56)
    template=Tiny()
    engine=make_engine(tmp_path,'alice',deepcopy(template))
    opt=torch.optim.AdamW(list(engine.adapter.trainable_parameters()),lr=0.001)
    for _ in range(3):loss_step(engine,opt)
    engine.seal_local(1)
    save_working(engine.journal,engine.adapter,opt,0,journal_tip=_tip(engine.journal))
    old_factors=raw_factor_state(engine)
    old_states=[opt.state[p]['exp_avg'].clone() for p in engine.adapter.trainable_parameters()]
    old_base=bases(engine)
    engine.journal.close()
    newer=make_engine(tmp_path,'alice',deepcopy(template));newer.replay()
    restored_opt=torch.optim.AdamW(list(newer.adapter.trainable_parameters()),lr=0.001)
    assert restore_working(newer.journal,newer.adapter,restored_opt)==0
    for p,q in zip(old_factors,raw_factor_state(newer)):
        assert torch.equal(p,q)
    for old,p in zip(old_states,newer.adapter.trainable_parameters()):
        assert torch.equal(old,restored_opt.state[p]['exp_avg'])
    for p,q in zip(old_base,bases(newer)):
        assert torch.allclose(p,q,atol=1e-6)
    # Malformed payload marked v1 cannot be interpreted as continual.
    _,meta,weights,_=unpack_event(next(newer.journal.events()),{},unsafe=True)
    from patchlib.core import encode_patch
    meta['elastic_protocol']='fedpara-elastic-v1'
    corrupt=pack_event(encode_patch(meta,weights),'alice')
    with pytest.raises(PatchError,match='continual protocol'):
        newer._check(corrupt)
    newer.journal.close()
