"""ReLoRA v4 properties: no lost learning, signed reset cadence, crash replay."""
import hashlib
from copy import deepcopy

import pytest
import torch

from patchlib.core import PatchError, UniversalAdapter, decode_patch, encode_patch
from patchlib.continual import ContinualElasticEngine, PROTOCOL
from patchlib.elastic import EventJournal, pack_event, unpack_event
from patchlib.effectiveness_benchmark import run_effectiveness_benchmark
from patchlib.relora import merge_and_reset_central, reset_committed_factors, reinit_seed
from patchlib.localprogress import save_working, restore_working, _tip
from test_livefed import Tiny


def engine(tmp_path, peer, base, every=2):
    model=deepcopy(base)
    ada=UniversalAdapter(model,'test','init',kind='loha',rank=3)
    genesis=hashlib.sha256(f'relora-test-{every}'.encode()).hexdigest()
    log=EventJournal(tmp_path/(peer+'.sqlite3'),genesis)
    return ContinualElasticEngine(ada,peer,genesis,log,reinit_every=every,unsafe=True)


def output(eng):
    with torch.no_grad():
        return eng.adapter.model.proj(eng.adapter.model.embed(torch.tensor([[0,1,2]]))).clone()


def step(eng, opt):
    with eng.adapter.lock:
        torch.manual_seed(190)
        loss=eng.adapter.model()
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)


def opt_for(eng):
    return torch.optim.AdamW(list(eng.adapter.trainable_parameters()),lr=.002)


def test_reinit_at_exact_signed_event_boundary_and_replay(tmp_path):
    torch.manual_seed(9)
    base=Tiny()
    a=engine(tmp_path,'alice',base,2)
    observer=engine(tmp_path,'bob',base,2)
    opt=opt_for(a)
    step(a,opt)
    e1=a.seal_local(1)
    assert any(float(opt.state[p]['step'])>0 for p in a.adapter.trainable_parameters())
    _,m1,_,_=unpack_event(e1,{},unsafe=True)
    assert m1['reinit_after'] is False and m1['reinit_every']==2
    first=output(a)
    first_factors=[p.clone() for p in a.adapter.trainable_parameters()]
    step(a,opt)
    before_refresh=output(a)
    e2=a.seal_local(1)
    assert output(a).allclose(before_refresh,atol=1e-5,rtol=1e-5)
    assert not all(torch.equal(x,p) for x,p in zip(first_factors,a.adapter.trainable_parameters()))
    for group in a.adapter.groups:
        for j,p in enumerate(group.overlay.local.parts):
            assert torch.equal(p,getattr(group.overlay.anchor,f'part{j}'))
    _,m2,_,_=unpack_event(e2,{},unsafe=True)
    assert m2['reinit_after'] is True and m2['elastic_protocol']==PROTOCOL
    assert observer.ingest(e1) and observer.ingest(e2)
    # The observer must only add weights, not reset ITS OWN factors.
    for x,y in zip(output(a).flatten(),output(observer).flatten()):
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
    assert not observer.ingest(e2)
    # A new optimizer starts with NO obsolete moments; subsequent event uses
    # new anchored factors and does not resend previously learned deltas.
    opt=opt_for(a)
    assert not opt.state
    step(a,opt)
    e3=a.seal_local(1)
    assert observer.ingest(e3)
    torch.testing.assert_close(output(a),output(observer),rtol=1e-5,atol=1e-5)
    old_out=output(a)
    a.journal.close()
    # Replay signed events from immutable base; own event 2 reproduces reset
    # using deterministic seed, even without working checkpoint.
    resumed=engine(tmp_path,'alice',base,2)
    resumed.replay()
    torch.testing.assert_close(output(resumed),old_out,rtol=1e-5,atol=1e-5)
    assert all(torch.equal(x,y) for x,y in zip(a.adapter.trainable_parameters(),resumed.adapter.trainable_parameters()))
    observer.journal.close();resumed.journal.close()


def test_reset_after_commit_requires_no_pending_local_effect(tmp_path):
    torch.manual_seed(11)
    eng=engine(tmp_path,'alice',Tiny(),1)
    opt=opt_for(eng)
    step(eng,opt)
    with pytest.raises(PatchError,match='uncommitted'):
        reset_committed_factors(eng.adapter,seed=11)
    before=output(eng)
    eng.seal_local(1)
    assert torch.allclose(output(eng),before,rtol=1e-5,atol=1e-5)
    old_out=output(eng)
    reset_committed_factors(eng.adapter,seed=11)
    assert torch.allclose(output(eng),old_out,rtol=1e-5,atol=1e-5)
    eng.journal.close()


def test_signed_refresh_flag_cannot_be_forged(tmp_path):
    eng=engine(tmp_path,'alice',Tiny(),2)
    step(eng,opt_for(eng))
    wire=eng.seal_local(1)
    _,meta,parts,_=unpack_event(wire,{},unsafe=True)
    meta['reinit_after']=True  # Not a boundary; signed metadata is invalid.
    wrong=pack_event(encode_patch(meta,parts),'alice')
    with pytest.raises(PatchError,match='invalid continual'):
        eng._check(wrong)
    eng.journal.close()


def test_unanchored_central_merge_preserves_predictions():
    torch.manual_seed(14)
    ada=UniversalAdapter(Tiny(),'test','init',rank=3)
    opt=torch.optim.AdamW(ada.trainable_parameters(),lr=.001)
    torch.manual_seed(17)
    loss=ada.model(); loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        a=ada.model.proj(ada.model.embed(torch.tensor([[0,1,2]]))).clone()
    before=[g.overlay.local.delta().detach().clone() for g in ada.groups]
    merge_and_reset_central(ada,seed=88)
    with torch.no_grad():
        b=ada.model.proj(ada.model.embed(torch.tensor([[0,1,2]]))).clone()
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    assert any(x.abs().max()>0 for x in before)
    for group in ada.groups:
        torch.testing.assert_close(group.overlay.local.delta(),torch.zeros_like(group.overlay.local.delta()))
    opt2=torch.optim.AdamW(ada.trainable_parameters(),lr=.001)
    assert len(opt2.state)==0


def test_offline_refresh_cadence_and_comparison(tmp_path):
    cfg=dict(peers=2,rounds=3,local_steps=1,rank=4,seed=99,
             output_dir=tmp_path,reinit_every=2)
    result=run_effectiveness_benchmark(**cfg)
    assert result['configuration']['reinit_every']==2
    assert [r['relora_reinitialized'] for r in result['trace']]==[False,True,False]
    assert result['fairness']['identical_training_samples']
    off=run_effectiveness_benchmark(**{**cfg,'output_dir':None,'reinit_every':0})
    assert not any(r['relora_reinitialized'] for r in off['trace'])
    assert result['loss']['p2p']!=off['loss']['p2p']


def test_reset_seed_independent_rng():
    torch.manual_seed(33)
    x=torch.rand(1)
    torch.manual_seed(33)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reinit_seed('a','b',10))
        _=torch.rand(100)
    torch.testing.assert_close(x,torch.rand(1))


def test_gui_live_and_benchmark_expose_frequency(tmp_path):
    from patchlib.webview_controller import WebviewController
    from patchlib.webview_gui import compose_html
    html=compose_html()
    assert 'id="trainReinitEvery"' in html  # Current UI has no benchmark form.
    ctl=WebviewController(tmp_path)
    with pytest.raises(ValueError,match='reinit_every'):
        ctl.start_benchmark({'reinit_every':-2})
    ctl.close()
