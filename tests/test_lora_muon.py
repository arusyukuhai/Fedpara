"""Paper Algorithm 1 / arXiv:2606.12921 regression tests."""
import math
import torch
import pytest

from patchlib.core import UniversalAdapter
from patchlib.lora_muon import LoRAMuon, gram_inverse_root, matrix_sign


def _adapter(seed=42):
    torch.manual_seed(seed)
    model=torch.nn.Linear(32,24,bias=False)
    return UniversalAdapter(model,'unit','unit',kind='lora',rank=16),model


def test_polar_express_matches_svd_matrix_sign():
    torch.manual_seed(0)
    for shape in ((24,16),(16,24),(16,16)):
        x=torch.randn(shape)
        actual=matrix_sign(x)
        exact=matrix_sign(x,method='svd')
        assert torch.allclose(actual,exact,atol=.035,rtol=.035)
    assert not matrix_sign(torch.zeros(24,16)).any()


def test_gram_root_and_singular_initialization():
    torch.manual_seed(2)
    x=torch.randn(24,16)
    root=gram_inverse_root(x,relative_epsilon=1e-7)
    whiten=root@(x.T@x)@root
    assert torch.allclose(whiten,torch.eye(16),atol=3e-3,rtol=3e-3)
    assert torch.isfinite(gram_inverse_root(torch.zeros_like(x))).all()


def test_muon_works_with_zero_initialized_lora_and_decreases_loss():
    adapter,model=_adapter()
    optimizer=LoRAMuon(adapter,lr=.002,momentum=.9,weight_decay=.01)
    x=torch.randn(96,32)
    with torch.no_grad():
        target=model(x)+.025*torch.randn(96,24)
    initial=((model(x)-target)**2).mean().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss=((model(x)-target)**2).mean()
        loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert ((model(x)-target)**2).mean().item() < initial
    assert len(optimizer.state)==2
    for state in optimizer.state.values():
        assert 'momentum' in state and 'exp_avg_sq' not in state
        assert state['step']==20


def test_muon_factor_update_is_gauge_equivariant():
    adapter1,_=_adapter(1); adapter2,_=_adapter(1)
    a1,b1=adapter1.groups[0].overlay.local.parts
    a2,b2=adapter2.groups[0].overlay.local.parts
    with torch.no_grad():
        # Need both sides full-rank, not paper's rank-deficient initial point.
        b1.normal_(0,.1);b2.copy_(b1)
        scale=2.5
        a2.copy_(a1*scale);b2.copy_(b1/scale)
    opt1=LoRAMuon(adapter1,lr=.004,momentum=.8,weight_decay=.03,orthogonalizer='svd',inverse_root_eps=1e-8)
    opt2=LoRAMuon(adapter2,lr=.004,momentum=.8,weight_decay=.03,orthogonalizer='svd',inverse_root_eps=1e-8)
    ga=torch.randn_like(a1)*.001;gb=torch.randn_like(b1)*.001
    a1.grad=ga;b1.grad=gb
    a2.grad=ga/scale;b2.grad=gb*scale
    opt1.step();opt2.step()
    assert torch.allclose(a1@b1.T,a2@b2.T,rtol=.03,atol=1e-3)


def test_muon_optimizer_state_survives_checkpoint(tmp_path):
    a,m=_adapter()
    o=LoRAMuon(a,lr=.004)
    x=torch.randn(8,32)
    (m(x).square().mean()).backward();o.step()
    saved=o.state_dict()
    torch.save(saved,tmp_path/'locally_created_state.pth')
    b,m2=_adapter()
    o2=LoRAMuon(b,lr=.004)
    o2.load_state_dict(torch.load(tmp_path/'locally_created_state.pth',weights_only=True))
    for p1,p2 in zip(o.param_groups[0]['params'],o2.param_groups[0]['params']):
        assert torch.equal(o.state[p1]['momentum'],o2.state[p2]['momentum'])
    assert saved['state']


def test_reject_loha_and_dense_fallback():
    m=torch.nn.Linear(32,24,bias=False)
    adapter=UniversalAdapter(m,'unit','unit',kind='loha',rank=16)
    with pytest.raises(ValueError,match='LoHa'):
        LoRAMuon(adapter)
    adapter2=UniversalAdapter(torch.nn.Linear(32,24), 'unit','unit',kind='lora',rank=16)
    with pytest.raises(ValueError,match='target-scope'):
        LoRAMuon(adapter2)


def test_tinystories_stub_muon_benchmark(monkeypatch):
    from test_tinystories_1m_effectiveness import install_fake_hf
    from patchlib.effectiveness_benchmark import run_effectiveness_benchmark
    install_fake_hf(monkeypatch)
    out=run_effectiveness_benchmark(model='tinystories-1m',dataset='builtin',
        peers=2,rounds=2,local_steps=2,rank=16,context=12,sequences=16,
        validation_sequences=32,batch_size=2,kind='lora',optimizer='lora-muon',lr=.002)
    assert out['configuration']['optimizer']=='lora-muon'
    assert out['configuration']['rank']==16
    assert out['configuration']['adapter_counts']['dense_fallback']==0
    assert math.isfinite(out['loss']['central']) and math.isfinite(out['loss']['p2p'])


def test_muon_benchmark_rejects_loha():
    from patchlib.effectiveness_benchmark import run_effectiveness_benchmark
    with pytest.raises(ValueError,match='LoRA-Muon'):
        run_effectiveness_benchmark(peers=2,rounds=1,optimizer='lora-muon',kind='loha')


def test_rank16_balanced_loha_has_live_initial_gradient_and_targeted_layers():
    from test_tinystories_1m_effectiveness import StubGPTNeo
    model=StubGPTNeo()
    adapter=UniversalAdapter(model,'tiny-stories','pinned',kind='loha',rank=16,
                             target_scope='gpt_neo_linear',init_style='balanced')
    info=adapter.counts()
    assert info['groups']==2 and info['dense_fallback']==0
    # For rank16 on narrow matrices LoHa may have MORE parameters than the
    # selected dense layer. Record honestly; scope still excludes embeddings.
    assert info['patch'] > 0
    assert all('transformer.h.' in group.canonical for group in adapter.groups)
    assert all(g.overlay.local.delta().abs().max().item() == 0 for g in adapter.groups)
    x=torch.randint(0,32,(2,12));label=torch.randint(0,32,(2,12))
    logits=model(x).logits
    torch.nn.functional.cross_entropy(logits.flatten(0,1),label.flatten()).backward()
    # LoHa's second CP branch is nonzero in one factor, so its zero factor
    # MUST get a gradient. Former both-branch tiny/zero initialization stalled.
    nonzero=[]
    for group in adapter.groups:
        parts=group.overlay.local.parts
        assert len(parts)==4
        assert parts[3].detach().abs().max()==0
        nonzero.append(parts[3].grad is not None and parts[3].grad.abs().sum().item()>0)
    assert all(nonzero)


def test_live_command_supports_explicit_lora_muon_without_legacy_accident(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.room_manifest import make_hf_manifest
    from patchlib.roomid import create_room_id
    room=create_room_id()
    # No actual network or checkpoints needed with auto_state=True.
    sources=[{'repo':'roneneldan/TinyStories','split':'train','revision':'main',
              'columns':{'text':'text'},'weight':1.0,'config':None}]
    manifest=make_hf_manifest('roneneldan/TinyStories-1M','main',sources,
                              architecture='causal-lm')
    mpath=tmp_path/'manifest.json'
    DesktopController.save_manifest(mpath,manifest)
    cmd=DesktopController.build_command(room=room,model='roneneldan/TinyStories-1M',
        revision='main',base=tmp_path/'genesis.fstate',manifest=mpath,
        ws='ws://127.0.0.1:8080/ws',task='causal-lm',kind='lora',rank=16,
        journal=tmp_path/'p.sqlite3',auto_state=True,optimizer='lora-muon')
    assert cmd[cmd.index('--optimizer')+1]=='lora-muon'
    assert cmd[cmd.index('--target-scope')+1]=='gpt_neo_linear'
    with pytest.raises(ValueError,match='LoRA-Muon'):
        DesktopController.build_command(room=room,model='roneneldan/TinyStories-1M',
            revision='main',base=tmp_path/'genesis.fstate',manifest=mpath,
            ws='ws://127.0.0.1:8080/ws',task='causal-lm',kind='loha',rank=16,
            journal=tmp_path/'p.sqlite3',auto_state=True,optimizer='lora-muon')
