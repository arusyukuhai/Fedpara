"""Regression checks for the previous rank32, std=.02, all-tensor failure mode."""
import torch
import pytest
from patchlib.core import UniversalAdapter
from patchlib.effectiveness_benchmark import scheduled_lr


def _first_gradient(init):
    torch.manual_seed(7)
    model=torch.nn.Linear(64,64,bias=False)
    original=model.weight.detach().clone()
    adapter=UniversalAdapter(model,'example','rev',rank=16,kind='loha',init_style=init)
    torch.testing.assert_close(model.weight.detach(),original,atol=0,rtol=0)
    g=adapter.groups[0]
    assert g.overlay.local.delta().detach().abs().max().item()==0
    x=torch.randn(32,64);y=torch.randn(32,64)
    ((model(x)-y)**2).mean().backward()
    return float(g.overlay.local.parts[-1].grad.norm()),adapter.counts()


def test_balanced_rank16_is_exact_zero_but_not_dead_gradient():
    legacy,_=_first_gradient('legacy')
    balanced,counts=_first_gradient('balanced')
    assert 0 < legacy < balanced/100
    assert counts['dense_fallback']==0
    assert counts['patch']==counts['original']  # rank16, square width64: NOT compression


def test_rank16_attention_mlp_scope_never_trains_embeddings():
    from test_tinystories_1m_effectiveness import StubGPTNeo
    m=StubGPTNeo()
    ada=UniversalAdapter(m,'tiny','rev',rank=16,kind='loha',
             init_style='balanced',target_scope='gpt_neo_linear')
    assert ada.counts()['dense_fallback']==0
    assert all('transformer.h.' in g.canonical for g in ada.groups)
    assert all('wte' not in g.canonical and 'lm_head' not in g.canonical for g in ada.groups)
    assert not any(p.requires_grad for p in m.transformer.wte.parameters())


def test_lr_warmup_and_cosine_do_not_abruptly_start_at_peak():
    values=[scheduled_lr(1e-4,i,100,.1,.1) for i in range(1,101)]
    assert values[0]==pytest.approx(1e-5)
    assert values[9]==pytest.approx(1e-4)
    assert values[-1]==pytest.approx(1e-5)
    assert all(values[i]>values[i-1] for i in range(1,10))
    assert all(values[i]<=values[i-1] for i in range(10,100))


def test_gui_live_loha_defaults_rank16_balanced_no_dense_scope(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.roomid import create_room_id
    from patchlib.room_manifest import make_hf_manifest
    room=create_room_id()
    sources=[{'repo':'roneneldan/TinyStories','split':'train','revision':'main',
              'columns':{'text':'text'},'weight':1.0,'config':None}]
    mf=make_hf_manifest('roneneldan/TinyStories-1M','main',sources,architecture='causal-lm')
    mp=tmp_path/'manifest.json';DesktopController.save_manifest(mp,mf)
    cmd=DesktopController.build_command(room=room,model='roneneldan/TinyStories-1M',
          revision='main',base=tmp_path/'b.fstate',manifest=mp,
          ws='ws://127.0.0.1:8080/ws',task='causal-lm',kind='loha',rank=16,
          journal=tmp_path/'j.sqlite3',auto_state=True)
    assert cmd[cmd.index('--rank')+1]=='16'
    assert cmd[cmd.index('--target-scope')+1]=='auto'
    assert cmd[cmd.index('--init-style')+1]=='balanced'
    assert cmd[cmd.index('--lr')+1]=='0.0001'


def test_gptneo_full_scope_ties_token_embedding_and_head_once():
    from test_tinystories_1m_effectiveness import StubGPTNeo
    from patchlib.continual import enable_anchors
    m=StubGPTNeo()
    initial=m.transformer.wte.weight.detach().clone()
    tied=m.transformer.wte.weight is m.lm_head.weight
    assert tied
    torch.manual_seed(2026)
    ada=UniversalAdapter(m,'random-neo','config',rank=16,kind='loha',
                         init_style='balanced',target_scope='gpt_neo_full')
    names=[g.canonical for g in ada.groups]
    assert 'transformer.wte.weight' in names
    assert 'transformer.wpe.weight' in names
    assert 'lm_head.weight' not in names   # shared Parameter, one unique adapter
    emb=next(g for g in ada.groups if g.canonical=='transformer.wte.weight')
    assert set(emb.aliases)=={'transformer.wte.weight','lm_head.weight'}
    assert m.transformer.wte.parametrizations.weight.original is m.lm_head.parametrizations.weight.original
    assert m.transformer.wte.parametrizations.weight[0] is m.lm_head.parametrizations.weight[0]
    assert not m.transformer.wte.parametrizations.weight.original.requires_grad
    assert all(not p.requires_grad for p in m.parameters() if id(p) not in
               {id(x) for x in ada.trainable_parameters()})
    torch.testing.assert_close(m.transformer.wte.weight.detach(),initial,rtol=0,atol=0)
    enable_anchors(ada)  # anchor marks last merged factor state BEFORE learning
    m.eval()  # deterministic comparison; original stub includes dropout
    optimizer=torch.optim.AdamW(list(ada.trainable_parameters()),lr=1e-2)
    tokens=torch.randint(0,32,(4,9))
    target=torch.randint(0,32,(4,9))
    logits=m(tokens).logits
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,32),target.reshape(-1))
    loss.backward()
    assert any(part.grad is not None and torch.count_nonzero(part.grad)>0
               for part in emb.overlay.local.parts)
    optimizer.step()
    assert not torch.equal(m.transformer.wte.weight.detach(),initial)
    # Full-mode ReLoRA-like rebase: merge current delta into frozen base,
    # move the anchor, preserving learned factors and optimizer state.
    with torch.no_grad():
        before=m(tokens).logits.clone()
        for g in ada.groups:
            module,_,attr=g.canonical.rpartition('.')
            original=m.get_submodule(module).parametrizations[attr].original
            delta=g.overlay.local.delta().detach().clone()
            original.add_(delta)
            for j,part in enumerate(g.overlay.local.parts):
                getattr(g.overlay.anchor,f'part{j}').copy_(part.detach())
        after=m(tokens).logits.clone()
    torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)


def test_full_scope_does_not_duplicate_untied_head():
    from test_tinystories_1m_effectiveness import StubGPTNeo
    m=StubGPTNeo()
    m.lm_head.weight=torch.nn.Parameter(m.lm_head.weight.detach().clone())
    ada=UniversalAdapter(m,'neo','rev',rank=16,kind='loha',target_scope='gpt_neo_full')
    names=[g.canonical for g in ada.groups]
    assert 'transformer.wte.weight' in names and 'lm_head.weight' in names
    assert sum('lm_head.weight' in g.aliases for g in ada.groups)==1
