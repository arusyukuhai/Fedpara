"""Real GUI→live CLI wiring, safe random config, pinned genesis and adapters.

Small in-process stand-in for GPT-Neo avoids a network dependency in CI.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.live_initialization import (
    TINY_STORIES_ID, TINY_STORIES_REVISION, TINYSTORIES_DATA_REVISION,
    create_random_tinystories_model, room_initialization,
)
from patchlib.gui import DesktopController
from patchlib.webview_controller import WebviewController
from patchlib.room_manifest import make_hf_manifest


class FakeGPTNeo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.transformer=nn.Module()
        self.transformer.wte=nn.Embedding(32,16)
        self.transformer.wpe=nn.Embedding(64,16)
        self.transformer.h=nn.ModuleList([nn.Module()])
        layer=self.transformer.h[0]
        layer.attn=nn.Module();layer.attn.attention=nn.Module()
        layer.attn.attention.q_proj=nn.Linear(16,16,bias=False)
        layer.mlp=nn.Module();layer.mlp.c_fc=nn.Linear(16,16)
        self.lm_head=nn.Linear(16,32,bias=False)
        self.lm_head.weight=self.transformer.wte.weight

    def forward(self,input_ids):
        seq=torch.arange(input_ids.shape[1],device=input_ids.device)
        x=self.transformer.wte(input_ids)+self.transformer.wpe(seq)
        x=self.transformer.h[0].attn.attention.q_proj(x)
        return types.SimpleNamespace(logits=self.lm_head(x))


def fake_transformers(monkeypatch):
    calls=[]
    def config_loader(name,**kw):
        assert name==TINY_STORIES_ID
        assert kw==dict(revision=TINY_STORIES_REVISION,trust_remote_code=False)
        calls.append('config')
        return types.SimpleNamespace(model_type='gpt_neo',use_cache=True)
    def model_loader(config,**kw):
        assert kw==dict(trust_remote_code=False)
        calls.append('random')
        return FakeGPTNeo(config)
    def no_pretrained(*a,**kw):
        raise AssertionError('random genesis must never fetch pretrained weights')
    fake=types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=config_loader),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_config=model_loader,from_pretrained=no_pretrained))
    monkeypatch.setitem(sys.modules,'transformers',fake)
    return calls


def test_random_config_is_deterministic_and_never_loads_weights(monkeypatch):
    calls=fake_transformers(monkeypatch)
    before=torch.get_rng_state().clone()
    m1,rev=create_random_tinystories_model(TINY_STORIES_ID,TINY_STORIES_REVISION,42)
    m2,_=create_random_tinystories_model(TINY_STORIES_ID,TINY_STORIES_REVISION,42)
    m3,_=create_random_tinystories_model(TINY_STORIES_ID,TINY_STORIES_REVISION,43)
    assert rev==TINY_STORIES_REVISION
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(m1.state_dict()[k],m2.state_dict()[k]) for k in m1.state_dict())
    assert any(not torch.equal(m1.state_dict()[k],m3.state_dict()[k]) for k in m1.state_dict())
    assert all(m.config.use_cache is False for m in (m1,m2,m3))
    assert calls==['config','random']*3
    with pytest.raises(ValueError,match='pinned'):
        create_random_tinystories_model('sshleifer/tiny-gpt2','main',42)


def test_main_load_model_full_scope_tied_embeddings_frozen(monkeypatch):
    fake_transformers(monkeypatch)
    from patchlib.cli import load_model
    args=types.SimpleNamespace(model=TINY_STORIES_ID,revision=TINY_STORIES_REVISION,
        model_id=None,initialization='random',initial_seed=42,
        factory_file=None,trust_remote_code=False,allow_legacy_bin=False,
        hf_task='causal-lm',auto_class=None,device='cpu',state_base=None,
        moe_plan=None,max_state_gb=32,kind='loha',rank=16,alpha=None,max_peers=8,
        target_scope='auto',init_style='balanced',gradient_checkpointing=False)
    model,adapter=load_model(args,use_adapter=True)
    assert args.target_scope=='auto'  # universal model-agnostic scope
    assert set(g.canonical for g in adapter.groups)>={'transformer.wte.weight', 'transformer.wpe.weight'}
    assert any('lm_head.weight' in g.aliases for g in adapter.groups)
    assert sum('lm_head.weight' in g.aliases for g in adapter.groups)==1
    assert all(not g.overlay.local.dense for g in adapter.groups)
    assert all(not p.requires_grad for n,p in model.named_parameters() if '.local.' not in n)
    inp=torch.tensor([[1,3,4]])
    init_logits=model(inp).logits.detach().clone()
    with torch.no_grad():
        group=next(g for g in adapter.groups if g.canonical=='transformer.wte.weight')
        group.overlay.local.parts[-1].add_(.04)  # nonzero first LoHa branch × newly activated second branch
    assert not torch.equal(init_logits,model(inp).logits)


def test_gui_new_room_pins_random_genesis_and_training_options(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    settings=c.settings()
    assert settings['model']==TINY_STORIES_ID
    assert settings['revision']==TINY_STORIES_REVISION
    assert settings['sources'][0]['revision']==TINYSTORIES_DATA_REVISION
    assert settings['rank']==16 and settings['reinit_every']==20
    data=c.save(settings)['manifest']
    assert room_initialization(data)=='random'
    got={}
    def start(args,log):
        got['args']=args
        return 54321
    monkeypatch.setattr(c.train,'start',start)
    response=c.start({'room':settings['room'],'ws':settings['ws'],
        'kind':'loha','rank':16,'reinit_every':20,
        'lr':.0001,'local_steps':48,'max_length':64})
    cmd=got['args']
    assert '--auto-state' in cmd and response['auto_state']
    assert '--initialization' in cmd and cmd[cmd.index('--initialization')+1]=='random'
    assert '--initial-seed' in cmd and cmd[cmd.index('--initial-seed')+1]=='42'
    assert cmd[cmd.index('--rank')+1]=='16'
    assert cmd[cmd.index('--reinit-every')+1]=='20'
    assert cmd[cmd.index('--local-steps')+1]=='48'
    assert cmd[cmd.index('--max-length')+1]=='64'
    assert cmd[cmd.index('--target-scope')+1]=='auto'
    assert response['base'].endswith('genesis.random.fstate')
    assert not Path(response['base']).exists()
    assert response['initialization']=='random'
    c.close()


def test_gui_cannot_switch_saved_room_from_random_to_pretrained(tmp_path):
    c=WebviewController(tmp_path)
    settings=c.settings();c.save(settings)
    with pytest.raises(ValueError,match='固定'):
        c.save({**settings,'initialization':'pretrained'})
    c.close()


def test_gui_loaded_pretrained_manifest_stays_pretrained(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    settings=c.settings()
    m=make_hf_manifest('org/model','main',settings['sources'],seed=21)
    path=c._room_path(settings['room']);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(__import__('json').dumps(m))
    got={}
    monkeypatch.setattr(c.train,'start',lambda cmd,log: got.setdefault('cmd',cmd) and 54321)
    out=c.start({'room':settings['room'],'ws':settings['ws'],'rank':16})
    assert out['initialization']=='pretrained'
    assert out['base'].endswith('genesis.fstate')
    assert got['cmd'][got['cmd'].index('--initialization')+1]=='pretrained'
    c.close()


def test_random_genesis_bootstrap_reuses_verified_same_state(monkeypatch,tmp_path):
    fake_transformers(monkeypatch)
    from patchlib.cli import load_model
    from patchlib.bootstrap import ensure_state
    from patchlib.state import semantic_state_sha256
    from patchlib.room_manifest import make_hf_manifest
    from patchlib.live_initialization import RANDOM_MANIFEST_DESCRIPTION
    sources=[{'repo':'roneneldan/TinyStories','revision':TINYSTORIES_DATA_REVISION,
              'config':None,'split':'train','columns':{'text':'text'},'weight':1.0}]
    manifest=make_hf_manifest(TINY_STORIES_ID,TINY_STORIES_REVISION,sources,
                              description=RANDOM_MANIFEST_DESCRIPTION,seed=42)
    manifest_path=tmp_path/'room.json';manifest_path.write_text(__import__('json').dumps(manifest))
    state_path=tmp_path/'genesis.random.fstate'
    args=types.SimpleNamespace(model=TINY_STORIES_ID,revision=TINY_STORIES_REVISION,
        model_id=None,initialization='random',initial_seed=42,
        factory_file=None,trust_remote_code=False,allow_legacy_bin=False,
        hf_task='causal-lm',auto_class=None,device='cpu',state_base=str(state_path),
        moe_plan=None,max_state_gb=1,kind='loha',rank=16,alpha=None,max_peers=8,
        target_scope='auto',init_style='balanced',gradient_checkpointing=False,
        auto_state=True,room_manifest=manifest_path)
    ensure_state(args,load_model=load_model)
    assert state_path.is_file()
    digest=semantic_state_sha256(state_path)
    ensure_state(args,load_model=load_model)
    assert digest==semantic_state_sha256(state_path)
    initial,adapter=load_model(args,use_adapter=True)
    assert adapter.revision.startswith('fstate-sha256:')
    assert digest in adapter.revision
    assert len(adapter.groups)>0
