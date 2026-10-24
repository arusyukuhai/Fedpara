from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.bootstrap import ensure_state
from patchlib.core import UniversalAdapter
from patchlib.state import export_state, semantic_state_sha256
from patchlib.cli import input_device, check_adapter_placement, resolve_device, run_step


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 5)
        self.out = nn.Linear(5, 2)
        self.register_buffer('seen', torch.ones(2, dtype=torch.int64))
    def forward(self, input_ids, labels):
        logits = self.out(self.emb(input_ids).mean(dim=1))
        return {'loss': nn.functional.mse_loss(logits,labels)}
    def get_input_embeddings(self): return self.emb


def test_adapter_and_batch_device_cpu():
    model = Tiny().to('cpu')
    adapter = UniversalAdapter(model,'tiny','fixed',kind='loha',rank=2)
    check_adapter_placement(adapter)
    for group in adapter.groups:
        assert all(p.device==torch.device('cpu') for p in group.overlay.local.parts)
    assert input_device(model,'auto')==torch.device('cpu')
    opt=torch.optim.AdamW(list(adapter.trainable_parameters()))
    args=SimpleNamespace(device='auto',grad_accum=1)
    loss=run_step(model,adapter,opt,{'input_ids':torch.randint(0,9,(2,3)),
                          'labels':torch.ones(2,2)},args)
    assert loss>=0
    assert any(p.grad is not None for p in adapter.trainable_parameters())
    assert resolve_device('cpu')==torch.device('cpu')


@pytest.mark.parametrize('dev', ['cuda','mps'])
def test_adapter_device_on_accelerator_when_available(dev):
    if not ((dev=='cuda' and torch.cuda.is_available()) or
            (dev=='mps' and torch.backends.mps.is_available())):
        pytest.skip('accelerator unavailable')
    model=Tiny().to(dev)
    adapter=UniversalAdapter(model,'tiny','fixed',kind='lora',rank=2)
    check_adapter_placement(adapter)
    opt=torch.optim.AdamW(list(adapter.trainable_parameters()))
    args=SimpleNamespace(device='auto',grad_accum=1)
    loss=run_step(model,adapter,opt,{'input_ids':torch.randint(0,9,(2,3)),
                  'labels':torch.ones(2,2)},args)
    assert loss>=0


def test_semantic_genesis_independent_of_zip_timestamp_and_peer(tmp_path):
    mdl=Tiny()
    a,b=tmp_path/'one.fstate',tmp_path/'two.fstate'
    export_state(mdl,a,model_id='tiny',revision='main',peer_id='alice')
    export_state(mdl,b,model_id='tiny',revision='main',peer_id='bob')
    assert semantic_state_sha256(a,verify=True)==semantic_state_sha256(b,verify=True)
    with torch.no_grad():mdl.out.bias.add_(1)
    c=tmp_path/'three.fstate'
    export_state(mdl,c,model_id='tiny',revision='main',peer_id='bob')
    assert semantic_state_sha256(a)!=semantic_state_sha256(c)


def test_bootstrap_creates_once_and_reuses_checkpoint(tmp_path):
    base=tmp_path/'genesis.fstate'
    args=SimpleNamespace(auto_state=True,factory_file=None,trust_remote_code=False,
       allow_legacy_bin=False,room_manifest=None,room_id='room_1234567890123456',
       state_base=str(base),model='tiny',model_id=None,revision='main',
       max_state_gb=1)
    calls=[]
    def load_model(local,use_adapter=False):
        assert not use_adapter and local.device=='cpu' and local.state_base is None
        calls.append(1)
        return Tiny(),'main'
    ensure_state(args,load_model=load_model)
    assert base.exists()
    assert len(calls)==1
    ensure_state(args,load_model=load_model)
    assert len(calls)==1
    assert not base.with_name('genesis.fstate.creating').exists()


def test_bootstrap_never_runs_user_code(tmp_path):
    args=SimpleNamespace(auto_state=True,trust_remote_code=False,
        factory_file='evil.py',allow_legacy_bin=False,room_manifest=None,
        state_base=str(tmp_path/'base.fstate'))
    with pytest.raises(PermissionError):
        ensure_state(args,load_model=lambda *_a,**_kw: None)
