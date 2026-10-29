"""Regression fixtures for architectures that load but previously could not train."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from patchlib.core import UniversalAdapter
from patchlib.training_compat_v84 import (enable_adapter_safe_checkpointing,
    causal_labels_from_attention)


class LegacyCheckpointLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(17,8)
        self.ff=nn.Linear(8,8)
        self.head=nn.Linear(8,17)
        self.reentrant=True
    def get_input_embeddings(self):return self.emb
    def gradient_checkpointing_enable(self):
        self.reentrant=True
    def forward(self,input_ids,labels):
        x=self.emb(input_ids)
        x=checkpoint(self.ff,x,use_reentrant=self.reentrant)
        logits=self.head(x)
        return SimpleNamespace(loss=F.cross_entropy(logits.reshape(-1,17),labels.flatten()),
                               logits=logits)


class ModernCheckpointLM(LegacyCheckpointLM):
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        assert gradient_checkpointing_kwargs == {'use_reentrant':False}
        self.reentrant=False


def sample():
    ids=torch.tensor([[1,2,3,4]],dtype=torch.long)
    return {'input_ids':ids, 'labels':ids.clone()}


@pytest.mark.parametrize('model_class,mode',[
    (LegacyCheckpointLM,'embedding_output_hook'),
    (ModernCheckpointLM,'non_reentrant'),
])
def test_frozen_embeddings_checkpointed_loha_still_trains(model_class,mode):
    model=model_class()
    assert enable_adapter_safe_checkpointing(model)==mode
    adapter=UniversalAdapter(model,'arch','version',kind='loha',rank=2,target_scope='auto')
    params=list(adapter.trainable_parameters())
    base=[p for p in model.parameters() if not p.requires_grad]
    assert base, 'frozen base weight required'
    out=model(**sample())
    out.loss.backward()
    assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in params)
    assert all(p.grad is None for p in base)


def test_checkpointing_unsupported_architecture_explained():
    with pytest.raises(RuntimeError,match='does not implement'):
        enable_adapter_safe_checkpointing(nn.Linear(3,4))


def test_eos_survives_pad_ids_without_token_type():
    ids=torch.tensor([[1,2,2,2],[3,2,4,2]])
    attention=torch.tensor([[1,1,0,0],[1,1,1,0]])
    assert causal_labels_from_attention(ids,attention).tolist()==[[1,2,-100,-100],[3,2,4,-100]]


def test_room_hf_causal_batches_keep_eos_and_drop_token_type_ids(monkeypatch):
    class FakeTokenizer:
        pad_token='<eos>'
        eos_token='<eos>'
        pad_token_id=2
        def __call__(self, texts,*, return_token_type_ids=True,**kwargs):
            assert not return_token_type_ids, 'decoder LM must not get synthetic token_type_ids'
            return {'input_ids':torch.tensor([[1,2,2,2],[3,2,4,2]]),
                    'attention_mask':torch.tensor([[1,1,0,0],[1,1,1,0]])}
    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls,*a,**kw):return FakeTokenizer()
    class ShouldNotConstructCollator:
        def __init__(self,*a,**kw):
            raise AssertionError('CLM must not mask target tokens based on ID')
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,DataCollatorForLanguageModeling=ShouldNotConstructCollator))
    from patchlib.hf_datasets import batches_from_room
    from patchlib.room_manifest import make_hf_manifest
    manifest=make_hf_manifest('org/model','main',[
        {'repo':'org/data','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1}])
    args=SimpleNamespace(hf_task='causal-lm',batch_size=2,max_length=16,peer_id='p')
    rows=iter([{'text':'abc'}, {'text':'def'}])
    batch=next(batches_from_room(args,manifest,rows=rows))
    assert 'token_type_ids' not in batch
    assert batch['labels'].tolist()==[[1,2,-100,-100],[3,2,4,-100]]


def test_no_targets_fails_with_rank_and_room_advice():
    model=nn.Sequential(nn.Linear(3,2),nn.ReLU())
    with pytest.raises(ValueError,match=r'--rank.*NEW P2P room'):
        UniversalAdapter(model,'small','rev',kind='loha',rank=16,target_scope='auto')


def test_preflight_command_probes_real_batch_without_updates(monkeypatch,capsys):
    from patchlib import cli
    model=ModernCheckpointLM()
    adapter=UniversalAdapter(model,'arch','version',kind='loha',rank=2,target_scope='auto')
    original=[p.detach().clone() for p in adapter.trainable_parameters()]
    monkeypatch.setattr(cli,'load',lambda args:(model,adapter))
    monkeypatch.setattr(cli,'training_batches',lambda args:iter([sample()]))
    monkeypatch.setattr(sys,'argv',['cli','training-preflight','--model','example/model',
                                   '--rank','2','--target-scope','auto'])
    cli.main()
    import json
    report=json.loads(capsys.readouterr().out)
    assert report['gradients_present']>0
    assert report['forward_backward']=='pass'
    assert all(torch.equal(before,after) for before,after in zip(original,adapter.trainable_parameters()))


def test_installed_builtin_transformers_model_mapping_recovers_newer_family(monkeypatch):
    # No network/remote code: metadata only; the implementation class is not imported.
    import patchlib.hf_config_v83 as hf
    modeling_auto=types.SimpleNamespace(
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES={'qwen3':'Qwen3ForCausalLM'},
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES={'vit':'ViTForImageClassification'})
    auto=types.SimpleNamespace(modeling_auto=modeling_auto)
    models=types.SimpleNamespace(auto=auto)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(models=models))
    monkeypatch.setitem(sys.modules,'transformers.models',models)
    monkeypatch.setitem(sys.modules,'transformers.models.auto',auto)
    assert hf.checked_model_type({'architectures':['Qwen3ForCausalLM']})=='qwen3'
    assert hf.checked_model_type({'architectures':['ViTForImageClassification']})=='vit'
    # Metadata conflicts are not silently resolved to one type.
    modeling_auto.MODEL_MAPPING_NAMES={'different':'Qwen3ForCausalLM'}
    with pytest.raises(hf.MissingModelTypeError):
        hf.checked_model_type({'architectures':['Qwen3ForCausalLM']})


def test_preflight_rejects_zero_gradient_adapter_even_if_backward_succeeds():
    from patchlib.model_contract_v64 import probe_model,ModelContractError
    class NoUsefulGradient(nn.Module):
        def __init__(self):
            super().__init__(); self.lin=nn.Linear(8,8)
        def forward(self,input_ids,labels):
            z=self.lin(torch.ones(1,8))
            return {'loss':(z*0).sum()}
    model=NoUsefulGradient()
    adapter=UniversalAdapter(model,'zero','rev',kind='loha',rank=2,target_scope='auto')
    with pytest.raises(ModelContractError,match='nonzero finite'):
        probe_model(model,adapter,{'input_ids':torch.ones(1,2,dtype=torch.long),
                                   'labels':torch.ones(1,2,dtype=torch.long)})
