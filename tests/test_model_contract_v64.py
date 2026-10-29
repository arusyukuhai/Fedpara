"""Unknown LM architectures, actual autograd, and fail-closed execution contracts."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from patchlib.core import UniversalAdapter, PatchError
from patchlib.model_contract_v64 import (ModelContractError, invoke_model,
    parse_input_map, probe_model, training_loss)


class MultiAxisBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(2, d, d)*.04)
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self,x):
        # Not an nn.Linear; an opaque custom differentiable LM operation.
        return torch.tanh(torch.einsum('bti,hij->bthj',x,self.kernel).mean(-2)+self.bias)


class UnknownLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(17, 8)
        self.block = MultiAxisBlock(8)
        self.head = nn.Linear(8,17,bias=False)
        self.head.weight = self.token.weight  # genuine weight tying
        self.extra_alias = self.block  # also share an entire Module
    def forward(self,tokens):
        x = self.block(self.token(tokens))
        return F.linear(x, self.head.weight)  # logits only; no loss field


def batch():
    ids=torch.tensor([[1,2,3,4],[2,3,4,5]],dtype=torch.long)
    return {'input_ids':ids,'labels':ids.clone()}


def make_adapter():
    torch.manual_seed(7)
    model=UnknownLM()
    adapter=UniversalAdapter(model,'custom-unseen-lm','fixed-revision',kind='lora',
                             rank=2,target_scope='universal')
    return model,adapter


def test_unseen_3d_1d_tied_module_forward_backward_and_export():
    model,adapter=make_adapter()
    assert len(adapter.groups)==3  # tied embedding+head, custom 3D and 1-D
    assert any(len(g.shape)==3 for g in adapter.groups)
    assert any(len(g.shape)==1 for g in adapter.groups)
    assert any(len(g.aliases)>1 for g in adapter.groups)
    before=[p.detach().clone() for p in adapter.trainable_parameters()]
    result=probe_model(model,adapter,batch(),task='causal-lm',
                       input_map='{"input_ids":"tokens"}')
    assert result['gradients_present']>0 and result['nonzero_gradients']>0
    opt=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.01)
    from patchlib.cli import run_step
    args=SimpleNamespace(device='cpu',grad_accum=1,hf_task='causal-lm',
                         input_map='{"input_ids":"tokens"}',loss_policy='auto',
                         telemetry_file=None)
    opt.zero_grad()
    loss=run_step(model,adapter,opt,batch(),args)
    assert loss>0
    opt.step()
    assert any(not torch.equal(old,new) for old,new in zip(before,adapter.trainable_parameters()))
    packet=adapter.export('peer1',seq=1)
    assert packet.startswith(b'FPAT1')
    other,recv=make_adapter()
    assert adapter.layout==recv.layout
    recv.load_local(packet)
    # Factors are transmitted under the same deterministic signed layout.
    assert torch.allclose(invoke_model(model,batch(),input_map={'input_ids':'tokens'}),
                          invoke_model(other,batch(),input_map={'input_ids':'tokens'}),atol=1e-4)


def test_logits_only_scalar_provided_and_aligned_losses():
    logits=torch.randn(2,4,17,requires_grad=True)
    sample=batch()
    loss=training_loss(logits,sample,task='causal-lm')
    expected=F.cross_entropy(logits[:,:-1].reshape(-1,17),sample['labels'][:,1:].reshape(-1))
    assert torch.allclose(loss,expected)
    masked=training_loss({'logits':logits},sample,task='masked-lm')
    assert masked.ndim==0 and masked.requires_grad
    provided=torch.tensor(2.,requires_grad=True)
    assert training_loss({'loss':provided},sample) is provided
    assert training_loss((provided,logits),sample) is provided
    # A user-selected explicit task loss overrides a model's own definition.
    assert torch.allclose(training_loss({'loss':provided,'logits':logits},sample,policy='causal'),expected)
    clf_logits=torch.randn(2,3,requires_grad=True)
    labels={'labels':torch.tensor([0,2])}
    assert training_loss(clf_logits,labels,task='sequence-classification').requires_grad


def test_contract_errors_are_actionable():
    model,adapter=make_adapter()
    with pytest.raises(ModelContractError,match='does not accept'):
        invoke_model(model,batch())
    assert invoke_model(model,{'input_ids':batch()['input_ids'],'attention_mask':torch.ones_like(batch()['input_ids'])},
                        input_map={'input_ids':'tokens','attention_mask':None}).shape==(2,4,17)
    with pytest.raises(ModelContractError,match='collides'):
        invoke_model(model,{'tokens':batch()['input_ids'],'input_ids':batch()['input_ids']},
                     input_map={'input_ids':'tokens'})
    with pytest.raises(ModelContractError,match='unknown|defined'):
        training_loss(torch.randn(2,4,17,requires_grad=True),batch(),task='base')
    with pytest.raises(ModelContractError,match='ignored'):
        training_loss(torch.randn(2,4,17,requires_grad=True),
                      {'labels':torch.full((2,4),-100)},task='causal-lm')
    with pytest.raises(ModelContractError,match='not differentiable'):
        training_loss(torch.randn(2,4,17),batch())
    with pytest.raises(ModelContractError,match='identifier'):
        parse_input_map('{"input_ids":"__import__"}')
    with pytest.raises(ModelContractError,match='JSON'):
        parse_input_map('{bad')


def test_large_dense_fallback_is_bounded():
    class LargeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias=nn.Parameter(torch.zeros(1_048_577))
    with pytest.raises(PatchError,match='dense adapter fallback'):
        UniversalAdapter(LargeLM(),'large','v1',kind='lora',rank=4,target_scope='universal')


def test_built_in_policy_unchanged():
    model=UnknownLM()
    auto=UniversalAdapter(model,'old','v1',kind='lora',rank=2,target_scope='auto')
    assert all(len(g.shape)==2 for g in auto.groups)
    assert len(auto.groups) == 1  # only tied embeddings+head


def test_architecture_name_never_blocks_bf16_builtin_forward():
    from patchlib.hf_compat_v63 import validate_training_config
    cfg=SimpleNamespace(model_type='deepseek_v3',quantization_config=None)
    assert validate_training_config(cfg,'somevendor/myLM',streaming=True)=='deepseek_v3'
    cfg.quantization_config={'quant_method':'fp8'}
    with pytest.raises(ValueError,match='quantized'):
        validate_training_config(cfg,'somevendor/myLM',streaming=True)


def test_no_unconditional_gradient_claim():
    class DetachedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.w=nn.Parameter(torch.randn(4,4))
        def forward(self,input_ids):
            return torch.zeros(input_ids.size(0),input_ids.size(1),4,requires_grad=True)
    model=DetachedLM()
    adapter=UniversalAdapter(model,'detached','v1',kind='lora',rank=2,target_scope='universal')
    with pytest.raises(ModelContractError,match='did not reach'):
        probe_model(model,adapter,{'input_ids':torch.zeros(2,3,dtype=torch.long),
                                   'labels':torch.zeros(2,3,dtype=torch.long)})


def test_gui_manifest_pins_universal_scope_across_peers(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.room_manifest import validate_manifest, manifest_digest
    from patchlib.roomid import create_room_id
    manifest=DesktopController.create_manifest('org/unknown','main','causal-lm',
        [{'repo':'org/data','revision':'main','config':None,'split':'train',
          'columns':{'text':'text'},'weight':1}])
    manifest['adapter']={'kind':'loha','rank':16,'wide_lora_rank':4,
                         'target_scope':'universal'}
    validate_manifest(manifest)
    import pathlib
    path=tmp_path/'room.json';DesktopController.save_manifest(path,manifest)
    command=DesktopController.build_command(room=create_room_id(),model='org/unknown',revision='main',
        base=None,manifest=path,ws='ws://127.0.0.1:8080/ws',task='causal-lm',
        kind='loha',rank=16,journal=tmp_path/'j.sqlite3',streaming_base=True)
    assert command[command.index('--target-scope')+1]=='universal'
    different=json.loads(path.read_text())
    del different['adapter']['target_scope']
    assert manifest_digest(different)!=manifest_digest(manifest)
    with pytest.raises(ValueError,match='requires AdamW'):
        DesktopController.build_command(room=create_room_id(),model='org/unknown',revision='main',
            base=None,manifest=path,ws='ws://127.0.0.1:8080/ws',task='causal-lm',
            kind='lora',rank=16,journal=tmp_path/'j.sqlite3',streaming_base=True,
            optimizer='lora-muon')
