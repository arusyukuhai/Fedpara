"""No-network model compatibility tests: Llama/Qwen-like weights and training."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from patchlib.cli import causal_labels_from_attention
from patchlib.streaming_base import materialize_meta_from_safetensors, StreamingModelError
from patchlib.hf_compat_v63 import validate_training_config
from patchlib.core import UniversalAdapter


class TinyRotaryEmbedding(nn.Module):
    """Mirrors the nonpersistent inv_freq contract of Llama-like architectures."""
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.rope_init_fn=lambda cfg, device: (
            (1. / (cfg.rope_theta**(torch.arange(0,8,2,device=device,dtype=torch.float32)/8))), 1.0)
        self.register_buffer('inv_freq',torch.empty(4),persistent=False)

class LlamaLikeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=8,num_attention_heads=1,rope_theta=10000.)
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(16,8)
        self.model.layers=nn.ModuleList([nn.Module()])
        attn=nn.Module()
        attn.q_proj=nn.Linear(8,8,bias=False)
        attn.rotary_emb=TinyRotaryEmbedding(self.config)
        self.model.layers[0].self_attn=attn
        self.lm_head=nn.Linear(8,16,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight
    def get_output_embeddings(self): return self.lm_head
    def forward(self, input_ids, labels=None):
        z=self.model.embed_tokens(input_ids)
        z=self.model.layers[0].self_attn.q_proj(z)
        logits=self.lm_head(z)
        loss=logits.float().square().mean()
        return SimpleNamespace(loss=loss, logits=logits)


def save_llama_like(path):
    model=LlamaLikeCausalLM().to(dtype=torch.bfloat16)
    save_file({
        'model.embed_tokens.weight': model.model.embed_tokens.weight.detach().contiguous(),
        'model.layers.0.self_attn.q_proj.weight':model.model.layers[0].self_attn.q_proj.weight.detach().contiguous(),
    },str(path/'model.safetensors'))
    return model


def test_bf16_checkpoint_rope_and_tied_head_train(tmp_path):
    src=save_llama_like(tmp_path)
    with torch.device('meta'):
        model=LlamaLikeCausalLM()
    assert model.model.embed_tokens.weight.dtype==torch.float32
    materialize_meta_from_safetensors(model,tmp_path,preserve_checkpoint_dtype=True)
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.layers[0].self_attn.rotary_emb.inv_freq.device.type=='cpu'
    assert model.lm_head.weight is model.model.embed_tokens.weight
    adapter=UniversalAdapter(model,'llama','fixed',kind='lora',rank=2,target_scope='auto')
    opt=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=1e-3)
    loss=model(torch.tensor([[1,2,3]]), labels=torch.tensor([[1,2,3]])).loss
    loss.backward()
    assert all(p.grad is not None for p in adapter.trainable_parameters())
    opt.step()
    assert torch.isfinite(loss)
    assert torch.allclose(model.model.layers[0].self_attn.rotary_emb.inv_freq,
                          torch.tensor([1.0,0.1,0.01,0.001]),atol=1e-6)


def test_dtype_strict_mode_still_rejects_mismatch(tmp_path):
    save_llama_like(tmp_path)
    with torch.device('meta'):model=LlamaLikeCausalLM()
    with pytest.raises(StreamingModelError,match='dtype mismatch'):
        materialize_meta_from_safetensors(model,tmp_path)


def test_missing_unknown_nonpersistent_stays_fail_closed(tmp_path):
    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('state',torch.ones(3),persistent=False)
            self.p=nn.Parameter(torch.ones(3))
    save_file({'p':torch.ones(3)},str(tmp_path/'model.safetensors'))
    with torch.device('meta'):model=Unknown()
    with pytest.raises(StreamingModelError,match='nonpersistent'):
        materialize_meta_from_safetensors(model,tmp_path,preserve_checkpoint_dtype=True)


def test_eos_token_labels_not_masked_like_padding():
    tokens=torch.tensor([[6,2,2,2], [6,2,7,2]])
    mask=torch.tensor([[1,1,0,0],[1,1,1,0]])
    labels=causal_labels_from_attention(tokens,mask)
    assert labels.tolist()==[[6,2,-100,-100],[6,2,7,-100]]
    with pytest.raises(ValueError):
        causal_labels_from_attention(tokens,mask[:1])


@pytest.mark.parametrize('architecture', ['llama','qwen2','mistral','gemma2'])
def test_builtin_distill_architectures_allowed(architecture):
    assert validate_training_config(SimpleNamespace(model_type=architecture,quantization_config=None),
                                    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',streaming=True)==architecture

@pytest.mark.parametrize('architecture',['deepseek_v2','deepseek_v3','deepseek_moe'])
def test_native_deepseek_is_not_rejected_by_architecture_name(architecture):
    # The installed loader/runtime, not a brand blacklist, decides support.
    assert validate_training_config(SimpleNamespace(model_type=architecture),
                                    'deepseek-ai/DeepSeek-V3',streaming=True)==architecture

def test_quantized_training_not_claimed_supported():
    with pytest.raises(ValueError,match='quantized'):
        validate_training_config(SimpleNamespace(model_type='llama',quantization_config={'quant_method':'fp8'}),
                                 'org/model',streaming=True)


def test_gui_streaming_base_with_no_shard_store(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.roomid import create_room_id
    ctl=DesktopController()
    m=ctl.create_manifest('org/model','main','causal-lm',[
        {'repo':'org/data','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1}])
    manifest=ctl.save_manifest(tmp_path/'room.json',m)
    cmd=ctl.build_command(room=create_room_id(),model='org/model',revision='main',
        base=None,manifest=manifest,ws='ws://127.0.0.1:8080/ws',task='causal-lm',
        kind='loha',rank=16,journal=tmp_path/'journal.sqlite3',streaming_base=True)
    assert '--streaming-scratch-dir' in cmd
    dest=cmd[cmd.index('--streaming-scratch-dir')+1]
    assert dest.startswith(str(tmp_path)) and dest.endswith('temporary-streaming-pages')


def test_projection_policy_finds_llama_linear_layers():
    model=LlamaLikeCausalLM()
    adapter=UniversalAdapter(model,'llama','sha',kind='lora',rank=2,
                             target_scope='linear')
    assert len(adapter.groups)==1
    assert adapter.groups[0].canonical=='model.layers.0.self_attn.q_proj.weight'
    from patchlib.lora_muon import LoRAMuon
    opt=LoRAMuon(adapter,lr=.001)
    out=model(torch.tensor([[1,3,4]])).loss
    out.backward()
    opt.step()
    assert torch.isfinite(out)


def test_gui_lora_muon_uses_architecture_neutral_projection(tmp_path):
    from patchlib.gui import DesktopController
    from patchlib.roomid import create_room_id
    ctl=DesktopController()
    m=ctl.create_manifest('org/llama','main','causal-lm',[
        {'repo':'org/data','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1}])
    manifest=ctl.save_manifest(tmp_path/'room.json',m)
    base=tmp_path/'base.fstate';base.touch()
    cmd=ctl.build_command(room=create_room_id(),model='org/llama',revision='main',
        base=base,manifest=manifest,ws='ws://127.0.0.1:8080/ws',task='causal-lm',
        kind='lora',rank=16,journal=tmp_path/'j.sqlite3',optimizer='lora-muon')
    assert cmd[cmd.index('--target-scope')+1]=='linear'


def test_offline_hf_diagnostic_supports_distilled_config_and_files(tmp_path):
    from patchlib.hf_compat_v63 import diagnose_hf_checkpoint
    (tmp_path/'config.json').write_text(json.dumps({
        'model_type':'llama','architectures':['LlamaForCausalLM']}))
    (tmp_path/'model.safetensors').write_bytes(b'for-metadata-presence-only')
    report=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert report['streaming_candidate']
    assert report['checkpoint_present']
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'deepseek_v3'}))
    reject=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert reject['streaming_candidate']
    assert 'verify' in ' '.join(reject['notes']).lower()


def test_backend_rejects_random_genesis_for_llama_early(tmp_path):
    from patchlib.webview_controller import WebviewController
    from patchlib.roomid import create_room_id
    ctl=WebviewController(tmp_path)
    with pytest.raises(ValueError,match='Select pretrained'):
        ctl.save({'room':create_room_id(),'model':'meta-llama/Llama-3.1-8B',
                  'revision':'main','task':'causal-lm','initialization':'random',
                  'sources':[{'repo':'org/data','revision':'main','config':None,
                              'split':'train','columns':{'text':'text'},'weight':1}]})
