"""Kagami v37: dtype-safe trained inference, graph responsiveness and detail."""
from __future__ import annotations
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.kagami_eval_worker import _direct_restore_dtype, load_snapshot_model
from patchlib.shards import export_state_sharded
from patchlib.webview_gui import compose_html
from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id


def test_direct_restore_dtype_supports_half_and_mixed():
    one={'tensors':[{'dtype':'torch.float16'},{'dtype':'torch.int64'}]}
    assert _direct_restore_dtype(one)=='torch.float16'
    assert _direct_restore_dtype({'tensors':[{'dtype':'torch.float16'},{'dtype':'torch.float32'}]}) is None


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_verified_trained_checkpoint_restores_original_precision(tmp_path,monkeypatch,dtype):
    class HalfToy(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(8,8)
        @classmethod
        def from_config(cls,_cfg,**kw):return cls()
        def tie_weights(self):pass
    original=HalfToy().to(dtype=dtype)
    with torch.no_grad():
        original.linear.weight.fill_(0.125)
        original.linear.bias.fill_(-0.25)
    saved=tmp_path/'effective.shards.json';store=tmp_path/'store'
    export_state_sharded(original,store,saved,model_id='example/tiny',
                         revision='main',peer_id='alice',shard_bytes=65536)
    fake=types.ModuleType('transformers')
    fake.AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())
    monkeypatch.setitem(sys.modules,'transformers',fake)
    monkeypatch.setattr('patchlib.kagami_eval_worker._model_class',lambda manifest:HalfToy)
    restored,state=load_snapshot_model({'checkpoint':str(saved),'store':str(store),
                                        'checkpoint_revision':'main'},torch,
                                        {'model':{'id':'example/tiny','revision':'main'}})
    assert restored.linear.weight.dtype==dtype
    assert torch.equal(restored.linear.weight,original.linear.weight)
    assert torch.equal(restored.linear.bias,original.linear.bias)


def test_snapshot_completed_status_checks_manifest_identity(tmp_path):
    ctl=WebviewController(tmp_path);room=create_room_id()
    ctl.save({'room':room,'model':'example/tiny','revision':'main','task':'causal-lm',
              'seed':42,'sources':[{'repo':'example/data','revision':'main',
              'config':None,'split':'train','columns':{'text':'text'},'weight':1.0}]})
    root=ctl._room_path(room).parent
    ctl._current_room=room
    ctl._snapshot_request_file=root/'export.request'
    ctl._checkpoint_path=root/'bad.shards.json'
    ctl._checkpoint_path.write_text('{"truncated": true}')
    ctl._snapshot_request_file.with_suffix('.status.json').write_text('{"state":"complete"}')
    result=ctl.eval_snapshot_status({'room':room})
    assert result['state']=='error' and 'manifest' in result['error']


def test_live_graph_refresh_and_legibility_in_embedded_html():
    html=compose_html()
    assert 'lastHistoryFetch<4200' in html
    assert '},900)' in html
    assert "latestTelemetry.slice(-8)" in html
    assert "inspectLine" in html and "chartAxisNumber" in html
    assert 'font-size' in html
    assert 'Inference status error:' in html

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_full_trained_inference_generation_reads_16bit_learned_weights(tmp_path,monkeypatch,dtype):
    """Prove the learned tensor actually affects generated output on CPU."""
    from patchlib.kagami_eval_worker import run
    from patchlib.room_manifest import make_hf_manifest
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(10,4)
            self.config=types.SimpleNamespace(max_position_embeddings=32)
        @classmethod
        def from_config(cls,_cfg,**kwargs):return cls()
        def tie_weights(self):pass
        def get_input_embeddings(self):return self.embedding
        def generate(self,input_ids,**kwargs):
            symbol=7 if float(self.embedding.weight[0,0])>.5 else 8
            return torch.cat([input_ids,torch.tensor([[symbol]],device=input_ids.device)],dim=1)
    class Tokenizer:
        model_max_length=32;pad_token_id=0;eos_token_id=1
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def __call__(self,text,**kwargs):return {'input_ids':torch.tensor([[0,2]])}
        def decode(self,ids,**kwargs):return ' '.join(str(x) for x in ids.tolist())
    transformers=types.ModuleType('transformers')
    transformers.AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**kw:types.SimpleNamespace())
    transformers.AutoModelForCausalLM=TinyLM
    transformers.AutoTokenizer=Tokenizer
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    source=TinyLM().to(dtype=dtype)
    with torch.no_grad():source.embedding.weight.fill_(1.0)
    store=tmp_path/'objects';snapshot=tmp_path/'trained.shards.json'
    export_state_sharded(source,store,snapshot,model_id='example/tiny',
                         revision='main',peer_id='alice',shard_bytes=65536)
    manifest=make_hf_manifest('example/tiny','main',
        [{'repo':'example/data','revision':'main','config':None,
          'split':'train','columns':{'text':'text'},'weight':1.0}])
    out=run({'manifest':manifest,'scope':'trained','checkpoint':str(snapshot),
             'checkpoint_revision':'main','store':str(store),'temp_dir':str(tmp_path),
             'precision':'auto','prompt':'test','max_tokens':1})
    assert out['text']=='7'
    assert out['checkpoint_seq']==0
