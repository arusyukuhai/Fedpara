"""Offline wiring regression tests: official GPT-Neo TinyStories-1M benchmark.
No external model is downloaded in CI; use TinyStories model locally for E2E.
"""
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.effectiveness_benchmark import (
    TINYSTORIES_1M_ID, TINYSTORIES_1M_REVISION, TINYSTORIES_DATASET_REVISION, run_effectiveness_benchmark,
)
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html


class StubGPTNeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=types.SimpleNamespace(vocab_size=32, hidden_size=24,
                max_position_embeddings=128, model_type='gpt_neo', use_cache=True)
        self.transformer=nn.Module()
        self.transformer.wte=nn.Embedding(32,24)
        self.transformer.wpe=nn.Embedding(128,24)
        # Mirror the real GPT-Neo transformer naming used by the selective
        # adapter target policy; head and embedding MUST remain untouched.
        self.transformer.h=nn.ModuleList([nn.Module()])
        self.transformer.h[0].attn=nn.Module()
        self.transformer.h[0].attn.attention=nn.Module()
        self.transformer.h[0].attn.attention.q_proj=nn.Linear(24,24)
        self.transformer.h[0].mlp=nn.Module()
        self.transformer.h[0].mlp.c_fc=nn.Linear(24,24)
        self.proj=nn.Linear(24,24)
        self.lm_head=nn.Linear(24,32,bias=False)
        self.lm_head.weight=self.transformer.wte.weight
        self.dropout=nn.Dropout(.3)

    def forward(self,input_ids):
        block=self.transformer.h[0]
        positions=torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        x=self.transformer.wte(input_ids)+self.transformer.wpe(positions)
        x=self.dropout(torch.tanh(self.proj(block.mlp.c_fc(block.attn.attention.q_proj(x)))))
        return types.SimpleNamespace(logits=self.lm_head(x))


class StubTokenizer:
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c)%32 for c in text]


def install_fake_hf(monkeypatch):
    def load_model(model_id,**kwargs):
        assert model_id==TINYSTORIES_1M_ID
        assert kwargs==dict(revision=TINYSTORIES_1M_REVISION,
                 trust_remote_code=False, use_safetensors=False,
                 weights_only=True, attn_implementation='eager')
        return StubGPTNeo()
    def load_tokenizer(model_id,**kwargs):
        assert model_id==TINYSTORIES_1M_ID
        assert kwargs==dict(revision=TINYSTORIES_1M_REVISION,trust_remote_code=False)
        return StubTokenizer()
    def load_config(model_id,**kwargs):
        assert model_id==TINYSTORIES_1M_ID
        assert kwargs=={'revision':TINYSTORIES_1M_REVISION,'trust_remote_code':False}
        return types.SimpleNamespace(model_type='gpt_neo',_attn_implementation='eager')
    def from_config(config,**kwargs):
        assert config.model_type=='gpt_neo'
        assert kwargs=={'trust_remote_code':False}
        return StubGPTNeo()
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=load_config),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model,from_config=from_config),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer)))


def test_tinystories_1m_pipeline_safe_pinned_and_persistent(monkeypatch,tmp_path):
    install_fake_hf(monkeypatch)
    kw=dict(model='tinystories-1m', dataset='builtin',peers=2,
                 rounds=2, local_steps=1,batch_size=2,context=12,rank=4,
                 sequences=16,seed=17,output_dir=tmp_path)
    out=run_effectiveness_benchmark(**kw)
    assert out['configuration']['model_id']==TINYSTORIES_1M_ID
    assert out['configuration']['model_revision']==TINYSTORIES_1M_REVISION
    assert out['configuration']['width']==24
    assert out['configuration']['dataset']['type']=='builtin-story-smoke-test'
    assert out['configuration']['adapter_counts']['groups']==4
    assert out['configuration']['initialization']=='random'
    assert out['configuration']['target_scope']=='gpt_neo_full'
    assert out['configuration']['adapter_counts']['dense_fallback']==0
    assert out['configuration']['adapter_counts']['dense_fallback']==0
    assert out['trace'][-1]['tokens_seen']==2*2*1*2*12
    assert out['compute']['p2p_divided_by_peers_s']==pytest.approx(out['compute']['p2p_serial_s']/2)
    assert math.isfinite(out['loss']['central']) and math.isfinite(out['loss']['p2p'])
    assert (tmp_path/'effectiveness-result.json').exists()
    assert json.loads((tmp_path/'effectiveness-result.json').read_text())['configuration']['model']=='tinystories-1m'
    again=run_effectiveness_benchmark(**{**kw,'output_dir':None})
    assert again['loss']==out['loss']


def test_tinystories_1m_local_file_bpe(monkeypatch,tmp_path):
    install_fake_hf(monkeypatch)
    source=tmp_path/'stories.txt'
    source.write_text('Once upon a time in a tiny town.\n'*250,encoding='utf8')
    data=run_effectiveness_benchmark(model='tinystories-1m',text_file=source,
        peers=2,rounds=1,local_steps=1,rank=4,context=16,sequences=16,seed=123)
    assert data['configuration']['dataset']['type']=='local-gpt2-bpe'
    assert data['configuration']['dataset']['heldout_disjoint']


def test_tinystories_1m_uses_disjoint_hf_splits(monkeypatch):
    install_fake_hf(monkeypatch)
    calls=[]
    def load_dataset(name,split,streaming,revision):
        calls.append((name,split,streaming,revision))
        return iter([{'text':'Here is a long example story. '*8} for _ in range(128)])
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=load_dataset))
    out=run_effectiveness_benchmark(model='tinystories-1m',dataset='tinystories',
             peers=2,rounds=1,local_steps=1,rank=4,context=12,sequences=16,
             validation_sequences=64)
    assert [x[1] for x in calls]==['train','validation']
    assert all(x[0]=='roneneldan/TinyStories' and x[2] and x[3]==TINYSTORIES_DATASET_REVISION for x in calls)
    assert out['configuration']['dataset']['heldout_split']=='validation'
    assert out['configuration']['dataset']['revision']==TINYSTORIES_DATASET_REVISION


def test_tinystories_1m_gui_defaults_and_validations(tmp_path):
    html=compose_html()
    assert 'id="benchModel"' in html and 'id="benchDataset"' in html
    assert 'option value="tinystories-1m" selected' in html
    assert 'option value="tinystories" selected' in html
    assert 'id="benchInitialization"' in html
    ctl=WebviewController(tmp_path)
    try:
        with pytest.raises(ValueError):ctl.start_benchmark({'model':'forged-model'})
        with pytest.raises(ValueError):ctl.start_benchmark({'model':'toy','dataset':'tinystories'})
    finally:ctl.close()


def test_tinystories_1m_rejects_unsafe_torch(monkeypatch):
    install_fake_hf(monkeypatch)
    monkeypatch.setattr(torch,'__version__','2.5.1',raising=False)
    with pytest.raises(RuntimeError,match='torch>=2.6'):
        run_effectiveness_benchmark(model='tinystories-1m',dataset='builtin',peers=2,rounds=1,initialization='pretrained')


def test_large_volume_tinystories_two_splits_and_exact_counts(monkeypatch):
    install_fake_hf(monkeypatch)
    requested=[]
    def load_dataset(name,split,streaming,revision):
        requested.append(split)
        n=2*256 if split=='train' else 384
        return iter([{'text':f'Story {i}: the little dog ran into the village. '*4}
                     for i in range(n)])
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=load_dataset))
    out=run_effectiveness_benchmark(model='tinystories-1m',dataset='tinystories',
        peers=2,rounds=1,local_steps=1,rank=4,context=12,sequences=256,
        validation_sequences=384)
    assert requested==['train','validation']
    assert out['configuration']['sequences_per_peer']==256
    assert out['configuration']['validation_sequences']==384
    assert out['configuration']['dataset']['training_sequences_total']==512
    assert out['configuration']['dataset']['validation_sequences']==384


def test_missing_data_fails_closed_instead_of_silent_resampling(monkeypatch):
    install_fake_hf(monkeypatch)
    def load_dataset(name,split,streaming,revision):
        return iter([{'text':'A short but valid story. '*8} for _ in range(16)])
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=load_dataset))
    with pytest.raises(RuntimeError,match='only 16/32'):
        run_effectiveness_benchmark(model='tinystories-1m',dataset='tinystories',
            peers=2,rounds=1,local_steps=1,rank=4,context=12,sequences=16,
            validation_sequences=64)


def test_training_batch_uses_without_replacement_within_cycle():
    from patchlib.effectiveness_benchmark import _batch
    partition=torch.arange(25*9).reshape(25,9)
    selected=[]
    for step in range(5):
        first,y=_batch(partition,epoch=0,step=step,batch_size=5,seed=4,peer=0)
        again,_=_batch(partition,epoch=0,step=step,batch_size=5,seed=4,peer=0)
        assert torch.equal(first,again)
        selected.extend(first[:,0].tolist())
    assert len(set(selected))==25


def test_gui_passes_volume_parameters(tmp_path,monkeypatch):
    ctl=WebviewController(tmp_path)
    calls=[]
    class DummyProcess:
        pid=43210
        def poll(self):return 0
    def launch(command,**kwargs):
        calls.append(command)
        return DummyProcess()
    monkeypatch.setattr('patchlib.webview_controller.subprocess.Popen',launch)
    try:
        response=ctl.start_benchmark({'model':'tinystories-1m','dataset':'tinystories',
            'peers':2,'rounds':1,'sequences':2048,'validation_sequences':1024})
        assert response['pid']==43210
        assert calls and calls[0][calls[0].index('--sequences')+1]=='2048'
        assert calls[0][calls[0].index('--validation-sequences')+1]=='1024'
        for payload in ({'sequences':8193}, {'validation_sequences':15},
                        {'sequences':2.5}):
            with pytest.raises(ValueError):ctl.start_benchmark(payload)
    finally:
        ctl.close()


def test_random_init_never_uses_pretrained_weight_loader(monkeypatch):
    install_fake_hf(monkeypatch)
    stub=sys.modules['transformers']
    def forbid(*a,**kw):
        raise AssertionError('random mode must NOT load pretrained model weights')
    stub.AutoModelForCausalLM.from_pretrained=forbid
    kw=dict(model='tinystories-1m',dataset='builtin',peers=2,rounds=1,
            local_steps=1,batch_size=2,context=12,rank=16,sequences=16,
            validation_sequences=16,seed=919,initialization='random')
    result=run_effectiveness_benchmark(**kw)
    assert result['configuration']['base_weights_frozen'] is True
    assert result['configuration']['initialization']=='random'
    assert result['configuration']['target_scope']=='gpt_neo_full'
    again=run_effectiveness_benchmark(**kw)
    assert result['loss']==again['loss']


def test_pretrained_explicit_optional_control(monkeypatch):
    install_fake_hf(monkeypatch)
    kw=dict(model='tinystories-1m',dataset='builtin',peers=2,rounds=1,
            local_steps=1,batch_size=2,context=12,rank=16,sequences=16,
            validation_sequences=16,seed=919,initialization='pretrained')
    out=run_effectiveness_benchmark(**kw)
    assert out['configuration']['initialization']=='pretrained'
