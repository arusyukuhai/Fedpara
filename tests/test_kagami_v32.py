"""V32 actual learned-state inference, architecture choice and security regressions."""
import base64
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.model_design import default_python_template, extract_config, validate_design
from patchlib.shards import export_state_sharded, _blob_path, shard_map, read_manifest
from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id
from patchlib.kagami_eval_worker import load_snapshot_model


def payload(room=None,**extra):
    return {'room':room or create_room_id(),'model':'demo/tiny','revision':'abcdef0123456789',
        'task':'causal-lm','seed':42,'sources':[{'repo':'demo/text','revision':'main',
            'config':None,'split':'train','columns':{'text':'text'},'weight':1.0}],**extra}


def test_three_creation_modes_and_pinned_peer_identity(tmp_path):
    ctrl=WebviewController(tmp_path)
    template=default_python_template()
    uploaded=ctrl.upload_code({'name':'architecture.py',
        'contents':base64.b64encode(template.encode()).decode()})
    for mode in ('weights','architecture','code'):
        settings=payload(creation_mode=mode,initialization='pretrained',
                         code_path=uploaded['path'] if mode=='code' else None)
        saved=ctrl.save(settings)
        manifest=saved['manifest']
        assert (manifest.get('design') is None)==(mode=='weights')
        assert ctrl.select_model({'room':settings['room']})['sha256']==saved['sha256']
        from patchlib.live_initialization import room_initialization
        assert room_initialization(manifest)==('pretrained' if mode=='weights' else 'config')
        if mode=='code':
            assert manifest['design']['model_type']=='gpt2'
            assert manifest['code']['sha256']==uploaded['sha256']
            assert ctrl.save(settings|{'code_path':None})['sha256']==saved['sha256']

def test_remote_join_python_manifest_needs_no_code_on_peer(tmp_path):
    author=WebviewController(tmp_path/'author')
    source=default_python_template()
    file=author.upload_code({'name':'model.py',
       'contents':base64.b64encode(source.encode('utf8')).decode()})
    args=payload(creation_mode='code',code_path=file['path'],initialization='config')
    upstream=author.save(args)
    joiner=WebviewController(tmp_path/'joiner')
    joiner.imported[args['room']]=upstream['manifest']
    joined=joiner.save(args|{'code_path':None})
    assert joined['sha256']==upstream['sha256']
    assert joined['manifest']['design']['model_type']=='gpt2'
    with pytest.raises(ValueError):
        WebviewController(tmp_path/'untrusted').save(args|{'code_path':None})
    with pytest.raises(ValueError):
        WebviewController(tmp_path/'mutated').save(args|{'code_path':None,
            'revision':'deadbeef'})


@pytest.mark.parametrize('source',[
    'from os import system\ndef build_config():\n return GPT2Config()',
    'def build_config():\n return eval("1")',
    'def build_config():\n return GPT2Config(n_embd=__import__("os"))',
    'def build_config():\n import os\n return GPT2Config()',
    'def build_config():\n return GPT2Config(**{"n_layer":2})',
])
def test_python_code_never_evaluated(source):
    with pytest.raises((ValueError,SyntaxError)):
        extract_config(source)


def test_code_config_is_small_and_fully_data_only():
    result=extract_config(default_python_template())
    assert result['kwargs']['n_layer']==6
    assert validate_design(json.loads(json.dumps(result)))==result


def test_trained_checkpoints_required_by_default_and_base_opt_in(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path);data=payload();ctrl.save(data)
    with pytest.raises(FileNotFoundError,match='学習済み'):
        ctrl.evaluate({'room':data['room'],'prompt':'hello','precision':'fp32'})
    fake=types.SimpleNamespace(pid=123,poll=lambda:None,terminate=lambda:None)
    monkeypatch.setattr('patchlib.webview_controller.subprocess.Popen',lambda *args,**kwargs:fake)
    result=ctrl.evaluate({'room':data['room'],'prompt':'hello','precision':'fp32','scope':'base'})
    assert result['scope']=='Explicit HF base'


def test_verified_effective_checkpoint_is_loaded_for_generation(tmp_path,monkeypatch):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(8,8,bias=False)
        @classmethod
        def from_config(cls,_cfg,**kw):return cls()
        def tie_weights(self):pass
    original=Toy()
    with torch.no_grad(): original.linear.weight.fill_(.73)
    ctrl=WebviewController(tmp_path);data=payload()
    ctrl.save(data)
    root=Path(ctrl._room_path(data['room']).parent)
    prof='testprofile'
    store=root/'models'/prof
    folder=root/'checkpoints'/prof
    folder.mkdir(parents=True)
    saved=folder/'effective-000.shards.json'
    export_state_sharded(original,store,saved,model_id=data['model'],
                         revision=data['revision'],peer_id='alice',seq=4,shard_bytes=65536)
    found=ctrl.available_checkpoints({'room':data['room']})['checkpoints']
    assert found[0]['seq']==4
    # Fake only Transformers *constructor*: genuine PyTorch and shard SHA-256
    # checks load actual bytes without loading any HF pretrained weights.
    transformers=types.ModuleType('transformers')
    transformers.AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**kw:object())
    transformers.AutoModelForCausalLM=Toy
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    options={'checkpoint':str(saved),'store':str(store),
             'checkpoint_revision':data['revision']}
    restored,state=load_snapshot_model(options,torch,ctrl.select_model({'room':data['room']})['manifest'])
    assert torch.equal(restored.linear.weight,original.linear.weight)
    assert state['seq']==4
    # A tampered shard must be rejected before any inference.
    blob=_blob_path(store,next(iter(shard_map(read_manifest(saved)))))
    with blob.open('r+b') as handle:handle.write(b'evil')
    with pytest.raises(Exception,match='corrupt|hash|shard'):
        load_snapshot_model(options,torch,ctrl.select_model({'room':data['room']})['manifest'])


def test_explicit_snapshot_request_is_nonblocking_and_relay_stays_online(tmp_path):
    c=WebviewController(tmp_path);data=payload();c.save(data)
    path=c._room_path(data['room']).parent/'request'
    c._current_room=data['room'];c._snapshot_request_file=path
    c._checkpoint_path=path.with_suffix('.shards.json')
    c.train.status=lambda:True
    a=c.request_eval_snapshot({'room':data['room']})
    b=c.request_eval_snapshot({'room':data['room']})
    assert a['requested'] and b['pending'] and path.is_file()


def test_streaming_safetensors_export_restores_actual_weights(tmp_path):
    from safetensors import safe_open
    from patchlib.snapshot_safetensors import export_hf_safetensors
    model=nn.Sequential(nn.Linear(31,29,bias=True),nn.LayerNorm(29))
    with torch.no_grad():
        for _,tensor in model.state_dict().items(): tensor.fill_(.125)
    source=tmp_path/'stored';checkpoint=tmp_path/'snapshot.shards.json'
    export_state_sharded(model,source,checkpoint,model_id='demo/tiny',
                         revision='0123456789abcdef',peer_id='alice',seq=5,shard_bytes=65536)
    output=tmp_path/'huggingface'
    report=export_hf_safetensors(checkpoint,source,output,max_shard_bytes=16*1024**2)
    assert report['tensors']==len(model.state_dict())
    index=json.loads((output/'model.safetensors.index.json').read_text())
    assert index['metadata']['total_size']>0
    assert set(index['weight_map'])==set(model.state_dict())
    for name,filename in index['weight_map'].items():
        with safe_open(str(output/filename),framework='pt',device='cpu') as f:
            assert torch.equal(f.get_tensor(name),model.state_dict()[name])


def test_streaming_safetensors_export_rejects_corrupted_shards(tmp_path):
    from patchlib.snapshot_safetensors import export_hf_safetensors
    model=nn.Linear(5,7)
    source=tmp_path/'stored';checkpoint=tmp_path/'snapshot.shards.json'
    export_state_sharded(model,source,checkpoint,model_id='demo/tiny',
                         revision='0123456789abcdef',peer_id='alice',shard_bytes=65536)
    blob=_blob_path(source,next(iter(shard_map(read_manifest(checkpoint)))))
    with blob.open('r+b') as handle:handle.write(b'CORRUPT')
    with pytest.raises(Exception,match='digest mismatch'):
        export_hf_safetensors(checkpoint,source,tmp_path/'out')
    assert not (tmp_path/'out'/'model.safetensors.index.json').exists()


def test_end_to_end_trained_scope_uses_learned_weights_not_hf_base(tmp_path,monkeypatch):
    from patchlib.kagami_eval_worker import run
    class MinimalLM(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding=nn.Embedding(5,4)
        @classmethod
        def from_config(cls,_cfg,**kw):return cls()
        @classmethod
        def from_pretrained(cls,*a,**kw):
            result=cls()
            with torch.no_grad():result.embedding.weight.fill_(-2)
            return result
        def tie_weights(self): pass
        def get_input_embeddings(self):return self.embedding
        def generate(self,input_ids,**kw):
            code=10 if float(self.embedding.weight[0,0])>0 else 20
            return torch.cat([input_ids,torch.tensor([[code]],device=input_ids.device)],dim=1)
        def eval(self):return self
    class FakeTokenizer:
        pad_token_id=0; eos_token_id=2
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def __call__(self,text,**kw):return {'input_ids':torch.tensor([[1,2]])}
        def decode(self,ids,**kw):return ','.join(map(str,ids.tolist()))
    def toy_model():
        model=MinimalLM()
        with torch.no_grad():model.embedding.weight.fill_(1.5)
        return model
    c=WebviewController(tmp_path);args=payload();manifest=c.save(args)['manifest']
    root=c._room_path(args['room']).parent
    store=root/'models'/'p';checkpoint=root/'checkpoints'/'p'/'trained.shards.json'
    checkpoint.parent.mkdir(parents=True)
    export_state_sharded(toy_model(),store,checkpoint,model_id=args['model'],
                         revision=args['revision'],peer_id='alice',seq=13,shard_bytes=65536)
    transformers=types.ModuleType('transformers')
    transformers.AutoTokenizer=FakeTokenizer
    transformers.AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**kw:types.SimpleNamespace())
    transformers.AutoModelForCausalLM=MinimalLM
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    base={'manifest':manifest,'precision':'fp32','prompt':'hello','max_tokens':1,
          'checkpoint':str(checkpoint),'store':str(store),'checkpoint_revision':args['revision'],
          'temp_dir':str(tmp_path)}
    result=run(base|{'scope':'trained'})
    assert result['text']=='10' and result['checkpoint_seq']==13
    result_base=run(base|{'scope':'base'})
    assert result_base['text']=='20' and result_base['checkpoint_seq'] is None


def test_tokenizer_provenance_pinned_to_code_model(tmp_path):
    from patchlib.model_design import tokenizer_identity
    ctrl=WebviewController(tmp_path)
    template=default_python_template()
    uploaded=ctrl.upload_code({'name':'config.py','contents':base64.b64encode(template.encode()).decode()})
    args=payload(creation_mode='code',code_path=uploaded['path'],
        tokenizer_model='roneneldan/TinyStories-1M',tokenizer_revision='77f1b168e219585646439073245fe87e56b3023e')
    manifest=ctrl.save(args)['manifest']
    assert tokenizer_identity(manifest)==(args['tokenizer_model'],args['tokenizer_revision'])
    assert manifest['model']['id']=='demo/tiny'
    broken=manifest.copy();broken['design']=dict(manifest['design'],tokenizer={'id':'../secrets','revision':'main'})
    from patchlib.room_manifest import validate_manifest
    with pytest.raises(ValueError):validate_manifest(broken)
