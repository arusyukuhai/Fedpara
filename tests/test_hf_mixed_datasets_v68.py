"""Hub selected SFT/PPO rows, GUI wiring, immutable dataset revision and local policies."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.hf_mixed_datasets_v68 import HFRowCycle, HFMixedDatasetError, source_spec
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options, objective_policy
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, MixedLearningError
from patchlib.webview_gui import FedparaBridge, compose_html
from test_gui_mixed_critic_v66 import make_controller

REV='abcdef0123456789'*2+'abcdef01' # 40 hex characters


def hf_settings(**updates):
    values={**DEFAULT,'enabled':True,'sft_hf_repo':'myorg/sft',
        'sft_hf_revision':REV,'sft_hf_config':'v1', 'sft_hf_split':'train',
        'sft_hf_prompt_column':'instruction','sft_hf_response_column':'output',
        'ppo_hf_repo':'myorg/ppo','ppo_hf_revision':REV,
        'ppo_hf_prompt_column':'question','ppo_hf_reference_column':'answer'}
    values.update(updates)
    return values


def test_gui_normalization_flags_policy_and_no_public_dataset_identity(tmp_path):
    v=normalize(hf_settings(),task='causal-lm')
    cli=command_options(v,state_dir=tmp_path/'critic')
    assert cli[cli.index('--sft-hf-revision')+1]==REV
    assert cli[cli.index('--ppo-hf-prompt-column')+1]=='question'
    assert '--ppo-jsonl' not in cli and '--sft-jsonl' not in cli
    policy=objective_policy(v)
    assert policy['objectives']==[True,True]
    assert 'myorg/sft' not in json.dumps(policy)
    assert 'myorg/ppo' not in json.dumps(policy)
    assert v['sft_jsonl']=='' and v['ppo_jsonl']==''
    # Mixing local/Hub is per-peer and source identity must not change consensus.
    path=tmp_path/'sft.jsonl';path.write_text('{"prompt":"q","response":"a"}\n')
    other=normalize({**v,'sft_hf_repo':'','sft_hf_revision':'', 'sft_hf_config':'',
                     'sft_jsonl':str(path)},task='causal-lm')
    assert objective_policy(other)==policy


def test_fail_closed_mixed_sources_invalid_revision_or_mapping(tmp_path):
    p=tmp_path/'local.jsonl';p.write_text('{"prompt":"q","response":"a"}\n')
    for change in ({'sft_hf_revision':'main'}, {'sft_hf_repo':'https://bad'},
                   {'sft_jsonl':str(p)}, {'sft_hf_prompt_column':'../../x'},
                   {'sft_hf_config':'foo/bar'}, {'sft_hf_prompt_column':'output'}):
        with pytest.raises(ValueError):normalize(hf_settings(**change),task='causal-lm')
    with pytest.raises(ValueError):normalize(hf_settings(ppo_hf_repo='',ppo_hf_revision=REV),task='causal-lm')
    assert source_spec(SimpleNamespace(sft_hf_repo='',sft_jsonl=str(p)),'sft') is None


def test_streaming_two_splits_nested_columns_and_cycle():
    calls=[]
    def factory(repo,**kw):
        calls.append((repo,kw))
        if repo=='myorg/sft':return iter([{'instruction':{'text':'Which?'},'output':'Answer!'}])
        return iter([{'question':[{'role':'user','content':'Hi'}], 'answer':'Hello'}])
    s=source_spec(SimpleNamespace(**hf_settings()),'sft')
    s['prompt_column']='instruction.text'
    a=HFRowCycle(s,'sft',dataset_factory=factory)
    assert a.next()=={'prompt':'Which?','response':'Answer!'}
    assert a.next()=={'prompt':'Which?','response':'Answer!'}
    p=HFRowCycle(source_spec(SimpleNamespace(**hf_settings()),'ppo'),'ppo',dataset_factory=factory)
    assert p.next()=={'prompt':'user: Hi','reference':'Hello'}
    assert len(calls)>=3
    assert calls[0][1]=={'name':'v1','split':'train','revision':REV,'streaming':True}
    a.close();p.close()


def test_ppo_without_reference_requires_external_reward():
    config=source_spec(SimpleNamespace(**hf_settings()),'ppo')
    cycle=HFRowCycle(config,'ppo',dataset_factory=lambda *a,**k:iter([{'question':'Hello'}]))
    assert cycle.next()=={'prompt':'Hello'}
    cycle.close()
    with pytest.raises(HFMixedDatasetError,match='Missing Hub column'):
        HFRowCycle(config,'ppo',dataset_factory=lambda *a,**k:iter([{'bad':'row'}])).next()
    with pytest.raises(HFMixedDatasetError,match='empty'):
        HFRowCycle(config,'ppo',dataset_factory=lambda *a,**k:iter([])).next()


def test_actual_sft_and_ppo_gradient_with_fake_hf_sources(monkeypatch):
    import patchlib.mixed_learning_v65 as mixed
    real_hf=mixed.HFRowCycle
    def mocked(spec,role):
        return real_hf(spec,role,dataset_factory=lambda *a,**kw: iter([
            {'instruction':'2+2?','output':'4'}] if role=='sft' else
            [{'question':'2+2?','answer':'4'}]))
    monkeypatch.setattr(mixed,'HFRowCycle',mocked)
    v=hf_settings()
    args=SimpleNamespace(**v,hf_task='causal-lm',loss_policy='auto',max_length=64,input_map='',
                         mix_policy=None,room_id='hf-test')
    class TinyTok:
        eos_token_id=1
        def encode(self,text,add_special_tokens=False):return [2+ord(c)%12 for c in text]
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(20,16);self.head=nn.Linear(16,20)
        def forward(self,input_ids,attention_mask=None,labels=None):
            return {'logits':self.head(self.emb(input_ids))}
    m=TinyModel();trainer=MixedObjectiveTrainer(args,TinyTok())
    loss=trainer.sft_loss(m,'cpu');assert torch.isfinite(loss);loss.backward()
    assert m.head.weight.grad is not None and m.head.weight.grad.abs().sum()>0
    rollout=trainer._new_rollout(m,'cpu')
    assert rollout is not None
    trainer.close()


def test_gui_real_bridge_hf_command_and_pinned_room(tmp_path,monkeypatch):
    c,room,commands=make_controller(tmp_path,monkeypatch)
    b=FedparaBridge(c)
    options=hf_settings()
    kwargs={'room':room,'ws':'ws://127.0.0.1:8080/ws','kind':'loha','rank':16,
            'streaming_base':True,'mixed_training':options}
    result=b.start_training(kwargs)
    assert result['mixed_enabled']
    assert '--sft-hf-repo' in commands[-1] and '--ppo-hf-repo' in commands[-1]
    assert b.mixed_settings({'room':room})['sft_hf_revision']==REV
    assert 'myorg/sft' not in json.dumps(c._room_path(room).read_text())
    # A returning peer may use local files with same objective policy.
    sft=tmp_path/'local.jsonl';sft.write_text('{"prompt":"Q","response":"A"}\n')
    more=dict(options,sft_jsonl=str(sft),sft_hf_repo='',sft_hf_revision='',sft_hf_config='')
    b.start_training({**kwargs,'mixed_training':more})


def test_gui_search_pin_uses_metadata_only(tmp_path,monkeypatch):
    c,room,_=make_controller(tmp_path,monkeypatch)
    class FakeApi:
        def dataset_info(self,repo):return SimpleNamespace(sha=REV,gated=False)
        def list_dataset_parquet_files(self,repo,revision):
            assert revision==REV
            return [SimpleNamespace(config='default',split='train')]
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=FakeApi))
    info=FedparaBridge(c).pin_mixed_dataset({'repo':'myorg/sft'})
    assert info['revision']==REV and info['variants']==[{'config':'default','split':'train'}]
    with pytest.raises(ValueError):c.pin_mixed_dataset({'repo':'https://example.com'})
    markup=compose_html()
    for name in ('mixSftHfSearch','mixPpoHfSearch','mixSftHfRevision','mixPpoHfPrompt',
                 'mixSftSource','mixPpoSource'):
        assert name in markup


def test_backward_compatible_local_v67_gui_settings(tmp_path,monkeypatch):
    c,room,_=make_controller(tmp_path,monkeypatch)
    from patchlib.webview_controller import allocate_profile
    folder=c._room_path(room).parent
    profile=allocate_profile(folder)
    c.profiles[room]=profile
    existing={key:value for key,value in DEFAULT.items() if not ('_hf_' in key)}
    (folder/('mixed-local-'+profile.name+'.json')).write_text(json.dumps({'room':room,**existing}))
    result=c.mixed_settings({'room':room})
    assert result['sft_hf_repo']=='' and result['ppo_hf_split']=='train'
