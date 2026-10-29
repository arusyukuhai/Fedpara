"""Reward Model v67: real PPO path, immutable peer policy and safe scorer."""
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.reward_model_v67 import HFRewardScorer, RewardModelError, reward_spec
from patchlib.mix_gui_v66 import DEFAULT, normalize, command_options, objective_policy
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, MixedLearningError

REV='a'*40


def args(**overrides):
    data=dict(ppo_reward='hf_model',reward_model_id='ExampleCorp/quality-rm',
              reward_model_revision=REV,reward_model_device='cpu',
              reward_model_output='logits',reward_model_format='plain',
              reward_model_index=0,reward_model_scale=1.,reward_model_max_length=64)
    data.update(overrides)
    return SimpleNamespace(**data)


class Tokenizer:
    eos_token_id=1
    truncation_side='right'
    def encode(self,text,add_special_tokens=False):
        return [max(2,ord(char)%17) for char in text]
    def decode(self,ids,skip_special_tokens=True):
        return ''.join(chr(ord('a')+int(i)%26) for i in ids)
    def __call__(self,text,**kwargs):
        ids=self.encode(text)[-kwargs.get('max_length',512):] if self.truncation_side=='left' else self.encode(text)[:kwargs.get('max_length',512)]
        return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
    def apply_chat_template(self,messages,**kwargs):
        return self.encode(messages[0]['content']+messages[1]['content'])


class RewardNet(nn.Module):
    def __init__(self,num_labels=1):
        super().__init__();self.bias=nn.Parameter(torch.ones(num_labels));self.config=SimpleNamespace(num_labels=num_labels)
    def forward(self,input_ids,attention_mask=None):
        val=input_ids.float().sum(dim=1,keepdim=True)*.005
        return SimpleNamespace(logits=val+self.bias)


def test_scorer_reuses_own_tokenizer_freezes_and_bounds():
    score=HFRewardScorer(args(),model=RewardNet(),tokenizer=Tokenizer())
    r=score.score('question','answer')
    assert isinstance(r,float) and r>1.0
    assert not any(p.requires_grad for p in score.model.parameters())
    assert score.calls==1 and score.tok.truncation_side=='right'
    assert HFRewardScorer(args(reward_model_format='chat'),model=RewardNet(),tokenizer=Tokenizer()).score('q','a')>0


def test_reward_model_pin_and_rejection():
    for changed in (dict(reward_model_revision='main'),dict(reward_model_id='/evil/file'),
                    dict(reward_model_format='unsafe'),dict(reward_model_scale=float('nan')),
                    dict(reward_model_output='hidden_states')):
        with pytest.raises(RewardModelError):reward_spec(args(**changed))
    scorer=HFRewardScorer(args(),model=RewardNet(2),tokenizer=Tokenizer())
    scorer.config['index']=2
    with pytest.raises(RewardModelError,match='smaller'):scorer.score('a','b')
    scorer.config['index']=0
    scorer.model.forward=lambda **kw:SimpleNamespace(logits=torch.full((1,1),float('nan')))
    with pytest.raises(RewardModelError,match='finite'):scorer.score('a','b')


def test_loader_flags_and_missing_reward_head(monkeypatch):
    records=[]
    class AutoModel:
        @staticmethod
        def from_pretrained(repo,**kwargs):
            records.append((repo,kwargs))
            return (RewardNet(),{'missing_keys':[]})
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(repo,**kwargs):
            records.append(('tok',kwargs));return Tokenizer()
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModelForSequenceClassification=AutoModel,AutoModel=AutoModel,AutoTokenizer=AutoTokenizer))
    scorer=HFRewardScorer(args())
    assert scorer.score('abc','xyz')>0
    assert records[0][1]['trust_remote_code'] is False
    assert records[0][1]['revision']==REV
    assert records[0][1]['output_loading_info'] is True
    assert records[0][1]['use_safetensors'] is True
    assert records[1][1]['trust_remote_code'] is False
    class UnsafeModel:
        @staticmethod
        def from_pretrained(*args,**kwargs):return RewardNet(),{'missing_keys':['score.weight']}
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModelForSequenceClassification=UnsafeModel,AutoModel=UnsafeModel,AutoTokenizer=AutoTokenizer))
    with pytest.raises(RewardModelError,match='missing/untrained'):HFRewardScorer(args())


def test_gui_reward_fields_cli_and_pin(tmp_path):
    f=tmp_path/'rollouts.jsonl';f.write_text(json.dumps({'prompt':'Explain gravity'})+'\n')
    raw={**DEFAULT,'enabled':True,'ppo_jsonl':str(f),'sft_weight':0.,
         'ppo_reward':'hf_model','reward_model_id':'ExampleCorp/quality-rm',
         'reward_model_revision':REV,'reward_model_format':'chat',
         'reward_model_scale':.5}
    normalized=normalize(raw,task='causal-lm')
    cli=command_options(normalized,state_dir=tmp_path/'critic')
    assert cli[cli.index('--reward-model-id')+1]=='ExampleCorp/quality-rm'
    assert cli[cli.index('--reward-model-format')+1]=='chat'
    policy=objective_policy(normalized)
    assert policy['reward_source']=='external-hf-sequence-scorer-v1'
    assert policy['reward_model']['revision']==REV
    assert policy['reward_model']['scale']==.5
    assert str(tmp_path) not in str(policy)
    with pytest.raises(ValueError,match='revision'):
        normalize({**raw,'reward_model_revision':'main'},task='causal-lm')
    other=normalize({**raw,'ppo_jsonl':'','ppo_weight':.2},task='causal-lm',allow_pt_only=True)
    pinned=tmp_path/'pinned.json';pinned.write_text(json.dumps(policy))
    # PT-only workers may share the reward policy without downloading scorer.
    assert objective_policy(other,pinned=pinned)==policy


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(20,12);self.head=nn.Linear(12,20)
    def forward(self,input_ids,attention_mask=None):
        return {'logits':self.head(self.emb(input_ids))}


def test_real_ppo_uses_external_reward_not_reference(tmp_path,monkeypatch):
    path=tmp_path/'ppo.jsonl';path.write_text('{"prompt":"abc"}\n')
    x=args(ppo_jsonl=path,sft_jsonl=None,pt_weight=1.,sft_weight=0.,ppo_weight=.2,
           ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
           ppo_gae_lambda=.95,ppo_max_tokens=3,ppo_reuse=2,max_length=32,
           hf_task='causal-lm',loss_policy='auto',critic_kind='mlp',critic_hidden=16,
           critic_lr=.001,model='toy')
    class MockRM:
        def __init__(self,args):self.calls=0
        def score(self,prompt,completion):
            self.calls+=1
            assert prompt=='abc' and completion
            return 2.25
    import patchlib.reward_model_v67 as mod
    monkeypatch.setattr(mod,'HFRewardScorer',MockRM)
    m=MixedObjectiveTrainer(x,Tokenizer())
    torch.manual_seed(2)
    actor=TinyActor()
    loss,critic,metrics=m.ppo_loss(actor,torch.device('cpu'))
    assert metrics['reward']==2.25
    assert loss.requires_grad and critic.requires_grad
    (loss+critic).backward()
    assert any(p.grad is not None for p in actor.parameters())
    m.ppo_loss(actor,torch.device('cpu'))
    assert m.reward_scorer.calls==1  # PPO reuses the same terminal reward
    m.close()


def test_gui_has_live_fields():
    from pathlib import Path
    html=(Path(__file__).parents[1]/'web/desktop_webview.html').read_text()
    js=(Path(__file__).parents[1]/'web/desktop_webview.js').read_text()
    for tag in ('mixRewardFields','mixRewardModelId','mixRewardRevision','mixRewardDevice',
                'mixRewardOutput','mixRewardLoader','mixRewardFormat','mixRewardIndex','mixRewardScale','mixRewardMaxLength'):
        assert f'id="{tag}"' in html and tag in js


def test_hf_reward_metadata_discovery_and_revision_pin(tmp_path,monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.webview_gui import FedparaBridge
    class PublicModel:
        id='ExampleCorp/quality-rm';sha=REV;pipeline_tag='text-classification';gated=False
    class Hub:
        def list_models(self,**kw):
            assert kw['limit']<=15
            return [PublicModel()]
        def model_info(self,repo,**kw):
            assert repo==PublicModel.id
            return PublicModel()
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=Hub))
    bridge=FedparaBridge(WebviewController(tmp_path))
    assert bridge.search_reward_models({'query':'quality'})['models'][0]['id']==PublicModel.id
    info=bridge.inspect_reward_model({'repo':PublicModel.id})
    assert info['revision']==REV
    with pytest.raises(ValueError):bridge.inspect_reward_model({'repo':'https://evil.example'})
    with pytest.raises(ValueError):bridge.search_reward_models({'query':'x'*1001})


def test_gui_to_real_training_command_with_selected_reward(tmp_path,monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.webview_gui import FedparaBridge
    from patchlib.roomid import create_room_id
    c=WebviewController(tmp_path)
    room=create_room_id()
    c.save({'room':room,'model':'org/ourmodel','revision':'main',
            'task':'causal-lm','initialization':'pretrained','seed':42,
            'sources':[{'repo':'org/pretraining','revision':'main','config':None,
                        'split':'train','columns':{'text':'text'},'weight':1.}]})
    called=[]
    monkeypatch.setattr(c.train,'start',lambda cmd,log:called.append(cmd) or 201)
    path=tmp_path/'ppo.jsonl'
    path.write_text('{"prompt":"What is one plus one?"}\n')
    values={**DEFAULT,'enabled':True,'ppo_jsonl':str(path),
            'ppo_reward':'hf_model','reward_model_id':'ExampleCorp/quality-rm',
            'reward_model_revision':REV,'sft_weight':0.}
    r=FedparaBridge(c).start_training({'room':room,'ws':'ws://127.0.0.1:8080/ws',
        'streaming_base':True,'kind':'loha','rank':16,'mixed_training':values})
    assert r['mixed_enabled'] is True
    cmd=called[-1]
    for name,value in (('--ppo-reward','hf_model'),('--reward-model-id','ExampleCorp/quality-rm'),
                       ('--reward-model-revision',REV),('--reward-model-loader','sequence_classification')):
        assert cmd[cmd.index(name)+1]==value
    pinned=__import__('pathlib').Path(cmd[cmd.index('--mix-policy')+1]);policy=json.loads(pinned.read_text())
    assert policy['reward_model']['repo']=='ExampleCorp/quality-rm'
    assert str(tmp_path) not in json.dumps(policy)
