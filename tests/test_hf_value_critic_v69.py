"""v69: real PPO value regression on separate Hugging Face-pinned critic."""
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from patchlib.hf_value_critic_v69 import HFValueCritic, HFValueError, critic_spec
from patchlib.hf_critic_state_v69 import save, load, snapshot_file
from patchlib.mix_gui_v66 import DEFAULT,normalize,command_options,objective_policy
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer

SHA='b'*40


class Tokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [max(2,ord(c)%19) for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(65+int(x)%26) for x in ids)
    def __call__(self,text,**kw):
        ids=self.encode(text)[-kw.get('max_length',32):]
        return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}


class ValueModel(nn.Module):
    def __init__(self,num_labels=1):
        super().__init__()
        self.config=SimpleNamespace(num_labels=num_labels)
        self.body=nn.Linear(1,4)
        self.score=nn.Linear(4,1)
    def forward(self,input_ids,attention_mask=None):
        x=input_ids.float().mean(dim=1,keepdim=True)*.01
        return SimpleNamespace(logits=self.score(torch.tanh(self.body(x))))


class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.e=nn.Embedding(24,6);self.l=nn.Linear(6,24)
    def forward(self,input_ids,attention_mask=None):return {'logits':self.l(self.e(input_ids))}


def settings(tmp_path,**extra):
    data=dict(critic_kind='hf',critic_hf_repo='Org/critic',critic_hf_revision=SHA,
              critic_hf_loader='sequence_classification',critic_hf_device='cpu',critic_hf_max_length=64,
              critic_hidden=32,critic_lr=.003,critic_state_dir=tmp_path/'checkpoint',
              ppo_jsonl=tmp_path/'ppo.jsonl',sft_jsonl=None,ppo_reward='token_match',
              pt_weight=1.,sft_weight=0.,ppo_weight=.1,ppo_clip=.2,ppo_kl_coef=.02,
              ppo_value_coef=.5,ppo_gamma=.99,ppo_gae_lambda=.95,ppo_max_tokens=3,
              ppo_reuse=2,max_length=32,hf_task='causal-lm',loss_policy='auto',model='actor')
    data.update(extra)
    return SimpleNamespace(**data)


def hf_mock(monkeypatch,missing=None):
    def construct(*args,**kw):
        assert kw['revision']==SHA and kw['trust_remote_code'] is False
        assert kw['use_safetensors'] is True and kw['output_loading_info'] is True
        return ValueModel(),{'missing_keys':missing or []}
    class Auto:
        from_pretrained=staticmethod(construct)
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args,**kw):return Tokenizer()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForSequenceClassification=Auto,AutoModelForTokenClassification=Auto,AutoTokenizer=AutoTokenizer))


def test_hf_critic_load_and_head_only_training(monkeypatch,tmp_path):
    hf_mock(monkeypatch)
    arg=settings(tmp_path)
    c=HFValueCritic(arg)
    assert all(not p.requires_grad for p in c.model.body.parameters())
    assert all(p.requires_grad for p in c.head.parameters())
    old_body=c.model.body.weight.detach().clone()
    old_head=c.head.weight.detach().clone()
    optim=torch.optim.AdamW(c.trainable_parameters(),lr=.03)
    losses=c.values(['hello','hello there'],actor_device=torch.device('cpu'))
    assert losses.shape==(2,) and losses.requires_grad
    losses.square().sum().backward();optim.step()
    assert torch.equal(c.model.body.weight,old_body)
    assert not torch.equal(c.head.weight,old_head)
    save(arg.critic_state_dir,c,optim,identity='room',steps=1)
    restored=HFValueCritic(arg); opt2=torch.optim.AdamW(restored.trainable_parameters())
    assert load(snapshot_file(arg.critic_state_dir),restored,opt2,identity='room')==1
    for a,b in zip(c.head.parameters(),restored.head.parameters()):assert torch.equal(a,b)
    with pytest.raises(HFValueError,match='mismatch'):load(snapshot_file(arg.critic_state_dir),restored,opt2,identity='other-room')
    with pytest.raises(HFValueError,match='mismatch'):
        alt=HFValueCritic(settings(tmp_path,critic_hf_revision='c'*40),model=ValueModel(),tokenizer=Tokenizer())
        load(snapshot_file(arg.critic_state_dir),alt,torch.optim.AdamW(alt.trainable_parameters()),identity='room')


def test_hf_ppo_actual_rollout_critic_and_restart(monkeypatch,tmp_path):
    hf_mock(monkeypatch)
    (tmp_path/'ppo.jsonl').write_text('{"prompt":"abc","reference":"abcd"}\n')
    arg=settings(tmp_path)
    torch.manual_seed(66)
    train=MixedObjectiveTrainer(arg,Tokenizer());actor=Actor()
    actor_loss,value_loss,metrics=train.ppo_loss(actor,torch.device('cpu'))
    assert actor_loss.requires_grad and value_loss.requires_grad
    assert len(train.cached['critic_prefixes'])==train.cached['tokens'].numel()
    (actor_loss+value_loss).backward()
    assert train.critic.head.weight.grad is not None
    assert all(p.grad is None for p in train.critic.model.body.parameters())
    train.critic_optim.step();train.critic_steps+=1
    orig=train.critic.snapshot_weights();train.close()
    new=MixedObjectiveTrainer(arg,Tokenizer())
    assert new.critic_steps==1
    for key,val in new.critic.snapshot_weights().items():assert torch.equal(val,orig[key])
    new.close()


def test_hf_value_guardrails_and_missing_head(monkeypatch,tmp_path):
    arg=settings(tmp_path)
    for kw in (dict(critic_hf_revision='main'),dict(critic_hf_repo='../bad'),
               dict(critic_hf_loader='unknown'),dict(critic_hf_max_length=2)):
        with pytest.raises(HFValueError):critic_spec(settings(tmp_path,**kw))
    hf_mock(monkeypatch,missing=['score.weight'])
    with pytest.raises(HFValueError,match='missing'):HFValueCritic(arg)
    with pytest.raises(HFValueError,match='single scalar'):HFValueCritic(arg,model=ValueModel(2),tokenizer=Tokenizer())
    c=HFValueCritic(arg,model=ValueModel(),tokenizer=Tokenizer())
    c.model.forward=lambda **kwargs:SimpleNamespace(logits=torch.tensor([[float('nan')]]))
    with pytest.raises(HFValueError,match='nonfinite'):c.values(['hello'],actor_device=torch.device('cpu'))


def test_gui_hf_critic_cli_pins_policy_and_keeps_local_source(tmp_path):
    row=tmp_path/'a.jsonl';row.write_text('{"prompt":"a","reference":"b"}\n')
    values=normalize({**DEFAULT,'enabled':True,'ppo_jsonl':str(row),'sft_weight':0.,
                      'critic_kind':'hf','critic_hf_repo':'Org/critic','critic_hf_revision':SHA,
                      'critic_hf_max_length':128},task='causal-lm')
    opts=command_options(values,state_dir=tmp_path/'state')
    assert opts[opts.index('--critic-hf-repo')+1]=='Org/critic'
    assert opts[opts.index('--critic-hf-loader')+1]=='sequence_classification'
    policy=objective_policy(values)
    assert policy['critic']['hf']['revision']==SHA
    assert 'critic_hf_device' not in json.dumps(policy)
    assert str(row) not in json.dumps(policy)
    from patchlib.webview_gui import compose_html
    html=compose_html()
    for key in ('mixCriticHfRepo','mixCriticHfRevision','mixCriticSearch','mixCriticPin','mixCriticHub'):
        assert key in html


def test_hf_token_critic_prefix_values_are_differentiable(tmp_path):
    class Tokens(ValueModel):
        def forward(self,input_ids,attention_mask=None):
            hidden=torch.tanh(self.body(input_ids.float().unsqueeze(-1)*.01))
            return SimpleNamespace(logits=self.score(hidden))
    cfg=settings(tmp_path,critic_hf_loader='token_classification',critic_hf_max_length=16)
    tok=Tokenizer();tok.truncation_side='right'
    critic=HFValueCritic(cfg,model=Tokens(),tokenizer=tok)
    assert critic.tok.truncation_side=='left'
    values=critic.values(['first','first second'],actor_device=torch.device('cpu'))
    assert values.shape==(2,)
    values.mean().backward()
    assert critic.head.weight.grad is not None
    assert critic.model.body.weight.grad is None
    critic.model.forward=lambda **kw:SimpleNamespace(logits=torch.zeros(1,1))
    with pytest.raises(HFValueError,match='Token HF Critic'):
        critic.values(['something'],actor_device=torch.device('cpu'))
