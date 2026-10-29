"""Regression cases for v73: HF KV-cache mask, frozen value-feature reuse, stale PPO policy."""
from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn

from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, MixedLearningError
from patchlib.hf_value_critic_v69 import HFValueCritic

SHA='d'*40

class Tok:
    eos_token_id=None
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):
        return [2+(ord(c)%29) for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(chr(65+int(x)%26) for x in ids)
    def __call__(self,text,*,return_tensors,truncation,max_length):
        items=self.encode(text)[-max_length:]
        return {'input_ids':torch.tensor([items]),'attention_mask':torch.ones(1,len(items),dtype=torch.long)}

class StrictCausalCache(nn.Module):
    """Emulates common HF contract: cache decode gets full-prefix mask."""
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(40,12)
        self.head=nn.Linear(12,40)
        self.tokens_seen=[]
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
        prior=0 if past_key_values is None else past_key_values.shape[1]
        expected=prior+input_ids.shape[1]
        assert attention_mask is not None and tuple(attention_mask.shape)==(1,expected), (
            f'expected full-prefix attention_mask of length {expected}, got {attention_mask.shape}')
        self.tokens_seen.append(input_ids.numel())
        history=torch.cat([past_key_values,input_ids],dim=1) if past_key_values is not None else input_ids
        logits=self.head(self.emb(history).cumsum(1))[:,-input_ids.shape[1]:]
        return {'logits':logits,'past_key_values':history.detach() if use_cache else None}

class SlowValue(nn.Module):
    def __init__(self,postprocess=False):
        super().__init__()
        self.config=SimpleNamespace(num_labels=1,pad_token_id=0)
        self.backbone=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh())
        self.score=nn.Linear(16,1)
        self.calls=0
        self.postprocess=postprocess
    def forward(self,input_ids,attention_mask=None):
        self.calls+=1
        weights=attention_mask.float()
        pooled=(input_ids.float()*weights).sum(1,keepdim=True)/weights.sum(1,keepdim=True)
        val=self.score(self.backbone(pooled/40.))
        return SimpleNamespace(logits=(2*val if self.postprocess else val))

def args_for_ppo(p,**kw):
    opts=dict(sft_jsonl=None,ppo_jsonl=p,pt_weight=0.,sft_weight=0.,ppo_weight=1.,
              ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
              ppo_gae_lambda=.95,ppo_max_tokens=5,ppo_reuse=2,ppo_reward='exact',
              max_length=48,hf_task='causal-lm',loss_policy='auto',critic_kind='mlp',
              critic_hidden=32,critic_lr=.001,ppo_kv_cache='auto',input_map='')
    opts.update(kw)
    return SimpleNamespace(**opts)

def critic_args():
    return SimpleNamespace(critic_kind='hf',critic_hf_repo='Org/value',critic_hf_revision=SHA,
                           critic_hf_loader='sequence_classification',critic_hf_device='cpu',
                           critic_hf_max_length=128)

def test_kv_cache_full_attention_mask_and_rollout_consistency(tmp_path):
    p=tmp_path/'samples.jsonl';p.write_text(json.dumps({'prompt':'abcdefgh','reference':'xyz'})+'\n')
    torch.manual_seed(40); fast=StrictCausalCache(); slow=StrictCausalCache()
    slow.load_state_dict(fast.state_dict())
    auto=MixedObjectiveTrainer(args_for_ppo(p),Tok())
    off=MixedObjectiveTrainer(args_for_ppo(p,ppo_kv_cache='off'),Tok())
    try:
        torch.manual_seed(55)
        first=auto._new_rollout(fast,torch.device('cpu'))
        torch.manual_seed(55)
        second=off._new_rollout(slow,torch.device('cpu'))
        assert torch.equal(first['tokens'],second['tokens'])
        assert torch.allclose(first['old_lp'],second['old_lp'],atol=1e-6)
        assert sum(fast.tokens_seen)<sum(slow.tokens_seen)
    finally:
        auto.close();off.close()

def test_frozen_hf_backbone_reuses_features_but_head_is_retrained():
    torch.manual_seed(8)
    model=SlowValue()
    critic=HFValueCritic(critic_args(),model=model,tokenizer=Tok())
    prefixes=['a','ab','abcd','hello','hello world','longish input']
    old=critic.values(prefixes,actor_device=torch.device('cpu'))
    calls=model.calls
    assert critic._cached_features is not None
    optimizer=torch.optim.SGD(critic.trainable_parameters(),lr=.25)
    loss=critic.values(prefixes,actor_device=torch.device('cpu')).square().sum()
    loss.backward()
    assert model.calls==calls, 'backbone must not run again on PPO reuse'
    assert model.score.weight.grad is not None and model.score.weight.grad.abs().sum()>0
    optimizer.step();optimizer.zero_grad()
    current=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert model.calls==calls, 'head updates alone must not invalidate frozen features'
    assert not torch.allclose(old,current)
    assert torch.allclose(current,model_value_reference(critic,prefixes),atol=1e-6)
    assert model.calls>calls
    critic.values(prefixes+['new state'],actor_device=torch.device('cpu'))
    assert model.calls>calls+1, 'changed rollout invalidates feature cache'

def model_value_reference(critic,prefixes):
    return torch.stack([critic.model(**critic.tok(s,return_tensors='pt',truncation=True,max_length=128)).logits[0,0] for s in prefixes])

def test_custom_postprocessing_disables_feature_shortcut():
    model=SlowValue(postprocess=True)
    critic=HFValueCritic(critic_args(),model=model,tokenizer=Tok())
    prefixes=['a','abc','xy']
    first=critic.values(prefixes,actor_device=torch.device('cpu'))
    before=model.calls
    second=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert model.calls>before
    assert critic._cached_features is None
    assert torch.equal(first,second)

def test_ppo_extreme_stale_replay_fails_closed(tmp_path):
    p=tmp_path/'samples.jsonl';p.write_text(json.dumps({'prompt':'abc','reference':'xyz'})+'\n')
    model=StrictCausalCache()
    trainer=MixedObjectiveTrainer(args_for_ppo(p,ppo_kv_cache='off'),Tok())
    try:
        actor_loss,_,_=trainer.ppo_loss(model,torch.device('cpu'))
        assert actor_loss.requires_grad
        trainer.cached['old_lp']-=100.
        with pytest.raises(MixedLearningError,match='stale rollout'):
            trainer.ppo_loss(model,torch.device('cpu'))
        assert trainer.cached is None
        a,_,_=trainer.ppo_loss(model,torch.device('cpu'))
        assert a.requires_grad
    finally:
        trainer.close()

class TokenWiseValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(num_labels=1,pad_token_id=0)
        self.proj=nn.Linear(1,12)
        self.score=nn.Linear(12,1)
        self.calls=0
    def forward(self,input_ids,attention_mask=None):
        self.calls+=1
        hidden=self.proj(input_ids.float().unsqueeze(-1)/40)
        return SimpleNamespace(logits=self.score(hidden))

def test_tokenwise_hf_critic_cache_respects_last_unpadded_token_and_gradient():
    options=critic_args();options.critic_hf_loader='token_classification'
    model=TokenWiseValue()
    critic=HFValueCritic(options,model=model,tokenizer=Tok())
    prefixes=['a','abcdef','bc','longer tokenwise prefix']
    expected=critic.values(prefixes,actor_device=torch.device('cpu'))
    calls=model.calls
    assert critic._cached_features is not None
    cached=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert torch.allclose(cached,expected)
    assert model.calls==calls
    cached.sum().backward()
    assert critic.head.weight.grad is not None and critic.head.weight.grad.abs().sum()>0
    for i,text in enumerate(prefixes):
        row=critic.tok(text,return_tensors='pt',truncation=True,max_length=128)
        assert torch.allclose(expected[i],model(**row).logits[0,-1,0])


def test_feature_cache_budget_fails_to_unoptimized_path_before_extra_clones(monkeypatch):
    import patchlib.hf_value_critic_v69 as module
    monkeypatch.setattr(module,'MAX_REPLAY_FEATURE_BYTES',128)
    model=SlowValue()
    critic=HFValueCritic(critic_args(),model=model,tokenizer=Tok())
    prefixes=['one','two','three','four']  # 4 * 16 * 4 = 256 bytes
    a=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert critic._cached_features is None
    calls=model.calls
    b=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert model.calls>calls and torch.allclose(a,b)
