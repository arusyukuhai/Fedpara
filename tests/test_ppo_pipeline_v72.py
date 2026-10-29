"""Safety/performance regressions for opt-in PPO actor cache and auxiliary CPU offload."""
import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from patchlib.ppo_pipeline_v72 import PrefetchedRows
from patchlib.reward_model_v67 import HFRewardScorer
from patchlib.mix_gui_v66 import DEFAULT,normalize,command_options


class SmallTokenizer:
    eos_token_id=None
    def encode(self,s,add_special_tokens=False):return [2+(ord(ch)%10) for ch in s]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(65+int(v)%26) for v in ids)


class CacheLM(nn.Module):
    """Deterministic causal LM, supports incremental cache with exact full-prefix parity."""
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(20,8)
        self.head=nn.Linear(8,20)
        self.lengths=[]
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
        self.lengths.append(input_ids.shape[1])
        tokens=(torch.cat((past_key_values,input_ids),dim=1)
                if past_key_values is not None else input_ids)
        hidden=self.emb(tokens).cumsum(dim=1)
        return {'logits':self.head(hidden[:,-input_ids.shape[1]:]),
                'past_key_values':tokens.detach() if use_cache else None}


def args(tmp_path,**kw):
    path=tmp_path/'ppo.jsonl'
    path.write_text(json.dumps({'prompt':'good','reference':'right'})+'\n')
    d=dict(ppo_jsonl=path,sft_jsonl=None,ppo_reward='token_match',
           max_length=36,ppo_max_tokens=6,ppo_reuse=2,
           ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
           ppo_gae_lambda=.95,ppo_kv_cache='auto',pt_weight=0.,
           sft_weight=0.,ppo_weight=1.,hf_task='causal-lm',loss_policy='auto',
           critic_kind='mlp',critic_hidden=12,critic_lr=.003,model='unit')
    d.update(kw)
    return SimpleNamespace(**d)


def test_kv_cache_old_probs_match_full_recompute(tmp_path):
    torch.manual_seed(333)
    cache_model=CacheLM()
    plain_model=CacheLM();plain_model.load_state_dict(cache_model.state_dict())
    cache=MixedObjectiveTrainer(args(tmp_path),SmallTokenizer())
    plain=MixedObjectiveTrainer(args(tmp_path,ppo_kv_cache='off'),SmallTokenizer())
    try:
        torch.manual_seed(902)
        loss1,critic1,_=cache.ppo_loss(cache_model,torch.device('cpu'))
        torch.manual_seed(902)
        loss2,critic2,_=plain.ppo_loss(plain_model,torch.device('cpu'))
        assert cache.cached['tokens'].tolist()==plain.cached['tokens'].tolist()
        assert torch.allclose(cache.cached['old_lp'],plain.cached['old_lp'],atol=1e-6)
        assert torch.allclose(loss1,loss2,atol=1e-6)
        assert cache_model.lengths[0]==4  # actor prefill + later 1-token steps
        assert cache_model.lengths[1:len(cache.cached['tokens'])]==[1]*(len(cache.cached['tokens'])-1)
        assert any(n>1 for n in plain_model.lengths[1:len(plain.cached['tokens'])])
        (loss1+critic1).backward()
        assert cache_model.head.weight.grad is not None
        assert torch.isfinite(cache_model.head.weight.grad).all()
        # The PPO actor optimization is always a full differentiable forward.
        assert cache_model.lengths[-1]==cache.cached['ids'].shape[1]
    finally:
        cache.close();plain.close()


def test_kv_cache_falls_back_if_model_never_returns_cache(tmp_path):
    class NoCache(CacheLM):
        def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
            result=super().forward(input_ids,attention_mask,past_key_values,use_cache=False)
            result['past_key_values']=None
            return result
    m=NoCache();tr=MixedObjectiveTrainer(args(tmp_path),SmallTokenizer())
    try:
        tr.ppo_loss(m,torch.device('cpu'))
        assert m.lengths[0]==4
        assert m.lengths[1]>1
    finally:tr.close()


def test_hub_prefetch_delivers_in_order_and_propagates_error():
    class Source:
        def __init__(self):self.n=0;self.closed=False
        def next(self):
            self.n+=1
            if self.n==4:raise RuntimeError('Hub connection failed')
            return {'prompt':str(self.n)}
        def close(self):self.closed=True
    src=Source();prefetch=PrefetchedRows(src,capacity=2)
    assert [prefetch.next()['prompt'] for _ in range(3)]==['1','2','3']
    with pytest.raises(RuntimeError,match='Hub connection failed'):prefetch.next()
    assert prefetch.closed
    prefetch.worker.join(timeout=1)
    assert src.closed


def test_prefetch_close_does_not_wait_on_hung_network():
    event=threading.Event();started=threading.Event()
    class Source:
        def next(self):started.set();event.wait(5);return {'prompt':'late'}
        def close(self):pass
    r=PrefetchedRows(Source())
    try:
        assert started.wait(1)
        begin=time.perf_counter();r.close();assert time.perf_counter()-begin<.4
    finally:
        event.set();r.worker.join(timeout=1)


def test_reward_idle_cpu_offload_returns_after_failure(monkeypatch):
    # Device travel can be tested without a GPU: a frozen module records .to().
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(1));self.config=SimpleNamespace(num_labels=1)
            self.moves=[]
        def to(self,device):self.moves.append(str(device));return self
    config=SimpleNamespace(ppo_reward='hf_model',reward_model_id='Org/test',
          reward_model_revision='a'*40,reward_model_loader='sequence_classification',
          reward_model_output='logits',reward_model_index=0,reward_model_scale=1.,
          reward_model_format='plain',reward_model_max_length=128,
          reward_model_device='cuda',reward_model_idle_offload=True)
    model=FakeModel()
    r=HFRewardScorer(config,model=model,tokenizer=object())
    assert model.moves==['cpu']
    monkeypatch.setattr(r,'_score_active',lambda a,b:2.5)
    assert r.score('foo','bar')==2.5
    assert model.moves[-2:]==['cuda','cpu']
    def fail(a,b):raise ValueError('bad score')
    monkeypatch.setattr(r,'_score_active',fail)
    with pytest.raises(ValueError,match='bad score'):r.score('foo','bar')
    assert model.moves[-1]=='cpu'


def test_gui_offload_local_only_and_cli(tmp_path):
    args={**DEFAULT,'enabled':True,'sft_weight':0.,'ppo_weight':1.,
          'ppo_jsonl':str(tmp_path/'rows.jsonl'),
          'ppo_reward':'hf_model','reward_model_id':'Org/reward',
          'reward_model_revision':'a'*40,'reward_model_idle_offload':True}
    (tmp_path/'rows.jsonl').write_text('{"prompt":"hello"}\n')
    v=normalize(args,task='causal-lm')
    cli=command_options(v,state_dir=tmp_path/'critic')
    assert '--reward-model-idle-offload' in cli
    from patchlib.mix_gui_v66 import objective_policy
    assert 'reward_model_idle_offload' not in json.dumps(objective_policy(v))
    with pytest.raises(ValueError,match='boolean'):
        normalize({**args,'reward_model_idle_offload':'yes'},task='causal-lm')


def test_hub_prefetch_timeout_fails_closed():
    event=threading.Event();started=threading.Event()
    class Stalled:
        def next(self):started.set();event.wait(1);return {'prompt':'eventually'}
        def close(self):pass
    data=PrefetchedRows(Stalled(),read_timeout=.03)
    try:
        assert started.wait(.5)
        with pytest.raises(TimeoutError,match='Hub dataset prefetch stalled'):
            data.next()
        assert data.closed
    finally:
        event.set();data.worker.join(timeout=1)
