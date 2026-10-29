"""v71 regression/performance correctness: bounded critic batching, Hub cache, PPO logit slices."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest
import torch
from torch import nn

from patchlib.hf_value_critic_v69 import HFValueCritic, HFValueError
from patchlib.hub_discovery_v70 import HubDiscoveryCache, discover_hub_assets
from patchlib.mixed_learning_v65 import _logits, MixedLearningError

SHA='c'*40

class Tokenizer:
    eos_token_id=1
    pad_token_id=0
    def __call__(self, s, *, return_tensors, truncation, max_length):
        tokens=[(ord(c)%29)+2 for c in s][-max_length:]
        return {'input_ids':torch.tensor([tokens]),'attention_mask':torch.ones((1,len(tokens)),dtype=torch.long)}

class SequenceValueModel(nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(num_labels=1,pad_token_id=0)
        self.body=nn.Linear(1,4);self.score=nn.Linear(4,1)
        self.forward_count=0
    def forward(self,input_ids,attention_mask=None):
        self.forward_count+=1
        m=(attention_mask if attention_mask is not None else torch.ones_like(input_ids)).float()
        avg=(input_ids.float()*m).sum(1,keepdim=True)/m.sum(1,keepdim=True)
        return SimpleNamespace(logits=self.score(self.body(avg*.01)).view(-1,1))

class TokenValueModel(SequenceValueModel):
    def forward(self,input_ids,attention_mask=None):
        self.forward_count+=1
        hidden=self.body(input_ids.float().unsqueeze(-1)*.01)
        return SimpleNamespace(logits=self.score(hidden))

def args(loader):
    return SimpleNamespace(critic_kind='hf',critic_hf_repo='Org/critic',critic_hf_revision=SHA,
        critic_hf_loader=loader,critic_hf_device='cpu',critic_hf_max_length=256)

@pytest.mark.parametrize('loader,model_cls',[('sequence_classification',SequenceValueModel),
                                              ('token_classification',TokenValueModel)])
def test_critic_batch_matches_unpadded_reference_and_preserves_grad(loader,model_cls):
    torch.manual_seed(7)
    critic=HFValueCritic(args(loader),model=model_cls(),tokenizer=Tokenizer())
    prefixes=['short','longer string','x'*24,'try again','any input here','hello world',
              'very long '*7,'one','two','abc','different lengths', 'abc'*21,'hello','goodbye','work']
    reference=[]
    for prefix in prefixes:
        inp=critic.tok(prefix,return_tensors='pt',truncation=True,max_length=256)
        logits=critic.model(**inp).logits
        reference.append(logits[0,0] if loader=='sequence_classification' else logits[0,-1,0])
    expected=torch.stack(reference)
    critic.model.forward_count=0
    actual=critic.values(prefixes,actor_device=torch.device('cpu'))
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
    assert critic.model.forward_count< len(prefixes), 'should batch several sequences together'
    actual.square().sum().backward()
    assert critic.head.weight.grad is not None and critic.head.weight.grad.abs().sum()>0
    assert critic.model.body.weight.grad is None


def test_critic_batch_rejects_nonfinite_and_oversized():
    critic=HFValueCritic(args('sequence_classification'),model=SequenceValueModel(),tokenizer=Tokenizer())
    critic.model.forward=lambda **kw: SimpleNamespace(logits=torch.full((kw['input_ids'].shape[0],1),float('nan')))
    with pytest.raises(HFValueError,match='nonfinite'):critic.values(['a','bc'],actor_device=torch.device('cpu'))
    with pytest.raises(HFValueError,match='oversized'):critic.values(['a'*65537],actor_device=torch.device('cpu'))


def test_ppo_logit_selection_checks_only_selected_window_and_preserves_autograd():
    class Model(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1))
        def forward(self,input_ids,attention_mask=None):
            result=self.weight*torch.ones((1,input_ids.shape[1],256))
            result[:,0,0]=float('nan')
            return {'logits':result.to(torch.bfloat16)}
    model=Model(); ids=torch.ones((1,1024),dtype=torch.long)
    select=_logits(model,ids,SimpleNamespace(),start=1020,length=4)
    assert select.shape==(1,4,256) and select.dtype==torch.float32
    select.sum().backward();assert model.weight.grad is not None
    with pytest.raises(MixedLearningError,match='nonfinite'):_logits(model,ids,SimpleNamespace(),start=0,length=1)
    with pytest.raises(MixedLearningError,match='selection'):_logits(model,ids,SimpleNamespace(),start=1024,length=1)


def test_hub_cache_single_flight_expiry_and_copy_isolation():
    tick=[0.]
    cache=HubDiscoveryCache(ttl=2,limit=2,clock=lambda:tick[0])
    calls=[0]; barrier=threading.Barrier(6)
    def load():
        calls[0]+=1
        time.sleep(.04)
        return {'candidates':[{'id':'Org/model','downloads':5}]}
    def task(_):
        barrier.wait()
        return cache.get_or_load(('reward','','recommended'),load)
    with ThreadPoolExecutor(max_workers=6) as ex:
        results=list(ex.map(task,range(6)))
    assert calls==[1]
    results[0]['candidates'][0]['id']='tampered'
    assert results[1]['candidates'][0]['id']=='Org/model'
    assert cache.get_or_load(('reward','','recommended'),load)['candidates'][0]['id']=='Org/model'
    tick[0]=3
    cache.get_or_load(('reward','','recommended'),load)
    assert calls==[2]


def test_hub_cache_network_errors_are_not_stored():
    cache=HubDiscoveryCache();calls=[0]
    def flaky():
        calls[0]+=1
        if calls[0]==1:raise RuntimeError('network unavailable')
        return {'candidates':[]}
    with pytest.raises(RuntimeError):cache.get_or_load(('sft','','recommended'),flaky)
    assert cache.get_or_load(('sft','','recommended'),flaky)=={'candidates':[]}
    assert calls==[2]


def test_cached_discovery_does_not_repeat_identical_hub_requests():
    class API:
        def __init__(self):self.count=0
        def list_models(self,**kw):
            self.count+=1
            return [SimpleNamespace(id='Org/reward-model',sha=SHA,downloads=10,pipeline_tag='text-classification',tags=['reward'])]
    api=API();cache=HubDiscoveryCache()
    a=discover_hub_assets({'role':'reward'},api=api,cache=cache)
    b=discover_hub_assets({'role':'reward'},api=api,cache=cache)
    assert a==b and api.count==3


def test_padless_causal_classifier_falls_back_to_single_sample_for_safety():
    model=SequenceValueModel()
    model.config.pad_token_id=None
    critic=HFValueCritic(args('sequence_classification'),model=model,tokenizer=Tokenizer())
    critic.values(['one','two','three'],actor_device=torch.device('cpu'))
    assert model.forward_count==3

def test_topology_sorted_fastpath_and_mutated_input_never_uses_stale_cache():
    from patchlib.topology import neighbors
    peers=['p%032x'%i for i in range(128)]
    a=neighbors(peers[0],peers)
    assert a==neighbors(peers[0],list(reversed(peers)))
    assert a==neighbors(peers[0],peers+peers)
    peers.append('p%032x'%128)
    assert neighbors(peers[0],peers)==neighbors(peers[0],tuple(peers))
