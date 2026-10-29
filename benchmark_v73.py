"""Local CPU synthetic estimate of PPO Critic reuse savings; no real HF/GPU involved."""
import json, statistics, time
from types import SimpleNamespace
import torch
from torch import nn
from patchlib.hf_value_critic_v69 import HFValueCritic

class Tok:
    pad_token_id=0
    eos_token_id=1
    def __call__(self, text, *, return_tensors, truncation, max_length):
        tokens=[2+ord(char)%40 for char in text][-max_length:]
        return {'input_ids':torch.tensor([tokens]), 'attention_mask':torch.ones(1,len(tokens),dtype=torch.long)}

class FrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(num_labels=1,pad_token_id=0)
        self.backbone=nn.Sequential(nn.Linear(1,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256))
        self.score=nn.Linear(256,1)
        self.calls=0
    def forward(self,input_ids,attention_mask=None):
        self.calls+=1
        weights=attention_mask.float()
        x=((input_ids.float()*weights).sum(dim=1,keepdim=True)/weights.sum(dim=1,keepdim=True))/48
        return SimpleNamespace(logits=self.score(self.backbone(x)))

def run():
    torch.set_num_threads(1); torch.manual_seed(73)
    args=SimpleNamespace(critic_kind='hf',critic_hf_repo='Lab/toy',critic_hf_revision='e'*40,
        critic_hf_loader='sequence_classification',critic_hf_device='cpu',critic_hf_max_length=512)
    critic=HFValueCritic(args,model=FrozenBackbone(),tokenizer=Tok())
    prompts=['synthetic state number '+str(i)+' very long prompt '*4 for i in range(32)]
    # Warm up and establish feature cache once per simulated rollout.
    critic.values(prompts,actor_device=torch.device('cpu'))
    t_cache=[]; t_full=[]
    for _ in range(9):
        t=time.perf_counter()
        critic.values(prompts,actor_device=torch.device('cpu')).sum().backward()
        t_cache.append(time.perf_counter()-t)
        critic.head.zero_grad(set_to_none=True)
        # Reference path performs full frozen-backbone classification each time.
        critic._cached_features=None
        critic._cached_prefix_key=None
        t=time.perf_counter()
        critic.values(prompts,actor_device=torch.device('cpu')).sum().backward()
        t_full.append(time.perf_counter()-t)
        critic.head.zero_grad(set_to_none=True)
    r={'synthetic_prefixes':32,'frozen_backbone_calls_per_full_evaluation':critic.model.calls//10,
       'cached_head_only_seconds_median':round(statistics.median(t_cache),6),
       'full_backbone_seconds_median':round(statistics.median(t_full),6),
       'cache_feature_bytes':critic._cached_features.numel()*critic._cached_features.element_size(),
       'scope':'CPU synthetic; non-HF, not representative of real model throughput'}
    print(json.dumps(r,indent=2))
    with open('BENCHMARK_V73.json','w') as f: json.dump(r,f,indent=2)

if __name__=='__main__':run()
