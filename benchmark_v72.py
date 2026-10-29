"""Reproducible CPU smoke measurements, not a WAN/GPU benchmark."""
from __future__ import annotations
import json, statistics, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from patchlib.ppo_pipeline_v72 import PrefetchedRows

class Tok:
    eos_token_id=None
    def encode(self,s,add_special_tokens=False):return [2+(ord(x)%61) for x in s]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(65+int(v)%26) for v in ids)

class Causal(nn.Module):
    def __init__(self):
        super().__init__();self.e=nn.Embedding(72,64);self.h=nn.Linear(64,72);self.n=0;self.tokens=0
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
        self.n+=1;self.tokens+=input_ids.numel()
        prefix=torch.cat([past_key_values,input_ids],dim=1) if past_key_values is not None else input_ids
        x=self.e(prefix).cumsum(dim=1)
        return {'logits':self.h(x[:,-input_ids.shape[1]:]),'past_key_values':prefix if use_cache else None}

def run(root,cached):
    p=root/'data.jsonl';p.write_text('{"prompt":"longer prompt data", "reference":"answer"}\n')
    args=SimpleNamespace(ppo_jsonl=p,sft_jsonl=None,ppo_reward='token_match',
        max_length=128,ppo_max_tokens=24,ppo_reuse=2,ppo_clip=.2,ppo_kl_coef=.02,
        ppo_value_coef=.5,ppo_gamma=.99,ppo_gae_lambda=.95,ppo_kv_cache='auto' if cached else 'off',
        pt_weight=0.,ppo_weight=1.,sft_weight=0.,hf_task='causal-lm',loss_policy='auto',
        critic_kind='mlp',critic_hidden=32,critic_lr=1e-3)
    m=Causal();tr=MixedObjectiveTrainer(args,Tok())
    torch.manual_seed(5)
    before=time.perf_counter();tr._new_rollout(m,torch.device('cpu'))
    seconds=time.perf_counter()-before
    report={'seconds':seconds,'actor_forward_calls':m.n,'forward_input_tokens':m.tokens}
    tr.close();return report

class Slow:
    def __init__(self,n):self.n=n;self.idx=0
    def next(self):
        time.sleep(.02)
        self.idx+=1;return {'prompt':str(self.idx)}
    def close(self):pass

def rows(prefetch):
    source=Slow(10)
    obj=PrefetchedRows(source) if prefetch else source
    start=time.perf_counter()
    for _ in range(10):
        obj.next();time.sleep(.02)
    elapsed=time.perf_counter()-start
    obj.close();return elapsed

if __name__=='__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        r=Path(d)
        results={}
        for setting in ('no_cache','cache'):
            values=[run(r,setting=='cache') for _ in range(5)]
            results[setting]={'seconds_median':statistics.median(x['seconds'] for x in values),
                              'actor_forward_calls':values[-1]['actor_forward_calls'],
                              'forward_input_tokens':values[-1]['forward_input_tokens']}
        results['sequential_hub_rows_seconds']=rows(False)
        results['prefetch_hub_rows_seconds']=rows(True)
        Path(__file__).with_name('BENCHMARK_V72.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))
