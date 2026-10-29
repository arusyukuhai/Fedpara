"""Local synthetic benchmark: v70 one-forward-per-prefix vs v71 bounded batching."""
import json
import statistics
import time
from types import SimpleNamespace
import torch
from torch import nn
from patchlib.hf_value_critic_v69 import HFValueCritic

class T:
    pad_token_id=0;eos_token_id=1
    def __call__(self,text,*,return_tensors,truncation,max_length):
        ids=torch.tensor([[3+(ord(ch)%100) for ch in text][-max_length:]],dtype=torch.long)
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}

class Net(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(num_labels=1,pad_token_id=0)
        self.body=nn.Sequential(nn.Linear(1,128),nn.Tanh(),nn.Linear(128,128))
        self.score=nn.Linear(128,1)
        self.calls=0
    def forward(self,input_ids,attention_mask=None):
        self.calls+=1
        m=attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids).float()
        means=((input_ids.float()*m).sum(1,keepdim=True)/m.sum(1,keepdim=True))*.01
        return SimpleNamespace(logits=self.score(self.body(means)))

def run():
    torch.manual_seed(14);torch.set_num_threads(1)
    args=SimpleNamespace(critic_kind='hf',critic_hf_repo='Org/model',critic_hf_revision='a'*40,
        critic_hf_loader='sequence_classification',critic_hf_device='cpu',critic_hf_max_length=256)
    model=Net();critic=HFValueCritic(args,model=model,tokenizer=T())
    # Variable-length strings mimic per-token PPO prefixes.
    prefixes=['The quick brown fox '+str(i)+' / '+('token '*i) for i in range(1,33)]
    def serial():
        result=[]
        for prefix in prefixes:
            inp=critic.tok(prefix,return_tensors='pt',truncation=True,max_length=256)
            result.append(model(**inp).logits[0,0])
        return torch.stack(result)
    def batched():return critic.values(prefixes,actor_device=torch.device('cpu'))
    old=serial();new=batched()
    assert torch.allclose(old,new,rtol=1e-5,atol=1e-6)
    def time_op(f):
        measures=[]
        for _ in range(30):
            t=time.perf_counter();f();measures.append((time.perf_counter()-t)*1000)
        return round(statistics.median(measures),4)
    model.calls=0
    old_ms=time_op(serial);old_calls=model.calls//30
    model.calls=0
    new_ms=time_op(batched);new_calls=model.calls//30
    return dict(platform='CPU synthetic 32 value prefixes; scalar head+128 hidden, no real HF model',
        serial_ms=old_ms,serial_forwards=old_calls,batch_ms=new_ms,batch_forwards=new_calls,
        speedup=round(old_ms/new_ms,2),repeats=30,
        notes='Synthetic batch-forward benchmark only; no real GPU or Hub weights')

if __name__=='__main__':
    result=run();print(json.dumps(result,indent=2));open('BENCHMARK_V71.json','w').write(json.dumps(result,indent=2)+'\n')
