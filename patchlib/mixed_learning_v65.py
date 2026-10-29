"""Opt-in concurrent-objective PT + response-masked SFT + on-policy clipped PPO.

Each local optimization microstep evaluates independent objectives against the
same current adapter.  P2P continues to distribute ONLY adapter updates;
the small auxiliary value estimator is local and never a signed parameter.
This deliberately uses *verifiable reference text* as the PPO reward source,
not the model's own self-judgment.
"""
from __future__ import annotations

import hashlib
import json
import math
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model_contract_v64 import invoke_model, training_loss, _loss_or_logits
from .hf_mixed_datasets_v68 import source_selected, source_spec, HFRowCycle
from .ppo_pipeline_v72 import PrefetchedRows
from .offpolicy_critic_v79 import resolved_mode, exact_trajectory_value_surrogate, OffPolicyCorrectionError


class MixedLearningError(ValueError):
    pass


def enabled(args):
    return source_selected(args,'sft') or source_selected(args,'ppo')


def spec(args):
    """No local dataset paths in genesis: peers can use distinct data shards."""
    if not enabled(args) and not getattr(args,'mix_policy',None):
        return None
    generated={'version':'pt-sft-clipped-ppo-v65',
            'objectives':[source_selected(args,'sft'),
                          source_selected(args,'ppo')],
            'weights':[float(getattr(args,'pt_weight',1.)),
                       float(getattr(args,'sft_weight',1.)),
                       float(getattr(args,'ppo_weight',.2))],
            'ppo_clip':float(getattr(args,'ppo_clip',.2)),
            'ppo_kl_coef':float(getattr(args,'ppo_kl_coef',.02)),
            'ppo_value_coef':float(getattr(args,'ppo_value_coef',.5)),
            'ppo_gamma':float(getattr(args,'ppo_gamma',.99)),
            'ppo_gae_lambda':float(getattr(args,'ppo_gae_lambda',.95)),
            'ppo_max_tokens':int(getattr(args,'ppo_max_tokens',8)),
            'ppo_reuse':int(getattr(args,'ppo_reuse',2)),
            'ppo_reward':str(getattr(args,'ppo_reward','token_match')),
            'input_map':str(getattr(args,'input_map','')),
            'reward_source':'ground-truth-reference-v1',
            'task':'causal-lm'}
    if getattr(args,'ppo_reward','token_match')=='hf_model':
        from .reward_model_v67 import reward_spec
        generated['version']='pt-sft-clipped-ppo-v67'
        generated['reward_source']='external-hf-sequence-scorer-v1'
        generated['reward_model']=reward_spec(args)
    if (getattr(args,'critic_kind','mlp')!='mlp' or
            int(getattr(args,'critic_hidden',32))!=32 or
            float(getattr(args,'critic_lr',1e-3))!=.001):
        if generated['version']!='pt-sft-clipped-ppo-v67':
            generated['version']='pt-sft-clipped-ppo-v66'
        generated['critic']={'kind':getattr(args,'critic_kind','mlp'),
                             'hidden':int(getattr(args,'critic_hidden',32)),
                             'lr':float(getattr(args,'critic_lr',1e-3))}
        if getattr(args,'critic_kind','mlp')=='hf':
            from .hf_value_critic_v69 import critic_spec
            generated['version']='pt-sft-clipped-ppo-v69'
            generated['critic']['hf']=critic_spec(args)
    if getattr(args,'distributed_critic',False):
        generated['critic_distribution']='signed-root-critic-gossip-v78'
        generated['version']='pt-sft-clipped-ppo-v78'
    if resolved_mode(args)=='trajectory_is':
        generated['critic_offpolicy']='unclipped-trajectory-is-mc-v79'
        generated['version']='pt-sft-clipped-ppo-v79'
    if getattr(args,'critic_rollout_gossip',False):
        generated['critic_distribution']='loha-complete-checkpoint-verified-trajectory-is-v82'
        generated['behavior_policy_proof']='content-addressed-loha-anchors-remotes-live-factors-frozen-base-sha256-v82'
        generated['version']='pt-sft-clipped-ppo-v82'
    if not getattr(args,'mix_policy',None):
        return generated
    path=Path(args.mix_policy)
    if not path.is_file() or path.stat().st_size>8192:
        raise MixedLearningError('--mix-policy must be a JSON file <=8KiB')
    try:
        policy=json.loads(path.read_text(encoding='utf8'))
    except (OSError,ValueError) as exc:
        raise MixedLearningError('--mix-policy is not readable JSON') from exc
    if not isinstance(policy,dict) or set(policy)!=set(generated):
        raise MixedLearningError('--mix-policy needs the exact v65 protocol fields')
    # All numerical/scoring settings are immutable room-wide. Per-peer dataset
    # selection is the ONLY permitted difference and is independently checked.
    if any(policy[k]!=generated[k] for k in generated if k!='objectives'):
        raise MixedLearningError('local objective hyperparameters disagree with --mix-policy')
    flags=policy.get('objectives')
    if not isinstance(flags,list) or len(flags)!=2 or any(type(f) is not bool for f in flags):
        raise MixedLearningError('--mix-policy objectives must be [SFT enabled, PPO enabled] booleans')
    if any(local and not agreed for local,agreed in zip(generated['objectives'],flags)):
        raise MixedLearningError('local dataset is not enabled in the shared --mix-policy')
    if not any(flags) or generated['weights'][0]+sum(w for active,w in zip(flags,generated['weights'][1:]) if active)<=0:
        raise MixedLearningError('--mix-policy contains no active mixed learning objectives')
    return policy


def validate_args(args):
    if not enabled(args) and not getattr(args,'mix_policy',None):
        return
    try:
        resolved_mode(args)
    except OffPolicyCorrectionError as exc:
        raise MixedLearningError(str(exc)) from exc
    if getattr(args,'critic_rollout_gossip',False):
        if not getattr(args,'distributed_critic',False):
            raise MixedLearningError('trajectory gossip requires --distributed-critic')
        if getattr(args,'ppo_reward','token_match') not in ('exact','token_match'):
            raise MixedLearningError('v81 trajectory relay requires receiver-verifiable reference rewards; HF reward not supported')
        if resolved_mode(args)!='trajectory_is':
            raise MixedLearningError('v81 distributed trajectory replay requires un-clipped trajectory_is')
        if int(getattr(args,'max_length',1024))>4096:
            raise MixedLearningError('v81 signed trajectory frame requires max_length <= 4096')
    if getattr(args,'distributed_critic',False):
        if not source_selected(args,'ppo') or float(getattr(args,'ppo_weight',.2))<=0:
            raise MixedLearningError('distributed Critic v78 currently needs a PPO dataset on every Critic participant')
        if getattr(args,'critic_init',None):
            raise MixedLearningError('distributed Critic requires a common pinned initial head; --critic-init is local-only')
        if getattr(args,'unsafe_unsigned',False):
            raise MixedLearningError('distributed Critic requires signed authenticated peers')
    if getattr(args,'ppo_kv_cache','auto') not in ('auto','off'):
        raise MixedLearningError('ppo_kv_cache must be auto or off')
    if getattr(args,'hf_task','causal-lm') != 'causal-lm':
        raise MixedLearningError('PT/SFT/PPO mixture currently requires --hf-task causal-lm')
    if getattr(args,'loss_policy','auto') not in ('auto','causal'):
        raise MixedLearningError('mixed causal-LM requires --loss-policy auto or causal')
    for field in ('pt_weight','sft_weight','ppo_weight'):
        val=float(getattr(args,field,0))
        if not math.isfinite(val) or val<0 or val>100:
            raise MixedLearningError(f'{field} must be finite and in [0,100]')
    if getattr(args,'ppo_reward','token_match') not in ('token_match','exact','hf_model'):
        raise MixedLearningError('PPO reward must be token_match, exact or hf_model')
    if getattr(args,'ppo_reward','token_match')=='hf_model':
        from .reward_model_v67 import reward_spec
        reward_spec(args)
        if getattr(args,'reward_model_device','cpu') not in ('cpu','cuda','mps'):
            raise MixedLearningError('Invalid reward model device')
    if getattr(args,'critic_kind','mlp') not in ('mlp','linear','hf'):
        raise MixedLearningError('critic_kind must be mlp, linear or hf')
    if getattr(args,'critic_kind','mlp')=='hf':
        from .hf_value_critic_v69 import critic_spec
        critic_spec(args)
    if type(getattr(args,'critic_hidden',32)) is not int or not 8<=getattr(args,'critic_hidden',32)<=512:
        raise MixedLearningError('critic_hidden must be 8..512')
    value_lr=float(getattr(args,'critic_lr',1e-3))
    if not math.isfinite(value_lr) or not 1e-7<=value_lr<=.1:
        raise MixedLearningError('critic_lr out of range')
    if not 0<float(args.ppo_clip)<=.5:
        raise MixedLearningError('ppo_clip must be in (0,.5]')
    for name in ('ppo_kl_coef','ppo_value_coef'):
        val=float(getattr(args,name))
        if not math.isfinite(val) or not 0<=val<=10:
            raise MixedLearningError(f'{name} must be in [0,10]')
    for name in ('ppo_gamma','ppo_gae_lambda'):
        val=float(getattr(args,name))
        if not 0<val<=1:
            raise MixedLearningError(f'{name} must be in (0,1]')
    if not 1<=int(args.ppo_max_tokens)<=128 or not 1<=int(args.ppo_reuse)<=8:
        raise MixedLearningError('PPO token cap must be 1..128, reuse 1..8')
    if not 8<=int(getattr(args,'max_length',128))<=131072:
        raise MixedLearningError('max_length must be >= 8 for mixed causal-LM')
    if not getattr(args,'mix_policy',None) and not source_selected(args,'sft') and getattr(args,'sft_weight',1.) not in (0.,1.):
        raise MixedLearningError('SFT weight supplied without --sft-jsonl')
    if not getattr(args,'mix_policy',None) and not source_selected(args,'ppo') and getattr(args,'ppo_weight',.2) not in (0.,.2):
        raise MixedLearningError('PPO weight supplied without --ppo-jsonl')
    if float(args.pt_weight)+ (float(args.sft_weight) if source_selected(args,'sft') else 0) + (float(args.ppo_weight) if source_selected(args,'ppo') else 0) == 0:
        raise MixedLearningError('at least one mixed objective must have positive weight')
    for name in ('sft_jsonl','ppo_jsonl'):
        path=getattr(args,name,None)
        if path and (not Path(path).is_file() or Path(path).stat().st_size>4*1024**3):
            raise MixedLearningError(f'{name} must be a readable local JSONL <= 4GiB')
    source_spec(args,'sft');source_spec(args,'ppo')
    spec(args)  # validate optional shared room policy before any network join


class JsonlCycle:
    """Streaming fixed-size lines; never keeps a multi-GB corpus in RAM."""
    def __init__(self, path):
        self.path=Path(path)
        self.file=self.path.open('r',encoding='utf8')
        self.line_number=0
        self.file.seek(0,2)
        if self.file.tell()==0:
            raise MixedLearningError(f'{path}: empty dataset')
        self.file.seek(0)

    def next(self):
        attempts=0
        while True:
            line=self.file.readline(65537)
            if not line:
                if attempts:
                    raise MixedLearningError(f'{self.path}: no nonempty JSONL rows')
                self.file.seek(0);self.line_number=0;attempts+=1;continue
            self.line_number+=1
            if len(line)>65536 or (line and not line.endswith('\n') and not self.file.tell()==self.path.stat().st_size):
                raise MixedLearningError(f'{self.path}:{self.line_number} JSONL line too large')
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except ValueError as exc:
                raise MixedLearningError(f'{self.path}:{self.line_number} invalid JSON') from exc
            if not isinstance(row,dict):
                raise MixedLearningError('training row must be a JSON object')
            return row

    def close(self):
        self.file.close()


def _tokens(tokenizer, text):
    if not isinstance(text,str) or not text or len(text)>16384:
        raise MixedLearningError('prompt/response/reference must be nonempty UTF-8 text <=16Ki chars')
    ids=tokenizer.encode(text, add_special_tokens=False)
    if not ids or not all(isinstance(i,int) and i>=0 for i in ids):
        raise MixedLearningError('tokenizer returned no valid token IDs')
    return list(ids)


def _logits(model, ids, args, *, start=None, length=None):
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    outputs=invoke_model(model,batch,input_map=getattr(args,'input_map',None))
    _,logits=_loss_or_logits(outputs)
    if not isinstance(logits,torch.Tensor) or logits.ndim!=3 or logits.shape[:2]!=ids.shape:
        raise MixedLearningError('PPO requires causal logits [1, tokens, vocab]')
    # Slice BEFORE FP32 promotion and finiteness scan. For long contexts,
    # promoting [batch,context,vocab] can temporarily use gigabytes even
    # though PPO needs only the generated token window.
    if start is not None:
        if not isinstance(start,int) or start<0 or not isinstance(length,int) or length<1 or start+length>ids.shape[1]:
            raise MixedLearningError('invalid PPO logits selection')
        logits=logits[:,start:start+length,:]
    if not torch.isfinite(logits).all():
        raise MixedLearningError('nonfinite PPO logits')
    return logits.float()


def _cached_step_logits(model, ids, args, cache_state):
    """Optional causal KV-cache fast path; never changes the PPO update graph.

    A generic PyTorch LM may not implement caching. Cache activation requires
    an explicit forward contract, a real returned cache and no custom input
    remapping. All logits used for actor gradients are still recomputed without
    cache during ppo_loss. We never share caches across policy updates.
    """
    if cache_state['active'] is False:
        return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
    if cache_state['active'] is None:
        if getattr(args,'ppo_kv_cache','auto')=='off' or getattr(args,'input_map',''):
            cache_state['active']=False
            return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
        try:
            params=inspect.signature(model.forward).parameters
        except (ValueError,TypeError):
            params={}
        # Explicit cache ABI, never guess from **kwargs alone.
        if not {'past_key_values','use_cache'}.issubset(params):
            cache_state['active']=False
            return _logits(model, ids, args, start=ids.shape[1]-1, length=1)
        data={'input_ids':ids,'attention_mask':torch.ones_like(ids),'use_cache':True}
    else:
        # Hugging Face causal LMs generally expect the *full prefix* attention
        # mask even though only the newest token is passed as input_ids. A
        # one-token mask can corrupt RoPE/SDPA attention or fail on decode.
        if ids.shape[1] != cache_state.get('last_len', -1)+1:
            raise MixedLearningError('PPO KV cache prefix length changed unexpectedly')
        data={'input_ids':ids[:,-1:],
              'attention_mask':torch.ones_like(ids),
              'past_key_values':cache_state['past'], 'use_cache':True}
    result=model(**data)
    _,logits=_loss_or_logits(result)
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=3 or
            logits.shape[0]!=1 or logits.shape[-1]<2 or logits.shape[1]!=data['input_ids'].shape[1]):
        raise MixedLearningError('PPO cached actor returned invalid logits')
    past=(result.get('past_key_values') if isinstance(result,dict)
          else getattr(result,'past_key_values',None))
    if past is None:
        if cache_state['active']:
            raise MixedLearningError('PPO actor stopped returning a live KV cache')
        cache_state['active']=False
    else:
        cache_state['active']=True
        cache_state['past']=past
        cache_state['last_len']=ids.shape[1]
    last=logits[:,-1:,:]
    if not bool(torch.isfinite(last).all()):
        raise MixedLearningError('nonfinite PPO generation logits')
    return last.float()


def _features(logits, *, offset):
    """Small bounded, detached critic features: no unbounded vocab-size head."""
    detached=logits.detach().float()
    logp=F.log_softmax(detached,dim=-1)
    p=logp.exp()
    entropy=-(p*logp).sum(-1)
    top2=detached.topk(k=min(2,detached.shape[-1]),dim=-1).values
    margin=(top2[...,0]-top2[...,-1]).clamp(0,30)
    normalized=-(logp.max(-1).values).clamp(0,30)
    steps=torch.arange(offset,offset+detached.size(1),device=detached.device,dtype=torch.float32)
    progress=(steps/512).clamp(0,1).unsqueeze(0).expand_as(entropy)
    return torch.stack((entropy.clamp(0,30)/30,margin/30,normalized/30,progress),dim=-1)


class LocalValueHead(nn.Module):
    def __init__(self, kind='mlp', hidden=32):
        super().__init__()
        if kind not in ('mlp','linear') or type(hidden) is not int or not 8 <= hidden <= 512:
            raise MixedLearningError('Critic architecture must be mlp/linear with 8..512 hidden units')
        self.net=(nn.Sequential(nn.Linear(4,hidden),nn.Tanh(),nn.Linear(hidden,1))
                  if kind=='mlp' else nn.Sequential(nn.Linear(4,1)))
    def forward(self,x):
        return self.net(x).squeeze(-1)


class MixedObjectiveTrainer:
    def __init__(self,args,tokenizer):
        # A HF reward/critic checkpoint may fail to load *after* dataset
        # prefetch workers have started. Release partial construction eagerly.
        self.sft=None
        self.ppo=None
        self.reward_scorer=None
        try:
            self._initialize(args,tokenizer)
        except BaseException:
            for item in (self.sft,self.ppo):
                if item is not None:
                    try:item.close()
                    except Exception:pass
            if self.reward_scorer is not None:
                try:self.reward_scorer.close()
                except Exception:pass
            raise

    def _initialize(self,args,tokenizer):
        validate_args(args)
        self.args=args
        self.tok=tokenizer
        self._closed=False
        self.sft=(PrefetchedRows(HFRowCycle(source_spec(args,'sft'),'sft')) if getattr(args,'sft_hf_repo','') else
                  JsonlCycle(args.sft_jsonl) if getattr(args,'sft_jsonl',None) else None)
        self.ppo=(PrefetchedRows(HFRowCycle(source_spec(args,'ppo'),'ppo')) if getattr(args,'ppo_hf_repo','') else
                  JsonlCycle(args.ppo_jsonl) if getattr(args,'ppo_jsonl',None) else None)
        self.reward_scorer=None
        if self.ppo and getattr(args,'ppo_reward','token_match')=='hf_model':
            from .reward_model_v67 import HFRewardScorer
            self.reward_scorer=HFRewardScorer(args)
        self.critic=None
        self.critic_optim=None
        self.cached=None
        self.reuses=0
        self.counts={'pt':0,'sft':0,'ppo':0}
        self.critic_steps=0
        if self.ppo:
            kind=getattr(args,'critic_kind','mlp')
            hidden=int(getattr(args,'critic_hidden',32))
            if kind=='hf':
                from .hf_value_critic_v69 import HFValueCritic
                self.critic=HFValueCritic(args)
                critic_params=self.critic.trainable_parameters()
            else:
                self.critic=LocalValueHead(kind,hidden)
                critic_params=list(self.critic.parameters())
            self.critic_optim=torch.optim.AdamW(critic_params,lr=float(getattr(args,'critic_lr',1e-3)))
            self._critic_dir=getattr(args,'critic_state_dir',None)
            self._critic_identity=str(getattr(args,'room_id',None) or getattr(args,'model','local'))
            if kind=='hf':
                from .hf_critic_state_v69 import snapshot_file, load
                if getattr(args,'critic_init',None):
                    raise MixedLearningError('HF Critic init is the pinned Hub model, not a local v66 checkpoint')
                if (not getattr(args,'_v75_skip_legacy_critic',False) and
                        self._critic_dir and snapshot_file(self._critic_dir).is_file()):
                    self.critic_steps=load(snapshot_file(self._critic_dir),self.critic,self.critic_optim,identity=self._critic_identity)
            else:
                from .critic_state_v66 import snapshot_file, load
                if (not getattr(args,'_v75_skip_legacy_critic',False) and
                        self._critic_dir and snapshot_file(self._critic_dir).is_file()):
                    self.critic_steps=load(snapshot_file(self._critic_dir),self.critic,self.critic_optim,
                                           kind=kind,hidden=hidden,identity=self._critic_identity)
                elif getattr(args,'critic_init',None):
                    load(args.critic_init,self.critic,self.critic_optim,kind=kind,hidden=hidden,
                         identity=self._critic_identity,allow_initial=True)

    def sft_loss(self,model,device):
        row=self.sft.next()
        if not isinstance(row.get('prompt'),str) or not isinstance(row.get('response'),str):
            raise MixedLearningError('SFT row requires prompt and response strings')
        prompt=_tokens(self.tok,row['prompt']);response=_tokens(self.tok,row['response'])
        eos=getattr(self.tok,'eos_token_id',None)
        if isinstance(eos,int) and eos>=0:
            response=response+[eos]
        if len(prompt)+len(response)>self.args.max_length:
            # Keep the newest prompt context but NEVER silently truncate the response.
            if len(response)>=self.args.max_length:
                raise MixedLearningError('SFT response exceeds max_length; would lose supervision')
            prompt=prompt[-(self.args.max_length-len(response)):]
        ids=torch.tensor([prompt+response],device=device,dtype=torch.long)
        labels=ids.clone();labels[:,:len(prompt)]=-100
        batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
        outputs=invoke_model(model,batch,input_map=getattr(self.args,'input_map',None))
        loss=training_loss(outputs,batch,task='causal-lm',policy='causal')
        self.counts['sft']+=1
        return loss

    def _new_rollout(self,model,device):
        row=self.ppo.next()
        is_hf=self.reward_scorer is not None
        if not isinstance(row.get('prompt'),str) or (not is_hf and not isinstance(row.get('reference'),str)):
            raise MixedLearningError('PPO row requires prompt and, for ground-truth reward, reference strings')
        prompt=_tokens(self.tok,row['prompt'])
        reference=(_tokens(self.tok,row['reference']) if not is_hf else None)
        if len(prompt)+2>self.args.max_length:
            prompt=prompt[-(self.args.max_length-2):]
        max_new=min(self.args.ppo_max_tokens,self.args.max_length-len(prompt))
        if max_new<=0:
            raise MixedLearningError('no room for PPO generation')
        # Snapshot the EXACT old Actor before its first sampling forward.
        # run_step holds adapter.lock, so no concurrent P2P/optimizer write
        # can change a tensor between this capture and rollout generation.
        if getattr(self.args,'critic_rollout_gossip',False):
            from .behavior_checkpoint_v82 import capture
            checkpoint=capture(model)
        else:
            checkpoint=None
        was_training=model.training
        model.eval()
        try:
            with torch.no_grad():
                # Avoid reallocating the complete token prefix on every
                # generation step (quadratic tensor copies for long PPO replies).
                ids_buffer=torch.empty((1,len(prompt)+max_new),dtype=torch.long,device=device)
                ids_buffer[0,:len(prompt)]=torch.tensor(prompt,dtype=torch.long,device=device)
                generated_length=len(prompt)
                chosen=[];old_lp=[];features=[]
                cache_state={'active':None,'past':None}
                eos=getattr(self.tok,'eos_token_id',None)
                for _ in range(max_new):
                    final_logits=_cached_step_logits(model,ids_buffer[:,:generated_length],self.args,cache_state)
                    lp=F.log_softmax(final_logits[:,0],dim=-1)
                    sample=torch.multinomial(lp.exp(),1)
                    token=int(sample.item());chosen.append(token)
                    old_lp.append(lp[0,token])
                    if getattr(self.args,'critic_kind','mlp')!='hf':
                        features.append(_features(final_logits,offset=generated_length-1)[0,0])
                    ids_buffer[0,generated_length]=token
                    generated_length+=1
                    if eos is not None and token==int(eos):
                        break
                if not chosen:
                    raise MixedLearningError('PPO generation returned no tokens')
                feat=(torch.stack(features) if features else None)
                if getattr(self.args,'critic_kind','mlp')!='hf':
                    self.critic.to(device)
                # AdamW moments restored from safe CPU checkpoint must follow
                # their parameter tensors when an accelerator is selected.
                for param,state in self.critic_optim.state.items():
                    for key in ('exp_avg','exp_avg_sq'):
                        if key in state:state[key]=state[key].to(param.device)
                self.critic_optim.zero_grad(set_to_none=True)
                prefixes=None
                if getattr(self.args,'critic_kind','mlp')=='hf':
                    # V(s_t) is conditioned on the prompt plus exactly t sampled tokens.
                    # The Critic uses its own tokenizer; token IDs cannot be shared.
                    prefixes=[self.tok.decode(prompt+chosen[:i],skip_special_tokens=False) for i in range(len(chosen))]
                    old_values=self.critic.values(prefixes,actor_device=device).detach()
                else:
                    old_values=self.critic(feat).detach()
                # Observable, deterministic reference-based reward. Never ask
                # this model to grade its own response.
                if is_hf:
                    completion=self.tok.decode(chosen,skip_special_tokens=True)
                    if not completion:
                        completion=self.tok.decode(chosen,skip_special_tokens=False)
                    reward=self.reward_scorer.score(row['prompt'],completion)
                else:
                    exact=int(chosen==reference)
                    token_matches=sum(a==b for a,b in zip(chosen,reference))
                    partial=token_matches/max(len(chosen),len(reference))
                    reward=float(exact if getattr(self.args,'ppo_reward','token_match')=='exact' else partial)
                T=len(chosen)
                rs=torch.zeros(T,device=device);rs[-1]=reward
                # Terminal-only verifiable reward; bootstrap V_T=0.
                adv=torch.zeros(T,device=device);gae=torch.zeros((),device=device)
                for i in reversed(range(T)):
                    next_v=old_values[i+1] if i+1<T else torch.zeros((),device=device)
                    delta=rs[i]+self.args.ppo_gamma*next_v-old_values[i]
                    gae=delta+self.args.ppo_gamma*self.args.ppo_gae_lambda*gae
                    adv[i]=gae
                returns=adv+old_values
                if T>1 and adv.std(unbiased=False)>1e-6:
                    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
                adv=adv.clamp(-10,10)
                return {'ids':ids_buffer[:,:generated_length].clone().detach(),'start':len(prompt),
                        'tokens':torch.tensor(chosen,device=device),
                        'old_lp':torch.stack(old_lp).detach(), 'old_values':old_values.detach(),
                        'adv':adv.detach(),'returns':returns.detach(), 'reward':reward,
                        'critic_prefixes':prefixes, 'reference_tokens':reference,
                        'behavior_checkpoint_v82':checkpoint, 'behavior_checkpoint_v81':checkpoint}
        finally:
            model.train(was_training)

    def ppo_loss(self,model,device):
        if self.cached is None or self.reuses>=self.args.ppo_reuse:
            self.cached=self._new_rollout(model,device)
            self.reuses=0
        rollout=self.cached
        ids=rollout['ids'];start=rollout['start'];N=rollout['tokens'].numel()
        if ids.device!=device:
            self.cached=None
            raise MixedLearningError('PPO device changed; rollouts must be recollected')
        was_training=model.training
        model.eval()  # avoid dropout mismatch between rollout and update
        try:
            logits=_logits(model,ids,self.args,start=start-1,length=N)
            lp=F.log_softmax(logits,dim=-1)
            chosen=lp.gather(-1,rollout['tokens'].view(1,N,1)).squeeze(0).squeeze(-1)
            # Never silently underreport a huge P2P policy shift by clipping
            # the log ratio *before* inspecting it. Extremely stale rollouts
            # must be recollected rather than replayed as if their KL were 20.
            raw_log_ratio=chosen-rollout['old_lp']
            if not bool(torch.isfinite(raw_log_ratio).all()):
                self.cached=None
                raise MixedLearningError('nonfinite PPO importance ratio; discard rollout')
            if bool((raw_log_ratio.detach().abs()>20).any()):
                self.cached=None
                raise MixedLearningError('PPO policy changed too far for a safe replay; discard stale rollout')
            log_ratio=raw_log_ratio
            ratio=log_ratio.exp()
            advantage=rollout['adv']
            clipped=ratio.clamp(1-self.args.ppo_clip,1+self.args.ppo_clip)
            pg_loss=-torch.minimum(ratio*advantage,clipped*advantage).mean()
            # Reference is the behavior policy at rollout time. Reuse is bounded.
            approx_kl=(log_ratio.square()).mean()/2
            entropy=-(lp.exp()*lp).sum(-1).mean()
            actor_loss=pg_loss+self.args.ppo_kl_coef*approx_kl
            # A small *local-only* auxiliary value function: trained for GAE,
            # not mixed into signatures, and not a model-as-judge.
            if getattr(self.args,'critic_kind','mlp')=='hf':
                values=self.critic.values(rollout['critic_prefixes'],actor_device=device)
            else:
                features=_features(logits,offset=start-1)[0]
                values=self.critic(features)
            metrics={'reward':rollout['reward'],
                     'kl':float(approx_kl.detach()),
                     'entropy':float(entropy.detach())}
            if resolved_mode(self.args)=='trajectory_is':
                # Exact (unclipped) full-trajectory density-ratio correction
                # for the Critic's terminal-reward Monte Carlo objective.
                # It corrects BOTH state visitation and future action
                # distribution, unlike suffix-only weighted regression.
                # The old policy's chosen-action log probabilities are stored
                # at sampling time and are immutable across PPO reuse.
                try:
                    value_loss, diagnostics=exact_trajectory_value_surrogate(
                        values.flatten(),rollout['old_lp'].flatten(),chosen.flatten(),
                        rollout['reward'],float(self.args.ppo_gamma))
                except OffPolicyCorrectionError as exc:
                    self.cached=None
                    raise MixedLearningError(str(exc)) from exc
                metrics.update(diagnostics)
            else:
                previous=rollout['old_values']
                clipped_values=previous+(values-previous).clamp(-self.args.ppo_clip,self.args.ppo_clip)
                value_loss=.5*torch.maximum((values-rollout['returns']).square(),
                                             (clipped_values-rollout['returns']).square()).mean()
            critic_loss=self.args.ppo_value_coef*value_loss
            if not bool(torch.isfinite(actor_loss+critic_loss)):
                self.cached=None
                raise MixedLearningError('nonfinite PPO objective')
            self.reuses+=1;self.counts['ppo']+=1
            return actor_loss,critic_loss,metrics
        finally:
            model.train(was_training)

    def export_rollout_v80(self):
        """Capture only source-sampled actions (once per PPO rollout).

        Never export activations/Gradients from the sender's Critic. The peer
        must evaluate its OWN Actor and re-derive its own value gradient.
        """
        from .trajectory_gossip_v80 import validate_rollout
        r=self.cached
        if r is None or r.get('reference_tokens') is None:
            raise MixedLearningError('v80 requires ground-truth reference rollout')
        data={'prompt':r['ids'][0,:r['start']].detach().cpu().tolist(),
              'tokens':r['tokens'].detach().cpu().tolist(),
              'behavior_logp':r['old_lp'].detach().cpu().float().tolist(),
              'reward':float(r['reward']),
              'reference':list(r['reference_tokens']),
              'reward_mode':str(self.args.ppo_reward)}
        validate_rollout(data,max_length=int(self.args.max_length),reward_mode=self.args.ppo_reward)
        return data

    def train_imported_rollout(self,model,device,data):
        """One local Critic-only AdamW step, on an authenticated remote sample.

        Actor never receives gradients. It is the TARGET pi, not the source's
        historic behavior mu, so its current action probabilities are used.
        """
        from .trajectory_gossip_v80 import validate_rollout
        validate_rollout(data,max_length=int(self.args.max_length),reward_mode=self.args.ppo_reward)
        prompt=data['prompt'];tokens=data['tokens'];n=len(tokens)
        # Network may try a different generation or vocab: safe input must
        # satisfy the current Actor's actual embedding indices.
        ids=torch.tensor([prompt+tokens],device=device,dtype=torch.long)
        was_training=model.training
        model.eval()
        try:
            with torch.no_grad():
                logits=_logits(model,ids,self.args,start=len(prompt)-1,length=n)
                target=F.log_softmax(logits,dim=-1).gather(
                    -1,torch.tensor(tokens,device=device).reshape(1,n,1)).reshape(n)
                if not bool(torch.isfinite(target).all()):
                    raise MixedLearningError('nonfinite recipient Actor probability')
                if getattr(self.args,'critic_kind','mlp')!='hf':
                    features=_features(logits,offset=len(prompt)-1)[0]
            if getattr(self.args,'critic_kind','mlp')=='hf':
                prefixes=[self.tok.decode(prompt+tokens[:i],skip_special_tokens=False) for i in range(n)]
                values=self.critic.values(prefixes,actor_device=device).flatten()
            else:
                self.critic.to(device)
                values=self.critic(features).flatten()
            for param,state in self.critic_optim.state.items():
                for key in ('exp_avg','exp_avg_sq'):
                    if key in state: state[key]=state[key].to(param.device)
            self.critic_optim.zero_grad(set_to_none=True)
            loss,diagnostics=exact_trajectory_value_surrogate(
                values,torch.tensor(data['behavior_logp'],device=device,dtype=torch.float64),
                target.detach(),float(data['reward']),float(self.args.ppo_gamma))
            (loss*float(self.args.ppo_value_coef)*float(self.args.ppo_weight)).backward()
            params=(self.critic.trainable_parameters() if getattr(self.args,'critic_kind','mlp')=='hf' else self.critic.parameters())
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                self.critic_optim.zero_grad(set_to_none=True)
                raise MixedLearningError('nonfinite received Critic gradient')
            self.critic_optim.step()
            self.critic_steps+=1
            self.cached=None;self.reuses=0
            return {'loss':float(loss.detach()),**diagnostics}
        finally:
            model.train(was_training)

    def save_critic(self):
        if (self.critic is not None and getattr(self,'_critic_dir',None) and
                not getattr(self,'_v75_paired',False)):
            if getattr(self.args,'critic_kind','mlp')=='hf':
                from .hf_critic_state_v69 import save
                save(self._critic_dir,self.critic,self.critic_optim,
                     identity=self._critic_identity,steps=self.critic_steps)
            else:
                from .critic_state_v66 import save
                save(self._critic_dir,self.critic,self.critic_optim,
                     kind=getattr(self.args,'critic_kind','mlp'),
                     hidden=int(getattr(self.args,'critic_hidden',32)),
                     identity=self._critic_identity,steps=self.critic_steps)

    def close(self):
        if self._closed:
            return
        self._closed=True
        try:
            if self.critic is not None:
                self.save_critic()
        finally:
            # Even a full disk/permission error must not strand open files
            # or background HF network readers during shutdown.
            try:
                if self.reward_scorer is not None:
                    close=getattr(self.reward_scorer,'close',None)
                    if callable(close):close()
            finally:
                for x in (self.sft,self.ppo):
                    if x:x.close()


def build(args, *, tokenizer=None):
    if not enabled(args):
        return None
    validate_args(args)
    if tokenizer is None:
        from .hf_load_v85 import load_hf_tokenizer
        tokenizer=load_hf_tokenizer(args.model,revision=args.revision,
                            trust_remote_code=bool(getattr(args,'trust_remote_code',False)),
                            local_files_only=bool(getattr(args,'trust_remote_code',False)))
    return MixedObjectiveTrainer(args,tokenizer)
