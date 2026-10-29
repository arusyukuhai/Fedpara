"""Pinned HF sequence value critic for PPO, trained locally and never shared as P2P adapter.

Reward Model scores finished responses. This model instead predicts per-action
state values using *prefixes* of the generated response and is optimized by PPO
value loss. The tokenizer can differ from the actor tokenizer. Only an already
trained scalar HF head is used; its frozen backbone never enters the optimizer.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import torch
from torch import nn

REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
SHA = re.compile(r'^[a-fA-F0-9]{40}$')
MAX_REPLAY_FEATURE_BYTES = 32 * 1024**2


class HFValueError(ValueError):
    pass


def critic_spec(args):
    if getattr(args, 'critic_kind', 'mlp') != 'hf':
        return None
    repo=str(getattr(args,'critic_hf_repo','') or '').strip()
    rev=str(getattr(args,'critic_hf_revision','') or '').strip()
    loader=str(getattr(args,'critic_hf_loader','sequence_classification'))
    device=str(getattr(args,'critic_hf_device','cpu'))
    limit=getattr(args,'critic_hf_max_length',512)
    if not REPO.fullmatch(repo) or not SHA.fullmatch(rev):
        raise HFValueError('HF Critic requires organization/model and a pinned 40-character commit SHA')
    if loader not in ('sequence_classification','token_classification'):
        raise HFValueError('HF Critic loader must be scalar sequence or token classification')
    if device not in ('cpu','cuda','mps'):
        raise HFValueError('HF Critic device must be cpu/cuda/mps')
    if type(limit) is not int or not 16<=limit<=8192:
        raise HFValueError('HF Critic max input must be 16..8192 tokens')
    return {'repo':repo,'revision':rev.lower(),'loader':loader,'max_length':limit,
            'trainable':'scalar_head_only-v1','trust_remote_code':False}


class HFValueCritic(nn.Module):
    def __init__(self,args,*,model=None,tokenizer=None):
        super().__init__()
        self.spec=critic_spec(args)
        if self.spec is None:raise HFValueError('HF Critic not selected')
        device=str(getattr(args,'critic_hf_device','cpu'))
        if device=='cuda' and not torch.cuda.is_available():
            raise HFValueError('Requested CUDA for HF Critic is unavailable')
        if device=='mps' and not torch.backends.mps.is_available():
            raise HFValueError('Requested MPS for HF Critic is unavailable')
        self.device=torch.device(device)
        if model is None or tokenizer is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
                from .hf_load_v85 import load_hf_model, load_hf_tokenizer
                kw={'revision':self.spec['revision'],'trust_remote_code':False,
                    'use_safetensors':True,'output_loading_info':True}
                cls=AutoModelForSequenceClassification if self.spec['loader']=='sequence_classification' else AutoModelForTokenClassification
                if model is None:
                    try:loaded=load_hf_model(cls,self.spec['repo'],dtype='auto',**kw)
                    except TypeError as exc:
                        if 'dtype' not in str(exc):raise
                        loaded=load_hf_model(cls,self.spec['repo'],torch_dtype='auto',**kw)
                    if not isinstance(loaded,tuple) or len(loaded)!=2:
                        raise HFValueError('HF Critic loader did not return weight-loading diagnostics')
                    model,info=loaded
                    if info.get('missing_keys') or info.get('mismatched_keys'):
                        raise HFValueError('HF Critic head/weights are missing or mismatched; refusing random initialization')
                if tokenizer is None:
                    tokenizer=load_hf_tokenizer(self.spec['repo'],revision=self.spec['revision'],trust_remote_code=False)
            except Exception as exc:
                if isinstance(exc,HFValueError):raise
                raise HFValueError(f'Cannot load pinned HF Critic safely: {exc}') from exc
        if getattr(getattr(model,'config',None),'num_labels',None)!=1:
            raise HFValueError('HF Critic must expose a PRETRAINED single scalar value head (num_labels=1)')
        candidates=[]
        for name,layer in model.named_modules():
            if (isinstance(layer,nn.Linear) and layer.out_features==1 and
                (name.split('.')[-1] in ('score','classifier','value_head','v_head','value','reward_head') or
                 'classifier' in name.lower())):
                candidates.append((name,layer))
        if len(candidates)!=1:
            raise HFValueError('Need exactly one trained scalar value head (Linear out_features=1); found '+str(len(candidates)))
        self.head_name,self.head=candidates[0]
        self.model=model.to(self.device)
        for p in self.model.parameters():p.requires_grad_(False)
        for p in self.head.parameters():p.requires_grad_(True)
        self.model.eval()  # deterministic old/current value predictions
        self.tok=tokenizer
        # New completions must stay visible when the shared context exceeds
        # the Critic's maximum input length. Its tokenizer is actor-independent.
        if hasattr(self.tok,'truncation_side'):
            self.tok.truncation_side='left'
        self.calls=0
        # PPO reuses a bounded set of prefixes. The backbone is frozen, so its
        # last hidden vectors can be reused while the scalar head is re-run on
        # every critic optimizer step. Never cache graphs or value predictions.
        self._cached_prefix_key=None
        self._cached_features=None

    def trainable_parameters(self):
        return list(self.head.parameters())

    def values(self,prefixes,*,actor_device):
        """Evaluate prefixes in bounded batches, retaining head gradients.

        The old path did one full backbone forward per generated token, again
        for every PPO reuse. We batch independent prefixes with right padding
        and an attention mask. Work is capped by both examples and padded token
        count to avoid a large temporary quadratic attention allocation.
        """
        if not isinstance(prefixes,list) or not 1<=len(prefixes)<=128:
            raise HFValueError('Expected 1..128 PPO state prefixes')
        self.model.eval()
        prefix_key=tuple(prefixes)
        if self._cached_features is not None and prefix_key==self._cached_prefix_key:
            # Keep the *head* in the graph: PPO still trains it on each reuse.
            values=self.head(self._cached_features).reshape(-1)
            if values.shape!=(len(prefixes),) or not bool(torch.isfinite(values).all()):
                self._cached_features=None
                self._cached_prefix_key=None
                raise HFValueError('Cached critic features produced invalid values')
            return values.float().to(actor_device)
        self._cached_features=None
        self._cached_prefix_key=None
        candidate_features=[None]*len(prefixes)
        can_cache=True
        cache_bytes=0
        samples=[]
        for text in prefixes:
            if not isinstance(text,str) or not text or len(text)>65536:
                raise HFValueError('Empty/oversized PPO critic prefix')
            batch=self.tok(text,return_tensors='pt',truncation=True,max_length=self.spec['max_length'])
            if not hasattr(batch,'items'):
                raise HFValueError('HF Critic tokenizer must return a tensor mapping')
            data={k:v for k,v in batch.items() if k in ('input_ids','attention_mask','token_type_ids')}
            ids=data.get('input_ids')
            if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]==0:
                raise HFValueError('HF Critic tokenizer returned invalid input_ids')
            length=int(ids.shape[1])
            if length>self.spec['max_length']:
                raise HFValueError('HF Critic tokenizer exceeded the configured input length')
            if 'attention_mask' in data and (data['attention_mask'].shape!=ids.shape or
                    not bool(torch.all(data['attention_mask']==1))):
                # Prefixes must be unpadded before the new right-padding step.
                raise HFValueError('HF Critic tokenizer returned a pre-padded or invalid prefix')
            for name,value in data.items():
                if not isinstance(value,torch.Tensor) or value.shape!=ids.shape:
                    raise HFValueError(f'Invalid HF Critic tokenizer field {name}')
            samples.append((length,data))
        # Sort by size so small prefixes are not padded to the longest one.
        # Do not detach, or gradients to the trainable scalar head would vanish.
        ordered=sorted(range(len(samples)),key=lambda idx:samples[idx][0])
        output=[None]*len(samples)
        # Some causal sequence-classifier implementations (e.g. Llama-derived)
        # explicitly reject batch>1 without a configured pad_token_id. Never
        # change the pinned checkpoint's config merely to force faster batches.
        configured_pad=getattr(getattr(self.model,'config',None),'pad_token_id',None)
        supports_batch=(self.spec['loader']=='token_classification' or
                        (isinstance(configured_pad,int) and configured_pad>=0))
        max_items=8 if supports_batch else 1
        n=0
        while n<len(ordered):
            group=[]
            max_length=0
            while n<len(ordered):
                idx=ordered[n]
                candidate=max(max_length,samples[idx][0])
                if group and (len(group)>=max_items or candidate*(len(group)+1)>2048):
                    break
                group.append(idx);max_length=candidate;n+=1
            pad_id=(configured_pad if self.spec['loader']=='sequence_classification'
                    else getattr(self.tok,'pad_token_id',None))
            if not isinstance(pad_id,int) or pad_id<0:
                pad_id=getattr(self.tok,'eos_token_id',None)
            if not isinstance(pad_id,int) or pad_id<0:
                pad_id=0  # mask=0; no padded token may contribute to prediction
            packed={'input_ids':[], 'attention_mask':[]}
            include_types=all('token_type_ids' in samples[idx][1] for idx in group)
            if include_types:packed['token_type_ids']=[]
            for idx in group:
                length,row=samples[idx]
                diff=max_length-length
                packed['input_ids'].append(torch.cat((row['input_ids'][0],
                    row['input_ids'].new_full((diff,),pad_id))))
                packed['attention_mask'].append(torch.cat((row['input_ids'].new_ones(length),
                    row['input_ids'].new_zeros(diff))))
                if include_types:
                    packed['token_type_ids'].append(torch.cat((row['token_type_ids'][0],
                        row['token_type_ids'].new_zeros(diff))))
            tensors={key:torch.stack(items).to(self.device) for key,items in packed.items()}
            head_inputs=[]
            def capture_head(_module, inputs):
                if len(inputs)==1 and isinstance(inputs[0],torch.Tensor):
                    head_inputs.append(inputs[0].detach())
            hook=self.head.register_forward_pre_hook(capture_head)
            try:
                result=self.model(**tensors)
            finally:
                hook.remove()
            logits=result.get('logits') if isinstance(result,dict) else getattr(result,'logits',None)
            if not isinstance(logits,torch.Tensor):
                raise HFValueError('HF Critic must return logits tensor')
            if self.spec['loader']=='sequence_classification':
                if logits.shape!=(len(group),1):
                    raise HFValueError('Sequence HF Critic logits must be [batch,1]')
                values=logits[:,0]
            else:
                if logits.ndim!=3 or logits.shape!=(len(group),max_length,1):
                    raise HFValueError('Token HF Critic logits must be [batch,tokens,1]')
                final_index=torch.tensor([samples[idx][0]-1 for idx in group],device=logits.device)
                values=logits[torch.arange(len(group),device=logits.device),final_index,0]
            if not bool(torch.isfinite(values).all()):
                raise HFValueError('HF Critic returned nonfinite state value')
            if can_cache:
                # Only bypass the backbone if the model's actual scalar output
                # is numerically identical to the selected Linear head output.
                # Custom post-head transforms remain on the conservative path.
                if len(head_inputs)!=1:
                    can_cache=False
                else:
                    hidden=head_inputs[0]
                    if hidden.ndim==3 and hidden.shape[:2]==(len(group),max_length):
                        selection=torch.tensor([samples[idx][0]-1 for idx in group],device=hidden.device)
                        hidden=hidden[torch.arange(len(group),device=hidden.device),selection]
                    if hidden.ndim!=2 or hidden.shape[0]!=len(group) or hidden.shape[1]!=self.head.in_features:
                        can_cache=False
                    elif cache_bytes + hidden.numel()*hidden.element_size() > MAX_REPLAY_FEATURE_BYTES:
                        # Enforce the bound *before* allocating any clones;
                        # inspecting it at the end could cause large spikes.
                        can_cache=False
                        candidate_features=[]
                    else:
                        with torch.no_grad():
                            direct=self.head(hidden).reshape(-1)
                            matches=bool(torch.allclose(direct.float(),values.detach().float(),
                                                        rtol=1e-4,atol=1e-4))
                        if not matches:
                            can_cache=False
                        else:
                            cache_bytes += hidden.numel()*hidden.element_size()
                            for idx,vector in zip(group,hidden.unbind(0)):
                                candidate_features[idx]=vector.detach().clone()
            for idx,value in zip(group,values.unbind(0)):
                output[idx]=value.float().to(actor_device)
            self.calls+=len(group)
        if can_cache and all(feature is not None for feature in candidate_features):
            self._cached_features=torch.stack(candidate_features).detach()
            self._cached_prefix_key=prefix_key
        return torch.stack(output)

    def snapshot_weights(self):
        return {name:p.detach().cpu().contiguous().clone() for name,p in self.head.named_parameters()}
