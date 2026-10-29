"""Locally evaluated, externally trained Hugging Face PPO reward models.

An independent scalar scorer, NOT the local PPO value critic.  No Hub-provided
Python is executed. A pinned immutable HF revision avoids cross-peer drift.
Only scores understood by the declared adapter are accepted; unexpected
outputs fail closed rather than silently becoming zero rewards.
"""
from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import torch


class RewardModelError(ValueError):
    pass


_HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def reward_spec(args):
    name=str(getattr(args,'ppo_reward','token_match'))
    if name!='hf_model':
        return None
    repo=str(getattr(args,'reward_model_id','')).strip()
    revision=str(getattr(args,'reward_model_revision','')).strip()
    if not _HF_REPO.fullmatch(repo):
        raise RewardModelError('Hugging Face reward model must be a repo ID of the form organization/name')
    if not _SHA.fullmatch(revision):
        raise RewardModelError('Reward model revision must be the immutable 40-character Hugging Face commit SHA for consistent P2P scores')
    loader=str(getattr(args,'reward_model_loader','sequence_classification'))
    if loader not in ('sequence_classification','auto_model'):
        raise RewardModelError('Reward loader must be sequence_classification or auto_model')
    output=str(getattr(args,'reward_model_output','logits'))
    if output not in ('logits','score','end_scores','rewards'):
        raise RewardModelError('Unsupported reward output key')
    index=int(getattr(args,'reward_model_index',0))
    if type(getattr(args,'reward_model_index',0)) is not int or not 0<=index<=65535:
        raise RewardModelError('Reward label index must be 0..65535')
    scale=float(getattr(args,'reward_model_scale',1.))
    if not math.isfinite(scale) or not 0<scale<=100:
        raise RewardModelError('Reward scale must be finite in (0,100]')
    max_length=getattr(args,'reward_model_max_length',512)
    if type(max_length) is not int or not 16<=max_length<=8192:
        raise RewardModelError('Reward input maximum length must be 16..8192 tokens')
    fmt=str(getattr(args,'reward_model_format','plain'))
    if fmt not in ('plain','chat'):
        raise RewardModelError('Reward format must be plain or chat')
    return {'repo':repo,'revision':revision.lower(),'loader':loader,'output':output,
            'index':index,'scale':scale,'max_length':max_length,
            'format':fmt, 'trust_remote_code':False}


def _read_output(result: Any, key: str):
    if isinstance(result,dict):
        value=result.get(key)
    else:
        value=getattr(result,key,None)
    if not isinstance(value,torch.Tensor):
        raise RewardModelError(f'Reward model did not return tensor field {key!r}; choose a compatible scalar reward head')
    if value.ndim==1:
        value=value.unsqueeze(0)
    if value.ndim!=2 or value.shape[0]!=1 or value.shape[1]<1:
        raise RewardModelError(f'Reward output {key} needs shape [1,num_scores], received {tuple(value.shape)}')
    return value


class HFRewardScorer:
    """Independent frozen scorer. Never adds parameters to the actor's optimizer."""
    def __init__(self,args, *, model=None, tokenizer=None):
        self.config=reward_spec(args)
        if self.config is None:
            raise RewardModelError('Hugging Face reward scoring is not enabled')
        device=str(getattr(args,'reward_model_device','cpu'))
        if device not in ('cpu','cuda','mps'):
            raise RewardModelError('Reward scoring device must be cpu, cuda, or mps')
        if device=='cuda' and not torch.cuda.is_available():
            raise RewardModelError('Reward Model CUDA device was requested but unavailable')
        if device=='mps' and not torch.backends.mps.is_available():
            raise RewardModelError('Reward Model MPS device was requested but unavailable')
        self.device=torch.device(device)
        self.idle_offload=bool(getattr(args,'reward_model_idle_offload',False)) and device!='cpu'
        if model is None or tokenizer is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoModel
                from .hf_load_v85 import load_hf_model, load_hf_tokenizer
                kw={'revision':self.config['revision'],'trust_remote_code':False,
                    'use_safetensors':True}
                if model is None:
                    # Check the trained score head was actually loaded, not randomly initialized.
                    model_class=(AutoModelForSequenceClassification if self.config['loader']=='sequence_classification' else AutoModel)
                    kw['output_loading_info']=True
                    try:
                        result=load_hf_model(model_class,self.config['repo'],dtype='auto',**kw)
                    except TypeError as exc:
                        if 'dtype' not in str(exc):raise
                        result=load_hf_model(model_class,self.config['repo'],torch_dtype='auto',**kw)
                    if not isinstance(result,tuple) or len(result)!=2:
                        raise RewardModelError('Reward checkpoint does not provide loading diagnostics')
                    model,info=result
                    missing=info.get('missing_keys',[])
                    if missing:
                        raise RewardModelError('Reward checkpoint has missing/untrained parameters: '+str(missing[:8]))
                if tokenizer is None:
                    tokenizer=load_hf_tokenizer(self.config['repo'],revision=self.config['revision'],
                                                          trust_remote_code=False)
            except Exception as exc:
                raise RewardModelError('Cannot load Hugging Face reward checkpoint/tokenizer (must have supported scalar sequence classification head and trusted built-in architecture): '+str(exc)) from exc
        self.model=model.to('cpu' if self.idle_offload else self.device).eval()
        self.tok=tokenizer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Avoid an untrained randomly initialized classification head.
        config=getattr(self.model,'config',None)
        count=getattr(config,'num_labels',None)
        if count is not None and (type(count) is not int or self.config['index']>=count):
            raise RewardModelError('Reward label index is not valid for the model classification head')
        self.calls=0

    def close(self):
        """Release an optional GPU resident copy without modifying frozen weights."""
        if self.idle_offload:
            self.model.to('cpu')

    @torch.inference_mode()
    def score(self, prompt: str, completion: str) -> float:
        if self.idle_offload:
            self.model.to(self.device)
        try:
            return self._score_active(prompt,completion)
        finally:
            if self.idle_offload:
                # Scorer is frozen; no autograd graph/optimizer may retain the GPU copy.
                self.model.to('cpu')

    @torch.inference_mode()
    def _score_active(self, prompt: str, completion: str) -> float:
        if not isinstance(prompt,str) or not isinstance(completion,str) or not prompt or not completion:
            raise RewardModelError('Reward scoring requires nonempty prompt and generated completion')
        try:
            if self.config['format']=='chat':
                if not hasattr(self.tok,'apply_chat_template'):
                    raise RewardModelError('This reward tokenizer has no chat template')
                ids=self.tok.apply_chat_template([
                    {'role':'user','content':prompt},
                    {'role':'assistant','content':completion}],
                    tokenize=True,add_generation_prompt=False)
                if not isinstance(ids,(list,tuple)) or not ids or not all(isinstance(i,int) for i in ids):
                    raise RewardModelError('Reward chat template must return token IDs')
                ids=ids[-self.config['max_length']:]
                input_ids=torch.tensor([ids],device=self.device,dtype=torch.long)
                kwargs={'input_ids':input_ids,'attention_mask':torch.ones_like(input_ids)}
            else:
                # Keep generated answer visible when the prompt exceeds the RM window.
                text=prompt+completion
                if len(text)>262144:
                    text=text[-262144:]
                old_side=getattr(self.tok,'truncation_side',None)
                try:
                    if old_side is not None:self.tok.truncation_side='left'
                    batch=self.tok(text,return_tensors='pt',truncation=True,
                                   max_length=self.config['max_length'])
                finally:
                    if old_side is not None:self.tok.truncation_side=old_side
                if not isinstance(batch,Mapping) or 'input_ids' not in batch:
                    raise RewardModelError('Reward tokenizer did not provide input_ids')
                if batch['input_ids'].shape[0]!=1 or batch['input_ids'].shape[1]==0:
                    raise RewardModelError('Reward model input is empty')
                kwargs={k:v.to(self.device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
            result=self.model(**kwargs)
            scores=_read_output(result,self.config['output'])
            if self.config['index']>=scores.shape[1]:
                raise RewardModelError('Reward model output is smaller than the configured score index')
            val=float(scores[0,self.config['index']].float().item())*self.config['scale']
        except RewardModelError:
            raise
        except Exception as exc:
            raise RewardModelError('Hugging Face reward inference failed: '+str(exc)) from exc
        if not math.isfinite(val):
            raise RewardModelError('Hugging Face reward output is not finite')
        self.calls+=1
        # Bound GAE/critic inputs even if a model is poorly calibrated.
        return max(-10.,min(10.,val))
