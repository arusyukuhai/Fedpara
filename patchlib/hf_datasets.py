"""Turn a room's pinned Hugging Face dataset collection into training batches.

Only data repositories are read; no dataset loading scripts or transforms run.
Every peer fetches data directly from Hugging Face, not from Mummy.
"""
from __future__ import annotations
import itertools
import random
from typing import Iterable

import torch
from .core import PatchError
from .room_manifest import validate_manifest


def _datasets_package():
    from importlib.metadata import version
    major = int(version('datasets').split('.')[0])
    if major < 4:
        raise RuntimeError('Safety requirement: install huggingface datasets>=4 (legacy dataset scripts unsupported)')
    import datasets
    return datasets


def rows_from_room(manifest: dict, *, peer_id: str='', loader=None, shuffle_buffer=512) -> Iterable[dict]:
    """Infinite, weighted, lazy interleave; fail loudly on empty sources.

    Individual peer seeds diversify samples without changing the room schema.
    Cache/credentials stay on the client; dataset records are never sent via Mummy.
    """
    validate_manifest(manifest)
    if manifest['version'] != 2:
        raise PatchError('room manifest v1 has no HF dataset references; create a v2 room manifest')
    if not 2 <= shuffle_buffer <= 100_000:
        raise ValueError('shuffle_buffer out of range')
    sources=manifest['datasets']['sources']
    if loader is None:
        loader=_datasets_package().load_dataset
    from .data_sampling import peer_sampling_seed
    seed=peer_sampling_seed(peer_id, manifest['datasets']['seed'])
    rng=random.Random(seed)
    streams=[]
    for i, source in enumerate(sources):
        opts=dict(path=source['repo'],split=source['split'],revision=source['revision'],
                  streaming=True)
        if source['config'] is not None:
            opts['name']=source['config']
        # datasets>=4 rejects datasets backed by executable repository scripts.
        ds=loader(**opts)
        if not hasattr(ds,'shuffle'):
            raise PatchError('HF loader must return a streaming iterable dataset')
        ds=ds.shuffle(seed=(seed+i) & 0x7FFFFFFF,buffer_size=shuffle_buffer)
        streams.append([ds,iter(ds)])
    weights=[src['weight'] for src in sources]
    while True:
        i=rng.choices(range(len(sources)),weights=weights,k=1)[0]
        ds,it=streams[i]
        try:
            row=next(it)
        except StopIteration:
            # Empty streams must raise rather than hang endlessly.
            it=iter(ds)
            try: row=next(it)
            except StopIteration as exc:
                raise PatchError('empty Hugging Face dataset: '+sources[i]['repo']) from exc
            streams[i][1]=it
        cols=sources[i]['columns']
        if not isinstance(row,dict):
            raise PatchError('HF dataset row must be a mapping')
        out={}
        for role,key in cols.items():
            value=row.get(key)
            if role=='label':
                if type(value) is not int or value < 0 or value > 2**31-1:
                    raise PatchError('dataset classification label must be a nonnegative integer')
            else:
                if not isinstance(value,str):
                    raise PatchError('dataset column '+key+' must contain text')
                value=value[:200_000]  # hard per-row memory guard
            out[role]=value
        yield out


def batches_from_room(args, manifest: dict, *, rows=None):
    """HF data -> model(**batch) for supported text Transformer tasks."""
    validate_manifest(manifest)
    task=args.hf_task
    required=({'text'} if task in ('causal-lm','masked-lm') else
              {'input','target'} if task == 'seq2seq-lm' else
              {'text','label'} if task == 'sequence-classification' else None)
    if required is None:
        raise ValueError('Other model tasks require --batches-dir until a task collator is provided')
    for spec in manifest['datasets']['sources']:
        if set(spec['columns']) != required:
            raise PatchError('task '+task+' requires dataset columns '+str(sorted(required)))
    from transformers import DataCollatorForLanguageModeling
    from .hf_load_v85 import load_hf_tokenizer
    from .model_design import tokenizer_identity
    tokenizer_id,tokenizer_revision=tokenizer_identity(manifest)
    tokenizer=load_hf_tokenizer(tokenizer_id,revision=tokenizer_revision,trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('tokenizer must have pad or eos token')
        tokenizer.pad_token=tokenizer.eos_token
    if rows is None:
        rows=rows_from_room(manifest,peer_id=getattr(args,'peer_id',None) or '',
                            shuffle_buffer=getattr(args,'hf_shuffle_buffer',512))
    mlm=task=='masked-lm'
    # Only masked LM needs HF's masking collator. Its CLM path masks by token
    # VALUE, which destroys real EOS targets if pad_token_id == eos_token_id.
    collator=(DataCollatorForLanguageModeling(tokenizer,mlm=True,mlm_probability=0.15)
              if mlm else None)
    while True:
        samples=list(itertools.islice(rows,args.batch_size))
        if not samples:
            raise PatchError('dataset iterator exhausted unexpectedly')
        if task in ('causal-lm','masked-lm'):
            encoded=tokenizer([x['text'] for x in samples],padding=True,truncation=True,
                              max_length=args.max_length,return_tensors='pt',
                              **({'return_token_type_ids':False} if not mlm else {}))
            if mlm:
                batch=collator([{k:v[j] for k,v in encoded.items()} for j in range(len(samples))])
            else:
                from .training_compat_v84 import causal_labels_from_attention
                batch=dict(encoded)
                batch['labels']=causal_labels_from_attention(batch['input_ids'],
                                                             batch['attention_mask'])
        elif task=='seq2seq-lm':
            batch=tokenizer([x['input'] for x in samples],padding=True,truncation=True,
                            max_length=args.max_length,return_tensors='pt',
                            return_token_type_ids=False)
            labels=tokenizer(text_target=[x['target'] for x in samples],padding=True,
                             truncation=True,max_length=args.max_length,return_tensors='pt')['input_ids']
            labels[labels==tokenizer.pad_token_id]=-100
            batch['labels']=labels
        else:
            batch=tokenizer([x['text'] for x in samples],padding=True,truncation=True,
                            max_length=args.max_length,return_tensors='pt')
            batch['labels']=torch.tensor([x['label'] for x in samples],dtype=torch.long)
        yield batch
