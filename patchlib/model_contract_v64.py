"""Architecture-neutral *execution* contract for an ordinary differentiable torch.nn.Module.

The patch format is tensor-based, not a claim that every model can be safely
executed.  This module separates invocation, loss semantics and a real gradient
probe from the architecture name.  No remote code is imported here.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping

import torch
from torch.nn import functional as F


class ModelContractError(ValueError):
    pass


def parse_input_map(spec):
    if spec is None or spec == '':
        return {}
    if isinstance(spec, str):
        if len(spec)>4096:
            raise ModelContractError('input map exceeds 4KiB')
        try:
            spec = json.loads(spec)
        except (ValueError, TypeError) as exc:
            raise ModelContractError('--input-map must be a JSON object') from exc
    if (not isinstance(spec, dict) or len(spec) > 32 or
            any(not isinstance(k,str) or
                (v is not None and not isinstance(v,str)) or
                not k.isidentifier() or (v is not None and not v.isidentifier()) or
                k.startswith('_') or (v is not None and v.startswith('_'))
                for k,v in spec.items()) or
            len(set(v for v in spec.values() if v is not None)) !=
            sum(v is not None for v in spec.values())):
        raise ModelContractError('input map must contain <=32 identifier-to-identifier or null(drop) entries')
    return dict(spec)


def invoke_model(model, batch: Mapping, *, input_map=None):
    """Do not send labels if a custom forward does not accept them.

    `labels` still belongs to the loss contract; other unexpected arguments
    are reported, never silently discarded. Explicit input remapping supports
    models with tokens/idx/etc instead of Hugging Face input_ids.
    """
    if not isinstance(batch, Mapping):
        raise ModelContractError('batch must be a mapping of model input tensors')
    mapping = parse_input_map(input_map)
    inputs = {}
    for name, value in batch.items():
        dest = mapping.get(name, name)
        if dest is None:  # explicitly drop e.g. unused attention_mask
            continue
        if dest in inputs:
            raise ModelContractError(f'input map collides at {dest!r}')
        inputs[dest] = value
    try:
        signature = inspect.signature(model.forward)
    except (ValueError, TypeError):
        # Native extension / TorchScript forwards may have no Python signature;
        # pass inputs unchanged instead of pretending to know their ABI.
        return model(**inputs)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    accepted = {k for k, p in signature.parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    if not accepts_kwargs:
        for name in list(inputs):
            if name == mapping.get('labels','labels') and name not in accepted:
                del inputs[name]  # labels remain in ORIGINAL batch for external loss
            elif name not in accepted:
                raise ModelContractError(f'model.forward does not accept {name!r}; '
                                         'use --input-map or a matching --batches-dir')
    return model(**inputs)


def _loss_or_logits(output):
    if isinstance(output, torch.Tensor):
        return (output if output.ndim == 0 else None,
                output if output.ndim >= 2 else None)
    if isinstance(output, Mapping):
        return output.get('loss'), output.get('logits')
    if hasattr(output, 'loss') or hasattr(output, 'logits'):
        return getattr(output, 'loss', None), getattr(output, 'logits', None)
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return None, None
        if isinstance(output[0], torch.Tensor) and output[0].ndim == 0:
            return output[0], output[1] if len(output)>1 else None
        if isinstance(output[0], torch.Tensor) and output[0].ndim >= 2:
            return None, output[0]
    return None, None


def training_loss(output, batch, *, task='causal-lm', policy='auto'):
    """Return differentiable scalar; never guess causality for unknown tasks.

    A bare logits model is supported without replacing its PyTorch forward:
    - causal-lm: predict token t+1 from logits at t
    - masked-lm/seq2seq-lm: predict the *aligned* token (no shift)
    - classification: predict per-example label
    When `policy=provided`, no external loss is allowed.
    """
    if policy not in ('auto','provided','causal','token','classification'):
        raise ModelContractError('unknown loss policy')
    provided, logits = _loss_or_logits(output)
    if provided is not None and policy in ('auto','provided'):
        if not isinstance(provided, torch.Tensor) or provided.ndim != 0:
            raise ModelContractError('model-provided loss must be a scalar tensor')
        loss = provided
    else:
        if policy == 'provided':
            raise ModelContractError('model returned no scalar loss and loss policy is provided')
        if not isinstance(logits, torch.Tensor):
            raise ModelContractError('model returned neither a scalar loss nor logits')
        labels = batch.get('labels')
        if not isinstance(labels, torch.Tensor):
            raise ModelContractError('a logits-only model requires tensor batch["labels"]')
        resolved = policy
        if resolved == 'auto':
            resolved = {'causal-lm':'causal', 'masked-lm':'token',
                        'seq2seq-lm':'token','sequence-classification':'classification',
                        'token-classification':'token'}.get(task)
        if resolved is None:
            raise ModelContractError(f'task {task!r} has no defined loss; use --loss-policy explicitly')
        if resolved in ('causal', 'token'):
            if (logits.ndim != labels.ndim+1 or
                    logits.shape[:-1] != labels.shape or logits.size(-1) < 2):
                raise ModelContractError('LM logits [batch,sequence,vocabulary] must match labels [batch,sequence]')
            if resolved == 'causal':
                if labels.ndim != 2 or labels.shape[-1] < 2:
                    raise ModelContractError('causal LM requires at least two tokens per sample')
                logits, labels = logits[:, :-1, :], labels[:, 1:]
            elif labels.ndim not in (1,2):
                raise ModelContractError('token labels must be 1-D or 2-D')
            if not bool((labels != -100).any().item()):
                raise ModelContractError('all labels are ignored; cannot train empty supervision')
            loss = F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),
                                   labels.reshape(-1).long(),ignore_index=-100)
        else:
            if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
                raise ModelContractError('classification expects logits [batch,classes], labels [batch]')
            if not bool((labels != -100).any().item()):
                raise ModelContractError('all labels are ignored')
            loss = F.cross_entropy(logits.float(), labels.long(),ignore_index=-100)
    if not isinstance(loss,torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ModelContractError('loss is not differentiable; model forward may detach weights')
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ModelContractError('nonfinite loss')
    return loss


def probe_model(model, adapter, batch, *, task='causal-lm', policy='auto', input_map=None):
    """One actual backward pass, no optimizer step; fail closed on missing grads.

    This proves only this input's forward/backward, not arbitrary models or
    sequence lengths. It is safe to use on a fresh model before starting a room.
    """
    model.train()
    for parameter in adapter.trainable_parameters():
        parameter.grad = None
    output = invoke_model(model, batch, input_map=input_map)
    loss = training_loss(output,batch,task=task,policy=policy)
    loss.backward()
    parts = list(adapter.trainable_parameters())
    reached = sum(p.grad is not None for p in parts)
    nonfinite = sum(p.grad is not None and not bool(torch.isfinite(p.grad).all().item()) for p in parts)
    useful = sum(p.grad is not None and bool((p.grad != 0).any().item()) for p in parts)
    report = {'adapter_layout':adapter.layout, 'groups':len(adapter.groups),
              'adapter_tensors':len(parts), 'gradients_present':reached,
              'nonzero_gradients':useful,'nonfinite_gradients':nonfinite,
              'loss':float(loss.detach().cpu()),'forward_backward':'pass' if reached and useful and not nonfinite else 'fail'}
    for p in parts:
        p.grad = None
    if not reached or not useful or nonfinite:
        raise ModelContractError('forward/backward did not reach trainable adapters with nonzero finite LoRA/LoHA gradients; '
                                 'check task/head alignment, token labels, frozen-base '
                                 'checkpointing and selected adapter targets')
    return report
