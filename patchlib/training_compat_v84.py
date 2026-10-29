"""Kagami v84: conservative training compatibility fixes.

No architecture is assumed to work merely because it can be constructed:
forward/backward must still be checked on a real representative training batch.
"""
from __future__ import annotations

import torch
from torch import nn


def causal_labels_from_attention(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask actual padding, NEVER every occurrence of an EOS-as-PAD token ID."""
    if not isinstance(input_ids,torch.Tensor) or not isinstance(attention_mask,torch.Tensor):
        raise TypeError('input_ids and attention_mask must be tensors')
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError('input IDs and attention masks must have identical [batch,sequence] shape')
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return labels


def _force_input_embedding_output_grad(model: nn.Module) -> str:
    """Legacy reentrant checkpointing needs an activation with requires_grad.

    The embedding's frozen weight is not made trainable. We only mark its
    FLOATING OUTPUT as an autograd leaf to allow gradients to reach later LoHA.
    """
    existing = getattr(model,'enable_input_require_grads',None)
    if callable(existing):
        existing()
        return 'input_require_grads'
    getter = getattr(model,'get_input_embeddings',None)
    embedding = getter() if callable(getter) else None
    if not isinstance(embedding,nn.Module):
        raise RuntimeError(
            'Legacy gradient checkpointing requires a usable input embedding ' 
            'or enable_input_require_grads(); disable --gradient-checkpointing '
            'or upgrade the model/Transformers implementation.')
    if getattr(model,'_kagami_v84_input_grad_hook',None) is None:
        def make_require_grad(_module, _inputs, output):
            if not isinstance(output,torch.Tensor) or not output.is_floating_point():
                raise RuntimeError('embedding output must be a floating tensor for legacy checkpointing')
            output.requires_grad_(True)
        handle = embedding.register_forward_hook(make_require_grad)
        object.__setattr__(model,'_kagami_v84_input_grad_hook',handle)
    return 'embedding_output_hook'


def enable_adapter_safe_checkpointing(model: nn.Module) -> str:
    """Prefer non-reentrant checkpointing with frozen bases + trainable adapters.

    Old HF implementations may not accept gradient_checkpointing_kwargs;
    their reentrant mode needs a gradient-bearing embedding output. No base
    Parameter's requires_grad flag is changed, and no optimizer is touched.
    """
    enable=getattr(model,'gradient_checkpointing_enable',None)
    if not callable(enable):
        raise RuntimeError('Model does not implement gradient_checkpointing_enable; '
                           'disable --gradient-checkpointing for this architecture')
    try:
        enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return 'non_reentrant'
    except TypeError as exc:
        # Do not hide TypeErrors thrown INSIDE the model; only an actual
        # signature incompatibility permits switching to legacy behavior.
        import inspect
        try: sig=inspect.signature(enable)
        except (TypeError,ValueError): raise exc
        if 'gradient_checkpointing_kwargs' in sig.parameters or any(
             p.kind==p.VAR_KEYWORD for p in sig.parameters.values()):
            raise
    enable()
    return _force_input_embedding_output_grad(model)
