"""Kagami v85: consistently recover *auditable* legacy HF configs on all loads.

Transformers AutoTokenizer.from_pretrained() can call AutoConfig internally when
no tokenizer_class is explicitly provided. The old Kagami workaround only
wrapped *model* config loads; thus data/eval/RM/Critic could still raise the
original ``Should have a model_type key in its config.json`` error.

Do not monkeypatch transformers globally, edit pinned model files, or invent
model_type from a repository name: recovery requires a unique built-in class.
"""
from __future__ import annotations

from .hf_config_v83 import load_hf_config, MissingModelTypeError


def is_missing_model_type_error(exc: BaseException) -> bool:
    """Match only the specific HF missing-config-identity failure."""
    return isinstance(exc, ValueError) and 'model_type' in str(exc).lower() and 'config.json' in str(exc).lower()


def _compat_config(repo, revision, local_files_only=False):
    try:
        return load_hf_config(repo, revision=revision,
                              trust_remote_code=False, local_files_only=local_files_only)
    except MissingModelTypeError as exc:
        raise MissingModelTypeError(
            f'{repo}@{revision}: Hugging Face configuration is missing a verifiable '
            'model_type. Check the source config.json and run '
            '`python -m patchlib.cli hf-diagnose --model <model>`; '
            'if it is a custom architecture, provide a trusted model factory '
            'rather than guessing a built-in model_type. Details: '+str(exc)) from exc


def load_hf_tokenizer(repo, *, revision='main', trust_remote_code=False,
                      local_files_only=False, config=None, **kwargs):
    """Retry missing-model_type tokenizer errors using an audited HF config.

    Only retry the specific ValueError; all other errors (missing tokenizer
    assets, gated auth, tokenization incompatibility) stay unchanged. A passed
    config permits deterministic newly-created models to load a tokenizer.
    """
    from transformers import AutoTokenizer
    opts=dict(revision=revision,trust_remote_code=trust_remote_code,
              local_files_only=local_files_only,**kwargs)
    if config is not None:
        return AutoTokenizer.from_pretrained(repo, config=config, **opts)
    try:
        return AutoTokenizer.from_pretrained(repo, **opts)
    except ValueError as exc:
        if not is_missing_model_type_error(exc) or trust_remote_code:
            raise
        config=_compat_config(repo, revision, local_files_only=local_files_only)
        return AutoTokenizer.from_pretrained(repo, config=config, **opts)


def load_hf_model(model_class, repo, *, revision='main', trust_remote_code=False,
                  local_files_only=False, **kwargs):
    """Retry missing-model_type AutoModel failures (e.g. separate RM/Critic)."""
    opts=dict(revision=revision,trust_remote_code=trust_remote_code,
              local_files_only=local_files_only,**kwargs)
    if 'config' in opts:
        return model_class.from_pretrained(repo, **opts)
    try:
        return model_class.from_pretrained(repo, **opts)
    except ValueError as exc:
        if not is_missing_model_type_error(exc) or trust_remote_code:
            raise
        opts['config']=_compat_config(repo, revision, local_files_only=local_files_only)
        return model_class.from_pretrained(repo, **opts)
