"""Audited Hugging Face configuration identity handling (Kagami v83).

A peer must never guess the architecture of a model from its weights or its
repository name.  The only supported legacy recovery is an unambiguous,
recognized *built-in* ``architectures`` declaration in config.json.  We never
execute remote model code or modify a cached/remote checkpoint on load.
"""
from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

# Deliberately small, audited list, not arbitrary attribute lookup/imports.
# The keys here are HF built-in model class names, not descriptions or model IDs.
_BUILTIN_ARCHITECTURES = {
    'GPT2Model': 'gpt2', 'GPT2LMHeadModel': 'gpt2',
    'GPT2ForSequenceClassification': 'gpt2',
    'GPTNeoModel': 'gpt_neo', 'GPTNeoForCausalLM': 'gpt_neo',
    'GPTNeoXModel': 'gpt_neox', 'GPTNeoXForCausalLM': 'gpt_neox',
    'LlamaModel': 'llama', 'LlamaForCausalLM': 'llama',
    'LlamaForSequenceClassification': 'llama',
    'MistralModel': 'mistral', 'MistralForCausalLM': 'mistral',
    'Qwen2Model': 'qwen2', 'Qwen2ForCausalLM': 'qwen2',
    'GemmaModel': 'gemma', 'GemmaForCausalLM': 'gemma',
    'Gemma2Model': 'gemma2', 'Gemma2ForCausalLM': 'gemma2',
    'Phi3Model': 'phi3', 'Phi3ForCausalLM': 'phi3',
    'OPTModel': 'opt', 'OPTForCausalLM': 'opt',
    'BloomModel': 'bloom', 'BloomForCausalLM': 'bloom',
    'BertModel': 'bert', 'BertForMaskedLM': 'bert',
    'BertForSequenceClassification': 'bert',
    'T5Model': 't5', 'T5ForConditionalGeneration': 't5',
    'BartModel': 'bart', 'BartForConditionalGeneration': 'bart',
}


class MissingModelTypeError(ValueError):
    """A repository config has no verifiable architecture identity."""


def _builtin_architecture_candidates():
    """Inspect ONLY installed Transformers built-in auto-model metadata.

    The short v83 allowlist could reject valid Qwen3, recent Gemma, ViT, etc.
    ``*_MAPPING_NAMES`` are built-in class-name strings, never imported model
    implementations and never arbitrary config.auto_map remote executable code.
    A class mapped to multiple model types is deliberately ambiguous.
    """
    candidates={name:{kind} for name,kind in _BUILTIN_ARCHITECTURES.items()}
    try:
        from transformers.models.auto import modeling_auto
    except (ImportError,AttributeError):
        return candidates
    for field in dir(modeling_auto):
        if not (field.startswith('MODEL_') and field.endswith('_MAPPING_NAMES')):
            continue
        records=getattr(modeling_auto,field,None)
        if not isinstance(records,Mapping):
            continue
        for kind,names in records.items():
            if not isinstance(kind,str):
                continue
            if isinstance(names,str):
                names=(names,)
            if not isinstance(names,(tuple,list)):
                continue
            for name in names:
                if isinstance(name,str) and name.isidentifier():
                    candidates.setdefault(name,set()).add(kind)
    return candidates


def checked_model_type(data: dict, *, source: str = 'config.json') -> str:
    """Return declared model_type, or infer it only from unambiguous HF classes.

    Never infer from path/name (Transformers' historic fallback does that) and
    never choose the first architecture when multiple candidates conflict.
    """
    if not isinstance(data, dict):
        raise MissingModelTypeError(f'{source}: config.json must contain a JSON object')
    supplied = data.get('model_type')
    if isinstance(supplied, str) and supplied.strip():
        return supplied
    if supplied is not None:
        raise MissingModelTypeError(f'{source}: model_type must be a nonempty string')
    names = data.get('architectures', [])
    if isinstance(names, list) and names and all(isinstance(x, str) for x in names):
        candidates=_builtin_architecture_candidates()
        inferred=set()
        for class_name in names:
            found=candidates.get(class_name)
            if not found:
                break
            inferred.update(found)
        else:
            if len(inferred)==1:
                return next(iter(inferred))
    raise MissingModelTypeError(
        f'{source}: missing model_type in config.json and no unambiguous recognized '
        'built-in architectures. Set model_type to the actual Hugging Face '
        'architecture (e.g. gpt2, gpt_neo, llama, qwen2); never guess from the '
        'repository name. Custom architectures require a separately audited loader.'
    )


def _read_checkpoint_config(model_id: str, revision: str, *, local_files_only: bool) -> dict:
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local / 'config.json'
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(model_id, 'config.json', revision=revision,
                                    local_files_only=local_files_only))
    if not path.is_file():
        raise FileNotFoundError(f'No config.json in {model_id}')
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('config.json is over the 16 MiB safety limit')
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise MissingModelTypeError(f'{model_id}: config.json must be a JSON object')
    return data


def load_hf_config(model_id, *, revision='main', trust_remote_code=False,
                   local_files_only=False):
    """Load model config; recover only missing model_type from built-in metadata.

    Custom remote code is never executed in this compatibility path. For normal
    configs the original AutoConfig behavior (incl. remote exceptions) remains.
    """
    from transformers import AutoConfig
    if trust_remote_code:
        # Custom remote code remains opt-in and must run only in the isolated,
        # offline environment validated by cli._untrusted_guard.
        return AutoConfig.from_pretrained(model_id, revision=revision,
            trust_remote_code=True, local_files_only=local_files_only)

    # v85: check the real config.json FIRST. Transformers may otherwise
    # silently guess a different model type from the repository's NAME, or
    # error before Kagami's later loaders (e.g. AutoTokenizer) are reached.
    # Never infer architecture identity from a repository name or weights.
    try:
        data = _read_checkpoint_config(str(model_id), str(revision),
                                       local_files_only=local_files_only)
    except (OSError, ValueError, ImportError):
        # Preserve the original HF diagnostics for missing, gated or invalid
        # configs; do not invent a config when config.json is absent.
        return AutoConfig.from_pretrained(model_id, revision=revision,
            trust_remote_code=False, local_files_only=local_files_only)
    if data.get('model_type') is not None:
        # An explicit type has precedence; unknown custom types are NOT
        # silently replaced by a similarly named built-in architecture.
        return AutoConfig.from_pretrained(model_id, revision=revision,
            trust_remote_code=False, local_files_only=local_files_only)

    model_type=checked_model_type(data,source=str(model_id))
    try:
        return AutoConfig.for_model(model_type, **{k: v for k,v in data.items()
                                                  if k not in ('model_type', 'auto_map')})
    except (ValueError, TypeError, KeyError) as exc:
        raise MissingModelTypeError(
            f'{model_id}: inferred built-in model_type={model_type!r} from '
            'architectures, but installed Transformers cannot construct it') from exc


def save_hf_config_with_model_type(config, destination: str | Path) -> str:
    """Save HF config with an explicit model_type key (atomic JSON correction).

    Never silently replace a conflicting declared type. Called only for a
    locally trusted config instance, not for untrusted remote JSON.
    """
    model_type = getattr(config, 'model_type', None)
    if not isinstance(model_type, str) or not model_type.strip():
        raise MissingModelTypeError('Cannot export Hugging Face config: model_type is missing')
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(folder)
    path = folder / 'config.json'
    if not path.is_file():
        raise MissingModelTypeError('save_pretrained did not create config.json')
    data = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(data, dict):
        raise MissingModelTypeError('Exported config.json is not a JSON object')
    if 'model_type' in data and data['model_type'] != model_type:
        raise MissingModelTypeError('Exported config.json model_type conflicts with config class')
    if data.get('model_type') == model_type:
        return model_type
    data['model_type'] = model_type
    temp = folder / ('.config-' + secrets.token_hex(8) + '.tmp')
    try:
        with temp.open('x', encoding='utf8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return model_type
