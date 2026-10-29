"""Kagami portable Transformers architecture designs.

Untrusted peer inputs are DATA, never Python to execute. The optional editor
accepts a deliberately restricted, statically parsed Python build_config()
whose single return constructs a built-in Hugging Face PretrainedConfig.
This is *not* general Python code execution or a security sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re

DESCRIPTION = {'weights': '', 'architecture': 'kagami:architecture:v1',
               'code': 'kagami:python-config:v1'}
# Only established Transformers config classes with AutoModel support; no dynamic imports.
CONFIG_CLASSES = {
    'GPT2Config': 'gpt2', 'GPTNeoConfig': 'gpt_neo', 'GPTNeoXConfig': 'gpt_neox',
    'LlamaConfig': 'llama', 'MistralConfig': 'mistral', 'Qwen2Config': 'qwen2',
    'GemmaConfig': 'gemma', 'Gemma2Config': 'gemma2', 'Phi3Config': 'phi3',
    'OPTConfig': 'opt', 'BloomConfig': 'bloom', 'BertConfig': 'bert',
    'T5Config': 't5', 'BartConfig': 'bart',
}
MAX_KWARGS = 70
MAX_DESIGN_BYTES = 2500
_HF_REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
_HF_REVISION = re.compile(r'^[A-Za-z0-9_./:+-]{1,120}$')


def _literal(node):
    # ast.literal_eval never executes Python; disallow containers beyond flat tuples/lists.
    value = ast.literal_eval(node)
    if type(value) in (int, float, bool, str) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError('non-finite config literal')
        if isinstance(value, str) and len(value) > 120:
            raise ValueError('config strings too long')
        return value
    if isinstance(value, (tuple, list)) and len(value) <= 64:
        return [_simple(x) for x in value]
    raise ValueError('only literal numbers, booleans, short strings, lists and tuples are supported')


def _simple(value):
    if type(value) in (int, float, bool, str) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError('non-finite number')
        return value
    raise ValueError('nested config values are not supported')


def extract_config(source: str) -> dict:
    """Parse ``def build_config(): return transformers.GPT2Config(...)`` safely.

    No user code, imports or function calls execute on the GUI or training peer.
    """
    if not isinstance(source, str) or not 0 < len(source.encode('utf8')) <= 256*1024:
        raise ValueError('Python source must be 1..256KiB')
    tree = ast.parse(source, filename='architecture.py')
    # Reject pathological ASTs before walking user-provided source. No
    # interpreter or import ever runs, but excessive nesting is still a DoS.
    if sum(1 for _ in ast.walk(tree)) > 2048:
        raise ValueError('Python configuration AST is too large')
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.ImportFrom) else
                     [a.name for a in node.names])
            if isinstance(node, ast.ImportFrom):
                if node.module != 'transformers' or not all(n in CONFIG_CLASSES for n in names):
                    raise ValueError('only built-in Transformers config imports are allowed')
            elif names != ['transformers'] or any(a.asname for a in node.names):
                raise ValueError('only import transformers is allowed')
        elif isinstance(node, ast.FunctionDef) and node.name == 'build_config' and fn is None:
            if node.decorator_list or node.args.args or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.posonlyargs:
                raise ValueError('build_config must not take arguments or decorators')
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
                raise ValueError('build_config must contain exactly one return expression')
            fn = node
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass  # module docstring
        else:
            raise ValueError('only import statements and a literal build_config factory are allowed')
    if fn is None or not isinstance(fn.body[0].value, ast.Call):
        raise ValueError('define build_config() returning a Transformers Config')
    call = fn.body[0].value
    cls = call.func
    if isinstance(cls, ast.Attribute) and isinstance(cls.value, ast.Name) and cls.value.id == 'transformers':
        config_class = cls.attr
    elif isinstance(cls, ast.Name):
        config_class = cls.id
    else:
        raise ValueError('unsupported config factory')
    if config_class not in CONFIG_CLASSES or call.args or len(call.keywords) > MAX_KWARGS:
        raise ValueError('only supported Config(...) constructors with keyword literals are allowed')
    kwargs = {}
    for kw in call.keywords:
        if kw.arg is None or kw.arg.startswith('_') or kw.arg in kwargs or len(kw.arg)>70 or not kw.arg.isidentifier():
            raise ValueError('invalid config argument name')
        value=_literal(kw.value)
        if type(value) is int and not -100_000_000 <= value <= 100_000_000:
            raise ValueError('unbounded model integer configuration')
        kwargs[kw.arg] = value
    result = {'mode': 'code', 'model_type': CONFIG_CLASSES[config_class], 'kwargs': kwargs}
    validate_design(result)
    return result


def validate_design(design) -> dict:
    if not isinstance(design, dict) or not 1 <= len(json.dumps(design, sort_keys=True, allow_nan=False)) <= MAX_DESIGN_BYTES:
        raise ValueError('architecture specification exceeds portable manifest budget')
    mode = design.get('mode')
    if mode == 'architecture':
        if set(design) != {'mode'}:
            raise ValueError('architecture copy has no custom parameters')
    elif mode == 'code':
        if set(design) not in ({'mode','model_type','kwargs'}, {'mode','model_type','kwargs','tokenizer'}) or design['model_type'] not in CONFIG_CLASSES.values():
            raise ValueError('unsupported Python config model type')
        if 'tokenizer' in design:
            tokenizer=design['tokenizer']
            if (not isinstance(tokenizer,dict) or set(tokenizer)!={'id','revision'} or
                not isinstance(tokenizer['id'],str) or not _HF_REPO.fullmatch(tokenizer['id']) or
                not isinstance(tokenizer['revision'],str) or not _HF_REVISION.fullmatch(tokenizer['revision']) or
                '..' in tokenizer['revision']):
                raise ValueError('invalid pinned tokenizer descriptor')
        obj = design['kwargs']
        if not isinstance(obj,dict) or len(obj)>MAX_KWARGS:
            raise ValueError('too many config parameters')
        for name,value in obj.items():
            if not isinstance(name,str) or not name.isidentifier() or name.startswith('_') or len(name)>70:
                raise ValueError('invalid configuration field')
            if isinstance(value,list):
                if len(value)>64:
                    raise ValueError('list parameter too large')
                for part in value: _simple(part)
            else:
                _simple(value)
                if type(value) is int and not -100_000_000 <= value <= 100_000_000:
                    raise ValueError('unbounded model integer configuration')
    else:
        raise ValueError('unknown architecture creation mode')
    return design


def create_config(manifest):
    """Build trusted Transformers config from immutable JSON, without code imports."""
    from transformers import AutoConfig
    from .hf_config_v83 import load_hf_config
    design = validate_design(manifest['design'])
    if design['mode'] == 'architecture':
        return load_hf_config(manifest['model']['id'],
                        revision=manifest['model']['revision'], trust_remote_code=False)
    return AutoConfig.for_model(design['model_type'], **design['kwargs'])


def tokenizer_identity(manifest):
    descriptor=manifest.get('design',{}).get('tokenizer')
    if descriptor:
        validate_design(manifest['design'])
        return descriptor['id'],descriptor['revision']
    return manifest['model']['id'],manifest['model']['revision']


def build_random_model(manifest, task='causal-lm'):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
    cls = {'causal-lm':AutoModelForCausalLM,'masked-lm':AutoModelForMaskedLM,
           'seq2seq-lm':AutoModelForSeq2SeqLM,
           'sequence-classification':AutoModelForSequenceClassification}[task]
    config = create_config(manifest)
    config.use_cache = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(manifest['datasets']['seed'])
        model=cls.from_config(config, trust_remote_code=False)
    return model, manifest['model']['revision']


def default_python_template():
    return ('from transformers import GPT2Config\n\n'
            'def build_config():\n'
            '    return GPT2Config(\n'
            '        vocab_size=50257, n_positions=512, n_embd=256,\n'
            '        n_layer=6, n_head=8\n'
            '    )\n')
