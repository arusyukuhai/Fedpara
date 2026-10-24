"""Pinned random TinyStories-1M model genesis, shared by all live peers.

Never downloads checkpoint weights, executes remote code, or uses untrusted pickle.
The immutable .fstate content digest is also checked in room genesis.
"""
from __future__ import annotations

import torch

TINY_STORIES_ID = 'roneneldan/TinyStories-1M'
TINY_STORIES_REVISION = '77f1b168e219585646439073245fe87e56b3023e'
TINYSTORIES_DATA_REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
RANDOM_MANIFEST_DESCRIPTION = 'fedpara:random-tinystories-1m:v1'


def room_initialization(manifest):
    desc=manifest['model'].get('description','')
    if desc == RANDOM_MANIFEST_DESCRIPTION:
        if manifest['model']['id'] != TINY_STORIES_ID or manifest['model']['revision'] != TINY_STORIES_REVISION or manifest['model']['architecture'] != 'causal-lm':
            raise ValueError('random TinyStories room model must use the pinned official GPT-Neo model/revision')
        return 'random'
    if desc.startswith('fedpara:random-'):
        raise ValueError('unsupported pinned random model architecture')
    return 'pretrained'


def create_random_tinystories_model(model_id, revision, seed):
    if model_id != TINY_STORIES_ID or revision != TINY_STORIES_REVISION:
        raise ValueError('random model generation is supported only for pinned TinyStories-1M')
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError('initial_seed must be an unsigned 31-bit integer')
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id, revision=revision,
                                       trust_remote_code=False)
    if getattr(config, 'model_type', None) != 'gpt_neo':
        raise RuntimeError('TinyStories-1M config is not GPT-Neo; refusing to generate a different architecture')
    config.use_cache = False
    # Strict deterministic CPU genesis independent of training accelerator RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    print(f'MODEL RANDOM-GENESIS: {model_id} revision={revision} seed={seed} '
          f'architecture={config.model_type} pretrained_weights=NO', flush=True)
    return model, revision
