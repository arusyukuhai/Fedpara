"""Purpose-aware, metadata-only Hugging Face discovery for the Kagami desktop.

Candidate ranking is heuristic. Never equate search matching with a validated
schema, scalar value head, trained reward head, model license or loadability.
No remote repository Python is executed by this module.
"""
from __future__ import annotations

import re
import copy
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any

ROLES = ('pt', 'sft', 'ppo', 'reward', 'critic')
_SHA = re.compile(r'[a-fA-F0-9]{40}\Z')
# Role-specific seed searches. No hard-coded dataset/model allowlist.
SEEDS = {
    'pt': ('pretraining', 'fineweb', 'text'),
    'sft': ('sft', 'instruction', 'conversation'),
    'ppo': ('prompts', 'rlhf', 'preference'),
    'reward': ('reward', 'rm', 'preference'),
    'critic': ('value', 'critic', 'value-head'),
}
SIGNALS = {
    'pt': ('pretrain', 'fineweb', 'corpus', 'text', 'language-model', 'webtext', 'tinystories'),
    'sft': ('sft', 'instruction', 'conversation', 'chat', 'instruct', 'assistant', 'alpaca'),
    'ppo': ('prompt', 'rlhf', 'preference', 'feedback', 'question', 'ranking'),
    'reward': ('reward', 'preference', 'ranker', 'scorer', 'rm-', 'rm_'),
    'critic': ('value-head', 'value_head', 'value-model', 'critic', 'valuefunction', 'value-function'),
}


def _string_attr(obj: Any, name: str) -> str:
    val = getattr(obj, name, '')
    return val if isinstance(val, str) else ''


def _classify(role: str, item: Any) -> tuple[int, str]:
    name = _string_attr(item, 'id').lower()
    tags = ' '.join(str(s).lower() for s in (getattr(item, 'tags', None) or ()) if isinstance(s, str))
    haystack = name + ' ' + tags
    hits = [x for x in SIGNALS[role] if x in haystack]
    match = hits[0] if hits else ''
    score = len(hits) * 3 + (5 if any(s in name for s in SIGNALS[role]) else 0)
    if role == 'pt' and ('language-modeling' in tags or 'language_modeling' in tags):
        score += 3
        match = match or 'language-modeling tag'
    if role in ('sft','ppo') and 'question-answering' in tags:
        score += 1
    if role in ('reward','critic'):
        pipeline = _string_attr(item,'pipeline_tag')
        if pipeline in ('text-classification', 'token-classification'):
            score += 2
        if role == 'critic' and ('reward' in name and not match):
            # A reward model is not automatically a trained value critic.
            score -= 3
    if match:
        reason = f'Metadata hint: {match}; columns/head must be checked'
    else:
        reason = 'Task category only; inspect columns/head before training'
    return score, reason



class HubDiscoveryCache:
    """Thread-safe short-lived Hub metadata cache with in-flight de-duplication.

    Stores *only* public search metadata, not dataset contents, access tokens,
    model weights, or private user files. Failed requests are never cached.
    """
    def __init__(self, *, ttl=75, limit=24, clock=None):
        if ttl<=0 or limit<1:raise ValueError('invalid discovery cache limits')
        self.ttl=float(ttl)
        self.limit=int(limit)
        self.clock=clock or time.monotonic
        self.lock=threading.Lock()
        self.items=OrderedDict()
        self.inflight={}

    def get_or_load(self,key,loader):
        with self.lock:
            now=self.clock()
            old=self.items.get(key)
            if old is not None:
                if old[0]>now:
                    self.items.move_to_end(key)
                    return copy.deepcopy(old[1])
                del self.items[key]
            pending=self.inflight.get(key)
            leader=pending is None
            if leader:
                pending=Future()
                self.inflight[key]=pending
        if not leader:
            return copy.deepcopy(pending.result())
        try:
            result=loader()
            with self.lock:
                self.items[key]=(self.clock()+self.ttl,copy.deepcopy(result))
                self.items.move_to_end(key)
                while len(self.items)>self.limit:self.items.popitem(last=False)
                self.inflight.pop(key,None)
            pending.set_result(result)
            return copy.deepcopy(result)
        except BaseException as exc:
            with self.lock:
                self.inflight.pop(key,None)
            pending.set_exception(exc)
            raise

def discover_hub_assets(payload: dict, *, api: Any = None, cache: HubDiscoveryCache | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError('Hub browser request must be an object')
    role = payload.get('role')
    if role not in ROLES:
        raise ValueError('Invalid Hugging Face purpose')
    query = payload.get('query', '')
    scope = payload.get('scope', 'recommended')
    if not isinstance(query, str) or len(query.strip()) > 100 or not isinstance(scope, str) or scope not in ('recommended', 'broader'):
        raise ValueError('Invalid Hub search query or filter')
    query = query.strip()
    if cache is not None:
        # Keep single-flight tracking out of the Hub's potentially slow network
        # operations. Cache is per-controller, so a private API credential cannot
        # accidentally influence the metadata shown by a different client.
        return cache.get_or_load((role,scope,query),
            lambda: discover_hub_assets(payload,api=api,cache=None))
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub to browse models and datasets') from exc
        api = HfApi()

    is_dataset = role in ('pt', 'sft', 'ppo')
    searches = (query,) if query else (SEEDS[role] if scope == 'recommended' else ('',))
    seen: dict[str, tuple[int, dict]] = {}
    errors: list[str] = []
    for search in searches:
        opts = {'sort': 'downloads', 'limit': 20, 'search': search or None}
        if scope == 'recommended':
            if role == 'pt':
                opts['filter'] = 'task_ids:language-modeling'
            elif not is_dataset:
                opts['pipeline_tag'] = 'text-classification'
        try:
            results = api.list_datasets(**opts) if is_dataset else api.list_models(**opts)
            for item in results:
                repo = _string_attr(item, 'id')
                # Defensive output validation; never inject malformed IDs into the GUI.
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
                    continue
                score, reason = _classify(role, item)
                if scope == 'recommended' and role in ('sft', 'ppo', 'reward', 'critic') and not query and score < 3:
                    continue
                sha = _string_attr(item, 'sha')
                downloads = getattr(item, 'downloads', 0)
                if not isinstance(downloads, int) or downloads < 0:
                    downloads = 0
                row = {
                    'id': repo, 'kind': 'dataset' if is_dataset else 'model',
                    'revision': sha.lower() if _SHA.fullmatch(sha) else '',
                    'pipeline': _string_attr(item, 'pipeline_tag') if not is_dataset else '',
                    'downloads': downloads, 'gated': bool(getattr(item, 'gated', False)),
                    'reason': reason, 'fit': 'candidate',
                    'requires_inspection': True,
                }
                old = seen.get(repo)
                if old is None or score > old[0]:
                    seen[repo] = score, row
        except Exception as exc:
            # A partial result list remains useful, but every unsuccessful query
            # is disclosed. Do not silently disguise a network failure as no matches.
            errors.append(f'{type(exc).__name__} during {search or "default"} search')
    ranked = sorted(seen.values(), key=lambda it: (-it[0], -it[1]['downloads'], it[1]['id']))
    if not ranked and errors:
        raise RuntimeError('Hugging Face discovery unavailable: ' + '; '.join(errors))
    return {'role': role, 'scope': scope, 'query': query, 'candidates': [r for _, r in ranked[:24]],
            'notice': 'Metadata suggestions only. Verify dataset columns and model head before use.',
            'partial_errors': errors[:3]}
