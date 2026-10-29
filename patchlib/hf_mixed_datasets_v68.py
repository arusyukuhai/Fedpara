"""Immutable-revision, bounded-memory Hugging Face iterable sources for SFT/PPO.

All HF loading happens within the local trainer; URLs, dataset scripts and
untrusted Python factories are not accepted as source IDs.  Dataset source
settings are local and intentionally absent from the room-wide objective policy.
"""
from __future__ import annotations
import re
from collections.abc import Mapping

_REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
_SHA = re.compile(r'^[0-9a-fA-F]{40}$')
_COMPONENT = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$')

class HFMixedDatasetError(ValueError):
    pass


def source_spec(args, role):
    """Validate and return pinned source for role, or None for local JSONL."""
    if role not in ('sft','ppo'):
        raise HFMixedDatasetError('unknown dataset role')
    prefix=role+'_hf_'
    repo=str(getattr(args,prefix+'repo','') or '').strip()
    local=getattr(args,role+'_jsonl',None)
    if not repo:
        if any(str(getattr(args,prefix+field,'') or '').strip() for field in ('revision','config')):
            raise HFMixedDatasetError(f'{role}: Hub revision/config provided without repository')
        return None
    if local:
        raise HFMixedDatasetError(f'{role}: choose local JSONL OR a Hub dataset, not both')
    sha=str(getattr(args,prefix+'revision','') or '').strip()
    config=str(getattr(args,prefix+'config','') or '').strip()
    split=str(getattr(args,prefix+'split','train') or '').strip()
    prompt=str(getattr(args,prefix+'prompt_column','prompt') or '').strip()
    other_name='response' if role=='sft' else 'reference'
    other=str(getattr(args,prefix+other_name+'_column',other_name) or '').strip()
    if not _REPO.fullmatch(repo):raise HFMixedDatasetError(f'{role}: Hub repo must be organization/name')
    if not _SHA.fullmatch(sha):raise HFMixedDatasetError(f'{role}: pin a 40-character immutable Hub commit SHA')
    if config and not _COMPONENT.fullmatch(config):raise HFMixedDatasetError(f'{role}: invalid dataset config')
    if not _COMPONENT.fullmatch(split):raise HFMixedDatasetError(f'{role}: split must be a single named split')
    for value in (prompt,other):
        if not _COMPONENT.fullmatch(value):raise HFMixedDatasetError(f'{role}: columns must be simple names or dotted fields')
    if role=='sft' and prompt==other:
        raise HFMixedDatasetError('SFT prompt and response columns must differ')
    return {'repo':repo,'revision':sha.lower(),'config':config or None,'split':split,
            'prompt_column':prompt,other_name+'_column':other}


def source_selected(args, role):
    return bool(getattr(args,role+'_jsonl',None) or getattr(args,role+'_hf_repo',''))


def _field(row,name):
    item=row
    for key in name.split('.'):
        if not isinstance(item,Mapping) or key not in item:
            raise HFMixedDatasetError(f'Missing Hub column {name!r} (check config, split and field mapping)')
        item=item[key]
    return item


def _text(value, field):
    # Chat turns in a single column can be projected to a plain text prompt/answer.
    # No hidden Jinja or arbitrary Python eval is performed.
    if isinstance(value,list) and value and all(isinstance(item,Mapping) for item in value):
        result=[]
        for item in value:
            content=item.get('content',item.get('text'))
            if not isinstance(content,str):
                raise HFMixedDatasetError(f'{field}: chat message needs string content')
            role=item.get('role','user')
            if role not in ('system','user','assistant','tool'):
                raise HFMixedDatasetError(f'{field}: unsupported chat role')
            result.append(f'{role}: {content}')
        value='\n'.join(result)
    if not isinstance(value,str) or not value or len(value)>16384:
        raise HFMixedDatasetError(f'{field}: expected nonempty string (or chat turns) <=16Ki chars')
    return value


class HFRowCycle:
    """Stream Parquet/Arrow-backed Hub rows without staging an entire corpus."""
    def __init__(self, spec, role, *, dataset_factory=None):
        if role not in ('sft','ppo'):raise HFMixedDatasetError('unknown role')
        self.spec=dict(spec)
        self.role=role
        if dataset_factory is None:
            try:from datasets import load_dataset
            except ImportError as exc:
                raise HFMixedDatasetError('Install huggingface datasets for SFT/PPO Hub streaming') from exc
            dataset_factory=load_dataset
        self._factory=dataset_factory
        self._open()

    def _open(self):
        s=self.spec
        # HF Datasets no longer runs remote dataset loading scripts. We never
        # set trust_remote_code=True and do not execute Hub-provided Python.
        try:
            self._dataset=self._factory(s['repo'],name=s['config'],split=s['split'],
                                        revision=s['revision'],streaming=True)
            self._rows=iter(self._dataset)
        except Exception as exc:
            raise HFMixedDatasetError(f"Cannot stream {self.role} Hub dataset {s['repo']} at pinned SHA: {exc}") from exc

    def next(self):
        for attempt in range(2):
            try:
                row=next(self._rows)
            except StopIteration:
                if attempt:break
                # Most HF streaming datasets are re-iterable. Reusing the pinned
                # dataset object avoids a new Hub metadata/config load every epoch.
                restarted=iter(self._dataset)
                if restarted is self._rows:
                    # Preserve compatibility with genuinely single-pass sources.
                    self._open()
                else:
                    self._rows=restarted
                continue
            except Exception as exc:
                raise HFMixedDatasetError(f'{self.role}: Hub streaming failed, check login/network/dataset: {exc}') from exc
            if not isinstance(row,Mapping):raise HFMixedDatasetError('Hub dataset must yield dict rows')
            prompt=_text(_field(row,self.spec['prompt_column']),'prompt')
            result={'prompt':prompt}
            other='response' if self.role=='sft' else 'reference'
            try:
                value=_field(row,self.spec[other+'_column'])
            except HFMixedDatasetError:
                if self.role=='ppo':return result  # external RM; ground-truth caller checks reference
                raise
            result[other]=_text(value,other)
            return result
        raise HFMixedDatasetError(f'{self.role}: selected Hub split is empty')

    def close(self):
        self._rows=iter(())
        self._dataset=None
