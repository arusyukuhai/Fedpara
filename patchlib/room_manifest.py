"""Immutable room descriptors; no remote code is executed or deserialized.

Only small JSON metadata transits signaling. Custom code is referred to by
SHA-256 + optional HTTPS URL; downloading requires explicit owner opt-in and
execution requires the existing offline container sandbox.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from .core import _canonical_json, PatchError

MAX_MANIFEST = 4096
HEX = re.compile(r'^[0-9a-f]{64}$')
ID = re.compile(r'^[A-Za-z0-9_.:/@-]{1,160}$')


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(data: dict) -> dict:
    if not isinstance(data, dict) or set(data) != {'version','model','dataset','code'} or data.get('version') != 1:
        raise PatchError('manifest must have version, model, dataset and code fields')
    model, ds, code = data['model'], data['dataset'], data['code']
    if not isinstance(model, dict) or set(model) != {'id','revision','description','architecture'}:
        raise PatchError('malformed model descriptor')
    if not isinstance(ds, dict) or set(ds) != {'name','description','license','privacy'}:
        raise PatchError('malformed dataset descriptor')
    if not isinstance(code, dict) or set(code) != {'sha256','source'}:
        raise PatchError('malformed code descriptor')
    for field in (model['id'],model['revision'],model['architecture']):
        if not isinstance(field, str) or not ID.fullmatch(field):
            raise PatchError('bad model id/revision/architecture')
    for field in (model['description'],ds['name'],ds['description'],ds['license'],ds['privacy']):
        if not isinstance(field, str) or len(field) > 240 or any(ord(c) < 32 for c in field):
            raise PatchError('bad dataset/model description')
    if ds['privacy'] not in ('local-only','public','licensed'):
        raise PatchError('dataset privacy must be local-only, public or licensed')
    digest, url=code['sha256'],code['source']
    if digest is not None and (not isinstance(digest,str) or not HEX.fullmatch(digest)):
        raise PatchError('invalid custom code digest')
    if not isinstance(url,str) or len(url)>512:
        raise PatchError('invalid custom code URI')
    if url:
        parsed=urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or not digest:
            raise PatchError('code URL must be HTTPS, hash-pinned, without credentials')
    raw=_canonical_json(data)
    if len(raw)>MAX_MANIFEST:raise PatchError('room manifest exceeds 4KiB')
    return data


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(_canonical_json(validate_manifest(manifest))).hexdigest()


def make_manifest(model: str, revision: str, dataset: str, *,
                  architecture: str = 'causal-lm', description: str = '',
                  dataset_description: str = '', license: str = 'unspecified',
                  privacy: str = 'local-only', code_path: Path | None = None,
                  code_url: str = '') -> dict:
    result={'version':1,
            'model':{'id':model,'revision':revision,'architecture':architecture,'description':description},
            'dataset':{'name':dataset,'description':dataset_description,'license':license,'privacy':privacy},
            'code':{'sha256':digest_file(code_path) if code_path else None,'source':code_url}}
    return validate_manifest(result)


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST + 256:raise PatchError('manifest too large')
    return validate_manifest(json.loads(path.read_text(encoding='utf-8')))


def verify_code(path: Path, manifest: dict):
    expected = validate_manifest(manifest)['code']['sha256']
    if not expected or digest_file(path)!=expected:
        raise PermissionError('custom code SHA-256 does not match room manifest')
    return True

# Version 2: executable *dataset references*, not human-readable descriptions.
# Kept separate from v1 to fail closed for older peers.
HF_REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')
DATA_PART = re.compile(r'^[A-Za-z0-9_./:+-]{1,120}$')
COLUMN = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]{0,79}$')
MAX_HF_SOURCES = 12


def validate_hf_sources(sources: list) -> list:
    """Validate immutable non-executable Hugging Face dataset descriptors.

    No credentials, expressions, arbitrary URLs, Python scripts or dataset
    transforms may appear in a descriptor shared with untrusted peers.
    """
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_HF_SOURCES:
        raise PatchError('Hugging Face dataset collection must contain 1..12 sources')
    normalized = []
    for entry in sources:
        if not isinstance(entry, dict) or set(entry) != {'repo', 'revision', 'config', 'split', 'columns', 'weight'}:
            raise PatchError('dataset source needs repo, revision, config, split, columns, weight')
        repo, revision, config, split = (entry[x] for x in ('repo', 'revision', 'config', 'split'))
        if not isinstance(repo, str) or not HF_REPO.fullmatch(repo):
            raise PatchError('dataset must be namespace/name on Hugging Face')
        # Commit hashes recommended. Branch names are allowed but may change.
        if not isinstance(revision, str) or not DATA_PART.fullmatch(revision) or '..' in revision:
            raise PatchError('invalid dataset revision')
        if config is not None and (not isinstance(config, str) or not DATA_PART.fullmatch(config)):
            raise PatchError('invalid dataset config')
        if not isinstance(split, str) or not DATA_PART.fullmatch(split) or '..' in split:
            raise PatchError('invalid dataset split')
        cols = entry['columns']
        if not isinstance(cols, dict) or set(cols) not in ({'text'}, {'input','target'}, {'text','label'}):
            raise PatchError('columns must map text, input+target, or text+label')
        if any(not isinstance(v, str) or not COLUMN.fullmatch(v) for v in cols.values()):
            raise PatchError('invalid dataset column name')
        weight = entry['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int,float)) or not 0 < weight <= 100:
            raise PatchError('weight must be between 0 and 100')
        # Reject NaN/Infinity through comparison and canonical JSON safety.
        normalized.append({'repo':repo,'revision':revision,'config':config,'split':split,
                           'columns':dict(cols),'weight':float(weight)})
    return normalized


def make_hf_manifest(model: str, revision: str, sources: list, *, architecture='causal-lm',
                     description='', seed=42, code_path=None, code_url='') -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31-1:
        raise PatchError('dataset seed must be an unsigned 31-bit integer')
    manifest={'version':2,
        'model':{'id':model,'revision':revision,'architecture':architecture,'description':description},
        'datasets':{'seed':seed,'sources':validate_hf_sources(sources)},
        'code':{'sha256':digest_file(code_path) if code_path else None,'source':code_url}}
    return validate_manifest(manifest)


_v1_validate_manifest = validate_manifest

def validate_manifest(data: dict) -> dict:
    if not isinstance(data, dict):
        raise PatchError('room manifest must be JSON object')
    if data.get('version') == 1:
        return _v1_validate_manifest(data)
    if data.get('version') != 2 or set(data) != {'version','model','datasets','code'}:
        raise PatchError('invalid room manifest version or fields')
    mdl, datasets, code = data['model'], data['datasets'], data['code']
    if not isinstance(mdl, dict) or set(mdl) != {'id','revision','architecture','description'}:
        raise PatchError('invalid model descriptor')
    for name in ('id','revision','architecture'):
        if not isinstance(mdl[name], str) or not ID.fullmatch(mdl[name]):
            raise PatchError('invalid model '+name)
    if not isinstance(mdl['description'],str) or len(mdl['description'])>240 or any(ord(c)<32 for c in mdl['description']):
        raise PatchError('invalid model description')
    if not isinstance(datasets, dict) or set(datasets) != {'seed','sources'}:
        raise PatchError('datasets must contain seed and sources')
    if type(datasets['seed']) is not int or not 0 <= datasets['seed'] <= 2**31-1:
        raise PatchError('bad dataset seed')
    validate_hf_sources(datasets['sources'])
    if not isinstance(code,dict) or set(code) != {'sha256','source'}:
        raise PatchError('invalid code descriptor')
    digest, url=code['sha256'],code['source']
    if digest is not None and (not isinstance(digest,str) or not HEX.fullmatch(digest)):
        raise PatchError('invalid custom code digest')
    if not isinstance(url,str) or len(url)>512:
        raise PatchError('invalid code URL')
    if url:
        parsed=urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or not digest:
            raise PatchError('custom code source must be hash-pinned HTTPS without credentials')
    if len(_canonical_json(data)) > MAX_MANIFEST:
        raise PatchError('room manifest exceeds 4KiB')
    return data
