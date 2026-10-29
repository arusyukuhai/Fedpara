"""Content-addressed no-copy epoch checkpoint installation."""
import hashlib
import os
from pathlib import Path

import pytest
import torch

from patchlib.core import PatchError,UniversalAdapter
from patchlib.shards import (export_state_sharded,read_manifest,
                             semantic_shard_sha256,shard_map,_blob_path)
from patchlib.epoch_local_stage_v53 import stage_verified_local_view


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(4,4)


def build(tmp_path):
    mdl=Net();adapter=UniversalAdapter(mdl,'net','r1',kind='loha',rank=2)
    store=tmp_path/'objects';incoming=tmp_path/'incoming';incoming.mkdir()
    manifest=incoming/'temp.fmanifest'
    export_state_sharded(mdl,store,manifest,model_id='net',revision='r1',
                         peer_id='alice',seq=0,adapter=adapter,shard_bytes=65536)
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.rename(incoming/(digest+'.fmanifest'))
    proposal={'manifest':digest,'semantic':semantic_shard_sha256(read_manifest(incoming/(digest+'.fmanifest')))}
    return store,incoming,proposal


def test_verified_view_is_hardlinked_not_copied(tmp_path):
    store,incoming,proposal=build(tmp_path)
    a=stage_verified_local_view(proposal,root=tmp_path/'epoch',incoming=incoming,objects=store)
    b=stage_verified_local_view(proposal,root=tmp_path/'epoch',incoming=incoming,objects=store)
    assert a.manifest==b.manifest
    meta=read_manifest(a.manifest)
    for digest in shard_map(meta):
        assert os.stat(_blob_path(store,digest)).st_ino==os.stat(_blob_path(a.store,digest)).st_ino


def test_certified_view_rejects_broken_shard(tmp_path):
    store,incoming,proposal=build(tmp_path)
    meta=read_manifest(incoming/(proposal['manifest']+'.fmanifest'))
    first=next(iter(shard_map(meta)))
    _blob_path(store,first).write_bytes(b'corrupt')
    with pytest.raises(PatchError):
        stage_verified_local_view(proposal,root=tmp_path/'epoch',incoming=incoming,objects=store)


def test_no_candidate_created_when_manifest_missing(tmp_path):
    store,incoming,proposal=build(tmp_path)
    (incoming/(proposal['manifest']+'.fmanifest')).unlink()
    with pytest.raises(FileNotFoundError):
        stage_verified_local_view(proposal,root=tmp_path/'epoch',incoming=incoming,objects=store)
    assert not (tmp_path/'epoch'/'checkpoints').exists()
