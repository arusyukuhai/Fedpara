from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
import pytest
import torch

from patchlib.core import PatchError, _canonical_json
from patchlib.elastic import EventJournal
from patchlib.roomid import auto_identity, create_room_id
from patchlib.room_manifest import make_hf_manifest
from patchlib.shards import export_state_sharded, read_manifest, verify_shards, semantic_shard_sha256
from patchlib.generation_rollover import prepare, _check_immutable_old_state


def fixture_state(tmp_path):
    journal_path=tmp_path/'old.sqlite3'
    peer, sign, _, _=auto_identity(Path(str(journal_path)+'.identity.pem'))
    journal=EventJournal(journal_path,'a'*64,max_bytes=0)
    journal.db.execute("INSERT OR REPLACE INTO meta(name,value) VALUES('owner',?)",(peer,))
    journal.db.commit()
    try:
        journal.append(peer,1,b'wire111')
        journal.append(peer,2,b'wire222')
        count=journal.count()
        sha=hashlib.sha256()
        for item in journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid'):
            sha.update(_canonical_json(list(item))+b'\n')
        store=tmp_path/'objects-store'
        manifest=tmp_path/f'elastic-{count:09d}.fmanifest'
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,3))
        export_state_sharded(model,store,manifest,model_id='myorg/my-model',revision='fstate-sha256:old-base',peer_id=peer,seq=count,shard_bytes=65536)
        payload={'protocol':'fedpara-elastic-checkpoint-v1','origin':peer,'genesis':'a'*64,
                 'event_count':count,'event_inventory_sha256':sha.hexdigest(),
                 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
        raw=_canonical_json(payload)
        payload['signature']=sign(len(raw),hashlib.sha256(raw).hexdigest())
        desc=manifest.with_suffix('.checkpoint.json')
        desc.write_text(json.dumps(payload))
        room=make_hf_manifest('myorg/my-model','main',[{'repo':'myorg/my-data','revision':'main',
                         'config':None,'split':'train','columns':{'text':'text'},'weight':1.0}])
        room_file=tmp_path/'room-manifest.json'
        room_file.write_text(json.dumps(room))
        return {'journal':journal_path,'checkpoint':manifest,'store':store,
                'room_manifest':room_file,'old_room_id':create_room_id(),
                'destination':tmp_path/'next-generation'}
    finally:
        journal.close()


def test_rollover_prepares_verified_new_base_without_deleting_old_log(tmp_path):
    args=fixture_state(tmp_path)
    oldlog=Path(str(args['journal'])+'.fplog')
    old_size=oldlog.stat().st_size
    result=prepare(**args)
    assert result['old_event_count']==2
    assert result['status']=='PREPARED_NOT_ACTIVATED'
    assert result['old_room_id']!=result['new_room_id']
    assert oldlog.stat().st_size==old_size
    assert args['journal'].exists()
    new=args['destination']
    base=read_manifest(new/'base.fmanifest')
    assert base['model_id']=='myorg/my-model' and base['revision']=='main'
    verify_shards(base,new/'store')
    assert result['base_semantic_sha256']==semantic_shard_sha256(base)
    assert (new/'generation.json').is_file()
    with pytest.raises(PatchError,match='overwrite'):
        prepare(**args)


def test_stale_snapshot_does_not_allow_rollover(tmp_path):
    args=fixture_state(tmp_path)
    j=EventJournal(args['journal'],'a'*64,max_bytes=0)
    try:j.append(j.db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()[0],3,b'wire333')
    finally:j.close()
    with pytest.raises(PatchError,match='ALL committed events'):
        prepare(**args)
    assert not args['destination'].exists()


def test_checkpoint_signature_tampering_fails_closed(tmp_path):
    args=fixture_state(tmp_path)
    desc=args['checkpoint'].with_suffix('.checkpoint.json')
    d=json.loads(desc.read_text());d['signature']='AA'*43+'=='
    desc.write_text(json.dumps(d))
    with pytest.raises(PatchError,match='signature'):
        prepare(**args)
    assert not args['destination'].exists()


def test_checkpoint_missing_shard_is_rejected(tmp_path):
    args=fixture_state(tmp_path)
    from patchlib.shards import _blob_path
    old=read_manifest(args['checkpoint'])
    digest=old['tensors'][0]['shards'][0]['sha256']
    _blob_path(args['store'],digest).unlink()
    with pytest.raises(PatchError,match='shards'):
        prepare(**args)


def test_live_trainer_lock_blocks_rollover(tmp_path):
    import fcntl, os
    args=fixture_state(tmp_path)
    fd=os.open(str(args['journal'])+'.train.lock', os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(PatchError,match='stop the trainer'):
            prepare(**args)
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def test_retire_requires_real_new_generation_progress_and_confirmation(tmp_path):
    from patchlib.generation_rollover import retire_old_history
    args=fixture_state(tmp_path)
    info=prepare(**args)
    newer=tmp_path/'fresh'/'events.sqlite3'
    fresh=EventJournal(newer,'b'*64,max_bytes=0)
    fresh.close()
    params=dict(journal=args['journal'],checkpoint=args['checkpoint'],store=args['store'],
                generation=args['destination'],new_journal=newer,confirmation=info['new_room_id'])
    with pytest.raises(PatchError,match='committed'):
        retire_old_history(**params)
    fresh=EventJournal(newer,'b'*64,max_bytes=0)
    try:fresh.append('alice',1,b'new-generation-durable-update')
    finally:fresh.close()
    with pytest.raises(PatchError,match='confirmation'):
        retire_old_history(**dict(params,confirmation='invalid'))
    oldlog=Path(str(args['journal'])+'.fplog')
    assert oldlog.is_file()
    result=retire_old_history(**params)
    assert result['retired'] and result['old_event_count']==2
    assert not args['journal'].exists() and not oldlog.exists()
    assert (args['destination']/'retirement.json').exists()
    assert (args['destination']/'base.fmanifest').exists()
    with pytest.raises(PatchError):retire_old_history(**params)


def test_seal_stopped_gui_checkpoint_and_prepare(tmp_path):
    from patchlib.generation_rollover import seal_stopped_checkpoint
    args=fixture_state(tmp_path)
    args['checkpoint'].with_suffix('.checkpoint.json').unlink()
    sealed=seal_stopped_checkpoint(journal=args['journal'],checkpoint=args['checkpoint'],store=args['store'])
    assert sealed['events']==2
    assert Path(sealed['descriptor']).exists()
    assert seal_stopped_checkpoint(journal=args['journal'],checkpoint=args['checkpoint'],store=args['store'])==sealed
    result=prepare(**args)
    assert result['old_event_count']==2


def test_seal_rejects_snapshot_not_at_journal_tip(tmp_path):
    from patchlib.generation_rollover import seal_stopped_checkpoint
    args=fixture_state(tmp_path)
    args['checkpoint'].with_suffix('.checkpoint.json').unlink()
    j=EventJournal(args['journal'],'a'*64,max_bytes=0)
    try:
        owner=j.db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()[0]
        j.append(owner,3,b'unsnapshotted')
    finally:j.close()
    with pytest.raises(PatchError,match='final committed event boundary'):
        seal_stopped_checkpoint(journal=args['journal'],checkpoint=args['checkpoint'],store=args['store'])


def test_import_generation_into_gui_and_refuse_duplicate(tmp_path):
    from patchlib.generation_rollover import import_gui_generation
    from patchlib.room_manifest import load_manifest
    args=fixture_state(tmp_path)
    info=prepare(**args)
    work=tmp_path/'gui_data'
    result=import_gui_generation(generation=args['destination'],workdir=work)
    assert result['room']==info['new_room_id']
    assert load_manifest(Path(result['path']))['model']['id']=='myorg/my-model'
    assert (Path(result['path']).parent/'generation-base.json').is_file()
    with pytest.raises(PatchError,match='already exists'):
        import_gui_generation(generation=args['destination'],workdir=work)


def test_corrupt_generation_blocks_import(tmp_path):
    from patchlib.generation_rollover import import_gui_generation
    args=fixture_state(tmp_path)
    prepare(**args)
    p=args['destination']/'base.fmanifest'
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(PatchError,match='digest changed'):
        import_gui_generation(generation=args['destination'],workdir=tmp_path/'gui_data')


def test_generation_command_path_uses_sharded_base(tmp_path):
    from patchlib.gui import DesktopController
    args=fixture_state(tmp_path)
    result=prepare(**args)
    root=args['destination']
    cmd=DesktopController.build_command(room=result['new_room_id'],model='myorg/my-model',
        revision='main',base=None,manifest=root/'room-manifest.json',
        ws='ws://127.0.0.1:8080/ws',task='causal-lm',kind='loha',rank=16,
        journal=tmp_path/'new-events.sqlite3',
        sharded_base_manifest=root/'base.fmanifest',sharded_base_store=root/'store')
    assert '--sharded-base-manifest' in cmd
    assert '--sharded-base-store' in cmd
    assert '--state-base' not in cmd


def test_imported_generation_gui_starts_on_new_sharded_base(tmp_path,monkeypatch):
    from patchlib.generation_rollover import import_gui_generation
    from patchlib.webview_controller import WebviewController
    args=fixture_state(tmp_path)
    prepared=prepare(**args)
    work=tmp_path/'ui'
    imported=import_gui_generation(generation=args['destination'],workdir=work)
    controller=WebviewController(work)
    calls=[]
    monkeypatch.setattr(controller.train,'start',lambda cmd, log: (calls.append((cmd,log)) or 12345))
    try:
        result=controller.start({'room':imported['room'],'ws':'ws://127.0.0.1:8080/ws'})
        assert result['pid']==12345
        assert '--sharded-base-manifest' in calls[0][0]
        assert '--state-base' not in calls[0][0]
        assert '--room-id' in calls[0][0]
        # A later GUI instance must reconstruct the same immutable base.
        controller2=WebviewController(work)
        try:
            monkeypatch.setattr(controller2.train,'start',lambda cmd, log: (calls.append((cmd,log)) or 23456))
            again=controller2.start({'room':imported['room'],'ws':'ws://127.0.0.1:8080/ws'})
            assert again['pid']==23456
            assert '--sharded-base-manifest' in calls[-1][0]
        finally: controller2.close()
    finally: controller.close()


def test_live_shutdown_signs_exact_boundary_and_rejects_older_checkpoint(tmp_path):
    from patchlib.generation_rollover import sign_live_checkpoint
    from patchlib.roomid import auto_identity
    args=fixture_state(tmp_path)
    args['checkpoint'].with_suffix('.checkpoint.json').unlink()
    owner, signer, _, _=auto_identity(Path(str(args['journal'])+'.identity.pem'))
    journal=EventJournal(args['journal'],'a'*64,max_bytes=0)
    try:
        descriptor=sign_live_checkpoint(journal,args['checkpoint'],signer=signer)
        assert descriptor.is_file()
        assert json.loads(descriptor.read_text())['event_count']==2
        journal.append(owner,3,b'newer-update')
        with pytest.raises(PatchError,match='stale'):
            sign_live_checkpoint(journal,args['checkpoint'],signer=signer)
    finally:journal.close()
