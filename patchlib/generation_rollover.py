"""Fail-closed, offline generation rollover for Kagami's v1 gossip protocol.

The v1 protocol requires the full signed history for anti-entropy.  It is NOT
safe to truncate a live v1 journal in place.  Instead prepare a fresh room
bound to a verified complete checkpoint.  The old room must be quiescent; peers
must explicitly join the *same* new room/checkpoint. Never implicitly delete.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .core import PatchError, _canonical_json
from .identity import _message
from .roomid import create_room_id, validate_room, id_from_raw_key
from .room_manifest import load_manifest
from .shards import (read_manifest, verify_shards, shard_map, _blob_path,
                     manifest_bytes, semantic_shard_sha256)

FORMAT = 'kagami-generation-rollover-v1'


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as fd:
        for chunk in iter(lambda: fd.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    with Path(path).open('rb') as fd:
        return json.load(fd)


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8')
    fd, name = tempfile.mkstemp(prefix='.kagami-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(dirfd)
        finally: os.close(dirfd)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def stopped_journal(path: Path):
    """Prevents active trainers from racing snapshot verification/rollover."""
    lock = Path(str(path) + '.train.lock')
    if not path.is_file() or path.is_symlink():
        raise PatchError('old journal is missing or a symbolic link')
    handle = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise PatchError('stop the trainer and its relay before rollover') from e
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def _validate_checkpoint_signature(desc: dict, journal_path: Path):
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key_path = Path(str(journal_path) + '.identity.pem')
    if not key_path.is_file() or key_path.is_symlink():
        raise PatchError('archived signing identity is missing; refuse unverified checkpoint rollover')
    key = load_pem_private_key(key_path.read_bytes(), password=None)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    if not isinstance(key, Ed25519PrivateKey):
        raise PatchError('old checkpoint identity is not Ed25519')
    raw=key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    if id_from_raw_key(raw) != desc.get('origin'):
        raise PatchError('checkpoint signer differs from journal owner')
    signature=desc.get('signature')
    if not isinstance(signature,str):
        raise PatchError('unsigned checkpoint cannot authorize history retirement')
    original={k:v for k,v in desc.items() if k!='signature'}
    data=_canonical_json(original)
    try:
        key.public_key().verify(base64.b64decode(signature, validate=True),
                                _message(len(data),hashlib.sha256(data).hexdigest()))
    except Exception as e:
        raise PatchError('checkpoint signature verification failed') from e


def _check_immutable_old_state(journal_path, checkpoint_path, store):
    """Only a checkpoint matching the EXACT last committed event is accepted."""
    descriptor=checkpoint_path.with_suffix('.checkpoint.json')
    if not descriptor.is_file():
        raise PatchError('signed .checkpoint.json is required; an arbitrary snapshot is unsafe')
    d=_json(descriptor)
    if d.get('protocol') != 'fedpara-elastic-checkpoint-v1':
        raise PatchError('unsupported checkpoint protocol')
    with sqlite3.connect(f'file:{journal_path.resolve()}?mode=ro',uri=True,timeout=1) as db:
        db.execute('PRAGMA query_only=ON')
        genesis=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
        owner=db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()
        if not genesis or not owner or d.get('genesis')!=genesis[0] or d.get('origin')!=owner[0]:
            raise PatchError('checkpoint does not belong to this peer/genesis')
        digest=hashlib.sha256(); n=0
        # checkpoint writer's inventory uses rowid order, distinct from journal_tip.
        for origin,seq,event_hash in db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid'):
            digest.update(_canonical_json([origin,seq,event_hash])+b'\n'); n+=1
        if d.get('event_count')!=n or d.get('event_inventory_sha256')!=digest.hexdigest():
            raise PatchError('checkpoint does not cover ALL committed events; create a fresh checkpoint')
        expected_total=db.execute("SELECT value FROM meta WHERE name='archive_used_bytes'").fetchone()
    if _sha(checkpoint_path)!=d.get('manifest_sha256'):
        raise PatchError('checkpoint descriptor does not match the saved manifest')
    _validate_checkpoint_signature(d,journal_path)
    meta=read_manifest(checkpoint_path, max_state_bytes=2*1024**4)
    if meta['peer_id']!=d['origin'] or meta['seq']!=n:
        raise PatchError('snapshot model does not match signed journal boundary')
    verify_shards(meta,store)
    return d,meta,(int(expected_total[0]) if expected_total else 0)


def prepare(*, journal: Path, checkpoint: Path, store: Path, room_manifest: Path,
            old_room_id: str, destination: Path, new_room_id: str | None = None):
    """Construct an independently verifiable generation; retain ALL old data."""
    journal=Path(journal).resolve(); checkpoint=Path(checkpoint).resolve()
    store=Path(store).resolve(); destination=Path(destination).resolve()
    old_room_id=validate_room(old_room_id)
    new_room_id=validate_room(new_room_id or create_room_id())
    if new_room_id==old_room_id: raise PatchError('new generation MUST have a new room ID')
    if destination.exists():
        raise PatchError('refusing to overwrite a generation, even partially')
    room=load_manifest(room_manifest)
    with stopped_journal(journal):
        descriptor,meta,old_bytes=_check_immutable_old_state(journal,checkpoint,store)
        if meta['model_id']!=room['model']['id']:
            raise PatchError('checkpoint model ID and room manifest disagree')
        # A checkpoint includes the previous fstate-derived adapter revision.
        # A NEW base must use the model's initial upstream revision instead.
        rebased=dict(meta)
        rebased['revision']=room['model']['revision']
        staging=destination.with_name(destination.name+'.pending-'+os.urandom(4).hex())
        staging.mkdir(parents=True)
        try:
            objects=shard_map(meta)
            for digest,size in objects.items():
                source=_blob_path(store,digest)
                target=_blob_path(staging/'store',digest)
                target.parent.mkdir(parents=True,exist_ok=True)
                try: os.link(source,target)
                except OSError: shutil.copyfile(source,target)
                # Contents verified against original hash above; check after copying.
                if target.stat().st_size!=size or _sha(target)!=digest:
                    raise PatchError('generation shard copy failed checksum')
            from .shards import _atomic_write
            target_manifest=staging/'base.fmanifest'
            _atomic_write(target_manifest,manifest_bytes(rebased))
            verify_shards(read_manifest(target_manifest,max_state_bytes=2*1024**4),staging/'store')
            _atomic_json(staging/'room-manifest.json',room)
            info={'format':FORMAT,'old_room_id':old_room_id,'new_room_id':new_room_id,
                  'old_genesis':descriptor['genesis'],
                  'old_inventory_sha256':descriptor['event_inventory_sha256'],
                  'old_event_count':descriptor['event_count'],
                  'old_event_bytes':old_bytes,
                  'original_checkpoint_sha256':descriptor['manifest_sha256'],
                  'base_sha256':_sha(target_manifest),
                  'base_semantic_sha256':semantic_shard_sha256(rebased),
                  'model_id':meta['model_id'],'model_revision':rebased['revision'],
                  'objects':len(objects),
                  'status':'PREPARED_NOT_ACTIVATED',
                  'safety':'Old room must be stopped; distribute identical generation files; start NEW room only; do not silently join the old v1 room.'}
            _atomic_json(staging/'generation.json',info)
            (staging/'START_HERE.txt').write_text(
                'Kagami generation rollover, OFFLINE COORDINATED RESTART\n'
                '1. STOP old-room trainers AND relays on ALL peers.\n'
                '2. Distribute the ENTIRE generation folder to all peers; SHA256-verify generation.json and base.fmanifest.\n'
                '3. Configure a NEW model/room using room ID: '+new_room_id+'\n'
                '4. All peers must use EXACTLY the same base.fmanifest and store/ objects.\n'
                '5. Start with --sharded-base-manifest <generation>/base.fmanifest --sharded-base-store <generation>/store\n'
                '   and --room-manifest <generation>/room-manifest.json; use a FRESH --journal for each peer.\n'
                '6. Keep old journals until peers have joined and restored from this base.\n'
                '   Do not reconnect an old-genesis peer to the new room.\n'
                'NOTE: v1 cannot coordinate automatic rolling epochs or Byzantine quorum safely.\n',
                encoding='utf-8')
            os.replace(staging,destination)
            return info
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True)
            raise






def sign_live_checkpoint(journal, checkpoint: Path, *, signer):
    """Called after a clean trainer shutdown has drained incoming updates.

    The writer already holds the exclusive journal lock. Use the same DB
    connection as the trainer to avoid cross-thread SQLite misuse.
    """
    checkpoint=Path(checkpoint)
    if signer is None:
        raise PatchError('cannot certify an unsigned training checkpoint')
    owner=journal.db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()
    if owner is None:
        raise PatchError('cannot certify checkpoint without a pinned owner')
    meta=read_manifest(checkpoint,max_state_bytes=2*1024**4)
    digest=hashlib.sha256(); n=0
    for origin,seq,sha in journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid'):
        digest.update(_canonical_json([origin,seq,sha])+b'\n');n+=1
    if meta['seq']!=n or meta['peer_id']!=owner[0]:
        raise PatchError('attempted to sign a checkpoint with a stale journal boundary')
    fields={'protocol':'fedpara-elastic-checkpoint-v1','origin':owner[0],
        'genesis':journal.genesis,'event_count':n,
        'event_inventory_sha256':digest.hexdigest(),
        'manifest_sha256':_sha(checkpoint)}
    raw=_canonical_json(fields)
    fields['signature']=signer(len(raw),hashlib.sha256(raw).hexdigest())
    descriptor=checkpoint.with_suffix('.checkpoint.json')
    if descriptor.exists():
        if _json(descriptor)!=fields:
            raise PatchError('refusing to replace a different signed checkpoint descriptor')
    else:
        _atomic_json(descriptor,fields)
    return descriptor


def verify_generation(generation: Path):
    generation=Path(generation).resolve()
    info=_json(generation/'generation.json')
    if info.get('format')!=FORMAT or info.get('status')!='PREPARED_NOT_ACTIVATED':
        raise PatchError('invalid or unsupported generation description')
    validate_room(info.get('new_room_id',''))
    if not _sha(generation/'base.fmanifest')==info.get('base_sha256'):
        raise PatchError('generation checkpoint manifest digest changed')
    meta=read_manifest(generation/'base.fmanifest',max_state_bytes=2*1024**4)
    if (semantic_shard_sha256(meta)!=info.get('base_semantic_sha256') or
        meta['model_id']!=info.get('model_id') or
        meta['revision']!=info.get('model_revision')):
        raise PatchError('generation checkpoint identity changed')
    room=load_manifest(generation/'room-manifest.json')
    if (room['model']['id']!=meta['model_id'] or
            room['model']['revision']!=meta['revision']):
        raise PatchError('generation room manifest differs from base checkpoint')
    verify_shards(meta,generation/'store')
    return info


def import_gui_generation(*, generation: Path, workdir: Path):
    """Register a prepared generation as one NEW Kagami GUI model.

    This only registers a local descriptor; peers must independently import the
    same generation into the SAME new room ID. No old model is altered.
    """
    generation=Path(generation).resolve()
    info=verify_generation(generation)
    path=Path(workdir).resolve()/'.fedpara-gui'/'rooms'/info['new_room_id']
    if path.exists():
        raise PatchError('new generation GUI model already exists; refusing overwrite')
    path.parent.mkdir(parents=True,exist_ok=True)
    os.mkdir(path,0o700)
    try:
        _atomic_json(path/'manifest.json',_json(generation/'room-manifest.json'))
        _atomic_json(path/'generation-base.json',{'generation':str(generation),
                  'base_sha256':info['base_sha256'],'format':FORMAT})
    except BaseException:
        shutil.rmtree(path)
        raise
    return {'room':info['new_room_id'],'path':str(path/'manifest.json'),
            'base':str(generation/'base.fmanifest')}


def seal_stopped_checkpoint(*, journal: Path, checkpoint: Path, store: Path):
    """Sign a COMPLETE v39 GUI stop-time snapshot for coordinated rollover.

    The trainer must have stopped and exported its effective weights AFTER
    draining all inbound updates. A seq-matched snapshot created manually at
    an arbitrary time is NOT independently provable from the journal alone.
    """
    journal=Path(journal).resolve(); checkpoint=Path(checkpoint).resolve()
    store=Path(store).resolve()
    descriptor=checkpoint.with_suffix('.checkpoint.json')
    with stopped_journal(journal):
        with sqlite3.connect(f'file:{journal}?mode=ro',uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            base=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
            owner=db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()
            if not base or not owner:
                raise PatchError('journal has no durable genesis/owner')
            digest=hashlib.sha256(); n=0
            for origin,seq,event_sha in db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid'):
                digest.update(_canonical_json([origin,seq,event_sha])+b'\n')
                n+=1
        meta=read_manifest(checkpoint,max_state_bytes=2*1024**4)
        if meta['peer_id']!=owner[0] or meta['seq']!=n:
            raise PatchError('GUI checkpoint is not the final committed event boundary; stop and save again')
        verify_shards(meta,store)
        from cryptography.hazmat.primitives.serialization import load_pem_private_key, Encoding, PublicFormat
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private=Path(str(journal)+'.identity.pem')
        if not private.is_file() or private.is_symlink():
            raise PatchError('local journal signing identity is unavailable')
        key=load_pem_private_key(private.read_bytes(),password=None)
        if not isinstance(key,Ed25519PrivateKey) or id_from_raw_key(
          key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw))!=owner[0]:
            raise PatchError('journal signing identity does not match the owner')
        payload={'protocol':'fedpara-elastic-checkpoint-v1','origin':owner[0],
                 'genesis':base[0],'event_count':n,
                 'event_inventory_sha256':digest.hexdigest(),
                 'manifest_sha256':_sha(checkpoint)}
        signed=_canonical_json(payload)
        payload['signature']=base64.b64encode(key.sign(_message(
            len(signed),hashlib.sha256(signed).hexdigest()))).decode('ascii')
        if descriptor.exists():
            if _json(descriptor)!=payload:
                raise PatchError('refusing to replace a different signed checkpoint descriptor')
        else:
            _atomic_json(descriptor,payload)
        return {'checkpoint':str(checkpoint),'descriptor':str(descriptor),
                'events':n,'manifest_sha256':payload['manifest_sha256']}


def retire_old_history(*, journal: Path, checkpoint: Path, store: Path,
                       generation: Path, new_journal: Path, confirmation: str):
    """Explicit destructive cleanup only AFTER a replacement room has progressed.

    This does NOT prove that remote/offline peers have migrated. The operator
    must verify the entire network has switched before authorizing retirement.
    Keep the signed checkpoint and generation store, not the old event bodies.
    """
    journal=Path(journal).resolve()
    checkpoint=Path(checkpoint).resolve()
    store=Path(store).resolve()
    generation=Path(generation).resolve()
    new_journal=Path(new_journal).resolve()
    info=_json(generation/'generation.json')
    if info.get('format')!=FORMAT or info.get('status')!='PREPARED_NOT_ACTIVATED':
        raise PatchError('invalid generation document')
    if confirmation!=info.get('new_room_id'):
        raise PatchError('confirmation must equal the NEW room ID')
    if journal==new_journal or not new_journal.is_file() or new_journal.is_symlink():
        raise PatchError('a DISTINCT, successfully used new-generation journal is required')
    manifest=generation/'base.fmanifest'
    if _sha(manifest)!=info['base_sha256']:
        raise PatchError('new generation checkpoint was modified')
    meta=read_manifest(manifest,max_state_bytes=2*1024**4)
    verify_shards(meta,generation/'store')
    if semantic_shard_sha256(meta)!=info['base_semantic_sha256']:
        raise PatchError('new generation semantic checkpoint differs')
    with stopped_journal(journal), stopped_journal(new_journal):
        old,_,_=_check_immutable_old_state(journal,checkpoint,store)
        if old['genesis']!=info['old_genesis'] or old['event_inventory_sha256']!=info['old_inventory_sha256']:
            raise PatchError('old journal changed since rollover preparation')
        with sqlite3.connect(f'file:{new_journal}?mode=ro',uri=True) as db:
            genesis=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
            count=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        if not genesis or genesis[0]==info['old_genesis'] or count<1:
            raise PatchError('replacement generation has not committed its own signed updates')
        retired=generation/'retirement.json'
        if retired.exists():
            raise PatchError('history retirement already started; inspect retirement.json')
        _atomic_json(retired,{'format':FORMAT+'-retired',
             'old_genesis':info['old_genesis'], 'new_genesis':genesis[0],
             'old_journal':str(journal), 'events_retired':info['old_event_count'],
             'bytes_retired':info['old_event_bytes'],
             'warning':'Old room can no longer restart; offline peers must join new generation.'})
        # Old training DB/sidecar are now *deliberately* retired. Keep identity
        # PEM, manifest and working directory untouched for incident response.
        paths=[Path(str(journal)+'.fplog'), journal,
               Path(str(journal)+'-wal'),Path(str(journal)+'-shm')]
        for target in paths:
            if target.is_symlink():
                raise PatchError('refusing to unlink a symbolic-link journal file')
        for target in paths:
            target.unlink(missing_ok=True)
        return {'retired':True,'old_event_count':info['old_event_count'],
                'old_event_bytes':info['old_event_bytes']}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Verified OFFLINE Kagami generation rollover. No automatic log deletion.')
    p=parser.add_subparsers(dest='command',required=True)
    prep=p.add_parser('prepare',help='Prepare a new room from a signed, full-cut checkpoint')
    for flag in ('journal','checkpoint','store','room-manifest','old-room-id','destination'):
        prep.add_argument('--'+flag,required=True)
    prep.add_argument('--new-room-id')
    imported=p.add_parser('import-gui',help='Register verified new generation as a Kagami GUI model')
    imported.add_argument('--generation',required=True)
    imported.add_argument('--workdir',required=True,help='Kagami data root (KAGAMI_DATA_DIR)')
    seal=p.add_parser('seal',help='Sign a stopped GUI snapshot at the exact journal tip')
    for flag in ('journal','checkpoint','store'):
        seal.add_argument('--'+flag,required=True)
    retire=p.add_parser('retire',help='IRREVERSIBLE: drop old history after verified next-generation training')
    for flag in ('journal','checkpoint','store','generation','new-journal','confirm-new-room-id'):
        retire.add_argument('--'+flag,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        data=prepare(journal=Path(args.journal),checkpoint=Path(args.checkpoint),
                     store=Path(args.store),room_manifest=Path(args.room_manifest),
                     old_room_id=args.old_room_id,destination=Path(args.destination),
                     new_room_id=args.new_room_id)
        print(json.dumps(data,ensure_ascii=False,indent=2))
    elif args.command=='import-gui':
        data=import_gui_generation(generation=Path(args.generation),workdir=Path(args.workdir))
        print(json.dumps(data,ensure_ascii=False,indent=2))
    elif args.command=='seal':
        data=seal_stopped_checkpoint(journal=Path(args.journal),
            checkpoint=Path(args.checkpoint),store=Path(args.store))
        print(json.dumps(data,ensure_ascii=False,indent=2))
    elif args.command=='retire':
        data=retire_old_history(journal=Path(args.journal),checkpoint=Path(args.checkpoint),
             store=Path(args.store),generation=Path(args.generation),
             new_journal=Path(args.new_journal),confirmation=args.confirm_new_room_id)
        print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
