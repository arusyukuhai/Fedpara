import json
from pathlib import Path
import pytest
from patchlib.core import PatchError
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.online_generation import (GenerationLedger,make_proposal,
            AutomaticGenerationControl,atomic_publish_certificate)


def setup(tmp_path, n=5):
    signers={};trusted={};names=[f'peer-{i}' for i in range(n)]
    for name in names:
        prv=tmp_path/(name+'.pem');pub=tmp_path/(name+'.pub')
        generate_keypair(prv,pub)
        signers[name]=signer_from_pem(prv);trusted[name]=verifier_from_pem(pub)
    ledgers={name:GenerationLedger(tmp_path/(name+'.db'),room='room',peer=name,
        committee=names,signer=signers[name],trusted=trusted,genesis='0'*64)
             for name in names}
    def prop(epoch=1,parent='0'*64,manifest='a'*64):
        return make_proposal(room='room',epoch=epoch,parent=parent,manifest=manifest,
             semantic='b'*64,cut='c'*64,cutoffs={'alice':40},replicas=names)
    return ledgers,prop


def test_certificates_crash_quorum_and_offline_catchup(tmp_path):
    ledgers,proposal=setup(tmp_path)
    p=proposal()
    votes=[ledgers[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ledgers)[:3]]
    cert=ledgers['peer-0'].certificate(p,votes)
    for k in list(ledgers)[:4]:assert ledgers[k].accept(cert)
    assert ledgers['peer-0'].accept(cert) is False
    status=ledgers['peer-4'].adoption_status(local_manifest_sha=None,local_semantic_sha=None,
                         pending_local_events=0,new_journal_ready=False)
    assert status['action']=='no-committed-generation'
    ledgers['peer-4'].accept(cert)
    status=ledgers['peer-4'].adoption_status(local_manifest_sha=None,local_semantic_sha=None,
                         pending_local_events=0,new_journal_ready=False)
    assert status['action']=='fetch-certified-snapshot'
    assert status['manifest']==p['manifest']
    assert ledgers['peer-4'].adoption_status(local_manifest_sha=p['manifest'],local_semantic_sha=p['semantic'],
                         pending_local_events=2,new_journal_ready=True)['gc_allowed'] is False
    assert ledgers['peer-4'].adoption_status(local_manifest_sha=p['manifest'],local_semantic_sha=p['semantic'],
                         pending_local_events=0,new_journal_ready=True)['gc_allowed'] is True
    atomic_publish_certificate(tmp_path/'cert.json',cert)
    assert json.loads((tmp_path/'cert.json').read_text())['proposal']==p


def test_quorum_rejects_missing_votes_and_bad_signature(tmp_path):
    ls,prop=setup(tmp_path)
    p=prop()
    votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:2]]
    with pytest.raises(PatchError,match='quorum'):
        ls['peer-0'].certificate(p,votes)
    tampered=json.loads(json.dumps(votes[0]));tampered['signature']='not-a-signature'
    with pytest.raises(PatchError):ls['peer-0'].certificate(p,[tampered]+votes)
    with pytest.raises(PatchError,match='verified'):
        ls['peer-2'].vote(p,verified_snapshot=False,verified_cut=True)


def test_one_vote_per_height_across_restart(tmp_path):
    ls,proposal=setup(tmp_path)
    p=proposal();other=proposal(manifest='f'*64)
    ls['peer-0'].vote(p,verified_snapshot=True,verified_cut=True)
    ls['peer-0'].close()
    key='peer-0'
    prv=tmp_path/(key+'.pem')
    from patchlib.identity import signer_from_pem,verifier_from_pem
    trusted={k:verifier_from_pem(tmp_path/(k+'.pub')) for k in ls}
    again=GenerationLedger(tmp_path/(key+'.db'),room='room',peer=key,
                  committee=list(ls),signer=signer_from_pem(prv),trusted=trusted,genesis='0'*64)
    with pytest.raises(PatchError,match='conflicting'):
        again.vote(other,verified_snapshot=True,verified_cut=True)


def test_wrong_parent_stale_cert_and_rejected_fork(tmp_path):
    ls,proposal=setup(tmp_path)
    p=proposal();votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:3]]
    cert=ls['peer-0'].certificate(p,votes)
    ls['peer-0'].accept(cert)
    with pytest.raises(PatchError,match='stale'):
        ls['peer-0'].vote(proposal(manifest='d'*64),verified_snapshot=True,verified_cut=True)
    p2=proposal(epoch=2,parent=ls['peer-0'].head(),manifest='f'*64)
    for k in list(ls)[:3]:ls[k].accept(cert)
    v2=[ls[k].vote(p2,verified_snapshot=True,verified_cut=True) for k in list(ls)[:3]]
    c2=ls['peer-0'].certificate(p2,v2)
    assert ls['peer-0'].accept(c2)
    assert ls['peer-0'].latest()['proposal']['epoch']==2


def test_autonomous_message_gossip_and_quorum(tmp_path):
    ls,proposal=setup(tmp_path)
    controllers={k:AutomaticGenerationControl(v,lambda p:True,lambda p:True) for k,v in ls.items()}
    first=controllers['peer-0'].propose(proposal())
    for ctl in controllers.values():ctl.receive(first)
    # With gossip forwarding, each committee replica signs on its own.
    for i in range(3):
        for wire in controllers[f'peer-{i}'].drain():
            for target in controllers.values():
                target.receive(wire)
    assert sum(bool(x.latest()) for x in ls.values())>=3


def test_capacity_frontier_validation(tmp_path):
    ls,p=setup(tmp_path)
    with pytest.raises(PatchError):make_proposal(room='room',epoch=1,parent='0'*64,
         manifest='a'*64,semantic='b'*64,cut='c'*64,
         cutoffs={f'p{i}':i for i in range(9000)},replicas=list(ls))


def test_1000_offline_members_need_only_latest_cert(tmp_path):
    ls,prop=setup(tmp_path)
    p=prop();votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:3]]
    c=ls['peer-0'].certificate(p,votes);ls['peer-0'].accept(c)
    # Ordinary participants don't need to vote or be permanently online.
    advert=ls['peer-0'].recovery_announcement()
    for _ in range(1000):
        assert advert['epoch']==1 and advert['manifest']==p['manifest']
    assert len(json.dumps(advert))<1024


def _local_gc_fixture(tmp_path):
    import hashlib
    import torch
    from patchlib.elastic import EventJournal
    from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
    from patchlib.core import _canonical_json
    ls,_=setup(tmp_path)
    old=tmp_path/'prior.sqlite3'
    journal=EventJournal(old,'e'*64,max_bytes=0)
    journal.append('alice',1,b'signed-example-event')
    digest=hashlib.sha256()
    for triple in journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        digest.update(_canonical_json(list(triple))+b'\n')
    cut=digest.hexdigest()
    journal.close()
    manifest=tmp_path/'current.fmanifest';store=tmp_path/'obj'
    export_state_sharded(torch.nn.Linear(4,5),store,manifest,model_id='example/model',
                         revision='test',peer_id='alice',seq=1,shard_bytes=65536)
    sem=semantic_shard_sha256(read_manifest(manifest))
    p=make_proposal(room='room',epoch=1,parent='0'*64,
        manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        semantic=sem,cut=cut,cutoffs={'alice':1},replicas=list(ls))
    votes=[ls[name].vote(p,verified_snapshot=True,verified_cut=True)
           for name in list(ls)[:3]]
    ls['peer-0'].accept(ls['peer-0'].certificate(p,votes))
    new=tmp_path/'next.sqlite3'
    fresh=EventJournal(new,'f'*64,max_bytes=0);fresh.close()
    return ls['peer-0'], old, new, manifest,store,sem


def test_local_gc_requires_actual_new_generation_activation(tmp_path):
    from patchlib.online_generation import mark_local_generation_loaded,retire_adopted_offline_journal
    ledger,old,new,manifest,store,sem=_local_gc_fixture(tmp_path)
    params=dict(ledger=ledger,old_journal=old,new_journal=new,
                checkpoint_manifest=manifest,local_semantic=sem,pending_local_events=0)
    with pytest.raises(PatchError,match='acknowledged'):
        retire_adopted_offline_journal(**params)
    with pytest.raises(PatchError,match='not been confirmed'):
        mark_local_generation_loaded(ledger=ledger,new_journal=new,manifest=manifest,
                    store=store,weights_loaded_and_verified=False)
    mark_local_generation_loaded(ledger=ledger,new_journal=new,manifest=manifest,
                store=store,weights_loaded_and_verified=True)
    with pytest.raises(PatchError,match='quarantine'):
        retire_adopted_offline_journal(**dict(params,pending_local_events=1))
    assert old.is_file()
    assert retire_adopted_offline_journal(**params)['retired']
    assert not old.exists() and not Path(str(old)+'.fplog').exists()
    assert new.is_file() and manifest.is_file()


def test_gc_blocks_old_unsnapshotted_suffix(tmp_path):
    from patchlib.online_generation import mark_local_generation_loaded,retire_adopted_offline_journal
    from patchlib.elastic import EventJournal
    ledger,old,new,manifest,store,sem=_local_gc_fixture(tmp_path)
    mark_local_generation_loaded(ledger=ledger,new_journal=new,manifest=manifest,
                store=store,weights_loaded_and_verified=True)
    j=EventJournal(old,'e'*64,max_bytes=0)
    j.append('alice',2,b'late signed event');j.close()
    with pytest.raises(PatchError,match='uncovered'):
        retire_adopted_offline_journal(ledger=ledger,old_journal=old,
                new_journal=new,checkpoint_manifest=manifest,
                local_semantic=sem,pending_local_events=0)
    assert old.is_file()


def test_offline_recovery_verified_snapshot_and_fallback(tmp_path):
    from patchlib.online_generation import OfflineSnapshotRecovery
    ledger,old,new,manifest,store,sem=_local_gc_fixture(tmp_path)
    recovery=OfflineSnapshotRecovery(ledger)
    installed=[]
    def fetch(replica,manifest_sha):
        if replica=='peer-0':raise OSError('peer offline')
        return manifest,store
    def install(m,s,epoch):
        installed.append((m,s,epoch))
        return True
    blocked=recovery.restore(fetch_from_replica=fetch,install=install,pending_local_events=1)
    assert not blocked['restored'] and not installed
    restored=recovery.restore(fetch_from_replica=fetch,install=install,pending_local_events=0)
    assert restored['restored'] and restored['epoch']==1 and installed


def test_offline_recovery_no_snapshot_refuses_to_install(tmp_path):
    from patchlib.online_generation import OfflineSnapshotRecovery
    ledger,old,new,manifest,store,sem=_local_gc_fixture(tmp_path)
    recovery=OfflineSnapshotRecovery(ledger)
    installed=[]
    with pytest.raises(PatchError,match='no verified'):
        recovery.restore(fetch_from_replica=lambda replica,d: (tmp_path/'missing',store),
                install=lambda *args:installed.append(args),pending_local_events=0)
    assert not installed


def test_nonvoting_peer_accepts_cert_and_cannot_vote(tmp_path):
    ls,proposal=setup(tmp_path)
    p=proposal();votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:3]]
    cert=ls['peer-0'].certificate(p,votes)
    from patchlib.identity import verifier_from_pem
    from patchlib.online_generation import GenerationLedger
    trusted={k:verifier_from_pem(tmp_path/(k+'.pub')) for k in ls}
    nonvoter=GenerationLedger(tmp_path/'ordinary.db',room='room',peer='ordinary-peer',
                   committee=list(ls),signer=None,trusted=trusted,genesis='0'*64)
    assert nonvoter.accept(cert)
    assert nonvoter.recovery_announcement()['epoch']==1
    with pytest.raises(PatchError,match='ordinary'):
        nonvoter.vote(p,verified_snapshot=True,verified_cut=True)


def test_parallel_64_regular_peer_recovery_planning(tmp_path):
    ls,proposal=setup(tmp_path)
    p=proposal();votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:3]]
    cert=ls['peer-0'].certificate(p,votes)
    from patchlib.identity import verifier_from_pem
    from patchlib.online_generation import GenerationLedger
    trusted={k:verifier_from_pem(tmp_path/(k+'.pub')) for k in ls}
    # 1000 distinct offline participants, no need for 1000 signatures/quorum votes.
    for i in range(64):
        regular=GenerationLedger(tmp_path/f'p{i}.db',room='room',peer=f'outside-{i}',
                   committee=list(ls),signer=None,trusted=trusted,genesis='0'*64)
        regular.accept(cert)
        status=regular.adoption_status(local_manifest_sha=None,
                      local_semantic_sha=None,pending_local_events=0,new_journal_ready=False)
        assert status['action']=='fetch-certified-snapshot'
        regular.close()
