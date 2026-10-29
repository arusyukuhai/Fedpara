import copy,hashlib,sqlite3
import pytest
from patchlib.core import PatchError,_canonical_json
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.frontier_commitment_v44 import build_frontier_pages
from patchlib.epoch_certificate_v46 import CompactEpochLedger,proposal_v2,verify_paged_candidate


def test_signed_quorum_for_4096_origin_merkle_cut(tmp_path):
    members=['a','b','c']; signers={};verifiers={}
    for peer in members:
        prv=tmp_path/(peer+'.pem');pub=tmp_path/(peer+'.pub')
        generate_keypair(prv,pub)
        signers[peer]=signer_from_pem(prv)
        verifiers[peer]=verifier_from_pem(pub)
    ledgers={p:CompactEpochLedger(tmp_path/(p+'.db'),room='room',peer=p,
        committee=members,signer=signers[p],trusted=verifiers,genesis='0'*64) for p in members}
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT)')
    frontier={'p%05d'%i:1 for i in range(4096)}
    db.executemany('INSERT INTO events VALUES(?,?,?)',[(p,1,hashlib.sha256(p.encode()).hexdigest()) for p in frontier])
    header,pages=build_frontier_pages(frontier)
    h=hashlib.sha256()
    for row in db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        h.update(_canonical_json(list(row))+b'\n')
    import torch
    from patchlib.shards import export_state_sharded, read_manifest, semantic_shard_sha256
    manifest=tmp_path/'state.fmanifest';store=tmp_path/'objects'
    export_state_sharded(torch.nn.Linear(2,3),store,manifest,model_id='tiny',
                         revision='v1',peer_id='alice',seq=4096,shard_bytes=65536)
    snapshot_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    semantic=semantic_shard_sha256(read_manifest(manifest))
    p=proposal_v2(room='room',epoch=1,parent='0'*64,manifest=snapshot_sha,
                  semantic=semantic,inventory=h.hexdigest(),frontier_header=header,replicas=members)
    assert len(_canonical_json(p))<1024
    signatures=[ledgers[k].vote(p,pages=pages,journal_db=db,manifest_path=manifest,shard_store=store) for k in ('a','b')]
    cert=ledgers['a'].certificate(p,signatures)
    for ledger in ledgers.values():
        assert ledger.accept(cert)
        assert ledger.accept(cert) is False
    broken=copy.deepcopy(pages);broken[2]['entries'][0][1]=2
    with pytest.raises(PatchError):verify_paged_candidate(p,broken,db)
    with pytest.raises(PatchError):verify_paged_candidate(p,pages[:-1],db)
    for ledger in ledgers.values():ledger.close()
    db.close()
