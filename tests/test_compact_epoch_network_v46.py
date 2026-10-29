import hashlib,sqlite3
import torch
from patchlib.core import _canonical_json
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
from patchlib.frontier_commitment_v44 import build_frontier_pages
from patchlib.online_generation import GenerationLedger,AutomaticGenerationControl
from patchlib.epoch_certificate_v46 import CompactEpochLedger,proposal_v2
from patchlib.live_generation_transport import GenerationWebRTCBridge
from tests.test_v44_low_cost_generation import DummyNode


def test_compact_signed_certificate_recovers_after_disconnection(tmp_path):
    names=['a','b','c'];keys={};verifiers={}
    for name in names:
        prv=tmp_path/(name+'.pem');pub=tmp_path/(name+'.pub')
        generate_keypair(prv,pub)
        keys[name]=signer_from_pem(prv)
        verifiers[name]=verifier_from_pem(pub)
    old={};new={};bridge={}
    for name in names:
        old[name]=GenerationLedger(tmp_path/(name+'.v1.db'),room='room',peer=name,
            committee=names,signer=keys[name],trusted=verifiers,genesis='0'*64)
        new[name]=CompactEpochLedger(tmp_path/(name+'.v2.db'),room='room',peer=name,
            committee=names,signer=keys[name],trusted=verifiers,genesis='0'*64)
        bridge[name]=GenerationWebRTCBridge(
            AutomaticGenerationControl(old[name],lambda _:False,lambda _:False),
            compact_ledger=new[name],interval=.1)
    manifest=tmp_path/'checkpoint.fmanifest';store=tmp_path/'store'
    export_state_sharded(torch.nn.Linear(2,3),store,manifest,model_id='tiny',
        revision='v1',peer_id='a',seq=1,shard_bytes=65536)
    h=hashlib.sha256();digest=hashlib.sha256(b'event').hexdigest()
    h.update(_canonical_json(['a',1,digest])+b'\n')
    header,pages=build_frontier_pages({'a':1})
    p=proposal_v2(room='room',epoch=1,parent='0'*64,
        manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        semantic=semantic_shard_sha256(read_manifest(manifest)),
        inventory=h.hexdigest(),frontier_header=header,replicas=names)
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT)')
    db.execute('INSERT INTO events VALUES(?,?,?)',('a',1,digest))
    votes=[new[n].vote(p,pages=pages,journal_db=db,manifest_path=manifest,shard_store=store) for n in ('a','b')]
    new['a'].accept(new['a'].certificate(p,votes))
    def deliver(source,target,now):
        wire=DummyNode();bridge[source].tick([(target,wire)],now=now)
        for name,data in wire.wires:
            bridge[target].receive(source,name,data)
        return wire.wires
    # b hears head, asks for range, receives signed certificate.
    deliver('a','b',1.)
    assert any(kind=='generation-compact-need' for kind,_ in deliver('b','a',2.))
    deliver('a','b',3.)
    assert new['b'].latest()['proposal']==p
    # c was disconnected during commit; replay the signed chain on return.
    deliver('a','c',4.)
    deliver('c','a',5.)
    deliver('a','c',6.)
    assert new['c'].latest()['proposal']==p
    for peer in names:
        old[peer].close();new[peer].close()
    db.close()
