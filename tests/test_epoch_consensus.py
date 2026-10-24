"""Deterministic 8-peer epoch agreement: signatures, crash, partition, replay."""
import hashlib
import json
from collections import deque

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from patchlib.core import PatchError
from patchlib.epoch_consensus import EpochPaxos, make_value
from patchlib.identity import _message
from patchlib.roomid import id_from_raw_key, raw_to_verifier


def setup(tmp_path,n=8):
    keys=[Ed25519PrivateKey.generate() for _ in range(n)]
    pubs=[k.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw) for k in keys]
    ids=[id_from_raw_key(raw) for raw in pubs]
    trusted=dict(zip(ids,[raw_to_verifier(raw) for raw in pubs]))
    signers={peer: (lambda size,digest,k=key: __import__('base64').b64encode(k.sign(_message(size,digest))).decode()) for peer,key in zip(ids,keys)}
    e=hashlib.sha256(b'epoch-1').hexdigest();base=hashlib.sha256(b'base').hexdigest()
    peers=sorted(ids)
    nodes={peer: EpochPaxos(tmp_path/peer/'epoch.sqlite',peer=peer,committee=peers,epoch=e,base=base,signer=signers[peer],trusted=trusted) for peer in peers}
    v=make_value(e,base,tuple(peers),[dict(origin=peers[0],seq=1,digest=hashlib.sha256(b'update').hexdigest(),samples=8)])
    return nodes,peers,v,dict(e=e,base=base,signers=signers,trusted=trusted)


def transport(nodes, *, stop=None, limit=12000):
    """Lossless unordered simulated transport, with optional dead peers."""
    stopped=set(stop or ())
    seen=set()
    for tick in range(limit):
        progress=False
        for peer,node in nodes.items():
            if peer in stopped:continue
            outgoing=node.outbox[:];node.outbox.clear()
            for w in outgoing:
                for receiver,remote in nodes.items():
                    if receiver in stopped:continue
                    token=(peer,receiver,hashlib.sha256(w.encode()).hexdigest())
                    if token in seen:continue
                    seen.add(token)
                    remote.receive(w)
                    progress=True
        if not progress:break
    return len(seen)


def start(nodes,peers,v,view=0):
    nodes[peers[view%len(peers)]].propose(v,view=view)


def test_8_peers_commit_with_two_crashes(tmp_path):
    nodes,peers,value,info=setup(tmp_path)
    start(nodes,peers,value)
    transport(nodes,stop=peers[-2:])
    # Latecomers recover via retained signed COMMIT frames.
    assert all(n.committed for p,n in nodes.items() if p not in peers[-2:])
    assert len({n.committed['digest'] for n in nodes.values() if n.committed})==1
    for node in nodes.values():
        node.outbox.extend(node.messages())  # reconnect anti-entropy replay
    transport(nodes)
    assert all(n.committed for n in nodes.values())
    for n in nodes.values():n.close()


def test_crash_restart_preserves_vote_and_committed_certificate(tmp_path):
    nodes,peers,v,info=setup(tmp_path)
    start(nodes,peers,v)
    transport(nodes)
    victim=peers[3];old=nodes[victim]
    saved=old.committed['digest'];old.close()
    nodes[victim]=EpochPaxos(tmp_path/victim/'epoch.sqlite',peer=victim,committee=peers,epoch=info['e'],base=info['base'],signer=info['signers'][victim],trusted=info['trusted'])
    assert nodes[victim].committed['digest']==saved
    bad=make_value(info['e'],info['base'],tuple(peers),[])
    with pytest.raises(PatchError):nodes[victim].propose(bad,view=peers.index(victim)) if peers.index(victim)!=0 else nodes[victim].propose(bad,view=len(peers))
    for n in nodes.values():n.close()


def test_minor_partition_cannot_commit(tmp_path):
    nodes,peers,v,info=setup(tmp_path)
    start(nodes,peers,v)
    # Only first four peers reachable: cannot obtain 5 of 8 signatures.
    transport(nodes,stop=peers[4:])
    assert not any(n.committed for p,n in nodes.items() if p in peers[:4])
    # Restore connectivity: signed messages plus unchanged ballot complete.
    for p,n in nodes.items():
        for w in n.messages():n.outbox.append(w)
    transport(nodes)
    assert all(n.committed for n in nodes.values())
    for n in nodes.values():n.close()


def test_signature_and_commit_certificate_rejections(tmp_path):
    nodes,peers,v,info=setup(tmp_path)
    leader=nodes[peers[0]]
    wire=leader._wire('accept',ballot=0,value=v)
    corrupted=wire.replace('epoch-v2-accept','epoch-v2-commit')
    with pytest.raises(PatchError):nodes[peers[1]].receive(corrupted)
    fake=leader._wire('commit',ballot=0,value=v,proofs=[])
    with pytest.raises(PatchError):nodes[peers[1]].receive(fake)
    rogue=leader._wire('commit',ballot=0,value=v,proofs=[leader._wire('accepted',ballot=0,digest=hashlib.sha256(b'wrong').hexdigest())]*5)
    with pytest.raises(PatchError):nodes[peers[1]].receive(rogue)
    for n in nodes.values():n.close()


def test_prepared_view_failover(tmp_path):
    nodes,peers,v,info=setup(tmp_path)
    start(nodes,peers,v)
    # Phase 1 only, leader disappears before accept broadcasts.
    leader=peers[0]
    prepare=nodes[leader].outbox.pop()
    for p,n in nodes.items():
        if p!=leader:n.receive(prepare)
    # View=1 is next designated coordinator; old accepted values should be adopted.
    nodes[peers[1]].propose(v,view=1)
    transport(nodes,stop=[leader])
    assert all(n.committed for p,n in nodes.items() if p!=leader)
    for n in nodes.values():n.close()


def test_deterministic_epoch_value_and_dedup(tmp_path):
    nodes,peers,v,info=setup(tmp_path)
    reversed_updates=list(reversed([dict(origin=p,seq=1,digest=hashlib.sha256(p.encode()).hexdigest(),samples=3) for p in peers]))
    a=make_value(info['e'],info['base'],tuple(peers),reversed_updates)
    b=make_value(info['e'],info['base'],tuple(peers),list(reversed(reversed_updates)))
    assert a==b
    with pytest.raises(PatchError):make_value(info['e'],info['base'],tuple(peers),[reversed_updates[0]]*2)
    for n in nodes.values():n.close()
