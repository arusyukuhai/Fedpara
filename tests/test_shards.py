import asyncio
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.core import PatchError
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.shards import (shard_state, read_manifest, shard_map, _blob_path,
                             parse_manifest, assemble_state, verify_shards)
from patchlib.shardp2p import (ShardSession, send_manifest, send_shard)
from patchlib.state import export_state, apply_state, verify_state

SMALL = 64*1024


class FakeChannel:
    readyState = 'open'
    bufferedAmount = 0
    def __init__(self):
        self.frames=[]
    def send(self,data):
        self.frames.append(data)


def packed(tmp_path, model=None, name='a', store=None):
    model = model or nn.Sequential(nn.Embedding(12_000,48),nn.Linear(48,32))
    archive = tmp_path/f'{name}.fstate'
    manifest = tmp_path/f'{name}.fmanifest'
    store = store or tmp_path/'store'
    export_state(model,archive,model_id='a/b',revision='1'*40,peer_id=name,seq=3)
    shard_state(archive,store,manifest,shard_bytes=SMALL)
    return model, archive, manifest, store


def test_roundtrip_and_legacy_compatibility(tmp_path):
    orig,archive,manifest,store = packed(tmp_path)
    meta=read_manifest(manifest)
    assert len(shard_map(meta)) > 2
    assert sum(x['bytes'] for x in meta['tensors']) == meta['total_tensor_bytes']
    verify_shards(meta,store)
    new=tmp_path/'assembled.fstate'
    assemble_state(manifest,store,new)
    assert verify_state(new)['total_tensor_bytes'] == verify_state(archive)['total_tensor_bytes']
    target=nn.Sequential(nn.Embedding(12_000,48),nn.Linear(48,32))
    apply_state(target,new,model_id='a/b',revision='1'*40)
    for x,y in zip(target.state_dict().values(),orig.state_dict().values()):
        assert torch.equal(x,y)


def test_unchanged_tensor_shards_are_reused(tmp_path):
    model,archive,manifest,store=packed(tmp_path)
    before=read_manifest(manifest)
    before_hashes={t['name']:[s['sha256'] for s in t['shards']] for t in before['tensors']}
    with torch.no_grad(): model[1].bias.add_(2)
    a2=tmp_path/'a2.fstate'; m2=tmp_path/'a2.fmanifest'
    export_state(model,a2,model_id='a/b',revision='1'*40,peer_id='a',seq=4)
    shard_state(a2,store,m2,shard_bytes=SMALL)
    after=read_manifest(m2)
    after_hashes={t['name']:[s['sha256'] for s in t['shards']] for t in after['tensors']}
    assert before_hashes['0.weight'] == after_hashes['0.weight']
    assert before_hashes['1.bias'] != after_hashes['1.bias']
    assert before_hashes['1.weight'] == after_hashes['1.weight']


def test_missing_corrupt_and_malicious_manifest(tmp_path):
    _, _, m, store=packed(tmp_path)
    original=m.read_bytes()
    meta=read_manifest(m)
    chunks=shard_map(meta)
    target_hash, size=next(iter(chunks.items()))
    target=_blob_path(store,target_hash)
    with target.open('r+b') as f:
        f.write(b'corrupted')
    with pytest.raises(PatchError,match='missing or corrupt'):
        verify_shards(meta,store)
    with pytest.raises(PatchError,match='missing or corrupt'):
        assemble_state(m,store,tmp_path/'bad.fstate')
    assert not (tmp_path/'bad.fstate').exists()
    tampered=json.loads(original)
    tampered['tensors'][0]['name']='../../escape'
    with pytest.raises(PatchError):
        parse_manifest(json.dumps(tampered).encode())
    tampered=json.loads(original)
    tampered['tensors'][0]['shards'][0]['size']=size+1
    with pytest.raises(PatchError):
        parse_manifest(json.dumps(tampered).encode())


def fake_receive(sender: ShardSession, receiver: ShardSession, signer=None):
    ch=FakeChannel()
    asyncio.run(send_manifest(ch,sender.local_data,signer))
    queue=[]
    for frame in ch.frames:
        outgoing,_ = receiver.feed(frame)
        queue.extend(outgoing)
    while queue:
        request=queue.pop(0)
        _, hashes=sender.feed(request)
        for digest in hashes:
            ch=FakeChannel()
            asyncio.run(send_shard(ch,sender.store,digest,sender.local_objects[digest]))
            for frame in ch.frames:
                outgoing,_=receiver.feed(frame)
                queue.extend(outgoing)
    assert receiver.remote_complete
    assert receiver.received_manifest
    control=json.dumps({'v':2,'type':'complete','manifest':sender.local_hash})
    sender.feed(control)
    assert sender.acked_local


def test_p2p_resume_and_signed_manifest(tmp_path):
    _,_,m,srcstore=packed(tmp_path)
    other=tmp_path/'recvstore'
    private=tmp_path/'signer.pem';public=tmp_path/'signer.pub'
    generate_keypair(private,public)
    sender=ShardSession(srcstore,tmp_path/'inbox1',local_manifest=m)
    receiver=ShardSession(other,tmp_path/'inbox2',verifier=verifier_from_pem(public))
    fake_receive(sender,receiver,signer_from_pem(private))
    meta=read_manifest(receiver.received_manifest)
    assert verify_shards(meta,other)[0] > 0
    assert receiver.received_objects == len(shard_map(meta))
    assert receiver.reused_objects == 0
    # A reconnect fetches nothing if the cache is complete.
    receiver2=ShardSession(other,tmp_path/'inbox3',verifier=verifier_from_pem(public))
    sender2=ShardSession(srcstore,tmp_path/'inbox4',local_manifest=m)
    fake_receive(sender2,receiver2,signer_from_pem(private))
    assert receiver2.received_objects == 0
    assert receiver2.reused_objects == len(shard_map(meta))
    # Remove one shard: just that shard is requested after reconnect.
    digest=next(iter(shard_map(meta)))
    _blob_path(other,digest).unlink()
    receiver3=ShardSession(other,tmp_path/'inbox5',verifier=verifier_from_pem(public))
    sender3=ShardSession(srcstore,tmp_path/'inbox6',local_manifest=m)
    fake_receive(sender3,receiver3,signer_from_pem(private))
    assert receiver3.received_objects == 1
    assert sender3.requested_shards == [digest]
    assert receiver3.reused_objects == len(shard_map(meta))-1


def test_bad_signature_wrong_model_and_truncated_shard(tmp_path):
    _,_,m,store=packed(tmp_path)
    private=tmp_path/'key.pem';public=tmp_path/'key.pub'
    generate_keypair(private,public)
    recv=ShardSession(tmp_path/'newstore',tmp_path/'received',verifier=verifier_from_pem(public))
    send=ShardSession(store,tmp_path/'send',local_manifest=m)
    ch=FakeChannel(); asyncio.run(send_manifest(ch,send.local_data,signer_from_pem(private)))
    data=json.loads(ch.frames[0]);data['sig']='AAAA'
    with pytest.raises(PatchError,match='signature'):
        recv.feed(json.dumps(data))
    recv=ShardSession(tmp_path/'newstore',tmp_path/'received')
    received=[]
    for frame in ch.frames:
        msgs,_=recv.feed(frame);received += msgs
    assert received and json.loads(received[0])['type']=='need'
    _, requested=send.feed(received[0])
    one=requested[0]
    assert one in recv.outstanding
    start={'v':2,'type':'shard-start','sha256':one,'size':send.local_objects[one]}
    recv.feed(json.dumps(start))
    recv.feed(b'partial')
    with pytest.raises(PatchError,match='corrupted/incomplete'):
        recv.feed(json.dumps({'v':2,'type':'shard-end','sha256':one}))
    recv.close()
    assert not list((tmp_path/'newstore').rglob('*.part'))
    wrong = json.loads(m.read_bytes())
    wrong['revision']='OTHER'
    wrong_manifest=tmp_path/'wrong.fmanifest';wrong_manifest.write_text(json.dumps(wrong))
    # Receiver offering the wrong revision must fail before transfer.
    existing=ShardSession(store,tmp_path/'existing',local_manifest=m)
    sender_wrong=ShardSession(store,tmp_path/'wrong',local_manifest=wrong_manifest)
    fake=FakeChannel();asyncio.run(send_manifest(fake,sender_wrong.local_data))
    with pytest.raises(PatchError,match='revision'):
        for fr in fake.frames: existing.feed(fr)


def test_direct_export_with_active_adapter(tmp_path):
    from patchlib.shards import export_state_sharded
    from patchlib.core import UniversalAdapter
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(100,8),nn.Linear(8,12),nn.Conv1d(4,3,2))
    adapter = UniversalAdapter(model,'a/b','1'*40,kind='loha',rank=3)
    with torch.no_grad():
        for p in adapter.trainable_parameters():
            p.add_(.007)
    full=tmp_path/'full.fstate'
    export_state(model,full,model_id='a/b',revision='1'*40,peer_id='alice',adapter=adapter)
    direct=tmp_path/'direct.fmanifest'
    export_state_sharded(model,tmp_path/'store',direct,model_id='a/b',revision='1'*40,
                         peer_id='alice',adapter=adapter,shard_bytes=SMALL)
    assembled=tmp_path/'direct.fstate'
    assemble_state(direct,tmp_path/'store',assembled)
    assert [(s['name'],s['sha256']) for s in verify_state(full)['tensors']] == [
        (s['name'],s['sha256']) for s in verify_state(assembled)['tensors']]


@pytest.mark.asyncio
async def test_live_bidirectional_datachannel_simulation(tmp_path):
    from patchlib.shardp2p import ShardNode
    _,_,ma,storea=packed(tmp_path,name='alice',store=tmp_path/'sa')
    # Same model structure and revision, but a different peer's snapshot.
    _,_,mb,storeb=packed(tmp_path,name='bob',store=tmp_path/'sb')
    a=ShardNode('ws://local','room1234567890123','',store=storea,
                recv_dir=tmp_path/'recv-a',local_manifest=ma)
    b=ShardNode('ws://local','room1234567890123','',store=storeb,
                recv_dir=tmp_path/'recv-b',local_manifest=mb)
    class Channel:
        def __init__(self):
            self.readyState='open';self.bufferedAmount=0;self.handlers={};self.other=None
        def on(self,event):
            def wrap(fn):
                self.handlers[event]=fn;return fn
            return wrap
        def send(self,payload):
            assert self.readyState=='open'
            asyncio.get_running_loop().call_soon(self.other.handlers['message'],payload)
        def close(self):
            if self.readyState!='closed':
                self.readyState='closed'
                self.handlers['close']()
    ca=Channel();cb=Channel();ca.other=cb;cb.other=ca
    a._attach_channel(ca);b._attach_channel(cb)
    ca.handlers['open']();cb.handlers['open']()
    recv_a,recv_b=await asyncio.wait_for(asyncio.gather(a.incoming.get(),b.incoming.get()),8)
    await asyncio.wait_for(asyncio.gather(a.local_acked.wait(),b.local_acked.wait()),8)
    assert read_manifest(recv_a)['peer_id']=='bob'
    assert read_manifest(recv_b)['peer_id']=='alice'
    assert a.session.received_objects > 0 and b.session.received_objects > 0
    for node in (a,b):
        node._send_task.cancel()
    ca.close();cb.close()
