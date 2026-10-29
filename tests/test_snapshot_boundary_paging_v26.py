"""Multi-origin signed boundary pages, partial delivery and reconnect regressions."""
import asyncio
import hashlib

import pytest

from patchlib.core import PatchError
from patchlib.elasticnet import (ElasticNode, CTRL_MAX, decode_control,
                                 send_snapshot_boundary)
from patchlib.roomid import auto_identity
from patchlib.snapshot_boundary import (SnapshotBoundaryAssembler,
                                        boundary_digest, boundary_pages)
from patchlib.snapshot_live import snapshot_inventory
from tests.test_snapshot_request_delivery import FakeChannel


def clock(n=4096):
    return {f'p{i:032x}': i + 1 for i in range(n)}


def test_4096_origin_pages_reassemble_and_fit_signed_wire(tmp_path):
    origin, signer, pub, verifier = auto_identity(tmp_path/'peer.pem')
    boundaries = clock()
    pages = list(boundary_pages('a'*64, 'b'*64, 4096, boundaries, '0123456789abcdef'))
    assert len(pages) > 80
    receiver = SnapshotBoundaryAssembler()
    completed = None
    for i, page in enumerate(pages):
        from patchlib.elasticnet import encode_control
        encoded = encode_control(dict(type='snapshot-tip-page',peer=origin,
                    genesis='d'*64,counter=i+1,**page), signer)
        assert len(encoded) <= CTRL_MAX
        decoded = decode_control(encoded,verifier)
        completed = receiver.accept(decoded)
        if i != len(pages)-1:
            assert completed is None
    assert completed is not None
    assert completed['cutoffs'] == boundaries
    assert completed['events'] == 4096


def test_missing_page_never_completes_and_tamper_fails():
    pages = list(boundary_pages('a'*64,'b'*64,4096,clock()))
    assembler=SnapshotBoundaryAssembler()
    for i,page in enumerate(pages):
        if i==7:continue
        assert assembler.accept(dict(type='snapshot-tip-page',counter=i+1,**page)) is None
    assert assembler.accept(dict(type='snapshot-tip-page',counter=999,**pages[7]))['events']==4096
    assembler=SnapshotBoundaryAssembler()
    for i,page in enumerate(pages):
        modified=dict(page)
        if i == 5:
            modified['cutoffs'] = [list(x) for x in page['cutoffs']]
            modified['cutoffs'][0][1] += 1
        if i != len(pages)-1:
            assert assembler.accept(dict(type='snapshot-tip-page',counter=i+1,**modified)) is None
        else:
            with pytest.raises(PatchError,match='digest mismatch'):
                assembler.accept(dict(type='snapshot-tip-page',counter=i+1,**modified))


def test_boundary_page_rejects_conflicting_duplicate():
    pages = list(boundary_pages('a'*64,'b'*64,96,clock(96)))
    asm=SnapshotBoundaryAssembler()
    assert asm.accept(dict(type='snapshot-tip-page',counter=1,**pages[0])) is None
    altered=dict(pages[0]);altered['cutoffs']=[list(x) for x in pages[0]['cutoffs']]
    altered['cutoffs'][0][1]+=1
    with pytest.raises(PatchError,match='conflicting'):
        asm.accept(dict(type='snapshot-tip-page',counter=2,**altered))


def test_snapshot_inventory_allows_4096_origins():
    from tempfile import TemporaryDirectory
    class Journal:
        class DB:
            def execute(self, sql):
                if 'MAX(seq)' in sql:
                    return ((f'p{i:032x}',1) for i in range(4096))
                return ((f'p{i:032x}',1,'a'*64) for i in range(4096))
        db=DB()
    digest,count,cutoffs=snapshot_inventory(Journal())
    assert len(cutoffs)==4096 and count==4096 and len(digest)==64


def test_real_signed_channel_4096_origin_delivery_and_reconnect(tmp_path):
    async def task():
        alice, signer_a, pub_a, verifier_a=auto_identity(tmp_path/'alice.pem')
        bob, signer_b, pub_b, verifier_b=auto_identity(tmp_path/'bob.pem')
        n=ElasticNode('ws://localhost/ws','r'*32,origin=bob,remote=alice,
                      genesis='9'*64,signer=signer_b,verifier=verifier_a)
        ch=FakeChannel(); n._attach_channel(ch);ch.handlers['open']()
        from patchlib.elasticnet import encode_control
        ch.handlers['message'](encode_control(dict(type='hello',peer=alice,
                genesis='9'*64,counter=1,pub=pub_a,snapshot_tip_pages_v1=True),signer_a))
        # Pre-fill the training queue to prove final tip is non-droppable.
        for i in range(2,132):
            ch.handlers['message'](encode_control(dict(type='heartbeat',peer=alice,
                genesis='9'*64,counter=i),signer_a))
        assert n.controls.full()
        pages=list(boundary_pages('a'*64,'b'*64,4096,clock(),'0123456789abcdef'))
        n.snapshot_request_id='0123456789abcdef'
        for i,page in enumerate(pages):
            ch.handlers['message'](encode_control(dict(type='snapshot-tip-page',peer=alice,
                genesis='9'*64,counter=132+i,**page),signer_a))
        assert not n.protocol_failure.is_set()
        assert n.snapshot_tip_latch['cutoffs']==clock()
        # Incoming new connection must discard previous partial boundary.
        for i,page in enumerate(pages[:2]):
            ch.handlers['message'](encode_control(dict(type='snapshot-tip-page',peer=alice,
                genesis='9'*64,counter=132+len(pages)+i,**page),signer_a))
        ch2=FakeChannel();n._attach_channel(ch2);ch2.handlers['open']()
        assert n.snapshot_boundary_assembler.identity is None
        assert n.snapshot_tip_latch is None
        ch.close();ch2.close()
        await asyncio.sleep(.01)
    asyncio.run(task())


def test_sender_paged_reliable_and_old_peer_rejected(tmp_path):
    async def task():
        a, signer_a, pub_a, verifier_a=auto_identity(tmp_path/'a.pem')
        b, signer_b, pub_b, verifier_b=auto_identity(tmp_path/'b.pem')
        src=ElasticNode('ws://localhost/ws','r'*32,origin=a,remote=b,
                        genesis='9'*64,signer=signer_a,verifier=verifier_b)
        dst=ElasticNode('ws://localhost/ws','r'*32,origin=b,remote=a,
                        genesis='9'*64,signer=signer_b,verifier=verifier_a)
        channel=FakeChannel();src._attach_channel(channel);channel.handlers['open']()
        remote=FakeChannel();dst._attach_channel(remote);remote.handlers['open']()
        src.remote_supports_snapshot_pages=True
        src.snapshot_pending_id='0123456789abcdef'
        dst.snapshot_request_id='0123456789abcdef'
        from patchlib.elasticnet import encode_control
        # Signed hello from sender received by receiver before any pages.
        remote.handlers['message'](channel.sent[0])
        assert await send_snapshot_boundary(src,'a'*64,'b'*64,4096,clock())
        for wire in channel.sent[1:]:remote.handlers['message'](wire)
        assert dst.snapshot_tip_latch['cutoffs']==clock()
        src.remote_supports_snapshot_pages=False
        with pytest.raises(PatchError,match='upgraded peer'):
            await send_snapshot_boundary(src,'a'*64,'b'*64,4096,clock())
        channel.close();remote.close()
        await asyncio.sleep(.01)
    asyncio.run(task())


def test_sharded_manifest_can_describe_state_over_32_gib_without_allocating_model():
    """The transfer protocol works beyond the default safety quota if configured."""
    from patchlib.shards import manifest_bytes, parse_manifest
    shard_bytes = 64 * 1024**2
    nshards = 1024  # 64 GiB of state represented by kilobytes of metadata.
    size = shard_bytes * nshards
    meta = {
        'format':'fedpara-sharded-state','version':1,
        'model_id':'huge', 'revision':'r1', 'peer_id':'p'+'a'*32,
        'seq':0,'shard_bytes':shard_bytes,'total_tensor_bytes':size,
        'tensors':[{'name':'embedding.weight','entry':'tensors/000000.bin',
                    'dtype':'torch.uint8','shape':[size],'bytes':size,
                    'sha256':'a'*64,
                    'shards':[{'sha256':f'{j:064x}','size':shard_bytes}
                              for j in range(nshards)]}]}
    encoded = manifest_bytes(meta)
    parsed = parse_manifest(encoded, max_state_bytes=128 * 1024**3)
    assert parsed['total_tensor_bytes'] == size
    with pytest.raises(PatchError, match='budget'):
        parse_manifest(encoded, max_state_bytes=32 * 1024**3)
