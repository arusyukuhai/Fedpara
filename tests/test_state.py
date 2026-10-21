from pathlib import Path
import json
import struct
import zipfile

import pytest
import torch
from torch import nn

from patchlib.core import UniversalAdapter, PatchError
from patchlib.state import export_state, inspect_state, apply_state, verify_state
from patchlib.p2p import DiskFrameReceiver, transmit_file
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem


def test_state_roundtrip_buffers_tied_and_custom_shape(tmp_path):
    torch.manual_seed(6)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 6)
            self.head = nn.Linear(6,8,bias=False)
            self.head.weight = self.emb.weight
            self.conv = nn.Conv2d(2, 4, 3)
            self.register_buffer('counts',torch.tensor([1,2,3],dtype=torch.int64))
        def forward(self, indices):
            return self.head(self.emb(indices))
    a,b = Toy(),Toy()
    adapter = UniversalAdapter(a, 'x/y','c'*40,rank=3,kind='loha')
    # Force a known perturbation in the effective weight and the dense fallback.
    with torch.no_grad():
        for p in adapter.trainable_parameters():
            p.add_(0.003)
    expected = a(torch.tensor([[1,2]]))
    path = tmp_path/'snapshot.fstate'
    export_state(a,path,model_id='x/y',revision='c'*40,peer_id='node1',seq=8,adapter=adapter)
    meta=verify_state(path)
    assert {s['name'] for s in meta['tensors']} == set(b.state_dict())
    with pytest.raises(PatchError,match='revision'):
        apply_state(b,path,model_id='x/y',revision='a'*40)
    apply_state(b,path,model_id='x/y',revision='c'*40)
    assert torch.allclose(b(torch.tensor([[1,2]])), expected,atol=1e-5)
    assert b.emb.weight is b.head.weight
    assert torch.equal(b.counts,a.counts)


def test_state_malformed_rejected_without_mutation(tmp_path):
    a=nn.Linear(3,4)
    state=tmp_path/'good.fstate'
    export_state(a,state,model_id='A',revision='1',peer_id='p')
    before = a.weight.detach().clone()
    bad=tmp_path/'bad.fstate'
    with zipfile.ZipFile(state) as src, zipfile.ZipFile(bad,'w') as dst:
        for name in src.namelist():
            data=src.read(name)
            if name.startswith('tensors/'):
                data=bytes([data[0]^1])+data[1:]
            dst.writestr(name,data,compress_type=zipfile.ZIP_STORED)
    with pytest.raises(PatchError,match='corrupt'):
        apply_state(a,bad,model_id='A',revision='1')
    assert torch.equal(before,a.weight)


def test_state_rejects_zip_extra_entries_and_compression(tmp_path):
    m=nn.Linear(3,2)
    good=tmp_path/'good.fstate'
    export_state(m,good,model_id='a',revision='1',peer_id='x')
    with zipfile.ZipFile(good,'a') as f:
        f.writestr('../evil.py',b'print(1)',compress_type=zipfile.ZIP_STORED)
    with pytest.raises(PatchError,match='unexpected'):
        inspect_state(good)


def test_signed_disk_transfer(tmp_path):
    private=tmp_path/'alice.pem'; public=tmp_path/'alice.pub.pem'
    generate_keypair(private,public)
    sign,verify=signer_from_pem(private),verifier_from_pem(public)
    m=nn.Linear(3,4)
    artifact=tmp_path/'a.fstate'
    export_state(m,artifact,model_id='a',revision='b',peer_id='alice')
    class FakeChannel:
        readyState='open'
        bufferedAmount=0
        def __init__(self): self.events=[]
        def send(self,x): self.events.append(x)
    import asyncio
    fake=FakeChannel()
    asyncio.run(transmit_file(fake,artifact,100_000,signer=sign))
    recv=DiskFrameReceiver(tmp_path/'received',100_000,verifier=verify)
    result=None
    for ev in fake.events:
        result=recv.feed(ev) or result
    assert result is not None
    assert result.read_bytes()==artifact.read_bytes()
    assert inspect_state(result)['peer_id']=='alice'
    fake.events[0]=json.dumps(json.loads(fake.events[0])|{'sig':'AAAA'})
    recv2=DiskFrameReceiver(tmp_path/'received',100_000,verifier=verify)
    with pytest.raises(PatchError,match='signature'):
        recv2.feed(fake.events[0])


def test_state_bfloat16_and_scalar(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w=nn.Parameter(torch.arange(12).reshape(3,4).to(torch.bfloat16))
            self.register_buffer('scalar',torch.tensor(42,dtype=torch.int64))
    a,b=Model(),Model()
    with torch.no_grad(): b.w.zero_(); b.scalar.zero_()
    p=tmp_path/'test.fstate'
    export_state(a,p,model_id='abc',revision='111',peer_id='foo')
    apply_state(b,p,model_id='abc',revision='111')
    assert torch.equal(a.w,b.w)
    assert torch.equal(a.scalar,b.scalar)


def test_state_rejects_nonfinite(tmp_path):
    m=nn.Linear(2,2)
    with torch.no_grad(): m.bias[0]=float('nan')
    p=tmp_path/'nan.fstate'
    export_state(m,p,model_id='m',revision='1',peer_id='x')
    with pytest.raises(PatchError,match='non-finite'):
        verify_state(p)
