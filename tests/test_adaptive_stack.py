import asyncio
import json
import time

import pytest
import torch
from torch import nn
from patchlib.adaptive import DeviceProfile, initial_policy, AdaptiveController
from patchlib.room_manifest import make_manifest,manifest_digest,validate_manifest,verify_code
from patchlib.roomid import RoomDiscovery
from patchlib.moe_light import balanced_experts, slice_moe_modulelist, UnavailableExpert
from patchlib.core import PatchError, UniversalAdapter


def test_adaptive_memory_and_other_peers():
    low=DeviceProfile(2,2.5,'cpu',0,30)
    high=DeviceProfile(16,128.,'cuda',48.,400)
    a=initial_policy(low,remote_tiers=[0,1])
    b=initial_policy(high,remote_tiers=[3])
    assert a.batch_size == 1 and a.gradient_checkpointing
    assert b.batch_size>=a.batch_size and b.preferred_neighbors>a.preferred_neighbors
    assert a.local_steps>b.local_steps
    assert low.public_summary()=={'tier':0,'accelerator':False}
    assert 'ram_free_gib' not in low.public_summary()


def test_adaptive_stale_untrusted_peer_data():
    p=DeviceProfile(16,32,'cuda',24,120)
    ctl=AdaptiveController(p,initial_policy(p,requested_local_steps=16))
    now=time.monotonic()
    ctl.observe('A',{'tier':-1},now)
    ctl.observe('B',{'tier':0,'some_poison':'x'},now-999)
    assert not ctl.peers.get('A')
    ctl.tune(now=now,timeout=10)
    assert 'B' not in ctl.peers
    assert ctl.policy.local_steps >= 4


def test_room_code_hash_no_execution(tmp_path):
    code=tmp_path/'factory.py'
    code.write_text('raise RuntimeError("NEVER EXECUTED")')
    m=make_manifest('some/model','abcdef012345','dataset','code-enabled','',code_path=code,
                    code_url='https://example.org/factory.py') if False else make_manifest(
                    'some/model','abcdef012345','dataset',description='test',code_path=code,
                    code_url='https://example.org/factory.py')
    assert verify_code(code,m)
    assert len(manifest_digest(m))==64
    code.write_text('print("different")')
    with pytest.raises(PermissionError):verify_code(code,m)
    with pytest.raises(PatchError):make_manifest('m','abc','ds',code_url='http://127.0.0.1/code.py',code_path=code)
    with pytest.raises(PatchError):validate_manifest({**m,'extra':1})


def test_room_descriptor_pin_and_members():
    m=make_manifest('foo/bar','deadbeef','dataset')
    o=make_manifest('foo/bar','deadbeef','other')
    d=RoomDiscovery('wss://example.invalid/ws','a'*32,'p'+'1'*32,manifest=m,
                    backup_urls=['wss://backup.invalid/ws'])
    assert len(d.urls)==2
    d.accept({'type':'group-members','topology':'bounded-v2','degree':8,
              'revision':1,'online':2,'peers':['p'+'2'*32],'manifest':m})
    assert 'p'+'2'*32 in d.members
    with pytest.raises(PatchError):d.accept({'type':'group-members','topology':'bounded-v2','degree':8,
              'revision':2,'online':1,'peers':[],'manifest':o})
    # Emergency: last-known roster remains available to existing channels.
    d.online=False;d.emergency=True
    assert 'p'+'2'*32 in d.members


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(3,4)
        self.experts=nn.ModuleList([nn.Linear(3,3) for _ in range(4)])
        self.proj=nn.Linear(3,1)
    def forward(self,x):
        logits=self.gate(x)
        ids=logits.topk(1).indices.squeeze(-1)
        result=torch.stack([self.experts[int(idx)](x[j]) for j,idx in enumerate(ids)])
        return self.proj(result)


def test_moe_selected_experts_and_gradients():
    m=ToyMoE()
    sl=slice_moe_modulelist(m,'experts','gate',count=2,seed='same-room')
    assert len(sl.keep)==2
    assert sum(isinstance(e,UnavailableExpert) for e in m.experts)==2
    inp=torch.randn(10,3,requires_grad=True)
    out=m(inp).sum();out.backward()
    assert inp.grad is not None
    # Same deterministic choice across peers
    assert sl.keep==balanced_experts(4,2,'same-room')
    # Adapter can attach to the pruned model, and layout includes only kept experts.
    a=UniversalAdapter(m,'toy','base',rank=2)
    assert a.counts()['patch']>0


def test_moe_fused_format_fail_closed():
    class Packed(nn.Module):
        def __init__(self):
            super().__init__();self.experts=nn.Linear(3,4);self.gate=nn.Linear(3,4)
    with pytest.raises(ValueError):slice_moe_modulelist(Packed(),'experts','gate',count=2)
