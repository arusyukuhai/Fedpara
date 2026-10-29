"""v62: authentic before/after factors remain virtual and export correctly."""
import hashlib
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from patchlib.core import UniversalAdapter
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.streaming_base import materialize_meta_from_safetensors
from patchlib.layer_offload import install_disk_linear_offload
from patchlib.factor_ledger_v62 import attach_factor_ledger, LazyWeightError
from patchlib.shards import export_state_sharded, apply_sharded_effective_state, read_manifest


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Linear(10,8)
        self.b=nn.Linear(8,6)
    def forward(self,x):return self.b(torch.tanh(self.a(x)))


def build(tmp,kind='loha',max_entries=32):
    torch.manual_seed(38)
    initial=Toy()
    source=tmp/'source';source.mkdir(parents=True)
    save_file({k:v.clone().detach().contiguous() for k,v in initial.state_dict().items()},str(source/'model.safetensors'))
    with torch.device('meta'):m=Toy()
    materialize_meta_from_safetensors(m,source,writable=False)
    a=UniversalAdapter(m,'test','v62',kind=kind,rank=3,target_scope='auto')
    install_disk_linear_offload(m,a,tmp/'disk',min_weight_mib=0.)
    ledger=attach_factor_ledger(m,a,scratch_dir=tmp/'scratch',max_entries=max_entries)
    base=Toy();base.load_state_dict(initial.state_dict())
    torch.manual_seed(111)
    b=UniversalAdapter(base,'test','v62',kind=kind,rank=3,target_scope='auto')
    return source,m,a,ledger,base,b


@pytest.mark.parametrize('kind',['lora','loha'])
def test_real_signed_events_no_dense_writes_and_gradients(tmp_path,kind):
    source,m,a,ledger,reference,b=build(tmp_path,kind)
    pristine=(source/'model.safetensors').read_bytes()
    ja=EventJournal(tmp_path/'journal-a.sqlite','e'*64)
    jb=EventJournal(tmp_path/'journal-b.sqlite','e'*64)
    eng_a=ContinualElasticEngine(a,'alice','e'*64,ja,unsafe=True)
    eng_b=ContinualElasticEngine(b,'alice','e'*64,jb,unsafe=True)
    x=torch.randn(3,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.01)
    m(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    wire=eng_a.seal_local(1)
    assert eng_b.ingest(wire)
    assert len(ledger.sources)==2
    assert ledger.stats()['fallback_bytes']==0
    assert ledger.stats()['factor_bytes']>0
    assert (source/'model.safetensors').read_bytes()==pristine
    assert torch.allclose(m(x),reference(x),atol=1e-5,rtol=1e-5)
    # Both virtual base and signed local anchor now agree for backward input gradient.
    z=x.clone().requires_grad_(True);q=x.clone().requires_grad_(True)
    m(z).square().mean().backward();reference(q).square().mean().backward()
    assert torch.allclose(z.grad,q.grad,atol=1e-5,rtol=1e-5)
    before=m(x).detach().clone()
    assert not eng_a.ingest(wire)
    assert torch.allclose(m(x),before,atol=1e-6)
    ja.close();jb.close()


def test_exact_compaction_then_more_updates_and_direct_shard_export(tmp_path):
    source,m,a,ledger,base,b=build(tmp_path,max_entries=1)
    ja=EventJournal(tmp_path/'a.sqlite','e'*64)
    jb=EventJournal(tmp_path/'b.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'alice','e'*64,ja,unsafe=True)
    ref=ContinualElasticEngine(b,'alice','e'*64,jb,unsafe=True)
    x=torch.randn(4,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.03)
    for i in range(3):
        m(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        wire=eng.seal_local(1)
        assert ref.ingest(wire)
        assert torch.allclose(m(x),base(x),atol=2e-5,rtol=2e-5)
    assert ledger.stats()['compacted_groups']==2
    assert ledger.stats()['factor_bytes']==0
    meta=export_state_sharded(m,tmp_path/'store',tmp_path/'manifest.fmanifest',
        model_id='test',revision='v62',peer_id='alice',seq=3,adapter=a,
        max_state_bytes=2*1024**2)
    assert meta['total_tensor_bytes']>0
    ja.close();jb.close()


def test_shard_export_reads_virtual_factor_not_stale_genesis(tmp_path):
    source,m,a,ledger,base,b=build(tmp_path)
    ja=EventJournal(tmp_path/'a.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'alice','e'*64,ja,unsafe=True)
    x=torch.randn(2,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.02)
    m(x).sum().backward();opt.step();eng.seal_local(1)
    expected=m(x).detach().clone()
    with pytest.raises(LazyWeightError,match='virtual low-rank ledger'):
        m.state_dict()
    meta=export_state_sharded(m,tmp_path/'store',tmp_path/'manifest.fmanifest',
        model_id='test',revision='v62',peer_id='alice',seq=1,adapter=a,
        max_state_bytes=2*1024**2)
    assert ledger.stats()['fallback_bytes']==0
    assert ledger.stats()['ledger_groups']>0
    rebuilt=Toy()
    new_adapter=UniversalAdapter(rebuilt,'test','v62',kind='loha',rank=3,target_scope='auto')
    apply_sharded_effective_state(rebuilt,new_adapter,tmp_path/'manifest.fmanifest',
        tmp_path/'store',model_id='test',revision='v62',max_state_bytes=2*1024**2)
    assert torch.allclose(rebuilt(x),expected,atol=2e-5,rtol=2e-5)
    ja.close()


def test_clean_journal_replay_and_bound_limit(tmp_path):
    source,m,a,ledger,base,b=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True)
    x=torch.randn(2,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.02)
    m(x).sum().backward();opt.step();wire=eng.seal_local(1)
    expected=m(x).detach().clone()
    root=tmp_path/'restart';root.mkdir()
    torch.manual_seed(44)
    with torch.device('meta'):m2=Toy()
    materialize_meta_from_safetensors(m2,source,writable=False)
    a2=UniversalAdapter(m2,'test','v62',kind='loha',rank=3,target_scope='auto')
    install_disk_linear_offload(m2,a2,root/'disk',min_weight_mib=0.)
    l2=attach_factor_ledger(m2,a2,scratch_dir=root/'scratch',max_entries=32)
    j2=EventJournal(root/'j.sqlite','e'*64)
    second=ContinualElasticEngine(a2,'bob','e'*64,j2,unsafe=True)
    assert second.ingest(wire)
    assert torch.allclose(m2(x),expected,atol=1e-5,rtol=1e-5)
    j.close();j2.close()


def test_batched_remote_and_duplicate_safety(tmp_path):
    import copy
    source,m,a,ledger,base,b=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    ref_j=EventJournal(tmp_path/'ref.sqlite','e'*64)
    target=ContinualElasticEngine(a,'target','e'*64,j,unsafe=True)
    reference=ContinualElasticEngine(b,'target','e'*64,ref_j,unsafe=True)
    remote=[]
    for k in (0,1):
        # A source's random local factors need not match the target's initial factors.
        src=Toy();src.load_state_dict({key:value.clone() for key,value in Toy().state_dict().items()})
        src_a=UniversalAdapter(src,'test','v62',kind='loha',rank=3,target_scope='auto')
        src_j=EventJournal(tmp_path/f'r{k}.sqlite','e'*64)
        src_eng=ContinualElasticEngine(src_a,f'origin{k}','e'*64,src_j,unsafe=True)
        with torch.no_grad():
            for group in src_a.groups:
                for p in group.overlay.local.parts:
                    p.add_(torch.randn_like(p)*.02)
        wire=src_eng.seal_local(1)
        remote.append((wire,src_j))
    prepared=[]
    for wire,_ in remote:
        info,parts=target._check(wire)
        prepared.append((info,parts))
        assert reference.ingest(wire)
    # The producer's signed Journal is the authority; the aggregation method
    # is fed only already authenticated/deduplicated records.
    target._apply_checked_many(prepared)
    x=torch.randn(3,10)
    assert torch.allclose(m(x),base(x),atol=1e-5,rtol=1e-5)
    assert ledger.stats()['ledger_groups']==2
    assert ledger.stats()['fallback_bytes']==0
    for _,journal in remote:journal.close()
    j.close();ref_j.close()


def test_bad_nonfinite_source_poisons_nothing_and_refuses_commit(tmp_path):
    from patchlib.core import PatchError
    source,m,a,ledger,base,b=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True)
    x=torch.randn(2,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.01)
    m(x).sum().backward();opt.step()
    wire=eng._make_wire(1,1)
    info,prepared=eng._check(wire)
    # Integrity check for prepared-update values: no factor is committed on
    # overflowing candidate, even if a prevalidated delta is mutated in RAM.
    prepared.deltas[0].view(-1)[0]=float('inf')
    with pytest.raises(PatchError,match='nonfinite'):
        eng._apply_checked(info,prepared)
    assert ledger.stats()['factor_bytes']==0
    j.close()


def test_cli_compatibility_gate():
    from types import SimpleNamespace
    from patchlib.cli import load_model
    args=SimpleNamespace(streaming_base=True,streaming_lazy_write=True,
       streaming_factor_ledger=True,streaming_page_cow=True,
       generation_v2_trainer=False,initialization='pretrained')
    with pytest.raises(ValueError,match='excludes'):
        load_model(args,use_adapter=True)
    # The bounded factor-ledger frozen source is now supported by v2 export;
    # unrelated CLI/model validation remains responsible for the committee.


def test_background_frozen_export_pins_exact_factor_epoch(tmp_path):
    from patchlib.epoch_frozen_export_v58 import capture
    source,m,a,ledger,_,_=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    engine=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True)
    x=torch.randn(4,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.01)
    m(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    engine.seal_local(1)
    exact_at_cut=m(x).detach().clone()
    frozen=capture(journal=j,model=m,adapter=a,max_factor_bytes=8*1024**2)
    assert frozen.frozen_base_sources
    # The trainer progresses after the frozen snapshot was pinned.
    m(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    engine.seal_local(1)
    frozen.assert_immutable_base()
    export_state_sharded(m,tmp_path/'frozen-objects',tmp_path/'frozen.fmanifest',
        model_id='test',revision='v62',peer_id='alice',seq=1,
        adapter=frozen.adapter,frozen_base_sources=frozen.frozen_base_sources,
        max_state_bytes=2*1024**2)
    restored=Toy()
    restored_adapter=UniversalAdapter(restored,'test','v62',kind='loha',rank=3,target_scope='auto')
    apply_sharded_effective_state(restored,restored_adapter,tmp_path/'frozen.fmanifest',
         tmp_path/'frozen-objects',model_id='test',revision='v62',max_state_bytes=2*1024**2)
    assert torch.allclose(restored(x),exact_at_cut,atol=2e-5,rtol=2e-5)
    assert len(next(iter(ledger.sources.values())).entries)==2
    j.close()


def test_capture_rejects_factor_budget(tmp_path):
    from patchlib.epoch_frozen_export_v58 import capture
    from patchlib.core import PatchError
    _,m,a,ledger,_,_=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True)
    x=torch.randn(2,10);opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.01)
    m(x).sum().backward();opt.step();eng.seal_local(1)
    with pytest.raises(PatchError,match='budget'):
        capture(journal=j,model=m,adapter=a,max_factor_bytes=10)
    j.close()


def test_signed_v2_frozen_export_preserves_virtual_cut(tmp_path):
    from patchlib.epoch_frozen_export_v58 import capture
    from patchlib.epoch_export_v57 import export_at_step_boundary,verify_receipt
    from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
    from patchlib.shards import _blob_path
    _,m,a,ledger,_,_=build(tmp_path)
    j=EventJournal(tmp_path/'j.sqlite','e'*64)
    engine=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True)
    x=torch.randn(3,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.01)
    m(x).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    engine.seal_local(1)
    frozen=capture(journal=j,model=m,adapter=a,max_factor_bytes=2*1024**2)
    expected=m(x).detach().clone()
    m(x).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    engine.seal_local(1)
    private=tmp_path/'a.key';public=tmp_path/'a.pub'
    generate_keypair(private,public)
    proposal,pages,receipt=export_at_step_boundary(journal=j,model=m,adapter=a,
       store=tmp_path/'signed-store',incoming=tmp_path/'signed-incoming',
       room='test',epoch=1,parent='0'*64,replicas=['alice','bob'],
       signer=signer_from_pem(private),peer='alice',
       max_state_bytes=2*1024**2,shard_bytes=65536,frozen=frozen)
    assert verify_receipt(receipt,proposal,
        trusted={'alice':verifier_from_pem(public)},committee=['alice','bob'])
    assert frozen.frontier=={'alice':1}
    new=Toy()
    new_adapter=UniversalAdapter(new,'test','v62',kind='loha',rank=3,target_scope='auto')
    apply_sharded_effective_state(new,new_adapter,
       tmp_path/'signed-incoming'/(proposal['manifest']+'.fmanifest'),
       tmp_path/'signed-store',model_id='test',revision='v62',max_state_bytes=2*1024**2)
    assert torch.allclose(new(x),expected,atol=2e-5,rtol=2e-5)
    j.close()


def test_linear_autograd_pins_virtual_revision_during_remote_append():
    from patchlib.factor_ledger_v62 import FactorLedgerSource
    from patchlib.core import FactorDelta
    from patchlib.layer_offload import _FrozenDiskLinear, _FrozenDiskConv1D
    torch.manual_seed(92)
    base=torch.randn(5,7)
    spec=FactorDelta((5,7),'lora',2,2)
    src=FactorLedgerSource(base,spec,max_entries=8)
    x=torch.randn(3,7,requires_grad=True)
    out=_FrozenDiskLinear.apply(x,None,src,tuple(base.shape),base.dtype,None)
    # Simulate a remote signed append between the forward and its backward.
    now=[torch.randn_like(p)*.1 for p in spec.parts]
    old=[torch.zeros_like(p) for p in spec.parts]
    src.append(now,old)
    out.sum().backward()
    assert torch.allclose(x.grad,torch.ones((3,5))@base,atol=1e-5)
    x2=torch.randn(2,5,requires_grad=True)
    old_weight=src.rows(0,5).clone()
    y=_FrozenDiskConv1D.apply(x2,None,src,tuple(base.shape),base.dtype,None)
    src.append([2*v for v in now],old)
    y.sum().backward()
    assert torch.allclose(x2.grad,torch.ones((2,7))@old_weight.T,atol=1e-5)


def test_very_wide_rows_validate_in_bounded_tiles(tmp_path):
    class Wide(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Linear(8193,2,bias=False)
        def forward(self,x):return self.proj(x)
    m=Wide()
    a=UniversalAdapter(m,'wide','base',kind='loha',rank=2,target_scope='all')
    j=EventJournal(tmp_path/'wide.sqlite','e'*64)
    engine=ContinualElasticEngine(a,'alice','e'*64,j,unsafe=True,block_elements=1024)
    with torch.no_grad():
        for group in a.groups:
            for part in group.overlay.local.parts:
                part.add_(.01)
    wire=engine._make_wire(1,1)
    info,prepared=engine._check(wire)
    assert all(value is None for value in prepared.deltas)
    assert engine.ingest(wire)
    assert torch.isfinite(m.proj(torch.randn(1,8193))).all()
    j.close()
