"""FedSum v3 regression tests: signed event replay, strict no-division semantics."""
from copy import deepcopy
import torch
import pytest
from patchlib.continual import ContinualElasticEngine, PROTOCOL
from patchlib.core import PatchError, encode_patch
from patchlib.elastic import unpack_event, pack_event
from patchlib.effectiveness_benchmark import _sum_effective_peer_updates, run_effectiveness_benchmark
from test_continual_elastic import make_engine, bases, effective_delta_after, loss_step
from test_livefed import Tiny


def test_p2p_effective_delta_sum_is_not_mean():
    dst={'a':torch.zeros(2,2), 'b':torch.zeros(3)}
    u1={'a':torch.tensor([[1.,2.],[3.,4.]]), 'b':torch.tensor([3.,4.,5.])}
    u2={'a':torch.tensor([[7.,6.],[5.,4.]]), 'b':torch.tensor([2.,1.,0.])}
    _sum_effective_peer_updates(dst,u1)
    _sum_effective_peer_updates(dst,u2)
    assert torch.equal(dst['a'],u1['a']+u2['a'])
    assert torch.equal(dst['b'],u1['b']+u2['b'])
    assert not torch.equal(dst['a'],(u1['a']+u2['a'])/2)
    with pytest.raises(ValueError,match='inventory'):
        _sum_effective_peer_updates(dst,{'a':torch.ones(2,2)})


def test_signed_v3_effects_are_exact_sum_and_dedup(tmp_path):
    torch.manual_seed(20261023)
    shared=Tiny()
    a=make_engine(tmp_path,'sum-alice',deepcopy(shared)); b=make_engine(tmp_path,'sum-bob',deepcopy(shared))
    observer=make_engine(tmp_path,'sum-third',deepcopy(shared))
    baseline=bases(observer)
    opt_a=torch.optim.AdamW(list(a.adapter.trainable_parameters()),lr=.001)
    opt_b=torch.optim.AdamW(list(b.adapter.trainable_parameters()),lr=.001)
    loss_step(a,opt_a); loss_step(b,opt_b)
    da=effective_delta_after(a); db=effective_delta_after(b)
    wa=a.seal_local(1);wb=b.seal_local(1)
    assert observer.ingest(wa)
    assert observer.ingest(wb)
    assert not observer.ingest(wb)  # a replay must never double-sum
    for orig, delta_a, delta_b, final in zip(baseline,da,db,bases(observer)):
        torch.testing.assert_close(final,orig+delta_a+delta_b,rtol=2e-5,atol=2e-6)
        if (delta_a+delta_b).abs().max()>1e-7:
            assert not torch.allclose(final,orig+(delta_a+delta_b)/2,rtol=1e-6,atol=1e-8)
    _,meta,parts,_=unpack_event(wa,{},unsafe=True)
    assert meta['aggregation']=='sum' and meta['denominator']==1 and meta['elastic_protocol']==PROTOCOL
    # A v2 averaging event is not a valid v3 event.
    meta['elastic_protocol']='fedpara-continual-elastic-v2'
    forged=pack_event(encode_patch(meta,parts),'sum-alice')
    with pytest.raises(PatchError,match='continual protocol'):
        observer._check(forged)
    with pytest.raises(PatchError,match='denominator=1'):
        a.seal_local(2)
    a.journal.close();b.journal.close();observer.journal.close()


def test_offline_benchmark_exports_explicit_fedsum(tmp_path):
    result=run_effectiveness_benchmark(model='toy',peers=2,rounds=1,local_steps=1,
              rank=4,seed=7911,output_dir=tmp_path)
    assert result['configuration']['p2p_aggregation']=='sum'
    assert result['configuration']['p2p_aggregation_scale']==1.
    assert 'SUM' in result['fairness']['peer_aggregation']
    assert result['fairness']['identical_total_tokens']
