"""Tests of real-gradient training and exact-once asynchronous FedSum simulation."""
import json

import pytest
import torch

from patchlib.core import UniversalAdapter
from patchlib.continual import enable_anchors
from patchlib.effectiveness_benchmark import TinyCausalLM
from patchlib.training_sim import Event, Peer, _apply_events, _commit_local, _original, run_training_sim


def _peer(seed=1):
    torch.manual_seed(seed)
    model=TinyCausalLM(32,24,8)
    adapter=UniversalAdapter(model,'toy','test',kind='loha',rank=16,
                             target_scope='auto',wide_lora_rank=4)
    enable_anchors(adapter)
    optimizer=torch.optim.AdamW(adapter.trainable_parameters(),lr=1e-3)
    return Peer(model,adapter,optimizer)


def test_mixed_adapter_types_and_no_double_apply():
    p=_peer()
    assert any(g.overlay.local.kind=='lora' for g in p.adapter.groups)
    assert any(g.overlay.local.kind=='loha' for g in p.adapter.groups)
    with torch.no_grad():
        for group in p.adapter.groups:
            for part in group.overlay.local.parts:
                part.add_(.001)
    local=_commit_local(p,0.1);local.origin=0
    before={g.canonical:_original(p.model,g).clone() for g in p.adapter.groups}
    remote_a=Event(1,1,{k:torch.ones_like(x)*.01 for k,x in local.deltas.items()},100,0.)
    remote_b=Event(2,1,{k:torch.ones_like(x)*.02 for k,x in local.deltas.items()},100,0.)
    assert _apply_events(p,[remote_a,remote_a,remote_b])==2
    assert _apply_events(p,[remote_b,remote_a])==0
    for g in p.adapter.groups:
        assert torch.allclose(_original(p.model,g),before[g.canonical]+.03,atol=1e-6)


def test_async_training_churn_reinit_and_catchup(tmp_path):
    cfg=dict(model='toy', peers=5, rounds=8, local_steps=4, reinit_every=2,
             batch_size=4,context=8,seed=12,rank=16,wide_lora_rank=4,
             sequences=48,validation_sequences=16,threads=1,
             churn_fraction=.4,loss_probability=.1,repair_every=.15,
             sim_steps_per_second=5,latency_ms=20,uplink_mbps=30,
             catchup_seconds=20,eval_every=2,output_dir=tmp_path)
    out=run_training_sim(**cfg)
    assert out['result']['all_events_received']
    assert out['result']['remote_updates_applied']==(cfg['peers']-1)*out['result']['total_published_events']
    assert out['result']['outstanding_events']==0
    assert out['result']['initial_loss']>out['result']['central_loss']
    assert out['result']['initial_loss']>out['result']['p2p_mean_loss']
    assert max(out['result']['p2p_losses']) - min(out['result']['p2p_losses']) < 1e-4
    assert out['config']['publish_steps']==4
    assert (tmp_path/'p2p-training-loss.png').is_file()
    assert (tmp_path/'p2p-training-trace.csv').is_file()
    assert json.loads((tmp_path/'p2p-training-result.json').read_text())['kind']==out['kind']


def test_invalid_conditions():
    with pytest.raises(ValueError):
        run_training_sim(model='toy',peers=1)
    with pytest.raises(ValueError):
        run_training_sim(model='toy',uplink_mbps=0)
    with pytest.raises(ValueError):
        run_training_sim(model='toy',loss_probability=1)


def test_low_bandwidth_reports_not_converged():
    out=run_training_sim(model='toy',peers=3,rounds=3,local_steps=2,
        reinit_every=1,validation_sequences=16,sequences=32,context=8,
        seed=123,eval_every=3,uplink_mbps=.01,catchup_seconds=.1)
    assert out['result']['total_published_events']==9
    assert not out['result']['all_events_received']
    assert out['result']['outstanding_events']>0
