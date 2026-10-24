"""Fair, reproducible 8-peer *optimizer* A/B on the same synthetic task.

Unlike the separately supplied REAL aiortc swarm harness, this runs mathematical
training updates in one process to isolate optimization quality from networking.
This must never be reported as real P2P, real LLM training, or a convergence proof.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from .elastic_v2 import EpochBuffer, EpochProposal


def run_optimizer_benchmark(*, peers=8, epochs=24, seed=20261022,
                            local_steps=4, lr=0.04, ridge=0.):
    if not 2 <= peers <= 96 or not 1 <= epochs <= 500 or not 1 <= local_steps <= 100:
        raise ValueError('invalid benchmark dimensions')
    if not 0 < lr <= 1 or not 0 <= ridge <= 100:
        raise ValueError('invalid optimizer hyperparameters')
    cpu_start=time.monotonic()
    gen=torch.Generator().manual_seed(seed)
    dimensions=12
    target=torch.randn((dimensions,1),generator=gen)
    data=[]
    # Pin per-peer heterogeneity and a held-out distribution. Neither
    # algorithm receives additional samples or a preferred data order.
    for i in range(peers):
        x=torch.randn((128,dimensions),generator=gen)+i*.07
        y=x@target+torch.randn((128,1),generator=gen)*.08
        data.append((x,y))
    hold_x=torch.randn((512,dimensions),generator=gen)
    hold_y=hold_x@target
    genesis=torch.zeros((dimensions,1))
    w_legacy,w_v2=genesis.clone(),genesis.clone()
    trace=[]
    epoch_durations={'legacy':0.,'v2':0.}

    def fit_delta(base:torch.Tensor, peer:int, epoch:int) -> torch.Tensor:
        x,y=data[peer]
        local=base.detach().clone().requires_grad_(True)
        for step in range(local_steps):
            # Deterministic, peer-specific samples; same schedule in both modes.
            start=(epoch*local_steps*16 + step*16 + peer*13) % x.shape[0]
            indices=(torch.arange(16)+start)%x.shape[0]
            err=x[indices]@local-y[indices]
            loss=(err.square()).mean()
            (gradient,)=torch.autograd.grad(loss,local)
            local=(local-lr*gradient).detach().requires_grad_(True)
        return (local.detach()-base)

    with torch.enable_grad():
        for epoch in range(epochs):
            # Two deterministic churn epochs: two absent participants, then rejoin.
            dropped={(seed+epoch)%peers,(seed+epoch+3)%peers} if epoch%5==2 else set()
            participants=[i for i in range(peers) if i not in dropped]
            # Old event-at-arrival behavior (counterfactual central delivery).
            t=time.monotonic()
            for peer in participants:
                delta=fit_delta(w_legacy,peer,epoch)
                w_legacy=w_legacy+delta/len(participants)
            epoch_durations['legacy']+=time.monotonic()-t
            # V2 microepoch: all updates reference ONE immutable W_epoch.
            t=time.monotonic()
            epoch_sha=hashlib.sha256(f'{seed}/{epoch}'.encode()).hexdigest()
            buffer=EpochBuffer(epoch_sha,{'weight':tuple(genesis.shape)},
                               max_updates_per_peer=1,max_events=peers)
            for peer in participants:
                delta=fit_delta(w_v2,peer,epoch)
                digest=hashlib.sha256(delta.contiguous().numpy().tobytes()).hexdigest()
                buffer.add(EpochProposal(f'p{peer}',epoch+1,epoch_sha,16*local_steps,
                                         {'weight':delta},digest))
            w_v2=buffer.materialize({'weight':w_v2},ridge=ridge,seal=True)['weight']
            epoch_durations['v2']+=time.monotonic()-t
            trace.append({'epoch':epoch+1,'online':len(participants),
                          'legacy_heldout_mse':float(((hold_x@w_legacy-hold_y)**2).mean()),
                          'v2_heldout_mse':float(((hold_x@w_v2-hold_y)**2).mean()),
                          'v2_event_set_hash':buffer.digest()})
    return {'status':'SYNTHETIC_OPTIMIZER_BENCHMARK','peers':peers,'epochs':epochs,
            'seed':seed,'lr':lr,'local_steps':local_steps,'ridge':ridge,
            'loss_first':{'legacy':trace[0]['legacy_heldout_mse'],'v2':trace[0]['v2_heldout_mse']},
            'loss_last':{'legacy':trace[-1]['legacy_heldout_mse'],'v2':trace[-1]['v2_heldout_mse']},
            'optimizer_time_s':epoch_durations,'elapsed_s':time.monotonic()-cpu_start,
            'trace':trace,
            'warning':'In-process synthetic SGD study, NOT WebRTC and NOT real LLM training. No quality advantage is assumed.'}
