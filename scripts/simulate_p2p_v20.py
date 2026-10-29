#!/usr/bin/env python3
"""Offline deterministic Fedpara network scale/churn simulation (stdlib only)."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
# Import directly: patchlib/__init__.py intentionally imports PyTorch, but the
# network simulator should run even without PyTorch or any third-party app.
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'patchlib'))
from scale_sim import Parameters, NetworkSimulation, write_report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--peers',type=int,default=20)
    p.add_argument('--events-per-peer',type=int,default=6)
    p.add_argument('--duration',type=float,default=16.)
    p.add_argument('--catchup-seconds',type=float,default=45.)
    p.add_argument('--steps-per-s',type=float,default=5.)
    p.add_argument('--local-steps',type=int,default=4)
    p.add_argument('--event-kib',type=int,default=1024)
    p.add_argument('--uplink-mbps',type=float,default=100.)
    p.add_argument('--latency-ms',type=float,default=20.)
    p.add_argument('--loss-probability',type=float,default=0.)
    p.add_argument('--churn-fraction',type=float,default=.10)
    p.add_argument('--compression-ratio',type=float,default=.75)
    p.add_argument('--repair-period',type=float,default=1.)
    p.add_argument('--max-backlog-seconds',type=float,default=30.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fanout',type=int,default=2)
    p.add_argument('--bundle-size',type=int,default=3)
    p.add_argument('--no-adaptive',action='store_true')
    p.add_argument('--compare',action='store_true',help='Compare v18-like raw fanout 8 with v19/v20-like bundles')
    p.add_argument('--output-dir',type=Path,default=Path('network-results/v20'))
    a=p.parse_args(argv)
    c=Parameters(peers=a.peers,events_per_peer=a.events_per_peer,duration_s=a.duration,
                 catchup_s=a.catchup_seconds,steps_per_s=a.steps_per_s,
                 baseline_steps=a.local_steps,event_kib=a.event_kib,
                 uplink_mbps=a.uplink_mbps,latency_ms=a.latency_ms,
                 loss_probability=a.loss_probability,churn_fraction=a.churn_fraction,
                 compression_ratio=a.compression_ratio,repair_period_s=a.repair_period,
                 max_backlog_s=a.max_backlog_seconds,seed=a.seed,
                 fanout=a.fanout,bundle_size=a.bundle_size,adaptive=not a.no_adaptive)
    scenarios={'adaptive-aggregate':c}
    if a.compare:
        scenarios={'flood-8-fixed':replace(c,fanout=8,bundle_size=1,adaptive=False),
                   'sparse-raw-same-cadence':replace(c,bundle_size=1),
                   'adaptive-aggregate':c}
    for name,config in scenarios.items():
        result=NetworkSimulation(config).run()
        summary=write_report(result,a.output_dir,prefix=name)
        print(json.dumps({'scenario':name,**{k:summary[k] for k in (
            'peers','publish_steps','published_events','coverage','exact_fedsum_peers',
            'converged','sent_mib','repair_mib','sent_packets','dropped_packets',
            'p95_arrival_lag_s')}},ensure_ascii=False))
    print('Reports:',a.output_dir.resolve())


if __name__=='__main__':
    main()
