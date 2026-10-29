#!/usr/bin/env python3
"""Actual-LM P2P training convergence simulation with virtual Gossip network."""
from __future__ import annotations
import argparse
import json
from patchlib.training_sim import run_training_sim


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['toy','tinystories-1m'],default='tinystories-1m')
    p.add_argument('--dataset',choices=['tinystories','builtin'],default='tinystories')
    p.add_argument('--text-file')
    p.add_argument('--peers',type=int,default=5)
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--local-steps',type=int,default=4)
    p.add_argument('--reinit-every',type=int,default=20)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--wide-lora-rank',type=int,default=4)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--vocab',type=int,default=32)
    p.add_argument('--sequences',type=int,default=2048)
    p.add_argument('--validation-sequences',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    p.add_argument('--fanout',type=int,default=2)
    p.add_argument('--bundle-size',type=int,default=3)
    p.add_argument('--uplink-mbps',type=float,default=600.)
    p.add_argument('--latency-ms',type=float,default=20.)
    p.add_argument('--loss-probability',type=float,default=0.)
    p.add_argument('--churn-fraction',type=float,default=0.)
    p.add_argument('--repair-every',type=float,default=1.)
    p.add_argument('--sim-steps-per-second',type=float,default=5.)
    p.add_argument('--compression-ratio',type=float,default=.75)
    p.add_argument('--max-outstanding-mib',type=float,default=512.)
    p.add_argument('--no-adaptive',action='store_true')
    p.add_argument('--catchup-seconds',type=float,default=30.)
    p.add_argument('--eval-every',type=int,default=1,help='Validation sampling interval in rounds (default: every round)')
    p.add_argument('--output-dir',default='benchmark-results/p2p-training')
    a=p.parse_args(argv)
    kw=vars(a)
    kw['adaptive']=not kw.pop('no_adaptive')
    if a.model=='toy' and a.context>64:
        # --model toy should work without extra arguments.
        kw['context']=12
        if a.sequences==2048:kw['sequences']=96
    result=run_training_sim(**kw)
    print(json.dumps(result['result'],indent=2,ensure_ascii=False))
    print('Saved results in:',a.output_dir)


if __name__=='__main__':
    main()
