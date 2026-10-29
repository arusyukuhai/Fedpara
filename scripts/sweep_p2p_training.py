#!/usr/bin/env python3
"""Run real-gradient P2P training simulations across multiple peer counts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from patchlib.training_sim import run_training_sim


def _parse_peers_list(text: str) -> list[int]:
    peers = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        peers.append(int(item))
    if not peers:
        raise argparse.ArgumentTypeError('expected at least one peer count')
    if len(set(peers)) != len(peers):
        raise argparse.ArgumentTypeError('peer counts must be unique')
    return peers


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--model',choices=['toy','tinystories-1m'],default='tinystories-1m')
    p.add_argument('--dataset',choices=['tinystories','builtin'],default='tinystories')
    p.add_argument('--text-file')
    p.add_argument('--peers-list',type=_parse_peers_list,default='2,4,8,16',
                   help='Comma-separated peer counts, e.g. 2,4,8,16')
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
    p.add_argument('--eval-every',type=int,default=1)
    p.add_argument('--output-dir',default='benchmark-results/p2p-peer-sweep')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    _add_common_args(p)
    a=p.parse_args(argv)
    peers_list = a.peers_list if isinstance(a.peers_list, list) else _parse_peers_list(a.peers_list)
    outdir=Path(a.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    common = vars(a).copy()
    common['adaptive'] = not common.pop('no_adaptive')
    common.pop('output_dir')
    common.pop('peers_list')

    runs=[]
    long_rows=[]
    summary=[]
    for peers in peers_list:
        run_dir = outdir / f'peers-{peers:04d}'
        kw = dict(common)
        kw['peers'] = peers
        kw['output_dir'] = str(run_dir)
        if a.model == 'toy' and a.context > 64:
            kw['context'] = 12
            if a.sequences == 2048:
                kw['sequences'] = 96
        result = run_training_sim(**kw)
        runs.append((peers, result, run_dir))
        for row in result['trace']:
            long_rows.append({
                'peers': peers,
                'round': row['round'],
                'tokens_seen': row['tokens_seen'],
                'central_compute_s': row['central_compute_s'],
                'p2p_compute_sum_s': row['p2p_compute_sum_s'],
                'p2p_compute_div_peers_s': row['p2p_compute_div_peers_s'],
                'p2p_compute_max_peer_s': row['p2p_compute_max_peer_s'],
                'central_valid_loss': row['central_valid_loss'],
                'p2p_mean_valid_loss': row['p2p_mean_valid_loss'],
                'p2p_worst_valid_loss': row['p2p_worst_valid_loss'],
                'p2p_best_valid_loss': row['p2p_best_valid_loss'],
                'updates_published': row['updates_published'],
                'applied_remote_updates': row['applied_remote_updates'],
                'network_sent_mib': row['network_sent_mib'],
                'active_peers': row['active_peers'],
                'min_seen_events': row['min_seen_events'],
                'max_seen_events': row['max_seen_events'],
            })
        last = result['trace'][-1]
        summary.append({
            'peers': peers,
            'publish_steps': result['config']['publish_steps'],
            'tokens_total': result['result']['total_tokens_each_arm'],
            'central_final_loss': result['result']['central_loss'],
            'p2p_mean_final_loss': result['result']['p2p_mean_loss'],
            'p2p_worst_final_loss': result['result']['p2p_worst_loss'],
            'central_compute_s': last['central_compute_s'],
            'p2p_compute_div_peers_s': last['p2p_compute_div_peers_s'],
            'p2p_compute_sum_s': last['p2p_compute_sum_s'],
            'simulated_network_elapsed_s': result['result']['simulated_network_elapsed_s'],
            'network_sent_mib': result['result']['sent_mib'],
            'repair_mib': result['result']['repair_mib'],
            'published_events': result['result']['total_published_events'],
            'all_events_received': result['result']['all_events_received'],
        })

    (outdir/'peer-sweep-result.json').write_text(
        json.dumps({
            'kind':'REAL_GRADIENT_P2P_TRAINING_PEER_SWEEP',
            'peers_list': peers_list,
            'config_template': common,
            'summary': summary,
        }, indent=2, ensure_ascii=False), encoding='utf-8')

    with (outdir/'peer-sweep-summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    with (outdir/'peer-sweep-trace.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader(); w.writerows(long_rows)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for peers, result, _ in runs:
            rows = result['trace']
            ax.plot([r['p2p_compute_div_peers_s'] for r in rows],
                    [r['p2p_mean_valid_loss'] for r in rows],
                    marker='.', markersize=2.5, label=f'P2P mean, peers={peers}')
        ax.set(xlabel='Training compute seconds (sum across peers / peers)',
               ylabel='Heldout cross-entropy',
               title='P2P convergence as peer count increases')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(outdir/'peer-sweep-loss.png', dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        xs = [row['peers'] for row in summary]
        ax.plot(xs, [row['central_final_loss'] for row in summary], marker='o', label='Centralized final loss')
        ax.plot(xs, [row['p2p_mean_final_loss'] for row in summary], marker='o', label='P2P mean final loss')
        ax.plot(xs, [row['p2p_worst_final_loss'] for row in summary], marker='o', linestyle=':', label='P2P worst final loss')
        ax.set(xlabel='Peer count', ylabel='Final heldout cross-entropy',
               title='Final convergence vs peer count')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(outdir/'peer-sweep-vs-peers.png', dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print(json.dumps({'peers_list': peers_list, 'output_dir': str(outdir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
