#!/usr/bin/env python3
"""Local control-topology audit, not a real WebRTC throughput benchmark."""
from __future__ import annotations
import argparse
from collections import deque
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patchlib.topology import neighbors,topology_metrics


def analyze(n:int, *, degree:int=8):
    roster=tuple(f'p{x:032x}' for x in range(n))
    edges={p: neighbors(p,roster,max_degree=degree) for p in roster}
    dist={roster[0]:0}
    queue=deque([roster[0]])
    while queue:
        p=queue.popleft()
        for q in edges[p]:
            if q not in dist:
                dist[q]=dist[p]+1
                queue.append(q)
    assert len(dist)==n, 'broken connectivity'
    if n == 4096:
        # At capacity, compare a replacement join after one departure.
        new_roster=(*roster[:-1], 'p'+f'{2*n+1:032x}')
        same=roster[:-1]
    else:
        new_roster=(*roster, 'p'+f'{2*n+1:032x}')
        same=roster
    changed=sum(edges[p]!=neighbors(p,new_roster,max_degree=degree)
                for p in same)
    return {**topology_metrics(roster,max_degree=degree),
            'source_mean_hops':round(sum(dist.values())/n,3),
            'source_max_hops':max(dist.values()),
            'existing_peers_rewired_on_join':changed,
            'complete_mesh_links':n*(n-1)//2,
            'not_measured':'latency, bandwidth, WebRTC/TURN and convergence'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--peers',type=int,default=4096,choices=[100,1000,4096])
    a=p.parse_args()
    print(json.dumps(analyze(a.peers),indent=2))

if __name__=='__main__':main()
