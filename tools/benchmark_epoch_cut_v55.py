#!/usr/bin/env python3
"""REPRODUCIBLE LOCAL METADATA-ONLY cut/receipt CPU benchmark.

No real signature verification, large tensor payload, WebRTC, or full 4k-peer
swarm is exercised. Synthetic SQLite rows intentionally have NO valid wire
archive; NEVER reuse the benchmark journal in production or for an epoch.

Run: python tools/benchmark_epoch_cut_v55.py --events 20000 80000 --repeats 20
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from patchlib.core import _canonical_json
from patchlib.elastic import EventJournal
from patchlib.epoch_live_rebase_v52 import StreamingCertifiedRebase, verify_certified_cut


def bench(size, repeats, origins):
    with TemporaryDirectory(prefix='kagami-v55-benchmark-') as root:
        journal=EventJournal(Path(root)/'source.db','a'*64,sync='NORMAL')
        try:
            per_origin=size//origins
            digest=hashlib.sha256()
            entries=[]
            for origin_index in range(origins):
                origin=f'peer{origin_index:06d}'
                for seq in range(1,per_origin+1):
                    event_hash=hashlib.sha256(f'{origin}/{seq}'.encode()).hexdigest()
                    digest.update(_canonical_json([origin,seq,event_hash])+b'\n')
                    entries.append((origin,seq,event_hash,b'',1))
            with journal.db:
                journal.db.executemany('INSERT INTO events (origin,seq,digest,wire,wire_len) VALUES(?,?,?,?,?)',entries)
            cutoffs={f'peer{i:06d}':per_origin for i in range(origins)}
            runner=object.__new__(StreamingCertifiedRebase)
            runner.source=journal
            runner.proposal={'cutoffs':cutoffs,'cut':digest.hexdigest()}
            runner._poisoned=False
            runner._validated=False
            runner._verified_cut=None
            runner._cut_guard=None
            runner._cached_inventory=None
            def measure(fn):
                t0=perf_counter_ns();r=fn();return (perf_counter_ns()-t0)/1_000_000,r
            first,result=measure(runner.validate_cut)
            baseline=[measure(lambda:verify_certified_cut(journal,cutoffs,digest.hexdigest()))[0] for _ in range(repeats)]
            cached=[measure(runner.validate_cut)[0] for _ in range(repeats)]
            # One post-cut row (new peer) must not invalidate a verified prefix.
            with journal.db:
                journal.db.execute('INSERT INTO events (origin,seq,digest,wire,wire_len) VALUES(?,?,?,?,?)',
                                   ('zz-new',1,hashlib.sha256(b'new').hexdigest(),b'',1))
            suffix,cached_result=measure(runner.validate_cut)
            assert cached_result==result
            return {'rows':size,'origins':origins,'first_verified_ms':round(first,4),
                    'full_rescan_median_ms':round(median(baseline),4),
                    'witness_check_median_ms':round(median(cached),4),
                    'witness_after_suffix_ms':round(suffix,4),
                    'speed_ratio_full_to_cached':round(median(baseline)/max(0.00001,median(cached)),1),
                    'benchmark_kind':'local_metadata_only_not_swarm'}
        finally:
            journal.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--events',type=int,nargs='+',default=[20000,80000])
    parser.add_argument('--origins',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    if not 1<=args.origins<=256 or not 1<=args.repeats<=200:
        parser.error('invalid repetitions or origin count')
    for size in args.events:
        if size<args.origins or size%args.origins or size>250_000:
            parser.error('event count must be multiple of origins, <=250000')
        print(json.dumps(bench(size,args.repeats,args.origins),sort_keys=True),flush=True)

if __name__=='__main__':main()
