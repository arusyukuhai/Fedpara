"""Compare v13-style SQLite BLOB writes with v14 external frames.

Run: python tools/bench_event_archive_v14.py --count 20 --mib 1 --sync normal
Outputs a local-only, environment-specific microbenchmark. No P2P traffic.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from patchlib.elastic import EventJournal


def run(count=20, mib=1, sync='NORMAL'):
    size = int(mib*1024*1024)
    payload = bytes((i % 251 for i in range(size)))
    with tempfile.TemporaryDirectory(prefix='fedpara-v14-bench-') as directory:
        root = Path(directory)
        # v13-style: one BIG signed bytes BLOB for each SQLite WAL transaction.
        db = sqlite3.connect(root/'legacy.sqlite3')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous='+sync)
        db.execute('PRAGMA wal_autocheckpoint=0')
        db.execute('CREATE TABLE events(origin TEXT,seq INT,digest TEXT,wire BLOB,wire_len INT,PRIMARY KEY(origin,seq))')
        start=time.perf_counter()
        for seq in range(1,count+1):
            with db:
                db.execute('INSERT INTO events VALUES (?,?,?,?,?)',
                           ('alice',seq,hashlib.sha256(payload).hexdigest(),payload,len(payload)))
        legacy=time.perf_counter()-start
        db.close()
        j=EventJournal(root/'new.sqlite3','a'*64,sync=sync)
        start=time.perf_counter()
        try:
            for seq in range(1,count+1):
                j.append('alice',seq,payload)
            modern=time.perf_counter()-start
            assert j.count()==count
            assert j.get('alice',count)==payload
        finally:
            j.close()
        print(f'{sync} | {count} events x {mib} MiB; '
              f'legacy SQLite WAL={legacy:.3f}s | v14 sidecar={modern:.3f}s | '
              f'ratio={legacy/modern:.2f}x (higher is better)')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--mib',type=float,default=1)
    parser.add_argument('--sync',choices=['full','normal'],default='normal')
    opts=parser.parse_args()
    if opts.count<1 or not 0<opts.mib<=64:
        parser.error('count>=1 and 0<mib<=64 required')
    run(opts.count,opts.mib,opts.sync.upper())
