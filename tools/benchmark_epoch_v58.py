"""Synthetic single-host fence benchmark; not a 4096-device WAN benchmark."""
import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from patchlib.core import UniversalAdapter
from patchlib.epoch_frozen_export_v58 import capture,finish_deferred_frontier


def run(origins=4096,steps=5):
    with tempfile.TemporaryDirectory(prefix='kagami-v58bench-') as temp:
        path=Path(temp)/'index.db'
        db=sqlite3.connect(path)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT,PRIMARY KEY(origin,seq))')
        rows=((f'p{i:05d}',seq,hashlib.sha256(f'{i}:{seq}'.encode()).hexdigest())
              for i in range(origins) for seq in range(1,steps+1))
        db.executemany('INSERT INTO events VALUES(?,?,?)',rows)
        db.commit()
        model=torch.nn.Linear(128,128)
        adapter=UniversalAdapter(model,'benchmark','v58',kind='loha',rank=8)
        clock=time.perf_counter
        start=clock()
        full=capture(journal=SimpleNamespace(db=db),model=model,adapter=adapter)
        full_ms=(clock()-start)*1000
        frozen_db=sqlite3.connect(path,check_same_thread=False)
        try:
            frozen_db.execute('BEGIN')
            frozen_db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()
            start=clock()
            fast=capture(journal=SimpleNamespace(db=frozen_db),model=model,adapter=adapter,
                         defer_frontier=True)
            fence_ms=(clock()-start)*1000
            start=clock()
            fast=finish_deferred_frontier(fast,frozen_db)
            scan_ms=(clock()-start)*1000
            assert fast.inventory==full.inventory
        finally:frozen_db.close();db.close()
        return {'origins':origins,'indexed_events':origins*steps,
                'legacy_fenced_capture_ms':round(full_ms,3),
                'v58_factor_fence_ms':round(fence_ms,3),
                'v58_background_index_ms':round(scan_ms,3),
                'frozen_factor_bytes':fast.factor_bytes,
                'note':'Synthetic metadata and tiny torch model; not model export or WAN throughput.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--origins',type=int,default=4096)
    p.add_argument('--steps',type=int,default=5)
    args=p.parse_args()
    print(json.dumps(run(args.origins,args.steps),indent=2))
