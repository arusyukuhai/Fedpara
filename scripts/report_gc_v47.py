#!/usr/bin/env python3
"""Read-only diagnostic: which Kagami generations are waiting for GC?"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path


def _read_db(p):
    return sqlite3.connect(f'file:{p.resolve()}?mode=ro',uri=True,timeout=3)


def inspect(root:Path):
    root=Path(root)
    runtime=root/'runtime.sqlite3'
    adoption=root/'adoption.sqlite3'
    if not runtime.is_file() or not adoption.is_file():
        raise FileNotFoundError('Expected runtime.sqlite3 and adoption.sqlite3 in a generation-state folder')
    with _read_db(runtime) as r, _read_db(adoption) as a:
        for epoch,state,old,manifest,store,new in r.execute(
            'SELECT epoch,state,old_journal,manifest,store,new_journal FROM handoffs ORDER BY epoch'):
            carried=dict(a.execute('SELECT status,COUNT(*) FROM carries WHERE epoch=? GROUP BY status',(epoch,)))
            ar=a.execute('SELECT state FROM epochs WHERE epoch=?',(epoch,)).fetchone()
            yield {'epoch':epoch,'runtime_state':state,'adoption_state':ar[0] if ar else None,
                   'pending':carried.get('pending',0),'carried':carried.get('carried',0),
                   'old_journal_present':Path(old).is_file() if old else False,
                   'checkpoint_present':Path(manifest).is_file() if manifest else False,
                   'store_present':Path(store).is_dir() if store else False,
                   'new_journal_present':Path(new).is_file() if new else False,
                   'can_attempt_auto_gc':state=='installed' and not carried.get('pending',0)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('generation_state',type=Path,
                   help='Path ending with .generation-state (does not modify files)')
    args=p.parse_args()
    print(json.dumps(list(inspect(args.generation_state)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
