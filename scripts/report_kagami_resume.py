"""Read-only local Kagami restart/log diagnostic. Never opens a journal for writing.

Usage: python scripts/report_kagami_resume.py [KAGAMI_DATA_DIR]
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path

from patchlib.storage_location import choose_data_root


def report(root: Path):
    root=Path(root).expanduser().resolve()
    rooms=root/'.fedpara-gui'/'rooms'
    output={'data_root':str(root),'rooms':[],'issues':[]}
    if not rooms.is_dir():
        output['issues'].append('No model storage at this location. Set KAGAMI_DATA_DIR to the previous Kagami working directory.')
        return output
    for manifest in sorted(rooms.glob('*/manifest.json')):
        entry={'room':manifest.parent.name,'manifest':str(manifest),'peers':[]}
        journals=[manifest.parent/'events.sqlite3']
        journals+=list((manifest.parent/'peers').glob('peer-*/events.sqlite3'))
        for journal in journals:
            if not journal.is_file():continue
            row={'journal':str(journal),'journal_bytes':journal.stat().st_size,
                 'archive_bytes':(Path(str(journal)+'.fplog').stat().st_size
                                  if Path(str(journal)+'.fplog').is_file() else 0),
                 'saved_policy':Path(str(journal)+'.launch.json').is_file()}
            try:
                db=sqlite3.connect(f'file:{journal.resolve()}?mode=ro',uri=True,timeout=1)
                try:
                    info=db.execute("SELECT value FROM meta WHERE name='archive_used_bytes'").fetchone()
                    row['signed_event_bytes']=int(info[0]) if info else None
                    row['event_count']=int(db.execute('SELECT COUNT(*) FROM events').fetchone()[0])
                finally: db.close()
            except (OSError,sqlite3.DatabaseError,ValueError) as exc:
                row['read_error']=str(exc)
            entry['peers'].append(row)
        output['rooms'].append(entry)
    return output


if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else choose_data_root()
    print(json.dumps(report(root),ensure_ascii=False,indent=2))
