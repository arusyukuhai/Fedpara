#!/usr/bin/env python3
"""Read-only report of Kagami model IDs with identical immutable manifests.

This does not delete, merge, move, or modify checkpoints or journal files.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patchlib.room_manifest import load_manifest, manifest_digest


def duplicate_groups(workdir: Path) -> list[dict]:
    base = Path(workdir) / '.fedpara-gui' / 'rooms'
    groups: dict[str,list[dict]] = collections.defaultdict(list)
    if not base.exists():
        return []
    for path in sorted(base.glob('*/manifest.json')):
        if path.is_symlink():
            continue
        try:
            manifest = load_manifest(path)
            if manifest.get('version') != 2:
                continue
            digest = manifest_digest(manifest)
            groups[digest].append({
                'room_id': path.parent.name,
                'model': manifest['model']['id'],
                'revision': manifest['model']['revision'],
                'datasets': [source['repo'] for source in manifest['datasets']['sources']],
                'path': str(path),
            })
        except (OSError, ValueError, KeyError):
            continue
    return [
        {'manifest_sha256': digest, 'count': len(rows), 'rooms': rows}
        for digest, rows in sorted(groups.items()) if len(rows) > 1
    ]


def main():
    parser=argparse.ArgumentParser(description='List identically configured Kagami model rooms without changing them')
    parser.add_argument('--workdir',default='.',help='Same workdir used to launch Kagami')
    args=parser.parse_args()
    result=duplicate_groups(Path(args.workdir))
    print(json.dumps({'duplicate_groups':result,'group_count':len(result),'read_only':True},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
