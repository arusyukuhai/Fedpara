"""Safe restart from a certified v52 live trainer pointer.

Runs BEFORE opening a new-generation optimizer or accepting peer traffic.
Only locally complete, fully verified data-only checkpoints are accepted.
The old-generation carry archive must have a durable final handoff receipt.
Neither a remote generation-head hint nor a naked pointer is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .core import PatchError
from .live_epoch_switch_v48 import certified_genesis,verify_candidate_files
from .epoch_live_rebase_v52 import verify_certified_cut


@dataclass(frozen=True)
class CertifiedEpochBoot:
    epoch: int
    current_genesis: str
    previous_genesis: str
    manifest_hash: str
    proposal: dict
    manifest: Path
    store: Path
    journal_path: Path
    carry_path: Path
    source_path: Path


def read_durable_epoch_pointer(path: Path):
    path=Path(path)
    if path.is_symlink():
        raise PatchError('unsafe committed epoch pointer symlink')
    if not path.exists():return None
    if not path.is_file() or path.stat().st_size > 4096:
        raise PatchError('unsafe committed epoch pointer')
    try:data=json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise PatchError('corrupt committed epoch pointer') from exc
    def hex64(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
    if (not isinstance(data,dict) or set(data)!={'format','epoch','genesis','checkpoint'}
            or data['format']!='kagami-hot-pointer-v1' or
            type(data['epoch']) is not int or not 1<=data['epoch']<2**32 or
            not hex64(data['genesis']) or not hex64(data['checkpoint'])):
        raise PatchError('invalid committed epoch pointer')
    return data


def resolve_certified_boot(*, root:Path, journal_base:Path, ledger,
                           root_genesis:str, max_state_bytes=2*1024**4):
    """Resolve local bytes for a committed epoch, refusing missing quorum links.

    Expected disk layout is stable across process restart. A staging worker
    stores hardlinked content-addressed checkpoints under
    ``root/checkpoints/views/<manifest-hash>``.
    """
    root=Path(root);journal_base=Path(journal_base)
    pointer=read_durable_epoch_pointer(root/'active-epoch.json')
    if pointer is None:return None
    epoch=pointer['epoch']
    cert=ledger.certified_epoch(epoch)
    if cert is None:
        raise PatchError('active epoch pointer has no locally verified quorum certificate')
    p=cert['proposal']
    if (p.get('epoch')!=epoch or p.get('manifest')!=pointer['checkpoint'] or
            certified_genesis(p)!=pointer['genesis']):
        raise PatchError('pointer does not identify the certified epoch')
    prev_cert=ledger.certified_epoch(epoch-1) if epoch>1 else None
    if epoch>1 and prev_cert is None:
        raise PatchError('missing prior epoch certificate for carry validation')
    previous=certified_genesis(prev_cert['proposal']) if prev_cert else root_genesis
    if p.get('parent') != (ledger.genesis if epoch==1 else __digest(prev_cert['proposal'])):
        raise PatchError('broken certified epoch parent chain')
    view=root/'checkpoints'/'views'/p['manifest']
    manifest=view/'checkpoint.fmanifest'
    verify_candidate_files(SimpleNamespace(manifest=manifest,store=view),p,
                           max_state_bytes=max_state_bytes)
    return CertifiedEpochBoot(epoch,pointer['genesis'],previous,p['manifest'],p,
        manifest,view,root/'journals'/f'epoch-{epoch:08d}.sqlite3',
        root/'carries'/f'epoch-{epoch:08d}.sqlite3',
        journal_base if epoch==1 else root/'journals'/f'epoch-{epoch-1:08d}.sqlite3')


def __digest(p):
    from .online_generation import _digest
    return _digest(p)


def verify_boot_carry(boot:CertifiedEpochBoot, archive, *, old_journal):
    """Reject restart if the source writer made any update after handoff.

    This is a *local* recovery gate, not a proof that every offline peer has
    migrated. Their old signed events still require carry-lane anti-entropy.
    """
    path=Path(str(archive.journal.path)+'.handoff.json')
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4096:
        raise PatchError('missing durable final carry receipt')
    try:receipt=json.loads(path.read_bytes())
    except (ValueError,UnicodeError) as exc:
        raise PatchError('malformed final carry receipt') from exc
    if (not isinstance(receipt,dict) or receipt.get('format')!='kagami-local-carry-attestation-v1'
            or receipt.get('epoch')!=boot.epoch or
            receipt.get('source_genesis')!=boot.previous_genesis or
            receipt.get('target_genesis')!=boot.current_genesis or
            receipt.get('certified_manifest')!=boot.manifest_hash or
            receipt.get('archive_genesis')!=archive.genesis):
        raise PatchError('durable carry receipt mismatches certified epoch')
    if type(receipt.get('source_rowid')) is not int or receipt['source_rowid']<0:
        raise PatchError('invalid certified old-writer boundary')
    cut_count,old_max=verify_certified_cut(old_journal,boot.proposal['cutoffs'],boot.proposal['cut'])
    if old_max!=receipt['source_rowid']:
        raise PatchError('old journal changed after committed epoch handoff')
    total=0;sha=hashlib.sha256()
    from .core import _canonical_json
    for origin,seq,digest in archive.journal.db.execute(
            'SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        sha.update(_canonical_json([origin,seq,digest])+b'\n')
        total+=1
    if (type(receipt.get('total_archived')) is not int or receipt['total_archived']!=total
            or receipt.get('archive_digest')!=sha.hexdigest()):
        raise PatchError('carried signed updates differ from final handoff receipt')
    return True
