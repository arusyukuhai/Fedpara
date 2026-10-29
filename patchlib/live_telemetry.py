"""Local-only, rate-limited training samples for the desktop graph.

No P2P packets, remote peer identifiers, datasets or model tensors are logged.
Each process owns a new file; partial last lines are ignored by the reader.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class LocalTelemetry:
    def __init__(self, path: Path | None, *, sample_seconds: float = 1.0,
                 append: bool = False):
        self.path = Path(path) if path is not None else None
        self.sample_seconds = sample_seconds
        self.started = time.monotonic()
        self.last_written = -float('inf')
        self.last_step_time = self.started
        self.last_step = 0
        self.file = None
        self.elapsed_offset = 0.0
        self.steps_offset = 0
        if self.path is not None:
            if append and self.path.is_file():
                # Single-writer ownership is enforced by the model profile's
                # journal lock. Retain *all* earlier sessions without loading
                # unbounded history into memory.
                old = recent_samples(self.path, max_samples=1)
                if old:
                    prev = old[-1]
                    self.steps_offset = max(0, int(prev['steps']) + 1)
                    gap = max(.001, min(31_536_000., time.time() -
                                float(prev.get('wall_time_epoch',time.time()))))
                    self.elapsed_offset = max(0., float(prev['elapsed_s'])) + gap
                # A crashed writer can leave an incomplete tail. Discard just
                # that malformed line so subsequent appends remain parseable.
                with self.path.open('r+b') as fh:
                    fh.seek(0, os.SEEK_END)
                    end=fh.tell()
                    if end:
                        fh.seek(end-1)
                        if fh.read(1)!=b'\n':
                            pos=max(0,end-16384)
                            fh.seek(pos)
                            data=fh.read(end-pos)
                            cut=data.rfind(b'\n')
                            fh.truncate(pos+cut+1 if cut>=0 else 0)
            # Preserve the unique-create contract for callers that did not
            # explicitly opt into process-restart persistence.
            flags=os.O_WRONLY|os.O_CREAT|(os.O_APPEND if append else os.O_EXCL)|getattr(os,'O_NOFOLLOW',0)
            fd=os.open(self.path,flags,0o600)
            self.file=os.fdopen(fd,'a' if append else 'w',encoding='utf8',buffering=1)

    def record(self, *, steps: int, loss: float, emitted: int,
               inbound_applied: int, inbox: int, channels: int, state: str = 'training',
               force: bool = False, verify_ms: float = 0., apply_ms: float = 0.,
               inbound_duplicates: int = 0, push_mib: float = 0., repair_mib: float = 0.,
               lock_wait_ms: float = 0.,
               journal_ms: float = 0., weight_apply_ms: float = 0.,
               local_seal_ms: float = 0., working_save_ms: float = 0.,
               local_lock_ms: float = 0., local_export_ms: float = 0.,
               local_verify_ms: float = 0., local_journal_ms: float = 0.,
               local_apply_ms: float = 0., publish_steps: int = 4,
               aggregation_events: int = 0, aggregation_batches: int = 0,
               aggregation_saved_mib: float = 0., accuracy: float | None = None,
               peer_id: str | None = None, connected_peers: list[str] | None = None,
               membership_count: int | None = None):
        if self.file is None:
            return False
        now = time.monotonic()
        if not force and now - self.last_written < self.sample_seconds:
            return False
        dt = now - self.last_step_time
        speed = max(0., (steps - self.last_step) / dt) if dt > 0 else 0.
        self.last_step_time, self.last_step = now, steps
        self.last_written = now
        value = float(loss)
        item = dict(accuracy=(float(accuracy) if accuracy is not None and math.isfinite(float(accuracy)) else None),
                    peer_id=peer_id, connected_peers=(connected_peers or [])[:8],
                    membership_count=(int(membership_count) if membership_count is not None else None),
                    wall_time_epoch=round(time.time(),3),
                    elapsed_s=round(self.elapsed_offset+now-self.started,3),
                    steps=self.steps_offset+int(steps),
                    loss=value if math.isfinite(value) else None,
                    steps_per_s=round(speed, 4), emitted=int(emitted),
                    inbound_applied=int(inbound_applied), inbox=int(inbox),
                    channels=int(channels), state=state,
                    verify_ms=round(float(verify_ms), 2),
                    apply_ms=round(float(apply_ms), 2),
                    inbound_duplicates=int(inbound_duplicates),
                    push_mib=round(float(push_mib), 2),
                    repair_mib=round(float(repair_mib), 2),
                    lock_wait_ms=round(float(lock_wait_ms), 2),
                    journal_ms=round(float(journal_ms), 2),
                    weight_apply_ms=round(float(weight_apply_ms), 2),
                    local_seal_ms=round(float(local_seal_ms), 2),
                    local_lock_ms=round(float(local_lock_ms), 2),
                    local_export_ms=round(float(local_export_ms), 2),
                    local_verify_ms=round(float(local_verify_ms), 2),
                    local_journal_ms=round(float(local_journal_ms), 2),
                    local_apply_ms=round(float(local_apply_ms), 2),
                    working_save_ms=round(float(working_save_ms), 2),
                    publish_steps=int(publish_steps),
                    aggregation_events=int(aggregation_events),
                    aggregation_batches=int(aggregation_batches),
                    aggregation_saved_mib=round(float(aggregation_saved_mib),2))
        try:
            self.file.write(json.dumps(item, separators=(',', ':'), allow_nan=False) + '\n')
        except OSError:
            # The visual monitor is best-effort. Disk trouble must not abort
            # the trainer or corrupt the signed training event journal.
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None
            return False
        return True

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def recent_samples(path: Path | None, *, max_samples: int = 320,
                   max_bytes: int = 160 * 1024) -> list[dict]:
    if path is None or not Path(path).is_file():
        return []
    # Tail a bounded amount. Skip first/last partial lines. No unbounded read.
    with Path(path).open('rb') as source:
        source.seek(0, os.SEEK_END)
        length = source.tell()
        source.seek(max(0, length - max_bytes))
        content = source.read(max_bytes)
    lines = content.split(b'\n')
    if length > max_bytes:
        lines = lines[1:]
    # split() produces a final empty item for every properly newline-terminated
    # JSONL file. With max_samples=1 this previously returned [] and disabled
    # all long-range graph fetching, leaving only the 320 recent status rows.
    lines = lines[:-1]  # drop trailing empty item OR unfinished final line
    output = []
    for raw in lines[-max_samples:]:
        if not raw:
            continue
        try:
            item = json.loads(raw)
            if (type(item) is dict and type(item.get('steps')) is int and
                    type(item.get('elapsed_s')) in (float, int) and
                    item.get('state') in ('training','waiting','paused','done')):
                output.append(item)
        except (ValueError, UnicodeError):
            continue
    return output


ALLOWED_HISTORY_WINDOWS = frozenset((0, 300, 1800, 7200, 21600, 86400, 604800))


def history_samples(path: Path | None, *, window_seconds: int = 1800,
                    max_points: int = 420) -> list[dict]:
    """Bounded-I/O plotting for a growing, single-writer JSONL run.

    Binary search by elapsed monotonic seconds, then sample positions across
    the on-disk interval. Number of reads and JSON decodes is O(max_points +
    log(file_bytes)), not O(total training updates). Latest point is preserved.
    This is a graph *preview*, not a lossless export or aggregate statistic.
    """
    if type(window_seconds) is not int or window_seconds not in ALLOWED_HISTORY_WINDOWS:
        raise ValueError('unsupported history period')
    if type(max_points) is not int or not 32 <= max_points <= 2400:
        raise ValueError('invalid history point budget')
    if path is None or not Path(path).is_file():
        return []
    def row_at(f, offset: int, size: int):
        if offset >= size:
            return None, size
        # Skip partial records when seeking into the middle of a JSONL line.
        f.seek(offset)
        if offset:
            f.readline(16384)
        start=f.tell()
        if start >= size:
            return None, start
        raw=f.readline(16384)
        if not raw.endswith(b'\n') or len(raw) >= 16384:
            return None, start
        try:
            item=json.loads(raw)
            if (type(item) is dict and type(item.get('steps')) is int and
                type(item.get('elapsed_s')) in (int,float) and
                math.isfinite(float(item['elapsed_s'])) and
                item.get('state') in ('training','waiting','paused','done')):
                return item,start
        except (ValueError, UnicodeError, TypeError):
            pass
        return None,start
    try:
        with Path(path).open('rb') as f:
            f.seek(0, os.SEEK_END)
            size=f.tell()
            if not size:return []
            # Last complete line (the writer can be appending a partial line).
            latest=recent_samples(path, max_samples=1, max_bytes=16384)
            if not latest:return []
            last=latest[-1]
            target=(float(last['elapsed_s'])-window_seconds) if window_seconds else -float('inf')
            begin=0
            if window_seconds:
                lo,hi=0,size
                for _ in range(40):
                    if hi-lo <= 4096:break
                    mid=(lo+hi)//2
                    sample,pos=row_at(f,mid,size)
                    if sample is None:hi=mid
                    elif float(sample['elapsed_s']) < target:lo=pos+1
                    else:hi=mid
                begin=max(0,lo-16384)
                # Binary search gives a vicinity; scan the bounded 20KiB
                # local region to find the first sample >= requested cutoff.
                f.seek(begin)
                if begin:f.readline(16384)
                for _ in range(128):
                    at=f.tell()
                    if at>=size:break
                    raw=f.readline(16384)
                    if not raw.endswith(b'\n'):break
                    try:
                        obj=json.loads(raw)
                        if float(obj.get('elapsed_s',-1))>=target:
                            begin=at;break
                    except (ValueError,TypeError):pass
            # Equal-byte-position stratified sampling: caps disk I/O even
            # after months of telemetry and preserves the final exact value.
            samples=[];seen=set()
            width=max(0,size-begin)
            for i in range(max_points-1):
                offset=begin+(width*i)//max(1,max_points-1)
                item,position=row_at(f,offset,size)
                if item is None or position in seen or float(item['elapsed_s'])<target:
                    continue
                seen.add(position)
                samples.append(item)
            # The exact most-recent measurement is always included.
            if (not samples or (samples[-1].get('elapsed_s'),samples[-1].get('steps'))
                    != (last['elapsed_s'],last['steps'])):
                samples.append(last)
            return sorted(samples,key=lambda o:(o['elapsed_s'],o['steps']))
    except (OSError,ValueError):
        return []
