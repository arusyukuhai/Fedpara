# Fedpara v14 — crash-consistent binary event archive

This revision targets slow LOCAL JOURNAL/JOURNAL WRITE on dual-window training.
It preserves the signed wire protocol, FedSum, LoHa/LoRA, ReLoRA and room genesis.

## Storage layout

For `--journal /path/to/events.sqlite3`, **two files together form one journal**:

- `events.sqlite3` (+ SQLite `-wal`/`-shm` while running): small metadata, `origin`, `seq`, `digest`, `wire_len`, `log_offset`, `log_size`, plus existing `meta`.
- `events.sqlite3.fplog`: append-only signed payload frames `[FPEV14\0\0 | u64 length | SHA256 | signed wire]`.

The SQLite schema retains the old `wire BLOB NOT NULL` column for migration; for
**new** events its value is `b''`. Older events that are already stored inline
continue to work. The wire bytes transmitted over the network have NOT changed.

**Back up the SQLite database, its WAL if active, and the `.fplog` together.**
Never delete `.fplog` or run v13 against a journal written by v14: v13 cannot
read the sidecar payloads. All programs sharing a local journal must use v14.
Different P2P nodes may use different on-disk journal formats because the
wire protocol is unchanged.

## Crash contract

Each event is committed as follows:

1. Verify event signature and update delta (unchanged).
2. Append raw frame to `.fplog` (no extra `wire` copy).
3. In **FULL** mode, fsync the `.fplog` frame before committing SQLite index.
4. Commit the small SQLite index and byte counter (SQLite synchronous=FULL).
5. Only then apply the update/ACK its successful commit.

The dedicated single-thread disk writer uses a persistent SQLite connection,
and an eight-item bounded async submission window. WebRTC/ICE/heartbeat and
verification tasks can continue while writes execute. Model application and
local seal still **await committed durability**, to avoid fabricating progress
that is not crash-replayable.

A crash between steps 2 and 4 may leave an unindexed tail. On next exclusive
open it is truncated to the last committed record. If the indexed last frame
is missing/truncated/corrupt, startup fails closed; older frames are checked
when replayed or fetched. FULL offers the strongest power-loss durability.
NORMAL is explicitly lower durability for experiments: SQLite and the sidecar
may lose their most recent committed updates on a power loss/OS crash.

## Existing v13 journals

v14 opens them without blocking startup on a full migration. Legacy rows stay
inline; all new rows are archived in `.fplog`.

Optionally migrate old BLOBs **while no trainer is running** (back up first):

```bash
python -m patchlib.event_archive migrate /path/to/events.sqlite3
```

Pass `--vacuum` to reclaim old SQLite database pages afterward. VACUUM may
need significant free disk space and time; back up the journal and `.fplog`.
Migration is resumable if interrupted; each moved event is committed separately.

## Throughput and compatibility

The former v13 default (`--journal-sync full`) remains the default; no silent
weakening of durability. The hot path avoids inserting large wire BLOBs into
SQLite WAL, avoids repeated SQLite connections, and avoids making an additional
full-sized `wire` copy for the archive frame. Actual laptop speedups depend on
SSD fsync latency and concurrent CPU/GPU contention; test FULL and NORMAL with
the same workload before drawing conclusions.

The GUI, local metrics and signed GOSSIP event format remain as in v13.
Separate progress files and individual peer journals continue to work.
The new `events.sqlite3.fplog` is essential for restart and anti-entropy.

## Synthetic microbenchmark

Run from the `fedpara` directory (storage-only, not a WebRTC throughput test):

```bash
PYTHONPATH=. python tools/bench_event_archive_v14.py --count 16 --mib 1 --sync normal
PYTHONPATH=. python tools/bench_event_archive_v14.py --count 16 --mib 1 --sync full
```

On this build environment, 16 x 1 MiB events: NORMAL 1.87x and FULL 1.38x
relative to a v13-style SQLite BLOB WAL loop after eliminating duplicate SHA
calculation. However, **12 x 5 MiB events in NORMAL mode was 0.86x** (i.e.,
sidecar slower than BLOB WAL). Real SSD/Mac results can differ substantially.
The implementation is primarily a safer, scalable basis for further I/O
pipelining; it is not a guarantee that all workloads will be faster.
