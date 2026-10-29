# Kagami v55 — certified-cut incremental reuse and measured epoch hot path (2026-10-27)

## What changed from v54

**A previously verified certified cut no longer requires O(total indexed events) CPU work again at final handoff, provided its trusted local SQLite audit witness has not changed.** The very first check remains the exact v52 canonical SHA-256 scan, including validation of every per-origin contiguous prefix against the *independently certified* cut digest. There is **no waiver** for missing certified rows and **no new consensus authority**.

1. `EventJournal` now installs durable SQLite mutation-audit triggers. UPDATE and DELETE of historical events bump `meta.epoch_index_mutation` *in the same transaction*. An INSERT guard refuses `INSERT OR REPLACE` for existing IDs; this closes SQLite's default non-recursive REPLACE loophole. Purely new append-only events do **not** bump the mutation counter. `epoch_cut_guard()` returns the SQLite schema version plus mutation revision.
2. `StreamingCertifiedRebase.validate_cut()` performs the full certificate check on first use. After success, repeat validation can reuse that exact digest only when schema and historic-event mutation revisions are unchanged. Because the entire high-water prefix was complete and composite IDs are unique, a new `(origin, seq)` INSERT cannot add a missing in-prefix row. If the guard changes, the code makes a **full check**; if it fails after previous success, it poisons the shadow and refuses further training. `tick()` checks the witness before changing shadow weights.
3. Handoff archive's canonical digest is likewise computed at least once. It can be reused when a local stamp (`sqlite3.total_changes`, `PRAGMA data_version`, `PRAGMA schema_version`) remains identical, detecting writes from the archive's thread-owned SQLite worker as well as its caller connection. On change, the full original sorted digest is recomputed. A concurrent change *during* a scan aborts rather than attesting an unstable snapshot.
4. Warm rebase reads the actual signed event payload using `EventJournal.get_async` (or `asyncio.to_thread` for compatible custom journals). Multi-megabyte event sidecar reads and hashing no longer block the trainer watcher loop.
5. Added 11 focused regression tests: cached verified prefix, post-cut append invariance, historical tamper from same/external connection, unsafe SQL REPLACE, schema trigger removal, legacy journal fallback, archive cache invalidation from worker writes or inline tampering, shadow poisoning, and asynchronous signed-wire carry.

### Correctness assumptions / fail-closed boundary

- Reuse is an **incremental *local audit* of an already authenticated SHA-256 commitment**, not a newly constructed Merkle proof and not a substitute for independently signed generation/cut certificates. SQLite is trusted local storage; an attacker with arbitrary DDL and direct DB-file mutation can bypass local audit machinery, so this is NOT an adversarial storage proof. The old whole-scan validation remains the required entry condition.
- The SQLite primary key and replacement guard make verified contiguous signed frontiers append-immutable, unless historical mutations are recorded. If a source lacks the guard, **the optimization is disabled** and full validation is retained.
- A writer/ingress barrier is still required at final handoff. The cached digest does not attest remote consensus, payload bytes or disk-media bitrot. It does not authorize GC.
- In a multi-process deployment, acquire the existing exclusive journal writer lock; do not run handoff against a separately mutated raw SQLite file.
- Existing `verify_candidate_files`, shard hashes, TLS/signature verification, and final network rekey remain unchanged. This revision does not skip their expensive but necessary work.

## Measured performance

`PYTHONPATH=. python tools/benchmark_epoch_cut_v55.py --events 20000 80000 --repeats 20`

| Synthetic indexed rows | Old full certificate-cut scan (median) | v55 trusted local witness check (median) | Initial complete scan |
| ---: | ---: | ---: | ---: |
| 20,000 | 115.318 ms | 0.0044 ms | 113.1082 ms |
| 80,000 | 443.3382 ms | 0.0046 ms | 474.6094 ms |

This benchmark measures **only in-memory-cache-friendly SQLite metadata and SHA-256 overhead** on the local test machine, with 20 synthetic origins, and excludes signatures, Tensor load, fsync, checkpoint verification, transport, consensus and WAN latency. The original full scan **still runs once** per newly prepared epoch or after an audit change. Numbers are not a forecast for 4,096 devices or overall training throughput.

A separate local `sync=NORMAL`, 512-byte payload append micro-benchmark (3 repeats, median) showed normal 4,000-event appends increasing from **129.2 ms (v54)** to **147.0 ms (v55)** (~14% in this fast synthetic microcase) due to the SQLite duplicate-ID guard and triggers. On real `sync=FULL` with durable fsync and larger updates, relative overhead must be measured separately; no speedup claim is made for append throughput.

## Still incomplete

- 4,096-origin/peer autonomous quorum checkpoint certification and proof of a globally safe cut. Multi-generation offline reconciliation and globally bounded log storage with never-rejoining peers.
- Production validation on a 4,096-device real WAN swarm. Zero-pause 300B-class full model/optimizer swap.
- Repeated full model-shard file hashing during checkpoint installation, plus long replay time when the *first* cut is huge. These intentionally remain exact, not guessed from file metadata.

See `VALIDATION_V55.md` for test output, `CHANGES_V55.patch` for the source changes, and `tools/benchmark_epoch_cut_v55.py` to reproduce the metadata-only benchmark.
