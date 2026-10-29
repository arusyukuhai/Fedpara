# Kagami v47 — verified, destructive **local** GC hardening

**Status: partial. Do not describe as completely autonomous live rollover.**

## What works

For a *previous* generation that has **actually been installed and acknowledged by the trainer**, the v43/v46 generation control loop already polls GC every 30s on a background thread. v47 makes the path safer and permits late collection after additional certified generations are committed:

- `GenerationLedger.certified_epoch(n)` reads and re-verifies committed historical quorum certificates (not unsigned votes or just the latest tip).
- A generation whose runtime state is `installed` can be retired even after the global certificate head advances beyond it.
- Right before deleting the old journal, the controller validates the new checkpoint manifest SHA-256, its semantic SHA, **every referenced tensor shard**, the replacement journal's certified epoch/manifest/proposal binding and SQLite integrity.
- It rejects missing/corrupt shards, a journal reusing the old genesis, missing journal index sequences and unsigned suffixes.
- A crash after storing `collecting` but before the durable tombstone will **repeat the complete cut check** rather than treating the database status as irreversible authorization to delete.
- An exclusive OS trainer lock and an fsynced retired marker precede unlinking `*.fplog`, the old journal SQLite DB and WAL files. Restarting a deleted old journal raises an error, not an empty replacement.
- `python scripts/report_gc_v47.py <journal>.generation-state` provides a read-only view of the current epoch state and obvious blockers. `can_attempt_auto_gc` is **only a preliminary diagnostic**, not an authorization to erase files.

## What is still NOT implemented

The original **live v1 training loop** in `patchlib/elasticnet.py` still uses one genesis, immutable signed event history, and a single attached adapter + optimizer. `TrainerStepRouter` and compact-v2 certificates are not connected to that live worker. As a consequence:

1. A certificate cannot automatically install a new model/optimizer/journal on a running worker.
2. Old signed events arriving after the checkpoint cut cannot be safely reassigned by the v1 anti-entropy channel; v47 declines GC with any such suffix.
3. Remote nodes offline at the cut may have **unseen updates**. Current wire protocol cannot conclusively determine which were applied or durably retained by successors.
4. The compact-v2 ledger does not yet drive GC on the legacy v1 worker. Destructively deleting its current journal would threaten reconnection correctness.

The new GC code therefore **only destroys old event logs after a previously completed trusted migration**; it never erases the active generation. There is no proof yet that thousands of peers can live-roll and recover from a new checkpoint with all writes accounted for. Avoid enabling indiscriminate deletion on current journals. No change migrates or deletes users' preexisting files on installation.

## Verification

```
PYTHONPATH=. python -m pytest -q tests/test_destructive_gc_v47.py tests/test_epoch_runtime_v43.py tests/test_online_generation_v41.py
```

Tests cover post-install shard corruption/disappearance, stale historical certificate tips, a crash leaving `collecting` without a tombstone, a replacement journal impersonating the old genesis, corruption of new SQLite, lock exclusion and crash-resume idempotence.
