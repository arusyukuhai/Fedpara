# Fedpara v12 — Faster durable IO for live P2P

This update targets the pathological slowdown when another peer connects:
`JOURNAL WRITE ≈ 2285 ms`, `LOCAL SEAL ≈ 2606 ms`,
`WORKING SAVE ≈ 2484 ms`, while `LOCK WAIT ≈ 0 ms` and
`WEIGHT APPLY ≈ 0 ms`.

## What changed

* **SQLite journal retains `WAL` and `synchronous=FULL`.** An authenticated,
  signed update is acknowledged, applied, and gossiped **only after its WAL
  transaction durably commits**. The update protocol and signed wire bytes
  are unchanged, so v11 journals remain readable.
* **No synchronous SQLite auto-checkpoints inside large INSERTs.** Set
  `wal_autocheckpoint=0`, and run PASSIVE checkpoints using an independent
  connection when the WAL exceeds 32 MiB. The WAL remains crash-replayable
  between checkpoints. The WAL may grow if readers prevent progress; monitor
  available disk space. The journal hard size cap still applies to signed
  events, not to SQLite temporary/WAL overhead.
* **The expensive database insert/fsync runs in a worker** with its own SQLite
  connection, so ICE/heartbeat/Gossip timers can continue. This does *not*
  remove the durability barrier before the local optimizer moves on to the
  next published block.
* **Only every 10 published updates** do we normally snapshot all local
  LoRA/LoHa factors and AdamW moments. Snapshot CPU capture occurs at an
  optimizer-safe boundary; safetensors serialization and fsync run in a
  separate task. Only one snapshot write is in flight; optional snapshots
  may be skipped while the previous one is still writing.
* **On a clean shutdown**, wait for the background writer and save a final
  complete local optimizer snapshot. A crash can lose the most recent local
  optimizer moments and unsealed training steps. Signed updates are still
  recoverable from the journal and are never silently dropped.
* GUI default launch adds `--working-save-every 10 --durable-every 0`.
  This means the *local* work checkpoint is optimized for speed, and is **not**
  the same durability level as v11. Signed updates remain fully durable.

## Overrides

For maximum local optimizer crash durability at a speed cost:

```
--working-save-every 1 --durable-every 1
```

For a middle ground:

```
--working-save-every 3 --durable-every 0
```

The CLI accepts these flags in `train-live`. GUI defaults prioritize speed;
advanced overrides can be supplied in the underlying trainer invocation.

## Real-machine validation

This is a structural performance change; the author's two-window Mac has **not
been benchmarked** in this sandbox. Compare single-window and two-window
`STEPS / SEC` with `JOURNAL WRITE`, `LOCAL SEAL`, and `WORKING SAVE`.
`WORKING SAVE` now reports **completed background disk writes**, not a current
blocking training operation. Local seal time still includes the full durable
SQLite commit. If that remains seconds long, benchmark the drive itself,
ensure the journal is on a fast local SSD (not a cloud-synchronized folder),
and record WAL filesize/checkpoint progress. A healthy inbound queue alone
is not proof of good distributed convergence.

Tests: `python -m pytest -q tests/test_durable_io_v12.py` and all existing
three regression batches (326 passed, 3 skipped in the current environment).
