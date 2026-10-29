# Validation — 2026-10-27

Result: **85 passed in 53.61s** on CPU (Python 3.12, torch 2.14.1+cpu).
Compilation of the four changed/added runtime modules also passed.

Reproduce from the archive's project root:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_epoch_checkpoint_v50.py \
  tests/test_epoch_objects_v50.py \
  tests/test_live_epoch_switch_v48.py \
  tests/test_hot_swap_v45.py \
  tests/test_epoch_carry_v46.py \
  tests/test_destructive_gc_v47.py \
  tests/test_epoch_runtime_v43.py \
  tests/test_compact_epoch_network_v46.py \
  tests/test_live_generation_transport_v42.py \
  tests/test_generation_multicast_v46.py \
  tests/test_epoch_ingest_v46.py \
  tests/test_snapshot_boundary_paging_v26.py \
  tests/test_v49_overlay_scale.py \
  tests/test_v49_gossip_cost.py
```

Install the existing project dependencies and `pytest-asyncio` for async tests.

## New evidence

- Actual subprocess `os._exit(13)` after partial append; reopen resumes the
  durable offset and reconstructs exactly the expected bytes.
- Final-block-before-publication, rename-before-index-commit, uncommitted file
  tail and corrupt cached object recovery.
- Quota exhaustion leaves pinned active bytes intact; content deduplication,
  object-count limits and exclusive cache writer checks.
- Download cancellation, neighboring-source failover, bad-source hash rejection,
  bounded retry and shared token-bucket concurrency.
- 300 sequential cache generations with explicit lifecycle retirement: unique
  payload storage stays at 600 bytes in that synthetic fixture and SQLite stays
  under 128 KiB. This proves cache behavior, NOT bounded full-system disk usage.
- Real checkpoint export, ranged download, full hash/semantic verification,
  PyTorch model loading and SGD steps. Four old-model optimizer steps execute
  while checkpoint transfer is blocked; new-model steps execute after handoff.
  Ledger, journal attestation and transport handoff are test doubles here.
- Reopening a completed checkpoint cache requires no redownload.
- Simulated pointer directory-fsync failure fails closed; same-process retry
  does not repeat flush or carry. Failed transport publication can retry safely
  with an idempotent callback.
- Lagging service selects the next intermediate certificate. A restart with a
  mismatched durable generation pointer is refused.

## Limits

The 4096-participant checks cover the existing overlay's connectivity, degree
and graph distances. They do NOT run 4096 training processes, WebRTC links or
simultaneous generation migrations. No full GUI end-to-end migration, GPU/VRAM
stress, real power-cut test or multi-day network soak has been performed.
The existing production training/transport integration is still incomplete;
see README_V50_STATUS.md before use.
