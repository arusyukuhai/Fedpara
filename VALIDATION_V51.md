# Kagami v51 validation — 2026-10-27

Run from the project root with pytest and the existing dependencies:

```sh
PYTHONPATH=. python -m pytest -q \
 tests/test_epoch_v51_lifecycle.py \
 tests/test_epoch_checkpoint_v50.py tests/test_epoch_objects_v50.py \
 tests/test_live_epoch_switch_v48.py tests/test_hot_swap_v45.py \
 tests/test_epoch_carry_v46.py tests/test_destructive_gc_v47.py \
 tests/test_epoch_runtime_v43.py tests/test_compact_epoch_network_v46.py \
 tests/test_live_generation_transport_v42.py tests/test_generation_multicast_v46.py \
 tests/test_epoch_ingest_v46.py tests/test_snapshot_boundary_paging_v26.py \
 tests/test_v49_overlay_scale.py tests/test_v49_gossip_cost.py
```

**Results**: 93 tests passed on this Linux CPU runner (Python 3.13). Includes:

- 500 sequential abandoned candidates reclaimed with at most one lease row
  retained and no orphan candidate views, across independently verified
  service-manager lifecycle tests (synthetic stage stubs).
- Pointer-rename-before-lease-commit recovery; active checkpoint pin protection
  across restarts; fail-closed retention budget; forged pointer mismatch.
- Authenticated mixed-generation signed Hello with explicitly matched root;
  only generation certificates/heartbeats accepted; v1 frame rejected;
  wrong immutable root rejected.
- A real signed second-generation certificate, actual PyTorch sharded checkpoint,
  SQLite journal installation, `EpochRuntimeV43.collect`, and separate worker
  thread invoking proven retirement of the old checkpoint view and its journal.
- Existing v50 actual PyTorch training steps while a checkpoint download is
  blocked, then independent optimizer activation, and crash/resume tests.
- Existing negative tests for corrupt/cut-mismatched shards, replays,
  incomplete carried updates, held journal lock, bad signer, and restart.

**4096-node graph benchmark:**

```sh
python scripts/benchmark_v49_overlay.py --peers 4096
```

Observed (graph simulation): max degree 8, 16,375 undirected connections,
mean shortest-path hops 4.346 from one source, max 6 hops; a complete mesh
would need 8,386,560 connections.

**Not measured:** 4096 live trainers, real multi-generation WebRTC throughput,
mobile storage consumption, end-to-end optimizer hot installation in the GUI,
real committee-proposed snapshot creation, hardware failures, multi-day uptime.
