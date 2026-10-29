# Kagami v62 validation — 2026-10-27

## Automated tests

- Whole repository (final code): `PYTHONPATH=. pytest -q tests` -> **752 passed, 3 skipped, 0 failed**, **79.97 seconds**. See full-run summary as recorded in this report.
- Focused v61 regression selection plus v62 dedicated tests, before the last wide-matrix fix: **306 passed**, 0 failed.
- v62 dedicated tests after the wide-matrix fix: **13 passed**, 0 failed.
- Frozen export / committee-v2 interaction regression: **25 passed**, 0 failed (subset of existing/focused suites; not additive to above counts).

The checks exercise:

- Signed local updates with both LoRA and LoHA; virtual disk kernels vs ordinary dense baseline outputs and input gradients.
- Cross-origin aggregated, authenticated updates with numerical reference comparison.
- Signed EventJournal replay and duplicate refusal.
- Exact one-time dense compaction after factor count cap, subsequent updates, and quota enforcement.
- Sharded effective checkpoint serialization directly from virtual weight sources; no stale model.state_dict allowed; reimport matches committed model output.
- Frozen factor snapshot followed by new trainer steps, signed v2 export receipt and manifest hash, reproduction of the exact cut checkpoint, local exporter-receipt verification.
- Pinned per-op Linear and GPT-2 Conv1D autograd base version across a remote append between forward/backward.
- A very wide matrix row validated in bounded columns, avoiding unbounded row materialization at `_check`.
- Legacy v61 page-COW tests, older v59/v60 disk optimizer/loader and v2 generation/recovery suites through repository regression test run.

## Synthetic benchmark

Run `PYTHONPATH=. python benchmarks/benchmark_v62_factor_ledger.py`.

Machine-local single-run example from `BENCHMARK_V62.json`:

- Tensor 2048x2048 FP32 = 16 MiB, 64-KiB COW pages, LoHA rank 8.
- One dense effective update changed all pages: page-COW payload **16,777,216 bytes**, low-rank factor ledger payload **524,288 bytes** (1/32 as much).
- *Only the update storage/write substep:* append low-rank factors ~**1.84 ms**; write fully calculated dense values into COW pages ~**72.81 ms**.
- *Only a 64-row/full-width read:* COW ~**0.70 ms**, ledger with one update ~**3.09 ms**, 8 updates ~**6.53 ms**, 32 updates ~**15.09 ms** (one run, non-controlled thermal/cache variance).
- Maximum absolute element difference between the two tile results: **0.0** for this synthetic input.

This benchmark deliberately excludes signature verification, factor payload decoding, full candidate-finiteness validation, optimizer time, network time, and snapshot fsync. It is **not** a complete event-throughput speedup measurement. The factor ledger's per-tile cost grows with retained events; dense compaction is the bounded fallback.

## Not validated

- 300B actual model in multi-device forward/backward or true no-pause hot swap.
- 4,096 real devices across a WAN over days/weeks with churn and consensus failover.
- Genuine end-to-end bandwidth, NVMe endurance, memory peaks under all architectures, or perfect global disk-space upper bounds across offline histories and checkpoints.
- Concurrent whole-network model forward atomicity beyond the tested per-operator revision pinning.
- Byzantine-safe committee behavior: quorum protocol remains crash-fault oriented.

## Key safety invariants

Only validated signed before/after factor event structures reach the ledger through `ContinualElasticEngine`. Failure after journal persistence poisons the in-memory manager; recovery requires clean reconstruction from the signed journal. Scratch is private to a running model/epoch; no unauthenticated low-rank delta cache is trusted on restart. Captured v2 exports preserve the frozen factors and immutable base and must verify the frozen-prefix journal cut. Direct arbitrary `model.state_dict()` is rejected while the original model tensors are virtualized.
