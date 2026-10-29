# Kagami v62 — bounded signed factor ledger, epoch-safe export, pinned disk autograd

## Status

An **opt-in experimental** low-rank factor-separated mutable base for `train-elastic --mode elastic`. The immutable safetensors genesis stays read-only. Rather than writing an almost entirely dense LoHA effective delta into every 64-KiB page, v62 retains the independently validated *before/after factor tensors* in a bounded CPU ledger and reconstructs only the requested Linear/Conv1D/Embedding tiles. This is exact factor algebra, not truncated SVD/approximate low-rank compression. Because different algebraic formulations have normal floating-point roundoff, expect tolerance-level rather than bit-identical results.

The ledger is **not authoritative persistent storage**. On a process restart it is reconstructed through the existing **authenticated signed EventJournal**. No new event is trusted merely because it appears in a local scratch directory.

## Invocation

Add to a valid elastic room/model invocation:

```bash
--streaming-base \
--streaming-lazy-write \
--streaming-factor-ledger \
--stream-factor-max-events 32 \
--streaming-scratch-dir /path/to/nvme \
--stream-lazy-limit-gb 100
```

`--streaming-factor-ledger` and `--streaming-page-cow` are mutually exclusive. `--stream-factor-max-events` accepts 1..256, defaults to 32. A smaller number reduces inference time but induces earlier dense compaction; larger values defer large NVMe writes but increase per-tile CPU computation and factor RAM use. The existing `--streaming-base` HF safetensors loader is still required, and all existing authenticated FedSum event validation, source genesis, signature, replay, duplicate, and quota controls remain enabled. This is an opt-in CLI feature, not enabled by default in the GUI.

## What changed

1. `patchlib/factor_ledger_v62.py`: `FactorLedgerSource` virtual tiled weight reads / transposed Conv1D / embedding row lookups; `FactorLedgerManager` delegates fully supported disk-backed 2D matrices and uses legacy safe fallbacks for unsupported tensors. It maintains a **bounded** number of signed event factor-pairs per group (not the whole dense effective delta). The signed apply validates finite candidate effective weights before appending. CPU factors are immutable private copies. A failed commit poisons the manager, requiring clean signed replay.
2. When an event/combined aggregation exceeds the configured per-group factor count or byte quota, the manager **compacts exactly** to a distinct, privately allocated dense v60 mutable backing, then continues ordinary authenticated FedSum. The fallback stays dense thereafter; v62 does **not** solve eternal fixed-size lossless compression. Compaction is bounded in both tensor dimensions, and capacity is checked before attempting a full copy.
3. `patchlib/continual.py`: the existing `_check`, single-event apply, and cross-origin batch apply remain signature-gated; opt-in groups are sent to the virtual ledger. Enormously wide matrix rows are validated in bounded 2D tiles rather than materializing one massive row.
4. `patchlib/layer_offload.py`: pinned per-forward ledger snapshots for disk Linear and Conv1D autograd. A signed update that arrives between the forward and backward of *one such operator* cannot silently change its input-gradient base revision.
5. `patchlib/shards.py`, `shard_runtime.py`, `epoch_frozen_export_v58.py`, `epoch_export_v57.py`: bounded **effective sharded export directly from virtual factors**, no eager dense mutable base needed. The frozen v58 checkpoint captures immutable event references under the fenced trainer cut; later signed updates do not pollute the earlier Merkle cut. The v57 exporter hashes and durably publishes full effective shards and issues its ordinary signed v2 export receipt. An ordinary `model.state_dict()` is rejected while virtual sources are active, preventing stale checkpoint publication. Importing a sharded checkpoint materializes to a private backing first.
6. Existing v59 disk AdamW, v61 sparse-page COW option, and certified v2 committee protocol remain available. The new mode selects a separate ledger; it does not change the quorum's Byzantine/CFT assumptions.

## Performance (synthetic **single update** only)

`BENCHMARK_V62.json` and `benchmarks/benchmark_v62_factor_ledger.py` reproduce this on CPU. Matrix 2,048x2,048 FP32 (16 MiB), LoHA rank 8, one dense effective change:

| Metric | Sparse page COW (v61) | Factor ledger (v62) |
|---|---:|---:|
| Added changed-base payload | 16 MiB pages | 512 KiB factors |
| Update/write stage | 72.81 ms page writes | 1.84 ms factor append |
| One 64-row/full-width tile read | 0.70 ms | 3.09 ms |

At 8 retained events the ledger tile read was ~6.53 ms, and at 32 events ~15.09 ms (single-run timings with cache and thermal variation). Update/write timings are **not a full signed FedSum event comparison**: v61 receives a precomputed dense delta, while v62's validation and base-finiteness checks are excluded from this microbenchmark. Factor-tile time increases with the number of ledger entries. No comparison here proves overall training throughput, 300B runtime, SSD endurance, or WAN behavior.

## Safety, scope, limitations

- CFT committee, signed Journal, and verified Merkle-bound snapshot are assumed. This does **not** make the consensus Byzantine-fault tolerant.
- Checkpoints can export the virtual effective state without pre-compaction. Frozen v2 export has a factor-byte budget; snapshot creation may fail/require retry under insufficient resources, and a simultaneous v60-style compaction that changes the source tensor storage causes the immutable-source check to reject that export rather than sign ambiguous weights.
- The signed validation path still performs O(number of updated model elements) bounded **computations** to reject nonfinite effective weights. We avoid most of the *writes*, not this validation cost.
- The disk kernel only intercepts compatible contiguous 2-D Linear/Conv1D/Embedding weight access. Direct arbitrary `.parametrizations.*.original` reads, fused third-party kernels, arbitrary packed MoE, and unsupported layouts do not become magically virtual-aware; unsupported weight groups use a dense private fallback.
- Per-layer operator forward/backward revision pinning is **not an atomic whole-model transaction** if another thread interleaves updates between different layers. The trainer must continue serializing optimizer steps and signed apply; concurrent model-wide inference requires an additional read epoch / step barrier.
- On a long enough run, exact compaction creates an O(model size) writable dense backing for changed groups; multi-epoch bounded journal GC, indefinite offline peer recovery, and long-run global disk limits require further work. Source genesis plus private mutable backing also require adequate disk free space.
- Still no real 300B end-to-end training, 4,096-device WAN soak, full production self-election, or true zero-pause all-layer hot swap test. This release is not a production guarantee.

Final repository tests: **752 passed, 3 skipped, 0 failed**. See `VALIDATION_V62.md`, `BENCHMARK_V62.json`, and `CHANGES_V62.patch` for independently checkable details.
