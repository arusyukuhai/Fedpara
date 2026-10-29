# Kagami v60 — lazy mutable safetensors / Conv1D streaming

## What's implemented

1. **Opt-in elastic training without an up-front writable copy of the entire pretrained base.**
   `--streaming-base --streaming-lazy-write` builds the HF model on `meta`,
   mmaps the **immutable, trusted local safetensors**, installs the existing
   per-layer disk kernels, and creates a separate writable `.raw` file **only
   when a signed continual FedSum commit first modifies that tensor**.
   Factors (LoRA/LoHA) and optimizer operate as before. Unmodified tensors
   continue to read from the immutable source. The update write gate handles
   both single signed updates and batch aggregation; sharded checkpoint import
   materializes all affected adapter groups before writing through their base.
2. **Crash isolation.** Each model/shadow has a separate locked scratch dir.
   Orphaned scratch dirs are removed when the next worker starts. No writable
   base storage is shared across generations. Existing signed journals remain
   authoritative for recovery; temporary mutable pages are NOT used as
   durable checkpoints.
3. **Capacity safety.** `--stream-lazy-limit-gb N` optionally refuses a
   first-write materialization before mutation. On supported POSIX systems,
   `posix_fallocate` reserves real file blocks before mmap, so ENOSPC produces
   an exception instead of a process-killing SIGBUS. A chunked zero-write
   fallback is used on platforms without file preallocation. After a failed
   copy, scratch is deleted and no weight pointers are changed.
4. **GPT-2-style Conv1D kernels.** Transformers' Conv1D stores weights as
   `[input,output]` rather than the `nn.Linear` `[output,input]` layout.
   New tiled CPU/GPU forward, input-gradient backward and LoRA/LoHA factor
   computation avoid constructing a dense effective weight tensor. Compatible
   with the existing frozen-weight cache, including its invalidation after
   signed updates. Fused/packed model-specific MoE kernels are NOT supported.
5. **Large-row safetensors copying.** Writable streaming initialization no
   longer fails when a single tensor row is wider than `--stream-chunk-mib`;
   it splits along a deeper dimension as necessary.

## How to use it

Append these flags to an **existing, otherwise complete and working**
`python -m patchlib.cli train-elastic --mode elastic ...` invocation:

```bash
--streaming-base \
--streaming-lazy-write \
--streaming-scratch-dir /path/to/fast/local/nvme \
--stream-lazy-limit-gb 100 \
--stream-chunk-mib 8
```

The source must be pinned, trusted pretrained **safetensors** and the model
must support `meta` initialization. Set `--stream-lazy-limit-gb` to a value
appropriate for available disk; 0 disables the local quota. The mode is
**experimental** and not enabled in the GUI. It does not support synchronous
training, `train-live` legacy mode, a separate `--state-base` genesis, or a
shared `--disk-linear-dir`. Invalid combinations are explicitly rejected.

## Local benchmark

See `BENCHMARK_V60.json` and `benchmarks/bench_v60_lazy_boot.py`.
Four 1536×1536 FP32 layers (~36 MiB) on a local CPU filesystem:
- v59 eager writable initialization: 37,748,736 bytes of initial mutable scratch, 0.0159 s.
- v60 lazy initialization **including adapter construction**: 0 bytes of initial mutable scratch, 0.003383 s.
- On first mutation of ONE layer: 9,437,184 bytes of scratch, 0.004312 s.

These timings are NOT apples-to-apples and are not a performance guarantee.
They do not measure checkpoint transfer, full-network training throughput,
300B hardware, optimizer I/O, or any WAN/GPU environment.

## Validation

- Scope-specific regression: **285 passed** (45 test modules), includes
  all epoch/streaming/shard/continual/FedSum/P2P-snapshot-specific tests
  selected for this scope; `tests/test_lazy_mutable_v60.py` contains 9 new tests.
- Full 731-test suite **attempted but not completed** within the tool execution
  window; no full-suite success claim is made. See `VALIDATION_V60.md`.

## Limitations / safety boundary

- This is **NOT** full 300B end-to-end training. A single enormous tensor's
  first modification still copies the **whole tensor**. After enough signed
  updates, mutable disk usage can approach the full model size.
- The model remains constructed in PyTorch with file-mapped tensors; model-
  specific forward operators, packed Expert layouts, and activation memory
  are not automatically paged. Some architectures may be rejected safely.
- Optimizer moments continue to use the v59 disk AdamW option where enabled;
  the v60 change does not page a single giant adapter factor more finely.
- The v58 frozen-step checkpoint method is not a mathematical zero-pause
  snapshot, and v2 leader failover, Byzantine fault tolerance, 4096 real
  connected devices and 300B hardware are **not** proven by this release.
- A process crash between an event's signed-journal append and its model
  application still requires the existing fail-closed journal replay; a local
  scratch file alone is never an authoritative resume point.
- Model files must not be modified by external writers or replaced after
  genesis validation. In-place mutation of the base outside the journaled
  continual engine is unsupported and can breach the signed-replay invariant.
