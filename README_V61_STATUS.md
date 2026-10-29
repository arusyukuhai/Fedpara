# Kagami v61 — experimental per-tensor sparse page COW + on-demand frozen MoE experts

## Implemented

1. Opt-in `--streaming-base --streaming-lazy-write --streaming-page-cow` for elastic continual signed FedSum. The immutable safetensors origin stays untouched. Eligible 2-D offloaded Linear/Conv1D/Embedding groups retain their CPU base and add a *private sparse page backing*. Page size defaults to 64 KiB and supports 4, 16, 64, 256 and 1024 KiB. The source for tiled CPU/GPU Linear/Conv1D kernels is an overlay view; incoming signed events apply to that view, not the original base. Disk quotas are checked before page allocation, sparse page stores are fail-closed after I/O failures, and a clean launch replays the existing authoritative signed Journal. No byte-level patch or synthetic signature is trusted.
2. Groups not safely intercepted by existing disk kernels continue to use the v60 whole-tensor first-write fallback. Legacy full-state exports must first **explicitly materialize all paged groups** into private whole-tensor files. Both direct model `state_dict()` and `export_state_sharded` refuse while sparse pages are active, rather than certifying stale original tensors. Importing a whole-state shard into a virtual page base is blocked until materialized. This materialization has O(model size) I/O cost and may require additional disk space. v2 autonomous trainer/export is **explicitly incompatible** with this opt-in mode until a proof-safe virtual-base exporter is integrated.
3. Optional `--streaming-expert-path SOME.EXPERTS` wraps a model's unpacked `ModuleList` or `ModuleDict` experts with a `StreamedFrozenExpert`. It reuses existing CPU tensor mappings (no extra per-expert raw copy), materializes only the *selected* expert's tensors on the compute device, and recomputes its forward in backward to propagate input/router gradients. Expert own weights and their trainable LoRA/LoHA factors are **not trained** in this path. Training-mode BatchNorm/dropout are blocked due to nondeterministic/stateful recomputation. Arbitrary packed/fused MoE is not supported. A model adapting expert parameters is rejected: users must keep those specific experts frozen, while training the router and eligible non-expert layers.
4. Signed local-event seals, dedup/replay, stale snapshot refusal, disk error poisoning, Conv1D gradient flow, expert input/router gradients and a genuine CLI streaming loader integration path have new tests.

## Invocation

To enable page-level COW, append to an existing valid `train-elastic --mode elastic` launch:

```bash
--streaming-base \
--streaming-lazy-write \
--streaming-page-cow \
--stream-page-kib 64 \
--streaming-scratch-dir /path/to/nvme \
--stream-lazy-limit-gb 100
```

For model-specific unpacked, **entirely frozen** experts, additionally specify `--streaming-expert-path transformer.layers.0.mlp.experts` with the *actual* named ModuleList/ModuleDict path. Not every MoE architecture has this layout; incompatible ones fail closed. This is experimental and not in the GUI.

## Performance and scope

See `BENCHMARK_V61.json`. Synthetic CPU data: 2048 x 2048 FP32 (16 MiB) and 64 KiB pages. A one-row sparse edit touches one 64 KiB page, allocating 64 KiB physical file blocks locally, versus 16 MiB for a full dense rewrite. If **all** pages are changed, physical data rises to ~16 MiB; dense LoHA/LoRA effective updates usually change most or all pages, so this is **not a general FedSum storage compression solution**. Reads through a virtual page view and per-page writes also add CPU/I/O overhead. No performance claim about 300B, SSD endurance, high-churn WAN, or end-to-end training speed is made.

## Validation

See `VALIDATION_V61.md`. Related selection based on v60's 45-module suite plus new v61 tests: **294 passed**. Full repository suite not claimed as passing; this environment has historical long-running integration tests that are not included in the focused result.

## Known limitations / not yet production-ready

- Not a true zero-pause 300B full training engine, nor GPU/NVMe benchmark of a 300B model. No real 4096-device WAN soak test.
- Sparse page view requires the installed disk kernel interception: arbitrary reads of `parametrizations.weight.original`, custom fused kernels, state_dict without explicit materialization, and generic model-specific state exporters do **not** see current virtual effective weights. The CLI v2 generation proposer is refused in this configuration; other model-specific exports must be reviewed before enabling.
- The sparse page set and scratch storage are intentionally ephemeral. A crash requires a clean scratch and complete authenticated signed journal replay; this design does not provide a bounded-replay compact checkpoint or universal long-term GC.
- `model.train()` with a stochastic frozen Expert (e.g., dropout) is refused; arbitrary custom nondeterministic modules are outside the supported pure-module contract. Expert weight and optimizer training are not paged by this proxy.
- Page writes are not a globally atomic transaction across every layer. An I/O exception poisons a page store and requires a clean replay. In-flight forward/commit consistency still depends on the existing training step serialization; page-tile locks alone do not give an entire forward a frozen version.
- Signed event deltas can be dense; all affected pages may allocate, so disk budget must be sized for worst-case full mutable base and snapshot materialization.
