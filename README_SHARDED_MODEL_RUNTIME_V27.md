# Fedpara v27 — bounded-memory sharded model state runtime

This revision makes **actual checkpoint export and import** operate on verified
content-addressed shards, without constructing full effective LoHa/LoRA weight
matrices or an intermediate ZIP `.fstate` archive in the default sharded path.

## What's different

- `patchlib.shard_runtime.effective_chunks` calculates bounded row ranges of
  effective adapted weights. For extreme rectangular matrices, it calculates
  flattened ranges using gathered low-rank factors. Active continual anchors,
  frozen remote overlays, and tied aliases preserve the previous semantics.
- `shard-export`, `--shard-state-manifest`, and live checkpoint creation use
  that streaming exporter automatically.
- A receiving live peer imports a verified `.fmanifest` **directly into the
  existing model**, no temporary multi-GB `.fstate` assembly.
- `apply_sharded_state` loads sharded snapshots directly into ordinary models;
  `apply_sharded_effective_state` loads into active parametrized LoHa/LoRA
  models and safely resets stale local factors and anchors.
- CLI model-loading flags:

  ```text
  --sharded-base-manifest PATH/weights.fmanifest
  --sharded-base-store PATH/store
  --max-state-gb 1024
  ```

  Both sharded-base arguments must be present together. They cannot be used
  with `--state-base`. The model architecture and base revision **must**
  correspond to the snapshot. The loaded model identity uses the same
  `fstate-sha256:` semantic hash as an assembled `.fstate` would.

- The data-only manifest/SHAs are validated *before* any Tensor changes. Tied
  parameters with conflicting tensor checksums are rejected. Interrupted
  installation after verification may still leave a partially changed model;
  treat a device/I/O failure as fatal and rebuild/restart the model.

## Size and device limitations (important)

- The live model **must already exist as PyTorch tensors**. Sharded loading
  avoids a *second* full model copy but **does not** independently provide
  expert parallelism, meta-device initialization, memory-mapped CPU offload,
  optimizer paging, or multi-device dispatch. Training a 300B+ dense model on a
  small device is **not** enabled by this revision.
- Working tensor blocks are capped at 131,072 elements (or smaller if requested
  shard size requires it) and live checkpoint shards can be set from 64KiB to
  64MiB. Actual CUDA/MPS intermediate memory also depends on factor rank and
  BLAS implementation.
- Large manifests/states require compatible `--max-state-gb` settings at every
  peer. The manifest size, tensor count and shard count retain finite limits.
- Legacy `.fstate` export/import remain available, but the optimized streaming
  functionality is specific to the **sharded** path.

## Testing

`PYTHONPATH=. python -m pytest -q tests/test_shard_runtime_v27.py tests/test_shards.py`

These verify content digests match legacy effective state, no eager dense
`FactorDelta.delta()` export, large rectangular matrices, tied heads, scalar
and noncontiguous tensors, corrupted shard rejection, and direct model import.
