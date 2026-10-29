# Fedpara v30 — memory-adaptive out-of-core model training (experimental)

This release introduces an **opt-in** execution path for models whose frozen
weight files exceed the available RAM/VRAM. v29's unlimited publication,
append-only journal, save-on-graceful-stop, and GUI checkpoint folders remain
unchanged. This is an engineering advance, **not** a claim that every Hugging
Face architecture, 300B+ model, or arbitrary custom MoE runs on every phone.

## New components

1. **Meta-device HF initialization + local safetensors mmap:**
   `patchlib.streaming_base` creates a supported HF architecture without its
   base weight allocations and binds parameter/buffer storage to verified-shape
   local sharded `.safetensors` tensors. No `torch.load`/pickle, full Python
   `state_dict` or full-model `.to(device)` is used in the streaming branch.
   A pretrained HF checkpoint can be downloaded into the HF cache first.
   A live room hashes **all safetensors files** into its genesis, so two peers
   with different initial bytes cannot silently merge their signed events.
   The immutable weights in the HF cache are never modified.
2. **Live, writable disk-backed base:** continual signed FedSum updates are
   written through private, temporary file mappings, in bounded tensor blocks.
   At each process start the original immutable safetensors is restored into
   fresh writable pages and all signed journal entries are replayed. This
   avoids double-applying old events after restart. A normal exit still writes
   a separate content-addressed sharded model checkpoint.
3. **Memory-adaptive accelerator placement:** supported offloaded `nn.Linear`
   and `nn.Embedding` keep frozen bases in CPU/disk mappings; other parameters
   (including trainable LoRA/LoHa factors) and small buffers are moved to
   GPU/MPS. If an unsupported parameter needs more accelerator memory than
   available, the process **fails rather than silently loading a giant tensor**.
   Frozen Linear projection forward and input-gradient backward use bounded
   **two-dimensional tiling** when an entire layer does not fit VRAM.
   Frozen Embedding reads only selected rows and applies LoRA/LoHa to the
   selected token IDs. Tied input/output embedding & Linear weights are handled
   when both module types have supported kernels.
4. **Use available memory:** GPU frozen-weight LRU budget is derived from
   *currently free* VRAM and user-selected `--disk-cache-fraction` (0..0.85,
   default 0.60), optionally capped with `--disk-cache-mib`. Hot whole-layer
   weights remain on GPU when they fit, rather than reloading every pass.
   CPU uses the OS page cache (which grows naturally into otherwise free RAM),
   and grows GEMM tile size with available physical/cgroup-adjusted memory.
   Signed model updates invalidate accelerator copies before they become stale.
5. **GUI:** new **巨大モデルの低メモリ実行** selector and GPU-cache fraction input.
   Normal v29 training remains the default. Streaming mode requires **a new
   room, pretrained initialization, and an empty initial `.fstate` field**;
   v29 rooms with `.fstate` genesis must not be converted in-place.

## Desktop usage

From `fedpara/`, start `python -m patchlib.cli gui`. Create a **new** room,
choose **pretrained** for initialization, use a compatible HF safetensors
model and leave the initial-state path blank. Select experimental streaming
mode, optionally raise the cache fraction toward `0.85` on a high-VRAM machine,
and start training. On graceful stop, wait for the **CHECKPOINT SAVE COMPLETE**
log message and verify the sharded manifest. Previous room logs/weights remain
in their original folders.

## CLI example (local CPU)

```bash
python -m patchlib.cli train \
  --model /path/to/local/hf-safetensors-checkpoint \
  --hf-task causal-lm --target-scope auto \
  --streaming-base --device cpu --kind loha --rank 16 \
  --text-file train.txt --steps 100 \
  --disk-optimizer-dir ./optimizer-pages \
  --disk-cache-fraction .75 \
  --shard-state-manifest ./trained.fmanifest --shard-store ./trained-shards
```

For GPU/MPS, use `--device auto` or the explicit device. `--streaming-base`
uses a meta-compatible HF model factory and a local or downloaded safetensors
snapshot, not arbitrary custom factory code. Live writable copies require a
fast SSD with approximately **one extra base-model-size of free capacity**, in
addition to the source weights and saved checkpoints. Specify
`--streaming-scratch-dir /path/to/fast/ssd` for control. Interrupted processes
can leave unused `fedpara-weights-*` directories behind; delete those only
when no trainer is using them.

Additional controls: `--stream-chunk-mib` (1..128; default 8),
`--disk-tile-mib` (1..256; default 16), `--disk-cache-mib` (0 for automatic),
`--disk-linear-min-mib` (streaming default 1 MiB threshold).

## Important limits and next work

- **Not arbitrary architecture support:** requires meta-safe HF construction,
  resolvable local safetensors tensor names, compatible persistent buffers and
  supported streaming execution kernels. Packed/fused MoE weights, many custom
  attention kernels, unusual shared parameters, `EmbeddingBag`, fused QKV
  operations, nonpersistent rotary buffers requiring bespoke initialization,
  and some custom heads need architecture-specific adapters. Unsupported
  situations should fail visibly, not claim a correct training result.
- **Training only adapters:** requires all trainable factor parameters,
  activation tensors and checkpointing workspace to fit on the compute device.
  v28's disk AdamW is available to local `train`, but **not** integrated with
  `train-live`'s crash journal. Whole-model base finetuning is not supported.
- **P2P is not limitless:** signed wire updates still use bounded individual
  messages (`MAX_PATCH`, `--max-patch-mb`), and the v29 durable journal and room
  membership budgets still exist. Very large LoHa layouts require a future
  *chunkwise signed multi-message transaction*; raising caps is not a fix.
- **Durability:** temporary mutable weight pages are NOT authoritative crash
  checkpoints. Only immutable model genesis plus the verified signed event
  journal provide restart recovery. Deleting the journal breaks continuity.
  Snapshot export also requires sufficient disk space and time.
- **Performance:** available RAM/VRAM changes during a run, and an unforeseen
  workload can still OOM. Repeated NVMe streaming or tiny output/input tiles
  can be extremely slow. Always benchmark a smaller model before scaling.
- **Testing:** tests in `tests/test_streaming_base_v30.py` cover meta mapping,
  shard index validation, shape/dtype failures, tied embedding and Linear
  gradients, two-dimensional tiling, scalar state, live genesis fingerprint,
  GUI command generation and recovery guards. CUDA/MPS kernels and a genuine
  300B+ training run **were not tested in this build environment**.

Run regression suites:

```bash
PYTHONPATH=. pytest -q tests/test_streaming_base_v30.py \
  tests/test_offload_v28.py tests/test_continual_elastic.py \
  tests/test_shard_runtime_v27.py tests/test_unlimited_durable_gui_v29.py
```
