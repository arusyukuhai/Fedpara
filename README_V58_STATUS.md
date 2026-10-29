# Kagami v58 — frozen-factor background checkpoint export

2026-10-27. **Research-grade integration; NOT a certified zero-pause 300B training engine.**

## What changed in the actual `train-live` path

1. `patchlib/epoch_frozen_export_v58.py` captures a bounded **independent CPU copy** of local LoRA/LoHA factors, their committed anchors, and currently installed remote overlays at a sealed optimizer-owner step boundary. It **shares** the presumed immutable original/base tensors; it does **not** duplicate the full model. The capture refuses to exceed **256 MiB** of additional factor memory (or lower configured state-size limit). Changes to PyTorch base tensor versions/storage/topology between capture and completion block signing.
2. `patchlib/elasticnet.py` pins a SQLite WAL **read snapshot** at the same fenced boundary as the factor capture. The O(history) index scan is then completed **outside** the writer/ingress fence. Post-capture new signed events remain in the original journal for certified suffix/carry; they must not become part of the snapshot's cut. The SQLite read transaction is released immediately after the index scan, **before** full shard export, to prevent long WAL retention.
3. The full effective-weight streaming, content-addressed shard storage, SHA verification, fsync, certified-cut validation, and signed v2 receipt now run in a **single bounded outstanding background task** while the trainer continues local optimizer steps and receives signed P2P updates. A candidate becomes visible to the committee only after the task succeeds. In-flight exports are drained before closing DB/key resources on clean shutdown, and an obsolete epoch/parent candidate is discarded.
4. `patchlib/epoch_export_v57.py` supports the frozen certified-prefix input without mislabeling its manifest with the *later* journal count, retains its original strict synchronous path for old callers, and refuses unfinished frozen captures.
5. The actual periodic `train-live` **local factor + AdamW moments** snapshots use the existing `AsyncWorkingSaver` rather than blocking on safetensors serialization and file fsync. A single pending snapshot writer is allowed; optional intermediate snapshots may be skipped under disk congestion, but signed journal events remain durable. The working-state writer fsyncs the enclosing directory after atomic publication. Its existing restart validation (genesis/tip/layout/parameter shapes) stays in force. This is **not yet a sharded/offloaded optimizer**; the moments remain ordinary in-memory PyTorch optimizer tensors while training.

## Experimental configuration

Use the same v57 pinned committee and opt-in flags, including `--generation-v1-committee FILE --generation-v2-trainer` and, for the pinned designated proposer, `--generation-v2-propose-every N`.

**No v1-to-v2 automatic pointer conversion or Byzantine committee tolerance** was added. The existing CFT trust assumptions are unchanged.

## Important limitations

- **Short local pauses still exist** for bounded low-rank factor capture, local unsigned gradient flush / signed event seal, and atomic epoch activation. For a dense-fallback adapter with large trainable tensors, capture may exceed the factor-memory budget and fail closed. This is *not* hard real-time or strict zero-pause.
- The model's **base storage must stay immutable during the export**. Normal PyTorch in-place changes are detected via tensor version/storage checks, but malicious `.data`/raw-memory writes can bypass version counters. Other mutators must honor the trusted trainer lock and immutable-base contract; arbitrary mutable layers (e.g. running-stat buffers) may cause the export to abort.
- A pinned WAL read snapshot can delay a WAL checkpoint during **frontier scanning**. The read transaction is short relative to full model export but scales with history length. Long offline histories can still require significant source journal scanning and disk.
- The live trainer's epoch handoff still instantiates a full PyTorch shadow model and resets new-generation AdamW state; actual layer-by-layer 300B training, moments offload, 300B model+optimizer no-double-load activation, live proposer failover, and 4,096 physical devices over WAN are **not implemented or verified**.
- Large checkpoints still incur a full effective-model write, full object integrity verification and fsync. There is no strict long-term disk bound while unlimited offline recovery is promised. No automatic old-object destructive GC is added.
- Background export during a clean shutdown is awaited so the event journal is not closed under its reader; this may delay exit on a huge model. Abrupt crashes simply leave unpublished/orphaned objects (not signed/activated).
- Signed receipts attest that a pinned **trusted** exporter carried out the local procedure, not a Byzantine-valid proof of independent deterministic model computation.

## Verification

- The relevant epoch/shard/optimizer/P2P-focused test selection: **103 passed** (see `VALIDATION_V58.md`).
- The two attempted full-suite runs did **not finish inside the execution budget**; there is no v58 claim of full-suite pass.
- Synthetic 4,096 origins / 20,480 indexed updates: original scan-in-fence capture **111.773 ms**; v58 low-rank factor fence **0.262 ms**; scan after release **111.152 ms**, on this runner with a small PyTorch linear model. This measurement does not include shard export, optimizer switching or WAN throughput.
- Reproduce with `python -m tools.benchmark_epoch_v58`.
