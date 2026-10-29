# Fedpara v10 — low-stall verified inbound path (24 Oct 2026)

This release is a source-level optimization based on the v9 local-telemetry GUI release.
It **does not** alter signed wire bytes, model layout, FedSum semantics, LoHa/LoRA ranks, periodic ReLoRA, the immutable genesis, or journal format. Existing v9 rooms and journals are compatible; back up files and stop trainers before installing.

## What changed

1. `patchlib.continual.ContinualElasticEngine._check` validates signed before/after factors and reconstructs each effective float32 delta on the **CPU without `adapter.lock`**. Structural group metadata is immutable for the lifetime of an adapter. Previously it reconstructed them while holding the same lock that protects forward/backward/AdamW, so a busy remote peer repeatedly blocked local training.
2. For ordinary small-to-medium weights, the verified dense effective delta is handed directly to `_apply_checked`: **the expensive CP/LoHa multiplication is no longer repeated during the locked commit**. A bounded 64 MiB per-event preparation budget protects memory on large models; oversized groups fall back to chunked locked reconstruction, after full preflight, without changing correctness.
3. `AsyncInboundPipeline` caches up to 2,048 **exact, already authenticated-and-committed event byte digests**. Repeat gossip delivery bypasses redundant safetensors/low-rank reconstruction and is still acknowledged as a duplicate. New bytes are always fully verified, and signed equivocation IDs are still rejected. The cache is process-local only and never replaces the SQLite journal.
4. The local-only GUI monitor now shows **last signature/tensor verification time, last journal/apply time, and the cumulative number of recognized duplicate events**. Measurements help distinguish CPU/disk stalls from networking backpressure.
5. Added regressions for verification while a trainer thread owns the model lock, no second low-rank reconstruction during cached commit, and exact duplicate deduplication without re-verification.

## Troubleshooting two-window slowdown

- Compare the `STEPS / SEC` graph with both GUI processes running *before* they connect and then after the connection is authenticated. Two trainers on the same computer can compete for CPU/GPU memory even with no network traffic.
- Watch `VERIFY (LATEST) ms`, `COMMIT (LATEST) ms`, `INBOUND QUEUE`, `GOSSIP DUPLICATES` and `EMITTED / APPLIED`. Large `VERIFY` means CPU reconstruction/validation is bottlenecking; large `COMMIT` means journal/disk or weight copy/lock still costs time. A growing inbound queue means production exceeds ingestion capacity.
- On **a new peer's first join**, a complete verified model snapshot can still pause local training while the snapshot is captured/exported. This release does not remove that one-time synchronization cost. Once the peer has the correct snapshot, signed event gossip should proceed with less blocking.
- Receiving updates does not ensure better held-out accuracy. FedSum's asynchronous stale updates and each peer's data distribution may affect convergence. Compare evaluation loss on one fixed validation set to substantiate benefits.

## Tests

Run `python -m pytest -q tests/test_fast_inbound_v10.py`. Whole suite was split into three batches due to a monolithic-run shutdown stall; the three runs collectively reported 318 passed, 3 skipped, 0 failures. Tests exercise mocks/CPU; there is **no completed measured WebRTC two-window speedup benchmark** in this deliverable.

No new dependencies. Launch as before: `python -m patchlib.cli gui`.
