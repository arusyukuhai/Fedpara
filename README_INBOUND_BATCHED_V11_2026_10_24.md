# Fedpara v11 — batched inbound commit and latency breakdown

The v10 GUI reported an example of 0.47 local steps/s, 265.6 ms VERIFY, and 3165.7 ms COMMIT when connected to another peer. These are **one user's sample**, not lab measurements. `COMMIT` combines lock wait, journal append, and application and does not isolate disk latency.

## Changes

1. Continual FedSum `_apply_checked()` checks all reconstructed candidate base tensors **in bounded 8 MiB batches**, rather than synchronizing an accelerator to the CPU after every individual layer/row block. Finite checks run before copying the corresponding bounded batch into the base model. This aims to reduce expensive GPU/MPS synchronization, without relaxing the finite-weight check.
2. Checked cached dense effective deltas up to 32 MiB are transferred to the model's device just once per group, rather than once per row slice.
3. Incoming `COMMIT` time is split into `LOCK WAIT`, `JOURNAL WRITE`, `WEIGHT APPLY`. The GUI also records `LOCAL SEAL` (publish own LoHa/LoRA update) and `WORKING SAVE` (local adapter/AdamW snapshot) times. `COMMIT` is still total time, including lock wait. The latter two times may include wait on the inbound commit lock.
4. Changes are local execution optimizations. Signed update wire bytes, protocol versions, LoHa/LoRA layout, ReLoRA period, genesis, journal format and FedSum semantics are unchanged. Existing v10 rooms are compatible once trainers are fully stopped and journal files are backed up.

## Reading timings

- `LOCK WAIT` high, `WEIGHT APPLY` low: asynchronous training and incoming commit contend for the shared update lock.
- `JOURNAL WRITE` high: durable SQLite WAL append/storage pressure. Do not turn off durability without explicitly accepting data-loss risk.
- `WEIGHT APPLY` high: accelerator/CPU reconstruction, weight copies and finite checks remain costly.
- `LOCAL SEAL` high: your own update publish event serializes factors and applies them, even without remote events.
- `WORKING SAVE` high: copying the local factor/AdamW state and fsyncing local snapshots may dominate a short `local_steps=4` interval.

The GUI reports the **latest** latency, not moving averages, and application time includes worker scheduling. On Apple MPS/other accelerators, actual performance changes must be measured on the user's hardware. Two trainers on the same computer still compete for CPU/GPU/SSD bandwidth. A TCP or QUIC swap alone cannot fix such contention.

## Verification

- `python -m pytest -q tests/test_batched_apply_v11.py` tests batched scalar synchronization, rejection of nonfinite candidate weights before any write of that batch, and nonnegative per-stage timings.
- Three-way suite run: 322 tests passed, 3 skipped (24 October 2026) in this CPU tool environment; no measured end-to-end two-window speedup yet.
- `node --check web/desktop_webview.js` verifies UI syntax.

Start: `python -m patchlib.cli gui`.
