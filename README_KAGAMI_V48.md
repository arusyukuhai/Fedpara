# Kagami v48 — certified in-process trainer generation activation

## Status (important)

**This is NOT a complete unattended multi-peer online generation rollover.**
The existing `patchlib.elasticnet.train_elastic` worker still pins its WebRTC
`ElasticNode`, incoming event engine, journal, and bootstrap snapshot handling
to the initial genesis; its automatic proposal/vote callbacks are intentionally
not enabled. Therefore it cannot yet migrate an entire live network and must
not automatically delete an active worker's journal.

### Implemented in this revision

- `patchlib/live_epoch_switch_v48.py`: `LiveEpochSwitchService`, an async
  generation watcher and activator. It uses signed, durable `GenerationLedger`
  certificates (rather than untrusted neighbor advertisements); requires the
  next consecutive epoch; derives a deterministic fresh genesis from the
  *whole* committed proposal; prepares a separate model/adapter/optimizer and
  journal; validates exact checkpoint manifest SHA-256, semantic content hash,
  and every referenced shard before activation; requests trusted local
  flush, durable carry, verified new journal and transport handoff callbacks.
- The v45 `TrainerStepRouter` now correctly clears its step admission barrier
  when its switch task is cancelled while waiting for an in-flight optimizer
  step. The previous implementation could deadlock every future step.
- A successful handoff publishes a fsynced, atomic active-generation pointer
  and replaces the live model/optimizer at a step boundary. Generation-aware
  network activation runs only after the durable pointer is written; if it
  fails, the step gate refuses all further training until recovery. A separate relay
  coroutine can remain alive while a trainer is being switched.
- No operation remotely requests file deletion. Existing v47 GC remains
  restricted to verified retired generations with a non-owning old journal.

### What an integrator MUST supply

`build_shadow`: a genuinely separate loaded model and optimizer constructed
from the certified checkpoint, not a wrapper over the old optimizer.

`flush_unpublished`: finish/record all partial local updates without losing
their exact origin and sequence IDs.

`carry_suffix`: cryptographically validate + durably archive/replay *all*
post-cut signed effective-weight deltas, ensuring no duplicate application.

`attest_journal`: verify the new journal, optimizer and checkpoint bytes are
durable and consistent with the certified proposal.

`transport_handoff`: atomically activate an ALREADY PREPARED dual-epoch WebRTC
route after the pointer is durable, with monotonic epoch tags and stale-update
quarantine; it must be crash-recoverable/idempotent and return `True` on success. The
legacy v1 `ElasticNode` is **NOT** such an implementation. Never pass a dummy
callback in production.

`verify_local`: confirm the staged *loaded* model and new optimizer agree with
the certified checkpoint, in addition to checking all referenced file bytes.

All hooks must be trusted local code, never downloaded network Python.

### Remaining required for real automatic rollover

1. Connect this activator and the trusted hooks to `train_elastic` instead of
   just providing callable primitives.
2. Implement dual-generation incoming/outgoing WebRTC event routing and
   snapshot negotiation while preserving the same user-visible model identity.
3. Implement a valid, distributed atomic checkpoint cut, including new epoch
   proposal/vote generation (currently explicitly disabled in the v47 worker).
4. Ensure old-generation event carry and late disconnected peers are handled
   across multiple generations with journal-based crash replay.
5. Only then enable GC of currently running workers' logs after verified
   new-generation acceptance. The existing fail-closed GC must remain.

### Tests

```bash
python -m pytest -q tests/test_live_epoch_switch_v48.py tests/test_hot_swap_v45.py \
  tests/test_generation_committee_v43.py tests/test_live_generation_transport_v42.py \
  tests/test_generation_multicast_v46.py tests/test_epoch_carry_v46.py \
  tests/test_destructive_gc_v47.py tests/test_epoch_runtime_v43.py \
  tests/test_continual_elastic.py tests/test_snapshot_reconnect_regression.py
```

The tests cover a running PyTorch AdamW step after switching to a new model,
a parallel relay task, a cancellation/deadlock regression, deterministic
version IDs, manifest and shard corruption, missing epoch certificates,
Gossip control and existing verified GC.

Do not delete old `.sqlite3`, `.fplog`, `.train.lock`, signed receipts, or
checkpoint stores based on the v48 trainer switch tests alone.
