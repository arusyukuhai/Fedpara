# Fedpara v16 — P2P connection churn hardening

This is a **robustness-only** update on top of the Tailwind v15 GUI. It does not
modify FedSum, adapter rank selection, signatures, genesis format, or the binary
journal on disk.

## Changes

- **Per-channel isolation in PatchNode:** old DataChannel message callbacks are
  ignored after replacement, including when callbacks arrive after pause/resume.
  Incomplete on-disk snapshot `.part` files belonging to a replaced channel are
  closed and removed even when the old close event never fires.
- **Per-channel control reset in ElasticNode:** when its DataChannel is replaced,
  old queued controls and snapshot status latches are removed. The new transport
  must authenticate a fresh signed hello. Old snapshot requests must be
  reissued rather than accidentally served to a newer RTC generation.
- **Critical snapshot progress during trainer stalls:** when the normal bounded
  control queue is full, the latest signed `snapshot-status` for the currently
  requested snapshot is kept separately. It is processed before old queue
  entries. Stale status reports for another request ID are ignored. Existing
  `snapshot-tip` overflow handling remains intact.
- **Bounded anti-entropy:** a signed inventory page cannot increase the
  set of missing identities above `FRONTIER_PENDING_LIMIT` (4096). Subsequent
  inventory/frontier cycles can discover deferred missing identities.
- **Requested update retry:** a requested Gossip event ID is put back at the
  head of its priority backlog if its SCTP send fails or is cancelled.
  The durable journal and frontier/need anti-entropy remain the ultimate source
  of truth across a completely new node instance.

## Tests

`python -m pytest -q tests`: **345 passed, 3 skipped** (this runner).
Additional tests cover stale patch callbacks, incomplete snapshot cleanup,
control queue pressure while paused, late signed control messages, bounded
missing-event lists, and forced Gossip transmit failure.

## Compatibility and limits

This version uses the same v15 on-wire signed event format, model layout and
journal schema, so **no fresh room is required**. Back up both SQLite and
`.fplog` first, and stop both training processes before replacing program files.

These unit and regression tests use fake RTC channels for injected failures;
full long-running, real five-peer ICE/TURN/WebRTC churn is not validated here.
Robustness does not imply zero-loss execution for a disconnected peer: recovery
of published updates relies on the signed journal and anti-entropy, and local
optimizer steps not checkpointed before a crash can still be lost.
