# Fedpara deep audit — 2026-10-22

## Scope

This is a targeted inspection of the **latest snapshot-join-recovery ZIP**, specifically Python P2P signaling, cold joins, checkpoint/shard transfer, the Elastic journal, and async liveness. It does **not** certify end-to-end readiness. Source was read and tests were executed on Linux/CPU. `aiortc` and the Nim compiler are **unavailable** here, so no real two-window WebRTC/Apple MPS test is claimed.

## Verified defects and changes

1. **Event-loop blocking during full shard verification — fixed.** In `ShardSession._finish_if_ready`, the last shard triggered `verify_shards` synchronously from the DataChannel message callback. A multi-GB snapshot can freeze the asyncio thread responsible for signaling, heartbeats, and channel callbacks. In the patched `LiveSnapshotLink` only, verification and manifest commit now run with `asyncio.to_thread`, and ACK is sent **only after verification succeeds**. The `ShardSession` synchronous default remains for standalone/offline consumers. Added a regression test with >4 shards and an intentionally slow hash verifier, asserting responsiveness while it verifies. Local shards continue to be SHA-256 checked.
2. **Streaming dataset fetching blocked the event loop — fixed.** The old expression `await asyncio.to_thread(run_step, ..., next(batch_iter), args)` evaluates `next(batch_iter)` *before* dispatching to a worker. Hugging Face network streaming and tokenization could therefore freeze the RTC event loop. The batch fetch and training step are both executed offloop in `run_step_offloop`. Added a slow streaming iterator test.
3. **Checkpoint export blocked the event loop — fixed.** Periodic `maybe_checkpoint` directly executed `export_state_sharded` synchronously; it now offloads to a thread. Snapshot-request export was already threaded.
4. **Verified snapshot did not necessarily release the cold-join barrier — fixed.** After signed manifest/shard verification and exact journal boundary match, the code marked `bootstrap.snapshot_installed` but could still await `node.ready` (independent rolling inventory). That could create an unnecessary or endless post-snapshot wait. It now commits `initial_ready` and sets `initial_complete` *after* applying the verified state and suffix updates, allowing normal asynchronous anti-entropy to continue.
5. **Live DataChannel unusable for snapshot request during signaling outage — fixed.** Previously the request loop was gated by `lobby.online`; it now accepts an already authenticated and live DataChannel even if Mummy signaling has gone down. The verified-snapshot emergency transition is also recognized on the training side. A new pure join predicate is tested.
6. **Legacy no-room-id mode regression introduced during the audit — detected and fixed.** A first implementation accessed `lobby.online` when `lobby` was `None`, raising AttributeError. Running old Elastic tests exposed it and it was corrected. Full-suite testing is mandatory after changing these branches.
7. **Hashing an outgoing shard could block the RTC thread — fixed.** Pre-transfer SHA-256 validation of the local shard now runs in a worker thread.

## Significant issues not resolved by this audit

- **Real RTC connection/ICE/DTLS/Mac WebView behavior unverified.** Mocked DataChannels and Python unit tests cannot establish that the user’s two real macOS processes will connect, authenticate, transfer all shards, and start training. Run the real two-peer diagnostic with both logs, and examine `phase/ice/rtc`, `SNAPSHOT REQUEST/OFFER/TIP/DOWNLOADED/HISTORY WAIT/READY` together.
- **Journal-history catch-up can remain blocked when the required signed updates are unavailable.** It is intentionally not safe to accept a snapshot whose vector-clock boundary differs from the local signed journal. If all holders of a required event leave permanently, this version cannot guarantee convergence. Automatic trusted epoch/checkpoint reconciliation with explicit quorum policy is still missing.
- **`seal_local` and received-event replay still perform substantial synchronous tensor/SQLite work on the asyncio thread.** Moving them naively to workers is unsafe because SQLite connections have thread affinity and the model+adapter require atomic model-generation boundaries. A proper worker-owned journal/model executor and a bounded control queue are needed for large-model real-time reliability.
- **Snapshot installation is not a fully transactional hot-swap.** The state archive is fully verified before applying, but if device allocation/copy or replay fails halfway through, the current process may hold partly changed weights. Treat it as a failed join, discard the partially initialized model object, and rebuild from trusted genesis and journal; a stronger implementation should swap a shadow model atomically.
- **No globally agreed 'latest model' under a network partition.** Elastic additive gossip can diverge between partitions; a newest peer snapshot is not necessarily globally authoritative. Signed model state and consistent local event history do not prove a consensus epoch across all peers.
- **Unexpected peer identity is self-certified, not owner-authorized.** A secret room ID gates participation; cryptographic signatures authenticate *a key*, not whether the room owner approved it. Private invite credentials, explicit admission, or a pinned room-owner signature are needed for adversarial deployments.

## Tests

- `python -m pytest -q`: full suite, including new audit regressions. CPU tests only, hardware-dependent MPS/CUDA tests skipped.
- `python scripts/check_group_presence.py` and `python scripts/check_live_p2p.py`: real local Mummy and `aiortc` verification **on the user's own machine**. A passing Python suite is **not** a substitute for these.
- Keep existing `.fedpara-gui`, keys, `.fstate` snapshots, object shards, and SQLite journals. Never solve a startup problem by deleting them.

## Recommended next acceptance gate

A real two-process E2E script should automatically assert: two distinct peer IDs; two successful signed Hellos; one authenticated `fedpara-state` channel; >4 shards delivered; identical manifest digest and prefix journal hash; `SNAPSHOT READY` and `JOIN READY` on newcomer; founder keeps training; then forcibly kill either peer mid-transfer and confirm resume from cached verified shards. Repeat with Mummy stopped after DataChannel establishment and with an actual MPS model. **Do not claim this gate passed until those logs exist.**
