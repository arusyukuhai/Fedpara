# Kagami v42 — authenticated epoch transport & conservative adoption controls

**Status: partial integration / experimental. Do not enable automatic deletion in production.**

The original request is *fully online consensus + live trainer rebase of signed updates + automatic GC + offline checkpoint recovery*. This release does **not** yet implement all of that. It connects epoch certificates to the actual v1 WebRTC signed DataChannels, adds bounded cert catch-up and exact-content snapshot requests, and adds deliberately fail-closed local adoption / GC components. It does **not** rotate the running optimizer and journal into a new epoch automatically, does not preserve all offline-only gradients, and cannot yet automatically GC the active v1 journal.

## Implemented

- `patchlib/live_generation_transport.py`: certificate proposal/vote/quorum-certificate gossip on already authenticated WebRTC controls; head announcements, chain catch-up with bounded indexed SQLite pages, retransmission after backpressure, link reconnect re-request. Direct channels negotiate `generation_v1`; old peers don't receive unknown messages. Thousands of historical participants do not accumulate per-peer memory: keep at most 256 link state entries, with bounded per-neighbor replay cache.
- `patchlib/elasticnet.py`: processes generation control frames outside the ordinary 128-entry worker control queue, negotiates capabilities in the signed hello, and services exact committed checkpoint requests over the existing hash-verified shard state DataChannel. Transport messages cannot mutate model weights directly.
- `patchlib/live_generation_adoption.py`: durable new-generation readiness and post-cut index; reconstructs journal suffix by SQLite cursor (not one giant Python list); refuses to GC an open old trainer, rejects a changed frontier or a missing activation marker. Does not pretend that arbitrary strings are proof of lossless gradient rebase. Automatic local GC polling (`collect_ready`) applies only to *already adopted*, entirely covered epochs without any post-cut events.
- `--generation-v1-committee ./committee.json` for authenticated `train-live` / elastic mode. The committee is a stable JSON array of 3–31 pinned Ed25519 peer IDs. All committee members must be seen and authenticated before the local bridge starts. No proposer is chosen automatically. The callback for casting votes is deliberately disabled until the trainer can produce a signed, atomic live checkpoint boundary that binds weight contents to an exact event cut. Do **not** enable this on production models assuming it performs generation rollover.

## Why live online rollover is not complete

1. **Snapshot/cut atomicity:** a shard manifest hash and matching event frontier do not on their own prove that the model weights correspond to that exact frontier. The actual trainer must export and sign them under the same model / inbound-commit lock, then receivers must verify it before voting.
2. **In-flight updates:** if a checkpoint covers origin A up to sequence 100 while A or another offline peer has independent updates not in the checkpoint, rebasing those into a new generation requires verifying the old signed delta, reproducing a new epoch-native signed delta, and making the old-to-new mapping durable. A receipt string alone is not adequate proof. For now, such suffixes block GC.
3. **Model activation:** v1 `ElasticEngine` still uses one fixed genesis and event journal. Its `AppliedJournalIndex`, optimizer state, LoHa factor anchors, DataChannel inventory / NEED, startup checkpoint parameters, and active incoming worker queues all need an epoch-aware atomic swap. Do not delete a v1 event history just because a quorum accepted a proposal.
4. **Cutoff scaling:** v41 proposals carry an inline `cutoffs` map with a 256-origin bound. Real 1,000+ origin epochs need a paged or Merkle-bound frontier and proof verification, not a larger JSON/SCTP frame.
5. **Consensus assumptions:** fixed committee majority is crash-fault tolerance, not Byzantine fault tolerance. Under a lasting network partition, one cannot both guarantee no loss of every disconnected peer's unshared update and unconditionally progress/GC. Offline-only updates must be quarantined and recomputed or deliberately omitted with an explicit rule.

## Tests

Run:

```bash
PYTHONPATH=. pytest -q tests/test_live_generation_transport_v42.py tests/test_online_generation_v41.py tests/test_generation_rollover_v40.py tests/test_elastic.py
```

Tests exercise signed control admission, offline certificate catch-up, fork detection, bounded link memory for 2,000 simulated addresses, exact snapshot request, suffix persistence after restart, no-GC with unverifiable suffix, and local GC once a separate activated epoch is durably acknowledged with no suffix. These are mock WebRTC channels / small local PyTorch models, **not** a 1,000-peer Internet/GPU test.

## Data safety

Leave previous `.fedpara-gui`, journal SQLite, `.fplog`, object store, checkpoint manifests and identity keys intact. Never run manual `rm` on them to free space unless a separate verified migration proves the running trainer will not need them. This release does not trigger GC automatically on a running Kagami room.
