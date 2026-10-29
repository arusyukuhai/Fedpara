# Kagami v44 — lower overhead generation-control patch

This patch responds to the concern that the *control plane* for online epochs
could slow learning down even when nothing changes. It is a compatible update
on the experimental v43 source; it does **not** pretend to complete seamless
hot model/optimizer rebase or automatic live-log deletion.

## Implemented and tested

- Change-triggered WebRTC generation-head notices, plus 30-second idle
  anti-entropy instead of 0.6-second repeated announcements. New connections
  receive a head immediately; committed generation changes are advertised on
  the next loop iteration.
- Cached generation certificate and checkpoint availability for each tick
  rather than repeating SQLite and filesystem I/O for every neighbor.
- 120-second idle checkpoint-availability notices (formerly 3 seconds),
  immediately emitted when the known locally available manifest changes.
- Exponential recovery and missing-checkpoint request backoff, up to 30 and
  60 seconds respectively; unanswered nodes cannot monopolize the control
  channel. Disconnection/reset restarts discovery promptly.
- `patchlib/frontier_commitment_v44.py`: independent paged Merkle frontier
  commitment. The header stays <256 bytes for a test frontier of 4,096 peers,
  with 32 peer IDs per page and integrity proofs. A committee must obtain and
  verify *every page* and independently verify the exact checkpoint cut,
  because one page proof is not enough to certify complete coverage.

## Engineering estimates (not observed Internet results)

Assume 1,000 peers, degree=8, no generation changes, all peers have the same
snapshot. Pre-patch, the steady-state head + have notifications alone would
be about `1000 × 8 × (1/0.6 + 1/3) = 16,000` directed control frames per second.
After this patch their steady-state rate is about
`1000 × 8 × (1/30 + 1/120) = 333` per second, ~48x fewer. Before a snapshot
exists, the change is ~50x for head notifications alone. Actual traffic may
include votes, event Gossip, snapshots, rejoin catch-up and non-generation
protocols; their bandwidth is **not** reduced by this patch.

## What remains before activating automatic live GC

1. Current v1 `make_proposal` still carries a **256-member** inline `cutoffs`
   mapping. The paged frontier module is deliberately isolated until protocol
   negotiation, complete proof downloads, and old-journal coverage scanning
   support the compact commitment format. No partial frontier is accepted as
   a GC authorization.
2. There is no atomic trainer/optimizer handoff. A checkpoint-creation worker
   must prepare weights asynchronously, validate that the checkpoint includes
   the stated immutable frontier, and use a brief durable publication barrier.
   A strictly instantaneous swap with zero compute interruption cannot be
   guaranteed. Optimizer state must either be correctly transformed for the
   new parameterization or reinitialized with corresponding learning-schedule
   considerations.
3. Applying LoHa factor deltas from an older base to new factors is **not**
   generally mathematically correct. Updates must be expressed as verified
   effective-weight deltas with the appropriate semantics or recomputed from
   training data. Unpublished local updates can be quarantined and replayed,
   but not silently marked incorporated.
4. Real automatic GC is still disabled for active v1 trainers. Require a
   fully verified new generation in durable storage, acknowledged suffix
   handling, release of the old trainer journal lock, and a crash-safe
   retirement tombstone before deleting payloads.
5. The existing 3–31-member committee is crash-fault tolerant, not Byzantine
   fault tolerant. A hostile committee model requires a different quorum
   threshold and security proof. Internet-scale load, GPU overhead, and
   real online rejoin have not been tested in this environment.

## Run targeted tests

```bash
python -m pytest -q tests/test_v44_low_cost_generation.py \
    tests/test_live_generation_transport_v42.py \
    tests/test_online_generation_v41.py tests/test_epoch_runtime_v43.py
```

No existing checkpoints, logs or trainer journal paths are modified by the
new Merkle-frontier module, and this package contains no migration that deletes
an existing v1 journal.
