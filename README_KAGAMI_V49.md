# Kagami v49 — sparse low-diameter control / transport patch

## Actual scope (important)

This version **does not** complete unattended online generation rollover. The
existing v48 trainer still pins its active optimization engine, checkpoint base,
SQLite journal and incoming event validation to its original genesis, and
`AutomaticGenerationControl` deliberately has no trusted way to attest a
concurrent checkpoint cut. Turning on automatic `gc()` before these are
transacted together would corrupt diverged learners and offline recovery.
No files are deleted by this patch. The previous v47 fail-closed GC remains.

### Implemented and exercised

1. Constant-degree 4-ring small-world neighbor overlay for >32 peers, with a
   deterministic 64-bit FNV1a+fmix permutation for each ring. Ring 0 is the
   sorted participant ID ring, which remains connected. Rings 1-3 are
   independent deterministic hashed permutations. The signaling server and
   Python simulator compute the same order. Eight direct neighbors maximum.
2. Membership joins/leaves recompute the sparse overlay once per update in
   the Nim signaling server. Each ring affects only two adjacent existing
   neighbors on insertion; up to 16 existing peers' neighbor sets change.
   There is no 4096-recipient broadcast of a full roster.
3. Adaptive, signed frontier advertisements: synced peers in large rooms
   slow redundant rotating vector-clock pages to 1.5 seconds, while an
   unsynced peer keeps the 0.3 second gap-repair schedule. During explicit
   swarm catch-up, the fast schedule remains unchanged.
4. Expensive full-event-inventory identity audits for synced, frontier-v1
   capable links run no more often than every 120 seconds in large rooms.
   Frontier/NEED hole repair and signed push updates remain active at all
   times. The old schedule remains when peers lack frontier support.
5. A stand-alone benchmark for graph diameter, degree, connectivity, and
   join rewiring: `python scripts/benchmark_v49_overlay.py --peers 4096`.

### Local simulation, 4096 peers

- 16,375 undirected direct links across the room (<=16,384).
- One sampled source's average shortest path: 4.346 hops; max 6 hops.
- 14 existing participants' neighbor lists changed during a one-out,
  one-in capacity-preserving membership update.
- A complete mesh would have 8,386,560 direct links.

These are **graph** metrics, not measured end-to-end WebRTC throughput or
wall-clock training throughput. Actual latency may be dominated by TURN,
SCTP buffering and aggregate tensor transfers.

### Integration / deployment

Update BOTH the Kagami trainers and the `src/main.nim` signaling server.
The bounded-v2 wire response remains syntactically compatible, but the
signaling process MUST be redeployed for new sparse long-distance links.
The long-lived client receives the server's neighbor list and does not
independently guess different links.

The Python side was tested here. This runtime does not include Nim, so the
modified Nim server was NOT compile-tested; compile before deploying:
`nim c -d:release --threads:on --mm:orc src/main.nim`

### Why full uninterrupted automatic rollover is not yet safe

The following transaction has to be implemented inside `train_elastic`, not
merely in detached utility modules:

- Freeze and sign an *atomic* applied-update frontier + corresponding exact
  model-weight checkpoint (while other peers keep training). Verify it on
  multiple durable replicas and bind the snapshot to a quorum certificate.
- Stage a fully loaded new model, optimizer, journal and signed event receive
  pipeline, while the old epoch still services offline or delayed peers.
- At an optimizer-step boundary, durably seal local LoHa state; partition
  exactly the before/after-cut events; rebase post-cut *effective-weight*
  deltas using original origin signatures and persistent deduplication.
- Atomically swap the entire live engine AND switch routing of outgoing AND
  incoming signed event traffic. Recover from a crash between disk commit and
  transport activation. No `genesis`-blind event may cross the boundary.
- Allow disconnected peers to fetch any missed certificates, then download a
  verified current checkpoint (with multiple available replicas).  Promote
  only when fully verified. Old logs are GC-eligible only after durable
  full-state installation and suffix accounting, never on quorum alone.

Exact model state equality is not implied by identical sets of FP32 deltas:
addition order affects rounding. Validators must attest one identical
canonical checkpoint, not individually export their own concurrent states.

**Do not enable destructive GC of an in-use live v1 journal or pretend that
all old versions will automatically catch up.** Full rollout and real
partition/churn tests are still required.

### Bandwidth limits still remaining

An all-to-all replica policy in which every one of N peers stores all N
independent signed updates has O(N^2 * update_size) *aggregate* network cost
per global round in the information-theoretic worst case. A better topology
and batched transport remove redundant links and some framing, but not this
cost. To reduce it in general, Kagami needs certified aggregated updates or
checkpoint-based bounded history with changed synchronization semantics.

### Tests

```
python -m pytest -q \
  tests/test_v49_overlay_scale.py tests/test_v49_gossip_cost.py \
  tests/test_large_room_bounded_topology_v17.py \
  tests/test_v44_low_cost_generation.py \
  tests/test_generation_multicast_v46.py \
  tests/test_live_epoch_switch_v48.py \
  tests/test_destructive_gc_v47.py \
  tests/test_continual_elastic.py \
  tests/test_sparse_gossip_v18.py
```
