# v50: bounded checkpoint preparation and safer epoch publication

**Status: partial implementation; not a completed end-to-end live generation migration.**

This archive contains the original v49 project plus the changes described below.
The existing GUI / `train-live` worker still uses v1 genesis-bound ElasticNode
connections. This patch DOES NOT make that production worker automatically
switch generations. Do not deploy it under the assumption that it does.

## Implemented and connected to the existing switch service

- `epoch_objects_v50.py`: exclusively owned, persistent content-addressed cache;
  byte reservations before download; object count and metadata page limits;
  explicit owner pins; only unpinned cache objects can be evicted.
- Transfers resume from the last durable block after disconnect or process exit.
  Publication requires full SHA-256 verification. Recovery handles both the
  last-block-before-rename and rename-before-SQL-commit crash windows.
- `epoch_checkpoint_v50.py`: a dedicated disk worker keeps fsync, hashing and
  SQLite off the relay event loop. Checkpoint views use hardlinks so unchanged
  shards occupy disk once. A shared per-node token bucket limits requested
  payload bandwidth, with a bounded burst. At most eight candidate neighbors
  and one in-flight shard block by default; retries and RPC timeouts are bounded.
- `LiveEpochSwitchService(checkpoint_stager=...)` uses this preparation path
  before calling the supplied model/optimizer constructor. The old trainer
  remains admitted while the new checkpoint downloads. The constructor must
  return the exact staged paths; independent manifest/shard checks still run.
- Rejoining switch services walk intermediate committed certificates in order
  instead of rejecting any head more than one generation ahead forever.
- Pointer publication now fails closed even if rename succeeds and directory
  fsync fails. `resume_publication()` retries only pointer/network publication;
  it does not flush or replay the carried suffix again. The network handoff
  callback must be idempotent.
- Watcher failures use bounded exponential retry delay with jitter. Restart
  refuses an initial trainer inconsistent with the durable active pointer.

## What is NOT implemented

1. A new generation-aware data plane in the actual ElasticNode / GUI worker.
   Its current signed hello rejects a different genesis. Relaxing that check
   without separating update validation and routing would be incorrect.
2. Automatic, verified candidate snapshot creation, voting and adoption from
   the production training loop. The existing bridge's vote guard remains
   disabled deliberately; it lacks an atomic model/certified-cut proof.
3. Complete transport of old-generation late contributions, with end-to-end
   deduplication across arbitrary generations and checkpoint cuts. Existing
   carry components have tests, but are not a complete production integration.
4. Automatic journal/carry retirement, checkpoint lease release and bounded
   certificate-history compaction. The new cache's `retire()` is LOCAL lifecycle
   functionality, not evidence authorizing retirement. It must only be called
   after independently proving no live model or recovery path needs the base.
5. Production-worker crash reconstruction from the new pointer. The service
   detects a mismatch instead of silently restarting the old generation.
6. Automatic liveness-preserving proposer recovery after split votes. The
   existing one-vote-per-epoch ledger alone does not implement a complete
   leader-election/ballot recovery protocol. Its safety checks must not be
   relaxed to recover from a split vote.
7. A real multi-thousand-device training/rollover deployment or long-duration
   hardware validation. The included 4096-node test is an overlay topology test.

These gaps are why this is not labelled a finished no-stop migration release.

## Resource semantics

`quota_bytes` limits reserved UNIQUE checkpoint payload bytes, including the
manifest and partial downloads. Active/candidate pins are protected. If both
cannot fit, preparation raises `CapacityError`; it does not delete the active
base. Optimizer RAM/VRAM, old journals, carry archives, and certified history
are outside this cache quota. SQLite has a separate page limit; its rollback
journal and filesystem overhead require additional disk headroom.

`bytes_per_second` limits this fetcher's requested checkpoint payload bytes.
Use one stager per node to share the budget. It does not limit all existing
WebRTC traffic, signatures, headers or the old gossip engine. Authentication
and rejection of oversized responses BEFORE buffering are responsibilities of
supplied transport callbacks. Manifest requests conservatively charge the
maximum accepted manifest size. Bad responses/retries also consume budget.

No local API can guarantee lossless retention of unlimited disconnected
training history in finite storage. The production policy still needs explicit
backpressure or a verified aggregation scheme. Never delete unincorporated
updates just to claim bounded disk usage.

## Integration API (not a GUI feature flag)

Create `BoundedCheckpointStager(root, peers=..., fetch_manifest=...,
fetch_block=..., quota_bytes=..., bytes_per_second=...)`. Callbacks:

- `peers()` yields authenticated CURRENT neighbors, not a room-wide roster.
- `await fetch_manifest(peer, digest, max_bytes)` returns bounded manifest bytes.
- `await fetch_block(peer, digest, offset, length)` returns exactly that range.

Pass this stager to `LiveEpochSwitchService(checkpoint_stager=stager, ...)`.
With this option, `build_shadow(proposal, checkpoint)` takes two arguments;
without it the original one-argument callback is unchanged. The checkpoint has
`manifest`, `store`, `owner`. Construct a separate model, adapter, optimizer and
durable journal, then return `PreparedEpoch`. Existing flush, carry, attestation,
verification and idempotent network handoff hooks remain mandatory.

After a separately verified retirement, `await stager.retire(checkpoint)` removes
its view and releases its cache pins. Never call it just because a neighbor
reported a newer head. Shutdown calls `await stager.close()`.

## Validation

See `VALIDATION_V50.md`. Tests include real PyTorch weight loading/optimizer
steps during blocked transfer; the transport and lifecycle hooks in that
integration test are test doubles, not an actual multi-peer WebRTC migration.
