# Kagami v54 — long-running epoch repair hardening (2026-10-27)

## Scope

This is a **hardening revision of v53**, not a claim of a fully automatic
4,096-origin/300B-scale system. The same certified-checkpoint prerequisites
and v53 limitations remain. No signing checks or epoch writer fences were
weakened to improve benchmark/test results.

### Production changes

1. **Constant-space authenticated carry index.** `CrossEpochCarryArchive`
   no longer allocates an ever-growing `set((origin, seq), ...)`. Its indexed
   durable EventJournal is the authority for deduplication, and completed
   replays retain only a counter. Concurrent asynchronous accepts share an
   archive-local commit lock. Any post-durable-append model-apply failure or
   cancellation permanently poisons the in-process archive, requiring clean
   checkpoint recovery; it cannot silently continue with partial tensors.
2. **Non-blocking cross-generation NEED service.** `AdjacentEpochCarryGossip`
   queues at most one bounded pending disk fetch per live neighbor. Original
   signed updates are read with `EventJournal.get_async` on the dedicated
   journal worker, rather than blocking WebRTC control callbacks while
   reading and checking 16 MiB of data. The relay wrappers in `elasticnet`
   expose the correct worker-thread method. No full mesh was introduced.
3. **Bounded delivery under backpressure.** A verified spool is loaded only
   once while an inbound queue is saturated, rather than reread every 3 ms.
   Submission retries are delayed 50 ms after failure; a disconnection
   frees spool, cached bytes, and pending requests. Temporary spool fragments
   remain untrusted and are discarded on restart.
4. **Asynchronous spool verification.** Finishing the last 8-KiB frame no
   longer fsyncs and SHA256-hashes a 16-MiB temporary file in a DataChannel
   callback. Verification is performed asynchronously before bytes can reach
   the signed ingestion pipeline. The pipeline's EventJournal still fsyncs
   authentic payloads before applying them; failure does not ACK data.
5. **Atomic, resumable local checkpoint-view publication.** The certified
   view is fully hardlinked and verified in a private `.building-*` sibling,
   and made visible with a rename only after directory durability barriers.
   A process-wide advisory flock serializes cooperative stagers. Abandoned
   staging views are reclaimed after crashes; existing published corrupt or
   legacy partial views fail closed instead of being silently rewritten.
   Replaced O(shards) directory fsync calls with O(containing directories).
6. **Desktop and test alignment.** UI defaults expose actual local steps
   and maximum context length. A scoped text-caret/focus correction was
   added for macOS WebView. Obsolete tests were updated to simulate a
   *completed authenticated HELLO*; they no longer demand unsafe pre-auth
   sending. Tests for an old benchmark tab and old CDN-reliant UI were
   updated to the current bundled/local-only desktop UI.

### New test scenarios

`tests/test_epoch_v54_hardening.py`: inbound saturation does not reread spool;
async worker-bound NEED reads; checksum-tampered spools fail closed; eight
concurrent signed carries apply at most once; injected disk failure cannot
publish a partial view and stale temporary directories are cleaned.

### Explicit remaining limitations (unchanged in v54)

- Autonomous global snapshot cut, certificate quorum formation and the >256
  origin v2 cut proof are not completed. **4,096 peers cannot yet perform an
  independently certified epoch transition end-to-end.**
- Multi-epoch offline reconciliation, indefinitely bounded storage while
  partitions never rejoin, production 300B out-of-core dual-model swaps,
  and worst-case zero-pause trainer handoff remain unsupported.
- Remote regular training and epoch-control connections have not been
  soak-tested in a real 4,096-GPU/devices swarm. The focused micro regressions
  demonstrate *removed redundant work*, NOT throughput acceleration in %.
- `verify_certified_cut`, final carry inventory digest, model-shard hashing,
  and WebRTC reconnection may still make the short final boundary expensive.
  No unsafe shortcuts were taken through these proofs.

### Operational notes

Use the same commands as v53. Restart normally with the durable pointer
checker; do not discard old signed histories unless `EpochRuntimeV43`
proves they are safe to GC. If upgrading from v53 and a legacy incomplete
`checkpoints/views/<manifest>` exists, quarantine/reconcile it manually; v54
refuses to overwrite a possibly active view. A `*.building-*` temporary
view from v54 is safe to discard when no stager holds the directory lock.
