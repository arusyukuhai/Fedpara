# Kagami v52 — Certified streaming carry, warm hot-swap, and restart recovery

## Scope and truth-in-testing

**This is NOT yet the promised fully automatic, 4096-node, no-stop generation
migration.** It is the next safe implementation increment over v51. In
particular, v52 **does not automatically commit checkpoints, generate quorum
votes, or invoke the live epoch switch from production `train-live`**. These
are remaining requirements, not features to toggle on optimistically.

## Implemented in v52

1. `patchlib/epoch_live_rebase_v52.py`
   - Streams the exact signed old-genesis suffix by increasing journal rowid,
     **not by OFFSET or an all-events Python list**. A signed update arriving
     late from a disconnected origin cannot hide behind an already-scanned
     origin/sequence cursor.
   - Compares *all* covered cut indexes against the certified SHA-256 cut,
     including exact per-origin terminal sequences. Scans memory O(origins).
   - Bounded per-tick processing: default at most 16 events / 16 MiB of wire
     payload (the tensor evaluation itself needs additional model memory).
   - Uses the existing independently authenticated `CrossEpochCarryArchive`:
     append signed original bytes durably before applying their effective
     weight deltas. On restart, reconstruct a clean certified base and replay.
   - On a failed signed-wire check or incomplete/corrupt carry, poisons the
     shadow, which cannot be published. Resuming a previously applied durable
     archive avoids re-decoding carried events but checks event identity hash.
   - Under a **caller-supplied verified old-writer barrier**, performs a final
     suffix sweep, rechecks the cut, and writes a crash-atomic + fsynced local
     carry receipt containing the exact archive inventory digest. The receipt
     is NOT authority to delete old signed history.

2. `patchlib/live_epoch_switch_v48.py` (service strengthened)
   - `PreparedEpoch.rebaser` supports streaming suffix staging while the old
     `TrainerStepRouter` still admits training steps. Each poll is bounded;
     the service keeps the same prepared shadow between polls rather than
     reloading and replaying it from scratch.
   - A locally controlled `old_ingress_frozen` hook must attest that the old
     receiver/writer is actually drained and frozen before final carry.
   - The trained checkpoint is reverified before final publication. A failed
     snapshot, unexpected old writer, conflicting certificate or unverified
     carry prevents activation. No old history is silently removed.
   - Background old-history GC does not poll for a nonexistent epoch-0 active
     pointer while shadow preparation is in progress.

3. `patchlib/epoch_trainer_boot_v52.py` and `patchlib/elasticnet.py`
   - `train-live` now detects a persisted pointer at
     `<journal>.live-epoch/active-epoch.json` **before opening the initial
     generation journal**.
   - It reconstructs the pinned committee's signed certificate chain, verifies
     the pointer against the exact epoch/manifest/genesis, hashes all shards,
     installs the certified base before new optimizer creation, and validates
     the durable old signed carry plus its final receipt before replaying the
     new-epoch journal.
   - Rejects unsafe paths, a missing epoch/certificate, missing source or new
     journal, an old writer that appended after the recorded handoff, a cut
     mismatch, and corrupt or missing carried events instead of guessing.
   - Pins the immutable original generation root in future generation-control
     `hello` frames even after restoring a successor epoch.
   - Plugs the existing signed old-epoch inbound carry lane into
     `AsyncInboundPipeline` on certified restart.

4. Regression tests
   - Added `tests/test_epoch_live_rebase_v52.py` with actual signed LoHA,
     three successive old updates, partial background staging, final flush,
     warm-switch service, offline/restart replay, and poisoned-shadow rejection.
   - Added `tests/test_epoch_trainer_boot_v52.py` with a real sharded checkpoint,
     pinned certified pointer verification, durable carry-receipt validation,
     old-writer resurrection rejection and a dangling symlink safety test.
   - Bounded live service tests cover incremental staged reuse, actual PyTorch
     optimizer swap and refusal to switch without a writer fence.

## Remaining work — must be completed before describing it as production-ready

- **Production trainer hook is not yet connected.** The real `train-live`
  optimization loop still needs to use `TrainerStepRouter.step()`, create and
  stage certified shadow models in background, invoke the prepared switch, and
  atomically rewire its shared `model`, `adapter`, `optimizer`, `journal`,
  `engine` and existing WebRTC sessions under a drain/fence. v52's
  `LiveEpochSwitchService` is a tested opt-in library, NOT an automatically
  called production switch.
- **No automated certifiable snapshot cut and quorum votes.** The v51 live
  generation bridge continues to refuse automatic votes without an atomic
  trainer-produced cut receipt; blindly enabling votes would risk
  irreversible model divergence.
- **Old-world/new-world anti-entropy is incomplete.** After adoption, the
  transport must relay old signed carry events and their inventory among
  *different generation* members (including long-offline peers). The current
  WebRTC bridge exchanges signed generation certificates but does not carry
  full cross-generation training history.
- The legacy `online_generation` v1 proposal embeds <=256 cutoffs and a
  <=40 KiB signed control payload; **a real 4096-peer certified cut needs the
  paged Merkle-frontier v2 validation and transport integrated end-to-end**.
- Checkpoint cache GC and old journal destruction remain conservative. Old
  data can continue growing until a separately proven finalized coverage
  checkpoint exists. A 300B+ shadow model needs explicit memory/disk capacity
  management; tests here use small networks, not large-model benchmarks.
- A future GC design must allow restart after legitimately verified old-source
  deletion. This version still requires the old source journal for verifying
  its durable final carry boundary.
- No 4096 physical devices, network partitions lasting months, multi-week
  storage-bounded benchmark, or sustained Byzantine failure test was run.

## Local file contract (not a manual activation shortcut)

A *previously* durable pointer is only a locator. For epoch E, local recovery
looks for:

```
<journal>.live-epoch/active-epoch.json
<journal>.live-epoch/checkpoints/views/<manifest-hash>/checkpoint.fmanifest
<journal>.live-epoch/checkpoints/views/<manifest-hash>/objects/...
<journal>.live-epoch/carries/epoch-XXXXXXXX.sqlite3
<journal>.live-epoch/carries/epoch-XXXXXXXX.sqlite3.handoff.json
<journal>.live-epoch/journals/epoch-XXXXXXXX.sqlite3
```

Never hand-edit the pointer or forge a handoff receipt. These files must be
created from a verified snapshot, durable signed carry, clean new journal and
properly fenced training writer. `train-live` intentionally refuses partial
state; it cannot safely recover a locally missing carry by guesswork.

## Tests

```
python -m pytest -q \
  tests/test_epoch_* tests/test_online_generation_v41.py \
  tests/test_live_generation_transport_v42.py \
  tests/test_v44_low_cost_generation.py \
  tests/test_generation_multicast_v46.py \
  tests/test_compact_epoch_network_v46.py \
  tests/test_live_epoch_switch_v48.py \
  tests/test_continual_elastic.py tests/test_elastic.py
```

Expected result on this source snapshot: **111 passed**.

## Performance expectations (not measurements)

- Gossip fanout remains bounded to <=8 direct neighbors per peer, not a
  complete graph; the original anti-entropy path is unchanged.
- Warm carry resident metadata is bounded by O(origins) for cut validation
  and O(batch size) for scanning; the per-event torch delta can be large.
- Warm carry sends no extra mandatory all-to-all wire traffic itself. It
  uses an existing local signed journal; actual old-event replay over peers
  still needs integration.
- During the handoff, the old event archive, new-generation carry and the
  staged certified model coexist. **Disk is not strictly bounded without
  completed, independently verified generation GC**. v52 refuses to erase
  unrecoverable signed history merely to meet a quota.
