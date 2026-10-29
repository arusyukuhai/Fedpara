# Kagami v53 — live trainer adoption and authenticated adjacent-epoch repair

## Status: functioning conditional hot-adoption path; NOT full 4,096-peer automatic finalization

**v53 is not a proven fully automatic, uninterrupted, indefinitely bounded-disk
4,096-device training system.** In contrast with v52, `train-live` now *does*
automatically prewarm and activate the next **already certified** epoch after
verifying its locally available checkpoint. This does **not** mint a certified
checkpoint, collect quorum votes, or solve large-cohort agreement.

### Newly integrated

1. `patchlib/elasticnet.py` and `patchlib/live_epoch_switch_v48.py`:
   - Detect the next **certified** generation in the live training loop.
   - Independently stage model + adapter + optimizer + new journal and replay the
     signed, not-yet-covered source journal suffix into a shadow model.
   - Prewarm incrementally in a separate asyncio task while the original
     optimizer and WebRTC sender keep running. Preparation never publishes
     active state or rekeys the active session.
   - Activate **only in the training coroutine between optimization steps**:
     seal local gradients, fence original writers, stop and drain the old
     WebRTC receiver including its bounded pending message queues, persist the
     final carry receipt, attest the newly staged journal, then write a durable
     active-epoch pointer before exposing the new model/optimizer/journal and
     reconnecting authenticated WebRTC peers under the new genesis.
   - If an irreversible fence or pointer publication is uncertain, fail closed
     and require durable recovery rather than resume an old writer.
   - Reuse the same staged shadow between slow anti-entropy polls; avoid model
     reloads/checkpoint rehash on each poll. Immutable shard hashes are checked
     again before the final swap.
   - Use the v52 restart checker to validate the active pointer, source fence,
     certified manifest, carry receipt and archive after a process restart.

2. `patchlib/epoch_carry_gossip_v53.py`:
   - An adjacent-epoch, **Ed25519 WebRTC-control authenticated** anti-entropy
     lane: offer signed event identities, request only missing items, stream
     8-KiB chunks with SHA-256 body verification, then submit entire ORIGINAL
     signed event to `AsyncInboundPipeline`. No unsigned cross-epoch optimizer
     update is ever applied.
   - Operates over existing bounded-neighbor connections, **not a full mesh**.
     Both old↔new and new↔new links may relay the *previous* generation's
     signed updates via persistent carry archives; recovery remains possible
     after the last old-generation sender leaves, provided some new peer kept
     the archive.
   - One requested item and one streamed body per direct neighbor. The wire
     limit is 16 MiB per event and four chunks per neighbor per ~50-ms tick;
     a larger patch is **not** supported by this repair lane. Per-neighbor
     stream memory is bounded by one event body; inbound fragments are spooled
     durably to disk and orphaned temporary fragments are removed on startup.
   - Fixed-size 16-item inventory: 8 newest updates for low tail latency +
     8 old updates from an indexed rowid cursor, to avoid starving long offline
     recovery or delaying fresh updates behind millions of old rows. Retry
     across disconnects and reverify all signatures upon ingestion.

3. `patchlib/epoch_local_stage_v53.py`:
   - Verify the certified content-addressed manifest, semantic shard hash and
     all shard checksums, and materialize a new checkpoint view with **hardlinks
     rather than copying** full model weights. This requires a single volume.
   - Check symlinks and content collisions, fsync directory metadata and keep
     source snapshot objects intact. The staged view is *not* a certificate.

### Safety guarantees and limitations

- Old and new epochs **never share an optimizer or writable event journal**.
  Old signed LoHA updates must be durable and independently verified before
  replay on the new certified base. No unverified weights or ambiguous carry
  may trigger an epoch transition. The old history is NOT automatically
  destroyed when a node switches.
- The actual local handoff includes a *bounded intended step-boundary pause*
  while writers drain, last carry is committed and WebRTC reconnects. Heavy
  tails or many missing updates can make this pause longer. It is **not** a
  mathematically zero-latency switch, although other connected peers keep
  training and the local process is not restarted as part of normal adoption.
- Epoch certificates and immutable cut/manifest inputs must already exist.
  Automatic crash-atomic checkpoint-cut formation, safe multi-node quorum
  voting, and automatic finalized frontier proofs are **not implemented**.
  Existing v1 cut encoding supports only <=256 origins: **4,096-origin
  certified transitions cannot work end to end yet**. A paged v2 proof protocol
  is necessary.
- Repair is for a *single adjacent source epoch*. A member offline across
  multiple epochs needs a verified certificate/checkpoint chain; this is not
  fully automated. If the last archive holder disappears, this version cannot
  conjure lost updates. Unproven old data is retained instead of GC'd, so disk
  use has no formal long-term global upper bound.
- Memory needs a second full model during shadow preparation. Large models
  require out-of-core or shard-wise warm activation not provided here.
- Maximum old carry event size is 16 MiB, regardless of a larger regular
  update configuration; no fragmentation of an oversized original signed
  event across higher-level signed fragments is implemented.
- The short synchronous finalization and WebRTC reconnect have not been
  benchmarked against thousands of real GPUs, browser endpoints or network
  partitions. The v51 4,096-node sparse-graph test was topology only.

### Operator notes

- Preserve the existing generation committee configuration and signing keys.
  No unsigned/manual "force new epoch" action has been added.
- Provision storage for checkpoint objects, at least one staging view, source
  and target SQLite journals, an adjacent signed-carry archive and the spool.
  Use the existing disk-budget limits. Do not delete source journals while
  unverified carry or a handoff is in progress.
- For certs received before the checkpoint shards, the trainer remains on the
  original generation and retries staging after missing content is fetched.
- Verify your exact model architecture supports loading an independent shadow
  model and optimizer; don't assume the small-model tests prove 300B support.

### Tests and files

- `tests/test_epoch_carry_gossip_v53.py`: 5 cases including real signed LoHA,
  disconnect/replay, tampered frames, same-generation archive relaying after
  original sender departure, and recent-event/fair-history inventory.
- `tests/test_epoch_local_stage_v53.py`: 3 cases, including hardlink reuse,
  checksum tampering and absent manifests.
- `tests/test_epoch_live_rebase_v52.py`: one new deferred-shadow test in
  addition to the existing signed carry suite.
- `VALIDATION_V53.md` records precise test commands and baseline unrelated
  failures. `CHANGES_V53.patch` has the v52→v53 code difference.
