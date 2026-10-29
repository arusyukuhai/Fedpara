# Kagami v56 — scalable epoch coordination and conservative recovery building blocks

**Status (2026-10-27): partial integration, NOT end-to-end 4096-device autonomous training or a validated 300B zero-pause trainer.** All candidate generation, model changes, and deletion remain subject to independently verified checkpoint/cut and writer-fence requirements. Missing evidence fails closed.

## Changes

### 1. 4096-origin compact autonomous epoch certificate assembly

- `patchlib/epoch_autonomous_v56.py`: durable, sparse P2P compact-candidate endorsements, on-demand authenticated Merkle frontier page requests, committee votes, quorum assembly, and recovery after disconnect. A 4096-origin frontier uses 128 pages of 32 entries. Each peer validates page proofs and **all pages** before voting. Proposals are first Ed25519-endorsed by a pinned committee exporter. Ordinary nodes may relay without forging votes.
- `patchlib/live_generation_transport.py`: `generation-wire` carries compact candidate/page/need/vote traffic over the existing authenticated control plane. Per-neighbor queues and page requests are bounded; large existing archives are not broadcast in bulk. Old `generation-compact-head/need/certificate` anti-entropy remains.
- `patchlib/epoch_certificate_v46.py`: verified historical certificate retrieval with explicit parent-chain checks.
- `patchlib/elasticnet.py`: train-live instantiates the compact ledger and authenticated wire relay when the existing pinned-generation committee is enabled. **Local export verification remains disabled in this production integration** (`local_export_verified=False`): a download matching hashes is NOT sufficient to claim a model/optimizer was exported at a specific cut. A real crash-consistent live exporter and trustworthy cut receipt must be wired before the committee may self-propose/vote.
- The committee has at most 31 pinned voters (crash-fault quorum), not 4096 voters. The test uses 4096 signed-update **origins**, not 4096 active WAN devices; not Byzantine tolerant.

### 2. Multi-epoch catch-up and conservative offline-update retention

- `patchlib/epoch_multihop_v56.py`: validates the signed parent chain before any transition; `MultiEpochCatchup` advances by adjacent certified epochs only, never skips missing checkpoints or carries. A bounded per-poll budget prevents an offline catch-up loop monopolizing local scheduling. Verified chain tips are cached, with each cached tip rechecked.
- `patchlib/elasticnet.py`: prewarmed conditional auto-adoption now calls this checked one-hop catch-up gate at the step boundary, retaining the v53 carry verification and writer fence.
- `DurableOfflineOutbox`: persists original (not rewritten) signed event bytes; same `(epoch, origin, seq)` must have identical bytes; quota is enforced using a durable SQLite trigger counter in O(1) per append. It *backpressures* on capacity, does not drop the oldest events. A removal requires a matching durable `CrossEpochCarryArchive` event, including identical bytes. **This new outbox is an available component; production train-live does not yet feed all its local unpublished events through it.**
- When all retainers of an event disappear or a device never rejoins, a certificate cannot recreate a missing original signed event. Indefinitely bounded storage and exact retention of all unknown peer updates cannot both be guaranteed without an explicit expiration/data-loss policy.

### 3. Sharded no-copy epoch pointer handoff primitive (300B preparation)

- `patchlib/epoch_sharded_rcu_v56.py`: `ShardedEpochRCU` independently verifies the next checkpoint/optimizer directory using an injected local verifier, prepares while old steps continue, fences admission at a local optimizer step boundary, drains old writers, atomically fsync-publishes a tiny durable pointer, and lets existing immutable readers finish on old shards. New readers receive only the new epoch. Memory used by this metadata router does not scale with total parameter count.
- This is **not yet bound to the normal PyTorch `train-live` full-model engine**. v55 still constructs a full shadow model and optimizer on a certified switch. The router does NOT itself offload each live layer, ensure optimizer moment crash consistency, merge arbitrary signed LoHA steps into shards, or verify 300B network throughput. The final local writer fence may introduce a short step-boundary pause; there is no mathematically strict zero-latency cutover.
- No automatic garbage collection of old shards is performed; GC must wait for verified retirement and all outstanding leases. `ShardedEpochRCU` currently tracks in-process readers only, not cross-process durable leases.

## Risk, scale and correctness

1. Trust model: pinned CFT committee (strict majority), integrity of local state/keys, signed transport, SHA-256 content integrity and a **trusted local exporter receipt**. Not proof against a Byzantine voter who intentionally signs invalid weights.
2. The v1 trainer still limits inline `cutoffs` to 256 origins. Compact v2 certificates are transmitted separately and **cannot yet activate the v1 trainer**. Using v2 proofs as a substitute for v1 in the current trainer would be unsafe; integration is blocked until a complete local v2-to-training cut validator and signed optimizer snapshot exporter exist.
3. Staged checkpoint verification can take O(model bytes) disk I/O, even though atomically swapping the verified *pointer* is O(1). Local committee votes currently perform synchronous verification and must be moved to an isolated disk/verification executor for 300B production.
4. Need storage for verifiable checkpoints and pending signed data. A disk budget cannot magically guarantee availability if all full copies are lost. WAN network partitions, adversarial origin equivocation, 4096 real peers, and 300B GPU execution are not benchmarked here.
5. The full v55 test suite was attempted but exceeded the execution budget. The targeted epoch, generation, compact, frontier and shard suite passed (139 tests). See `VALIDATION_V56.md`.

The new modules are intentionally safe-to-integrate control-plane components, not unverified replacements for the existing model training loop. Do not enable v2 quorum voting by replacing `local_export_verified` with a constant True in production.
