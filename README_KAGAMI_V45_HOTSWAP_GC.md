# Kagami v45 — bounded-cost hot-swap primitives and safe-GC wiring

**Status: incremental / experimental. Not a finished autonomous, thousand-peer, zero-downtime generation protocol.**

This revision preserves v44 data, network protocol and user-facing behavior. The prior releases explicitly did not have a live worker handoff. Rather than activate unsafe history deletion, v45 adds executable, tested components that can be reused when the multi-generation WebRTC protocol is deployed.

## Changes

- `patchlib/epoch_hot_swap_v45.py` introduces `TrainerStepRouter`. An active learner can obtain an immutable `TrainerGeneration` for one forward/backward/optimizer step. A candidate model/adapter/fresh optimizer/new journal can be built separately. The router pauses admission of **new local steps** just long enough to await in-flight steps, seal unfinished local work, run an authenticated carry callback, durably attest the new journal and checkpoint, atomically persist the new pointer, then continue local steps on the new model and optimizer. The network relay is external to the step gate and is not closed by the router. In an existing worker, each step and `optimizer.step()` MUST use the router; v44's training loop does not yet do so.
- Checks reject missing verification, aliased old optimizer/model/journal, conflicting or non-monotonic epochs, concurrent switches, and revalidation failures. Old history is **not** automatically destroyed on a pointer swap.
- `rebase_authenticated_additive_suffix` validates old *signed* continual FedSum events with the existing engine's `_check`, reconstructs **effective-weight deltas** (`f(after)-f(before)`) instead of misapplying raw LoHa factor differences, and adds them onto a clean candidate base via `_apply_checked`. It requires original event bytes to be present in a durable archive. A fresh optimizer is required, and the original signed archive must be replayed on restart. This is local carry functionality, **not** cross-genesis Gossip dissemination.
- `checked_paged_cut`: verifies a 4,096+ origin frontier only after receiving and verifying **all** Merkle pages. `stream_cut_summary`: scans journal indexes without materializing every event in RAM; O(origins) memory. Page verification does **not** replace cryptographic binding between checkpoint bytes and exact event coverage.
- `verify_certified_checkpoint` re-hashes the checkpoint manifest, checks its semantic digest and verifies all tensor shards against the locally accepted quorum certificate.
- `elasticnet.train_elastic`: if authenticated generation control is enabled, an existing *already installed* local transition is polled for safe GC about every 30 seconds. Expensive verification/deletion opens its own SQLite handles on an off-loop worker thread; idle normal training does no full journal scans. Original `EpochRuntimeV43` checks the old writer flock, complete cut, installed new journal, and absence of post-cut events. Any uncertainty **blocks deletion**. Old v1 trainers are never garbage-collected merely because a certificate was gossiped.

## Tests

`python -m pytest tests/test_hot_swap_v45.py tests/test_v44_low_cost_generation.py tests/test_epoch_runtime_v43.py tests/test_generation_committee_v43.py tests/test_live_generation_transport_v42.py tests/test_online_generation_v41.py tests/test_continual_elastic.py tests/test_continual_fedsum_v3.py tests/test_generation_rollover_v40.py tests/test_unlimited_durable_gui_v29.py -q`

69 targeted tests passed in the build environment. The new tests include a real Torch AdamW step across in-process generation change, 4096-origin complete-page verification, additively correct LoHa effective-weight rebase, race rejection and journal suffix scanning. **No real GPU, 1000-peer WebRTC network, model-serving failover, or chaotic network partition test was performed.**

## Still missing for the requested complete behavior

1. A v2 certificate format whose *signed* proposal commits to the Merkle frontier (v1 retains a maximum of 256 inline cutoffs). The committee must verify the complete paged frontier **and** the snapshot's actual immutable cut receipt before voting; existing production callbacks deliberately decline to vote.
2. Per-epoch WebRTC session routing/re-handshake. Existing `elasticnet` links are pinned to one genesis. The local trainer must consume `TrainerStepRouter` as its step gate, and inbound queues, anti-entropy cursors, key identity and local journal pointers must be migrated together. The router must be staged by a valid quorum and a verified checkpoint, not a local untrusted file.
3. A cross-epoch authenticated wire format for additive carries. v1 event signatures bind the *old* genesis: original signed updates cannot just be relabeled as new-genesis events. Offline peers need this format, deduplication and a newly certified snapshot they can discover. Until then GC must refuse uncovered suffixes.
4. A generation-aware optimizer and journal restart loader, plus real on-disk / remote recovery tests with crashes between every handoff phase. The new optimizer can be safely reset, but **transferring AdamW moments without a mathematically defined transformation is not correct**.
5. True continuous training without even a short safe-point admission barrier is not attainable with this implementation. The model is prepared while training continues, but the final publication needs a brief, explicit optimizer boundary. It is not appropriate to guarantee zero pauses or unbounded throughput across arbitrary devices.

**Do not manually delete old `.sqlite3`, `.fplog`, `.fedpara-gui`, or checkpoint shards.** Back up existing v44/v43 work before using experimental generation APIs. The v45 helper is not enabled as an automatic production hot switch.
