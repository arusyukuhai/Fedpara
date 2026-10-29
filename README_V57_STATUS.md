# Kagami v57 — experimental compact-v2 trainer integration

Date: 2026-10-27. **This is a research integration, not a production-proven 4,096-device / 300B / strictly zero-pause training solution.** The existing v1 path remains the default.

## New

1. `patchlib/epoch_export_v57.py` creates a *signed, content-addressed local trainer checkpoint* from an actual optimizer-owner step boundary. The trainer seals its unpublished local update, takes `inbound_commit_lock` against concurrent signed-update model mutation, then streams effective model weights into sharded objects on a worker thread. It verifies the *entire contiguous indexed journal prefix* before and after export and revalidates the manifest/shard digests. The pinned exporter signs the manifest, 4,096-origin Merkle root and exact event-inventory digest. The lock is maintained throughout the export: **local training is paused during this potentially long operation**; P2P event-loop progress continues. Optimizer *moments* are NOT part of the certified model-only sharded snapshot; a new low-rank optimizer is explicitly reset at generation activation, as already done by v52/v53.
2. `patchlib/epoch_autonomous_v56.py`: persists and validates the exporter's signed receipt; v2 candidate pages remain retained after certification, and post-certification reconnects can request missing historical Merkle pages. Added bounded cursors to avoid scanning every old completed frontier on every gossip tick. Heavy committee voter verification is deferred to a dedicated worker in the actual trainer; it opens its own SQLite connection and verifies the entire local snapshot and exact journal prefix before persisting its Ed25519 vote. The older standalone test behavior remains supported.
3. `patchlib/epoch_compact_trainer_v57.py`: reconstructs the **full 4,096-origin cutoff** only from a locally verified compact-v2 certificate PLUS every authentic Merkle page. The resulting local-only projection includes `cutoffs` and `cut` consumed by the existing v52 signed-carry/rebase and v53 optimizer-boundary switch. It is not a new signed certificate: its trust comes *solely* from checking the original v2 quorum certificate and complete Merkle frontier. No downgrade to a forged v1 certificate.
4. `patchlib/elasticnet.py`: opt-in compact boot, certificate-driven asynchronous prewarm, signed carry and guarded trainer handoff, plus pinned leader checkpoint proposals. It will not vote for an epoch other than the next epoch of the currently running model/journal. A compact cut missing an intermediate certificate or one page **does not authorize activation**.
5. `patchlib/live_generation_transport.py` + elasticnet signed-control handlers: bounded `generation-compact-have` / `generation-compact-snapshot-request` control messages. A certified manifest and complete hashes are checked before the existing resumable shard DataChannel serves it. No bulk checkpoint bytes are added to the small gossip-control messages.

## Configuration (experimental)

All members use `--generation-v1-committee ./committee.json` and `--generation-v2-trainer`. To let the lexicographically first pinned committee member automatically export a candidate every N locally emitted signed events, *on that committee member only*, additionally pass `--generation-v2-propose-every N` (N > 0). This **is not leader failover**. To avoid a global duplicate candidate or unsupported coordinator changes, v57 does not automatically appoint a replacement proposer when that member is offline. Checkpoint export may require substantially increasing `--max-state-gb` and available local disk.

Use a NEW journal/run configuration, or explicitly migrate an existing v1 committed epoch using a separately audited procedure; **do not toggle the v2 trainer flag on top of an existing v1 active-epoch pointer**. No guarantee of backward-compatible automatic pointer conversion is made.

## Outstanding blockers / limitations

- A signed receipt attests the honest pinned exporter ran the fenced capture; it is **not Byzantine proof** that model computation is the unique consequence of the event history. The committee's majority is crash-fault tolerant only, not Byzantine-fault tolerant.
- The current automatic proposer uses a **fixed designated committee peer**. Robust leader failover, competing-proposal resolution, session/view changes and adversarial quorum behavior remain unimplemented.
- The proof that the sharded snapshot is the exact result of the signed event history ultimately rests on the correct model/journal locking and trusted exporter implementation. Full system/integration fault injection is still needed.
- The v52/v53 trainer still reloads a full in-memory shadow PyTorch model and resets the optimizer when changing epochs. The v56 disk-shard RCU primitive does not replace that training engine. **No 300B end-to-end training or strict zero-pause model/optimizer export has been implemented or tested**.
- The snapshot-copy path reuses the existing serialized sharded model transport. WAN throughput, 300B storage, multiple simultaneous multi-TB checkpoint transfers, and actual 4,096 WAN devices are untested.
- Old signed updates must remain available if peers can return after indefinitely long disconnects. No bounded-disk exact-recovery guarantee exists without an explicit retention policy or replication contract. Certified frontier pages are deliberately not garbage-collected until such a contract exists.
- Full CFT certificates alone do not mean all origins have contributed; a frontier certifies only the actual contiguous cut. A permanently missing signed update cannot be recreated by quorum votes.
- No automatic v1↔v2 consensus upgrade on an already committed v1 trainer pointer. Keep feature flag choices consistent across restarts.

## Tests

`tests/test_epoch_v57_trainer.py` adds checks for 4,096-origin local projection, incomplete-page fail-closed behavior, invalid Merkle pages, signed exporter receipt tampering, journal sequence holes, v2 step-boundary adapter activation and compact snapshot request controls.

See `VALIDATION_V57.md` for executed tests and `CHANGES_V57.patch` for diff against v56.
