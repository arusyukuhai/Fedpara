# Kagami v41 — Online generation control plane (experimental, fail-closed)

This release contains a tested **generation-consensus and checkpoint-recovery foundation**, **not** a completed automatic live epoch-switching trainer. The older v40 training engine and its v1 gossip history are intentionally unchanged; no automated destructive migration is enabled.

## What is implemented

- `patchlib/online_generation.py`: durable crash-fault fixed-committee quorum approvals (Ed25519), strictly increasing checkpoint parent hashes, one vote per `(peer, epoch)` persisted in SQLite; signed quorum certificates; ordinary nonvoting members; bounded epidemic message cache. Up to 256 cut-off owners per direct signed proposal (bounded by wire size), up to 31 approved committee IDs. Larger vector clocks require a separate paged/Merkle availability implementation, which is not yet present. These are schema limits, not tested network throughput.
- `OfflineSnapshotRecovery`: verifies a chain of certificates and obtains the latest approved checkpoint from a verified replica; checks manifest SHA-256, every shard, and semantic tensor digest. Refuses restoration if there are unresolved local updates, and refuses to proceed if the runtime does not explicitly confirm loading the weights.
- `mark_local_generation_loaded` and `retire_adopted_offline_journal`: proof-of-local-installation marker, old-journal lock check, certified-cut inventory hash check and no-remaining-update check before local previous-generation GC. No deletion of an active v1 journal, even if a certificate exists.
- `tests/test_online_generation_v41.py`: positive/negative tests including 5 voting replicas, missing/bad signatures, rebooted voter double-voting, parent-chain validation, fallback snapshot replicas, offline peer recovery, local GC with and without uncommitted events, ordinary participants. The `test_1000_offline_members_need_only_latest_cert` test is a lightweight protocol/state-advertisement simulation and **not** a 1000-peer network load test.

## Known missing functionality (do not claim these features are production-ready)

1. The current `patchlib/elasticnet.py` v1 optimization and WebRTC gossip loops do **not** automatically instantiate/use `GenerationLedger`, move to an agreed base, or broadcast the new control messages. The module provides the core control-plane API but is not yet connected to an automatic room-level epoch transition.
2. The current v1 snapshot receiver requires matching full event history. It must be adapted to install a verified certified generation without replaying its pruned prefix, and restart from the correct new-gen checkpoint and journal. The wire handshake also rejects mismatched genesis; a certified redirection mechanism is still needed for old peers.
3. The old optimizer needs a consistent cut, rebase for updates produced during transfer, and deduplication of events across epochs. Simply accepting a majority certificate and resetting its additive weights would lose or double-apply updates. This is not implemented.
4. Old generation GC must be triggered by a **successful** local new-gen activation, not by a remote certificate alone; unattended automatic GC is intentionally disabled.
5. No real P2P tests under network partitions, crashes, or 1,000–4,000 concurrently connected peers have been run. A fixed committee assumes crash failures (not a Byzantine adversary); dynamic committee reconfiguration needs its own consensus.

## Security invariant

A certificate attests that a quorum of pinned committee identities signed one *specific* checkpoint digest. It **does not** prove that all disconnected, still-unpublished local gradients were included. Offline peers with such updates are flagged `quarantine-or-rebase-uncommitted-updates` and cannot delete the old journal until those updates are explicitly recovered or safely abandoned by policy.

## Verification

```bash
PYTHONPATH=. pytest -q tests/test_online_generation_v41.py tests/test_generation_rollover_v40.py tests/test_durable_io_v12.py tests/test_audit_snapshot_liveness.py tests/test_snapshot_boundary_paging_v26.py
```

This subset passed 38/38 in the build environment. Existing online v1 trainer behavior remains as in v40; there is no dangerous default auto-compaction.
