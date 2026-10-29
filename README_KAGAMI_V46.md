# Kagami v46 — authenticated epoch handoff transport / carry patch

**Status: PARTIAL / opt-in infrastructure. NOT autonomous zero-downtime rollover.**
Do not use this release as authorization to delete any v43–v45 journal.

## Changed code

- `patchlib/live_generation_transport.py`: fixed gossip fanout bug. Earlier, `AutomaticGenerationControl.drain()` was processed independently per neighbor, consuming the shared outbox for the first peer. The patch stages a bounded broadcast set once per tick, forwards to every authenticated neighbor, retries failed sends and suppresses stale committed proposals. Optional compact-v2 certificate heads, signed certificate recovery and range requests use existing authenticated `generation-wire` DataChannel controls.
- `patchlib/elasticnet.py`: added the two compact-v2 control types to authenticated inbound generation message routing. **The old main training loop still does not stage/switch epochs automatically.**
- `patchlib/epoch_certificate_v46.py`: independent compact-v2 quorum ledger. Votes commit to a complete Merkle-paged 4096+ origin frontier, a SHA256 journal-inventory digest and shard-manifest/semantic hashes. Local voting re-verifies all received pages, full per-origin contiguous sequence coverage and every checkpoint shard. Committee signatures, parent chaining and durable single-vote semantics are checked. **Snapshot-to-optimizer-cut equivalence still needs a trusted atomic training export receipt before this is production-safe.**
- `patchlib/epoch_carry_v46.py`: persistent, authenticated rebase of old-generation signed LoHa FedSum events onto a new certified base. Retains ORIGINAL signed bytes, uses the existing fsync-before-index event archive, prevents double application and supports clean-base replay on crash/restart. Archive identity binds source/target genesis, epoch, certified manifest and cutoffs. This archive is a required input for later checkpointing; it does NOT by itself prove that all disconnected peers have received the carries.
- `patchlib/continual.py`: optional `authenticated_source_genesis` check so the unchanged signed old update can be converted into its effective-weight delta and applied to the new base; the normal new-gen event path still enforces its native genesis.
- `patchlib/async_ingest.py`: optional preverified carry lane with threaded signature/delta preparation, fsync-before-apply, duplicate suppression, and no blocking GPU apply on the WebRTC asyncio event loop. Previous update events covered by the checkpoint cut are ignored rather than applied twice. No carry support is activated unless an upstream caller supplies a replayed archive.

## What STILL MUST be built before true unattended rollover

1. Produce a consistent **atomic snapshot + exact authenticated event-cut receipt** from the active worker, disseminate the complete frontier proofs, and automatically obtain quorum signatures. Existing `elasticnet.py` still disables production votes (callbacks return False), intentionally.
2. Stage a second model/adapter/optimizer/journal off-loop, atomically change the actual worker's model and journal pointers at an optimizer safe point, persist its restart configuration, and update all live WebRTC/anti-entropy session routing. `TrainerStepRouter` exists but is **not attached to the old training loop**.
3. Cross-epoch *carry discovery and retransmission*, including returning peers who skipped multiple epochs. The new archive accepts old signed events, but old anti-entropy does not currently advertise archive inventories for all generations. Therefore peer convergence is **not proven**.
4. Replicated checkpoint availability and retention rules. Do not retire an old journal until a newer checkpoint covers its carries or safely replicated peers durably serve them; no fully automatic destructive GC is enabled in this release.
5. Real multi-node / GPU / packet-loss / process-crash integration testing and adaptive generation scheduling. A brief optimizer-step safe-point fence is unavoidable for atomic pointer publication; peer communication can remain live.

## Safety / compatibility

- No old data directories, signed events, checkpoints or identity keys are deleted or migrated implicitly.
- Existing v45 running models continue to use the same wire/optimizer state unless the new opt-in APIs are explicitly exercised.
- The compact-v2 ledger is separate from legacy v1 ledgers, so original certificates cannot be interpreted as compact approvals.
- The experimental carry archive must be replayed onto a freshly loaded, independently verified snapshot **before** using its `AsyncInboundPipeline` hook. Do not replay on top of a working model that has already incorporated the deltas.
- Model code distributed to peers is still not automatically executed; only trusted pinned keys can authorize epoch votes.

## Validation

```
PYTHONPATH=. python -m pytest tests/test_epoch_ingest_v46.py tests/test_epoch_carry_v46.py tests/test_compact_epoch_network_v46.py tests/test_compact_epoch_certificate_v46.py tests/test_generation_multicast_v46.py tests/test_hot_swap_v45.py tests/test_live_generation_transport_v42.py tests/test_v44_low_cost_generation.py tests/test_epoch_runtime_v43.py tests/test_online_generation_v41.py tests/test_continual_elastic.py tests/test_continual_fedsum_v3.py tests/test_generation_committee_v43.py tests/test_snapshot_reconnect_regression.py -q
```

65 targeted tests passed at packaging time. No 1000-peer real RTC test or live 300B-model test was run.
