# Kagami v51 — migration transport isolation and durable bounded checkpoint lifecycle

**Status: substantial migration infrastructure, but NOT a complete unattended production `train-live` rollover.**

This revision builds on v50. It does NOT disable the original genesis-bound
training-engine protections, enable unverified committee votes, falsely
reinterpret old LoHA factors as new-epoch deltas, or delete live journals.

## Connected changes

1. **Authenticated mixed-generation certificate transport.** `ElasticNode`
   accepts a *different* signed `genesis` only when both sides explicitly
   negotiate `generation_v1` and name the same immutable, caller-configured
   `generation_root`. In that state, the DataChannel is *bridge-only*:
   signed generation heads/needs/certificates and heartbeats work, but normal
   FedSum frames, frontier repair, snapshot-tip, and ordinary `need` do not.
   A remote party cannot switch its own genesis mid-connection. No legacy
   peer or unrelated model can bypass strict genesis checking. The production
   `train-live` connection factory now supplies its room genesis as
   `generation_root` when `--generation-v1-committee` is configured. A *new*
   epoch worker would need to preserve that original root explicitly.
2. **Automatic durable staging leases.** A v50 `BoundedCheckpointStager`
   now automatically obtains an SQLite-backed `DurableEpochLifecycle` when
   passed to `LiveEpochSwitchService`. Metadata is fsynced and capped;
   candidate download is resumable. After a new certificate supersedes a
   failed stage, orphaned checkpoint views and pins are actually released.
   A pointer referring to an uncertain published checkpoint is NEVER pruned.
3. **Crash-window repair.** A published hot pointer can be reconciled with a
   lease marked `preparing` or `staged` after a crash. A v50 checkpoint without
   an existing lease can be verified through the bounded stager and adopted
   on restart. The old trainer cannot be silently started over a newer pointer.
4. **Bounded retention of formerly active checkpoints.** By default a node
   retains no more than two old verified checkpoint views plus the active one
   and two candidate staging records. If retirement evidence does not arrive,
   future migrations are rejected before old recovery state is destroyed.
   This is deliberate *backpressure*, not an infinite-disk promise.
5. **Automated proof-gated old-checkpoint cleanup.** Supply an
   `gc_runtime_factory` which opens a NEW `EpochRuntimeV43` and its ledger in
   the worker thread. The live-switch watcher attempts GC at most every
   `gc_interval` seconds (default 60). The runtime independently checks the
   signed certificate, exact checkpoint shards, new journal, certified cut,
   carry coverage, and exclusive old trainer lock before retiring an old log.
   Only then does this layer unlink the obsolete checkpoint view/unpin its
   immutable objects. An old cache view is never released on timer, remote
   report, quorum alone, or merely after pointer advance.
6. **No WebRTC-thread fsync for large checkpoint I/O.** The existing v50
   single-thread disk worker, resumable transfer, full SHA-256 checks,
   immutable hardlink reuse, token bucket and ≤8 neighbors remain in use;
   the new runtime GC proof is also executed off the asyncio relay thread.

## Resource budgets and limits

- The *direct connection graph* for 4,096 peers is not complete: the v49
  four-ring overlay uses at most eight neighbors per peer; the tested graph
  had 16,375 undirected connections instead of 8,386,560 for a full mesh.
  These are graph metrics only.
- Checkpoint download is bounded locally by `BoundedCheckpointStager`:
  4 MiB/s requested checkpoint payload, 8 GiB unique object reservation and
  at most eight candidate sources by default. These are **per-device**
  defaults, not measured network-wide bandwidth. Control traffic, old gossip,
  optimizer memory, and journal/carry archives have separate budgets.
- Local lease metadata is capped at 4 MiB of SQLite database pages by default.
  Retired rows are removed, not stored forever. Filesystem overhead, temporary
  files and SQLite rollback journals need additional headroom.
- `max_held=2`, `max_candidates=2` provide fail-closed long-term checkpoint
  retention. Without verifiable coverage of old signed updates, a finite quota
  cannot both retain arbitrary offline work *and* always progress.
- A small-world overlay prevents a direct all-to-all mesh; it does **not**
  eliminate the inherent cost of replicating every independently signed event
  to all peers. Complete replacement with certified aggregates and bounded
  offline local-update semantics remains future work.

## Usage of the connected switch service

```python
stager = BoundedCheckpointStager(
    root, peers=authenticated_neighbors,
    fetch_manifest=fetch_verified_manifest,
    fetch_block=fetch_authenticated_bounded_block,
)
service = LiveEpochSwitchService(
    ledger=ledger, router=trainer_router, root=runtime_path,
    checkpoint_stager=stager,
    build_shadow=build_independent_model_optimizer_and_journal,
    verify_local=verify_actual_checkpoint_and_journal,
    flush_unpublished=flush_signed_local_updates,
    carry_suffix=archive_and_rebase_signed_suffix,
    transport_handoff=activate_generation_aware_ingress_egress,
    attest_journal=verify_durable_new_journal,
    gc_runtime_factory=open_new_epoch_runtime_in_disk_thread,
)
await service.run(shutdown)
await service.close()
await stager.close()
```

These locally supplied callbacks are **not optional no-ops** in production.
They must have the real mathematical/durable properties indicated by their
names. The example is integration API, not a fully wired GUI feature.

## Still incomplete (do not enable autonomous destructive GC on v1)

- Actual GUI/`train-live` optimization loop is still pinned to its original
  adapter, optimizer, journal and original genesis; this revision does **not**
  automatically instantiate/install the next model generation in that loop.
- There is no proven atomic model+frontier producer that automatically submits
  and wins new quorum proposals. `locally_verified`/`locally_covered` in the
  existing control-plane bridge deliberately remain `False`.
- The bridge-only cross-generation connection passes **certificate traffic**,
  not authenticated old-generation training-event carries. A separate
  generation-tagged bounded update stream and original signature validation
  still need to be connected end to end.
- Correct replay of an offline peer's unincorporated steps, canonical
  effective-weight carry accounting, and durable reconstruction of the actual
  production optimizer across a power loss remain application obligations.
- No thousand-GPU/phone WebRTC training rollout or multiweek soak has been run.
  The topology tests, test-double migrations and a local real-torch/SQLite
  handoff are not a substitute for that deployment.

**Safety policy:** on missing proof, stop *migrating or collecting*, never
silently discard uncommitted events. Training can continue in the existing
admitted generation until its separate journal quota is reached.

See [VALIDATION_V51.md](VALIDATION_V51.md) for reproducible test coverage.
