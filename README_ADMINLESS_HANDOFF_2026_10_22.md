# Fedpara: leaderless room handoff + two-GUI fix (2026-10-22)

## Resulting protocol

**No permanent room administrator or leader is elected.** The first genuinely
joined peer only establishes the initial genesis state, and persists its own
`initial_ready` in its independent SQLite journal. After that, *every peer*
that has successfully installed a verified full-model snapshot and persisted
its own `initial_ready` can offer a full snapshot to other newcomers. The
signaling server stores immutable room metadata and peer presence, not model
weights or an administrator identity. All peers share signed update gossip.

When the initial peer leaves after B has verified state, B continues training,
serves snapshots, and signs gossip just like A did. The overlay automatically
recomputes neighbor links after membership removal; it stays connected with
up to eight direct neighbors per peer in a 16-peer room.

If a selected state provider disconnects, its SCTP state channel stalls, or
it explicitly reports `source-not-ready`, the receiver places it on a bounded
cooldown and automatically selects another authenticated verified holder.
Successful selection preserves a single active source (no unbounded parallel
multi-GB transfers). When a peer becomes eligible to serve state, it advertises
that transition immediately through a signed control heartbeat.

**Fail-closed edge case:** if the first peer disappears before *any* other
peer obtains a verified full-model snapshot and durable event history, there
is no safe state source. A fresh GUI must wait for a peer with verified history
to return, or perform an explicit administrator-approved *new room* reset;
this patch deliberately does **not** silently train from stale genesis. A
network partition can likewise lead to independently progressing replicas;
this change does not implement globally consistent consensus of model updates.

## Fixes also bundled for two simultaneous GUI windows

- `group-describe` reads a manifest **without** registering a fake training
  member or consuming the unique empty-roster bootstrap transition. Restart
  the **Nim signaling server** from this archive when installing.
- A room manifest is immutable after first save; concurrent saves are
  cross-process serialized and mismatched settings rejected.
- Each GUI allocates a distinct, lock-protected persistent peer profile,
  identity, and journal. Both share one immutable genesis file; checkpoint
  creation is serialized with a kernel lock.

## Install on macOS

1. Stop GUI training and the old signaling server. Back up your project and
   persistent room data, then copy the archive's `fedpara/` sources over the
   corresponding sources in your checkout.
2. **Do not delete** `.fedpara-gui`, existing `.fstate`, `.identity.pem`, SQLite
   journals, sharded snapshots, or your saved room manifests.
3. Recompile/restart the updated Nim signaling server:

   ```sh
   nim c -d:release --threads:on --mm:orc -r src/main.nim
   ```

4. Open two terminal windows in the SAME project directory and run in both:

   ```sh
   python -m patchlib.cli gui
   ```

5. Save/register a NEW room in the first GUI, then use **Load room** in the
   second (or use a room that already has a verified trainer online). Each
   GUI must show a different profile/journal. When the second reaches
   `SNAPSHOT READY` and `JOIN READY`, quit the first GUI. The second must
   continue generating signed updates, and a third newcomer must reach
   `SNAPSHOT READY` via the second peer.

Expected log marker during automatic provider selection:
`SNAPSHOT SOURCE HANDOFF: none -> p...` (or `p... -> p...` when the former
source becomes unavailable).

A room whose previously joined peers all disappeared *before* any verified
peer source remains cannot be automatically repaired by this code. If you see
`waiting-for-original-model-source`, do not delete existing persistent data;
bring back a prior verified identity/journal or select an explicitly new room.

## Verification

`python -m pytest -q`: **225 passed, 3 skipped** (CPU test environment).
The new tests verify no permanent founder authority, creator departure,
source cooldown/rotation, refusal to promote an unverified peer, and
16-member bounded overlay connectivity after any one departure. Tests for
read-only lookup and simultaneous GUI profile/genesis creation are included.

**Not executed here:** macOS dual native WebViews + real `aiortc` 16-process
end-to-end experiment; Nim compiler not installed. This is a tested source
fix, not a claim of end-to-end live WebRTC validation.
