# Fedpara: first-member bootstrap / cold-join deadlock fix

## Symptom and root cause

A room's first member starts training alone, then a second member joins.
The second member can begin operating while the first repeatedly displays
`waiting-for-verified-full-model-snapshot`.

Root cause: the runtime captured `fresh_join = not persisted_ready` at boot and
never cleared it when the first participant successfully initialized an empty
room. When the second member appeared, that *founder* incorrectly entered the
newcomer-only snapshot wait guard. The founder's persisted `initial_ready` flag
was set but the in-memory gate was stale until process restart.

A second race: the signaling server can send an empty `group-members` roster,
then immediately notify the first peer of another arrival **before** the
training loop examines room membership. The founder must remember the *first*
authoritative empty roster. Current membership alone is insufficient.

## Fix

- `patchlib/join_guard.py`: explicit `BootstrapState` with one-way promotion.
- `patchlib/roomid.py`: remembers whether the initial server roster was empty,
  independently of later joins/leaves/reconnects.
- `patchlib/elasticnet.py`: the founder persistently becomes an established peer
  before joining the optimizer loop and NEVER waits to download the joiner's
  snapshot. A new peer cannot train before a verified snapshot.
- `patchlib/elasticnet.py`: if the initial roster already had a peer but that
  peer goes away, the newcomer MUST NOT silently initialize from genesis.
- `patchlib/elasticnet.py`: more detailed snapshot progress diagnostics in
  `JOIN WAIT` (channel / signed tip / manifest / missing shards / error).
- `tests/test_cold_join_guard.py`: regression tests for each order of arrival.

## Install

Replace the Python project files from this archive with the same paths in your
existing Fedpara tree. Stop running training processes first. Do not delete
`.fedpara-gui`, SQLite journals, locally generated identities, or checkpoints.

No Nim/Mummy source change is required for this particular fix.

## Quick check

1. Start GUI A on a **fresh** room. Wait for the new log message
   `ROOM GENESIS READY:` followed by `JOIN READY` and training steps.
2. Start GUI B in the SAME room. A must continue training and should emit
   `SNAPSHOT OFFER` in response to B's `SNAPSHOT REQUEST`.
3. B must stay paused until it emits `SNAPSHOT READY` and `JOIN READY`.
4. If B stays waiting, inspect the `snapshot_progress=` field in `JOIN WAIT`
   for its peer. It distinguishes missing state channel, signed tip, manifest,
   shard data, and verification errors.
5. Close B. A should continue. Reopen B and check restart history.

If B has never received a snapshot and no peers with the room's historical
weights remain online, the system deliberately refuses to initialize B from
an obsolete genesis checkpoint. No safe network partition convergence is
claimed without a surviving authoritative model source.

## Test status

`python -m pytest -q`: 116 passed, 3 skipped in the implementation environment.
`aiortc` and Nim were not installed there; a macOS two-window WebRTC test is
still necessary. This is a state machine fix backed by regression tests, not a
claim that all possible network transport problems have been eliminated.
