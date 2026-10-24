# Fedpara — cold join, missing events and stalled DataChannel repair

This release fixes independent problems that caused two desktop windows to show
`active=1 living=1` and caused the second peer to optimize from an empty
`genesis.fstate` instead of the state already trained by the room.

## What changes

* **No silent offline start.** A brand-new, empty-journal participant cannot
  start optimizing until Mummy responds to `group-join` with `group-members`.
  Old Mummy servers that do not implement the group protocol now yield
  repeated `JOIN WAIT: waiting-for-room-signaling` diagnostics instead of a
  deceptive loss value.
* **New participant catch-up barrier.** If Mummy reports another member,
  the new peer waits until signed, SHA-256-verified journal inventory/patches
  from its direct connected neighbors have been applied. It then prints
  `JOIN READY: initial-inventory-verified` and begins training. The existing
  peer does not stop optimizing. The `--sync-timeout` alone no longer allows
  a cold joiner to train on stale genesis.
* **Lost journal requests.** Missing-event requests now retry after SCTP
  backpressure, rather than being marked pending forever with no request
  actually reaching the other peer. Each event has an origin/sequence and
  signature, and remains idempotent on retransmission.
* **Stalled ICE/DataChannel reconnect.** A permanently open signaling socket
  with a dead/unopened DataChannel is reset after a bounded interval, with
  increasing retry backoff.
* **Emergency mode.** Room membership confirmation is remembered in the
  local SQLite meta table. A *previously established* peer with a durable
  history can keep training if signaling dies, including after restarting.
  New/empty-journal peers cannot assume they are alone simply because the
  signaling server cannot be contacted.
* **Diagnostics.** Logs now distinguish `JOIN WAIT`, `JOIN READY`, `living`
  (server membership including self), `active` (authenticated ready direct
  connections including self), received journal events, and errors returned
  by the actual DataChannel tasks.

## Install and diagnose

Extract this ZIP and replace the old source **without deleting** the hidden
`.fedpara-gui` directory, which contains the room manifests, cryptographic
identities, SQLite WAL history, optimizer snapshots and initial checkpoints.

Start the **updated** Nim server separately:

```sh
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Then check it from a separate terminal:

```sh
python scripts/check_group_presence.py --ws ws://127.0.0.1:8080/ws
```

It uses a fresh random room and asserts two independent WebSocket connections
can join and that the first connection gets the `group-peer-joined` notification.
If it fails, the training GUI cannot discover peers either. Check whether an
older Nim executable is still listening on port 8080, or whether the WS URL,
proxy and/or APP_TOKEN are different. The script accepts `--token`.

Open two GUI processes with `python -m patchlib.cli gui`. Use the **same**
room ID, exact model/revision, same data source manifest, same LoRA/LoHA
kind/rank, and the same genesis tensor content. Watch both logs:

```
room presence: ... living=2
peer registered: p...
connecting p...
JOIN WAIT: waiting-for-initial-P2P-sync ...
JOIN READY: initial-inventory-verified; synced_events=N living=2 active=2 ...
```

`JOIN WAIT: waiting-for-room-signaling` means Mummy membership is not working.
`living=2 active=1` means discovery works, but a working authenticated
DataChannel / event sync is still missing. Look for `peer_errors` and ICE/DTLS
warnings. `JOIN READY` proves the finite journal inventory received from direct
neighbors was exhausted and all currently missing announced events were
applied; it is **not** a guarantee of globally linearizable consensus.

## Important remaining limits

- The code continues using the append-only `.fpatch` journal for synchronization,
  not automatic transfer of the last full `.fstate` from a peer. If history was
  compacted/deleted, the peers cannot converge using the available journal alone.
- A remote peer may have an unsealed in-progress optimizer/LoHA block; that
  unfinished work is not visible until it publishes a signed update event.
  The new peer should be close to the last *published* model, not guaranteed
  to have exactly the same loss immediately (different batches/order).
- `living` includes self and is Mummy membership; it is not a synchronized
  global membership view after a partition. In a room with more than five
  peers, synchronization with the local four neighbors is **not** proof that
  distant peers have no additional updates. Gossip continues afterward.
- Actual `aiortc`+Mummy+macOS interprocess tests could not run in this
  environment. Unit tests verify the guard, retries and WebSocket membership
  using a protocol-compatible Python test server. Nim itself is not compiled
  here; compile and run the provided signaling check on your Mac.

## Test suite

```sh
python -m pytest -q tests/test_cold_join_guard.py tests/test_roomid.py tests/test_join_resume_fix.py
```

All existing files under `.fedpara-gui` must be retained: deleting them **loses
local update history** and invalidates resumability.
