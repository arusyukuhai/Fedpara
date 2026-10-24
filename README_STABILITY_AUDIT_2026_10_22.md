# Fedpara: bounded reconnect / snapshot flow-control stabilization (2026-10-22)

## Scope and evidence

Built **on top of** `fedpara_dual_gui_keepalive_repair_full.zip`, which already contains the 16-peer frontier Gossip fix and adminless snapshot handoff. No journals, local keys or checkpoints are replaced. This is a targeted robustness patch, not a claim that all network/hardware conditions have been proven.

Observed in the user's previous two-window logs: signed HELLO, `JOIN READY` and snapshot receipt succeeded; subsequent signaling `group-peer-left` / rejoin events were followed by closed RTC channels and `living=2 channels=0`. A signaling presence notification is not itself evidence that the actual WebRTC channel has failed. A separate source code review found send-window loops that wait as long as `readyState == 'open'`, regardless of whether SCTP's buffered bytes have drained.

## Changes

1. **Roster membership hysteresis** (`patchlib/elasticnet.py`): retain a disappeared member for 20s while the signaling roster is online; preserve last known roster during a complete signaling outage. New members appear immediately. Apply the SAME stable roster to the topology so the 8-neighbor ceiling is maintained. This avoids destroying a still-authenticated transport on a one-second `group-peer-left`/`group-peer-joined` oscillation. Actual `DataChannel` close/failure handling is unchanged. An authentically departed peer is removed after 20s of online absence.
2. **Finite SCTP send stalls** (`patchlib/shardp2p.py`, `patchlib/p2p.py`): fail with a `TimeoutError` after 45s without outgoing buffer progress, even if SCTP reports open. Progress resets the deadline. The snapshot link surfaces send failure to the existing recovery system, which reconnects and reuses content-addressed shards. Previously these send-buffer wait loops could persist indefinitely.
3. **Self-amplifying room-discovery reconnections** (`patchlib/roomid.py`): short-lived successful joins no longer reset exponential backoff to 1s; only a healthy connection lasting at least 60s resets it. Each identity gets a small, deterministic jitter to avoid simultaneous reconnects. Wrong room manifest or permanently rejected authentication now surfaces as a fatal error instead of an endless `JOIN WAIT`. The server's generic `Group full, identity already online, or server busy` response remains retryable.
4. **Do not throw away signed snapshot boundaries** (`patchlib/elasticnet.py`): a saturated application control queue previously made `snapshot-tip` fatal, terminating a healthy RTC link. The latest signed snapshot tip is now retained separately if that queue is full; it is processed before the ordinary queue when training resumes. Normal unsaturated control delivery stays FIFO-compatible.
5. **Async file checksum** (`patchlib/p2p.py`): hashing a multi-GB outgoing `.fstate` is offloaded from the event loop, preserving signaling timers while the archive is checked.

**Unchanged**: immutable room manifest, Ed25519 hello authentication, journal/SQLite durability, content-addressed snapshot verification, frontier NEED retransmission, no-administrator topology, existing numerical training/evaluation semantics.

## Installation

Stop the trainer in each GUI first. Back up the repository and unzip the `fedpara/` tree over the checkout **or** apply the `.patch` against the *immediately previous* keepalive repair version. **Never remove `.fedpara-gui`, `events.sqlite3`, `.identity.pem`, `*.fstate` or snapshot object stores**. This revision does **not** modify `src/main.nim` and does not require recompilation beyond what the previous revision already required.

For the two-window WebView GUI:

```bash
python -m patchlib.cli gui
```

Open a second terminal and launch the same command again, joining the same room. The identities/journals MUST be different between the two training children. Watch **both** trainer logs for `SIGNED HELLO OK`, `SNAPSHOT READY` and `JOIN READY` for a new member; once established, each should report `living=2 active=2 channels=1 authed=1`. During a signaling hiccup, `living` and `active` can differ momentarily. They should eventually converge without an RTC reconnection storm. Verify signed `elastic local event` continues on both sides.

Then exit the original window; the survivor should continue and serve a fresh joiner without a fixed administrator. A 20s topology debounce following the departure is expected. To verify robust snapshot recovery, test a mid-transfer disconnect, then rejoin and verify the signed state boundary / journal suffix.

## Verification and limitations

`python -m pytest -q`: new failure-injection tests for (a) flapping memberships, (b) max-degree topology, (c) silent SCTP send buffer stalls, (d) saturated controls and signed snapshot-tip preservation, (e) invalid-vs-transient signaling rejections. Full CPU test suite is executed separately. Runtime **aiortc**, Nim server, Hugging Face downloads and Apple MPS are not available in this Linux container; those two-window end-to-end scenarios remain an acceptance gate on macOS, not proven outcomes.

This intentionally avoids changing model-update atomicity, SQLite threading semantics or consensus policy. If a CPU/HF worker blocks permanently, `asyncio.to_thread` cannot safely kill it; preserving model state takes precedence over aggressively restarting the thread. A complete worker-owned training/data execution subsystem with cooperative cancellation would be a larger separate redesign.
