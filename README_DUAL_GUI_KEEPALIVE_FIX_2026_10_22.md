# 2026-10-22 — two-GUI WebRTC keepalive / trainer-loop repair

## What the supplied logs prove

The second GUI has successfully connected its WebRTC channel, verified a
signed HELLO, downloaded a full model snapshot, and reached JOIN READY.
Afterward, `RoomDiscovery` reported an HTTP/WebSocket **client-side keepalive
ping timeout (1011)**, and the peer-side signaling socket received `peer-left`
while its DataChannel was still open. Shortly afterward aiortc reported the
DataChannel and ICE/DTLS closed. A later status showed `living=2 channels=0
active=1`: **two names in signaling presence is not an authenticated direct
transport**. The log does not alone prove whether Hugging Face streaming or
model compute caused the initial starvation, nor that the signaling server
never stalls; it does demonstrate a ping-timeout / link teardown cascade.

## Changes

1. **On localhost**, the signaling WebSocket does not fail its own connection
   because of a delayed Pong (set `ping_interval=None` and `ping_timeout=None`).
   On non-local WSS, native ping is retained with a 30s interval / 150s timeout.
   Both presence and P2P signaling send a lightweight application `{type:ping}`
   every 30 seconds; Mummy already answers with `{type:pong}`. TCP disconnects,
   explicit peer leaves, failed ICE/DTLS, signed heartbeat expiry, and the
   existing bounded reconnect backoff are still handled. This avoids one
   event-loop hiccup being mistaken for a permanent peer departure.
2. **Off-event-loop model operations**: expensive tensor export, delta
   signature validation, factor reconstruction and application, AdamW steps,
   and the safetensors working-state write no longer block asyncio's RTC/WS
   progress. SQLite journal access remains strictly on its creating thread;
   the existing append-before-apply recovery invariant is preserved.
3. A watchdog warns `TRAINER EVENT LOOP STALL <seconds>s` for any remaining
   synchronous code path that blocks communication by more than four seconds.
4. The previously passing demand-driven frontier gossip, snapshot provider
   handoff, 8-neighbor overlay and per-window peer profiles are unchanged.

## Install

1. Back up your working directory, then close **both** GUI training windows.
2. Overlay this archive's `fedpara/` directory onto your existing checkout.
   Do NOT delete `.fedpara-gui/`, peer identity PEMs, journals, snapshots, or
   persistent model files. The signed event history is essential to recovery.
3. This version does not modify `src/main.nim`, but the already-updated Mummy
   signaling service (with `group-describe`) must still be running. Restart
   the signaling service if it is still an old version.
4. Run each WebView GUI in a different terminal (same project checkout):

   ```sh
   python -m patchlib.cli gui
   ```

5. Start an existing room in GUI 1; load and join that same room in GUI 2.
   Confirm **different** peer IDs/journal locations in the two worker logs.

## Expected observation

- GUI 1 and GUI 2 both show `living=2 active=2 channels=1 authed=1`
  (once each has completed its initial connect; `synced` can lag during
  anti-entropy catchup). `elastic local event=N` continues on both.
- A newly joined GUI should print `SIGNED HELLO OK`, `SNAPSHOT READY`,
  and `JOIN READY`.
- After closing GUI 1, GUI 2 continues with `living=1 active=1`;
  it remains an authorized, snapshot-serving peer. A new GUI 3 can join it.
- If you still encounter disconnections, check the *worker logs of both
  windows* for `TRAINER EVENT LOOP STALL`, `WebRTC link failure`, `peer-left`,
  `negotiation timeout`, `SIGNED HELLO FAILED`, or the state-channel error.

## Tests

`python -m pytest -q`: **228 passed, 3 skipped** in the CPU container.
New regression tests simulate a 270ms model export and 250ms working-state
serialization and check the event loop's progress, signed event convergence,
and working-state integrity. A macOS real-WebRTC dual-GUI execution was not
performed here (aiortc/Nim are not installed in the test environment).

## Known limitations

- An entire operating-system process blocked beyond several minutes cannot
  transmit application-level pings; no protocol can guarantee connectivity
  if both P2P and signaling are unable to make progress.
- An abnormal signaling-server crash can still interrupt negotiation for
  entirely new peers until the service restarts. Already-established direct
  SCTP links do not require an active server, but any link that actually fails
  must be renegotiated through signaling.
- This is an asynchronous, eventual-consistency scheme, not consensus under
  permanent network partitions.
