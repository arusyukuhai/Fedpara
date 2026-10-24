# 2026-10-22: `waiting-state-channel` / `full-model DataChannel never opened`

## Observed from user log

- `ICE=completed`, `rtc=connected`, and `SIGNED HELLO OK` indicate that the authenticated control transport is live.
- `SNAPSHOT SOURCE stage=waiting-state-channel` on the joining side shows the source cannot use its state channel.
- After 25 seconds the joiner sets `protocol_failure` and destroys the working peer connection.
- The log **does not** indicate a NAT, TURN, STUN, or signed-Hello failure.

## Root cause verified from aiortc source

`RTCSctpTransport._data_channel_receive()` calls `channel._setReadyState('open')`
*before* emitting the peer connection's `'datachannel'` event. The receiving
application installs its `@channel.on('open')` listener only when that event
arrives; no second `open` notification is generated.

Our previous `LiveSnapshotLink.attach()` waited exclusively for that missed
notification. As a consequence, the answerer's `snapshot_link.connected` never
became true, even with SCTP already working. This also applies to the control
channel in `ElasticNode._attach_channel()` and to the basic `PatchNode`.

## Fix

On channel attachment, **after registering all event handlers**, call the
`open` initialization immediately if `ch.readyState == 'open'`. Guard that
initialization to avoid duplicate send workers / Signed Hello packets if a test
or an implementation subsequently emits a redundant `open` event.

On channel replacement, reset the connection flag and cancel the superseded
worker before attaching the new channel; old-channel close events must not
reset the new channel. The snapshot link already rejects obsolete callbacks.

Changes:
- `patchlib/snapshot_live.py`: immediate state channel initialization + logging.
- `patchlib/elasticnet.py`: immediate authenticated control initialization.
- `patchlib/p2p.py`: same for generic patch transfer.
- `tests/test_incoming_preopened_channels.py`: regression tests for the actual
  aiortc event order and end-to-end manifest/shard transfer with a pre-opened
  incoming channel on either endpoint.
- `tests/test_elastic.py`, `tests/test_roomid.py`,
  `tests/test_hello_clipboard_regression.py`: update unrealistic synchronous
  fake-channel event ordering; behavior and assertions are preserved.

## Running

Stop both GUI/training processes, overwrite the Python files from this ZIP,
then relaunch both GUI windows; the Mummy server does not need recompiling for
this fix. Do NOT delete `.fedpara-gui`, SQLite files, stored keys, or snapshots.

```bash
python -m patchlib.cli gui
```

For a focused regression run:
```bash
python -m pytest -q tests/test_incoming_preopened_channels.py \
    tests/test_snapshot_request_delivery.py
```

Expected new diagnostic message when state channel is recognized:
`full-model state DataChannel ready label=fedpara-state id=...`.

Look for source-side `SNAPSHOT SEND PENDING` to transition from
`waiting-state-channel` to `SNAPSHOT OFFER`, followed by recipient-side
`SNAPSHOT DOWNLOADED`, `SNAPSHOT READY`, and `JOIN READY`.

## Verification limits

Mocked DataChannel tests, Python compilation, and Python regression tests pass.
We cannot run a real Mac two-window aiortc / Mummy / GPU integration test
here, so real-device behavior is not yet certified. If another stall remains,
provide the *source-side* state-channel-ready and snapshot-export logs.

Official implementation reference:
https://github.com/aiortc/aiortc/blob/main/src/aiortc/rtcsctptransport.py
(`_data_channel_receive`, DCEP OPEN branch).
