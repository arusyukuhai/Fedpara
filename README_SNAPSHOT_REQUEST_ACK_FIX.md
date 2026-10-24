# Fedpara full-model snapshot request/ACK fix (2026-10-22)

## Failure observed

The new peer reaches `active=2`, `phase=authenticated`, `ice=completed`,
`rtc=connected`, `state_channel=True`, but remains at
`waiting-for-verified-full-model-snapshot`, with `tip_received=False` and
`manifest_received=False`.

This means WebRTC is established, but there is no evidence that the source
processed the signed snapshot request or started transferring the model.

## Specific defect repaired

The older sender handled `snapshot-request` by calling `send_full_snapshot()`
only once. If the separate `fedpara-state` DataChannel was not open at that
instant, `send_full_snapshot()` returned without sending anything or preserving
the request. DataChannels can open at different times, even when ICE/DTLS and
the signed control channel are already healthy. A newcomer could repeatedly
wait, especially if subsequent requests raced similarly.

## Changes

- Receipt of a signed, authenticated request is acknowledged **immediately in
  the control channel message callback**. Slow dataset downloads and training
  steps cannot delay this receipt ACK.
- The sender **latches** the request until the state DataChannel opens, then
  attempts to export and offer the full model. Failed sends/backpressure keep
  the request pending.
- A not-yet-authoritative cold joiner will NOT answer model requests with its
  unverified genesis state.
- Signed `snapshot-status` messages report `accepted`, `waiting-state-channel`,
  `source-not-ready`, `exporting`, or `offered`.
- The receiver retries unacknowledged requests every 5s; acknowledged requests
  are retried less aggressively (20s) while transfer preparation progresses.
- `JOIN WAIT` includes `source_stage` and `request_age_s`. A missing error is
  displayed as an empty string instead of `None`.

No signatures, SHA-256 checks or model-history verification were bypassed.
No changes to Mummy signaling or persisted journal/keys.

## Install

Shut down both trainer/GUI Python processes, replace `patchlib/elasticnet.py`
(or unpack the full project ZIP over the corresponding project tree), and
restart both. The existing Mummy server need not be changed for this patch.
Do not delete `.fedpara-gui`, SQLite journals, local models or peer identities.

Run `python -m patchlib.cli gui` in two independent GUI processes with the
same room, wait for `SNAPSHOT READY`/`JOIN READY` on the cold joiner.

## New source-side diagnostic lines

- `SNAPSHOT RX REQUEST`: the source received a signed request.
- `SNAPSHOT SEND PENDING stage=waiting-state-channel`: the second channel
  hasn't opened yet, but the request has **not** been discarded.
- `SNAPSHOT EXPORT START`: the source is preparing the snapshot.
- `SNAPSHOT OFFER`: manifest queued for state-channel transfer.
- `SNAPSHOT SEND FAILED`: an explicit export or send error has occurred.

## New receiver diagnostic lines

- `SNAPSHOT SOURCE ... stage=accepted`: source got the request.
- `... stage=exporting`: model materialization is happening on the source.
- `... stage=offered`: source has queued the snapshot manifest.
- `source_stage=no-ack`: no source receipt has been confirmed yet.
- `tip_received=True, manifest_received=False`: investigate the state
  DataChannel, shard sender errors, or any unexpected disconnect.

## Verification

`python -m pytest -q tests` -> **128 passed, 3 skipped** (CPU environment).
A paired signed-control + fake-state-DataChannel test covers a request arriving
BEFORE the state channel opens and the subsequent signed manifest/shard transfer.

**Not verified**: real macOS 2-window aiortc/Mummy transfer or large-model
transfer. A test passing locally is not proof of real ICE/network behavior.
