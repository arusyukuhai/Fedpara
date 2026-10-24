# Snapshot join recovery hotfix

This revision addresses repeatable cold-join stalls after a joined peer connects:

1. Each WebRTC state DataChannel has its own `ShardSession`. A reconnect no longer carries a partially downloaded manifest or an `outstanding` request window into the next channel, causing the next `manifest-start` to be rejected as a duplicate. Fully verified SHA-256 shard blobs remain cached and are reused.
2. Stale WebRTC channel callbacks cannot clear a newer state channel. State-channel sender tasks are tied to their exact channel.
3. Unexpected Python exceptions in `aiortc` state-channel callbacks, including `AttributeError: 'NoneType' object has no attribute ...`, are logged with tracebacks and trigger a clean peer reconnect instead of disappearing into an event callback.
4. New members detect silent state-channel closure (25 s after the control channel opens) and stalled transfers (60 s with no activity after requesting a snapshot), and reconnect automatically. Outgoing shard/manifest chunks refresh activity, so a large transfer is not mistaken for stalled solely because the command takes time.
5. A fully downloaded state is retained while its signed event-vector history catches up, instead of redownloading that state every 30 s. `SNAPSHOT HISTORY WAIT` prints verified-history counts and distinguishes incomplete history from incomplete shard transfer.

## Usage

Replace the source files with this package. **Do not delete** `.fedpara-gui`, keys, journal databases, or checkpoint files. Quit all old GUI/training processes, restart the same Nim signaling server if necessary, then launch each GUI in a separate Python process:

```sh
python -m patchlib.cli gui
```

When a newcomer is blocked, inspect the **new** logs:

- `SNAPSHOT REQUEST`: authenticated control channel requested latest model.
- `SNAPSHOT OFFER`: existing peer produced a manifest.
- `SNAPSHOT TIP`: manifest hash and signed journal-vector boundary arrived.
- `SNAPSHOT DOWNLOADED`: all model shards were received and SHA-256 verified.
- `SNAPSHOT HISTORY WAIT ... events=x/y`: all model shards are present but signed historic patch records have not caught up. Do **not** delete the received state to fix this.
- `SNAPSHOT READY`: signed model and event cutoff verified, applied to PyTorch model.
- `SNAPSHOT ERROR ...`: state channel crashed. Full exception type is logged, link will reconnect.
- `SNAPSHOT STALLED`: transfer no-progress watchdog will reconnect and reuse SHA-256 blobs.

For network-stage diagnostics run `python scripts/check_live_p2p.py` against the running Mummy server. Capture the **complete traceback**, not only the `AttributeError` summary, if another exception appears.

## Verification and limitations

Python state/join regression tests and a signed full manifest transfer over mocked bidirectional DataChannels pass. No real `aiortc` connection or Apple Silicon MPS test was possible in the build environment. This release cannot prove that every unknown `NoneType` failure is fixed. It preserves signature/hashes and never treats a failed download as valid model state. Large initial model exports may take time; shard transfer resume is per verified shard, not per partial shard. This is an experimental P2P training framework and should not be used as the sole copy of valuable model weights.

Run focused checks:

```sh
python -m pytest -q tests/test_snapshot_reconnect_regression.py tests/test_live_snapshot.py tests/test_cold_join_guard.py tests/test_connection_stability.py
```
