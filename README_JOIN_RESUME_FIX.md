# Fedpara — independent GUI peers and durable local resume

## What was broken

Previous WebView GUI windows used the same `.fedpara-gui/rooms/ROOM/events.sqlite3` and therefore the same `events.sqlite3.identity.pem`. They were not distinct peers. Stopping a child process via SIGTERM also could drop the partially trained LoRA/LoHA adapter between two journal events. The UI's `emitted` counter reset to zero even after earlier events were replayed.

## How it works now

- **One stable identity per window:** two instances of `python -m patchlib.cli gui` on the same Mac claim separate, locked peer profiles. The room's *immutable* genesis and dataset manifest remain shared. Identity, events journal, optimizer working snapshot, and trust database are **private to each peer**.
- **Reclaim after restart:** after the GUI closes, its locked profile becomes available for later use. If there was an older single-window journal, the first GUI keeps using its **existing location**, not a new empty journal. Running two windows produces two independent peer IDs and logs; both may reuse those previous identities after a full restart.
- **Durable updates:** committed peer patches are written to SQLite WAL (`synchronous=FULL`) and replayed from the original genesis on every restart. On every complete optimizer step, the in-progress local factor and AdamW tensors are stored as a local-only, atomic `.working.safetensors` file without pickle. After a crash, they are reloaded **only if the journal tip still matches**, preventing a stale in-progress delta being replayed on the wrong model.
- **Graceful exit:** SIGTERM / SIGINT from GUI shutdown is handled by the training loop. On exit the current accumulated gradients are stepped (if necessary) and the pending LoRA/LoHA patch is sealed into the journal before process shutdown. `GUI.close()` waits for the child to complete before unlocking the identity.
- **Exclusive journal:** a second process that accidentally tries to write to the *same* peer's SQLite journal fails with an explicit message, rather than corrupting sequence numbers.
- **Clear metrics:** logs distinguish `living` (server-reported presence, when online) from `active` (connected, authenticated, caught-up direct P2P neighbors, including self). `emitted` on restart is the recovered count of local events, not zero.

## Launch and verify two windows

```bash
python -m patchlib.cli gui
# Open a second terminal in the same project directory:
python -m patchlib.cli gui
```

Both windows choose **the same room ID**. Start training in both. Look for lines like:

```
elastic peer=p... recovered=... events; log=.../peers/peer-.../events.sqlite3
peer registered: p...
elastic steps=... emitted=... active=2 living=2 peer=p...
```

The `peer=` and `log=` paths must differ between the two windows. `living=2` should appear after room presence is established. `active=2` requires a working authenticated WebRTC DataChannel; NAT/firewall/coturn failures can leave `active=1, living=2` even though both peers registered successfully. To debug, check both log tails for `peer registered`, `connecting`, `hello`, `ice`, and WebSocket connection warnings.

You can use `--durable-every N` for direct CLI `train-live` runs. Default is `1` optimizer step, prioritizing crash recovery; `0` disables local snapshots, at the cost of losing uncommitted steps after a hard kill. The working snapshot is never distributed and may cost noticeable disk writes for very large models.

## Limits

- A hard crash in the narrow interval between optimizer.step and the following durable save can still lose that single step. Gradient accumulation not yet stepped is not persisted. A graceful stop does flush it.
- Dataset iterator position and random-number generator state are *not* yet snapshotted, so data may be sampled again after restarting; model weights and optimizer moments do recover at the last durable point.
- The model remains temporarily inconsistent between disconnected peers until all verified journal events arrive through P2P gossip. This remains **asynchronous additive FedAvg approximation**, not synchronous FedAvg.
- The signaller/server and `aiortc` are not available for real multiprocess WebRTC integration verification in this execution environment. Two local profiles and simulated room membership are covered by Python tests, not physical network tests.
- Profiles are automatically reclaimed in oldest-first order. For deterministic mapping between a particular human and a profile on a shared Mac, an explicit profile selector UI is a future improvement. Do not delete `.fedpara-gui/rooms/ROOM/peers/` or the original `events.sqlite3`; that is the learning history.

## Tests

```bash
python -m pytest -q tests/test_join_resume_fix.py \
    tests/test_webview_gui.py tests/test_roomid.py tests/test_elastic.py
```
