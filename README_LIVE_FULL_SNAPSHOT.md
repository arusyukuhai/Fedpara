# Fedpara — live P2P full-weight bootstrapping (experimental)

This release changes actual data flow, not just `active` log formatting. Room peers now open two **separate ordered WebRTC DataChannels on the same RTCPeerConnection**:

* `fedpara-patch`: authenticated hello, elastic event inventory and LoRA/LoHA deltas.
* `fedpara-state`: authenticated sharded full *effective* model transfer, streamed to a content-addressed disk store.

Neither payload goes via Mummy (except when a separately deployed TURN server is relaying traffic for NAT traversal).

## How a new peer starts

1. GUI / `train-live` creates or validates its *immutable genesis* (`.fstate`), pins the room model manifest and opens a **distinct** per-window identity/journal.
2. Mummy `group-join` establishes room membership. A new peer cannot assume the room is empty without confirmation.
3. Each neighboring pair creates a normal WebRTC connection, DTLS and an authenticated Ed25519 `hello`. The receiving peer requests `snapshot-request`.
4. An existing peer **seals uncommitted local LoRA/LoHA** and saves its then-current effective weights (embedding, buffers, all other registered parameters) directly as `.fmanifest` + SHA-256-addressed blobs. It signs the checkpoint *event vector clock* on `fedpara-patch` and signs the manifest transfer header on `fedpara-state`.
5. The incoming peer requests **only missing shards**. Each shard is checked against the manifest. Completed chunks can be reused across interrupted transfers.
6. Only when a full manifest, its signature, every shard, model ID/revision, and all source-included signed event IDs/digests match does Fedpara assemble `.fstate` and apply it to the existing parametrized PyTorch model. Later journal events beyond the signed snapshot cutoffs are replayed **once** on top of it.
7. `SNAPSHOT READY` appears, followed by `JOIN READY`. **No new-room-peer training before verified state synchronization.** The full signed event log is retained for restart replay; it is not truncated or compacted.

### Important limitations

* This is an **experimental integration** of the sharded transfer protocol into live Elastic training. Python model/shard tests pass, but true `aiortc`/macOS connections, Nim signaling compilation and full multi-process live training have not been tested in the build environment.
* The local architecture and immutable **genesis checkpoint must still exist** before an incoming model can be instantiated. For supported packaged Hugging Face models, `--auto-state` makes this automatically. The new transfer replaces the **current effective weights**, not the model implementation, tokenizer, configuration, or missing genesis bootstrap. It is **not** a universal from-zero code/model bootstrap for custom/untrusted models.
* New peers still download the origin-signed event log to establish the exact snapshot boundary. **A snapshot is not yet a signed compacted history epoch**. Sending both the full snapshot and old event log costs extra bandwidth for long-lived rooms; compaction/epoch migration is follow-up work.
* Streaming starts only once authenticated DataChannels exist. A failing ICE/DTLS/TURN/WS connection cannot be repaired by adding checkpoint logic. For real-world security, a shared room ID is **not** an authorization system against a malicious participant. This mode trusts an authenticated room member's provided weights; signatures and checksums do not prove its model is honest or high quality.
* The default four-neighbor gossip topology is not an atomic global consensus mechanism. In a partition, models can diverge until updates reconcile. Model equivalence is not inferred from loss values.

## Install and try two windows

Preserve **all** existing `.fedpara-gui` files (keys, SQLite journals, checkpoints, working-state files); **do not delete** them.

```sh
python -m pip install -r requirements-patches.txt
python -m pip install -r requirements-gui.txt
```

Start the updated Mummy server in one terminal. Confirm you do not have an **older** server still listening on TCP 8080.

```sh
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Before involving models, diagnose *both* stages:

```sh
python scripts/check_group_presence.py --ws ws://127.0.0.1:8080/ws
python scripts/check_live_p2p.py --ws ws://127.0.0.1:8080/ws
```

The second script creates **two independent cryptographic peers and real aiortc RTCPeerConnections** against Mummy and expects both to establish a DataChannel and exchange signed control frames. It reports per-peer errors if anything fails. On the build host, `aiortc` is missing so the diagnostic **was not executed**.

Then launch **two independent processes**:

```sh
python -m patchlib.cli gui
python -m patchlib.cli gui
```

Join **the exact same room ID**, model, revision and LoRA/LoHA settings. Logs should progress approximately as:

```text
room presence: ... living=2
peer registered: p...
P2P room=... ICE/DTLS state=connected
SNAPSHOT REQUEST peer=p...
SNAPSHOT TIP peer=p... events=...
SNAPSHOT DOWNLOADED peer=p...
SNAPSHOT READY: verified full model from ...
JOIN READY: ...
elastic steps=... living=2 active=2 synced=2 channels=1 authed=1 ...
```

Interpret counters: `living` = Mummy's room membership including self, `active` = **authenticated** WebRTC peer links plus self, `synced` = anti-entropy caught up plus self, `channels` = connected neighbor count excluding self, `authed` = signed-hello verified neighbor count excluding self. No logged number on its own proves the two training models match. `SNAPSHOT READY` is the actual full-state import point.

If `check_group_presence.py` fails, the issue is Mummy group discovery or connection parameters. If it passes but `check_live_p2p.py` fails, inspect the `P2P room=... ICE/DTLS state=...` and per-peer errors: the problem is below the model layer (pair signaling, ICE, DTLS, SCTP, or python dependencies). If both pass but the GUI cannot sync, inspect the `JOIN WAIT`, `SNAPSHOT`, journal and matching-genesis logs.

## Tests

```sh
python -m pytest -q tests/test_live_snapshot.py tests/test_roomid.py tests/test_cold_join_guard.py
```

No new server-side model/database storage is used; the centralized Mummy server still handles signaling and bounded membership.
