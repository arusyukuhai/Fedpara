# Fedpara: bounded gossip and persistent room directory

This update builds on the tested `fedpara_reconnect_peer_seed_fixed.zip` source. It preserves the WebView GUI and the elastic LoRA/LoHA training implementation.

## Core changes

- `MaxGroupMembers` is now **96** in Mummy; room discovery's Python validator accepts 96.
- Direct WebRTC connections use a symmetric, deterministic **power-of-two ring** with up to **eight neighboring peers** (`±1, ±2, ±4, ±8`). On 96 members this is at most 384 *peer-to-peer links*; not 4,560 full-mesh links. Two signaling WebSockets may be required per link. The overlay is connected given an identical authoritative roster and functioning edges. Different in-flight rosters are eventually reconciled by the room-presence protocol.
- A registered room that has **never had a training peer** can be founded from genesis. Once any actual peer has joined, that fact is persisted. If the room later goes empty, a newly joining peer cannot silently treat it as a brand-new model.
- Joining peers request a *full-model snapshot from at most one authenticated, signed-advertised ready source* at a time; a peer without a verified current model cannot advertise source readiness. After it has verified a snapshot, it may act as a source for its neighbors.
- Gossip inventory cadence is less aggressive (`0.4` seconds per page and `60` seconds between repeat cycles) to mitigate many-peer data-channel pressure. Full-history gossip is **not** yet amortized to sublinear cost. Large histories still need checkpoint/compaction work.
- Once a peer permanently leaves a room, inactive connection slots can be discarded after their tasks close. Signed history remains in SQLite.

## Persistent room directory

Run Mummy with `ROOM_DATA_DIR` pointing at a backed-up, writable directory; its default is `./fedpara-rooms`, **relative to the server process's working directory**. The server creates owner-only JSON files on the room's first successful join and restores these manifests at startup. The models' weights, datasets, Python executable code, keys and LoHA updates are **not** stored on this server. A registered room does not automatically have a live model checkpoint after every participant leaves.

Only a room explicitly marked public at initial registration becomes visible at `GET /api/rooms` (up to 256 bounded results). Private room IDs are bearer secrets and are not listed. The public directory includes the room ID, model ID/revision, dataset repository references and online count. Publishing an ID lets other people *enter that room*, so it is a security-sensitive opt-in. To publish from a remotely accessible Mummy instance, require a `ROOM_PUBLISH_TOKEN` environment variable set on the server, and pass the same token as an environment variable to the authorized GUI/training client. Neither secret is stored in the manifest. Publication status is immutable; create a different room to change it.

The WebView now has **公開ルーム一覧を更新** (fetch public rooms) and **このルームを公開一覧へ掲載** (publish). Clicking a listed room fills in its ID; use the existing '既存ルームを読み込む' to fetch its dataset configuration. A new room is persisted when either **サーバーにルーム登録** is clicked in the GUI, or its first training peer joins. Local **ルーム設定を保存** only saves locally and works offline. Standalone metadata registration does not pretend to be a trainer: the first genuine training peer can still initialize a new room. Room directories have a total cap of 4096 entries. Directory searches return a bounded page of 256; there is not yet pagination or a room-deletion/admin endpoint.

For CLI:

```bash
python -m patchlib.cli room-list --ws ws://127.0.0.1:8080/ws
python -m patchlib.cli train-live --room-public --room-id YOUR_NEW_ROOM_ID ...
```

For real signaling/load testing, first run Mummy, then:

```bash
python scripts/load_test_room_presence.py --ws ws://127.0.0.1:8080/ws --peers 48
python scripts/load_test_room_presence.py --ws ws://127.0.0.1:8080/ws --peers 96
```

This verifies server-side join/leave membership convergence for many WebSockets, **not** 96 real aiortc channels or simultaneous training workloads. It creates an unrelated throwaway room, not an existing training room. To compile Mummy on your Mac, run `bash scripts/check_nim.sh` first; the current artifact environment has no Nim compiler.

## Safety, limits and operational risks

- **No unconditional 96-peer runtime claim.** The tests prove topology degree/connectivity in pure Python and preserve existing unit tests. Actual 96-device throughput depends on upload bandwidth, RTT, TURN, model size, dataset streaming, failure rates and hardware resources.
- The signaling server still sends O(N) small membership notifications per join and retains O(N) membership IDs per room. Total clients and message sizes are bounded. Multiple Mummy instances do **not** yet share an authoritative global room directory/roster; use one primary server or add a transactional shared datastore before active-active deployment.
- Frozen/permanently absent previous model owners can leave a room with metadata but without recoverable weights. A **new** participant must wait for a verified existing model source; it must not silently reset a previously established room to genesis.
- Encryption and signatures do not prevent poisoned model updates from an invited malicious participant. A public room is not appropriate for untrusted contribution without additional policy/validation.
- The directory is a lightweight read-only API; the reverse proxy should limit request/connection rates. The included Nginx example is tuned upward so peers behind one NAT can join in larger batches. Application-layer limits do not replace upstream DDoS mitigation.
- A full event journal must eventually be compacted at an agreed signed checkpoint epoch. This prototype still has linear anti-entropy over old event histories, no Byzantine agreement and no WAN long-run soak tests.

## Files most relevant to this change

`src/main.nim`, `patchlib/topology.py`, `patchlib/elasticnet.py`, `patchlib/roomid.py`, `patchlib/webview_controller.py`, `patchlib/webview_gui.py`, `patchlib/gui.py`, `patchlib/cli.py`, `web/desktop_webview.*`, `scripts/load_test_room_presence.py` and the added topology tests.
