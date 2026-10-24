**2026-10-23 LoHa rank16 audit:** [README_LOHA_RANK16_AUDIT_2026_10_23.md](README_LOHA_RANK16_AUDIT_2026_10_23.md). Balanced LoHa initialization, GPT-Neo target selection, AdamW LR schedule, baseline/training-loss diagnostics.

**新機能 — 分散学習の有効性ベンチマーク（同一トークン予算、LoHa、ピア数補正時間）＋黒背景/emerald-500 GUI:** [README_EFFECTIVENESS_BENCHMARK_2026_10_23.md](README_EFFECTIVENESS_BENCHMARK_2026_10_23.md)。

**Two-window peer discovery and crash-resume fixes:** see [README_JOIN_RESUME_FIX.md](README_JOIN_RESUME_FIX.md). GUI windows now acquire independent persistent profiles; current local LoHA and AdamW tensors are snapshotted on each optimizer step.

**Apple MPS update-validation fix:** See [README_MPS_FLOAT64_FIX.md](README_MPS_FLOAT64_FIX.md).

# Automatic genesis and MPS/CUDA fix

**New:** See [README_AUTOBOOT_DEVICE.md](README_AUTOBOOT_DEVICE.md). The WebView initial `.fstate` field is optional; automatic HF genesis creation runs in a child process. LoRA/LoHA parameter-device placement is fixed. This does NOT yet implement automatic peer-assisted initial model transfer.

# Fedpara — WebView Desktop GUI + Hugging Face dataset collections

Run `python -m patchlib.cli gui` after installing `requirements-gui.txt` to open a standalone desktop window rendered from bundled HTML/CSS/JavaScript with a Python bridge. **No localhost HTTP server or GUI port is used.** See [README_WEBVIEW.md](README_WEBVIEW.md).

The room v2 manifest stores actionable Hugging Face dataset repo/config/split/revision/columns/weights, not just descriptive text. The training process directly consumes these via the Python `datasets` loader.

# Fedpara Adaptive P2P

**New:** See [README_ADAPTIVE.md](README_ADAPTIVE.md) for device-aware elastic training, bounded room descriptors, opt-in MoE pruning, the local GUI, DDoS protections and signaling outage mode.

```bash
python -m patchlib.cli gui
```

---

**New default connection setup:** `--room-id` joins automatic room discovery with no manual keys or roster. See [README_ROOM_ID.md](README_ROOM_ID.md).

## Elastic P2P live training (new default)

`python -m patchlib.cli train-live` now uses asynchronous `--mode elastic` by default; see [README_ELASTIC.md](README_ELASTIC.md) for peer joins, offline/persistent disconnects, signed gossip, SQLite replay and checkpoints. Set `--mode synchronous --rounds N` to use the original fixed-roster exact FedAvg path.

# WebRTC server written with Nim + Mummy

This is a small **WebRTC service**, not a home-grown implementation of every wire protocol. The HTTP/WebSocket signaling is Nim/Mummy, STUN and TURN are served by coturn, and the optional native WebRTC endpoint (ICE + DTLS + SCTP + SRTP) is implemented through the C API of libdatachannel. It includes a Japanese browser test page.

**Status:** Reference implementation. The Mummy and libdatachannel APIs and coturn REST authentication format were checked against their documentation. The sandbox used to prepare this project does **not** have the Nim compiler, coturn, or libdatachannel installed, so end-to-end execution and Nim compilation have **not** been verified here. Test and harden before deploying.

## Features / protocol ownership

| Function | Actual implementation |
|---|---|
| SDP offer/answer and room signaling | Mummy, `src/main.nim` |
| Trickle ICE signaling | WebSocket forwarding and native C callback handling |
| STUN and TURN allocation / media relaying | coturn (UDP/TCP 3478 and relayed UDP 49160-49200) |
| Temporary TURN credentials | Mummy, HMAC-SHA1 TURN REST API, 1-hour validity |
| ICE connectivity checks | Browsers / libdatachannel ICE stack; coturn relays packets |
| DTLS and SRTP for P2P calls | Browser `RTCPeerConnection` on both peers; relays forward ciphertext |
| Server-side DTLS, SCTP, SRTP termination | Optional `-d:rtcNative` / libdatachannel. DataChannel echo; SRTP receive RTP packet counter (when client includes media tracks) |
| SFU, transcoding, mixing, video recording | **Not implemented** |

STUN and TURN should NOT be re-implemented by a few ad-hoc UDP handlers. TURN needs authenticated allocations, permissions, channel binding, refresh/expiry, TCP/UDP relays, rate limiting, and defenses against abuse. Similarly, use a mature DTLS and SRTP engine.

## 1. Fast local P2P experiment

Install **Nim 2.x**, `nimble`, and OpenSSL (required by `std/openssl` at runtime) on your system.

```sh
cd webrtc_mummy
nimble install -y mummy
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Open `http://127.0.0.1:8080` in two tabs with the same random room ID. By default, signaling binds only to localhost; the browser can attempt a host-candidate connection even if coturn is not running. This local configuration is not a valid test of TURN connectivity or NAT traversal.

Click **connect** in both tabs. The first receives `peer-joined`, creates an offer and a DataChannel; the second answers. Send messages once the DataChannel opens. Checking audio/video asks for camera and microphone access.

## 2. TURN/STUN for a public Linux server

Provide a real reachable IPv4 address, DNS hostname, unique random TURN secret (at least 32 chars), and a separate APP_TOKEN for the signaling endpoint. The example Docker compose uses Linux host networking and exposes the relay port range without Docker NAT. `PUBLIC_IP` must match coturn's reachable external address; complex multi-homed setups require more configuration.

```sh
cp .env.example .env
# Edit .env: PUBLIC_IP, TURN_HOST, TURN_REALM, TURN_SECRET, APP_TOKEN, ALLOWED_ORIGIN
set -a; . ./.env; set +a
# On Linux with Docker:
docker compose --env-file .env -f deploy/compose.yml up -d
# In a separate terminal (with the same environment):
BIND_ADDR=127.0.0.1 nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Place an HTTPS reverse proxy in front of `127.0.0.1:8080`, forwarding WebSocket upgrade requests (WSS). Serve the browser test page on the configured `ALLOWED_ORIGIN`. Type the APP_TOKEN into the test page; the server checks it before returning TURN credentials or joining rooms.

Open network security rules for **TCP/UDP 3478** and **UDP 49160-49200**. These port ranges are an example and may be small for production traffic. TCP TURN listener is enabled; relayed traffic is UDP in this example. HTTPS TCP 443 belongs to your reverse proxy. DTLS/SRTP between WebRTC peers uses the endpoints' negotiated ICE transport, **not** coturn's optional TURN-over-DTLS listener (disabled in this example).

The server issues `timestamp:peer` + base64(HMAC-SHA1(TURN_SECRET, username)), matching coturn's `--use-auth-secret`. The one-hour expiry is suitable for an initial allocation, but production sessions longer than an hour should reissue ICE configuration and coordinate renewals/restarts as needed.

The configured TURN credential is bearer capability for your TURN relay; protect the signaling join token and rotate secrets if exposed. With one shared `APP_TOKEN`, every holder can request relay credentials; in production issue app-specific authenticated identities and apply quotas/rate limits. `ALLOWED_ORIGIN` is an additional browser origin restriction, not authentication.

## 3. Optional server-native DTLS/SRTP endpoint

Install [libdatachannel](https://github.com/paullouisageneau/libdatachannel) **with its C API and shared library** on the machine where you run Nim (the loader must find `libdatachannel.so` or `libdatachannel.dylib`). You can build it via CMake; on macOS make sure the install location is in the dynamic loader path. The Nim source intentionally binds dynamically, so compiling does not require a static C link step.

```sh
nim c -d:release -d:rtcNative --threads:on --mm:orc -r src/main.nim
```

Select **Nim native echo** in the browser demo and connect. A WebRTC DataChannel offer arrives over Mummy WebSocket; the Nim server uses the libdatachannel C interface to set the remote offer, generate an SDP answer and its ICE candidates, establish DTLS/SCTP, and echo messages over the DataChannel.

If you turn on video/audio in this mode, the server also installs a remote-track callback and counts decrypted RTP packets delivered by libdatachannel (printed every 250 packets). It **does not** decode, play, echo, mix, or forward the media track. Thus the native path exercises SRTP reception when properly negotiated, but **is not an SFU**.

To receive connections from the public internet, the native host needs reachable ICE candidates (public network address / proper NAT traversal and firewall rules). The minimal native `rtcConfiguration` does not yet specify outbound STUN/TURN servers; extend the config if behind restrictive NAT. Native mode handles one SDP negotiation per WebSocket. Renegotiation and ICE restart are not implemented.

## Message protocol

Browser sends:

```json
{"type":"join","mode":"p2p","room":"32-character-random-room-id","token":"APP_TOKEN"}
{"type":"offer","sdp":"v=0..."}
{"type":"answer","sdp":"v=0..."}
{"type":"candidate","candidate":{"candidate":"candidate:...","sdpMid":"0","sdpMLineIndex":0}}
{"type":"candidate","candidate":null}
```

For native endpoint use `"mode":"echo"` with no room field. Server sends `joined` containing `iceServers`, `role`, and optionally `room`; `peer-joined` or `peer-left`; and forwarded `offer`, `answer`, `candidate`. Native mode sends its own answer, candidates, state and optional `rtp-stats` reports.

## Further production work

- Per-user authentication and TURN rate/quota enforcement, metrics, heartbeat and connection timeouts.
- Structured logging, reverse proxy TLS, host hardening and monitored TURN capacity.
- Client reconnect and perfect negotiation for glare/renegotiation; full ICE restart and temporary TURN credential renewal.
- A true multi-peer **SFU** needs track/routing/RTCP/congestion control beyond this example.

### API references

- [Mummy WebSocket example](https://github.com/guzba/mummy/blob/master/examples/basic_websockets.nim)
- [libdatachannel C API](https://libdatachannel.org/pages/reference.html)
- [coturn TURN REST credentials](https://github.com/coturn/coturn/blob/master/README.turnserver)

## NEW: Sharded model-state synchronization

See **[README_SHARDS.md](README_SHARDS.md)** for direct tensor-by-tensor state sharding, content-addressed deduplication, SHA-256 verification, Ed25519-authenticated P2P transfer, cached-shard resume, and assembly back into `.fstate`. New CLI commands: `shard-export`, `shard-create`, `shard-inspect`, `shard-exchange`, `shard-assemble`.

## Expert Parallelism extension (experimental)

See [README_EXPERT_PARALLEL.md](README_EXPERT_PARALLEL.md) for selective
safetensors loading, signed P2P expert forward/backward primitives and demo,
quantized expert bases, CPU/NVMe paging, packed-vs-unpacked MoE router support,
and GUI custom source upload. These are intentionally NOT described as complete
integration with the earlier asynchronous LoHA FedAvg trainer.

## New: Verified live full-model catch-up

See [README_LIVE_FULL_SNAPSHOT.md](README_LIVE_FULL_SNAPSHOT.md). A newly joined
room peer now downloads a signed, sharded **current effective model** on a
second DataChannel, checks the event vector clock, and only then may train.
Diagnose persistent `active=1` with `scripts/check_live_p2p.py`.

### Experimental Elastic v2 evaluation and redesigned room directory

See [README_ELASTIC_V2.md](README_ELASTIC_V2.md) for the new deterministic
buffered aggregation library, signed epoch proposal codec, reproducible optimizer
A/B, opt-in Swarm counterfactual analysis and searchable room directory.
Live training remains on the stable V1 protocol until epoch consensus is implemented.
