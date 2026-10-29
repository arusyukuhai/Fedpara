# Fedpara v17 — large-room signaling and bounded overlay

This release makes **bounded-v2 signaling the sole supported room topology**
in the existing Tailwind GUI / v16 P2P client. No changes to adapter ranks, FedSum, ReLoRA,
Genesis, signed model updates, binary journal, or event replay.

## What scales and what does not

- Bounded-v2 rooms admit up to **4,096 simultaneous room participants per Mummy instance** (a **configured cap**, not unlimited capacity). A signaling server has a **global 65,536 WebSocket client cap** including temporary pair-negotiation sockets; actual resource constraints will usually be lower. Raise server file-descriptor limits and provision RAM before large experiments. A production environment needs monitoring, admission/rate limits, and resource-aware deployment.
- The signaling service calculates a deterministic symmetric ring/chord overlay for each membership change. Each member receives only its **2/4/6/8 direct neighbors**, not the entire room roster. Join/leave events notify only peers whose neighbor set changes (at most about 40 candidates per change), rather than broadcasting every presence event to everyone.
- The signaling server keeps the full active peer registry; each training client keeps only the directly relevant identities. Lobby heartbeat updates report the approximate current online count every 30s, without transmitting the entire roster. Stale out-of-order neighbor revisions are rejected.
- Direct WebRTC neighbor degree remains bounded. Per-peer **connection bookkeeping** no longer grows with room population. Gossip still propagates through direct neighbors and must repair missing events after churn.
- **Crucial limitation:** distributing every training update to *every* peer is still at least O(N) received updates per participant if every participant contributes at a fixed rate. Total bandwidth, persistence, validation cost, snapshot availability, model convergence and eventual anti-entropy still **do not scale unboundedly**. Large rooms need update batching, checkpoint/compaction, adaptive update rates and potentially hierarchical/federated aggregation to achieve useful throughput.
- The existing single Mummy server is an admission and membership authority; it is not an active-active distributed directory. High availability across server failures is **not** solved. A room can still lose recoverable models if all authorized snapshot sources disappear.

## Signaling protocol

New clients send `topology_v2: true, topology_degree: 8` in `group-join`.
They receive `group-members` with `topology: "bounded-v2"`, a small `peers`
(neighbor-only) set, `online` (total), `revision`, `degree`, pinned `manifest`
and `existing_room`. When affected by joins or leaves, they receive
`group-neighbors` with a newer revision. `ping` returns a small `group-count`
frame for compact clients. Counts and membership are **server hints, not
cryptographic attestations**; DataChannel peer identities and signed training
updates still require separate verification.

**No legacy full-roster fallback remains.** The updated signaling server
requires `topology_v2: true` on every `group-join`, and updated clients reject
responses that do not contain `topology: "bounded-v2"`. The old signaling
server and old clients must both be upgraded; do not mix versions within an
active room. Fully stop old v16 peers and create a **new room ID** for v17.
A different neighbor degree within a live room is rejected.

## Validation and load-test instructions

```bash
# From a directory containing the Fedpara project; rebuild/restart Mummy.
bash scripts/check_nim.sh
python scripts/load_test_room_presence.py --ws ws://127.0.0.1:8080/ws --peers 256 --depart 32
python scripts/load_test_room_presence.py --ws ws://127.0.0.1:8080/ws --peers 1000 --depart 200
python scripts/load_test_room_presence.py --ws ws://127.0.0.1:8080/ws --peers 4096 --depart 512
```

The script tests **signaling membership and bounded adjacency** with real
WebSocket sessions; it does not create aiortc WebRTC channels or run training.
Start with tens of peers before hundreds/thousands. Avoid public deployments
without explicit admission, authentication and reverse-proxy limits. Do not
run stress tests against other people's public signaling servers.

Pure-Python topology and reconnect regression tests can run with:

```bash
python -m pytest -q tests/test_large_room_bounded_topology_v17.py
python -m pytest -q
```

**Validation caveat:** the patch environment may not have a Nim compiler or an
operating WebRTC signaling service. In that case, the source needs an actual
Nim 2.x compilation and multi-process signaling integration test on the user's
machine before production deployment. Unit tests cannot verify the Nim server
or 4,096 simultaneous real model-training peers.
