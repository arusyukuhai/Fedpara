# Fedpara Swarm Simulator — 8 real peer processes

This is a **real-transport E2E test harness**, not a mathematical-only simulation.
It runs the existing `train-live` Elastic LoHA/LoRA implementation eight times in
separate OS processes, each with its own peer key, SQLite journal, model and
optimizer. Peers exchange Signed Hello, WebRTC DataChannels, shard snapshots and
signed Elastic event inventories using existing Fedpara code. Every peer is given
the same tiny, deterministic **packaged** Hugging Face GPT-2 fixture and local
safetensors batches; no Hub downloads or arbitrary Python are required.

The harness does **not** claim that a successful 8-peer run proves 96-peer scale,
WAN/TURN compatibility, resistance to malicious peers or convergence to a good
loss. Churn changes online population and therefore the existing asynchronous
averaging denominator; even event-exact replicas can have numerical rounding
differences. The final report tests these rather than assuming success.

## Install and run

On the computer with Fedpara installed:

```bash
python -m pip install -r requirements-swarm-sim.txt

# OPTION A: use existing Mummy (best fidelity to your real deployment)
nim c -d:release --threads:on --mm:orc -r src/main.nim
# In ANOTHER terminal:
python -m patchlib.cli swarm-sim --peers 8 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/run-001

# OPTION B: test even without starting Nim; real local WebRTC/UDP/DTLS/SCTP
# connections, but a small Mummy-compatible WebSocket signaling implementation.
python -m patchlib.cli swarm-sim --peers 8 \
  --embedded-signaling --output ./swarm-runs/run-002
```

**Outputs must be new or empty directories.** This prevents a test from
accidentally overwriting production `.fedpara-gui` identities, checkpoints, or
journals. All subprocesses are stopped on normal completion and on exceptions.

On macOS the run creates 8 PyTorch processes: plan sufficient free RAM (several
GiB in aggregate), and ensure that macOS allows the local WebRTC UDP traffic.
The local built-in signaling server binds to `127.0.0.1` only and forwards no
model weights, data batches or LoHA patches. `--ws` lets you test the **actual
Nim Mummy** version instead. Non-local signaling requires `wss://`; no insecure
remote signaling fallback occurs.

## Scenario

1. Generate one immutable genesis `.fstate`, 24 synthetic local tensor batches,
   and a pinned room manifest.
2. Start the first peer and verify its `JOIN READY`; start seven more, staggered.
3. Require verified `SNAPSHOT READY` for every newcomer before the E2E test can PASS.
4. Continue local LoHA updates for the warm-up period.
5. Select `--faults` distinct non-founder peers using the deterministic `--seed`.
   Force-terminate each with SIGKILL (simulating power loss), leave it down
   for `--down-time`, then restart the *same* peer journal and identity.
6. After reconnecting, create a `pause.flag`; existing training code first
   seals partial local steps, then continues networking and anti-entropy only.
7. Wait for signed update inventories to converge and stay stable.
8. Terminate cleanly, export all eight full effective `.fstate` snapshots,
   verify **each tensor's actual SHA-256**, and compare **numerical tensor values**.

The E2E run fails if **any** of the following are false: all peers joined,
all seven new peers received verified snapshots, all peers exchanged a signed
Hello over DataChannel, crashed peers restarted, event inventories converge,
origin/sequence event identities have no conflicting digests, or effective
model tensors agree within `max_abs <= 1e-3` (with non-floating buffers exactly
identical). The conservative tolerance acknowledges different update arrival
orders and float32 accumulation. It is not evidence of model quality.

## Artifact structure

```
swarm-runs/run-001/
  scenario.json                 random room ID and test parameters
  fixture/model/                local tiny pretrained HF model
  fixture/genesis.fstate        immutable common starting weights
  fixture/batches/              pretokenized local safetensors data
  fixture/room.json             immutable room manifest
  peer-00/events.sqlite3        isolated, signed event journal
  peer-00/run-01.log            individual process log and warnings
  peer-00/final.fstate          final *actual* model weights
  ... peer-01 through peer-07 ...
  report.json                   only on completed evaluation: PASS or FAIL
  failure-report.json           on startup/snapshot/other fatal errors
```

Inspect: `python -m json.tool swarm-runs/run-001/report.json | less`

`report.gossip.per_peer_missing` must be all zeros and `report.gossip.all_equal`
true. `report.model.max_abs` measures real tensor differences, not mismatched
ZIP timestamps. `workers[*].snapshot_received`, `signed_hello_count`,
`max_active`, `max_living`, `loss_first`, `loss_last` and log files provide
communication/progress evidence. A synthetic data loss value is not a
meaningful LM benchmark.

## Other scenarios

```bash
# Three different sudden peer outages; fast local smoke test
python -m patchlib.cli swarm-sim --embedded-signaling --output ./swarm-runs/run-003 \
 --peers 8 --faults 3 --seed 123 --warmup 5 --down-time 3 \
 --between-faults 3 --settle-timeout 60

# Use LoRA rather than LoHA (same transport and test logic)
python -m patchlib.cli swarm-sim --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/lora-001 --kind lora --rank 2
```

For a fast connection check without eight PyTorch workers:

```bash
python -m pytest -q tests/test_swarm_sim.py
```

That smaller test uses **real TCP WebSocket connections and SDP forwarding**
and covers group join/leave/rejoin, journaling, and full weight comparison. It
**does not** create real aiortc PeerConnections or claim a full E2E PASS.

## Known limitations

- This harness **requires** local `aiortc`, `transformers`, PyTorch,
  safetensors, and websockets. It fails clearly if they are missing. The
  built-in signaling server is not a substitute for real WebRTC, TURN, or Mummy.
- Eight concurrent WebRTC processes may consume more RAM than a typical laptop
  can spare. Reduce to `--peers 2` for a first check. Eight peers must be run
  for the requested coverage.
- The isolated scenario tests CPU training and direct local UDP connections;
  hardware-dependent MPS/CUDA and Internet NATs/TURN need separate E2E runs.
- If the real implementation stalls or diverges, this harness reports **FAIL**,
  preserving full logs; it does not forge agreement or silently reset learning.
- This does not automatically test an actual WAN network partition; it
  exercises abrupt process departures and reconnects. A link-failure or
  network-partition scenario is suitable for a later iteration.
