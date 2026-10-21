# Fedpara adaptive / MoE / room-descriptor / emergency extension

This extends the preceding room-ID elastic P2P training code.  Existing commands
continue to work. The new local GUI is **not** hosted by Mummy and is bound to
`127.0.0.1`; Mummy still carries only signaling and <=4KiB room descriptors.

## Install and start

```bash
python -m pip install -r requirements-patches.txt
python -m patchlib.cli gui                  # http://127.0.0.1:8765
# start signaling server in another terminal:
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

The GUI can generate a private 192-bit room ID, save the room descriptor,
start/stop the local training process, show local logs and device policy.
It does NOT execute/download the custom model's Python code.

Alternatively create the room descriptor explicitly:

```bash
python -m patchlib.cli room-create
python -m patchlib.cli room-manifest-create \
  --model sshleifer/tiny-gpt2 --revision main \
  --dataset 'consented-text-v1' --dataset-description 'Private per-peer text' \
  --privacy local-only --output room-manifest.json
# If you have custom model code, add --code-file local_factory.py and
# optionally --code-url https://trusted.example/path/to/factory.py
```

Start the first peer:

```bash
python -m patchlib.cli train-live \
  --room-id 'RANDOM_ROOM_FROM_ROOM_CREATE' \
  --room-manifest room-manifest.json \
  --model sshleifer/tiny-gpt2 --revision main \
  --state-base genesis.fstate --text-file train-alice.txt \
  --kind loha --rank 32 --rounds 0 --auto-tune \
  --journal alice.sqlite3
```

Other peers should use the **same** room ID, room manifest bytes/semantics,
initial `.fstate`, model/revision, LoRA/LoHA kind/rank, and (if used) MoE plan.
They may have different private dataset files, different devices and different
batch sizes. Connect to the public WSS endpoint with `--ws wss://HOST/ws`.

## Device-aware tuning

- Auto-detection: CPU thread count, free RAM, accessible CUDA GPU free memory,
  approximate shared-memory MPS capacity; GPU availability is platform-specific.
- Startup auto-picks device (when default CPU was passed), safe heuristic batch
  size, gradient accumulation, activation checkpointing, patch wire dtype,
  and local update cadence. No exact peak-memory guarantee.
- On existing **signed encrypted DataChannels**, only coarse machine tier and
  accelerator presence are gossiped. No precise RAM/VRAM/CPU fingerprint is
  disclosed. A signed but dishonest peer can misreport its tier; remote tiers
  only affect local pacing, never trust or access control.
- The peer changes only update cadence (`local_steps`) at runtime, via
  hysteresis and stale-heartbeat expiry. Batch size, rank, device, effective
  architecture and optimizer state are *not* changed unsafely mid-run.
- Local network DataChannels can supply live telemetry; the server neither
  collects device profiles nor computes global placement.

**Critical protocol constraint:** The current Fedpara elastic patch codec still
requires the SAME kind, rank, alpha, exact parameter layout, and immutable
shared genesis at all nodes. Different per-peer ranks or Expert partitions
**cannot** be mixed by this release. The name FedAvg here means asynchronous
additive scaled updates, not synchronous consensus FedAvg.

## Selective MoE experts (experimental)

Supported: **explicit** MoE expert `torch.nn.ModuleList` and an `nn.Linear`
router whose output last dimension consists of raw expert logits. The pruning
pass leaves a fixed subset and forces the router's non-selected logits to
`torch.finfo(dtype).min`. Removed slots are replaced with fail-closed sentinels.

Example plan (`moe-plan.json`):

```json
[
  {"experts":"model.layers.0.block_sparse_moe.experts",
   "gate":"model.layers.0.block_sparse_moe.gate",
   "count":2,"seed":"SHARED_ROOM_SEED"}
]
```

Pass `--moe-plan moe-plan.json`. Only use paths that actually exist in your
model, with true `ModuleList` experts and a compatible gate. The model loader
checks types, dimensions and tied-parameter safety and otherwise raises rather
than silently using the wrong weights. The plan is hashed into the elastic
genesis fingerprint. ALL nodes must use identical settings to exchange updates.

Important limitations:

1. Pruning currently happens **AFTER loading the full pretrained checkpoint**;
   it reduces *steady-state* memory, not peak initial download/load memory.
2. It changes model predictions, and router behavior may change significantly;
   validate against quality metrics.
3. It does not support packed/fused expert tensors, alternative gating
   implementations, load balancing auxiliary losses, or distributed inference.
4. No automatic optimization or re-training of pruned router thresholds is
   performed. It is an explicit opt-in, not a generic MoE correctness guarantee.
5. Participant-specific Expert subsets require an expert-parallel training
   protocol and sparse cross-partition patch format, which are not implemented.

For massive MoE checkpoints, efficient **preload-time** skipping requires a
model-specific loader that reads selected expert tensors from safetensors
shards (or a trusted expert-backend like a model's native expert-parallel
support). We do not claim that feature here.

## Immutable room descriptor / custom code

`--room-manifest room-manifest.json` is announced on `group-join`. The Mummy
server stores **at most 4KiB** of structured, non-executable model/dataset/code
metadata per active room, including a SHA-256 for optional Python source and
an optional pinned HTTPS URL. It does not store the code, weights or datasets.
Subsequent participants with a different descriptor are rejected by the
server; clients also pin/validate it. When the final member leaves, the server
forgets the descriptor. This is **first-writer** semantics, not strong room
ownership or a durable registry. Persist/pin the descriptor out of band.

The room only contains dataset **descriptions**, not private datasets. If you
specify a local `.py` code file, a SHA-256 pins the code. A user can verify a
local code file using `patchlib.room_manifest.verify_code`. Never execute
remote-received custom code in the P2P network process; use
`sandbox/run_untrusted.sh` and the existing offline build path. A hash alone
cannot establish authorship or trust, and URLs are not auto-fetched.

## Signaling outage / emergency mode

When the lobby is unreachable, it retains last-known membership and tries
`--ws-backup wss://BACKUP/ws` as an optional, **independent** signaling server.
Existing WebRTC DataChannels are kept open after the signaling WebSocket dies.
Authenticated, recently active DataChannel neighbors determine the effective
peer denominator during outage; if none survive, local learning continues
alone. Updates remain in the durable SQLite journal for catch-up.

**Connectivity limitation:** An entirely disconnected new node cannot establish
brand-new NAT-traversing WebRTC sessions without any rendezvous path. Backup
signaling endpoints help if another server remains up. If TURN was required and
its relay disappears, the associated DataChannel can also die. Network
partitions cause differing models until anti-entropy catches up, and there is
no consensus safety guarantee.

## DDoS defenses

Mummy limits total sockets, group sizes, message bytes, and per-socket
packet/byte rates. See `deploy/nginx-security.conf` for TLS/WSS origin-side
per-IP connection and handshake limits and `deploy/compose.yml` for coturn
quotas. These are **application-layer defenses**, not volumetric mitigation.
Deploy behind a managed upstream provider, firewall, edge L3/L4 filters, and
restrict access to the Mummy origin so an attacker cannot bypass the proxy.
Set `ALLOWED_ORIGIN` where browser access should be restricted. Never rely on
room IDs as a perfect authorization system; Sybil/poisoning resistance needs
separate admission rules.

## Extra lightening paths (proposed, not implemented)

- 4-bit or 8-bit frozen base weights (QLoRA-compatible loader); the current
  full-weight code does not automatically quantize every arbitrary model.
- Safetensors expert-by-expert *preload*, eliminating peak-memory costs.
- CPU/NVMe expert offloading plus a cache and hot-expert replication.
- Expert-parallel routing by placement/capacity (requires remote activations,
  possibly privacy-sensitive, and a different protocol).
- Smaller activation context/batch, checkpointing and selective-layer updates.
- Elastic *heterogeneous-rank* overlays represented in a common canonical
  dense/sparse delta schema before averaging.
- Periodic quiescent checkpoint/epoch transition to compact old event logs;
  current history still has a hard bounded cap and no automatic epoch reset.

## Testing

`PYTHONPATH=. pytest -q` covers adaptation, signed-control compatibility,
room descriptor pinning, model code hash, MoE toy expert pruning, and existing
patch/shard/elastic tests. Nim compilation, actual RTC/TURN failover, mixed
hardware, model-specific MoE checkpoints and distributed DDoS need separate
integration/load testing in the deployment environment.
