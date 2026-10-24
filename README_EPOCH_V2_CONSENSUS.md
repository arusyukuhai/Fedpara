# Fedpara Epoch v2 — signed crash-fault consensus probe

This release introduces an **experimental, opt-in** crash-fault-tolerant, fixed-roster *single-decree Paxos* agreement primitive. It is wired to the existing authenticated WebRTC control channel for a no-op 8-peer end-to-end probe. **It does not replace live V1 optimization, commit V2 model tensors, or prove live V2 convergence.**

## Scope and security model

- Immutable committee of 2–16 peers in the live probe. Majority certificate: 5 of 8 for an eight-peer cohort. At most 3 crashes can be tolerated while preserving quorum availability. Other ensemble sizes require a new test.
- Every prepare / promise / accept / accepted / commit is Ed25519 signed; peers verify sender, epoch ID, base-model semantic hash, roster hash, ballot and quorum of ACCEPTED proofs.
- Promise/accepted/committed records and signed envelopes are persisted with SQLite WAL and synchronous=FULL *before* the dependent reply. Retained signed envelopes can be retransmitted after a disconnect without impersonating their original signer.
- A new ballot cannot adopt an arbitrary new value if a quorum already reported an accepted value: the proposer carries the highest prior accepted value forward.
- A separated minority cannot certify a new epoch; its local training may continue only under the existing V1 path. Do not treat this no-op certification as an instruction to mutate the model.
- Assumes crash-stop / recover, authenticated non-Byzantine peers, pinned stable roster and persistent identity keys. Key theft, Byzantine/equivocating quorums, membership reconfiguration, rolling upgrades, real snapshot replication and multi-epoch V2 training are **not** addressed.

## Test with the real eight-process Swarm harness

```bash
python -m patchlib.cli swarm-sim \
  --peers 8 --faults 2 --seed 20261022 \
  --max-events-per-peer 256 --settle-timeout 180 \
  --epoch-v2-probe --epoch-v2-timeout 120 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/epoch-probe-001
```

This runs the existing **V1** P2P training and random churn, settles signed V1 Gossip, then reaches a **no-op epoch vote** about a fixed base and committee. Look for `EPOCH V2 PROBE ARMED` and `EPOCH V2 CERTIFIED` in each peer log, `epoch-v2-certified` in console, and `epoch_v2_probe.certified = true` and identical `epoch_v2_probe.digests` in `report.json`.

For an isolated protocol-only test that does *not* require eight GPU/WebRTC workers:

```bash
PYTHONPATH=. python -m pytest -q tests/test_epoch_consensus.py
```

Protocol tests simulate eight distinct Ed25519 identities, packet exchange, two crashed peers, recovery/anti-entropy, a four-peer partition, coordinator failover, signed witness tampering, immutable update manifest sorting and crash-safe retained certificate reuse. These are **simulations** of message delivery, not a measured live aiortc result. Real E2E run must be performed in an environment with `aiortc`, `transformers`, `torch` and the signaling server.

## Next engineering milestone

1. Pin a verified `W_e` semantic hash and close an exact signed *V2 update* cohort; reject changes after a vote has been proposed.
2. Train local V2 patches **against exactly that checkpoint**, not V1's continuously changing bases. Make update descriptors and content hashes match actual verifiable LoHA payloads.
3. Aggregate certified updates deterministically; atomically persist the resulting checkpoint with rollback on failure.
4. Build an epoch-by-epoch recovery protocol, durable membership changes and a genuine 8-peer V1-vs-V2 WebRTC quality benchmark.

The no-op probe is intentionally separate from the model update path until all of the above exist.

## Appearance

The desktop WebView now follows the permanent **Fedpara MONO-01** design guide (`DESIGN_SYSTEM.md`): black and white, Chicago-style display headings, Courier-style interface typography, neutral greys, accessible focus cues and no remote fonts. Legacy `web/dashboard.html` was also restyled. The GUI bridge and signaling APIs are unchanged.
