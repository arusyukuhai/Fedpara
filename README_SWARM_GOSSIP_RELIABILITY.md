# Fedpara Swarm Simulator: 8-peer Gossip convergence fix

This update is based on a **real** run report: 8 peers joined, two crashed and rejoined, 5,471 distinct signed update events existed, but each peer was missing 1,117–1,418 events when the final 90-second drain expired. The run also reported unusually frequent connection closures. It failed **gossip convergence**, not initial snapshot verification.

## Changes

- `ElasticNode.retry_missing` now rotates through batches of at most 32 missing event IDs. Previously it always requested `sorted(pending)[:32]`, which could starve all later IDs when the first requests remained unsatisfied. Failed sends do not move the cursor.
- When the incoming control queue is full, **repeatable** controls (heartbeats, inventory, need, hello, hello-ack, snapshot request/status) are allowed to be dropped and recovered instead of tearing down otherwise authenticated SCTP links. A missing signed model TIP is **not** silently ignored: failures of non-repeatable critical controls still fail explicitly. Full received event frames are also recoverable through the durable inventory mechanism. No signature/identity verification is disabled.
- Rebroadcast skips the immediate sender and the original event producer; this reduces redundant copies in a mesh without weakening journal-based repair.
- In the simulator's explicit pause/drain phase, inventory is walked more frequently and missing event IDs are requested at shorter intervals. Existing trainers are unchanged outside the swarm test phase.
- `SWARM CATCHUP` reports the number of durable events, pending event IDs, outgoing queue entries, and observed backpressure drops every ~5 seconds during the drain.
- `gossip-diagnostic.json` is written if event inventories still disagree. It groups missing event counts by original signer, so the next failure is actionable. Report parsing records exception final lines, not only the first `Traceback` header.
- `--max-events-per-peer` provides an **optional** deterministic capacity test; `0` is the default and preserves the unlimited load stress test. A capped test is NOT equivalent to the unlimited stress scenario.

## Commands

Apply the patch to an **unchanged working copy** of the prior `fedpara_swarm_rejoin_fixed.zip` version (or manually merge these files if local edits exist). Do not remove `.fedpara-gui`, SQLite histories, keys, or existing checkpoint files. Run tests from the repository root:

```bash
python -m pytest -q tests/test_swarm_gossip_fairness.py
python -m pytest -q tests
```

First, run a bounded 8-peer recovery/convergence test (no changes to regular train-live):

```bash
python -m patchlib.cli swarm-sim \
  --peers 8 --faults 2 --seed 20261022 \
  --max-events-per-peer 256 \
  --settle-timeout 180 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/gossip-bounded-002
```

Then rerun the **original, unlimited** stress case. Do not treat success in the bounded case as equivalent to success at 5,471+ events:

```bash
python -m patchlib.cli swarm-sim \
  --peers 8 --faults 2 --seed 20261022 \
  --max-events-per-peer 0 \
  --settle-timeout 180 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/gossip-unlimited-002
```

Both tests use **real aiortc/ICE/DTLS/WebRTC** when dependencies are installed and a working signaling server is reachable. Every run requires a *new* output directory.

On failure, compare `report.json`, `gossip-diagnostic.json`, and the lines near any actual exception **at the end of** `peer-XX/run-YY.log`. Exception details were absent from the uploaded summary report and could explain remaining disconnects. If inventories eventually agree but `max_abs` is still above tolerance, investigate order-dependent numeric application and add a canonical signed-event replay barrier (not included in this patch).

## Verified and unverified

The current development environment ran the entire Python test suite: **175 passed, 3 skipped**, including new fairness, backpressure, and diagnostic tests. Real 8-process WebRTC has **not** been rerun here because `aiortc` and `transformers` are absent. The new code improves provable starvation/loss conditions but does NOT guarantee that the macOS stress trial now passes. It does not claim strict mathematical FedAvg or identical float32 sums under different orderings.
