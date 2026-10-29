# Fedpara v20: deterministic network simulation + signed aggregate performance fixes

## Run immediately (no Internet or GPU needed)

```bash
python scripts/simulate_p2p_v20.py \
  --peers 20 --events-per-peer 6 --duration 16 \
  --uplink-mbps 100 --latency-ms 20 \
  --event-kib 1024 --churn-fraction 0.1 \
  --loss-probability 0.01 --compare \
  --output-dir network-results/test-20
```

The simulation uses **only the Python standard library**. When installed, matplotlib additionally generates `*-coverage.png`. It always writes `*-timeseries.csv` and `*-summary.json`.

```bash
# Higher-load network-only 80-peer scenario:
python scripts/simulate_p2p_v20.py \
  --peers 80 --duration 12 --events-per-peer 3 \
  --event-kib 256 --uplink-mbps 600 \
  --churn-fraction 0.2 --loss-probability 0.03 --compare \
  --output-dir network-results/test-80

# Unit/regression tests:
python -m pytest -q tests/test_scale_sim_and_batch_journal_v20.py \
  tests/test_cross_origin_aggregate_v19.py \
  tests/test_binary_journal_v14.py \
  tests/test_swarm_gossip_fairness.py
```

The `--compare` scenarios are deliberately separated:
- `flood-8-fixed`: 8-neighbor immediate push; unbundled events, fixed original 4-step publication (heavy/flood baseline, **v17-like**, not v18).
- `sparse-raw-same-cadence`: 2-neighbor immediate push, same *adaptive* event production as the new scenario, individual raw events.
- `adaptive-aggregate`: same sparse/adaptive cadence, up to 3 distinct original updates per transport packet, configurable compression-ratio assumption.

The middle-vs-last comparison isolates bundling; first-vs-middle **does not isolate** bundling because both fanout and publication frequency differ. Compare `sent_mib`, `repair_mib`, `coverage`, `exact_fedsum_peers`, `dropped_packets`, `p95_arrival_lag_s` together. Events carry exactly additive **integer** updates for correctness checks, not learned gradients.

## Model assumptions (do not confuse simulated with real network measurements)

- Each peer stores *all* accepted signed event IDs and applies every identity once. Virtual payloads use `--event-kib`, no large model tensors are allocated. Link upload contention, configurable bandwidth/latency/drop, bounded 8-neighbor ring, push fanout, delayed adaptive publication and two temporary outage windows on a random fraction of peers are modeled.
- `NEED` reconciliation is optimistic: the simulator can compare neighbors' durable event-ID sets directly. Real network state requires signed inventory/frontier/NEED traffic, so real convergence can be slower and repair can cost more bytes. One fixed small control overhead per simulated repair frame is modeled. No SDP/ICE/TURN, crypto CPU, actual Python serialization, GPU contention, true queue sizes, Byzantine peers, or real model perplexity is modeled.
- Compression ratio is a **user-supplied assumption** (`--compression-ratio 0.75`), not a measured ratio. The real v19/v20 sender only sends aggregate frames if compression genuinely saves bytes and fits the wire cap.
- No claimed loss convergence is valid: `exact_fedsum_peers == peers` means identity-deduplicated integer delta sums converge, **not** that a neural net trained better.
- The simulator stores replica knowledge in RAM and is meant for moderate experiments. A 4,096-peer full replica with multiple events per peer can require very large RAM and simulation time: total distinct event copies can scale as `O(peers² × events_per_peer)`. Use lower `--events-per-peer` or a machine with plenty of memory; don't mistake node-degree scaling for update dissemination scaling.

## Live code changes

1. **One durable journal commit for a verified cross-origin bundle**: 2–8 original signed `FPEL1` events are appended to `.fplog`, one `fsync` is issued in FULL mode, and one SQLite transaction commits the individual indexes. Already seen IDs remain duplicates; conflicting IDs reject the entire batch; a crash between logging and committing loses no *indexed* event and orphan log tail is reclaimed. The signature/tensor checks still happen per original author.
2. **Fix redundant NEED requests**: an in-flight event ID is requested at most once per 6 s unless a send failed, and all still-missing IDs rotate fairly. A brand-new RTC transport generation clears old in-flight markers so recovery can start immediately.
3. **Coalesce signed NEED repair responses**: up to three separately authenticated originals may use the same compressed aggregate as ordinary Gossip. Oversize/incompressible aggregates gracefully fall back to individual events and requeue remaining requested IDs.
4. **Off-loop durable event loads**: `EventJournal.get_async` uses its own serial SQLite worker, not the aiortc event loop, when serving expensive historical repairs.

### Compatibility and durability

No change to signed `FPEL1`/`FPAG1` layouts, genesis, model architecture, ReLoRA, FedSum or SQLite/.fplog on-disk schema. v19 authenticated peers can still exchange *individual* signed events. FULL mode still fsyncs payload frames **before** their indexes become durable. As before, the NORMAL mode can lose recent events during an OS/power crash.

For actual training-loss simulations, use the existing `benchmark-effectiveness` command (with the chosen dataset and model), and for real multi-process transport stress use the existing `swarm-sim` command. Those are separate from the quick, pure-Python network model introduced here. Results from this script must not be presented as real WebRTC/GPU benchmarks.
