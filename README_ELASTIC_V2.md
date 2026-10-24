# Fedpara Elastic v2 (experimental aggregation engine) + Room Directory UX

This release adds an independently testable *epoch-anchored* aggregator and two
comparison workflows. **It does not silently replace the live Elastic v1
trainer.** A real decentralized v2 must agree on checkpoint epochs, distribute
signed cohort closures, and train every local update against its pinned base.
Those distributed epoch-commit operations are **not** in this release. All
existing `train-live`, `swarm-sim`, P2P transfers, historical signed journals,
state snapshots and Mummy APIs remain unchanged unless opt-in analysis is used.

## 1. Algorithms

For proposals from an agreed checkpoint `W_epoch`:

- Every update includes `origin`, monotonically increasing `seq`, the 64-byte
  checkpoint hash `epoch`, sample count, model schema and a verified digest.
- The caller must verify Ed25519 before aggregation. The helper
  `pack_epoch_proposal` / `ingest_signed_epoch_proposal` reuses the existing
  signed data-only event envelope; unsigned or malformed input is rejected.
- At most K smallest sequence IDs of a given origin are selected. This
  selection is independent of arrival order. In a production implementation,
  peer epoch proposals must only be committed once the chosen set is agreed.
- Each participant contributes its **sample-weighted mean update**. The
  participants are combined with equal weight:

  `W_new = W_epoch + eta * sum_peer(sum_update(samples*delta)/sum_samples) / (lambda + number_of_peers)`

- Identical verified proposal *sets* produce deterministic CPU aggregation.
  This is not a guarantee of matching future proposals, convergence, or
  Byzantine tolerance.
- Stale epoch proposals, equivocation, nonfinite tensors, inconsistent
  model layouts and oversized event/model buffers are rejected.
- EpochBuffer is currently deliberately limited to 2 million floating-point
  parameters, to avoid an unbounded dense expansion of LoHA factors. It is a
  research/reference implementation, **not a 300B-scale live aggregator**.

## 2. Compare optimizers from scratch (CPU-only; no network)

Run with no Hugging Face downloads:

```bash
python -m patchlib.cli elastic-v2-benchmark \
  --peers 8 --epochs 24 --seed 20261022 \
  --output ./swarm-runs/v2-ab-synthetic.json
```

The A/B trains a *small regression model* separately with the same peer-specific
synthetic examples, local steps, seeds and deterministic dropout/rejoins. The
first arm uses the old `delta / online` semantics, and the second buffers
proposals against a frozen base and applies the V2 cohort mean. It outputs
per-epoch held-out MSE, dropout counts and optimizer times. **These tests are
mathematical simulations, not real WebRTC, LLM perplexity, or a proof that V2
outperforms V1 in general**. CPU timings also exclude network time.

## 3. Analyze a real 8-peer Swarm log (counterfactual)

You can reuse an already completed `swarm-sim` run, including one with real
WebRTC, without rerunning expensive peer training:

```bash
python -m patchlib.cli elastic-v2-compare \
  --run ./swarm-runs/gossip-unlimited-002 \
  --cap-per-peer 8
```

Output: `elastic-v2-counterfactual.json` (V1 final held-out loss, V2 formula
held-out loss on a capped selection, cohort hash, warning). Ed25519 signatures
on the selected archived events are verified. Requires the original run's
`fixture/model`, `fixture/genesis.fstate`, `peer-00/final.fstate`, and
`peer-00/events.sqlite3` plus `transformers` and `safetensors`.

**Important:** V1's updates were actually trained against constantly changing
local bases, rather than a shared epoch checkpoint. Thus, replaying them with
V2's equation is an instructive *counterfactual* rather than a fair training
A/B comparison. It must never be reported as "V2 P2P training succeeded".
This replay uses the test fixture's synthetic held-out tokens and not a
published language-model benchmark. Only the small synthetic Swarm fixture is
currently supported.

To request it during the existing 8-peer end-to-end experiment:

```bash
python -m patchlib.cli swarm-sim \
  --peers 8 --faults 2 --seed 20261022 \
  --max-events-per-peer 256 --settle-timeout 180 \
  --v2-analysis --v2-cap-per-peer 8 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/gossip-v2-003
```

The normal Swarm PASS/FAIL still depends on **real** aiortc connectivity,
churn/recovery, signed-event convergence and matching final weights. An optional
counterfactual error cannot turn a valid V1 PASS into a FAIL, but will be shown
as `elastic_v2_counterfactual.status=ERROR` for investigation.

## 4. Desktop WebView: room directory improvements

The existing `python -m patchlib.cli gui` interface still uses pywebview and
opens **no additional HTTP server**. Room-directory UI now supports:

- search across model names, room IDs and Hugging Face dataset repositories;
- sort by online count, remaining seats or model name;
- display only rooms with capacity;
- per-room model, data sources, occupancy and abbreviated ID;
- select room ID or directly fetch its verified model/dataset manifest;
- preserve the last list if refreshing fails;
- all filtering is local; the Mummy directory API is queried only when the
  Refresh button is used.

The room ID of a public room is a bearer-like secret. **Do not publish private
rooms**. Manifest fingerprints still need independent confirmation by the room
creator. User-supplied room metadata is inserted as `textContent`, never HTML.

## 5. Tests and unimplemented production pieces

```bash
python -m pytest -q tests/test_elastic_v2.py \
  tests/test_elastic_v2_integration.py tests/test_elastic_v2_benchmark.py
node --check web/desktop_webview.js
node tests/webview_room_directory.js
python -m pytest -q
```

A genuinely live V2 implementation requires, in order:

1. shared checkpoint epoch identifiers and reliable signed epoch-start/closure;
2. frozen-epoch local optimization, with stale updates deferred or rejected;
3. conflict handling during partition, leaderless checkpoint agreement, and
   durable epoch seals in the existing SQLite journal;
4. epoch-aware snapshot serving and replay, without treating new V1 epochs as
   compatible with old V2 events;
5. a new **real** eight-process WebRTC Swarm test that trains using V2, then
   compares quality and wall time with V1 under matched budgets.

These items are deliberately not claimed as finished. Do not change the
`train-live` engine without all of them: silently mixing protocol versions
would corrupt distributed model state.
