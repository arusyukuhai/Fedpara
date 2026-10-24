# Fedpara — 16 peers × 8 direct neighbors; MONO-01 caret fix

This patch is based on the **Elastic v2 Epoch + MONO-01** source release. It does not enable real V2 training; `swarm-sim` continues to exercise the existing real V1 train-live path, and can additionally run the optional signed *no-op* Epoch v2 probe.

## What changed

- The 16-peer overlay is a symmetric, connected, **8-regular graph: 64 undirected direct links**. The prior (1,2,4,8) stride layout had only seven distinct neighbors at 16 peers because both +/-8 target the same opposite peer. A deterministic parity-based perfect matching (+/-3) fills that gap without increasing the cap. The established 8-peer layout is unchanged.
- `swarm-sim --peers 16 --max-neighbors 8` starts 16 independent real `train-live` workers. `--allow-over-8` remains accepted for older scripts but is no longer required. The launcher forwards `--max-neighbors` to every worker instead of hard-coding 8.
- Final `report.json` now has an `overlay` section recording expected links and topology symmetry plus **observed** maximum authenticated direct neighbors, gathered from training-process logs. `checks.bounded_overlay_verified` fails if an observed sample exceeds eight direct neighbors, no authenticated peer connection was observed for one or more participants, or the graph is invalid. `active` includes the local peer, so the cap is `active <= 9`.
- The `active`/`living` monitor recognizes both `active=... living=...` and `living=... active=...` log orders. Start and settle timeouts have higher defaults suitable for a larger group.
- MONO-01 WebView now explicitly applies a solid black text-insertion caret, black native WebKit text fill, inverse selection, I-beam pointer, and a visible `:focus` outline to text inputs and textareas. Does not force focus or alter the Python bridge.

## Run a genuine 16-peer test on your Mac

Install the existing project's swarm requirements if needed and start the **updated** Mummy server. Open a terminal in the Fedpara root:

```bash
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 \
  --faults 2 --seed 20261022 \
  --max-events-per-peer 256 \
  --startup-timeout 360 --settle-timeout 300 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/16x8-bounded-001
```

If this passes, test the unlimited-load scenario with a **fresh** output folder:

```bash
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 \
  --faults 2 --seed 20261022 \
  --max-events-per-peer 0 \
  --startup-timeout 360 --settle-timeout 360 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/16x8-unlimited-001
```

The test uses **real aiortc WebRTC** in each independent subprocess. The optional `--embedded-signaling` mode substitutes only the local signaling server, not the P2P transport. For the no-op Epoch v2 signature/certificate test, add `--epoch-v2-probe --epoch-v2-timeout 180` on a subsequent dedicated run; this does NOT switch the live learner to v2.

## Inspect `report.json`

Check:
- `status == "PASS"`
- `checks.bounded_overlay_verified == true`
- `overlay.expected_links == 64`, `overlay.expected_max_neighbors == 8`
- `overlay.observed_max_direct_neighbors <= 8`
- `gossip.all_equal == true`
- `model.ok == true`
- `checks.churn_restarts == true`

`max_active`, `observed_max_direct_neighbors`, and `signed_hello_count` are derived from samples and do not prove all 64 WebRTC edges were simultaneously connected. The expected graph calculation is a topology specification, not a packet capture. Verify the worker logs for ICE/DTLS if the test fails. A 16-process test is much heavier than the proven 8-process run; successful 8-peer tests do not guarantee 16-peer convergence.

## GUI testing

```bash
python -m patchlib.cli gui
```

Click the room ID, model ID, dataset search, and other text inputs and check the **blinking black insertion caret**, I-beam pointer, text selection contrast, keyboard Tab navigation, and Japanese IME input. These were covered by static/composed-HTML tests; macOS WKWebView drawing is not accessible in the provided test environment. Do not remove `.fedpara-gui` or existing SQLite journals.

## Validation scope

Python unit/regression tests exercise the 16-node connected/symmetric overlay, 16-client real TCP group join/leave/rejoin, all 64 pair signaling WebSockets/SDP exchanges, worker argument propagation, monitoring parser and GUI CSS. The `aiortc` and `transformers` packages are unavailable here, so full 16-process WebRTC/LLM execution and native macOS caret visibility must be verified on your Mac; neither is claimed as already passed.
