# Fedpara 16-peer bootstrap / Snapshot verification fix

## Scope and interpretation

This version includes the **16 peer / 8 neighbors** topology and the **MONO-01 WebView text caret repair**, plus a new **staged startup barrier** for 16-process E2E tests. It is based on the previously supplied Elastic-v2 Epoch + MONO-01 version. The training engine continues to use Elastic v1; an Epoch-v2 probe is optional and does not represent v2 live learning.

The symptom `SimulationError: at least one newcomer could not verify full-model snapshot` means some worker failed to emit `JOIN READY` by the startup deadline. It is *not sufficient to identify the underlying error* without per-worker logs. Previous swarm-sim launched all 15 newcomers at ~1.5-second intervals while earlier joiners were still downloading full weights and existing peers were continuing to train. Every admission changed the bounded-degree overlay and could interrupt a still-live snapshot source connection. Growing journal tips also increased the synchronization load. These are plausible causes, not proof of the particular user's failure.

## Changes

1. For **more than eight** peers, pause optimization at the beginning of the scenario. Full-state transfer and signed WebRTC control/gossip **remain active**. This keeps initial snapshots anchored to a stable source model and signed history cut-off.
2. Join newcomers in **verified waves of four** (16 peers: groups 1–4, 5–8, 9–12, 13–15). The next wave is not started until everyone in the current wave has emitted `SNAPSHOT READY` and `JOIN READY`. Avoid topology edits mid-transfer caused solely by launching the next wave too soon.
3. When all 15 newcomers have verified their snapshots, remove the temporary bootstrap pause and run the usual **16 concurrently training peers**, two SIGKILL/rejoin faults, anti-entropy convergence and model tensor comparison. This is not a fake or mocked WebRTC test.
4. If any wave fails, create `startup-diagnostic.json` with each worker's latest snapshot progress, authenticated-link count, process state, and log path. Exit early if one of the affected processes has terminated.
5. Keep the Epoch-v2 consensus probe from incorrectly treating the initial bootstrap pause as the post-training convergence barrier; it only begins when the explicit quiescence marker is written.
6. Retain previous WebView high-contrast caret/selection/focus CSS and the exact 16-vertex 8-regular overlay.

**Production effect:** the staged admission and temporary pause apply only to `swarm-sim` with >8 peers, or when the user explicitly enables batches. Normal GUI `train-live` rooms are unchanged. For 2–8 peers, default launcher behavior remains unchanged. This is an E2E **harness reliability improvement**, not a guarantee that large production rooms will always synchronize.

## Run 16 peers, 8 direct neighbors

From the Fedpara root, with the updated Mummy server running:

```bash
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 --bootstrap-batch-size 4 \
  --faults 2 --seed 20261022 --max-events-per-peer 256 \
  --startup-timeout 360 --settle-timeout 300 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/16x8-staged-001
```

`--bootstrap-batch-size 0` uses an *automatic* default: four newcomers per wave when peers>8, all newcomers for peers<=8. Setting `--bootstrap-batch-size 2` reduces initial transfer pressure further, at the expense of longer startup. `startup-timeout` applies to **each wave**, not the entire test. Do not reuse an existing non-empty output directory; each scenario stores its own identities and event histories.

Run the unbounded-load comparison only after the bounded test has passed:

```bash
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 --bootstrap-batch-size 4 \
  --faults 2 --seed 20261022 --max-events-per-peer 0 \
  --startup-timeout 360 --settle-timeout 360 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/16x8-staged-unbounded-001
```

Look for `bootstrap-training-paused`, `bootstrap-wave-start`, `bootstrap-wave-complete`, `bootstrap-training-resumed`, `all-joined`, `peer-rejoined`, `gossip-converged`, `overlay-verified` and final `PASS`.

If it fails before `all-joined`, supply both `startup-diagnostic.json` and the affected `peer-XX/run-01.log`. The diagnostic is written even if the scenario raises `SimulationError`. It records no model tensor bytes, but may contain peer identifiers and configuration metadata.

## Validation limits

Python regression tests cover wave gating, stalled wave diagnostics, TCP signaling for 16 peers, the 8-regular topology, and the GUI CSS contract. The native 16-process WebRTC + ML training E2E test **cannot be executed here** because `aiortc` and `transformers` are unavailable in the runtime. The Mac-specific text caret display is also not natively tested here.

Do not delete your existing `.fedpara-gui` data, identities, SQLite journals, or snapshots.
