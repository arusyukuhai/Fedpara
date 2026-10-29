# Kagami v37 — faster training charts and trained-weight EVAL recovery

Based on v36; no Gossip protocol migration and no automatic fallback from trained checkpoints to original weights.

## Training charts

- Status/metrics refresh ~0.9 s while the app is visible, rather than 1.75 s. The status endpoint reads at most 32 local samples and 32 KiB, instead of fetching an entire 320-point chart on every status tick.
- Immediately merges the newest 8 actual telemetry records onto the longer history preview. History re-fetch is throttled to 4.2 s rather than 12 s, independently of live sampling. Expensive long-history queries remain bounded to <=1600 points.
- Loss and Top-1 use independent adaptive Y scales with labelled minimum, maximum and intermediate guides; five X-axis step labels, higher-contrast green traces, light area fill, last-point marker and a hover/touch raw-value cursor. All values are actual measurements.
- The graph toolbar now has **Both / Enlarge Loss / Enlarge Top-1**. Focusing one graph uses the available height without scrolling the desktop window. Display-range selection remains intact.
- Graph rendering is skipped in non-training tabs; metrics and relay continue regardless of active screen.

## Inference reliability

- Verified FP16/BF16 learned checkpoints no longer fail solely because Transformers constructs a default-FP32 shell with `torch.device('meta')`. The in-memory restore first installs parameters in the snapshot's saved floating dtype; it may subsequently cast to the requested inference dtype.
- Mixed-floating-dtype effective checkpoints are dispatched through the verified safetensors/Accelerate path. Converting the whole model to a single dtype *before* installing such a snapshot would lose information.
- If a direct GPU restore runs out of VRAM, EVAL retries with the bounded-memory streaming/Accelerate offload path rather than returning an immediate OOM.
- EVAL snapshot completion now validates the exported manifest's data shape/identity and reports invalid manifests as a **SAVE ERROR** instead of remaining in a pending state forever.
- Failure to return a result JSON no longer masquerades as completed successful inference; the exit code and diagnostic hint are surfaced. Polling errors appear in EVAL instead of being swallowed. GUI evaluation RPC timeout is raised for initial job setup; inference itself runs asynchronously.

### What has / has not been validated

- Targeted PyTorch/GUI/restore tests: `PYTHONPATH=. python -m pytest -q tests/test_kagami_v37.py tests/test_kagami_v36.py tests/test_kagami_v35_eval_errors.py tests/test_kagami_v32.py tests/test_kagami_v34.py tests/test_shards.py`.
- Chromium UI checks for 1024×660 and 1366×768, no uncaught JS errors, graph zoom/hover/grid rendering.
- **A real Transformers + Accelerate runtime is not installed in the test container**. End-to-end learned inference has been tested with real PyTorch tensor export, hash verification, restoration and generation, using a mocked Transformers factory. This does **not** prove compatibility with every live Hugging Face architecture, tokenizer, quantizer or checkpoint.
- Actual slow disk I/O, GPU memory pressure, huge checkpoints and live P2P conditions have not been reproduced. The user's specific error trace is still unavailable and may indicate another model-specific issue.

### Installation

Use the Python interpreter that launches Kagami. Install PyTorch appropriate to your platform, then:

```bash
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-eval.txt
python kagami.py
```

For 4-bit/8-bit quantization, additionally install compatible bitsandbytes or torchao and their GPU/backend dependencies. If trained weights do not yet exist, begin/resume the corresponding model and allow the snapshot to complete before EVAL; *base inference* is a distinct manual selection.
