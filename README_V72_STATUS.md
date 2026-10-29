# Kagami v72 — PPO pipeline / auxiliary model memory audit

## What changed

1. **Optional causal KV-cache for PPO rollout (default `auto`).** Inside the existing adapter lock and `torch.no_grad()`, a model whose Python forward *explicitly* declares both `past_key_values` and `use_cache` can prefill the prompt once and decode each subsequent token with a 1-token input plus the returned cache. If no cache is returned, v71's full-context recomputation remains active. A cache that disappears halfway through a rollout fails closed. Models with custom `input_map`, nonstandard or opaque forwards use the old path. The differentiable full forward for the clipped PPO actor update is retained, as are old-policy log-probabilities. CLI opt-out: `--ppo-kv-cache off`.

2. **Bounded Hub SFT/PPO data prefetch.** Hub readers now run on their own daemon reader thread with a two-row bounded queue. Only validated rows are prepared ahead; **no Actor/Critic inference runs asynchronously against a changing policy**. Rows and errors are delivered in order. A 120-second per-row exhaustion timeout prevents indefinite main-worker blocking but cannot forcibly cancel a hung underlying Hub network request. At trainer shutdown the reader is signaled, without a potentially unbounded thread join. Local JSONL remains synchronous.

3. **Reward Model idle CPU offload.** Optional `--reward-model-idle-offload` causes a frozen HF Reward Model assigned to CUDA/MPS to live on CPU when not scoring, temporarily moving to its chosen scoring device for inference, then returning to CPU even if the inference raises. The TRAINING GUI Reward section has a *Memory mode* setting. Model ID, fixed revision, and scoring semantics stay pinned, but hardware memory preferences remain per-peer local. Default is *off* to avoid repeated host/device transfers. The HF Value Critic continues to support CPU residency via its existing device choice (default CPU); **no unsafe GPU migration of a trainable Critic during a live backward graph was attempted**.

4. **Old GUIs and protocols.** The new GUI setting defaults to false when loading v71 data; old room policy and signature formats remain unchanged. PPO cache and hardware offload are local execution optimizations, not a change to the signed objective function.

## Measured local CPU smoke results

See `BENCHMARK_V72.json` and `benchmark_v72.py` for reproducible instructions.

| CPU synthetic case | v71-style | v72 |
|---|---:|---:|
| 24 PPO generated tokens, 17-token prompt, sum of tokens sent to Actor | 708 | 41 |
| Rollout generation median of five runs | 7.594 ms | 6.112 ms |
| 10 buffered network-like SFT/PPO rows with 20ms artificial network and 20ms compute | 0.403 s | 0.221 s |

These are intentionally *toy CPU* demonstrations (72-token vocabulary, 64-dimensional embedding for the Actor); they do not predict wall-clock speedups for HF Transformers on GPU, real Hub streaming, or 300B models. Cache memory grows with context length and model architecture. Inference/offload GPU memory was **not** benchmarked on a physical GPU.

## Validation

* `PYTHONPATH=. python -m pytest -q tests/test_ppo_pipeline_v72.py`: 7/7 passed, covering probability and gradient parity, explicit fallback, bounded queue/error delivery, prompt dataset failure, blocked stream shutdown, timeout, GUI CLI wiring, offload teardown.
* Mixed HF/PT/PPO, GUI, P2P LiveFed, v54-v58 epoch-related regression selection: **122 passed** after the final change. See `PYTEST_V72_FOCUSED.log`.
* `node --check web/desktop_webview.js` succeeded.
* An attempted all-tests run was not completed within the runtime allocation. A detailed run reached **98% without failures** before its timeout; **do not** interpret this as a passed full suite.

## Unresolved

* KV caching is automatically activated only for the explicitly supported forward protocol. Custom Transformers implementations may still require an architecture-specific cache adapter; no user-provided Hub Python code is executed.
* The Actor full-context logits allocation for the *differentiable* PPO update still exists, as does model-wide parameter locking. Generation remains sequential with one token per decoding operation; this is **not** a fully concurrent actor/evaluator distributed rollout service.
* GPU Reward Model CPU staging reduces *resident tensor* VRAM but incurs PCIe/unified-memory transfers and PyTorch's caching allocator may reserve freed blocks. CUDA/MPS performance and OOM handling must be verified on user hardware.
* The HF Critic still requires a trained scalar head. Running it on CPU avoids GPU residency but may be slower. Training-time zero-copy or NVMe paging of an arbitrary frozen HF Critic is not implemented.
* Live Hub request hangs are not forcibly canceled. After 120 seconds without a row the trainer fails closed; the daemon I/O worker may persist until its transport returns.
* Neither 4,096-peer real WAN endurance nor 300B no-pause PPO training was run.
