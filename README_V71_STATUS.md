# Kagami v71 — Reliability and throughput hardening

This version is based on v70. It focuses on four measurable hotspots without changing the signed P2P update protocol, generation certificate validation, or PT/SFT/PPO mathematical objectives.

## 1. Bounded batched Hugging Face PPO value critic

`patchlib/hf_value_critic_v69.py`: replaces one full Critic forward for every rollout prefix with bounded batches of at most 8 prefixes and 2,048 padded tokens. Prefixes are grouped by length to reduce wasted padding. Each prefix's actual nonpadding token is used for token classifiers; sequence classifiers receive attention masks and use their checkpoint's `pad_token_id`. The result order and autograd path to the trained scalar value head are preserved. The backbone remains frozen.

**Compatibility fallback:** Sequence classifiers with no configured `pad_token_id` are evaluated one prefix at a time because some models reject batch sizes greater than 1. This fallback takes priority over speed. Incompatible output shapes and nonfinite predictions still fail closed. The real Hugging Face weights/GPU paths have **not** been tested in this environment.

## 2. PPO actor logit memory

`patchlib/mixed_learning_v65.py`: `_logits(..., start, length)` validates and selects only needed generated-token positions **before** FP32 promotion and finite-value scans. This reduces avoidable temporary memory during sampling (last position) and PPO actor loss (generation window). It does not change the underlying actor model's full logits allocation; architectures that generate a full `[batch,context,vocabulary]` output still pay for that initial output.

## 3. Hub candidate browser and search

`patchlib/hub_discovery_v70.py`, `patchlib/webview_controller.py`: per-controller, thread-safe, 75-second TTL metadata cache with 24-entry cap and in-flight identical-request coalescing. Does not cache weights, dataset rows or private user files. The cache's failures are not persisted. Discovery is still heuristic, not actual checkpoint compatibility verification.

`web/desktop_webview.js`: result tickets prevent a cancelled in-flight selection from mutating form fields when the user closes the picker or switches to another purpose. Concurrent selection is guarded. The PT dataset screen also rejects stale search responses. GUI bridge behavior was checked with real headless Chromium and simulated Hub replies.

## 4. Large-room topology

`patchlib/topology.py`: avoid rebuilding and resorting a large set for peer listings that are *already strictly sorted and unique*; copy those listings into an immutable tuple for cache-safety. Other list/iterator inputs retain canonical sort+dedup semantics. A 4,096-roster test improved from 13.05s to 6.14s in separate local runs (test process wall clock and environment differ; not a WAN network benchmark).

## Tests and benchmark

- All project Python tests: **842 passed, 3 skipped, 0 failed**, in 75.73s. See `VALIDATION_V71.md` and `PYTEST_V71_ALL.log`.
- Chromium GUI scenario: cancelled SFT selection followed by Critic selection, and existing v70 picker scenario, both passed.
- Synthetic CPU value-critic benchmark of 32 prefixes: 32 model forwards -> 4, median 2.1592ms -> 1.9241ms (1.12x); the measured speedup depends strongly on the model/backend. This does not predict GPU speedup.
- Tests added for output-equivalent Critic batching, token classifier indexing, head-only gradients, empty pad-ID fallback, nonfinite checks, pre-FP32 PPO slicing, Hub cache singleflight/expiry/error handling, and sorted/modified topology inputs.

## Remaining limitations

1. Real pretrained Hub Critic/Reward/Actor GPU execution remains unverified, as do end-to-end LM training and WAN churn at 4,096 simultaneous peers.
2. HF streaming datasets can still stall if underlying networking blocks; a bounded asynchronous prefetcher with independent download deadlines is not added in this release.
3. The frozen Critic backbone is loaded and resident according to Transformers; there is no general sharded critic inference or per-layer NVMe offload of that backbone here.
4. PPO actor models may still allocate full-context logits internally; this release only avoids a *second* full-context FP32 tensor.
5. Cache entries may temporarily be stale for up to 75 seconds; user selection must still pin/check revisions before use.
6. No verified performance guarantees for DeepSeek 671B, 300B training, infinite history GC or zero-pause generation rollover.

Security constraints are unchanged: no untrusted Hub `trust_remote_code` enabled, no relaxation of signed journal/certificate checks, and Critic state never enters signed actor P2P patches.
