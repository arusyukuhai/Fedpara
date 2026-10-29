# Kagami v73 validation report

## Verified

- v73-specific regressions: 6 passed, including a strict HF-like full-prefix attention-mask contract, cached vs uncached actor sampling/logprobs, scalar head optimizer updates with reused frozen Critic features, token-classifier last-token correctness, unsupported post-head transform fallback, and extreme stale PPO rollout rejection (some assertions share tests).
- Relevant functional regressions: 76 passed (GUI mixed learning, HF Critic/Reward, Hub SFT/PPO datasets, discovery UI, v71 audit, v72 prefetch/KV, v73).
- P2P/epoch/trainer/GUI focused suite: 120 passed; after that, an additional token-classifier regression also passed.
- **Broad regression prior to the final feature-budget guard, excluding three very costly existing P2P simulation/topology modules: 840 passed, 3 skipped, 0 failures, 62.05 seconds.** The excluded modules were `test_v49_overlay_scale.py`, `test_p2p_training_sim_v21.py` and `test_p2p_training_peer_sweep_v23.py`. This is NOT a full-suite pass.
- After the final hard 32 MiB pre-allocation cache guard and its regression test, **30 targeted PPO/HF-Critic tests passed**, with no failures. The complete 840-case broad suite has not been rerun after this narrow change.
- Source compiles; see validation log in distribution. Tested with PyTorch 2.10 CPU and pytest 9.
- The complete test suite was attempted but did **not** finish in the tool execution window. An instrumented run passed through 73% before a long P2P training simulation; not counted as a full pass. The broad 840-pass run excludes that simulation and two other heavy modules.

## Synthetic CPU benchmark

`python benchmark_v73.py` measures 32 simulated Critic prefixes using a **toy frozen backbone**. Results are in `BENCHMARK_V73.json`:

- Full frozen-backbone re-evaluation median: ~3.188 ms.
- Cached head-only re-evaluation median: ~0.138 ms.
- Cached feature storage: 32,768 bytes.

This is ONLY the repeated value-inference portion. It excludes Actor, generation, reward computation, network, real Hugging Face models and accelerator behavior. Do not extrapolate to overall PPO step speed.

A separate A/B run of the pre-existing v72 toy rollout benchmark showed cache-path medians of ~7.304 ms on original v72 versus ~6.756 ms in this v73 working environment. Variance and the additional correctness checks make this an illustrative timing only; it does not establish an overall speedup.

## Remaining risks

- Real HF cache implementations vary (dynamic cache positions, flash-attention masks), so model-specific validation remains necessary.
- Inference-only frozen features may grow up to 32 MiB. The model backbone itself is still resident on its selected device.
- HF Hub streaming upstream I/O cannot always be force-cancelled, although v72 limits prefetch wait.
- No live large-model CUDA/MPS performance verification or 4,096-peer churn/long-term disk study.
