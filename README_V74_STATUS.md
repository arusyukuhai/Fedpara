# Kagami v74 — mixed-learning lifecycle, fail-closed gradients and HF streaming audit

Based on v73. This release preserves the P2P adapter wire format, room policy fields, on-policy PPO/G​​AE objective, and the existing GUI configuration. No claimed support for arbitrary HF custom kernels or distributed 300B production training.

## Confirmed bugs fixed

1. **Critic head and AdamW moments were not saved at exit in `train`/`train_p2p`/`train_live`.** Only the periodic ten-step checkpoint was written during `run_step`, so a normally completed 1–9-step run could lose its Critic progress. Entry points now finalize the mixed trainer in `finally` (including error exits). The elastic trainer already had a shutdown finalizer and remains unchanged. Completed stats remain readable and an already-closed mixer is rebuilt if a CLI args object is reused.
2. **Partial initialization leaked HF dataset worker threads.** When opening the Reward Model or Critic raised after SFT/PPO `PrefetchedRows` creation, the constructor abandoned the open stream. Initialization now closes partially opened sources, and `close()` is idempotent. A checkpoint save exception still causes independent source and scorer cleanup.
3. **Nonfinite loss / Critic gradients could corrupt Critic optimizer state.** Pretraining and SFT objectives are checked before backward, PPO actor and value losses are checked before backward, and the trainable Critic gradients are verified before AdamW.step. Invalid gradients cause fail-closed rejection, not Critic writes. The adapter's existing optimizer has a separate path and is *not* an atomic transaction with local Critic state; abrupt crash consistency is still limited.

## Performance

4. **Repeated metadata reloading of finite HF streaming splits.** `HFRowCycle` previously re-invoked `load_dataset(...)` whenever a finite split was exhausted. v74 restarts `iter(the_same_streaming_dataset)` for re-iterable sources. One-shot iterators still fall back to reloading with the existing pinned revision. Networked dataset rows are still re-read on each epoch; this optimization saves metadata/initialization calls, not training-data bytes.

## Synthetic benchmark

See `BENCHMARK_V74.json` and `benchmarks/benchmark_v74_hf_reiteration.py`. A two-row fake split is repeated 100 times with 1ms synthetic setup delay per `load_dataset`. Initialization calls: v73-equivalent 100, v74 1; run time 110.05ms versus 0.58ms. This synthetic benchmark is NOT a Hub networking measurement or a realistic throughput speedup estimate.

## Remaining risks / work

- Actor/critic checkpoints do not form one atomic multi-component transaction with the signed event journal. A crash between commits can leave the local Critic at an earlier step than the adapter; this is a substantial correctness limitation to address with a shared commit boundary.
- `asyncio.to_thread` training steps cannot be forcibly canceled and complete OS network requests may outlive caller cancellation; true cancellable bounded upstream I/O is not implemented.
- Optimizer operations and periodic Critic disk fsync still happen under the adapter training lock. For large heads, checkpoint serialization can pause inference or inbound P2P updates. A verified immutable snapshot/async commit is needed to reduce lock hold without compromising state consistency.
- Real HF checkpoint, GPU/CUDA/MPS and WAN stress tests were not available here. The test suite is primarily mocked/tiny PyTorch models.

## How to test

```bash
cd kagami-v74
python -m pytest -q
PYTHONPATH=. python benchmarks/benchmark_v74_hf_reiteration.py
```
