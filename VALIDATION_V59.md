# v59 test and performance validation — 2026-10-27

## Verification

- A full pytest run after implementing the live offload integration finished with **721 PASSED, 3 SKIPPED, 0 FAILED** in **84.90 s**, after the new integration tests and cleanup/fail-closed guards were included.
- Post-addition, a focused set including live disk optimizer, disk Linear, continual FedSum, v57/v58 epoch export, v52 carry, hot switch and durable journal state completed with **70 PASSED**.
- Nine v59-specific tests passed, including: numeric AdamW parity; immutable moment snapshots under continued training; missing or corrupted optimizer snapshot rejection; journal-tip mismatch; async snapshot/recovery; optimizer format switching; signed FedSum application into separate disk Linear mappings; leased scratch isolation; and a real (offline-neighbor) elastic `train-live` integration with disk AdamW.
- **Do not mistake any simulated or disconnected-neighbor test for 4,096 physical peers or a 300B model.**

## Small local CPU microbenchmark

Run `PYTHONPATH=. python benchmarks/bench_live_disk_v59.py --output BENCHMARK_V59.json` from the project root.

- 4 dense 256×256 FP32 adapter-like matrices = 262,144 trainable parameters, 12 identical-gradient steps; no model forward/backward, epoch switch, signed networking, WebRTC, or disk working checkpoint included.
- Conventional torch AdamW step median: **0.332 ms**.
- Disk AdamW step median: **1.028 ms** (~3.1× slower on this filesystem/CPU workload).
- Maximum absolute parameter difference: **2.38e-7**.
- Conventional AdamW moment tensors: **2,097,168 bytes** counted within PyTorch optimizer state; disk AdamW optimizer state moment tensors: **0 bytes** counted within `optimizer.state`, with **2,097,152 bytes** of moment files. This **excludes OS page cache, file mappings, I/O buffers and factor tensors**.

## Regression caveats

The codebase contains long-running WAN/swarm/integration tests. Report only completed invocations and keep these results distinct from a genuine multi-host endurance test. In particular, no crash fault injection at every filesystem syscall, zero-pause SLA or 300B end-to-end training throughput has been established.
