# Kagami v59 — opt-in live disk AdamW + isolated mutable disk Linear

2026-10-27. **Experimental integration, not a 300B or zero-pause certification.**

## What is now connected to the actual elastic `train-live` route

- `--disk-optimizer-dir PATH` opts into `LiveDiskAdamW` in `train-live --mode elastic --optimizer adamw`. No separate full optimizer moment tensors are held in PyTorch `optimizer.state`; first and second FP32 moments reside in per-factor files via `torch.from_file(shared=True)`. For each step, only that factor's moments are staged on the compute device. CPU OS page cache may still consume RAM, and factor *parameters/gradients* continue to reside on their compute device. Does **not** work with LoRA-Muon or synchronous mode.
- At durable working snapshots the optimizer flushes its mmap files and creates independent filesystem copy-on-write clones (`FICLONE` when supported, otherwise a streamed copy). **Never hardlinks live-writable optimizer mmaps.** CPU copies of low-rank factors and cloned moments correspond to the same optimizer/journal fence. The expensive SHA-256 of immutable snapshot moment files runs in the asynchronous snapshot writer, outside the optimizer lock. The safetensors working snapshot only publishes after the moment clones are checksummed. Publication is atomic and fsynced; it references the moment snapshot's unique local folder. Missing, truncated, modified, layout-incompatible or schema-incompatible moment files cause startup to refuse recovery. An obsolete journal tip causes both the old factors and moments to be ignored. The previous snapshot stays intact if the new writer fails before publication.
- Every optimizer reset (including ReLoRA and new-generation shadow preparation) creates **new scratch files**; a restart never interprets dirty/uncommitted files from a previous run as authoritative. Scratch allocations are limited by `--disk-optimizer-max-gb` (default 32 GiB) and available disk; active scratch is deleted on clean shutdown; orphan scratch directories from crashed processes are reclaimed only if their OS file lease is no longer held.
- `--disk-linear-dir PATH` now opts into **live elastic mutable base Linear mmap**, using the existing per-layer/row-tiled frozen-weight autograd and the already bounded FedSum event application. The live model and its generation shadow use different, lock-protected scratch subdirectories, so preparing a shadow cannot truncate a current worker's base files. A per-model cap `--live-disk-linear-max-gb` (default 32 GiB) refuses excess scratch storage. Only supported `nn.Linear` modules are offloaded; other layers/buffers may remain fully resident. GPU cache is invalidated on signed base mutations. Because live FedSum **changes** the effective base, the `--streaming-base` read-only safetensors mapping is intentionally **not** accepted with live `--disk-linear-dir`.
- The standard signed-journal, prior certified generation checks, source event verification and cut/carry/recovery logic remain in force. No unsigned update or uncertified epoch is admitted for the sake of offloading.

## How to opt in

Pass these extra flags to an existing, otherwise valid `train-live` / `train-elastic` command:

```
--mode elastic --optimizer adamw \
--disk-optimizer-dir /fast-nvme/kagami/optimizer \
--disk-optimizer-max-gb 32 \
--disk-linear-dir /fast-nvme/kagami/model-base \
--live-disk-linear-max-gb 32
```

Each command still needs its normal pinned genesis, dataset source, trusted peer identity and journal arguments. Avoid sharing a journal or mutable weight scratch between independent active trainers. `--resume-disk-optimizer` is prohibited in live mode: **only a matching, verified `.working.safetensors` snapshot may restore optimizer moments**.

### Performance and storage tradeoffs

- Disk moment files require ~`8 × adapter parameter count` bytes (two FP32 arrays), in addition to the factors, gradients, model, journal and temporary working snapshots. While a working snapshot is captured, CoW clones can preserve historical pages, and asynchronous fsync/hash consumes I/O. Filesystem reflink unsupported → streamed copy under the writer fence, which can **significantly pause the local worker**; v59 is not strict zero-pause.
- Mutable base Linear offload needs up to one full offloaded copy *per active/shadow generation*. At initialization, the ordinary PyTorch loader may still materialize **the full base once** before moving eligible Linear weights to disk. Thus this is *not* a 300B model loader. For read-only safetensors streaming in existing modes, see v30, but do not mix it with this live mutable base path.
- DiskAdamW accesses at least one whole factor's moment arrays per step, then writes updated moments back. There is no sparse optimizer paging within an individual factor; very wide factors can still exceed available RAM/VRAM. The CPU filesystem cache is intentionally not counted as `optimizer.state` residency. Neither bounded I/O latency nor NVMe bandwidth is guaranteed.
- A failed optimizer step poisons the optimizer; the caller must restart from a journal-matched working checkpoint. Optional intermediate working snapshots may be skipped under backpressure; signed events remain durable. The system is **not** a transactional exactly-once model+optimizer commit at each individual SGD step.

## Remaining work

**Not implemented or proven:** 300B model loader/execution without peak full-model allocation; tensor/expert parallel training over multiple physical peers; strict bounded-memory/latency per *arbitrarily large single tensor*; true zero-pause model/Optimizer checkpoint+switch; end-to-end 4,096 real-device WAN benchmarking; malicious/Byzantine committee resilience; hard finite disk bound combined with indefinite offline recovery. Existing v2 epoch consensus still requires its pinned committee safety assumptions.

See `VALIDATION_V59.md` for reproducible test scope and `BENCHMARK_V59.json` for a local synthetic CPU comparison. Neither constitutes a 300B production validation.
