# Kagami v60 — validation report

The model uses the Python 3.13 / PyTorch 2.10 CPU environment present here;
`transformers` is not installed. The true Hugging Face implementation was
simulated in one CLI-loader integration test via a fake module factory; the
actual streaming loader, live adapter, disk kernel, optimizer and journal were
NOT mocked. Therefore this does not verify real HF architectures.

## Results

| Execution | Result |
|---|---|
| New tests plus disk offload | 23 passed |
| 45 selected modules (epoch, live generation, signed gossip, checkpoint, sharded, streaming) | **285 passed** in 18.40 s |
| Full test suite, 731 collected | **Incomplete**: first run stopped in the ~50% range at the tool timeout; retry also did not complete in the allowed window |
| `python -m compileall -q patchlib` | pass |
| Synthetic 4-layer boot benchmark | complete; see `BENCHMARK_V60.json` |

## Explicit new coverage

- No mutable disk clone on launch; first signed LoHA update allocates only
  touched weight tensors and leaves safetensors bytes unchanged.
- Duplicate signed update does not modify weight again, and replay on a fresh
  model recovers the effective base.
- Disk quota refusal does not corrupt immutable source or mutate pointers.
- Isolated writable scratch per generation/model, with matching kernel source
  remapping.
- Very wide tensor row no longer violates bounded stream chunking.
- Incorrect CLI flag combinations fail closed.
- GPT-2 Conv1D forward and backward match regular PyTorch outputs and
  LoHA factor/input gradients; Conv1D signed update targets writable storage.
- Actual `load_model(...,use_adapter=True)` meta-streaming opt-in path is
  exercised with a stand-in HF factory.
- `posix_fallocate` ENOSPC fault injection fails before publishing a mapping.

## Reproduction

```bash
PYTHONPATH=. python -m pytest -q $(cat tests-selection-v60.txt)
PYTHONPATH=. python benchmarks/bench_v60_lazy_boot.py --output BENCHMARK_V60.json
```

Re-check v60 against production HF architecture and real NVMe/WAN conditions
before deploying to more than controlled test peers. No 300B training or
multi-thousand-device soak test was performed.
