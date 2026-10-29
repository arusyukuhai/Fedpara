# Kagami v55 validation (2026-10-27)

## Verification commands

```bash
python -m pytest -q --tb=short
python -m pytest -q tests/test_epoch_v55_incremental_cut.py tests/test_epoch_live_rebase_v52.py --tb=short
PYTHONPATH=. python tools/benchmark_epoch_cut_v55.py --events 20000 80000 --repeats 20
```

### Focused tests

- 18 passed in 2.27 s (before the final full-suite re-run).
- Checks both original signed torch/LoHA warm-carry path and newly added v55 guard/dedup/poison scenarios.

### Measured local metadata performance

- 20,000 indexed events, 20 origins: first scan 113.1082 ms; repeated whole scan median 115.3180 ms; validated guard median 0.0044 ms.
- 80,000 indexed events, 20 origins: first scan 474.6094 ms; repeated whole scan median 443.3382 ms; validated guard median 0.0046 ms.
- Appending one post-cut origin after validation leaves the witness valid (20k 0.0077 ms / 80k 0.0075 ms).
- A separate micro-case using `sync=NORMAL`, 4,000 sequential 512-byte appends gives v54 ~129.2 ms vs v55 ~147.0 ms (3-repeat median). **The extra SQLite mutation audit has a measurable write-side cost in synthetic low-latency tests.**

Synthetic rows for the cut benchmark contain no signed wire payload; do not interpret this as complete-generation adoption speed or production security proof. The full cut is still hashed on first use and on audit invalidation.

### Full suite

**691 passed, 3 skipped in 76.92 s (0:01:16), failures: 0**. Full local Python 3.13 test suite, not a live 4,096-peer proof.
