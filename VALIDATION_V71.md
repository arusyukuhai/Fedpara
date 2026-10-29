# Kagami v71 validation — 2026-10-27

## All Python tests

```sh
python -m pytest -q tests
```

Result: `842 passed, 3 skipped in 75.73s (0:01:15)`; the raw output is in `PYTEST_V71_ALL.log`.

The baseline v70 suite and the v71 new tests were run in the active container. Test counts include the v71 regression additions. No failures were observed.

## Browser operation

```sh
python tests/manual_browser_v70.py
python tests/manual_browser_v71.py
node --check web/desktop_webview.js
```

Result: both Chromium Playwright scenarios passed without JavaScript exceptions. The new scenario deliberately keeps an SFT dataset pin request unresolved, closes the picker, opens the Critic picker, resolves the old request, and asserts that no stale SFT value overwrites the GUI.

## New regression tests

```sh
python -m pytest -q tests/test_performance_safety_v71.py
```

Covers numeric agreement against unpadded single-prefix Critic evaluation, per-token output selection, scalar-head gradients with frozen backbone, nonfinite errors, padless fallback, logit subset conversion, same-key cache singleflight and expiry, network-error retry, cache copy isolation, and topology mutation safety.

## Synthetic benchmark

```sh
python benchmark_v71.py
```

CPU only, 32 variable-length prefixes, scalar classifier with two 128-unit hidden transforms; no downloaded HF checkpoints. 30 repetitions/variant, median:

| Variant | Forward calls | Median duration |
|---|---:|---:|
| v70-style independent evaluation | 32 | 2.1592 ms |
| v71 bounded batching | 4 | 1.9241 ms |

Observed ratio: 1.12x. The main benefit for larger real models may be reduced dispatch overhead, but that remains unmeasured.

## 4,096-roster test

`pytest -q tests/test_large_room_bounded_topology_v17.py -k 4096_roster_middle_insertion`:

- Before sorted-list fast path: 13.05s pytest, 15.99s process wall clock.
- After sorted-list fast path: 6.14s pytest, 9.61s process wall clock.
- All 18 tests in `test_large_room_bounded_topology_v17.py`: passed in 10.13s (13.62s process wall clock).

These are sequential local observations, not statistical benchmarks or WAN measurements.

## Limits

The HF Critic unit tests use mocked model/tokenizer weights. The browser tests stub Hub traffic. No current production Hub or GPU measurements were obtained. See `README_V71_STATUS.md`.
