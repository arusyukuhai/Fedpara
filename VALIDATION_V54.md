# Kagami v54 validation (2026-10-27)

## All tests

Run from repository root:

```shell
python -m pytest -q --tb=short
```

**680 passed, 3 skipped in 74.86s** (Python 3.13 environment, localhost
unit/integration tests; no 4,096-peer WAN experiment).

Focused epoch suite:

```shell
python -m pytest -q tests/test_epoch_* tests/test_online_generation_v41.py \
  tests/test_live_generation_transport_v42.py \
  tests/test_v44_low_cost_generation.py \
  tests/test_generation_multicast_v46.py \
  tests/test_compact_epoch_network_v46.py \
  tests/test_live_epoch_switch_v48.py \
  tests/test_continual_elastic.py tests/test_elastic.py
```

**125 passed in 13.69s**.

The five new v54 cases in `tests/test_epoch_v54_hardening.py` cover real
signed LoHA race and failure paths, a disk I/O delayed WebRTC repair callback,
backpressure write amplification, spool checksum damage, and atomic local
view publishing under injected I/O failure.

## Prior failures and interpretation

- v53 full-suite initially had two previous failures: GUI defaults test
  expected absent keys, and one aggregate test transmitted without HELLO.
- First v54 full run: 542 passed, 3 skipped, 4 failed because other tests
  used old unauthenticated channel mocks or defunct benchmark tabs.
- Second run: 674 passed, 3 skipped, 6 failed in similarly stale assumptions.
- Final run: 680 passed, 3 skipped. Test fixtures now establish authenticated
  state when testing message pump behavior; no pre-auth send gate was relaxed.
  UI tests target bundled CSS and current desktop tabs; removed bench inputs
  are tested via the supported backend validation separately.

The 3 skipped tests are environment-dependent; this run is *not* proof of
real-network performance or mathematically complete GC/liveness at scale.
