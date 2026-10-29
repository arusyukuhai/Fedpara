# Kagami v53 — test record (2026-10-27)

## Focused regression suite

```
python -m pytest -q tests/test_epoch_* \
  tests/test_online_generation_v41.py \
  tests/test_live_generation_transport_v42.py \
  tests/test_v44_low_cost_generation.py \
  tests/test_generation_multicast_v46.py \
  tests/test_compact_epoch_network_v46.py \
  tests/test_live_epoch_switch_v48.py \
  tests/test_continual_elastic.py tests/test_elastic.py
```
**120 passed in 12.25 seconds**. This includes all added v53 tests,
new deferred hot-swap test and prior lifecycle, certificate, epoch bootstrap,
network isolation, copy-on-stage and streaming replay checks.

`python -m pytest -q tests/test_epoch_carry_gossip_v53.py`:
**5 passed**, including a signed event relayed new→new after the source left.

## Full-suite limitations

Attempted unfiltered `python -m pytest -q`; stopped after a 200s timeout
(~42% of collected tests had been started) with two failures seen before
that point:

```
tests/test_async_mixed_v6.py::test_gui_live_real_command_async_defaults
  KeyError: 'local_steps'
tests/test_cross_origin_aggregate_v19.py::test_outbound_aggregates_original_signed_frames
  AssertionError: expected imports but got []
```

**Both failures were reproduced unchanged in a clean extracted original v52
archive**, under the same environment. They are pre-existing test issues,
not newly passing nor attributable to v53, and this does not constitute a
full-suite pass. The remainder of the full suite was NOT exhaustively run.

## Unproven

- 4,096 concurrently training real peers under churn and partitions
- a certified 4,096-origin epoch cut, automatic checkpoint quorum consensus
- multi-epoch offline repair and data retention bounded for arbitrary time
- guaranteed seamless zero-pause GPU or optimizer replacement at 300B scale

These are separate acceptance tests before a production-ready claim.
