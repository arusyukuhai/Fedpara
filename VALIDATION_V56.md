# v56 validation — 2026-10-27

Run from project root:

```
python -m pytest -q tests/test_epoch_v56_autonomy.py tests/test_epoch_v56_recovery_shards.py
# 7 passed
python -m pytest -q tests/test_epoch* tests/test_generation* tests/test_frontier* tests/test_live_generation* tests/test_live_epoch* tests/test_compact* tests/test_shard_runtime_v27.py --disable-warnings --maxfail=1
# 139 passed (17.80s)
python -m pytest -q tests/test_epoch_v56_autonomy.py tests/test_live_generation_transport_v42.py
# 10 passed after control-plane generator compatibility correction
```

### Added tests
- Real 4096-origin Merkle frontier (128 pages), locally verified source votes; remote committee requests pages through signed bridge messages and assembles an independently authenticated majority certificate. Another peer reconnects and retrieves the certificate. This test is not a 4096-node network stress test.
- Reject malformed page/unknown page index, unendorsed or equivocal candidates.
- Three+ adjacent generation recovery, missing certificate/parent-chain mismatch fail-closed, interruption and retry.
- Durable signed-update quarantine with duplicate, conflicting event and quota tests.
- Immutable-reader/sharded-pointer switch with live writer drain, independent re-verification, fail-closed on bad fence or altered checkpoint.

### Limits
The general 700-ish case test suite did **not** complete in the available execution period. A partial run through approximately the first 30% emitted no failures but was terminated by the tool timeout. No claim that all existing tests or real-world 300B/4096 device benchmarks passed.
