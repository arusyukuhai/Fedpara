# Validation v58 (2026-10-27)

- `python -m compileall -q patchlib`: PASS.
- Focus: `python -m pytest -q tests/test_epoch_v58_frozen_export.py tests/test_epoch_v57_trainer.py tests/test_durable_io_v12.py`: **18 passed**.
- Broad regression (v58, v57/v56 trainer/quorum/recovery, certified carry, cut audit, local-progress, checkpoint/RCU/shards): **103 passed, 0 failed**, about 6.56 seconds on runner.
- **Full-suite attempted twice but exceeded per-run execution time.** First run at approximately 50–55% including a high-cost large-room topology test; second run omitting that test module reached roughly 61% but also did not finish. Their incomplete logs are not evidence of a successful full-suite run.
- `python -m tools.benchmark_epoch_v58` (synthetic index, not actual P2P/300B):

```json
{
  "origins": 4096,
  "indexed_events": 20480,
  "legacy_fenced_capture_ms": 111.773,
  "v58_factor_fence_ms": 0.262,
  "v58_background_index_ms": 111.152,
  "frozen_factor_bytes": 16896
}
```

New regression tests include: snapshot-exact effective weights despite optimizer-factor mutation; future journal suffix exclusion; incomplete deferred snapshot denial; pinned WAL snapshot cut consistency; missing signed sequence denial; model original in-place mutation denial; independent remote overlay copy; cloned-factor budget rejection; forged prefix mutation rejection; and artificially delayed shard I/O while training and journal keep progressing. Old signed v57 export behavior is covered by prior tests.
