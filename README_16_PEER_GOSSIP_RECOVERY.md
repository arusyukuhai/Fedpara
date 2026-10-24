# 16-peer bounded Gossip recovery — 2026-10-22

## Observed on the attached Mac report (not a reproduced E2E result)

- 16 peers; full snapshots and two SIGKILL recoveries: successful.
- 64 expected undirected overlay links; all observed degrees <= 8: successful.
- 3,967 unique signed journal updates. Seven peers had all 3,967; the other nine together missed 343 updates (each up to 111 missing).
- No origin/sequence digest conflicts detected.
- Convergence failed after 300 seconds of quiesce and model max_abs was 0.00654572.
- Across all workers, report recorded 1,650 disconnection log entries and 1,541 signed-Hello successes; those are log-event counts, not exact unique failed transports.
- Most workers also reported aiortc teardown errors involving NoneType.sendto / NoneType.call_exception_handler. The report lacks full stack traces. These errors are not proven to be the primary cause of missing updates.

## Fixed

1. Old `need` handling enqueued at most the remaining slots in the ordinary 3-item broadcast queue and discarded the rest of every 32-ID demand. Now authenticated requests queue bounded **event identities** (max 4096 per neighbor), not event bytes.
2. The transmitter reads on-demand signed bytes from the durable SQLite journal and serves them ahead of speculative fanout, with 8:1 priority/fairness. A failed/dropped request eventually retries from the receiver's `pending` set.
3. Transmit-pump exceptions now mark the link failed so that a channel that silently stopped sending is eligible for a reconnect.
4. WebRTC state-change callback no longer attempts to close the signaling WebSocket during normal `pc.close()` teardown (a potential source of re-entrant teardown), and handles `failed` by notifying the receiving loop instead.
5. `SWARM CATCHUP` now reports `demand` and `demand_sent` to distinguish unknown missing updates from queued responses.
6. `scripts/analyze_swarm_connection_failures.py` makes connection-error summaries and traceback excerpts from the **actual macOS run logs**. No private key or model bytes are transmitted by this utility.

## Important limits

- These changes are regression-tested with fake DataChannels; **16 live aiortc peers were not rerun** in the build environment. There is no claim of real-world PASS or of an aiortc `NoneType.sendto` fix. Capture traceback excerpts with the bundled script if that error persists.
- The V1 Elastic optimizer/LoHA math, Ed25519 authentication, SQLite journal format, and Epoch-v2 probe are not changed.
- Use a new Swarm run directory; old runs are useful evidence but must not be reused as a new scenario. Do not delete persistent `.fedpara-gui`/key/journal records from real Fedpara accounts.

## Re-test

```bash
python -m patchlib.cli swarm-sim \
 --peers 16 --max-neighbors 8 --faults 2 --seed 20261022 \
 --max-events-per-peer 256 --settle-timeout 300 \
 --ws ws://127.0.0.1:8080/ws \
 --output ./swarm-runs/gossip-16-demand-003

python scripts/analyze_swarm_connection_failures.py \
 ./swarm-runs/gossip-16-demand-003
```

If FAIL, provide `report.json`, `gossip-diagnostic.json`, and `connection-diagnostic.json`, optionally full peer logs if safe to share. The `NoneType` error requires complete stack traces to locate the first failing object.
