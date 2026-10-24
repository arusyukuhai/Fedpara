# Signed Gossip convergence repair — 2026-10-22

## Evidence from the supplied run

- 16 real WebRTC peers and 8-neighbor overlay; initial snapshots and both SIGKILL rejoin checks passed.
- Union: 3,967 distinct signed events; seven peers contained all events, nine missed a combined 343 journal entries. Peer 10 was missing 111, peer 0 missed 78, peer 15 missed 64.
- No conflicting `(origin, sequence)` digests were reported.
- The reporter observed 1,650 aggregate "disconnected" log markers and aiortc teardown exceptions. **Without the original run-*.log tracebacks, their exact root cause cannot be established.**
- After the 300-second quiesce, journal convergence was false and the attached report's max absolute weight divergence was 0.006545722484588623.

## Verified design flaw and modifications

`ElasticNode._attach_channel.on_message` previously placed authenticated `need` controls into a **128-element shared control queue**, where a full queue quietly discarded them. The training loop only serviced the queued requests later, while the sender's ordinary outgoing queue has just three slots. Its retry machinery could recover under some conditions, but a busy or reconnecting swarm could repeatedly lose legitimate requests.

1. On receipt of an authenticated, signed `need`, validate the *whole* request and place the 1–32 event IDs immediately in the bounded durable-response backlog. No model-weight read or optimizer interaction occurs inside the callback.
2. Add `frontier` announcements (signed, max sequence per origin), accepted immediately after authenticated hello. Detect missing older holes *even when the highest local event matches the sender's highest*, scan origins and sequences fairly, and send normal signed `need` requests. Each scan is bounded and uses the journal's indexed SQLite key.
3. Preserve the full digest inventory/anti-equivocation path and existing durable, original-signed event verification. An advertised high-water mark does not attest to intermediate updates or authorize application of model bytes.
4. Advertise `frontier_v1` in the signed hello; only send frontiers to peers that negotiate support. Older v3 peers continue to use the ordinary inventory path.
5. Reduce the high-frequency quiesce inventory rate (0.08 -> 0.25 seconds per page) because compact frontiers now drive focused repair; keep ordinary inventory for digest/conflict checks and mixed-version compatibility.
6. Add a bounded timeout to each signed-event DataChannel transmit, so permanently stalled SCTP buffers trigger the existing authenticated-link reconnect path rather than silently hanging the transmitter forever.
7. Emit `need_rx` and `frontier_scanned` in `SWARM CATCHUP` logs to help locate remaining failures.

## Tests performed in the build environment

- `PYTHONPATH=. python -m pytest -q`: **213 passed, 3 skipped** (includes 7 new frontier regression tests). Updated test count should be confirmed after packaging.
- New tests cover: full control queue + authenticated NEED; malformed/duplicate authenticated frames; internal holes below equal maxima; fairness across 16 origins; frontier delivery despite a full control queue; and deterministic 16-node, degree-8 gossip repair with deliberately dropped frontier notices and no inventory pages.
- **A live 16-process aiortc swarm was not executed here**, because the environment lacks `aiortc` and `transformers`. Thus the fixed archive is a validated candidate, not a claimed macOS E2E PASS.

## Re-run on your Mac

Use a **new** output directory. Keep old run directories and user journals intact.

```bash
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 --faults 2 --seed 20261022 \
  --max-events-per-peer 256 --settle-timeout 300 \
  --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/gossip-16-frontier-004

python scripts/analyze_swarm_connection_failures.py \
  ./swarm-runs/gossip-16-frontier-004
```

If it still fails, inspect `SWARM CATCHUP` (especially `pending`, `demand`, `need_rx`, `frontier_scanned`, `backpressure_drops`) and supply the new `report.json`, `gossip-diagnostic.json`, and `connection-diagnostic.json`, plus redacted `peer-*/run-*.log` excerpts if possible. The aiortc `NoneType.sendto` teardown exception cannot be conclusively attributed without its original traceback and lifecycle context.
