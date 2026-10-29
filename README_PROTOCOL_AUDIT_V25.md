# Fedpara v25: communication protocol audit

This revision changes only the transport/anti-entropy hot path. It does **not**
change the signed event format, journal format, LoHa/LoRA layout, FedSum, or UI.

## Fixes

1. **Streaming SHA-256 for received signed frames.** `FrameReceiver` hashes
   each bounded 16 KiB piece as it arrives. It no longer makes a blocking
   SHA-256 pass across an entire large update in aiortc's synchronous receive
   callback at END. The final digest and length are still checked.
2. **Large outgoing SHA-256 off the event loop.** For transfers >= 1 MiB,
   `transmit()` computes the initial digest using `asyncio.to_thread` so
   heartbeats and remote control are not stalled by whole-message hashing.
3. **Bound inbound pipeline RAM by bytes as well as event count.** The default
   soft cap is 128 MiB of queued/in-flight signed packet bytes, in addition
   to 32 items. One oversized but otherwise accepted packet is permitted
   only when nothing else is in flight. Rejected packets are left in the
   source DataChannel receive queue and retried; they are not acknowledged
   or lost. Note that the upper layer still has its own bounded receive queue.
4. **Prevent repair traffic from starving ordinary Gossip.** The sender's
   fairness condition now accounts for events already removed from the
   outgoing queue and stored in the local `deferred` deque. With persistent
   NEED traffic, those deferred pushes are still scheduled after a bounded
   burst of repairs.
5. **Avoid full signed-history scans while waiting for a snapshot.** A cold
   joiner that lacks journal events no longer recomputes its whole inventory
   every training-loop iteration. The check resumes once its append-only
   journal count or the signed snapshot boundary changes.
6. **Reduce control overhead and limit control inputs.** Control envelope
   SHA-256 is calculated once, and transfer control JSON is capped at 4 KiB;
   boolean frame sizes are rejected.

## Tests

- `tests/test_protocol_audit_v25.py`: malformed controls, incremental hashing,
  send-loop hash offloading, byte backpressure and recovery, repair fairness,
  snapshot catch-up invalidation.
- Full regression test suite includes the existing Gossip, swarm, churn,
  signed events, snapshots, journal and GUI tests.

## Limitations

- This is **not** a measured WebRTC throughput speedup. Large model tensor
  application and fsync are still potentially expensive.
- The full snapshot `snapshot-tip` control still carries an origin cutoff map
  (currently bounded to 256 origins and a 16 KiB control envelope); paging
  this authenticated snapshot boundary is a separate protocol migration
  needed for cold joins after substantially more than 256 active origins.
- Unsolicited traffic is repairable through signed anti-entropy but recovery
  latency still depends on available bandwidth and neighbors.
- Run cross-host, long-duration tests for real ICE/TURN, disk stalls and many
  live participants before treating the protocol as production hardened.
