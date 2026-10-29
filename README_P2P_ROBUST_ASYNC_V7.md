# Fedpara v7: RTC churn hardening + faster SCTP flow control

Baseline: `fedpara_async_mixed_lora_v6_full.zip`. This is a **transport and lifecycle** revision; the signed event protocol, model/LoHa/LoRA layout, room genesis, FedSum, ReLoRA period, and default GUI model stay unchanged. It does NOT provide a guarantee of eventual convergence under permanently disconnected or Byzantine peers.

## What changed

1. **SDP negotiation fencing** (`patchlib/negotiation_guard.py`). Each *new* `RTCPeerConnection` owns a fresh role/state guard. Late/duplicate SDP offers and answers are ignored when no longer in an applicable signaling state, instead of raising and poisoning a healthy link. Early ICE candidates are bounded (96 max) and deduplicated; the cache is bounded even under injected floods. After the remote SDP is set, pending ICE candidates are replayed. Misordered, duplicate, or stale signaling messages never authorize weights; signed HELLO and event verification still apply.
2. **Background retirement of disconnected RTC generations.** `stop_slot()` invalidates `slot.node` and cancels the obsolete task immediately, then tracks a supervised cleanup task. A slow `aiortc.close()` cannot hold up the remaining gossip peers; a new generation waits for cleanup (up to the 15s retirement watchdog). Slot deletion also respects in-flight cleanup. All slots are retired concurrently on process shutdown.
3. **SCTP flow-control fast path** (`patchlib/p2p.py`, `patchlib/shardp2p.py`). The old 15ms bufferedAmount polling sleep is replaced with aiortc `bufferedamountlow` event waiting, and 2ms fallback polling for older / fake transports, retaining the original 256KiB high-water mark and 45s no-progress timeout. Both normal signed updates and snapshot shard transfers share the new waiting mechanism. This is a latency reduction opportunity, **not a measured network speedup**.
4. **Cancellation-safe sends and nonblocking relay.** An interrupted send is opportunistically requeued when room remains, with the authenticated SQLite anti-entropy inventory/need as the durable recovery route. Fanout uses nonblocking per-peer queues rather than suspending the training loop while a queue fills. Saturation is counted rather than treated as an immediate disconnect. Network ingress polling is 3ms rather than 10ms for lower fanout-to-ingest latency.
5. **Failure-injection tests.** SDP answer-before-offer, repeated offers, 100 premature candidates, pause/resume while another participant is joining, SCTP close amid buffer wait, bufferedamountlow listener cleanup, mid-transfer cancel/requeue, and stale slot generations.

## Operation / deployment

```bash
python -m patchlib.cli gui
# or inspect the test suite
PYTHONPATH=. python -m pytest -q tests/test_churn_transport_v7.py tests/test_connection_stability.py tests/test_no_reconnect_storms.py
```

**Migration:** v7 does **not** change model layouts or signed update formats relative to v6. The existing v6 room and journal are compatible on disk, so creating a new room is optional. Still back up any ongoing journals before upgrading a live installation. Restart peers intentionally; mixed v6/v7 transport releases can result in more reconnects during a rolling upgrade.

**Caveats:** The 5-peer real WebRTC/TURN benchmark and true throughput on macOS remain untested here. For meaningful latency numbers compare `bytes/s`, loss/token, inbound queue depth, SCTP bufferedAmount and reconnect counts on the same machine/network. Repeated reconnects can delay updates temporarily; peer journals use signed inventory + frontier repair rather than TCP-level guaranteed seamless delivery. The receiver still serializes model mutations with training to prevent tensor races. No lock-free training or forced peer-level consensus is promised.
6. **GUI stop/start race protection** (`patchlib/gui.py`, `patchlib/webview_controller.py`). Stop is idempotent: repeated clicks do not send repeated SIGTERM. An immediate restart waits briefly for the previous child to release its journal/profile and refuses a duplicate worker if the old one is still winding down. UI status includes `stopping`. It never force-kills an unsealed update to win a restart race.
