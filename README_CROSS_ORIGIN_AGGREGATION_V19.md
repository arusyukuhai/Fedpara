# Fedpara v19 — signed cross-origin aggregation and adaptive publication

## What changed

- **Different origins are truly added together** in the live model application path: receive a compressed bundle of 2–3 individually signed updates; verify each *original* source event, its immutable genesis, tensor ranges and limits; filter `(origin, sequence)` identities already present in the durable journal; commit fresh originals; sum their effective LoHa/LoRA weight deltas inside one model lock; apply exactly once per constituent event.
- An inbound compressed aggregate is a transport envelope (`FPAG1`) of independently authenticated `FPEL1` originals, **not a relay's unverified assertion of a dense sum**. Each event remains independently recoverable from the journal, can be requested by `NEED`, and can be rechecked during replay. A relay's signed SCTP transfer does not confer authority to modify an origin's event. Checksums of the aggregate detect bit corruption in addition to each origin's signature.
- The sending peer chooses at most 3 ordinary queued events, uses bounded zlib compression in a worker thread, and sends a combined frame **only when it is actually shorter** than the raw events and within configured patch cap; otherwise the existing individual event path is used. Demand-driven `NEED` repair retains strict priority. Incoming decompression is bounded to at most 64 MiB. Both sending and receiving are opt-in through a signed `aggregate_v1` hello capability.
- Inbound partial overlap is safe: a receiver that already has `alice:1` but not `bob:2` only adds `bob:2` from an aggregate containing both. Exact replays do not multiply updates; ID conflicts are rejected. Crash replay reconstructs the same sum from durable individual signed events. FedSum remains unaveraged (`denominator=1`).
- **Adaptive local publication** is enabled by default: with `--local-steps 4`, room sizes 1–19 use 4 steps per event, 20–79 use 8, 80–319 use 16, 320–1279 use 32, and >=1280 use 64 (default `--max-sync-steps 64`). ReLoRA still runs every 20 local **published events**; optimizer/gradient steps therefore increase between resets as peers grow. Peer heartbeat, signed `NEED`, frontier reconciliation and join recovery are never throttled by this option.
- GUI realtime monitor shows effective steps per published event, signed inbound aggregate count and total saved push bytes.

## Settings

```bash
python -m patchlib.cli train-live ... --local-steps 4 --max-sync-steps 64
# Disable scaling if needed:
python -m patchlib.cli train-live ... --fixed-sync --local-steps 4
```

New GUI rooms default to the adaptive policy. The GUI's `local steps` setting is the baseline for about 5 peers. Existing journals and signed event layouts remain valid. Both endpoints must advertise `aggregate_v1` before compression bundles are used; older clients continue receiving normal individual events.

## Exactness, security and bandwidth limitations

There is no way to cryptographically validate a relay's arbitrary **single** dense summed tensor from signatures over different original factor tensors without also furnishing the original signed contributions or adopting a more complex multi-party aggregation proof/attestation protocol. This version intentionally retains the signed originals inside the compressed bundle. Consequently, it reduces **redundant framing, compressible bytes and aggregate application overhead**, but does **not** remove the information-theoretic O(N) dissemination cost of having every peer verify every unique contributor. Incompressible or oversized bundles fall back to single-event gossip. No fixed throughput improvement is guaranteed on real WebRTC or Apple MPS.

The model treats events as additive updates, but floating-point summation order may produce negligible numerical differences from sequential application. It is *not* round-synchronous training and does not remove stale-update convergence concerns.

## Tests

- `tests/test_cross_origin_aggregate_v19.py`: 3-origin exact sum, independent source signature checks, partial overlap/no double sum, journal replay, bounded decompression, sender coalescing, room-size publish table.
- All tests distributed in four batches: **393 passed, 3 skipped** (Python test environment). No real multi-machine aiortc throughput or Nim compiler verification performed in this container. Nim signaling server sources are unchanged from the syntax-corrected v18 baseline.
- Small synthetic three-author aggregate: 17,981 raw signed bytes vs 7,137 compressed bytes (60.31% fewer) in this environment. This is NOT representative of large neural network updates and excludes WebRTC header/transport costs.
