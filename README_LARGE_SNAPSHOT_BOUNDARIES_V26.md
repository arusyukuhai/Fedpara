# Fedpara v26 — paged signed snapshot cutoffs

For an authenticated cold join, the serving peer sends a signed snapshot boundary,
including its per-origin journal high-water marks. In v25 this was a single 16 KiB
control frame, and `snapshot_inventory` rejected more than 256 origins. Larger
rooms could not complete snapshot installation.

v26 adds `snapshot-tip-pages-v1` capability to the authenticated Hello. Large
boundary vectors are sent as `snapshot-tip-page` controls, each separately
signed, with at most **48 origins per page** and **16 KiB per control frame**.
The complete ordered vector, manifest SHA-256, journal inventory SHA-256,
event count, and request ID are committed by a second SHA-256 digest.

- Up to **16,384 origin IDs** in one signed boundary (covers the configured
  4,096-member room plus some churn history).
- A partially transferred boundary cannot authorize installation.
- Page digest mismatch, inconsistent header, duplicate origins, invalid
  sequences and conflicting duplicates are rejected.
- Transfer state is cleared when the RTC control DataChannel generation
  changes. Pages from earlier request IDs are ignored.
- The sender retries the same page on SCTP backpressure. It sends the model
  manifest only *after* all signed pages were enqueued to the control channel.
- Peers that do not advertise this capability cannot accept large boundaries;
  upgrade both applications before a large-room cold join. A small inline
  v25-compatible boundary still works with older nodes.

## Larger model files

The actual model-state transport remains content-addressed, sharded and
resumable. The default state budget is **32 GiB** as a safety limit, NOT a
fundamental restriction of the new boundary protocol. It can be raised, e.g.

```bash
python -m patchlib.cli gui
# In your live-training CLI invocation, for example:
#   --max-state-gb 1024 --shard-mb 32
```

This requests a 1 TiB state byte budget and 32 MiB content-addressed shards.
The snapshot file still needs sufficient local disk, and the machine must be
able to instantiate its assigned model weights and effective adapter matrices.
Current shard transfer has bounded limits (up to 750,000 shards, 96 MiB manifest,
200,000 tensor entries), so **unbounded model size is not guaranteed**. This
is not expert-parallel or streamed model instantiation: a client without enough
RAM/VRAM to hold its model allocation will still fail. Snapshot export may
also allocate a full effective dense tensor for a parametrized layer.

Run the focused tests with:

```bash
python -m pytest -q tests/test_snapshot_boundary_paging_v26.py
```
