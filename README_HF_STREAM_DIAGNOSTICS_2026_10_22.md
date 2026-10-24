# TinyStories streaming diagnosis (2026-10-22)

This fix builds on `fedpara_stability_anti_loop_full.zip` and preserves the
previous signed Gossip, bounded WebRTC overlay, journal and recovery semantics.

## What the pasted logs do and do not mean

`HTTP/1.1 302 Found` is a successful redirect to Hugging Face's content CDN.
`HTTP/1.1 206 Partial Content` is a successful byte-range read, as expected
for streaming Parquet. Repeated GETs for the *same* 249MB shard can be normal:
Parquet requires seeking among footer metadata / row groups / column pages.
Neither response proves an infinite loop or an error.

Fedpara uses `load_dataset(..., streaming=True)` and a 512-example shuffle
buffer; the first example can take time to appear while this buffer is filled.
There is also no promise that streaming requests use a persistent offline cache.
Two independent GUI workers stream independently and can compete for bandwidth.

## Changes

- `run_step_offloop()` still fetches and trains in a background thread; now
  emits `DATA WAIT stage=data-fetch seconds=...` every 15s for an unusually
  slow next batch, and `DATA WAIT stage=model-step ...` if the model step is slow.
  The report is a *heartbeat*, not a timeout or an unsuccessful request.
- Does **not** issue a second call to `next(batch_iter)` while the previous
  one is still blocked. Python cannot safely cancel a synchronous HTTP read
  inside `asyncio.to_thread`; concurrent generator access could corrupt the
  sample stream. This is why no automatic per-request retry is added here.
- In train-live, the `httpx` INFO logger is reduced to WARNING to avoid
  confusing normal redirects/range reads with errors, and to avoid displaying
  extremely long signed CDN query URLs in peer logs. Other logs remain enabled.
- The `--hf-shuffle-buffer` flag (2..100000, default 512) allows cold-start
  tradeoffs without changing room manifests or existing defaults. Smaller
  buffers such as 64 may reduce initial prefill, at the cost of less local
  shuffling. The desktop GUIs can opt in via `FEDPARA_HF_SHUFFLE_BUFFER=64`.
- `--data-wait-report-seconds N` controls progress cadence (0 disables).

## Run two GUIs with a smaller HF startup buffer (optional)

```bash
cd /path/to/Fedpara
FEDPARA_HF_SHUFFLE_BUFFER=64 python -m patchlib.cli gui
# Launch the second window in another terminal with the same setting.
```

Do not delete `.fedpara-gui`, your journals or downloaded checkpoints.

If the worker prints `DATA WAIT stage=data-fetch` for a long time without
`DATA BATCH READY` or an `elastic steps` increment, suspect HF streaming,
network throughput or an upstream HTTP stall. If it prints
`DATA WAIT stage=model-step`, suspect local compute / optimizer work.
If neither appears but `channels=0`, investigate P2P separately.

This is an *observability and tuning fix*, not a claim that the HTTP stack,
macOS two-GUI or Hugging Face remote servers were reproduced here.

Tests: 240 passed, 3 skipped in this Linux environment; no live network
requests were made by the test suite. No Nim/server changes.
