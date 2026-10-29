# Fedpara v9 — self-peer live telemetry

The live-training GUI now renders only **its own process**: local minibatch training loss, local steps/s, cumulative local optimizer steps, local signed events emitted, inbound updates applied, inbound queue depth, and connected data channels. No remote-peer loss, data or telemetry is requested.

The worker writes a private, unique, append-only `.jsonl` file approximately once per second, flushed linewise. File size in each GUI poll is bounded to 160 KiB and the last 320 samples. Charting is pure local SVG in WebView (it uses no charting CDN; Tailwind CSS itself is now CDN-loaded); GUI polling every 1.5 seconds does not touch the training model or journal.

GUI metrics are **training loss**, not independent validation loss, and 0 channel connections may be normal during join. Local steps/s is computed from monotonic clock between samples and is not an estimate of total group throughput. On rapid stop/restart a new telemetry file is created, preserving prior runs on disk.

The CLI may opt in with `train-live ... --telemetry-file /private/path/unique.jsonl`. The path must not exist, to avoid mixing runs. The live P2P protocol and signed update wire format remain unchanged.
