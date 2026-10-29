# Kagami v34 — Automatic shared adapter rank, square UI, long-span metrics, online audit

## Changes

- **Adapter rank**: Model creation estimates LoHa or LoRA rank using architecture dimensions, available accelerator/system memory, optimizer/gradient reserve and predicted Gossip delta size. Choices: 4, 8, 16, 32, 64. The chosen rank and adapter type are **pinned into the immutable model manifest**; other peers must use the exact wire layout. A local peer whose estimated capacity is below the pinned rank is considered relay-only by the automatic training planner. This does *not* change existing signed model manifests or automatically migrate them. Legacy models without a pinned layout preserve their old behavior.
- **Square UI**: All rounded corners are removed. Japanese body UI prefers **Noto Sans JP**, then Noto Sans CJK JP or Noto Sans. Code remains monospaced. To keep the app offline-capable, there are no external font requests; users should install Noto Sans JP on the OS if not already present. The font binary is *not* distributed with the project.
- **Loss/Top-1 graphs**: Both chart y-axes now show numeric lower and upper bounds plus intermediate ticks. Accuracy scales dynamically around observed values (clamped to [0, 100%]), rather than being hard-coded to 0–100%. The model detail summary shows *only the active model's* most recent locally observed loss and Top-1 accuracy; unmeasured or other-model metrics remain `—`.
- **Selectable chart history**: Shared period selector for both charts: 5 minutes, 30 minutes, 2 hours, 6 hours, 24 hours, 7 days or the entire current training run. A separately exposed, local-only `training_history` route reads a sparse sample from the private append-only JSONL log using bounded seek operations and a maximum plot-point budget (420 by default). It doesn't stream the entire training history to the frontend on each refresh. Longer periods display retained samples from the current run; they cannot show data from before the current log began. The raw per-run telemetry file can grow with prolonged training and needs a future archival/retention policy.
- **Online configuration clarity**: The GUI shows a warning when the signaling URL is loopback-only, or a remote insecure `ws://` URL is used. The bridge being alive is not proof that internet-facing WebRTC connectivity is established.
- **Identity and sizing correction**: An imported network model retains its complete immutable adapter descriptor. Planning cannot accidentally reuse the rank of a different saved model. Presence is displayed as *online membership* rather than interpreting bounded neighbor count as the total population.

## Limits of the audit / security and scaling

This release **does not claim verified internet-scale operation**. Peer discovery and model metadata use a signaling service; signaling is necessary for joining and rejoining peers. Remote instances need the same reachable `wss://` server, valid room credentials/configuration and, for difficult NAT configurations, TURN (coturn or equivalent) with appropriate ports and relay bandwidth. The included deployment samples are not a hosted production service. Local WebRTC unit and simulation tests do not demonstrate success behind carrier NAT, corporate firewalls, or with thousands of live participants.

The adapter rank estimate is a heuristic. Real memory peaks, model-specific kernels, attention KV/activation footprints, optimizer and tokenizer behavior can exceed estimates. A too-small device should relay instead of independently changing the shared adapter rank. Publication steps/context are still auto-estimated as in v33. Large-model attention and MoE support are constrained by actual architecture/runtime implementations.

`NEW MODEL` Python is still a *data-only Transformers Config* subset. It is deliberately **not** arbitrary downloaded Python execution on peers. The learning checkpoint and Eval limitations from v32/v33 also continue to apply. A verified checkpoint is needed for inference on the learned weights.

## Verification on this build

- 35 targeted new/previous Python tests passed (v34 + v33 + local telemetry + v32 + WebView).
- 70 focused network/room/recovery tests passed.
- 23 tests passed in a final targeted regression after the last code tweaks (overlaps previous sets).
- Chromium rendered 1024×660 and 1366×768. All six tabs, square edges, axis labels, range selection, unique model metric IDs and absence of page-level scrolling passed with no JS runtime errors.
- Python `compileall` and Node `--check` passed.
- Did **not** run real multi-site WebRTC, public WSS+TURN deployment, 1000-peer scale, high-memory GPU training or the entire full test suite.

## Launch

```sh
python kagami.py
```

For remote use, start a compatible bounded-v2 signaling server behind a secure public WSS URL, deploy and configure TURN/STUN as appropriate, and set the same endpoint on the participating devices. See `README_ROOM_ID.md`, `deploy/`, and `README_KAGAMI_V33.md`.
