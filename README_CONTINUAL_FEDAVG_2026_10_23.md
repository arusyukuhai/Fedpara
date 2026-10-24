> **Historical v2 documentation.** The active protocol is now FedSum v3; see [`README_FEDSUM_2026_10_23.md`](README_FEDSUM_2026_10_23.md).

# Fedpara: continual elastic averaging (v2, 2026-10-23)

## What changed

The `train-live` / `train-elastic` **elastic** mode now always uses
`fedpara-continual-elastic-v2`.  There is no option to fall back to the old
reset-per-update protocol in live elastic mode. The previously shipped signed
v1 journal bodies are NOT silently imported and old genesis IDs are rejected.
There is no automatic old checkpoint migration.

- Keep each peer's trainable LoHa/LoRA factors and **AdamW first/second moments**
  across local publications and inbound signed updates.
- Maintain a frozen *anchor* of the last published local factors, so the model
  computes `W_global + (D(local) - D(anchor))` between updates.
- Each signed event contains full-precision **before** and **after** factors;
  the committed dense effect is `(D(after) - D(before)) / live_denominator`.
  Never average factor matrices. The update is idempotent under event
  retransmission, and the SQLite journal is still the replay authority.
- A fresh peer installing a verified full-state snapshot resets both factors
  and anchor **together**, and clears optimizer state only at this initial
  bootstrap boundary.
- The offline effectiveness benchmark also uses persistent LoHa and AdamW.
  It synchronizes every peer each round; the *real* gossip network is an
  **asynchronous, denominator-scaled additive protocol**, not strict
  barrier-synchronized mathematical FedAvg.

## Migration (IMPORTANT)

1. Stop all running GUI trainers and all old-version peers.
2. **Back up** `.fedpara-gui/`, `.sqlite3` journals, state snapshots, and
   cryptographic identity keys. Do not delete old training work.
3. Start with a **new room ID** and **new per-peer journal paths** (GUI room
   creation normally allocates these). Old journals intentionally fail with
   `different genesis checkpoint`; do not edit their SQLite metadata.
4. If retaining already learned weights matters, first export an approved,
   verified merged full-model state and use it as the new room's immutable
   genesis. Otherwise a new room trains from its selected pretrained model.
5. All participants must run this v2 code; mixed v1/v2 rooms will fail closed.

Source install: unzip the full archive, then from the `fedpara/` directory:

```bash
python -m patchlib.cli gui
```

No `--continual-fedavg` switch is necessary. For offline validation:

```bash
python -m patchlib.cli benchmark-effectiveness \
  --peers 4 --rounds 40 --local-steps 4 \
  --batch-size 8 --kind loha --rank 8 \
  --output-dir ./benchmark-continual
```

Compare `effectiveness-result.json`, `effectiveness-trace.csv`, and the two
PNGs. The reported `p2p_compute_s_div_peers` is the *idealized compute time*,
not real wall time. The more conservative `p2p_max_peer_s_sum` is also supplied.

## Caveats

- Signed before/after events serialize **both** sets of factors, roughly
  doubling factor payload compared with reset-v1. Repeated signed events are
  still propagated using the existing anti-entropy/inventory flow.
- Concurrent local training and inbound events are serialized at the
  `adapter.lock` boundary. When new remote events arrive during a local
  block, the local block is sealed before the remote event is applied.
- The per-peer denominator is the estimated live membership count; dynamic
  membership / stale bases prevent guarantees of strict synchronous FedAvg.
- A process crash between an event's SQLite commit and the subsequent
  local safetensors checkpoint can recover its committed model effect but may
  lose the freshest AdamW moments; graceful restart checkpoints are retained.
- The CI test suite uses simulated transports for most network tests.
  **Real macOS GUI two-window, 16-peer network and long-running stability
  still require verification on actual hardware.**
