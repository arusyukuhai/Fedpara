# Fedpara v21 — P2P **training** simulator

This is **not** `scripts/simulate_p2p_v20.py` (which only simulated packets).
`python -m scripts.simulate_p2p_training` performs real PyTorch language-model training with independent models and AdamW states at each peer.

## Main experiment (same settings as the GUI study)

From the repository root, after installing `requirements-patches.txt` and the Hugging Face datasets/transformers dependencies:

```bash
python -m scripts.simulate_p2p_training \
  --model tinystories-1m --dataset tinystories \
  --peers 5 --rounds 200 --local-steps 4 \
  --reinit-every 20 --rank 16 --wide-lora-rank 4 \
  --lr 0.0001 --batch-size 4 --context 64 \
  --sequences 2048 --validation-sequences 256 \
  --uplink-mbps 600 --latency-ms 20 \
  --fanout 2 --bundle-size 3 \
  --eval-every 1 \
  --output-dir benchmark-results/relora_2
```

The **TinyStories-1M architecture is randomly initialized**, never loaded from its pretrained checkpoint. The training and validation examples come from the real TinyStories dataset's disjoint splits. Attention and MLP use **LoHa rank 16**, while Embeddings/output heads/strongly rectangular layers use **LoRA rank 4** via `UniversalAdapter(target_scope='auto')`. Frozen original weights accumulate the FedSum effective deltas. Each peer maintains its own AdamW state and LoHa/LoRA anchors, periodically rebasing and reinitializing its factors. The central baseline consumes **the same batch IDs from active peers**.

To test churn, latency and packet loss, add:

```bash
--churn-fraction 0.4 --loss-probability 0.05 \
--latency-ms 80 --uplink-mbps 50 --repair-every 0.5
```

For a small, **no-download** end-to-end smoke test:

```bash
python -m scripts.simulate_p2p_training \
  --model toy --peers 5 --rounds 15 --local-steps 4 \
  --context 12 --sequences 128 --validation-sequences 64 \
  --rank 16 --wide-lora-rank 4 --lr 0.0001 \
  --eval-every 1 --output-dir benchmark-results/toy-training
```

`--device mps` (macOS) and `--device cuda` are also available, subject to memory and PyTorch support. `--model toy` allows up to 64 peers; TinyStories model allows up to 16 (large memory footprint). `--max-outstanding-mib` stops experiments before unsent dense effective deltas exhaust RAM.

## What is simulated versus real?

**Real:** PyTorch forward/backward, cross-entropy, AdamW optimizer, per-layer mixed LoHa/LoRA factors, periodic merge/reinitialization, deterministic non-IID training partitions, independent local training, per-peer heldout losses and a centralized comparator; FedSum of exactly-once *effective* weight deltas received from multiple origins.

**Simulated:** WebRTC/Gossip packet delivery, finite uplink bandwidth, latency, packet loss, churn, repair/inventory gossip, factor wire sizes, and multi-event bundle compression (compression ratio is an **input assumption**, not measured). Peer training happens sequentially in a single Python process; the ideal parallel time in the JSON result is a **projection**. Peers apply network updates at safe round boundaries, not during an optimizer step. No Ed25519 signing, Nim server, crash durability, RTC handshakes or real GPU sharing is tested here.

### Outputs

- `p2p-training-loss.png`: Validation CE vs **measured synchronized training compute seconds**: centralized uses its actual accumulated compute, P2P uses sum of all peers' compute **divided by `--peers`** (ideal parallel; excludes network and validation). **One point per round by default** plus the initial point.
- `p2p-training-loss-tokens.png`: Separate sample-efficiency figure using the same total training tokens for both arms.
- `p2p-training-trace.csv`: round-by-round training tokens, validation losses, separate central/P2P compute seconds (`central_compute_s`, `p2p_compute_sum_s`, `p2p_compute_div_peers_s`, `p2p_compute_max_peer_s`), update coverage, event count, network estimate and local training loss.
- `p2p-training-result.json`: full configuration, limitations, per-peer heldout loss, loss after final catch-up, completion status, network traffic and optimizer timing.

**Critical:** A successful `all_events_received=true` means every peer has received each event ID, **not** that asynchronous optimization converged to the same solution as centralized training. If updates are delayed, peers trained using stale weights. Results are a comparison of model quality, not a certificate of correctness for the live WebRTC implementation. To avoid benchmarking a partially synchronized run, check `all_events_received`, `outstanding_events`, and `p2p_worst_loss`.

### Comparison caveat

Centralized training applies each active peer's batches to one continuously optimized model. P2P uses separately learned factors, applies remote FedSum updates asynchronously, and persists separate AdamW moments. Both process the **same total training tokens**, but have different update order and optimizer trajectories. Centralized ReLoRA resets every 20 rounds; P2P resets every 20 **local publications**, matching the v19/v20 live protocol and thus not identical reset-step counts when adaptive publication is enabled. This asymmetry is reported explicitly in the JSON.

### Performance safeguards

Dense effective deltas are produced only at publication, and are freed when all peers have received the event; events already delivered once are never re-applied. The simulator enforces a configurable outstanding-delta memory limit instead of consuming RAM unboundedly when network capacity is insufficient. Validation evaluation now runs **each round by default (`--eval-every 1`)**. This is more expensive at 16 peers, especially on TinyStories; for a quick exploratory run, explicitly use `--eval-every 10`. Training-time axes exclude validation/evaluation, serialization, and simulated network time. CUDA/MPS timings synchronize their devices before/after each train step, but the /peers curve is still an idealized estimate (not a measured multi-machine wall-clock speedup).
