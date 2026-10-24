# TinyStories-1M benchmark: larger training and validation sets

The default dataset *pool* was too small (64 examples per peer; at most 128
validation examples). This update changes the TinyStories-1M **benchmark** to
2,048 distinct-story training sequences per peer and 1,024 disjoint heldout
validation sequences. The benchmark requests separate official `train` and
`validation` splits of the pinned TinyStories dataset. It rejects incomplete
streamed retrievals instead of silently repeating the small available set.

- Training pool: 2,048 sequences × 4 peers = **8,192** sequences (default).
- Heldout validation: **1,024** sequences (default; shared between both arms).
- Sequence length: `--context 64`, so 64 predicted tokens per sequence.
- Tokenized samples remain deterministic for a fixed seed; each input story
  contributes one training window. Within a peer and epoch, sample selection
  shuffles without replacement until its pool is exhausted.
- Both central and ideal FedAvg simulations receive precisely identical input
  batches. The checkpoint, adapter rank, evaluation and network/compute-time
  definitions are otherwise unchanged.
- The toy model retains a lightweight default: 64 training sequences per peer
  and 64 validation sequences.

## CLI

```bash
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --peers 4 --sequences 2048 --validation-sequences 1024 \
  --rounds 40 --local-steps 8 --batch-size 4 \
  --rank 8 --context 64 \
  --output-dir benchmark-results/tinystories-large
```

In the black/emerald GUI, open `05 · 分散学習の有効性検証` and edit
`学習系列 / ピア` (1–8192) and `検証系列` (16–8192). Defaults are 2048 and 1024.
The values are passed explicitly to the benchmark worker and are recorded
in `effectiveness-result.json` (`configuration` and `dataset`).

## Critical distinction: data pool vs actual training budget

**Increasing the dataset pool does not automatically increase the number of
training updates.** With the original GUI defaults (5 rounds × 2 local steps
× batch 4), only 40 sample selections per peer occur, even when the pool has
2,048 sequences. To actually make use of more distinct training examples,
increase `--rounds`, `--local-steps`, or `--batch-size` as appropriate. A new
`sample_draws_per_peer` field and console log clarify this distinction.

Increasing validation size also makes every evaluation slower, since 1,024
sequences are scored instead of 128. Evaluation time is **not** part of the
previously defined training-compute comparison. The real wall time can grow
significantly, especially on a CPU. Leave available RAM for model copies.

The new `--sequences` and `--validation-sequences` flags affect the offline
benchmark only; they do **not** change the live WebRTC trainers' HF dataset
streaming or any signed Gossip/journal behavior.

## Tests

```
python -m pytest -q tests/test_tinystories_1m_effectiveness.py tests/test_effectiveness_benchmark.py tests/test_webview_gui.py
```

The test suite uses a small mocked GPT-Neo model and local fake TinyStories
streams so it does not download external models or datasets. The real
TinyStories-1M+1024-validation run must be completed in your own environment.
