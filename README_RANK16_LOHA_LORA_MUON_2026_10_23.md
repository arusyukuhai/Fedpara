# Fedpara rank16 LoHa / LoRA-Muon experiment (2026-10-23)

This release contains two distinct optimization methods:

- **LoHa + AdamW**: existing Fedpara continual LoHa, rank 16, balanced LoHa initialization, warmup + cosine decay, grad clipping 1.0, TinyStories-1M **default learning rate 1e-4**, and **GPT-Neo Attention/MLP projection weights only**. It omits Embedding, LM Head, LayerNorm and Bias parameters from the benchmark, preventing the previous huge dense-fallback embeddings. Not every LoHa update is smaller than its corresponding original dense layer: with rank 16, some small matrices have equal or higher adapter parameter count. Always inspect `configuration.adapter_counts`.
- **LoRA + LoRA-Muon**: paper *LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold*, https://arxiv.org/abs/2606.12921, Algorithm 1 (June 2026). Separate option with rank 16. **It does not operate on LoHa**, since the paper is about the LoRA matrix manifold. Requires `--kind lora --optimizer lora-muon`. Default LR in the benchmark is **0.01** (a starting point only, tune it separately from AdamW).

## Paper algorithm implementation details

File `patchlib/lora_muon.py`. For matrices `A[m, r]`, `B[n, r]` and EMA first moments:

```
RA = (A.T @ A)^(-1/2)   RB = (B.T @ B)^(-1/2)
dA = -(lr/2) * msign(momentum_A @ RB) @ RB
dB = -(lr/2) * msign(momentum_B @ RA) @ RA
s = sqrt(1 - lr*weight_decay)
A <- s*A + dA/s
B <- s*B + dB/s
```

The matrix sign `msign` uses the paper's 8-stage **Polar Express Newton–Schulz coefficients** by default; a small-SVD option exists for tests. Gram inverse roots use small `rank × rank` symmetric eigendecompositions instead of the **Appendix B.4 Newton–Schulz inverse-root kernel**. This is algebraically the same inverse-root operation (with diagonal damping), but **not identical numerical code** to the paper. No QR or large dense-SVD and no second-moment optimizer state. First-moment state, weight decay and learning-rate schedule are saved and restored through PyTorch `state_dict`.

The usual zero-initialized second LoRA factor is rank-deficient, where the strict rank-r manifold derivation is singular. Our regularized inverse Gram root enables a safe startup, but this damping is a **practical extension**. The paper's exact gauge invariance applies in the full-rank idealized case; numerical root damping / approximate orthogonalization can introduce differences. The repo has a regression test checking near-invariance.

## Run comparisons

```bash
# Main LoHa experiment: rank 16, LR 1e-4 (default)
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --kind loha --optimizer adamw --rank 16 \
  --peers 4 --rounds 40 --local-steps 8 \
  --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/loha-r16

# Paper-inspired LoRA-Muon: same rank, data and training budget
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --kind lora --optimizer lora-muon --rank 16 --lr 0.01 \
  --peers 4 --rounds 40 --local-steps 8 \
  --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/lora-muon-r16

# Re-run LR sweep for each method. Useful Muon candidates: 0.003, 0.01, 0.03, 0.1;
# LoHa+AdamW candidates: 3e-5, 1e-4, 3e-4.
```

Both methods report training/validation loss and an **unchanged baseline validation loss**. They use an idealized synchronous peer-delta aggregation (not actual WebRTC) and report `P2P compute seconds / peers`; this is **not measured parallel wall time**. Ensure both arms use the same optimizer within a single run.

## Live P2P (experimental optimizer option)

CLI `train-live` now accepts `--kind lora --rank 16 --optimizer lora-muon --target-scope gpt_neo_linear --lr 0.01` (for official GPT-Neo-style models only). The GUI also exposes optimizer choice for live training; selecting Muon sets the correct GPT-Neo linear target scope. Default live LoHa+AdamW is unchanged.

Live LoRA-Muon uses the **same signed continual before/after factor-delta gossip protocol**. For safety, Muon rooms derive a distinct genesis ID to avoid accidentally replaying AdamW journals with incompatible optimizer state. **Use a new room and fresh journal when switching optimizers.** The older default AdamW genesis IDs are not altered. In the event of restart, optimizer's first moment is restored through the existing working-state checkpoint path.

**Testing**: `python -m pytest -q` includes real small PyTorch matrix tests and a mocked pinned-HF GPT-Neo benchmark; neither tests a live macOS 16-peer aiortc swarm nor downloads the actual model. Those remain required acceptance tests before claiming real-world convergence.
