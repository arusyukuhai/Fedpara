# Fedpara LoHa rank-16 initialization and optimization audit (2026-10-23)

## Summary

The preceding TinyStories-1M benchmark (`effectiveness-result.json`) recorded
an initial validation loss of 2.482287, central 2.555152, peer-mean 2.487016.
**Neither beat the pretrained baseline.** That experiment trained rank-32
adapters on 108 universal parameter groups, with 58 full-density fallbacks and
3,950,784 trainable patch parameters for 3,745,984 original parameters.
Thus it did not isolate LoHa in Attention/MLP.

This release changes TinyStories-1M benchmark defaults to:

| Hyperparameter | Default |
|---|---|
| LoHa rank | **16** |
| LoHa scaling `alpha/rank` | **16/16 = 1** |
| Trainable scope | GPT-Neo q/k/v/out and MLP c_fc/c_proj **weights only** |
| Initialization | **balanced**: one CP product nonzero, other initially exactly zero; first nonzero factors scaled by rank |
| Learning rate | **1e-4** for AdamW |
| AdamW weight decay | **0** for benchmark |
| LR schedule | **5% linear warmup + cosine decay to 10%** of peak |
| Gradient clipping | **global norm 1.0** |
| Data per peer | 2,048 sequences, validation 1,024 sequences |
| Optimizer moments | retained across rounds for *both* centralized and peer arms |
| Aggregation | mean of **after-minus-before effective LoHa deltas**, never averages of factors |
| Report | original loss baseline line, training/validation loss, LR, throughput and estimated traffic |

## Audit: the zero-initialization failure mode

Previous LoHa initialization used a Gaussian std of `0.02` for both branches,
setting one factor in the second branch to zero. Zero initialization is
correct *for the base output* (no pretrained model perturbation), but in this
four-factor multiplicative parameterization the first gradient back to the
zero factor is proportional to the product of **three** small initial factors.
It can be extremely weak, particularly with AdamW epsilon.

The new **balanced LoHa** initialization uses (for 2D matrices) std
`rank**(-1/4)` for the two factors of the first CP product and std
`rank**(-1/2)` for the nonzero factor of the second CP product; the second
factor of the second CP product is still zero. Thus the effective update at
step zero is **exactly zero**, without the triple-0.02 derivative suppression.
The initial gradient of the zero factor is nonzero; the other three factors
naturally receive their gradients on subsequent steps. This is a pragmatic
Fedpara-specific initial scaling, **not a claim of equivalence to PEFT's
LoHa initializer**.

A deterministic PyTorch regression using a 64x64 linear layer, rank16,
reports gradient norm **5.35e-5** (legacy) and **0.418** (balanced) at the
first initially-zero factor; this is a controlled numerical check, not a
TinyStories-1M validation result.

## Scope, efficiency and a newly found freezing bug

The former universal adapter selected *every* trainable floating tensor,
including embeddings, LayerNorm, biases and many Dense fallbacks. The new
`target_scope=gpt_neo_linear` matches official GPT-Neo Attention/MLP
projection **weights**. This generally avoids huge embedding and LM-head
patches, LayerNorm and dense fallback. The previously excluded parameters
were not all frozen; this release explicitly sets excluded parameters'
`requires_grad=False`, ensuring their gradients are not computed.

**Important:** rank16 is not universally smaller than a dense layer. On a
64x64 matrix a LoHa rank16 has `2*r*(64+64)=4096` factor parameters —
exactly the full 4096-weight matrix; on a 256x64 matrix it has 10,240 factors
versus 16,384 dense values. Assuming the standard 8-layer GPT-Neo structure
with 4 Attention and 2 MLP projections per layer, these six projection matrices
contain approximately **393,216 dense weights / 294,912 LoHa parameters**.
Do not confuse target-only count with the entire 3.7M-model count, and do not
claim arbitrary compression at rank16. Signed continual gossip transmits
**both** before and after factors.

## Live default vs offline benchmark

The benchmark now defaults to these safe settings. Additionally, `train-live`
/ `train-elastic` defaults are updated to lr=1e-4 and `init-style=balanced`.
`--target-scope auto` selects `gpt_neo_linear` for actual GPT-Neo models and
`all` for models of other architectures. The desktop GUI uses the same safe
training flags by default, and prints the resolved scope at startup.
Explicit overrides are available via the CLI (`--lr`, `--target-scope`,
`--init-style`). The balanced style is also the default for new `UniversalAdapter` instances; `legacy` remains available only as an explicit comparison setting. The optional LoRA-Muon experiment is independent of these
LoHa+AdamW defaults and should not be used as a LoHa optimizer.

**Migration:** GPT-Neo adapter layout and rank differ from previous releases.
Keep old journals and snapshots backed up; use a **new room/journal** for the
new target scope and rank16. Old journal events are **not** migrated by this
revision. Do not delete old identity keys or persistent state without backups.

## Run

```bash
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --kind loha --optimizer adamw --rank 16 \
  --peers 4 --rounds 40 --local-steps 8 --batch-size 4 \
  --sequences 2048 --validation-sequences 1024 \
  --lr 0.0001 --warmup-ratio 0.05 --min-lr-ratio 0.1 \
  --target-scope gpt_neo_linear \
  --output-dir benchmark-results/loha-rank16
```

For sensitivity analysis run the same experiment at `--lr 0.00003` and
`--lr 0.0003` with separate output dirs; report all three results rather
than selecting the best seed ex post. `p2p_compute_s_div_peers` is idealized
compute time only, **not** measured distributed wall time.

## Verification and limitations

`python -m pytest -q`: **271 passed, 3 skipped** in this CPU environment.
Checks include exact-zero initialization, live gradients, tensor selection,
excluded-gradient freezing, LR scheduling, optimizer state and original
signed Gossip / 16-peer regression tests. **Actual macOS/GPU 16-peer
TinyStories-1M WebRTC training and real Hugging Face checkpoint downloading
were not performed here.** Better validation loss is not yet demonstrated:
run the benchmark and compare against the original 2.4823 baseline on the
exact same fixed validation split before making accuracy claims.
