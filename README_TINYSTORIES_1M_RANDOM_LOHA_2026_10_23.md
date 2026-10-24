# TinyStories-1M random frozen-base LoHa, with Embedding/LM Head

This revision builds on `fedpara_loha_rank16_audited_full.zip`. The **default** benchmark is now scratch-initialized GPT-Neo of the same TinyStories-1M architecture. It does NOT load TinyStories pretrained `pytorch_model.bin` weights in `--initialization random` mode. The tokenizer and model `config.json` are loaded from the pinned revision. Run uses the same random seed and exactly the same base model in the centralized and simulated multi-peer arms.

The pretrained model remains an **explicit opt-in** control experiment (`--initialization pretrained`), subject to the existing PyTorch >=2.6 safe weights-only loader.

## Training semantics

- The randomly initialized Transformer backbone weights are **frozen** as trainable Parameters; only LoHa factors receive gradients. The algorithm nevertheless updates the frozen **storage values** during each merge of authenticated/in-process LoHa effects. This is an intentional *merge-to-base* operation, not backpropagation into the base.
- LoHa target scope `gpt_neo_full` includes the 8-layer Attention Q/K/V/Out and MLP FC/Proj linear weights, `transformer.wte` token embeddings, `transformer.wpe` positional embeddings, and `lm_head` weight.
- In GPT-Neo the LM Head is typically *tied* to token embeddings. The adapter detects identical `Parameter` identities; it installs ONE shared LoHa overlay, not two additive overlays. Untied head weights are adapted separately.
- The local LoHa and AdamW factor moments persist between rounds. On merge, the effective delta `D(current) - D(previous_anchor)` is accumulated into the global base, then the anchor advances. This is **ReLoRA-like cumulative merging**, not conventional ReLoRA's periodic rank reinitialization. Local LoHa factors are not reset.
- Rank 16 is the default. Embedding/LM Head LoHa on a `[50257,64]` matrix needs `4*16*(50257+64)` factor values, nearly the same size as the dense matrix. Adding embedding/head LoHa can be very costly; multiple peer replicas and optimizers increase RAM.
- The simulation is synchronous and in-process, *not* a live P2P/WebRTC test. Idealized parallel compute time uses the peer sum divided by N, excluding network and data/evaluation time.
- The desktop live-learning `--target-scope auto` is changed to `gpt_neo_full` for LoHa on GPT-Neo. This produces a different adapter layout digest; **old rooms/journals/snapshots using `gpt_neo_linear` should not be reused**. Back up old states before creating a new room. For LoRA-Muon, `auto` retains linear-only scope.

## Usage

Install Transformers/PyTorch, safetensors, datasets, matplotlib and GUI requirements from project READMEs, then:

```bash
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --initialization random \
  --dataset tinystories --kind loha --optimizer adamw \
  --target-scope gpt_neo_full --rank 16 --lr 0.0001 \
  --peers 4 --rounds 40 --local-steps 8 --batch-size 4 \
  --context 64 --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/random-neo-loha
```

Or select **ランダム（本体固定・LoHaのみ）** from the benchmark GUI. Use `--initialization pretrained` only for a separate fine-tuning control. `--model toy` continues to be synthetic and randomly initialized.

For diagnosis, check `configuration.initialization`, `configuration.target_scope`, `loss.initial`, `trace[].central_train_batch_loss`, `trace[].p2p_train_batch_loss`, and losses. Random backbone+LoHa-only training is a severe capacity constraint and *may fail to match true pretraining*, so improvement is an empirical question, not guaranteed.

## Verification

Tests exercise deterministic config-only initialization with a mocked Transformers interface, tied Embedding/Head aliasing, excluded base gradient freezing, first factor gradients, no-double-merge, untied head fallback, and adapter replay compatibility. **Real HF checkpoint/config downloads or macOS multi-process WebRTC were not performed** in the test runtime.
