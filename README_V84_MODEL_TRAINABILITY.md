# Kagami v84 — trainability fixes across Hugging Face model families

v83 only repaired some `config.json` missing-`model_type` scenarios. It **did not** certify that a model could perform a valid training step. V84 improves several observable compatibility problems and adds a one-batch real-gradient preflight. It does *not* promise all models, quantization formats, tokenizers, or distributed setups will train.

## Changes

1. **Frozen-base activation checkpointing and LoHA/LoRA gradients.** Prefer `gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})`. Legacy HF implementations that do not support that argument get input embedding activation-grad enablement without unfreezing the base embedding weights. Models lacking both paths fail with a clear error instead of silently losing gradients.
2. **Causal LM data labels.** HF dataset collection v2's CLM path used `DataCollatorForLanguageModeling(mlm=False)` even for tokenizers with `pad_token_id == eos_token_id`; it masks *genuine* EOS training targets. Now mask using `attention_mask`, preserving genuine EOS. The MLM path still uses the HF MLM collator. Local text training already used this principle.
3. **Tokenizer-produced segment IDs.** Causal LM and seq2seq text tokenizer calls use `return_token_type_ids=False` to avoid unwanted `token_type_ids` crashing decoder-only/encoder-decoder forward methods. MLM/classification still retain supported segment IDs; explicit `--batches-dir` remains untouched.
4. **Newer built-in model families.** When `config.json` truly lacks `model_type` but declares built-in `architectures`, allow inference from the installed Transformers library's `MODEL_*_MAPPING_NAMES` registries. Conflicting or unknown mappings still fail. No model implementation classes or `auto_map` custom code are imported/used for identity recovery.
5. **Trainability probe.** `training-preflight` runs *one real dataset minibatch* forward+backward, and refuses missing, nonfinite **or identically zero** adapter gradients. No optimizer update, no peer connection and no journal mutation. It does not establish that all future batches, long sequences, or other devices will work.
6. **Helpful zero-target error.** When the selected rank and target scope produce no compatible LoHA/LoRA groups, the exception reports the chosen rank/scope and suggests the proper new-room options. No automatic per-peer rank alteration or changing signed adapter layouts.

## To diagnose a particular model

From inside the unpacked `kagami-v84` folder:

```bash
python -m patchlib.cli hf-diagnose --model org/model
```

Run the same architecture/task, LoHA scope, dataset, and device as your intended room, without creating a running room:

```bash
python -m patchlib.cli training-preflight \
  --model org/model --revision main --hf-task causal-lm \
  --kind loha --rank 16 --target-scope auto \
  --text-file ./sample.txt --batch-size 1 --max-length 128 --device cpu
```

For an HF dataset v2 room, replace `--text-file` with `--room-manifest /path/to/room.json`. Models requiring a different task must set `--hf-task` to a matching head. Existing `architecture-probe --probe-batch /path/to/batch.safetensors` remains available for custom input mappings. Quantized native FP8/4bit paths may still require a separately audited differentiable backend; unsafe custom remote code remains blocked on networked peers.

If a probe fails, record the *model repository + pinned revision*, `config.json`'s `model_type` and `architectures`, selected `--hf-task` and adapter scope, device/backend, and the **full traceback**. A model may load yet fail for entirely different reasons: nontrainable quantization, unsupported multimodal inputs, incompatible task head, missing tokenizer resources, sharded offload, OOM, or external operator kernels.

## Distributed compatibility

- KCP82 / KTR82 / signatures / LoHA checkpoint format / epoch consensus were not modified by v84.
- Training data supervision changes for HF-dataset causal LM. Upgrade all peers in a room *consistently*; mixing v83 and v84 training workers can yield different objectives even if cryptographic signatures remain valid.
- Never auto-reduce rank or silently broaden adapter scope in an existing room. That changes its signed layout. Start a **new room/genesis** with the desired rank/scope if required.
- Local model configuration repair still invalidates any config-bound genesis hash or signed model identity: publish an authenticated revision.

## Tests & limitations

Relevant offline regression suite: **111 passed** (including v80–v82 trajectory/behavior checkpoint, GUI, LoHA, CLI and new v84 regression tests). The environment does not have `transformers` installed or the user's actual failing model checkpoints. Real hub/model-specific end-to-end training, GPU compatibility, memory sufficiency, and full-suite regression remain **unverified**. The new built-in architecture registry is unit-tested with a stub mapping, not with every live Transformers version.
