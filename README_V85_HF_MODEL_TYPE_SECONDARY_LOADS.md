# Kagami v85 — `Should have model_type key in config.json` recurrence fix

## Why v83/v84 still failed

Hugging Face `AutoTokenizer.from_pretrained()` often invokes `AutoConfig.from_pretrained()` internally when `tokenizer_config.json` does not identify the tokenizer class. The v83/v84 recovery wrapped the **primary model** configuration, but **not the tokenizer** in these paths:

- CLI local text training / adapter preflight (`patchlib.cli.training_batches`)
- P2P Hugging Face dataset training (`patchlib.hf_datasets.batches_from_room`)
- SFT / PPO mixed learning (`patchlib.mixed_learning_v65.build`)
- GUI evaluation/inference (`patchlib.kagami_eval_worker.run`)
- separate external reward model and Hugging Face Critic (both model and tokenizer loads)
- pinned TinyStories random initialization / benchmark and training simulator utilities

The error thus resurfaced when the model loaded correctly but an independent `AutoTokenizer` or external reward model was loaded later. **It is an architecture identity error, not a LoHA numerical issue.**

## What v85 changes

1. `patchlib.hf_load_v85.load_hf_tokenizer` and `load_hf_model` retry only the *specific* missing-`model_type` ValueError with `config=load_hf_config(...)` supplied explicitly. No global Transformers monkeypatch, no inference from model directory name, and no mutation to original HF files.
2. Primary `load_hf_config` checks the source `config.json` itself **before** Hugging Face's legacy path-name heuristic can silently supply the wrong model type. When `model_type` is missing, it allows a unique, recognized built-in `architectures` mapping; otherwise it refuses and gives the source model ID.
3. Recovery remains opt-out for custom code: `trust_remote_code=True` follows its prior sandbox-only path and is not silently treated as a built-in model.
4. Reward and HF Critic auxiliary model loads now receive the independently verified config on the same error, rather than loading a model type guessed from their repo names.
5. Fixed test reproductions cover tokenizer recursion, reward/critic model loader behavior, an HF path whose name misleadingly contains `gpt2`, an unknown custom architecture, and preservation of original source bytes and pinned revision.

## Verify the specific failing model

Inside the extracted `kagami-v85` folder:

```bash
python -m patchlib.cli hf-diagnose --model org/model --revision main
python -m patchlib.cli training-preflight --model org/model --revision main \
  --hf-task causal-lm --kind loha --rank 16 --target-scope auto \
  --text-file sample.txt --batch-size 1 --max-length 128 --device cpu
```

The exact second command depends on the real task, device and dataset. Use the *same model ID and revision* as the failed room. The preflight verifies one actual gradient step without modifying trained weights.

**Important: not every missing `model_type` can be fixed automatically.** A config with neither a valid `model_type` nor a uniquely recognized built-in `architectures` list is under-specified. A custom architecture cannot safely be reinterpreted as GPT-2, Llama, etc., just to make the exception disappear. To explicitly repair a known built-in **local** model directory (only if independently verified):

```bash
python scripts/fix_hf_model_type_v83.py /path/to/model --model-type gpt2
python scripts/fix_hf_model_type_v83.py /path/to/model --model-type gpt2 --write
```

`--write` creates a backup and changes `config.json`. In a signed room or when checkpoints pin file hashes, this requires a new authenticated model revision/genesis, not an in-place change under an old identity. Never alter a remote model repo implicitly.

## Validation and limitations

- Added v85 regression tests; 89 targeted tests pass (from HF adapters, configs, reward/Critic, dataset, LoHA trainability, room identity).
- The legacy `test_kagami_v35_eval_errors.py` test module contains 4 pre-existing failures with a fake Transformers module that does not expose `AutoConfig`; reproduces on unmodified v84. This is not evidence of a real user model failure or of v85 correcting it.
- This environment does not contain the actual Transformers package or the user's failing model weights, so an end-to-end load for that particular model remains unverified. Supply the model repository ID/commit and complete traceback to make a model-specific diagnosis.
- Model checkpoint, LoHA math, signatures, P2P protocol and the trained-weight representation are unchanged. Consistently deploy to every affected peer to avoid mixed runtime behavior.
