# Kagami v83 — Explicit `model_type` in Hugging Face `config.json`

Hugging Face Transformers can reject a model with a message similar to
`Should have a model_type key in its config.json`.  Kagami room descriptors
(`manifest.json` / `design.model_type`) are **not** a substitute for the
Hugging Face model's own `config.json`.

## What changed

1. `patchlib/hf_config_v83.py` provides a built-in-only compatibility loader.
   It reads valid `model_type` unchanged. For *legacy config.json without that
   key* it accepts **only** a known, unambiguous built-in model class from
   `architectures` (such as `GPT2LMHeadModel`, `LlamaForCausalLM`,
   `GPTNeoForCausalLM`) and constructs its corresponding config. An unknown or
   ambiguous architecture fails with a clear diagnostic. **The source model
   files are never edited on load.** A declared unknown model_type still fails.
2. Kagami's CLI train/pretrained loading, streaming base loading, GUI config
   discovery, code/architecture initializers, direct inference and trained
   snapshot inference use the compatible loader. Pretrained weights loading
   receives the verified config explicitly rather than causing Transformers to
   try reading the same missing `model_type` a second time.
3. Temporary Hugging Face safetensors exports used by trained evaluation and
   ordinary merged model exports now enforce an explicit `model_type` key.
   Conflicting existing keys are rejected, never overwritten silently.
4. The offline diagnostic no longer says a missing/unknown model identity is
   a trainable streaming candidate. Where safely inferable, it shows the
   inferred architecture and a warning.

## Repairing a *local* legacy model (only when desired)

First inspect without writing:

```bash
python scripts/fix_hf_model_type_v83.py /path/to/local/model
```

If the known built-in `architectures` field unambiguously identifies the type,
repair the local `config.json`:

```bash
python scripts/fix_hf_model_type_v83.py /path/to/local/model --write
```

When **you know** the architecture but the JSON has no usable `architectures`
field, you may specify a built-in model type explicitly:

```bash
python scripts/fix_hf_model_type_v83.py /path/to/local/model --model-type gpt_neo --write
```

The script makes `config.json.before-v83.bak` before modifying `config.json`.
It never modifies weights. Editing a signed or hash-pinned model's config
invalidates its original commitment: publish a new authenticated revision.
Do **not** set `model_type` to a guessed value, because a mismatched
architecture can cause load failures or unsafe/inaccurate inference.

## LoHA/PPO replay implications

This v83 patch does **not** rewrite KCP82/KTR82, actor signatures, frozen-base
SHA-256 manifests, adapters or checkpoint tensor bytes. A local legacy config
repair must be treated as a *new* model identity/revision if config is part
of a signed genesis commitment. Missing `model_type` is a model architecture
metadata issue, not a LoHA factor-replay correctness issue.

## Verification

```bash
PYTHONPATH=. pytest -q tests/test_hf_config_v83.py tests/test_hf_compat_v63.py tests/test_kagami_v32.py tests/test_cli.py
```

Tests run offline with stub Transformers config factories. This environment
has no `transformers` installation; a real AutoConfig/weights integration
smoke test still needs the target Transformers version and model files.
