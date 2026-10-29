# Kagami v35 — EVAL ValueError repair

This is a targeted patch on v34. It preserves the networking, training, dataset and GUI implementations and their stored data formats.

## Fixed

1. EVAL now defaults to **AUTO** precision, choosing BF16/FP16 on supported CUDA, FP16 on Apple MPS, or FP32 on CPU. Explicit CPU FP16 yields an actionable diagnostic instead of a confusing downstream failure. Explicit FP4/NF4/INT8 still requires a compatible bitsandbytes backend.
2. Prompt tokenization budgets the **input + output** against the model's finite context, excluding tokenizer sentinel `model_max_length`. For a very short custom model, excessive requested output tokens are *clamped*, not passed into `generate()` to raise errors. The returned result indicates requested and effective budgets.
3. Empty prompts can start from a BOS/EOS token where a model provides one; the GUI also shows a nonempty sample prompt initially.
4. EVAL dispatches via trusted Transformers AutoModel classes for `causal-lm`, `seq2seq-lm`, `masked-lm` (requires a `[MASK]` token), and `sequence-classification`. Seq2Seq generation does not incorrectly strip encoder input length from generated text.
5. Hugging Face `dtype` vs `torch_dtype` arguments are selected based on installed Transformers version; the minimum supported dependencies from `requirements-eval.txt` remain.
6. The latest learned-weight checkpoint list now matches both model ID and revision. A stale checkpoint from another revision is not silently used.
7. Structured worker errors include the **original exception**, a Japanese action hint and the last traceback lines in the EVAL interface. Errors are NOT silently swallowed or switched to pretrained HF weights.
8. Quantization compatibility checks run before an expensive streamed conversion, and Accelerator disk-offloaded embeddings can accept input IDs on CPU instead of erroneously trying to construct meta tensors.

## Install / launch

```bash
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-patches.txt
python -m pip install -r requirements-eval.txt
python kagami.py
```

Quantization requires compatible optional packages/device. `FP32` or `AUTO` is recommended as the initial troubleshooting setting. To evaluate learned weights, the model must have a complete matching snapshot; `Save current weights` then `EVAL SNAPSHOT COMPLETE` as in prior versions. Select "original HF base" explicitly if learned weights are unavailable and you intend to test the original pretrained checkpoint instead.

**Limitations:** The exact user-side ValueError traceback was not supplied, so this release addresses identifiable paths in source code; it does not prove that every future ValueError is eliminated. This environment lacks `transformers` and `accelerate`, so real Hugging Face model inference and GPU quantization could not be end-to-end validated. Learned-state integrity checks remain mandatory, and snapshots with a different architecture or tensor naming cannot be loaded safely. Non-generative EVAL results use a single-input local prediction, not validation-set accuracy.

## Validation

- 13 v35 regression tests cover CPU/CUDA/MPS policy mocks, context-budget adaptation, error messages, empty prompt inference and generation/classification/masked/seq2seq paths using real PyTorch with mocked Transformers factories.
- Re-ran existing GUI, shards, streaming, and Kagami v31-v34 regression tests (see test output for coverage).
- Python syntax compilation and Node.js JS syntax checking passed.
