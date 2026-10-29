# Kagami v64 — architecture-neutral language-model contract

## Goal

Move from named-family compatibility patches to a model-independent adaptation contract. The foundation is **any executable, differentiable `torch.nn.Module` exposing ordinary floating `nn.Parameter` tensors**, rather than a list of Llama/Qwen/DeepSeek model classes. This is a capability condition, **not** a claim that literally every language model is executable.

## Changes

1. `--target-scope universal` opts into every supported float Parameter found in a model's module tree: 1-D biases/gates, 2-D embeddings/projections, convolution kernels, 3-D expert kernels and higher-order tensors. Standard LoRA/LoHA factors are used for shapes supporting the configured rank; small unusual tensors use the existing exact dense adapter fallback (with a 1,048,576-element safety cap). Float8, unavailable/opaque weights, and non-PyTorch kernels are not silently accepted. Tied weights and modules are visited without attaching a second adapter to the same module field. **Old room layouts and the existing `auto`, `linear`, `all` modes are unchanged.**
2. `model_contract_v64.py` separates the model's `forward(**batch)` ABI from the supervised loss ABI. A logits-only causal LM can now use a correct next-token shift; token, seq2seq and classification logits have explicit loss semantics. A model-provided scalar loss is preferred in `auto` mode. Invalid/nonfinite/detached loss is rejected. `--input-map '{"input_ids":"tokens","attention_mask":null}'` can rename or explicitly drop a batch field (rather than dropping unexpected arguments by accident).
3. `architecture-probe --probe-batch ...` runs **real forward and backward on representative tensors** with no optimizer step and reports adapter layout, reached gradients, missing/non-finite gradients and loss. This is not a complete model safety certification.
4. The DeepSeek architecture-name blacklist is removed. An installed, trusted execution backend may now attempt any known architecture; the actual implementation and tensor format determine compatibility. Quantized checkpoints still fail with an actionable message until a tested dequantization/training backend is supplied.
5. Desktop GUI offers Auto (traditional 2-D scope) / Universal (all float parameter tensors). New-room `adapter.target_scope=universal` is pinned in the immutable room manifest; existing rooms retain the original digest and layout. Universal is incompatible with matrix-only LoRA-Muon, so the GUI refuses that combination.

## CLI examples

Run a preflight on an installed HF architecture with a local representative safetensors batch:

```bash
python -m patchlib.cli architecture-probe \
  --model /models/my-language-model --revision PINNED_REVISION \
  --hf-task causal-lm --target-scope universal --kind lora --rank 8 \
  --probe-batch /data/sample.safetensors
```

For a locally trusted custom model that calls its input `tokens` and returns logits without `loss`, use an explicitly sandboxed **offline** factory (never execute code from a P2P peer):

```bash
# Only in an explicitly isolated and offline runtime described in sandbox/run_untrusted.sh
python -m patchlib.cli architecture-probe \
  --model local-custom --factory-file /trusted/architecture.py \
  --target-scope universal --rank 8 \
  --input-map '{"input_ids":"tokens","attention_mask":null}' \
  --probe-batch /data/sample.safetensors
```

For a new room, `train-live --target-scope universal ...` uses exactly the same tensor patch, signature, journal and generation mechanisms as v63. Use the same pinned model weights and adapter policy across peers. Do not change existing room settings mid-journal: a different adapter layout requires a **new room**.

## Limitations — essential to interpretation

- There is no universal loader for *all possible* binary, opaque or proprietary models. One needs a compatible local execution backend; PyTorch autograd must observe the model's weight usage. Tensor parallelism, arbitrary custom CUDA kernels, exotic sparse/quantized FP8 formats and runtime-specific key layouts may require audited backend implementations. **Do not enable untrusted Hugging Face remote model code in the networked P2P worker.** The existing sandbox guard remains in place.
- `universal` is a functional compatibility mode, not 300B out-of-core support. The v63 tensor/Expert offloader still has limited supported module types; an unrecognized, very large operation can allocate the whole effective weight at once. Unknown custom ops require separate tiling/expert-dispatch integration.
- A probe proves only the *sample* input path. Conditional experts not selected by the probe remain untested. The report can show gradients present even if some kernels have zero gradient for that particular data. Real model checkpoint / GPU / WAN testing is still required.
- No benchmark of real DeepSeek V3/R1 or Llama 8B fine-tuning was run in this environment. Native FP8/quantized model backends remain unsupported without tested conversion kernels. Model family names are no longer an acceptance criterion, but that alone does not supply missing execution operators.
- `--input-map` and `--loss-policy` define local training semantics. Every participant should use a consistent contract; P2P model identity does not yet hash arbitrary per-peer local batch remappings.

## Files

- `patchlib/model_contract_v64.py`: architecture-neutral forward and supervised loss adapters, actual gradient probe.
- `patchlib/core.py`: universal tensor discovery and tied-module handling.
- `patchlib/cli.py`: runtime integration and preflight command.
- `patchlib/gui.py`, `patchlib/webview_controller.py`, `patchlib/room_manifest.py`, `web/desktop_webview.*`: immutable GUI scope.
- `VALIDATION_V64.md`, `CHANGES_V64.patch`: checks and full change summary.
