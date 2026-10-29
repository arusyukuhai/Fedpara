# Kagami v63 — Hugging Face compatibility and training repair

**Status:** Compatibility hotfix; real model integration requires transformers, checkpoint weights, and sufficient hardware.

## What was fixed

1. GUI streaming training passed `Path(None)` to the scratch directory when the checkpoint shard destination was unset (normal new rooms). It now derives a per-profile private path from the journal.
2. Changing the NEW MODEL repository away from TinyStories no longer silently keeps the TinyStories-only random genesis selection; the UI selects pretrained genesis and enables streaming, while the backend rejects explicitly inconsistent random initialization at save time.
3. Streaming meta loading for ordinary BF16/FP16 pretrained safetensors now preserves the CHECKPOINT'S per-tensor floating dtype (opt-in at load_model; strict default API preserved). Both read-only and private writable backing paths use the actual checkpoint dtype.
4. Nonpersistent rotary `inv_freq` buffers (typical of Llama/Qwen variants) are reconstructed from the built-in module's local `rope_init_fn`; for simple default RoPE a bounded deterministic fallback is available. Unknown or scaled RoPE without a supported initializer is rejected, rather than silently corrupted.
5. Causal LM labels now mask padding positions by `attention_mask`, not by token ID, preserving genuine EOS targets when the model reuses EOS for padding.
6. LoRA-Muon new Llama/Qwen rooms use an architecture-neutral 2D projection scope (`linear`), excluding tied embeddings/LM heads. Existing TinyStories rooms keep `gpt_neo_linear` to preserve their old layout digest.
7. Standard `from_pretrained` chooses `dtype='auto'` in Transformers v5 or `torch_dtype='auto'` in v4, instead of needlessly promoting pretrained BF16/FP16 weights to full FP32 memory.
8. Added `python -m patchlib.cli hf-diagnose --model <ID-or-directory> [--revision SHA] [--local-files-only]`. Reads config metadata without executing remote model code, shows preliminary compatibility and gated access diagnostics. Diagnostic success is **not** a claim that a full training run has been performed.
9. Native DeepSeek-V2/V3/R1 FP8/MoE configs and other checkpoint-native quantization configurations now fail early with guidance; the `DeepSeek-R1-Distill-Qwen` / `DeepSeek-R1-Distill-Llama` series use standard Qwen2/Llama architectures and are the appropriate targets for first real-device testing.
10. Unsupported FP8 tensor dtypes are rejected early; no untrusted `trust_remote_code` execution was enabled on P2P workers.

## Suggested GUI workflow

1. NEW MODEL: paste a regular *Llama* checkpoint or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
2. Select **Copy pretrained weights** and **Pretrained** initialization, not TinyStories random.
3. For larger-than-RAM models enable **Streaming base**, use a pinned revision SHA, select LoRA rank8 or LoHA rank16 appropriate to device, and configure a valid dataset.
4. Start a **new room** if the previous failed room pinned an incompatible initialization/adapter layout. Never delete an established working Journal merely to force a restart.
5. For gated Meta Llama models, accept license terms on Hugging Face and authenticate using `hf auth login`.

Example local diagnostics:

```bash
python -m patchlib.cli hf-diagnose --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --revision main
python -m patchlib.cli hf-diagnose --model /path/to/local/safetensors-checkpoint --local-files-only
```

## Limitations

- The build environment did not include the `transformers` package, HF model weights, external networking for downloading real checkpoints, or GPUs. **No real Llama 8B, DeepSeek-R1-Distill 1.5B/8B, or native 671B DeepSeek training succeeded in this environment.** A tiny Llama-like PyTorch model with authentic BF16 safetensors layout, tied embedding/head and nonpersistent RoPE underwent actual forward, backward, optimizer step.
- Untested built-in architectures or nonpersistent buffer patterns still fail closed. Hugging Face Transformers-version differences can require additional per-model adapters.
- Full DeepSeek V3/R1 MoE/FP8 models, low-memory long-context attention, full expert training and 300B true stop-free training remain unsupported/unverified. Generic meta initialization alone is **not** proof that a full architecture can execute out-of-core.
- Old room manifests/identities are intentionally not silently rewritten. Llama and DeepSeek gated repositories require explicit access.
- Some user systems need newer PyTorch / Transformers for particular built-in architectures; repository requirements remain general minimum bounds.

See VALIDATION_V63.md and CHANGES_V63.patch for tests and code changes.
