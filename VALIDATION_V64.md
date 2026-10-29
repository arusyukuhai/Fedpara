# Kagami v64 validation — 2026-10-27

## Confirmed

- Architecture-neutral custom PyTorch model: 3-D learned expert kernel, 1-D learned bias, shared embedding/head Parameter and shared Module. Real causal-logits-only forward, backward, AdamW update, FPAT patch export/import and identical layout: **PASS**.
- Logits-only causal token shift, noncausal token alignment, seq2seq-compatible token loss, classification, explicit loss override; rejection of incorrect/detached/NaN/all-masked supervision: **PASS**.
- Input remap and explicit drop, malformed remap refusal, default auto-policy backward compatibility, bounded dense fallback, quantized preflight, unknown model_type acceptance: **PASS**.
- Actual CLI `architecture-probe` with a temporary local Python factory under explicitly offline isolated environment, representative safetensors batch: **PASS**. Returned 3 adapter groups, reached all 9 factor tensors and finite loss (not a GPU result).
- New GUI room universal manifest is digest-pinned; LoRA-Muon incompatibility refused: **PASS**.
- Python compileall of package/tests: **PASS**; JavaScript syntax `node --check web/desktop_webview.js`: **PASS**.
- Focused regression suite covering legacy adapters, HF compatibility, CLI, GUI, custom loaders, wide-target selection, Muon, factor ledger, streaming base, v57/v58 epoch, continual FedSum: **95 PASSED / 0 FAILED** after final edits.
- Other focused subset (63 tests) passed earlier and overlaps the 95 tests; not summed.

## Not confirmed

The full existing test suite was attempted, but exceeded a 180-second execution limit partway through. It cannot be called passing; long-running/network tests were not exhaustively verified. A subsequent broader test command had an execution transport error; no results are claimed for it.

No real 8B/300B-model GPU training, native FP8/DeepSeek MoE training, 4,096-device WAN soak test, or multi-generation model-wide uninterrupted train test was run.

## Important protocol rule

A v64 universal room has a new adapter target set and a new signed layout digest. Do not change the layout or append universal events to a v63 `auto` room's existing Journal. Recreate the room / genesis after approving a new model-wide contract.
