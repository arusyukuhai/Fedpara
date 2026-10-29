# Fedpara v8 — Mixed-rank snapshot repair and architecture-neutral auto LoRA/LoHa selection

This revision addresses the live crash:

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (16) at non-singleton dimension 1
```

## Root cause

The live adapter can mix rank-16 LoHa layers with rank-4 LoRA layers. However,
`apply_effective_state` in `patchlib/state.py` previously reset *every* adapter
group using the global `adapter.kind`, `adapter.rank` and `adapter.alpha`, even
when that group was configured as rank-4 LoRA. On snapshot import or rejoin,
copying the regenerated rank-16 factors to rank-4 parameters raised the shape
mismatch. The legacy elastic and synchronous merge paths had the same unsafe
assumption.

Each reset now reconstructs the group's own `FactorDelta` (`spec.kind`,
`spec.rank`, `spec.alpha`, `spec.init_style`). After an effective snapshot,
local factors, anchors and obsolete remote overlays are reset consistently so
that already-merged updates are never added twice.

## Model-agnostic `--target-scope auto`

`train-live` defaults to `--target-scope auto`. For any PyTorch/Transformers
model (not only GPT-Neo), it inspects original Parameter identity and their
owning modules *before* parametrization:

- `nn.Embedding` / `nn.EmbeddingBag` weights → rank-4 LoRA by default;
- output weights discovered using `get_output_embeddings()` → rank-4 LoRA;
- conventional output-head names such as `lm_head` / `embed_out` → rank-4 LoRA;
- any 2D weight whose larger dimension is at least 8x its smaller dimension
  → rank-4 LoRA;
- remaining eligible 2D weights → rank-16 LoHa (when `--kind loha`);
- 1D bias/normalization and unrepresentable small dimensions are skipped, not
  expanded into an expensive dense trainable fallback.

The LoRA threshold uses `--wide-lora-rank` (default 4); LoHa rank uses `--rank`
(default 16). Tied parameters always share one overlay, and the per-layer
kind/rank is included in the patch layout identifier.

Explicit `--target-scope all`, `gpt_neo_linear` and `gpt_neo_full` modes are
retained for compatibility. Other models using the live GUI or `train-live`
with `--target-scope auto` now get automatic architecture-neutral detection.

## Migration / compatibility

- Stop the live process completely and back up the `.sqlite3` journal and
  `.fstate` snapshots before replacing a running version.
- GPT-Neo layouts with only the expected Attention/MLP/Embedding/Head weights
  match v7's `gpt_neo_full` layout, allowing normal resume when all other room
  settings and state digests match.
- For other architectures or models with additional 2D weights, the auto
  policy may produce a **new layout**. Do not mix peers with different layout
  digests; start a fresh room/journal if the layout changes.
- Snapshot files are verified before being applied. An incompatible snapshot
  should fail closed rather than silently changing the model layout.

## Verification

- Mixed-rank snapshot import reproduces the old failure path and verifies
  exact effective model outputs afterward, including anchor consistency.
- A non-GPT-Neo model with tied Embedding/output head and elongated projections
  demonstrates architecture-neutral selection and frozen excluded weights.
- Legacy elastic and synchronous round resets are regression-tested with
  mixed rank-4 LoRA/rank-16 LoHa groups.
- All test groups: 311 passed, 3 skipped (non-swarm: 281 passed, 3 skipped;
  swarm: 30 passed). These are local unit/integration tests; a live five-peer
  extended WebRTC run has not been performed.
