# Fedpara v28 — disk-resident frozen weights, paged AdamW and MoE planning

This is an **opt-in experimental runtime**. It does not claim unrestricted 300B training on a smartphone. HF model construction, some non-Linear layers, activations, and FedSum's live dense-merge path may still require full model/weight residency.

## Working local training: frozen Linear + LoHa/LoRA

The ordinary `train` / legacy `train-p2p` paths now accept:

```bash
python -m patchlib.cli train \
  --model YOUR_PINNED_MODEL --revision YOUR_REVISION \
  --peer-id local-1 --hf-task causal-lm --text-file train.txt \
  --target-scope auto --kind loha --rank 16 \
  --device auto --disk-linear-dir ./base-weight-pages \
  --disk-linear-min-mib 2 \
  --disk-optimizer-dir ./adamw-moments \
  --resource-report device-budget.json \
  --steps 1000 --output local.fpatch
```

- Linear *frozen* bases are saved as raw, disk-backed CPU tensors. Each layer's forward loads just that layer onto the active device. Backward reloads it to compute the input gradient; the autograd graph **does not retain the base weights**.
- LoRA factors use two compact projections; LoHa applies the rank² factorized contraction instead of materializing the dense effective LoHa matrix. This changes compute complexity; at large rank the rank² contraction may become expensive.
- AdamW first and second moments are stored in files under `--disk-optimizer-dir` and paged per trainable factor. At the end of `train`, moments are flushed. Use `--resume-disk-optimizer` **only** alongside an exactly matching factor checkpoint/layout; the disk moments alone do not restore the adapter weights.
- `--auto-rank` is **local-only** and picks a conservative default from the current device's free-memory estimate. For a P2P room, negotiate and freeze a shared rank in its signed genesis. Devices must not independently alter their factor layout.
- `--resource-report FILE` produces a machine-readable rank/expert budget. A report depends on currently available memory; it is *not* an exact OOM predictor.

**Important constraints:** Initial HF `from_pretrained` and `model.to(device)` happen before disk offload. This saves **steady-state** residency, not the initial peak. The new disk Linear path currently supports frozen, untied `nn.Linear` bases; embeddings, fused QKV and arbitrary operators need specific kernels. The continual **`train-live`/`train-elastic`** path explicitly rejects `--disk-linear-dir` and `--disk-optimizer-dir`: its FedSum merge and crash journal are not yet safe with these new modes. No silent fallback is attempted.

## MoE: heterogeneous experts, packed shapes and placement

`patchlib.moe_planner.inspect_expert_shapes(model)` discovers independently sized ModuleList and Switch ModuleDict experts as well as packed tensors with a shared leading expert axis. Architecture-specific detectors may be registered with `ExpertAdapterRegistry.register(model_type, detector)`.

- **Unpacked experts:** Existing `patchlib.expert_parallel.RemoteExpert` + `ExpertOwner` support remote forward/backward using an authenticated transport. `patchlib.expert_disk.DiskExpertProxy` can execute **frozen** arbitrary nested `expert(x) -> Tensor` modules using on-demand disk reads and backward recomputation without keeping base weights in the forward graph.
- **Packed experts:** Shape-only detection is safe for arbitrary tensors, but not sufficient for execution. `packed_swiglu_expert_views` supplies an opt-in executable adapter **only for validated 3-D w1/w2/w3 SwiGLU** tensors, usable by existing ExpertOwner. Other packed/fused experts require a locally trusted model-specific adapter. Their layout and router semantics are not guessed.
- Placement: `plan_expert_placement` is deterministic given the versioned roster and provides capacity-weighted rendezvous assignment with optional strict per-peer byte budgets. It throws an error if an expert cannot fit, rather than pretending the allocation succeeded. Compute/activation/network capacity are not estimated by placement.

Obtain the inventory and rank/expert recommendation from `--resource-report`. Prepare a roster JSON:

```json
{
  "peer-a": {"weight": 10, "expert_budget_bytes": 4294967296},
  "peer-b": {"weight": 4, "expert_budget_bytes": 1073741824}
}
```

Then:

```bash
python -m scripts.plan_expert_roster \
  --report device-budget.json --roster roster.json \
  --room MY_ROOM --generation GENESIS_DIGEST \
  --replicas 1 --output expert-owners.json
```

The generated plan is **not auto-deployed**. The program or runtime owning the verified P2P cohort must distribute and authenticate it, then install remote proxies for supported architectures. A remote call during backward requires a live owner and a verified generation; it does not support a mid-step failover without rollback.

## Remaining work for arbitrary massive models

1. Start HF models on `meta` and load **directly from a verified shard store**, without a full initial resident model; not implemented in v28.
2. Implement offload-aware frozen embeddings, conv/fused projections, and model-specific graph checkpointing where needed; not universal in v28.
3. Integrate disk AdamW and chunkwise FedSum into the **durable `train-live`** state machine so checkpoint replay and ReLoRA reset cannot corrupt optimizer state; not implemented in v28.
4. For custom packed experts, supply trusted router/kernel adapters and integrate distributed placements with live room roster messaging; only the validated SwiGLU packed format has a generic executor in v28.

Run `PYTHONPATH=. python -m pytest -q tests/test_offload_v28.py` for dedicated tests.
