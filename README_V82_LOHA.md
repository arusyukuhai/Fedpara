# Kagami v82 — LoHA-aware Actor probability replay (2026-10-28)

## Summary

v81 captured requires_grad tensors, but treated all other behavior-policy
parameters and buffers (including remote overlays and ReLoRA anchors) as one
frozen-base digest. Peers may have different remote overlays or old anchors even
when their unmodified frozen weights are identical. This could unnecessarily
reject legitimate trajectories, and was not a complete LoHA state commitment.

v82 explicitly snapshots:

* all trainable tensors including *every* LoHA factor (A/B/C/D, and the LoRA
  factors used for auto-selected embeddings/output heads);
* the exact immutable anchor factors for each parametrized group;
* the exact frozen remote contribution factors and the remote count, in sorted,
  deterministic order; and
* a SHA-256 commitment to all non-overlay frozen parameters and buffers.

All fields are stored as safely bounded safetensors, and the complete
checkpoint blob is SHA-256-addressed in KCP82 and KTR82 frames. Each frame is
independently authenticated with new domain-separated signatures. The rollout
is associated with a certified Actor epoch, and the verified replayed log-probs
(not the sender's original claims) enter the off-policy Critic objective.

## LoHA equation and reconstruction

For each adapted parameter with local, anchor and remote contributions:

    D_LoHA(A,B,C,D) = (alpha/r) * ((A @ B.T) * (C @ D.T))

    W_effective = B_frozen + (D_local - D_anchor + sum D_remote) / (1 + n_remote)

The second formula is exactly the current Overlay.forward implementation.
The v82 verifier installs historical `D_local`, `D_anchor`, and
`D_remote` on the recipient (under the existing adapter writer lock), computes
the old Actor likelihoods in eval/no-grad mode, then restores all original
weights, remote modules, anchors and heterogeneous training/eval mode flags
in a `finally` block. Each field is checked before the first mutation.

This format deliberately cannot be read as a v81 checkpoint and v82 peers
advertise `trajectory_v82` separately. Older v81/v80 frames are not trusted.

## Optional old-frozen-base replay across epochs

`behavior_checkpoint_v82.capture_frozen_base(model)` exports a separate
safetensors artifact (max 256 MiB) whose content is committed under the same
base SHA-256 as the behavior checkpoint. `FrozenBaseCAS(directory)` stores it
immutably by hash. A caller can supply `base_resolver=store` to
`replay_logps(...)` or to `LohaVerifiedTrajectoryGossip(...)`. The receiver
checks the full tensor digest, names, dtypes and shapes, substitutes the old
base temporarily, verifies the sample probabilities, then restores its
current base and all other live state, including on exception.

**The old-base artifact is NOT part of the per-rollout wire transfer.**
It must be obtained from a trusted/certified epoch snapshot process, and the
source Actor-epoch ledger must authenticate the rollout. This base CAS is a
bounded in-process recovery codec, not yet a remote-snapshot distribution
implementation for multi-GiB/terabyte models.

**Disk-mapped virtual mutable bases, streaming offload and Factor Ledger
require a dedicated signed virtual-base replay manifest.** v82 explicitly
rejects these configurations rather than producing false verification. On
an unattested frozen-base mismatch it rejects the update. Critic receipts are
never committed for such trajectories.

## Next large-scale design (NOT claimed as shipped)

1. Derive immutable, certified effective-base manifests from an Actor epoch's
   quorum-certified snapshot root and its exact signed journal cut. Store
   4-MiB content-addressed shards or journal-replayable factor deltas, using
   SHA-256 Merkle proofs. Avoid whole-base copies on rollout boundaries.
2. Separate **base-root** (epoch/cut, large and deduplicated) from **behavior
   overlay root** (LoHA/LoRA factors, small, one per sampled Actor state).
   Include both in the trajectory signature with tokenizer/model ABI.
3. Request only missing base shards/committed deltas over the bounded gossip
   overlay, with resumable chunk ranges and a rolling per-peer traffic budget.
   Rely on snapshot seeding rather than every sender broadcasting N copies.
4. Verify each streamed base tile, the authenticated Actor journal cut, and
   the LoHA manifest before reconstituting the old model in an isolated
   shadow evaluator. Do not mutate the live model during large old-base replay.
5. Keep GC pins until all referenced trajectory/cut evidence is finalized,
   expired under an explicit retention policy or safely compacted to a
   replacement certified snapshot. Never GC an unacknowledged dependency.

For `P` base parameters and LoHA rank `r`, the local factor payload is roughly
`O(r sum_i (in_i+out_i))`, plus anchors/remotes. The base itself is still `O(P)`
for a completely new, incomparable epoch: no lossless algorithm can promise a
sublinear amount of independent new base information without shared state.

## Security and mathematical limitations

* Reconstructed log-probs certify a *replayable sampled-policy distribution*;
  they do **not** prove that the sender actually drew tokens from that
  distribution, or that the overlay factors came from a valid historical
  model update journal. Exact historical journal lineage requires a future
  proof binding `actor_tip` to the actual overlay manifest.
* The off-policy trajectory-IS Critic expectation statement still requires
  correct behavior sampling, shared tokenizer/action space, stationary reward
  semantics, and support coverage. It does not guarantee finite-sample
  convergence or PPO Actor update unbiasedness.
* Nonstandard/opaque custom forward paths and mutable virtual weights are
  not automatically covered. The 2e-3 log-prob tolerance is for numerical
  cross-backend differences, NOT a target-policy importance-weight clipping.
* 64 MiB is still the factor checkpoint cap. LoHA with many remote overlays,
  huge dense fallback groups or ranks beyond this budget needs separately
  sharded factor checkpoints before the mode can be enabled at scale.

## Verification

`PYTHONPATH=.:tests python -m pytest -q tests/test_behavior_checkpoint_v82.py`

Tests cover explicit LoHA/LoRA overlay state, anchors/remotes differing by peer,
a mixed auto scope with embedding/head LoRA, temporary old-base CAS replay,
wrong base and wrong likelihood rejection, missing pages, the live CLI export,
exception-safe state restoration and a fail-closed virtual-base guard.
