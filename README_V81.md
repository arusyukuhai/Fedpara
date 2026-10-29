# Kagami v81 — Reproducible Old-Actor Probability Checkpoints

## What changed

The v80 `behavior_logp` **signed claims** are not accepted as sufficient evidence for a recipient-side Critic update. v81 adds a NEW, room-policy-pinned, checkpoint-verification protocol:

1. Immediately **before** a new PPO rollout samples its first token, the source Actor copies **all trainable model parameter tensors** under the existing `adapter.lock`. The source also hashes *every nontrainable model parameter and buffer*, including dtype, shape and stable tensor name. The resulting safetensors blob includes the frozen-base fingerprint and is SHA-256 content-addressed.
2. `KCP81` signed, bounded 48-KiB checkpoint pages are persisted by the separate durable trajectory journal; `KTR81` contains the source's complete signed rollout and checkpoint digest. Both bind to the room's v81 trajectory genesis and a certified Actor epoch. Each event is signed by its origin; relays cannot change event payloads. No Critic gradient/head delta is received.
3. Existing bounded `trajectory-frontier/trajectory-need` anti-entropy retransmits missing pages, even across reconnects. Exact-repeat snapshots are reused within the same source Actor epoch. Journal persistence and a rebuildable SQLite page index tolerate interrupted index transactions.
4. A recipient waits until all signed checkpoint pages are present, concatenates them, checks the signed provenance and **full** content SHA-256, verifies the certified Actor epoch, and validates all trainable names/dtypes/shapes/finite values *before* touching its current Actor.
5. The recipient **requires its own frozen-base tensors and buffers to have an identical hash**, temporarily loads the old trainable Actor tensors, does an `eval()` full-forward pass on the exact prompt and chosen-token prefixes, computes `log(mu(a_t | s_t))`, compares to signed reported values (2e-3 absolute per-token guard), and **uses the independently recomputed values** in the unclipped trajectory IS Critic objective. The current Actor is restored in `finally`; its gradients/optimizer are not updated during imported-Critic replay.
6. The local Critic value head/AdamW state and applied receipt remain stored atomically as in v80. Checkpoint-missing, mismatched, corrupted, unrepresentable and legacy-v80 events never generate successful Critic update receipts.

## Conditional math

For actions `a_0...a_(T-1)` sampled from the recorded old Actor `mu`, current recipient Actor `pi`, and terminal reward `R`, let

`w_<t = exp(sum_(j<t) [log pi(a_j|s_j) - log mu(a_j|s_j)])` and
`W = exp(sum_(j<T) [log pi(a_j|s_j) - log mu(a_j|s_j)])`.

The existing v79 loss is

`L = 1/(2T) * sum_t [ w_<t * V(s_t)^2 - 2*W*V(s_t)*G_t + W*G_t^2 ]`, with `G_t = gamma^(T-1-t) * R`.

With genuinely sampled trajectories, a fixed target Actor per receiver step, a common prompt distribution/tokenizer/action space, identical reward semantics, exact probabilities, support and integrability, its Critic **gradient is unbiased in expectation** for the target-policy Monte Carlo value objective. In v81 **the recipient can now validate/recalculate the denominator policy probabilities from an immutable tensor commitment**, rather than taking signed decimal values at face value. Floating-point backend differences and the explicit bounded numerical guard mean this is a conditional mathematical result, not exact real-number computation.

**What v81 DOES NOT prove:** A sender's signature and reproducible weights do not cryptographically prove it actually sampled its reported actions from that Actor or submitted an unbiased prompt population. Arbitrary fabricated trajectories can still bias empirical training. This is not a proof of arbitrary Byzantine data honesty, global Critic convergence, exact PPO Actor replay, or finite-variance IS. Weight clipping is intentionally absent; numeric overflow aborts the update rather than silently introducing a clipped IS bias.

## Important limits / operator expectations

- **v81 is a new protocol and new room**, pinned by `version=pt-sft-clipped-ppo-v81`, `critic_distribution=checkpoint-verified-trajectory-is-v81`, and `behavior_policy_proof=content-addressed-trainable-checkpoint-frozen-base-sha256-v81`. Old v80 rooms/events cannot be silently upgraded.
- **Common frozen base required.** This version cannot reconstruct a previous epoch with different *frozen* base tensors on the recipient. It rejects such samples. For cross-base migration, a future version must add durable, separately verified, immutable full-base snapshot archives and compatible-model execution replay. Merely knowing the actor journal cursor is insufficient.
- **Max per behavior snapshot: 64 MiB**, 48-KiB chunks, max 2,048 pages. Large adapters fail closed rather than weakening authenticity. This can be a major bandwidth/disk/time cost: approximately checkpoint-size bytes per new changed Actor rollout per receiving edge, plus fanout/retries. Snapshots are only deduplicated when identical *within the same epoch*. No delta compression or safe distributed archival GC is implemented yet. The v80 512-MiB trajectory journal cap remains; long-running use can exhaust storage.
- **Byte-identical model base is necessary but not sufficient** to guarantee matching execution. The root model revision, forward implementation, tokenizer IDs, operator semantics and nondeterministic GPU behavior must also be compatible. v81 uses 2e-3 tolerance to distinguish ordinary low-precision backend discrepancies from a false reported probability; it always trains from its *recomputed* log probabilities.
- **Receiver verification is expensive**: load actor adapter, compute frozen-base fingerprint, replay full-forward selected actions, restore current adapter, then run its current Actor and Critic again. Existing event loop currently performs this synchronously under the lock, and should be moved to a bounded accelerator-aware worker for production clusters.
- Existing v80 caveats persist for privacy of tokenized prompts and reference labels, unbounded un-clipped IS variance, independent peers' optimization orders, and lack of real WAN/GPU/long-duration distributed fault-injection tests.
- This is a **strict opt-in** distributed Critic PPO feature. Local/non-trajectory training is unchanged.

## Main files

- `patchlib/behavior_checkpoint_v81.py`: immutable trainable snapshot and frozen-base tensor digest, safe temporary restoration and independent log-probability replay.
- `patchlib/trajectory_gossip_v81.py`: checkpoint-page and trajectory signed protocols, content-addressed replay, reconstruction and resilient receiving.
- `patchlib/mixed_learning_v65.py`, `patchlib/cli.py`, `patchlib/elasticnet.py`: pinned v81 policy, pre-sampling capture and signed publication/anti-entropy/fenced replay.
- `web/desktop_webview.js`, `web/desktop_webview.html`: v81 GUI support and explicit size/cost warning.
- `tests/test_behavior_checkpoint_v81.py`: working two-peer CPU toy PPO replay; forged logp, source/recipient divergence, frozen-base mismatch, cross-restart gaps, tampering, conflict, page reindexing and unchanged-checkpoint reuse.

## Validation

Targeted CPU tests: `PYTEST_V81_FOCUSED.log` (**71 passed**; run before the final probability-substitution adjustment), and a subsequent 28-test subset **passed** after that adjustment. Full `pytest tests` started but exceeded the environment's execution time cap at 15% with no failure printed in the partial log; do NOT interpret this as a full-suite pass. No multi-host real-network stress test performed.
