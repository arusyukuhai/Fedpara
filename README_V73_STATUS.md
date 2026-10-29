# Kagami v73 — PPO safety and replay performance audit

Derived from Kagami v72, with a conservative, opt-in-free correction to the existing PPO pipeline. The signed update wire format, checkpoint format, room policy and existing CLI/GUI settings are unchanged.

## Fixes

1. **Cached HF decoding attention mask:** During one-token decode with `past_key_values`, `input_ids` remains one token but `attention_mask` now includes *the entire current token prefix*. A monotonic prefix-length check also prevents accidental reuse of an incompatible cache. Previously `attention_mask` was one token long, which is incorrect for many HF attention implementations. If the model doesn't expose a supported cache contract, existing full-prefix decoding remains the fallback.
2. **PPO rollout token allocations:** Use one bounded preallocated token buffer for sampling rather than allocating a new full-prefix tensor after every token. The final saved rollout is cloned to an independent immutable tensor. This does not change the actor-gradient forward or how old log-probabilities are stored.
3. **HF Critic frozen-backbone replay:** Existing HF Critic only trains its scalar head. After one backbone evaluation, cache at most 32 MiB of *detached per-prefix final features*, enforcing the budget before allocating each group, keyed to the precise prefix contents. On PPO reuse recompute the trainable scalar head on these features (including its gradient), without re-running the frozen backbone or re-tokenizing. Bypass is enabled only if recomputing `head(features)` numerically matches the model's actual scalar outputs. Unsupported custom heads or nonlinear post-processing transparently use the previous whole-model code path. A new rollout replaces the previous cache; model-specific optimizer state remains unchanged.
4. **Extreme stale policy handling:** Don't silently clamp raw PPO log importance ratios when current and behavior policies are separated by more than 20 natural-log units, or when ratios are nonfinite. Discard the stale rollout and fail closed rather than reporting a misleading bounded KL or applying an unstable update. Normal PPO ratios and clipping remain unchanged. A new subsequent call recollects a rollout.

## Limitations

- This is **not** a fully parallel Actor/Critic/Reward scheduler, not a new PPO implementation, and not a demonstration of 300B-model or thousands-of-peers long-running training.
- Feature caching requires the **Critic backbone to stay frozen** while the critic head trains; this is the current v69/v72 HF Critic contract. It is not valid for full-backbone Critic fine-tuning or arbitrary stateful/stochastic score heads, where the validation gate must reject caching.
- The feature cache uses at most 32 MiB of feature tensors, in addition to the existing frozen model. Tokenization and backbone inference for **new** PPO rollouts are still required.
- Severe P2P policy drift is now a reported error instead of being hidden. Callers should restart/recollect from their durable checkpoint in such a case; there is no distributed PPO policy snapshot protocol added here.
- The broad suite has long-running P2P simulation and topology tests; see `VALIDATION_V73.md` for exact tested scope. No live HF checkpoint, GPU benchmark or WAN cluster test was available.
