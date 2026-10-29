# Kagami v65 — concurrent pretraining + SFT + clipped PPO (experimental)

## Scope

This revision is an opt-in mixed-objective training path on top of v64. It computes
three independent losses **inside each local optimizer microstep** and sends their
accumulated gradients into the **same LoRA/LoHA adapter**. There is no global PT→SFT→PPO
phase barrier: pretraining can continue while SFT and PPO are applied.

- **PT:** retains the existing source (HF room dataset, pretokenized safetensors, or raw text).
- **SFT:** local JSONL lines `{"prompt":"Question: ","response":"Answer"}`; loss masks all prompt tokens. Labels cover the response and EOS when present. Large examples error instead of silently removing response supervision.
- **PPO:** local JSONL lines `{"prompt":"2+2=", "reference":"4"}`. Creates a fresh bounded on-policy completion, snapshots sampled action log-probabilities and local critic estimates, computes terminal ground-truth reward (position-wise token match, or exact token sequence), generalized advantage estimates, clipped policy ratio objective, separate clipped critic value loss, and bounded approximate KL relative to the **recorded behavior policy**. Maximum default reuse of each trajectory: **2 microsteps**. Samples/old logprobs are always detached.
- **Value estimator:** compact local-only trainable scalar critic from bounded predictive-statistics features. It is NOT the language model grading itself, is not included in P2P signed patches, and currently is **not checkpointed durably**. Each restart reinitializes it and starts collecting new trajectories. Reinitialize on model/adapter generation changes as well.
- **Security:** never load arbitrary reward Python scripts into the P2P process. PPO is *verifiable-answer reinforcement learning*, **not** human-feedback RLHF; a human-preference reward model or a persistent full-model critic would be a separate feature. Do not treat the proxy reward as human alignment.
- **P2P:** each signed adapter event still uses the existing durable journal, rank layout, epoch protocol and origin signature. When mixed objectives are enabled, an exact shared training-protocol descriptor becomes part of the root genesis. Incorrect settings cannot silently join the same signed room. Local data paths are excluded from that identity. When using the optional `--mix-policy`, peers can specialize by providing different subsets of local datasets, so PT-only and SFT/PPO peers continue training at once.

**No change to the existing behavior** when neither `--sft-jsonl`, `--ppo-jsonl` nor `--mix-policy` is used.

## Usage

Append the following options to an otherwise valid `python -m patchlib.cli train-elastic ...` invocation that already specifies model, revision, room and pretraining data:

```bash
--hf-task causal-lm \
--sft-jsonl ./data/sft.jsonl \
--ppo-jsonl ./data/ppo_prompts.jsonl \
--pt-weight 1.0 \
--sft-weight 0.5 \
--ppo-weight 0.1 \
--ppo-max-tokens 8 \
--ppo-reuse 2 \
--ppo-kl-coef 0.02
```

All three objectives are evaluated within **each** local training microstep. `--grad-accum` still controls when the shared adapter optimizer commits an update. To disable PT on a specialized peer, use `--pt-weight 0` and supply `--sft-jsonl` and/or `--ppo-jsonl`; no PT dataset is then needed for local `train` / `train-p2p`, and a PT fetch is skipped.

The current **causal decoder logits** contract is required for autoregressive PPO. Arbitrary differentiable model parameters are still supported as LoRA/LoHA targets, but encoder-only, diffusion, opaque generation kernels, and seq2seq PPO require task-specific rollout adapters. Some Hugging Face model implementations may also need `--input-map` from v64.

### P2P task specialization with shared policy

Create one policy JSON from the model owner's selected settings (same field names and values as `patchlib.mixed_learning_v65.spec(args)`) and distribute the identical policy file to all peers. For example, for the default v65 objective coefficients and rollout options, a simple Python snippet can produce the complete file:

```python
import argparse, json
from patchlib.mixed_learning_v65 import spec
config = argparse.Namespace(sft_jsonl='enabled', ppo_jsonl='enabled', mix_policy=None,
    pt_weight=1.0, sft_weight=1.0, ppo_weight=0.2,
    ppo_clip=0.2, ppo_kl_coef=0.02, ppo_value_coef=0.5,
    ppo_gamma=0.99, ppo_gae_lambda=0.95, ppo_max_tokens=8,
    ppo_reuse=2, ppo_reward='token_match', input_map='')
open('shared_mix_policy.json','w').write(json.dumps(spec(config), indent=2))
```

All members start their normal training command with `--mix-policy shared_mix_policy.json` and **matching weights, PPO options, input mappings**. Each peer may independently provide `--sft-jsonl`, `--ppo-jsonl`, either or neither (pretraining only), so long as the common policy enables that task. No file paths, private examples or live rewards are transmitted by this mechanism. If the policy differs, the room genesis differs and the existing authenticated handshake refuses to treat the peers as one model.

### Runtime cost

For each mixed microstep, there are up to 3 training forwards/backwards plus up to `--ppo-max-tokens` autoregressive sampling forwards. This is substantially more compute than PT alone, especially for large models. PPO is opt-in and should be first validated on a small model; `--ppo-max-tokens 4 --ppo-reuse 2` is a reasonable smoke-test configuration. Numeric weights scale gradient contributions, **not** computation cost.

### Supported reward functions

Only reference-based rewards `--ppo-reward token_match` (default) and `--ppo-reward exact` exist in v65. This supports automatically graded tasks. Arbitrary numeric ratings in JSONL are **not** trusted for model-generated completions: claiming a fixed reward for a different sampled response would train the wrong policy. The model itself never decides reward quality.

## Remaining limitations / not implemented

- Real Llama/DeepSeek model weights and large GPU training were **not exercised** in this environment.
- PPO uses a deliberately small local value estimator; no durable critic optimizer state, distributed critic merging, preference reward model, or off-policy replay correction beyond bounded clipped reuse. It is **not yet an RLHF training stack**.
- A new root genesis is required to enable mixed objectives in an existing P2P room. Existing v64 rooms/keys/journals must not silently be reinterpreted under a new objective protocol.
- `--mix-policy` is a CLI feature. GUI configuration / remote local-file selection were **not** added in this revision.
- Task-specialized peers can use different datasets, but each step within one peer is synchronous in execution (not three independently running model copies). Networked peers of course train concurrently.
- The prior 300B, thousands-of-real-WAN-peers, infinite-duration storage and zero-pause epoch guarantees remain unproven.

See `VALIDATION_V65.md` and `CHANGES_V65.patch` for tests and diff.
