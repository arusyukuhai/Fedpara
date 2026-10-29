# Kagami v67 — Hugging Face Reward Model selection and PPO scoring

## What is implemented

- TRAINING → PT + SFT + PPO → Reward = **Hugging Face Reward Model**.
- Browse Hub model metadata by search, select a repository, resolve its **40-character immutable commit hash**, choose inference device (CPU/CUDA/MPS), loader (`AutoModelForSequenceClassification` or generic `AutoModel`), score field (`logits`, `score`, `end_scores`, `rewards`), score index, prompt format (plain/chat-template), multiplier, and context limit.
- Scorer executes in local Python inside **`torch.inference_mode()`**, has **no gradients** and is not part of the Actor, P2P LoRA/LoHA signatures, or Critic AdamW optimizer. The Critic remains a separate *local value function* used by GAE. PPO uses the external RM for terminal rewards and caches scores for PPO trajectory reuse.
- `ppo_jsonl` rows can be `{ "prompt": "..." }` without `reference` in this mode. Legacy token-match and exact reference based scores remain unchanged and still require `{ "prompt": "...", "reference": "..." }`.
- WebView → controller schema validation → CLI flags → actual mixed learning PPO rollout scoring. The signed room objective policy pins RM repository ID, immutable revision and scoring interpretation; peer-local **device** is excluded, and private data/paths are not transmitted.
- `trust_remote_code=False` always. Safe `safetensors` checkpoints only; do not load arbitrary remote Python. A checkpoint with missing (uninitialized) parameters is rejected instead of returning meaningless rewards.
- Nonfinite/malformed/unsupported reward outputs **fail closed**. Scores are multiplied by a positive finite scale and bounded to `[-10,+10]` before GAE, and are **not used to update the reward model itself**.

## How to use

1. Open TRAINING → PT + SFT + PPO. Select **Concurrent PT / SFT / PPO**.
2. Provide a PPO `.jsonl` with `{"prompt":"Describe ..."}`. SFT can be a separate `.jsonl` or left empty where allowed by the shared policy.
3. Under PPO choose **Hugging Face Reward Model**, search for a model by ID, pick it, then choose **Pin current commit**. A 40-hex immutable SHA is mandatory for shared P2P scoring.
4. Use `Sequence classification scorer` for models with a trained scalar sequence-classification head. Use `Generic AutoModel` only if its built-in class **actually returns one of the configured score tensor fields**. `trust_remote_code` cannot be enabled from GUI.
5. Choose `logits` and index `0` for a standard `[batch,1]` scorer, or another explicit tensor field/index appropriate to that model. Choose `Chat` for reward models whose tokenizer includes the intended chat template; otherwise use `Plain`.
6. Choose the reward inference device. Start the training worker; it downloads/loads this reward model independently from the Actor and Value Critic. If the pretrained scorer is incompatible or unavailable, launch fails with a descriptive exception (it does not silently use a substitute reward).

## Limitations and security guarantees

- **Arbitrary selectable Hub repository** does not mean every repository is runnable. Only models supported by the installed `transformers` built-in AutoClasses, exposing the chosen scalar score, and using safetensors, are eligible. Custom remote-code architectures, weight-only causal LMs without trained reward heads, unusual scoring APIs, quantized-only checkpoints, PEFT adapter-only reward heads, and multimodal models may require an explicit trusted adapter or additional runtime.
- The `AutoModel` loader is an extensibility path, **not** a guarantee that every AutoModel exposes a reward score. Output must be an explicit tensor shaped `[1,num_scores]` (or `[num_scores]`). We do not reinterpret hidden states as rewards.
- Hub search/inspection fetches metadata only. Gated or private repositories require appropriate local Hugging Face authentication, and may fail without approval.
- The HF reward model and tokenizer are intentionally loaded in the training process. They may consume substantial RAM, VRAM and download bandwidth; CPU is the default. This release is **not** an asynchronous batched reward evaluation service, and scoring adds latency.
- As in v66, the PPO Value Critic is local and saved periodically; actor-signed updates and local Critic checkpoints are not guaranteed to be at exactly the same step after an abrupt crash. Model/reward configs are pinned per room. Changing an existing room's reward model or scoring policy requires a new room.
- A SHA pin prevents revision drift but does not prove the rewards are human-aligned or consistent to floating point precision across different hardware.
- No real Hub RM weights, `transformers` install, GPU, or large distributed WAN systems were available for the test run. Scorer loading/inference was validated with mocked Hugging Face API and actual PyTorch test modules; full external-repository compatibility remains to be measured.

## Files

`patchlib/reward_model_v67.py`, `patchlib/mixed_learning_v65.py`, `patchlib/mix_gui_v66.py`, `patchlib/cli.py`, `patchlib/webview_controller.py`, `patchlib/webview_gui.py`, `web/desktop_webview.html`, `web/desktop_webview.js`, `tests/test_hf_reward_v67.py`.
