# Kagami v66 — GUI mixed-dataset + independent local PPO Value Critic

## What works

The default embedded WebView `TRAINING` tab now has a collapsible **PT + SFT + PPO** section.

1. **PT corpus**: the selected immutable room's existing Hugging Face dataset manifest (selected in `NEW MODEL` / `DATASETS`). Its dataset mix still belongs to the room's immutable definition.
2. **SFT corpus**: a separate local UTF-8 JSONL, each row `{"prompt":"...","response":"..."}`; response-only token supervision. You can browse using the native file picker or type a local path.
3. **PPO corpus**: a separate local UTF-8 JSONL, each row `{"prompt":"...","reference":"..."}`; on-policy sampled rollouts with ground-truth token/exact match scoring. It is *not* RLHF with an arbitrary preference reward model.
4. **PPO Value Critic**: choose an independent `mlp` or `linear` feature-based value network; configure hidden width, learning rate, value loss coefficient, discount `gamma` and GAE lambda. Optionally initialize it from another **Kagami v66** safetensors Critic checkpoint (not an arbitrary HF model). This Critic is **not** a self-rewarding judge.
5. Configure PT/SFT/PPO objective weights, PPO KL coefficient, PPO clipping, rollout length and reuse in GUI. `Start` passes validated settings to `train-live`, and the actual v65 objective mixer consumes the settings; no mock or UI-only scheduling.

The local critic weights and AdamW first/second moments are checkpointed in `rooms/<room>/critics/<peer-profile>/critic-v66.safetensors` at every 10 PPO updates and on clean shutdown. Checkpoints are written to a temporary file, fsynced, and atomically published. Reload rejects unexpected tensors, layout, model type or room/profile identity. An optional `critic-init` loads only data-only Kagami critic checkpoints. **A crash may rewind at most the last few unsaved local critic optimizer updates** (more under an I/O error); actor's signed P2P adapter history is separate.

## Getting started

1. In `NEW MODEL`, attach the desired PT Hub datasets and save a **new** causal-LM room.
2. Select that room under `MODELS`, then open `TRAINING` → `PT + SFT + PPO`.
3. Change `Training objectives` to `Concurrent PT / SFT / PPO`.
4. Choose SFT and PPO JSONL files independently. On the first worker of a new room, either can be omitted by setting its weight to **0**. On later peers joining an already pinned mixed room, either corpus can be absent even when that objective's room-wide weight is nonzero; no local gradients from that missing corpus are computed. PT remains driven by the selected room's Hub sources. For SFT/PPO-specialized peers, set the PT coefficient to 0 when creating the room.
5. Select the PPO Critic architecture, hidden size, learning rate and optional initial Kagami v66 `.safetensors` file. Set the reward style (deterministic `token_match` or `exact`).
6. Press `Start`. The GUI calls Python's `WebviewController.start`, validates all paths and hyperparameters, pins the shared objective specification and launches `patchlib.cli train-live` with the appropriate flags. The location of the original model and each data shard stays local.

When the model is selected again on the same machine, its latest **per-profile** private dataset and Critic settings are restored. If a room's shared mixed policy is already pinned but this peer has no local dataset selections yet, the GUI prepopulates its room-wide coefficients/Critic configuration; the user can attach only the datasets actually available locally. For a previously trained PT-only room, adding SFT/PPO changes its signed training genesis: create a **new room** instead of silently reusing the previous journal. The shared objective policy is fixed for a room; local filenames can change between restarts without changing the protocol, so peers can train on different local samples. This v66 GUI does not distribute the room's objective policy file between independent machines; securely provision the identical policy on joining peers if running mixed P2P objectives.

## What is *not* supported or verified

- The separately selectable Critic is a small **local value function**, not an arbitrary pretrained Hugging Face reward/critic LM or a remote reward grader. Self-Critic is not implemented. The initial checkpoint must be a Kagami v66 local Critic safetensors file.
- SFT and PPO input sources are currently **local JSONL paths**; selecting Hub-hosted SFT/PPO splits and remote streaming is future work. No large dataset is copied into JavaScript; native file dialogs return only file paths.
- Critic optimizer state is local, is not aggregated by P2P and has bounded-save intervals; the current feature-based value function is intentionally small and does not take LM hidden states.
- The objective protocol is immutable per room to prevent mixing incompatible signed histories. Changing PPO reward, Critic architecture or weights requires creating a fresh room.
- No 4,096-peer WAN live test, real Hugging Face large-LM mixed-training benchmark, arbitrary reward-model interface, or 300B full zero-pause validation was performed.
- The full test suite was attempted but did not finish within the execution limit; see `VALIDATION_V66.md`.

## Implementation details / boundaries

- The objective-policy JSON is included in the peer's signed training identity, but it is **not transferred automatically across distinct devices**. A second physical device needs the same trusted policy and local inputs. GUI local file paths are not sent over the network.
- Choosing another PT corpus after training has started would change the immutable room manifest: create a separate room, rather than silently altering a signed shared training history.
- Local PPO Critic state is saved every 10 updates and at graceful shutdown, so power loss may rewind the critic relative to an already signed actor update. The critic is auxiliary and is not a part of the signed update graph.
- First-row JSONL schema is checked before worker startup. Subsequent lines are validated by the streaming loader as encountered, with errors halting the local worker rather than silently skipping examples.
