# Kagami v80 — Signed Trajectory Gossip and Receiver-Side Off-Policy Critic Replay

## Implemented

v80 introduces a separate **opt-in** Critic distribution protocol. In a new `train-elastic` room use `--distributed-critic --critic-rollout-gossip --critic-offpolicy trajectory_is` (or GUI: TRAINING → VALUE CRITIC → Critic training → `P2P · signed trajectories / receiver-side IS`). The shared objective policy pins `critic_distribution=signed-trajectory-recipient-is-v80` and `version=pt-sft-clipped-ppo-v80`, disallowing accidental joining of older parameter-delta rooms.

Unlike v78/v79, a v80 participant publishes a **signed source PPO trajectory**, NOT a Critic value-head parameter delta. Each authenticated receiving peer independently computes the sampled action log-probabilities using its own **current Actor**, then trains only its own Critic using the non-clipped terminal-reward trajectory importance-sampling Monte Carlo value surrogate from v79. Actor gradients are not propagated from incoming trajectories.

The signed `KTR80` event contains: room/critic genesis fingerprint, originating peer + per-origin sequence, certified source Actor epoch, optional signed observed Actor journal cursor, bounded **tokenized prompt**, bounded sampled action IDs, source-reported behavior action log-probabilities, reward kind, reward value and reference token IDs. The receiver recomputes the reward for `exact` and `token_match` modes; Hugging Face scorer rewards are intentionally refused because they require a different proof of immutable scorer and inputs. Frame size <=64KiB and <=128 generated actions, with max input length <=4096; policy and size bounds are enforced.

The original v78 Critic certificate-ledger machinery remains to authenticate certified Actor epoch history and Critic head architecture/initialization, but in this v80 mode **no KVC78 Critic parameter deltas may be imported or broadcast**. The per-room trajectory lane is separately signed and stored durably via EventJournal's SQLite + fsynced binary archive. Anti-entropy uses paged `trajectory-frontier`/`trajectory-need` with bounded request and outgoing queues for reconnecting peers. Duplicate receipts do not update the value head again and equivocation is rejected.

Local receiver state is stored as a bounded `safetensors` value-head + AdamW moment snapshot, together with an applied `(origin,seq)` receipt in **one SQLite transaction**. On restart, the prior snapshot is restored and only pending events are eligible to be applied. A failure during the Critic update or persistence rolls back weights and optimizer moments to the prior local snapshot. This is a **local atomic receipt/state commit**, NOT a transaction across the Actor's signed journal and the Critic lane. A source's uncommitted latest local step can still be lost after a crash.

## Mathematical guarantee: precise scope

For truthful, complete behavior action probabilities `mu`, a fixed target Actor policy `pi` during evaluation, a fixed prompt distribution, identical autoregressive action space/tokenizer, unchanged reward semantics, and the support condition, v79's *unclipped trajectory IS* Critic surrogate has a **target-on-policy unbiased expected value gradient** (under integrability and a valid terminal-reward objective). The receiving peer supplies `pi` from its own Actor and obtains its own locally estimated Critic gradient. This is **not** sample-wise exact recovery of unobserved trajectories or future rewards; off-policy IS has potentially unbounded variance and finite precision guards may reject extreme samples. Training on already-received trajectories cannot guarantee the same global update order or parameter convergence across peers.

**Especially important:** A digital signature attests that a peer *claimed* a behavior probability and Actor journal cursor. It does **not** prove the probability actually resulted from the referenced Actor checkpoint or the sampling process. Authenticating that claim against a reproducible immutable Actor snapshot/replay would require an additional verification protocol. A Byzantine peer can lie about log probabilities, even with a valid signature. `KTR80` should therefore be used within an appropriately trusted room; source honesty is a mathematical assumption, not a cryptographic guarantee.

## Limitations and operating safeguards

- New rooms only; v78/v79 delta-sharing rooms are not auto-migrated.
- Only `train-elastic` / `train-live --mode elastic`; nonelastic commands fail CLI validation.
- Only `exact` and `token_match` verifiable PPO reward modes in v80. No remote Hugging Face scorer reward sharing yet.
- Only bounded, trainable Critic value heads; HF Critic backbone remains frozen (as in v79). The recipient pays the inference cost of its own current Actor and Critic for each imported sample. This implementation's synchronous recipient replay can transiently stall the event loop, especially on large HF models; a GPU-safe background scheduler remains future work.
- Independent Critic heads will **not** necessarily converge to the same weights, because peers can have different Actor targets, sample receipt order, and optimizers. This is distributed training by sharing experiences and recomputing per-peer gradients, not synchronous data-parallel training with a global optimizer.
- Local Critic state + receipt are atomically committed but not globally synchronized with Actor journal checkpoints, nor is the full remote PPO dataset checkpointed. A local Critic step not yet persisted can be lost on power failure.
- The v80 trajectory journal is capped to the smaller of the configured journal maximum or 512MiB; currently no safe archival GC/retention consensus. This limits long-running training until a bounded checkpoint/GC protocol is introduced.
- Receiving a peer's signed token IDs and references **exposes reversible tokenized prompts and reference data** to participants. Only use approved datasets whose content may be distributed. Trajectory sharing is opt-in.
- No real multi-host WAN endurance, GPU pretrained HF, or long-term Byzantine-stress experiments were performed in this environment. All direct replay tests use small toy CPU models and simulated signed peers.

## Main code paths

- `patchlib/trajectory_gossip_v80.py`: signed events, reward verification, journal, anti-entropy, local checkpoint and atomic receipt, replay.
- `patchlib/mixed_learning_v65.py`: sampled source trajectory export, receiving Actor probability re-evaluation, Critic-only IS update, v80 objective policy.
- `patchlib/elasticnet.py`: WebRTC message/control integration, epoch certificate gating, relay, local publish, restart and replay.
- `patchlib/cli.py`: new `--critic-rollout-gossip` opt-in and mode validation.
- `patchlib/mix_gui_v66.py`, `web/desktop_webview.js`, `web/desktop_webview.html`: v80 GUI selection and policy display.
- `tests/test_trajectory_gossip_v80.py`: source/recipient proof-of-concept, update/restart, malformed reward, tampering, equivocation, missing event recovery, source probability change, failed persistence rollback and CLI/GUI policy pinning.
