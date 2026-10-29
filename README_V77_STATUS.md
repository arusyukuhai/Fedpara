# Kagami v77 — Distributed PPO Critic value-head training

## Summary

Opt-in `--distributed-critic` makes the PPO **value-head parameters** a second, signed, durable P2P training stream, independent of the Actor's `FPEL1` LoRA/LoHA stream. New rooms can enable it in **TRAINING → VALUE CRITIC → Critic training → P2P distributed**. Existing v76 rooms retain local-only Critic.

Each PPO peer trains its Critic head with the existing PPO GAE/value loss. After AdamW steps, the **actual parameter delta** (CPU float32) is signed, committed to its separate event journal and sent over the existing authenticated WebRTC data channel. Remote peers verify the identity, model/initial-weight fingerprint, origin and sequence, payload digest, tensor shapes and finite values before adding the delta. Repeated deliveries are not applied twice. An `origin/seq` equivocation fails closed. Reconnection repairs missing per-origin sequences through bounded signed `critic-frontier` and `critic-need` controls. This is **asynchronous additive gossip**, not synchronous data-parallel SGD or averaged AdamW.

For Hugging Face Critics the *pretrained scalar head* is distributed (backbone remains frozen); for local MLP and Linear critics the entire learnable local value head is shared. There is no Self-Critic reward scoring: Critic = PPO state-value estimator; Reward Model remains separate. Local AdamW optimizer moments are never transmitted. Foreign updates invalidate old PPO rollouts and reset the Critic AdamW moments to avoid carrying stale moments into the merged head.

## Enabling

- Create a **new** room with a PPO dataset and positive PPO loss weight. Every Critic-gossip participant currently requires a PPO dataset; PT-only peers are not yet Critic relays.
- Choose **TRAINING → PT + SFT + PPO → VALUE CRITIC → Critic training: P2P distributed**. This setting is pinned into the room's immutable objective policy. Local-only remains the default for backward compatibility.
- Or add `--distributed-critic` to `train-elastic` and all participating PPO peers. The independently signed Critic archive lives alongside the Actor journal as `<journal>.critic-v77.sqlite3` (and sidecar). No Critic data is included in Actor `FPEL1` events.
- An existing local-only v76 paired checkpoint cannot be silently migrated into distributed Critic; start a new v77 room. A restart of the same v77 room replays its independently signed Critic archive from the pinned pretrained/deterministic common base.

## Protocol details and safety

- Wire `KVC77\0`: domain-separated signature over **all event metadata and payload SHA-256**; safetensors delta payload with 128-tensor/1 MiB uncompressed budget and 2 MiB wire cap. Identity is pinned to actor genesis, room Critic configuration, trainable shapes/dtypes, and initial tensor digest. Unsafe/unsigned Critic mode is forbidden.
- The existing `EventJournal` is used for durable append, `origin/seq` idempotency and historical replay. This is a **separate journal** so Critic events never enter Actor epoch/rebase proof logic.
- Only authenticated v77-capable neighbors exchange Critic events; periodic bounded frontiers request missing indexed ranges and enable late arrivals to catch up. On reception, `critic_step_lock` plus `adapter.lock` exclude simultaneous local Critic optimization and application.
- Local signed critic updates are journaled **once** after each Critic optimizer step, but are **not** applied twice (the optimizer already modified the local head).
- During restart, the v76 Actor replay/recovery still validates Actor events. The v77 Critic is **separately rebuilt from its pinned baseline + all signed Critic events**, not restored twice from an older paired value snapshot. Critic AdamW moments are reset on replay/foreign updates. Existing v75 paired snapshots still capture local states, but are not dual-journal atomic commits.
- Signed Critic events record their producer's Actor journal-tip metadata, but v77 does **not** enforce that all participating Actors had the exact same tip at training time. PPO staleness is bounded by the existing v73 stale-rollout defenses; no V-trace or mathematically exact off-policy correction is claimed.

## Limitations (important)

1. **No automatic Critic cross-generation rebase.** `--distributed-critic` with generation-v1 committee / generation-v2 trainer is rejected explicitly rather than mixing old/new-generation value updates.
2. **No simultaneous Actor+Critic journal transaction.** Two independently durable histories are replayed safely, but cannot recover unpublished local gradients or exactly reproduce every optimizer step after a power loss.
3. **No per-peer privacy-preserving secure aggregation or poisoning-proof robust consensus.** Signatures detect forgery/equivocation, not a legitimately authenticated peer intentionally sending misleading finite gradients. Values are bounded per update.
4. Only the **trainable Critic value head** participates. Fine-tuning the Hugging Face Critic backbone (LoRA/LoHA) is not implemented.
5. Current event fanout is sparse, gossip is eventually consistent under continued connectivity, and CPU tensor arithmetic can have tiny order-dependent floating-point differences. Strict bitwise convergence at all peers is not guaranteed.
6. No physical WAN/4,096-peer/GPU/real Hub Critic distributed training run in this environment. Tests cover real PPO gradients on a tiny PyTorch LM, signed lanes, replay, gap repair and failure cases.

## Where to read

- `patchlib/distributed_critic_v77.py` — wire and ledger
- `patchlib/elasticnet.py` — WebRTC routing, anti-entropy, local step publication and epoch guard
- `patchlib/mixed_learning_v65.py` / `patchlib/cli.py` — immutable policy and local value gradient capture
- `patchlib/mix_gui_v66.py`, `web/desktop_webview.js`, `web/desktop_webview.html` — GUI selection
- `tests/test_distributed_critic_v77.py` — reproductions and multi-peer convergence
- `VALIDATION_V77.md` — exact test status
