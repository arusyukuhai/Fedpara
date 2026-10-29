# Kagami v76 — verified signed-log replay and automatic paired-state recovery

## Fixed: crash between durable journal append and paired checkpoint publication

v75 correctly refused to load an Actor/Critic pair if the signed SQLite journal had moved beyond the pair's saved tip, but it then permanently blocked local training. v76 introduces `patchlib/paired_recovery_v76.py`, invoked during `train-elastic` startup **after** the existing authenticated `engine.replay()` reconstructs the signed Actor state.

When the journal has advanced by a **verified append-only extension** of the stored checkpoint boundary:

1. Validate snapshot identity (genesis, adapter layout, room/critic identity), step counters, all tensors, AdamW moment shapes, and SHA-256 payload digest. No stale tensors are applied.
2. Check that the previous tip's complete `(origin, seq, digest)` entry still occurs at the same journal position, and that the current event count is strictly larger.
3. Retain the Actor **already reconstructed from the committed, signed journal**. Use newly initialized Actor optimizer and PPO Critic/optimizer. The stale, uncommitted local states are explicitly discarded.
4. Before the first new training step, capture the clean Actor/Critic pair at the *new* journal tip. Write and fsync it to a unique temporary file, replace the old pair atomically, then fsync the directory.
5. If the process dies during this publication, the old file remains and the procedure can retry. If it dies after publication, the new file matches the current journal tip and restores normally.

No signed event is invented, rewritten, deleted, or rolled back. No stale gradients or optimizer moments are silently replayed on a different base.

## Fail-closed cases

- Journal shorter than the checkpoint (loss/rollback), or forked/divergent event at the checkpoint boundary.
- Incompatible genesis, model factor layout, or Critic identity.
- Tampered/corrupt checkpoint bytes, missing/malformed AdamW state, or unsafe symlink/oversized file.
- Failure of signed journal verification/replay prior to recovery, or failure to publish the replacement snapshot.
- Unsupported optimizer formats and non-PPO rooms retain the v75 behavior.

## What automatic recovery **does not** promise

**This is not an atomic transaction across SQLite and the safetensors file.** Instead, it is *redo of authenticated committed events + bounded local-state reset* on restart. Critic updates and optimizer moments that have not been included in a compatible checkpoint cannot be reconstructed from signed Actor events; the reset may lose progress and alter the continuation trajectory. A previous checkpoint's `steps` is a recorded local-step count, **not an exact measure of how many steps were lost**.

This is a recovery from process death during the journal-versus-checkpoint crash window, not fault-tolerant distributed exact-once PPO and not a zero-step-loss guarantee. True reproducible local PPO replay would require persistent rollout observations/rewards/random state and a separate fully ordered actor+critic training log. Network identity and certified epoch replay checks still apply. Actual power-loss, real GPU HF models, large-model long-WAN and 4,096-peer runs remain unverified.

## Usage

No new flag: restart the same `train-elastic` PPO+AdamW room with the same room, tokenizer, signed journal, and pinned Critic policy. The log prints `PAIRED AUTO-RECOVERY` with previous and current event counts if an append-only extension requires a reset. A correctly matching pair restores all supported moments without reset. A fork or corrupt file fails loudly rather than continuing. Existing `*.paired-v75.safetensors` paths remain compatible.

See `VALIDATION_V76.md` for explicit crash-window scenarios and tests.
