# Kagami v75 — paired Actor/Critic local checkpoint & async snapshot writer

## What changed

- `patchlib/paired_checkpoint_v75.py`: captures immutable CPU copies of **Actor trainable LoRA/LoHA factors**, Actor AdamW moments, **PPO Critic value-head parameters**, and Critic AdamW moments at the **same awaited optimizer boundary**. A single safetensors file contains both, with format/version, room/critic identity, adapter layout, local steps, genesis, and durable signed-journal tip.
- The writer computes a SHA-256 digest over the actual tensor names/dtypes/shapes/bytes on a background thread, writes a unique temp file, `fsync`s its bytes, atomically replaces the paired file, then `fsync`s the parent directory. In-memory training no longer holds the adapter lock for that file I/O. Only tensor **capture** (GPU→CPU copies) pauses that model briefly; writer may still consume CPU/I/O resources.
- Restore verifies **the exact current journal tip**, genesis, immutable critic HF SHA/room, adapter layout, tensor integrity, shape/dtype, all AdamW moments and step counters before changing the Actor or Critic. Tensor transfers to their target devices are staged before either state is applied. On mismatch, it **refuses** mixed recovery instead of silently loading a stale value function.
- `train-elastic` integration: when a local PPO data source is present and AdamW is in use without `--disk-optimizer-dir`, the paired snapshot is used at configured durable/working intervals and at clean shutdown. CLI rooms get a journal-scoped Critic save directory automatically if none was specified. Interrupted previous v74 standalone Critic files remain a migration starting point when no paired checkpoint exists.
- The `AsyncPairedSaver` allows **at most one** disk-writer task; optional snapshots are skipped under backpressure rather than growing an unbounded queue. Its force mode drains pending work and writes a final capture on **clean exit**.
- On certified epoch/adaptor switches, the prior generation's old standalone Critic state is not silently loaded under the new Actor baseline; the new generation uses its own checkpoint.
- Legacy PPO-only CLI, local training, HF Critic/Reward, unrelated P2P protocol, and existing non-PPO snapshots remain unchanged.

## Important limits

**Not a globally atomic P2P transaction.** The signed SQLite journal may commit an event before its local paired checkpoint has finished publishing. A process crash in that window may leave a newer journal tip than the last paired state. v75 detects this **and refuses automatic restore**. It does not roll back signed events, invent missing Critic optimizer states, or automatically replay Critic gradients. This is **fail-closed correctness**, not fully autonomous recovery from every crash window.

The actor optimizer must be standard AdamW with tensor moments. LoRA-Muon and disk/NVMe optimizer snapshots need their own atomic factor/moment protocol and continue on the pre-v75 path unless there is already a paired checkpoint, in which case changing incompatible settings is rejected. 256 MiB is the bounded combined CPU tensor budget for paired snapshots; larger model states need sharded snapshots. A writer backlog can skip optional saves; clean shutdown forces the final one. The experimental PPO Critic remains **local to each peer** and is not included in the signed actor updates.

Actual Hugging Face GPU or WAN runs and induced power loss under a real signed swarm have **not** been performed. No claim of fully interruption-free millions-of-peer journaling, consistent distributed Actor/Critic updates, or zero-pause capture is made.

## Launch

Use the existing `train-elastic` workflow and GUI settings with PPO + AdamW. The joint snapshot is stored beside the active epoch journal as `*.paired-v75.safetensors`. No new GUI field is needed. If you restart at an unmatched signed tip, inspect the mismatch; do not delete recovery files without keeping a copy and choosing an explicit recovery policy. For very large states or incompatible optimizers v75 does not enable paired saves.
