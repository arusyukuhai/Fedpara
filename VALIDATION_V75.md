# Kagami v75 validation

## Verified tests

- 88 passed, 0 failed: `tests/test_paired_checkpoint_v75.py`, `test_audit_v74.py`, `test_mixed_learning_v65.py`, `test_gui_mixed_critic_v66.py`, `test_hf_value_critic_v69.py`, `test_hf_reward_v67.py`, `test_ppo_correctness_performance_v73.py`, `test_ppo_pipeline_v72.py`, `test_hf_mixed_datasets_v68.py`, `test_continual_elastic.py`, `test_epoch_v57_trainer.py`, and `test_live_disk_optimizer_v59.py`. The log is `VALIDATION_V75_TARGETED_PYTEST.log`.
- New v75 tests cover safe reset on signed-generation change and both optimizers' exact tensor roundtrip, stale journal rejection without partial actor mutation, tampered finite tensor bytes, invalid payload/shape, mismatched room/genesis, failed `os.replace` retaining previous snapshot, and deliberately slowed writer with a bounded queue and immutable capture.
- `python -m compileall -q patchlib` passed.

## Not completed

Two full-suite attempts encountered long existing network/topology simulations and hit execution-time limits: the first reached ~49% without a reported failure, and a second excluding several heavy modules reached ~25% without a reported failure. These are **not** full-suite passes and cannot establish long-running P2P reliability. No large real HF model, GPU, or WAN swarm test was run.

## Model consistency contract

The entire paired file is atomically published; an individual journal append and this paired file are **not** atomically committed across SQLite and the filesystem. v75 explicitly refuses mismatched journal-tip restores. The async writer only decouples safetensors serialization/fsync; bounded CPU capture and forced shutdown save still incur latency.
