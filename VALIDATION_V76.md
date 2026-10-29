# v76 validation

- New `tests/test_paired_recovery_v76.py`: exact-tip round-trip, advanced signed prefix (empty and nonempty prefix), no stale Actor/Critic/optimizer application, persistent reset and restart idempotence, fork rejection, rollback rejection, corrupt payload rejection, cross-room rejection, requirement for prior signed replay, simulated write crash and retry, absent snapshot.
- v75 local paired checkpoint tests included to ensure compatibility with its format.
- `train-elastic` connects replay-first recovery after `ContinualElasticEngine.replay()`.
- Validation is on CPU with synthetic P2P journal rows; `verified_replay` is a contract enforced in code, and the production caller supplies it only after the signed replay call returns successfully. Synthetic rows themselves are not Ed25519 signatures, so tests do **not** prove on-the-wire signature verification.
- Full pytest outcome is recorded in `PYTEST_V76_FULL.log`. No real Hugging Face large-model checkpoint, CPU/GPU mixed device or genuine WAN crash test was performed.
