# Kagami v65 — verification

Test environment: Python 3.11 / PyTorch 2.10.0+cpu, pytest 9.0.2; no real HF Llama/DeepSeek weights, WAN peers, or multi-GPU cluster provided.

```
cd kagami-v65
PYTHONPATH=. pytest -q --disable-warnings
787 passed, 3 skipped in 88.75s
```

Focused command:

```
PYTHONPATH=. pytest -q tests/test_mixed_learning_v65.py \
  tests/test_model_contract_v64.py tests/test_cli.py \
  tests/test_continual_elastic.py tests/test_epoch_v56_autonomy.py \
  tests/test_epoch_v57_trainer.py tests/test_epoch_v58_frozen_export.py \
  tests/test_factor_ledger_v62.py tests/test_live_disk_optimizer_v59.py \
  tests/test_paged_mutable_v61.py tests/test_webview_gui.py tests/test_livefed.py
85 passed in 10.94s
```

New tests (`tests/test_mixed_learning_v65.py`) cover:

1. Mixed PT+response-masked SFT+clipped PPO actually backpropagates into an existing signed UniversalAdapter on every microstep; the independent critic is updated locally only.
2. SFT prompt masking and response-only supervision for a causal logits model.
3. PPO samples using current policy, freezes old log probabilities, constrains update reuse, computes finite clipped loss and KL, updates critic estimates.
4. Objective-protocol hash excludes private local data paths and changes when clip coefficients change.
5. Invalid JSONL, missing ground-truth references, incompatible task and dangerous hyperparameters fail early.
6. Legacy PT-only run_step loss remains unchanged.
7. Shared JSON room policy lets independent PT, SFT and PPO peers retain the same genesis contract, while mismatching parameters are rejected.
8. SFT-only training uses no useless PT batch fetch.
9. Full `cli.train` operation with PT/SFT/PPO produces a valid `.fpatch` update after multiple actual optimizer steps.
10. A fabricated numeric reward without reference/ground truth is rejected (prevents pretending an off-policy constant score is a valid on-policy reward).

**Not established by these tests**: reward-model RLHF, language-model-as-judge, general human preference optimization, a learned deep critic with distributed state, actual full 300B training, stable 4,096 WAN peers, live mixed training with external HF checkpoints. The PPO implementation uses an on-policy sample, GAE, a small local value estimator and a clipped actor ratio; it is a restricted algorithm for verifiable text answers.

The GUI does not yet expose the new objective settings: launch through CLI. Older room genesi cannot silently enable new objectives in-place. Mixed training adds substantial computation per microstep compared with PT alone.
