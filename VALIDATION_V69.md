# Kagami v69 validation

- `python -m pytest -q tests/test_hf_value_critic_v69.py`: 5 passed.
- `python -m pytest -q tests/test_hf_value_critic_v69.py tests/test_gui_mixed_critic_v66.py tests/test_hf_reward_v67.py tests/test_hf_mixed_datasets_v68.py tests/test_mixed_learning_v65.py tests/test_model_contract_v64.py tests/test_webview_gui.py tests/test_gui.py`: 59 passed (before final small optimization and added token-classification test).
- `python -m pytest -q tests --ignore=tests/test_large_room_bounded_topology_v17.py`: 800 passed, 3 skipped, 0 failed (before a small subsequent optimization that removes unneeded logit features from HF Critic mode; its focused tests pass).
- Full suite attempted: did not complete within local test execution constraints. A 4096-peer topology test consumed significant time; no full-suite pass claim.
- JavaScript `node --check web/desktop_webview.js` and Python `compileall patchlib`: pass.
- Tests cover HF loader/diagnostic flags, scalar trained head requirement, frozen backbone vs trainable value head, per-prefix PPO gradient, separate tokenizer, local optimizer snapshot/restart, room/revision mismatch rejection, NaN/missing-head failures, GUI->CLI, HF policy pin, old GUI and JSONL compatibility, token-classification values, multi-purpose PPO/Reward coexistence.
- No external pretrained HF Critic was downloaded or run with CUDA/MPS. End-to-end GUI clicks in native WebView not exercised.
