# Kagami v67 validation

- New tests cover immutable revision validation, sequence scorer inference, chat formatting, frozen model parameters, missing reward head rejection, score shape/nonfinite errors, GUI/CLI policy pinning, a PPO rollout using external model rewards (no reference field), repeat rollout reward caching, public Hub metadata discovery, and real GUI bridge → CLI flags.
- No actual Hugging Face reward checkpoint was downloaded in this container. `transformers` and official model weight runtime are not part of this mocked integration test.
- Checkpoints requiring `trust_remote_code=True` are deliberately not executed.
- Full pytest suite (after implementation and Hub discovery wiring): **806 passed / 3 skipped / 0 failed**, 87.00s.
- Focused reward model + previous mixed-critic/PPO regression tests: **29 passed**, 0 failed.
- `python -m compileall -q patchlib` and `node --check web/desktop_webview.js` passed.
