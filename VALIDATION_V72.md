# Kagami v72 validation notes

- Main focused regression run after final changes: **122 passed / 0 failed**, `PYTEST_V72_FOCUSED.log`.
- New v72 regression tests: **7 passed** (included in 122).
- Static checks: Python `compileall` and Node.js `node --check web/desktop_webview.js` passed.
- Full suite was attempted but timed out. A verbose run passed up to 98% before reaching the long-running large-room overlay tests. This does not constitute a full-suite pass.
- Synthetic CPU benchmark values are in `BENCHMARK_V72.json` and can be regenerated with `PYTHONPATH=. python benchmark_v72.py`.
- There was no live GPU, physical model/Reward checkpoint, or actual Hugging Face network test.
