# Kagami v66 verification — 2026-10-27

**Confirmed in the current final tree:** 83 focused regression tests PASSED, 0 failed (GUI, mixer, HF datasets/compatibility, CLI, room persistence, live GUI, elastic and epoch). An additional focused run of the 11 v66-specific integration tests PASSED. The counts overlap and must not be added together.

**Full suite:** attempted multiple times, not completed within the execution time limit. The latest run got through approximately 54% without recording a failure; no full-suite pass claim is made.

## Verified behavior

- Real WebView form state -> restricted Python bridge -> controller input validation -> `train-live` CLI argv containing separate SFT/PPO JSONL and Critic flags. PT dataset references are taken from the immutable HF room manifest.
- Critic (independent local MLP or linear value function) updated by a toy PPO loop and saved/restored including optimizer AdamW moments, without inclusion in signed LoRA/LoHA P2P updates.
- Configurable Critic initialization from a data-only Kagami safetensors checkpoint; unknown structure/identity is rejected.
- Per-peer local dataset paths and critic checkpoint paths are excluded from the published room manifest and shared objective policy.
- Shared mixed-objective policy remains immutable; PT-only and SFT-only peers can participate in a pinned room without changing the room-wide objective semantics.
- GUI settings and pinned policy survive app restart; dataset first-row schema is checked before starting the worker.
- `python -m compileall -q patchlib` and `node --check web/desktop_webview.js` passed.

## Not verified

No actual native OS WebView click test, real large HF model/GPU PPO, long-duration 4096-peer WAN evaluation, arbitrary external reward-model integration, or full suite completion.
