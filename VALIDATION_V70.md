# v70 validation

- Python package `compileall`: **PASS**.
- `node --check web/desktop_webview.js`: **PASS**.
- `pytest` focused modules (v70 discovery, v68 datasets, v67 Reward, v69 Critic, v66 GUI, v31 Hub metadata, WebView, Tailwind GUI, GUI): **68 passed**.
- Purpose discovery: 5 task defaults tested; broad opt-in; invalid requests rejected; metadata-only queries and bounded search counts verified; partial/network failure handled.
- Headless Chromium smoke with a simulated pywebview bridge: **PASS** (purpose-specific dialog display, SFT repo pin to immutable SHA, config/split transfer, Critic selection, no JS page errors). Not an OS WebView or a live Hub backend test.
- Wide pytest run excluding 4,096-peer topology stress file advanced beyond 70% without a visible failure, then stalled in existing long-running cases and was stopped. **Not counted as a passing full-suite run.**
- Existing legacy Node smoke scripts rely on a DOM mock without `querySelectorAll`, so they fail on the existing app startup before reaching the new GUI. Real DOM interaction was exercised using Chromium instead.

Privacy/safety: No custom repository scripts, checkpoint weights or dataset rows are fetched by the discovery endpoint; applied selections still go through v68/v67 commit inspection. No auto-inference of scalar-head compatibility from repo names.
