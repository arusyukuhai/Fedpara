# pywebview bridge hotfix (2026-10-21)

Problem: the local HTML loaded, but UI operations displayed "Pythonブリッジがまだ使用できません" and the `pywebview.api` methods were missing.

Root cause: the prior Content Security Policy specified `script-src 'unsafe-inline'` and omitted `unsafe-eval`. pywebview's JavaScript bridge generates the exposed callable methods with `new Function(...)`, which CSP blocks without `unsafe-eval`.

Changes:

- `patchlib/webview_gui.py`: allow `script-src 'unsafe-inline' 'unsafe-eval'` **only for the bundled local UI**. The other restrictions (no connections, no external scripts, no forms/frames) remain.
- `web/desktop_webview.js`: wait until **all declared bridge methods** are callable. Handle delayed injection, a ready event before method registration, and ready-event absence. Show a diagnostic after timeout rather than fail instantly.
- `web/desktop_webview.html`: include a `Python接続を再試行` action without reloading or opening an HTTP port.
- `tests/test_webview_gui.py`: check bridge-compatible CSP.
- `tests/webview_bridge_smoke.js`: simulate both ready-event and silent delayed method injection.

Install:

```sh
cd fedpara
python -m pip install -r requirements-gui.txt
python -m patchlib.cli gui
```

If still failing on macOS, run `PYWEBVIEW_LOG=DEBUG python -m patchlib.cli gui` and inspect traceback. Opening `web/desktop_webview.html` directly in a normal browser will not provide a Python bridge.

Validation: `python -m pytest -q` (75 passed), `node --check web/desktop_webview.js`, `node tests/webview_bridge_smoke.js` (pass). Actual macOS WKWebView execution was not available in the build environment.
