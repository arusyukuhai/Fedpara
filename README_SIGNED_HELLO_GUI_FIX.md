# Fedpara — Signed Hello & GUI log repair

This build is a **focused repair** of the 2026-10-21 `fedpara_connect_ui_stability_fixed.zip` release. It retains all prior model checkpoints, user journals, profiles, and signatures.

## What changed

* Incoming Ed25519-signed hello can legitimately be delivered before the receiver's own local `DataChannel.on('open')`. Authentication state now initializes when the channel is attached, not when its open callback fires.
* An obsolete channel's close or message callback cannot alter the replacement channel's state.
* An outgoing hello that fails once is retried **before** sending any heartbeat, inventory, or model-update control frame; it is no longer silently skipped.
* Verified self-certified public keys are stored in the peer trust registry; no keys need manual provisioning.
* If the hello signature, peer ID, or base-model fingerprint really is incompatible, the transport aborts promptly and reports `PEER BLOCKED`, rather than blindly reconnecting forever. **No authentication or genesis checks are disabled.** Restart after resolving the incompatible model/settings or versions.
* `logs` now has native *Copy logs* and *Save logs* buttons, plus a *Select visible logs* button. macOS uses the built-in `pbcopy` command through Python, not unsupported WebView clipboard permissions. You can also find full worker logs under `.fedpara-gui/rooms/<room-id>/profiles/.../*.gui.log`. Diagnostic exports go to `.fedpara-gui/diagnostics/`, mode 0600.
* While text is selected in the on-screen log, auto-refresh avoids replacing the selection.

## Upgrade

1. Stop **both** desktop apps and their training child processes. Stop the Nim signaling server only if it needs restarting; this patch does **not** change its protocol.
2. Unzip and copy the `fedpara/` sources into your project, preserving `.fedpara-gui/`, SQLite databases, `genesis.fstate`, personal keys, datasets, and model weights. Do not overwrite or delete the state directory.
3. Both windows must run this new version; mixing old hello ordering and new settings can cause compatibility errors.
4. Start the signaling server as usual and run: `python -m patchlib.cli gui` in two terminals. Join the same room ID. Check the logs for `SIGNED HELLO OK`, `active=2` and `SNAPSHOT READY`. Connection counts or a hello success do **not** alone guarantee a correct model sync.
5. Use **ログをコピー** in each GUI, or **ログをファイルに保存**. Before sharing, redact any room IDs, access tokens, private URLs, dataset credentials, or personal file paths.

## Diagnosing a persistent connection failure

* `SIGNED HELLO FAILED`: invalid first control frame or incompatible handshake versions. Full error is displayed in the log.
* `GENESIS MISMATCH`: clients do not agree on the initial tensor content, model revision, architecture, rank, or pinned room definition. Do not merge their updates.
* `PEER BLOCKED`: an authenticated-control/identity check failed and auto-retry has been intentionally stopped. Resolve the underlying mismatch first.
* `ice=checking`, `rtc=connecting` but **no** signed hello: transport/ICE/DTLS problem, not an Ed25519 signature problem.

To test ICE + hello without loading model data, run `python scripts/check_live_p2p.py` on the Mac while Mummy is running. A passing diagnostic against a synthetic pair does not establish that the real room has matching base-model genesis.

## Verification

Python compile and JS syntax checks pass. Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_connection_stability.py tests/test_cold_join_guard.py tests/test_elastic.py tests/test_hello_clipboard_regression.py tests/test_webview_gui.py tests/test_roomid.py
node tests/webview_bridge_smoke.js
node tests/webview_click_smoke.js
```

Live two-window macOS + WebRTC and Nim compilation still require testing on the user's machine. No claim of live connection success is made.
