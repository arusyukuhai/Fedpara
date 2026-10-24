# Fedpara Desktop — Embedded WebView (Python + HTML/CSS/JS)

This replaces both the temporary standalone browser dashboard and the default Tkinter UI. `python -m patchlib.cli gui` opens a **desktop application window** using the operating system's native WebView: WebKit on macOS, WebView2 on Windows, and an appropriate WebKit/Qt backend on Linux.

The dashboard is rendered from bundled HTML/CSS/JavaScript, and actions invoke **explicit Python methods** via `window.pywebview.api`. It is like an Electron/Tauri-style desktop shell, but uses Python and pywebview instead of a Node/Rust runtime. It is **not** a standalone Electron/Tauri package.

## 1. Installation (macOS)

```bash
cd fedpara
python -m pip install -r requirements-patches.txt
python -m pip install -r requirements-gui.txt
python -m patchlib.cli gui
```

You do **not** need Tkinter, a browser tab, Flask, FastAPI, or an HTTP port for this window. The WebView HTML is provided as an **in-memory string** (not a filesystem path): pywebview's local static HTTP server is not started by this GUI. macOS requires a normal GUI desktop session; this will not render from a headless SSH terminal.

If you already have the Python training dependencies, install only the GUI package with `python -m pip install -r requirements-gui.txt`.

The old Tkinter prototype can still be used with `python -m patchlib.cli gui-desktop` if Tkinter is installed. The old HTTP browser dashboard has been removed from this build.

## 2. What the window does

- Generate or enter a room ID, specify the signaling WebSocket (`ws://127.0.0.1:8080/ws` locally, `wss://` for remote use).
- Choose a Hugging Face model/revision and text training task.
- Add up to 12 Hugging Face **dataset sources** with repo, config, split, revision, column mapping, sampling weight, and shared seed; import sources from JSON.
- Fetch and inspect a room's model/dataset manifest via its room ID and signaling connection. Verify its SHA-256 fingerprint with the room creator over a separate channel.
- Register local `.py`/`.zip` custom model source as a **hash-pinned, inert artifact**. Nothing is imported or executed just by registering or joining a room.
- Save a room manifest locally and launch `train-live` with the chosen local `.fstate` initial checkpoint, LoRA/LoHA and rank; stop the child process and view its logs.

**Important:** a new room manifest is advertised to peers when a training peer actually connects to the Mummy signaling server. Saving the manifest in the GUI alone does **not** upload it to Mummy. Each participant still needs an initial checkpoint; this UI does not implement automatic seed-weight distribution.

## 3. Architecture and trust boundary

```text
Fedpara Desktop window (HTML/CSS/JS)
       | pywebview.api (fixed methods, no arbitrary command execution)
Python controller -> validated JSON room descriptors
                  -> vetted code hash registry (not execution)
                  -> subprocess: patchlib.cli train-live (fixed argv, no shell)
                                      | WebRTC DataChannels and Mummy signaling
                                      | downloads HF datasets directly at each peer
```

All HTML, JavaScript, and CSS are bundled and inlined from local trusted project files. The GUI **does not bind localhost** or any other HTTP port. No web pages are loaded from arbitrary URLs within the WebView. The static HTML uses a restrictive CSP with a carefully scoped `unsafe-eval` exception **required by pywebview**, which dynamically registers its API with `new Function()`. External script loads and browser-side network requests remain disallowed. Signaling and dataset downloads are initiated by separate Python training / room-join functions when explicitly requested, not by `fetch()` from page JavaScript.

The WebView is a privileged UI. **Never load untrusted HTML into it**, as scripts could invoke its Python API. The room metadata is displayed using DOM `textContent`, never `innerHTML`. Neither room manifests nor uploaded model source files are injected into the page as HTML or JavaScript. Imported model code is not executed by the GUI; when intentionally executed for training, it must use the separate isolation mechanism described in `README_CUSTOM_STATE.md`.

Room IDs are bearer invitations, not access control. DTLS encrypts live WebRTC transport; update validation, malicious-peer resistance, and TURN usage limits remain separate security concerns.

## Creating the initial checkpoint

The initial-state field requires a real `.fstate` FILE, **not** the repository directory and not an empty string. From the project directory, run:

```bash
python -m patchlib.cli state-export --model sshleifer/tiny-gpt2 --revision main --peer-id bootstrap --output genesis.fstate
python -m patchlib.cli state-inspect genesis.fstate
```

Paste the absolute path to `genesis.fstate` into the GUI. If you use a different model/revision in the GUI, use that same model/revision in `state-export`.

All peers must start from the **same checkpoint bytes**; do not separately export files for each peer, because checkpoint manifests/ZIP bytes may differ. Distribute one canonical checkpoint (or assemble a verified shard transfer into that exact file) before starting `train-live`. Automatic initial-state fetching on joining a room is **not yet implemented**.

## 4. Troubleshooting

- **「Pythonブリッジがまだ使用できません」 / bridge unavailable**: The previous release omitted the CSP `unsafe-eval` allowance required by pywebview. Use this patched archive, reinstall pywebview with the same Python interpreter, and launch the actual desktop app (not the HTML file in Safari/Chrome). The UI now waits for bridge functions, and has a **Python接続を再試行** button with meaningful diagnostics.
- Check the installed pywebview version with `python -m pip show pywebview`. If the bridge still does not appear, capture the terminal exception/traceback and your macOS version. Use `PYWEBVIEW_LOG=DEBUG python -m patchlib.cli gui` to get debug logging.


- **`GUI requires pywebview`**: `python -m pip install -r requirements-gui.txt` using the same Python used to launch the app.
- **macOS UI never opens**: start from a logged-in graphical session; check your Python/macOS GUI compatibility and dependencies. This archive was not tested on a live macOS machine.
- **`Address already in use`**: this WebView GUI does *not* use port 8765. Check for an old browser dashboard or Mummy itself (often `8080`) still running.
- **Room cannot load**: verify the Mummy signaling server URL is reachable and the room has been published by an active peer. Remote unencrypted `ws://` is rejected.
- **Training does not start**: select an existing local `.fstate` and check the GUI log area; missing model/dataset credentials and dependencies must be resolved locally.

## 5. Tests and limitations

`python -m pytest -q tests/test_webview_gui.py` checks the in-memory page, fixed JS bridge, room metadata, code registration, native window startup arguments, absence of TCP binding by the UI, and CLI routing without a GUI display. `node --check web/desktop_webview.js` verifies JavaScript syntax, and `node tests/webview_bridge_smoke.js` tests delayed API injection (both with and without the native ready event).

The actual macOS WebView renderer, native window interaction, Mummy compilation, and end-to-end WebRTC learning are **not verified** in the test container.
