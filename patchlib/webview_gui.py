"""Embedded WebView desktop application (Electron/Tauri-like architecture).

User interface: bundled HTML/CSS/JS rendered by the operating system's WebView.
Backend: a limited pywebview JS-Python bridge, no HTTP socket, no localhost API.
Remote model code is never executed from the UI; only content hash is registered.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

from .webview_controller import WebviewController

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'web'


class FedparaBridge:
    """Minimal JS API; exposed method names are intentionally explicit.

    pywebview calls exposed methods on background threads. WebviewController
    serializes operations needing shared state using its internal RLock.
    """

    def __init__(self, controller: WebviewController | None = None):
        self._controller = controller or WebviewController()

    def ping(self, _payload=None):
        return {'ok': True, 'backend': 'Fedpara Python', 'python': sys.executable}

    def defaults(self, _payload=None):
        return self._controller.settings()

    def new_room(self, _payload=None):
        from .roomid import create_room_id
        return {'room': create_room_id()}

    def register_room(self, payload: dict):
        return self._controller.register_room(payload)

    def list_rooms(self, payload: dict):
        return self._controller.list_rooms(payload)

    def save_room(self, payload: dict):
        return self._controller.save(payload)

    def join_room(self, payload: dict):
        return self._controller.load_room(payload)

    def register_code(self, payload: dict):
        return self._controller.upload_code(payload)

    def start_training(self, payload: dict):
        return self._controller.start(payload)

    def training_status(self, _payload=None):
        return self._controller.status()

    def copy_logs(self, _payload=None):
        return self._controller.copy_logs()

    def save_logs(self, _payload=None):
        return self._controller.save_logs()

    def start_benchmark(self, payload: dict):
        return self._controller.start_benchmark(payload)

    def benchmark_status(self, _payload=None):
        return self._controller.benchmark_status()

    def stop_training(self, _payload=None):
        return self._controller.stop()


def compose_html() -> str:
    """Embed static UI assets inline so even pywebview's file server is unused."""
    template = (WEB / 'desktop_webview.html').read_text(encoding='utf-8')
    css = (WEB / 'desktop_webview.css').read_text(encoding='utf-8')
    js = (WEB / 'desktop_webview.js').read_text(encoding='utf-8')
    # Static trusted build artifacts, never include room/user-supplied content.
    if '</style' in css.lower() or '</script' in js.lower():
        raise ValueError('unexpected closing HTML tag in a bundled resource')
    # pywebview dynamically builds its JS API with `new Function(...)`; it needs
    # 'unsafe-eval'. Without it the HTML loads, but Python methods never appear.
    # No untrusted HTML is interpolated; the page embeds only vetted local assets.
    # JS calls to Python do not require a socket/HTTP connection.
    meta = '''<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'; object-src 'none'; frame-src 'none'">'''
    template = template.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n'+meta, 1)
    template = template.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)
    template = template.replace('</body>', f'<script>\n{js}\n</script>\n</body>', 1)
    return template


def run_webview(*, workdir: Path | None = None, webview_module=None):
    """Launch native window. Import pywebview lazily; unit tests can use a stub."""
    if webview_module is None:
        try:
            import webview as webview_module
        except ImportError as exc:
            raise RuntimeError(
                'GUI requires pywebview. Install: python -m pip install "pywebview>=5,<7"'
            ) from exc
    controller = WebviewController(workdir)
    bridge = FedparaBridge(controller)
    window = webview_module.create_window(
        'Fedpara — Distributed AI Training',
        html=compose_html(),
        js_api=bridge,
        width=1100,
        height=850,
        min_size=(760, 570),
        resizable=True,
        focus=True,
        background_color='#080e0b',
    )
    try:
        debug = os.environ.get('FEDPARA_GUI_DEBUG', '').strip() == '1'
        if debug:
            print('[Fedpara GUI] Python:', sys.executable, file=sys.stderr)
            print('[Fedpara GUI] Starting pywebview in debug mode', file=sys.stderr)
        webview_module.start(debug=debug)
    finally:
        # Clean up only the subprocess launched by this UI, not the P2P server.
        controller.close()
    return window
