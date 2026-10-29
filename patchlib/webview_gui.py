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
        return {'ok': True, 'backend': 'Kagami Python', 'python': sys.executable}

    def defaults(self, _payload=None):
        return self._controller.settings()

    def new_room(self, _payload=None):
        from .roomid import create_room_id
        return {'room': create_room_id()}

    def select_model(self,payload):
        return self._controller.select_model(payload)

    def model_compatibility(self,payload):
        return self._controller.model_compatibility(payload)

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

    def pick_training_file(self, payload: dict):
        """Native file selector; never read a large dataset into the WebView."""
        if not isinstance(payload,dict) or payload.get('kind') not in ('sft','ppo','critic'):
            raise ValueError('Invalid training file kind')
        import webview
        if not getattr(webview,'windows',None):
            raise RuntimeError('A desktop window is required for file selection')
        window=webview.windows[0]
        if payload['kind']=='critic':
            mask=('Safetensors (*.safetensors)', 'All files (*.*)')
        else:
            mask=('JSON Lines (*.jsonl)', 'All files (*.*)')
        dialog_type=getattr(webview,'OPEN_DIALOG',None)
        if dialog_type is None:
            dialog_type=webview.FileDialog.OPEN
        result=window.create_file_dialog(dialog_type,allow_multiple=False,file_types=mask)
        return {'path':str(Path(result[0]).expanduser().resolve()) if result else ''}

    def mixed_settings(self, payload: dict):
        return self._controller.mixed_settings(payload)

    def start_training(self, payload: dict):
        return self._controller.start(payload)

    def training_status(self, _payload=None):
        return self._controller.status()

    def training_history(self, payload):
        return self._controller.training_history(payload)

    def storage_paths(self, payload: dict):
        return self._controller.storage_paths(payload)

    def open_storage(self, payload: dict):
        return self._controller.open_storage(payload)

    def copy_logs(self, _payload=None):
        return self._controller.copy_logs()

    def save_logs(self, _payload=None):
        return self._controller.save_logs()

    def start_benchmark(self, payload: dict):
        return self._controller.start_benchmark(payload)

    def benchmark_status(self, _payload=None):
        return self._controller.benchmark_status()

    def stop_training(self, _payload=None):
        return self._controller.pause_training()

    def resume_training(self, _payload=None):
        return self._controller.resume_training()

    def disconnect_training(self, _payload=None):
        return self._controller.stop()

    def local_models(self, _payload=None):
        return self._controller.local_models()

    def device_capabilities(self, _payload=None):
        return self._controller.device_capabilities()

    def inspect_dataset(self, payload):
        return self._controller.inspect_dataset(payload)

    def recommend_training(self, payload):
        return self._controller.recommend_training(payload)

    def pin_mixed_dataset(self, payload):
        return self._controller.pin_mixed_dataset(payload)

    def discover_hub_assets(self, payload):
        return self._controller.discover_hub_assets(payload)

    def search_reward_models(self, payload):
        return self._controller.search_reward_models(payload)

    def inspect_reward_model(self, payload):
        return self._controller.inspect_reward_model(payload)

    def search_datasets(self, payload):
        return self._controller.search_datasets(payload)

    def evaluate(self, payload):
        return self._controller.evaluate(payload)

    def available_checkpoints(self, payload):
        return self._controller.available_checkpoints(payload)

    def request_eval_snapshot(self, payload):
        return self._controller.request_eval_snapshot(payload)

    def eval_snapshot_status(self, payload):
        return self._controller.eval_snapshot_status(payload)

    def eval_status(self, _payload=None):
        return self._controller.eval_status()

    def cancel_eval(self, _payload=None):
        return self._controller.cancel_eval()



def compose_html() -> str:
    """Use exclusively bundled GUI CSS/JS; no third-party privileged scripts."""
    template = (WEB / 'desktop_webview.html').read_text(encoding='utf-8')
    css = (WEB / 'tailwind.cdn.css').read_text(encoding='utf-8')
    js = (WEB / 'desktop_webview.js').read_text(encoding='utf-8')
    # Trusted UI assets only; no user-supplied markup is ever interpolated.
    if '</style' in css.lower() or '</script' in js.lower():
        raise ValueError('unexpected closing HTML tag in a bundled resource')
    # pywebview dynamically builds its JS API with `new Function(...)`; it needs
    # 'unsafe-eval'. Without it the HTML loads, but Python methods never appear.
    # No untrusted HTML is interpolated; the page embeds only vetted local assets.
    # JS calls to Python do not require a socket/HTTP connection.
    meta = '''<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline' https://fonts.googleapis.com https://fonts.cdnfonts.com; font-src https://fonts.gstatic.com https://fonts.cdnfonts.com; img-src data:; connect-src 'none'; worker-src 'none'; form-action 'none'; base-uri 'none'; object-src 'none'; frame-src 'none'">'''
    template = template.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n'+meta, 1)
    template = template.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)
    template = template.replace('</body>', f'<script>\n{js}\n</script>\n</body>', 1)
    # External font stylesheets must NEVER block a local-first window opening:
    # synchronous CSS @import would wait for CDN DNS/timeout before DOM ready.
    # Inject ONLY these two hard-coded styles after the document loads. The
    # restrictive CSP still disallows any remote JavaScript or data upload.
    font_loader = """<script>
window.addEventListener('load', function () {
  const jp = document.createElement('style');
  jp.textContent = "@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;600;700&display=swap')";
  document.head.appendChild(jp);
  const latin = document.createElement('style');
  latin.textContent = "@import url('https://fonts.cdnfonts.com/css/myriad-pro')";
  document.head.appendChild(latin);
});
</script>"""
    return template.replace('</body>', font_loader+'\n</body>', 1)


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
        'Kagami — Distributed Learning',
        html=compose_html(),
        js_api=bridge,
        width=1380,
        height=860,
        min_size=(960, 620),
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
