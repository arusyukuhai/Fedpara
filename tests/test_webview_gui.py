"""No actual display required; the real macOS/Windows WebView is not installed in CI."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge, compose_html, run_webview


def test_serverless_inline_webview_page():
    html = compose_html()
    assert '<style>' in html
    assert '<script>' in html
    assert 'window.pywebview' in html
    assert 'pywebviewready' in html
    assert "connect-src 'none'" in html
    # pywebview generates the JS API dynamically with new Function().
    assert "script-src 'unsafe-inline' 'unsafe-eval'" in html
    assert "id=\"retryBridge\"" in html
    assert '<link rel="stylesheet"' not in html
    assert '<script src=' not in html
    assert 'fetch(path' not in html


def test_bridge_roundtrip_room_dataset_and_code(tmp_path):
    controller = WebviewController(tmp_path)
    bridge = FedparaBridge(controller)
    rid = bridge.new_room()['room']
    defaults = bridge.defaults()
    assert defaults['sources']
    source = [{'repo': 'org/data', 'revision': 'main', 'config': None,
               'split': 'train', 'columns': {'text': 'text'}, 'weight': 1.0}]
    blob = b'def build_model():\n    return None\n'
    registered = bridge.register_code({'name': 'model.py', 'contents':
                                       base64.b64encode(blob).decode('ascii')})
    assert Path(registered['path']).is_file()
    saved = bridge.save_room({'room': rid, 'model': 'org/model', 'revision': 'main',
        'task': 'causal-lm', 'seed': 42, 'sources': source,
        'code_path': registered['path'], 'code_url': ''})
    assert saved['manifest']['datasets']['sources'] == source
    assert saved['manifest']['code']['sha256'] == registered['sha256']
    assert saved['sha256']
    assert not bridge.training_status()['running']
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        bridge.start_training({'room': rid, 'base': str(tmp_path / 'missing.fstate'),
                               'ws': 'ws://127.0.0.1:8080/ws', 'kind': 'loha', 'rank': 32})
    assert bridge.stop_training()['stopping']


def test_webview_launch_does_not_bind_tcp_port(tmp_path, monkeypatch):
    class FakeWebview:
        def __init__(self):
            self.info = {}
        def create_window(self, title, **kw):
            self.info = {'title': title, **kw}
            return 'window'
        def start(self, **kw):
            assert kw['debug'] is False
            assert self.info.get('url') is None
            assert 'html' in self.info
            assert isinstance(self.info['js_api'], FedparaBridge)
    view = FakeWebview()
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('desktop UI must not bind any TCP socket')
    monkeypatch.setattr(socket.socket, 'bind', forbidden)
    assert run_webview(workdir=tmp_path, webview_module=view) == 'window'


def test_join_reads_room_v2_without_code_execution(tmp_path, monkeypatch):
    import patchlib.webview_controller as vc
    controller = vc.WebviewController(tmp_path)
    bridge = FedparaBridge(controller)
    manifest = vc.make_hf_manifest('org/model', 'abc123', [{
        'repo':'org/dataset','revision':'abc123','config':None,'split':'train',
        'columns':{'text':'text'},'weight':1.0
    }])
    async def mocked_fetch(ws, room, token, timeout):
        assert ws == 'wss://example.com/ws'
        assert timeout == 8
        return manifest
    monkeypatch.setattr(vc, 'fetch_room_manifest', mocked_fetch)
    room=bridge.new_room()['room']
    loaded=bridge.join_room({'room':room,'ws':'wss://example.com/ws'})
    assert loaded['manifest']==manifest
    assert loaded['sha256']==vc.manifest_digest(manifest)
    assert 'Verify' in loaded['warning']


def test_cli_gui_uses_webview(monkeypatch):
    import sys
    import patchlib.cli as cli
    import patchlib.webview_gui as wg
    called=[]
    monkeypatch.setattr(wg, 'run_webview', lambda: called.append(True))
    monkeypatch.setattr(sys, 'argv', ['fedpara','gui'])
    cli.main()
    assert called == [True]


def test_webview_auto_checkpoint_does_not_use_workdir(tmp_path, monkeypatch):
    from patchlib.roomid import create_room_id
    controller = WebviewController(tmp_path)
    room = create_room_id()
    controller.save({'room':room, 'model':'org/model', 'revision':'main',
        'task':'causal-lm', 'seed':42, 'sources':[{'repo':'org/data',
        'revision':'main','config':None,'split':'train',
        'columns':{'text':'text'}, 'weight':1.0}]})
    cmds=[]
    monkeypatch.setattr(controller.train,'start',lambda cmd,log: cmds.append(cmd) or 1234)
    base_args={'room':room, 'ws':'ws://127.0.0.1:8080/ws',
               'kind':'loha','rank':32}
    for value in ('', '  ', None):
        result=controller.start({**base_args, 'base':value})
        assert result['auto_state']
        assert Path(result['base']).name=='genesis.fstate'
        assert Path(result['base']).parent.name==room
        assert '--auto-state' in cmds[-1]
        assert Path(result['base']) != tmp_path
    with pytest.raises(IsADirectoryError, match='フォルダ'):
        controller.start({**base_args, 'base':str(tmp_path)})
    with pytest.raises(FileNotFoundError, match='fstate'):
        controller.start({**base_args, 'base':str(tmp_path/'missing.fstate')})

def test_webview_rejects_non_fstate_file(tmp_path):
    from patchlib.roomid import create_room_id
    controller = WebviewController(tmp_path)
    room=create_room_id()
    controller.save({'room':room, 'model':'org/model', 'revision':'main',
        'task':'causal-lm', 'seed':42, 'sources':[{'repo':'org/data',
        'revision':'main','config':None,'split':'train',
        'columns':{'text':'text'}, 'weight':1.0}]})
    file=tmp_path/'notes.txt'; file.write_text('not a checkpoint')
    with pytest.raises(ValueError, match='fstate'):
        controller.start({'room':room, 'base':str(file),
                          'ws':'ws://127.0.0.1:8080/ws',
                          'kind':'loha','rank':32})


def test_noir_emerald_design_system_is_inline_and_no_external_fonts():
    # A single invariant protects the requested black/white retro-modern
    # appearance against a return to the old blue/dark dashboard theme.
    html = compose_html()
    assert 'Chicago' in html and 'Courier New' in html
    assert '--paper: #080e0b' in html
    assert '--ink: #e5f6ec' in html
    assert 'color-scheme: dark' in html
    assert '#10b981' in html
    assert 'caret-color: #34d399' in html
    assert '.room-card' in html and '.directory-toolbar' in html
    assert 'prefers-reduced-motion' in html
    assert 'https://fonts.' not in html
    assert "background_color='#080e0b'" in Path(__import__('patchlib.webview_gui',fromlist=['__file__']).__file__).read_text()
    assert 'status-error' in html and 'status-ok' in html
    assert '#527cf0' not in html
