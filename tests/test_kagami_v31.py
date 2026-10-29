"""Kagami v31: new UI RPC contract, relay-only state, inference and privacy."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.webview_gui import compose_html, FedparaBridge
from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id
from patchlib.live_telemetry import LocalTelemetry, recent_samples


def _save(ctl):
    rid=create_room_id()
    ctl.save({'room':rid, 'model':'org/model', 'revision':'main', 'task':'causal-lm',
       'seed':42,'sources':[{'repo':'org/data','revision':'main','config':None,
           'split':'train','columns':{'text':'text'},'weight':1.0}]})
    return rid


def test_kagami_screen_and_localization_contract():
    h=compose_html()
    for tab in ('models','training','datasets','eval','network','new'):
        assert f'id="panel-{tab}"' in h
        assert f'data-tab="{tab}"' in h
    for control in ('start','stop','disconnect','dockToggle','modelCards','networkGraph',
                    'codeEditor','evalPrompt','evalPrecision','searchDatasets','language','openGenesis'):
        assert f'id="{control}"' in h
    assert 'KAGAMI' in h and '日本語' in h and 'English' in h
    assert '"Noto Sans CJK JP"' in h
    assert 'overflow:hidden' in h and 'height:100dvh' in h
    assert 'document.querySelectorAll' in h
    assert 'trust_remote_code=True' not in h
    assert '<style>' in h # usable when Tailwind CDN is unavailable


def test_relay_toggle_creates_file_and_keeps_training_worker(tmp_path,monkeypatch):
    ctl=WebviewController(tmp_path)
    rid=_save(ctl)
    args=[]
    monkeypatch.setattr(ctl.train,'start',lambda cmd,log: args.append(cmd) or 123)
    monkeypatch.setattr(ctl.train,'status',lambda: bool(args))
    ctl.start({'room':rid,'base':'','ws':'ws://127.0.0.1:8080/ws','kind':'loha','rank':16})
    assert '--swarm-pause-file' in args[-1]
    sentinel=Path(args[-1][args[-1].index('--swarm-pause-file')+1])
    assert not sentinel.exists()
    bridge=FedparaBridge(ctl)
    stopped=bridge.stop_training()
    assert stopped['relay_only'] is True
    assert sentinel.exists()
    assert bridge.training_status()['relay_only'] is True
    resumed=bridge.resume_training()
    assert resumed['relay_only'] is False
    assert not sentinel.exists()
    ctl.profiles[rid].close()


def test_relay_noop_without_running_worker(tmp_path):
    ctl=WebviewController(tmp_path);bridge=FedparaBridge(ctl)
    assert not bridge.stop_training()['relay_only']
    assert not bridge.resume_training()['running']


def test_local_model_catalog_and_selection(tmp_path):
    ctl=WebviewController(tmp_path);rid=_save(ctl);bridge=FedparaBridge(ctl)
    result=bridge.local_models()
    assert len(result['models'])==1
    assert result['models'][0]['datasets']==['org/data']
    info=bridge.select_model({'room':rid})
    assert info['local'] and info['manifest']['model']['id']=='org/model'
    assert info['sha256']


def test_dataset_search_bounded(monkeypatch,tmp_path):
    import huggingface_hub
    class MockHub:
        def list_datasets(self,**opts):
            assert opts['limit']==25 and opts['search']=='story'
            return [SimpleNamespace(id='roneneldan/TinyStories',downloads=1234,sha='abc',gated=False)]
    monkeypatch.setattr(huggingface_hub,'HfApi', MockHub)
    ctl=WebviewController(tmp_path)
    found=ctl.search_datasets({'query':'story'})
    assert found['datasets'][0]['revision']=='abc'
    with pytest.raises(ValueError):ctl.search_datasets({'query':'a'*101})


def test_inference_validates_model_and_starts_isolated_process(tmp_path,monkeypatch):
    import patchlib.webview_controller as wvc
    ctl=WebviewController(tmp_path);rid=_save(ctl)
    calls=[]
    class FakeProcess:
        pid=456
        def poll(self):return None
        def terminate(self):pass
    def fake_popen(cmd, **kwargs):
        calls.append((cmd,kwargs));return FakeProcess()
    monkeypatch.setattr(wvc.subprocess,'Popen',fake_popen)
    result=ctl.evaluate({'room':rid,'prompt':'Once upon a time','precision':'nf4','max_tokens':32,'scope':'base'})
    assert result['scope'].startswith('Explicit HF base')
    assert calls[0][0][1:3]==['-m','patchlib.kagami_eval_worker']
    options=json.loads(Path(calls[0][0][3]).read_text())
    assert options['precision']=='nf4' and options['model']=='org/model' and options['scope']=='base'
    with pytest.raises(RuntimeError,match='already running'):
        ctl.evaluate({'room':rid,'prompt':'Again','precision':'fp16','scope':'base'})
    with pytest.raises(ValueError):ctl.evaluate({'room':rid,'prompt':'bad','precision':'made-up'})


def test_real_neighbor_telemetry_no_invented_global_edges(tmp_path):
    path=tmp_path/'telemetry.jsonl'
    record=LocalTelemetry(path)
    record.record(steps=11,loss=1.2,emitted=2,inbound_applied=1,inbox=0,
           channels=2,accuracy=.47,peer_id='local-abc',connected_peers=['peer-x','peer-y'],force=True)
    record.close()
    rows=recent_samples(path)
    assert len(rows)==1
    assert rows[0]['accuracy']==.47
    assert rows[0]['connected_peers']==['peer-x','peer-y']
    assert rows[0]['peer_id']=='local-abc'


def test_worker_quant_options_exist():
    worker=Path(__file__).resolve().parent.parent/'patchlib'/'kagami_eval_worker.py'
    src=worker.read_text()
    for name in ('fp4','nf4','int8','torchao-int4','fp16','bf16'):
        assert name in src
    assert 'trust_remote_code' in src and 'False' in src
