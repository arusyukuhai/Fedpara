"""Regression tests for fixed-pane scrolling, auto planning and dataset composition."""
import sys
import types
from pathlib import Path

import pytest

from patchlib.auto_policy import estimate_training_policy
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge, compose_html


def test_context_respects_model_and_device_and_publication_size():
    config={'n_positions':1024,'n_embd':1024,'n_layer':12}
    low=estimate_training_policy(config=config, available_ram_bytes=1*1024**3,rank=8,kind='lora')
    high=estimate_training_policy(config=config,available_vram_bytes=96*1024**3,
                                  available_ram_bytes=96*1024**3,rank=64,kind='loha')
    assert 32<=low['max_length']<high['max_length']<=1024
    assert high['local_steps'] > low['local_steps']
    missing=estimate_training_policy(config=None,available_vram_bytes=512*1024**3)
    assert missing['max_length']<=1024
    with pytest.raises(ValueError):estimate_training_policy(config=None,rank=-1)


def test_auto_plan_peers_not_user_supplied(tmp_path,monkeypatch):
    ctrl=WebviewController(tmp_path)
    monkeypatch.setattr(ctrl,'device_capabilities',lambda *a:{'ram_bytes':8*1024**3,'vram_bytes':0})
    # No remote Python / model code. Only cached standard HF Config metadata.
    module=types.ModuleType('transformers')
    module.AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**kw:
        types.SimpleNamespace(to_dict=lambda:{'n_positions':4096,'n_embd':768,'n_layer':12}))
    monkeypatch.setitem(sys.modules,'transformers',module)
    info=ctrl.recommend_training({'model':'org/model','revision':'main','rank':16,'kind':'loha','peers':9999})
    assert info['max_length']>=32 and info['max_length']<=4096
    assert info['online_peers'] is None
    assert 'local_steps' in info
    cmd=Path(__import__('patchlib.gui',fromlist=['__file__']).__file__).read_text()
    assert "'--expected-peers',str(peers)" not in cmd


def test_hf_inspection_selects_real_metadata_not_guessed_split(monkeypatch,tmp_path):
    ctrl=WebviewController(tmp_path)
    class Client:
        def list_dataset_parquet_files(self,repo,revision):
            assert repo=='owner/corpus' and revision=='abcdef12345'
            return [types.SimpleNamespace(config='standard',split='validation'),
                    types.SimpleNamespace(config='extra',split='train'),
                    types.SimpleNamespace(config='extra',split='train')]
    module=types.ModuleType('huggingface_hub')
    module.HfApi=Client
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    result=FedparaBridge(ctrl).inspect_dataset({'repo':'owner/corpus','revision':'abcdef12345'})
    assert result['variants']==[{'config':'standard','split':'validation'},
                                {'config':'extra','split':'train'}]
    with pytest.raises(ValueError):ctrl.inspect_dataset({'repo':'https://example.com','revision':'main'})


def test_inner_scroll_works_and_code_editor_is_never_disabled():
    page=compose_html()
    assert '.draft-list{overflow-y:auto' in page
    assert '.code-pane textarea#codeEditor' in page
    assert "editor.disabled=false" in page
    assert "trainAutoContext" in page and "trainAutoPublish" in page
    assert 'id="trainPeers"' not in page
    assert 'id="trainLocalSteps"' not in page
    assert 'id="trainMaxLength"' not in page
    assert 'inspect_dataset' in page
