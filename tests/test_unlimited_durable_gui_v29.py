"""v29 regression: no GUI publication end cap, persistent effective checkpoints, scale."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from patchlib.elastic import AppliedJournalIndex
from patchlib.roomid import create_room_id
from patchlib.webview_gui import FedparaBridge, compose_html
from patchlib.webview_controller import WebviewController


def _new_room(controller):
    room=create_room_id()
    controller.save(dict(room=room,model='org/model',revision='main',task='causal-lm',
                         sources=[dict(repo='org/data',revision='main',config=None,
                                       split='train',columns={'text':'text'},weight=1)]))
    return room


def test_unlimited_training_always_creates_separate_durable_shard_manifest(tmp_path,monkeypatch):
    ctl=WebviewController(tmp_path)
    rid=_new_room(ctl)
    cmds=[]
    monkeypatch.setattr(ctl.train,'start',lambda cmd,log: cmds.append(cmd) or 9001)
    assert ctl.settings()['rounds']==0
    result=[]
    for _ in range(2):
        r=ctl.start(dict(room=rid,base='',ws='ws://127.0.0.1:8080/ws',
                         kind='loha',rank=16,rounds=200))
        result.append(r)
        assert cmds[-1][cmds[-1].index('--rounds')+1]=='0'
        assert cmds[-1][cmds[-1].index('--shard-state-manifest')+1]==r['checkpoint_file']
        assert cmds[-1][cmds[-1].index('--shard-store')+1]==r['model_storage_dir']
        assert int(cmds[-1][cmds[-1].index('--max-state-gb')+1])>=512
        assert Path(r['checkpoint_file']).parent.is_dir()
        assert Path(r['model_storage_dir']).is_dir()
        assert '--full-state-output' not in cmds[-1]
    assert result[0]['checkpoint_file']!=result[1]['checkpoint_file']
    assert result[0]['model_storage_dir']==result[1]['model_storage_dir']
    assert ctl.status()['checkpoint_ready'] is False
    Path(result[-1]['checkpoint_file']).write_text('{}')
    assert ctl.status()['checkpoint_ready'] is True


def test_file_manager_actions_are_whitelisted_and_expose_real_room_paths(tmp_path,monkeypatch):
    import patchlib.webview_controller as wvc
    ctl=WebviewController(tmp_path)
    bridge=FedparaBridge(ctl)
    room=_new_room(ctl)
    commands=[]
    monkeypatch.setattr(wvc.sys,'platform','darwin')
    monkeypatch.setattr(wvc.subprocess,'Popen',lambda cmd,**kw: commands.append(cmd))
    paths=bridge.storage_paths({'room':room})
    for kind in ('checkpoints','models','genesis'):
        r=bridge.open_storage({'room':room,'kind':kind})
        assert Path(r['path']).is_dir()
        assert r['path']==paths[kind]
        assert commands[-1]==['open',r['path']]
    with pytest.raises(ValueError,match='storage kind'):
        bridge.open_storage({'room':room,'kind':'/etc'})
    with pytest.raises(ValueError):
        bridge.open_storage({'room':'../wrong','kind':'models'})
    html=compose_html()
    for element in ('openCheckpoints','openModels','openGenesis'):
        assert f'id="{element}"' in html
    assert 'id="trainRounds"' not in html
    assert 'id="storageStatus"' in html


def test_applied_index_is_bounded_even_after_millions_of_signed_events():
    class Journal:
        def __init__(self): self.ids=set()
        def has(self,origin,seq):
            return ('digest',) if (origin,seq) in self.ids else None
    journal=Journal()
    cache=AppliedJournalIndex(journal,capacity=128)
    for i in range(100_000):
        ident=('alice',i)
        journal.ids.add(ident)
        cache.add(ident)
    assert len(cache.hot)==128
    assert ('alice',0) in cache  # indexed durable event, evicted from RAM
    assert ('unknown',0) not in cache
    cache.replaying=True
    assert ('alice',0) not in cache  # journal entries aren't yet applied during replay
    assert ('alice',99_999) in cache
    cache.replaying=False
