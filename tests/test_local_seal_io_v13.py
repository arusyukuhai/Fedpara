"""Regression coverage for local publish timing and opt-in WAL durability."""
import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from patchlib.elastic import EventJournal
from patchlib.continual import ContinualElasticEngine
from patchlib.core import UniversalAdapter
from patchlib.live_telemetry import LocalTelemetry, recent_samples
from patchlib.gui import DesktopController
from test_livefed import Tiny


def test_journal_full_is_default_and_normal_is_explicit(tmp_path):
    genesis = 'a'*64
    path = tmp_path/'events.sqlite3'
    with pytest.raises(ValueError, match='sync'):
        EventJournal(tmp_path/'bad.sqlite3',genesis,sync='OFF')
    full=EventJournal(path,genesis)
    assert full.sync=='FULL'
    assert full.db.execute('PRAGMA synchronous').fetchone()[0] == 2
    full.close()
    normal=EventJournal(path,genesis,sync='NORMAL')
    try:
        assert normal.db.execute('PRAGMA synchronous').fetchone()[0] == 1
        assert asyncio.run(normal.append_async('alice',1,b'payload'))
    finally:
        normal.close()
    again=EventJournal(path,genesis)
    try:
        assert again.get('alice',1) == b'payload'
    finally:
        again.close()


def test_local_seal_reports_export_validate_journal_apply(tmp_path):
    async def run():
        engine=ContinualElasticEngine(UniversalAdapter(Tiny(),'tiny','base',kind='loha',rank=3),
                                     'alice',hashlib.sha256(b'base').hexdigest(),
                                     EventJournal(tmp_path/'peer.sqlite3',hashlib.sha256(b'base').hexdigest()),
                                     unsafe=True)
        try:
            assert await engine.seal_local_responsive(1)
            ms = engine.last_local_seal_stages_ms
            assert len(ms)==4
            assert all(isinstance(s,float) and s>=0 for s in ms)
            assert engine.journal.count()==1
        finally:
            engine.journal.close()
    asyncio.run(run())


def test_local_breakdown_visible_in_own_trace(tmp_path):
    path=tmp_path/'own.jsonl'
    trace=LocalTelemetry(path,sample_seconds=0)
    trace.record(steps=4,loss=2.,emitted=1,inbound_applied=0,inbox=0,channels=1,
                 local_seal_ms=2450.,local_lock_ms=60.,local_export_ms=120.,
                 local_verify_ms=230.,local_journal_ms=1980.,local_apply_ms=60.)
    trace.close()
    item=recent_samples(path)[0]
    assert item['local_journal_ms']==1980.
    assert item['local_export_ms']==120.
    assert item['local_lock_ms']==60.


def test_gui_normal_mode_reaches_live_cli(tmp_path, monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.roomid import create_room_id
    controller=WebviewController(tmp_path)
    room=create_room_id()
    controller.save({'room':room, 'model':'org/model', 'revision':'main',
        'task':'causal-lm', 'seed':42, 'sources':[{'repo':'org/data',
        'revision':'main','config':None,'split':'train',
        'columns':{'text':'text'},'weight':1.0}]})
    captured=[]
    monkeypatch.setattr(controller.train,'start',lambda cmd,log: captured.append(cmd) or 900)
    controller.start({'room':room,'ws':'ws://127.0.0.1:8080/ws',
                      'kind':'loha','rank':16,'journal_sync':'normal'})
    cmd=captured[-1]
    assert cmd[cmd.index('--journal-sync')+1] == 'normal'
    with pytest.raises(ValueError,match='journal_sync'):
        controller.start({'room':room,'ws':'ws://127.0.0.1:8080/ws',
                          'kind':'loha','rank':16,'journal_sync':'off'})
