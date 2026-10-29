"""v39: unbounded signed archive and safe recovery after GUI/application restart."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from patchlib.elastic import EventJournal
from patchlib.profiles import allocate_profile
from patchlib.roomid import create_room_id
from patchlib.storage_location import choose_data_root, pin_data_root
from patchlib.webview_controller import WebviewController
from patchlib.restart_policy import policy_path

GENESIS=hashlib.sha256(b'fixed-genesis').hexdigest()


def payload(room):
    return {'room':room,'model':'org/model','revision':'main','task':'causal-lm',
            'sources':[{'repo':'org/dataset','revision':'main','config':None,
                        'split':'train','columns':{'text':'text'},'weight':1.0}]}


def test_no_fixed_archive_cap_and_no_pruning_when_reopened(tmp_path):
    path=tmp_path/'events.sqlite3'
    journal=EventJournal(path,GENESIS,max_bytes=0)
    try:
        for index in range(1,17):
            assert journal.append('alice',index,b'x'*3072+str(index).encode())
        assert journal.total_bytes()>40_000
        assert journal.count()==16
        assert journal.next_seq('alice')==17
        assert journal.get('alice',1)==b'x'*3072+b'1'
    finally: journal.close()
    recovered=EventJournal(path,GENESIS,max_bytes=0)
    try:
        assert recovered.count()==16
        assert recovered.get('alice',16)==b'x'*3072+b'16'
        assert recovered.append('alice',17,b'continued')
        assert recovered.next_seq('alice')==18
    finally: recovered.close()


def test_explicit_hard_cap_still_prevents_silent_history_loss(tmp_path):
    journal=EventJournal(tmp_path/'capped.sqlite3',GENESIS,max_bytes=1024)
    try:
        assert journal.append('alice',1,b'a'*1024)
        with pytest.raises(Exception,match='log is full'):
            journal.append('alice',2,b'b')
        assert journal.count()==1
    finally: journal.close()


def test_reopen_dormant_peer_with_most_history(tmp_path):
    root=tmp_path/'room'
    for name,length in [('peer-older',32000),('peer-empty',0)]:
        directory=root/'peers'/name
        directory.mkdir(parents=True)
        (directory/'events.sqlite3').touch()
        if length:
            (directory/'events.sqlite3.fplog').write_bytes(b'0'*length)
    old=allocate_profile(root)
    try:
        assert old.name=='peer-older'
    finally:old.close()
    new=allocate_profile(root)
    try:
        assert new.name==old.name
        assert new.journal==old.journal
    finally:new.close()


def test_resume_pins_launch_and_retains_room_after_gui_restart(tmp_path,monkeypatch):
    room=create_room_id()
    first=WebviewController(tmp_path)
    first.save(payload(room))
    commands=[]
    monkeypatch.setattr(first.train,'start',lambda cmd,log: commands.append(cmd) or 101)
    previous=first.start({'room':room,'ws':'ws://127.0.0.1:8080/ws',
                          'base':'','streaming_base':True,'optimizer':'adamw',
                          'reinit_every':13})
    assert '--streaming-base' in commands[-1]
    assert policy_path(Path(previous['journal'])).is_file()
    assert first.settings()['last_room']==room
    first.close()
    second=WebviewController(tmp_path)
    monkeypatch.setattr(second.train,'start',lambda cmd,log: commands.append(cmd) or 102)
    try:
        assert second.settings()['last_room']==room
        after=second.start({'room':room,'ws':'ws://127.0.0.1:8080/ws',
                            'base':'','streaming_base':False,'optimizer':'lora-muon',
                            'reinit_every':20})
        cmd=commands[-1]
        assert after['journal']==previous['journal']
        assert '--streaming-base' in cmd
        assert cmd[cmd.index('--optimizer')+1]=='adamw'
        assert cmd[cmd.index('--reinit-every')+1]=='13'
        assert cmd[cmd.index('--max-journal-gb')+1]=='0'
    finally:second.close()


def test_data_root_discovery_and_pinning_survives_extract_upgrade(tmp_path):
    old=tmp_path/'kagami-v38'
    new=tmp_path/'kagami-v39'
    old.joinpath('.fedpara-gui','rooms').mkdir(parents=True)
    new.mkdir()
    h=tmp_path/'home'; h.mkdir()
    assert choose_data_root(cwd=new,home=h,environment={})==old.resolve()
    pin_data_root(old,home=h)
    moved=tmp_path/'other';moved.mkdir()
    assert choose_data_root(cwd=moved,home=h,environment={})==old.resolve()
    assert choose_data_root(cwd=moved,home=h,environment={'KAGAMI_DATA_DIR':str(new)})==new.resolve()
