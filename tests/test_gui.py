from pathlib import Path
import pytest
from patchlib.gui import DesktopController
from patchlib.roomid import create_room_id


def test_native_desktop_manifest_and_command(tmp_path):
    ctl=DesktopController()
    m=ctl.create_manifest('org/model','main','causal-lm',[
        {'repo':'org/dataset','revision':'abc123','config':None,'split':'train',
         'columns':{'text':'content'},'weight':1.0}])
    fp=ctl.save_manifest(tmp_path/'room.json',m)
    base=tmp_path/'genesis.fstate';base.write_bytes(b'fake-only-command-test')
    cmd=ctl.build_command(room=create_room_id(),model='org/model',revision='main',
           base=base,manifest=fp,ws='wss://localhost/ws',task='causal-lm',
           kind='loha',rank=32,journal=tmp_path/'journal.sqlite3')
    assert '--room-manifest' in cmd
    assert '--text-file' not in cmd
    assert '--room-id' in cmd
    assert not hasattr(ctl,'server')


def test_desktop_rejects_insecure_remote_ws_and_legacy_v1(tmp_path):
    ctl=DesktopController()
    m=ctl.create_manifest('org/model','main','causal-lm',[
        {'repo':'org/dataset','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1}])
    p=ctl.save_manifest(tmp_path/'room.json',m)
    base=tmp_path/'genesis.fstate';base.touch()
    with pytest.raises(ValueError,match='wss'):
        ctl.build_command(room=create_room_id(),model='org/model',revision='main',
           base=base,manifest=p,ws='ws://example.org/ws',task='causal-lm',
           kind='loha',rank=32,journal='x.sqlite3')
