"""Regression tests for saved-weight availability and complete training history."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from patchlib.live_telemetry import LocalTelemetry, history_samples, recent_samples
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html
from patchlib.shards import export_state_sharded
from patchlib.roomid import create_room_id


def model_params(room):
    return {'room':room,'model':'example/tiny','revision':'main',
            'task':'causal-lm','seed':42,
            'sources':[{'repo':'example/data','revision':'main','config':None,
                        'split':'train','columns':{'text':'text'},'weight':1.0}]}


def test_graph_history_preserved_across_restarts_and_long_ranges(tmp_path):
    path=tmp_path/'history-room-peer.jsonl'
    for run in range(4):
        writer=LocalTelemetry(path,sample_seconds=0,append=True)
        for step in range(450):
            writer.record(steps=step,loss=4-(run*450+step)*0.001,
                          accuracy=step/500,emitted=step//4,
                          inbound_applied=0,inbox=0,channels=1,force=True)
        writer.close()
    head=history_samples(path,window_seconds=0,max_points=1600)
    assert head[-1]['steps']==1799
    assert head[0]['steps']==0
    assert len(head)>1000
    assert all(x['steps']<=y['steps'] for x,y in zip(head,head[1:]))
    assert len(recent_samples(path)) <= 320
    # A short window is allowed to contain fewer points but not to rewrite history.
    assert history_samples(path,window_seconds=300,max_points=800)[-1]['steps']==1799


def test_model_select_recovers_old_history_without_running_worker(tmp_path):
    controller=WebviewController(tmp_path)
    room=create_room_id()
    controller.save(model_params(room))
    history=tmp_path/'.fedpara-gui'/'telemetry'/f'history-{room}-peer-abc.jsonl'
    history.parent.mkdir(parents=True,exist_ok=True)
    writer=LocalTelemetry(history,sample_seconds=0,append=True)
    writer.record(steps=700,loss=2,emitted=4,inbound_applied=2,inbox=0,channels=0)
    writer.close()
    controller.select_model({'room':room})
    results=controller.training_history({'window_seconds':0,'max_points':100})
    assert results['model_id']==room
    assert results['telemetry'][-1]['steps']==700


def test_pause_enqueues_model_snapshot_and_tracks_failure(tmp_path):
    ctl=WebviewController(tmp_path);room=create_room_id()
    ctl.save(model_params(room))
    directory=tmp_path/'.fedpara-gui'/'rooms'/room
    ctl._current_room=room
    ctl._pause_file=directory/'pause.flag'
    ctl._snapshot_request_file=directory/'eval.request'
    ctl._checkpoint_path=directory/'effective.shards.json'
    ctl.train.status=lambda:True
    first=ctl.pause_training()
    assert first['relay_only'] and first['save_requested']
    assert ctl._pause_file.is_file() and ctl._snapshot_request_file.is_file()
    assert ctl.eval_snapshot_status({'room':room})['state']=='queued'
    progress=ctl._snapshot_request_file.with_suffix('.status.json')
    ctl._snapshot_request_file.unlink()
    progress.write_text(json.dumps({'state':'error','error':'disk full'}))
    assert ctl.eval_snapshot_status({'room':room})['error']=='disk full'
    ctl.request_eval_snapshot({'room':room})
    assert ctl._snapshot_request_file.is_file()
    assert not progress.exists()


def test_completed_checkpoint_visible_to_eval(tmp_path):
    ctl=WebviewController(tmp_path);room=create_room_id()
    ctl.save(model_params(room))
    profile='peer-test'
    root=ctl._room_path(room).parent
    store=root/'models'/profile
    dest=root/'checkpoints'/profile/'effective-000.shards.json'
    dest.parent.mkdir(parents=True)
    export_state_sharded(nn.Linear(8,8),store,dest,model_id='example/tiny',
                         revision='main',peer_id='peer-test',seq=7,shard_bytes=65536)
    listing=ctl.available_checkpoints({'room':room})['checkpoints']
    assert len(listing)==1 and listing[0]['seq']==7
    assert listing[0]['store']==str(store)


def test_font_and_ui_are_csp_permitted_and_no_courier():
    html=compose_html()
    assert "@import url('https://fonts.cdnfonts.com/css/myriad-pro')" in html
    assert "family=Noto+Sans+JP" in html
    assert 'font-src https://fonts.gstatic.com https://fonts.cdnfonts.com' in html
    assert 'script-src \'unsafe-inline\' \'unsafe-eval\'' in html
    assert 'html,body,button,input,select,textarea,pre,code,svg text{font-family:var(--jp)!important}' in html
    assert 'SAVE FAILED:' in html


def test_one_click_trained_inference_queues_save_when_not_yet_present(tmp_path):
    ctl=WebviewController(tmp_path);room=create_room_id()
    ctl.save(model_params(room))
    root=ctl._room_path(room).parent
    ctl._current_room=room
    ctl._snapshot_request_file=root/'eval-request.flag'
    ctl._checkpoint_path=root/'checkpoints'/'peer-1'/'effective.shards.json'
    ctl.train.status=lambda:True
    result=ctl.evaluate({'room':room,'scope':'trained','prompt':'Hi'})
    assert result['pending_snapshot']
    assert ctl._snapshot_request_file.is_file()
    assert ctl.eval_snapshot_status({'room':room})['state']=='queued'
