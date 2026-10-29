"""Kagami v34: pinned rank, private long history, square UI, live metrics."""
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from patchlib.auto_policy import estimate_adapter_rank
from patchlib.live_telemetry import history_samples
from patchlib.room_manifest import validate_manifest, make_hf_manifest
from patchlib.webview_gui import compose_html


def test_rank_estimates_use_memory_and_network_ceilings():
    cfg={'hidden_size':4096,'num_hidden_layers':32}
    low=estimate_adapter_rank(config=cfg,available_vram_bytes=1*1024**3)
    high=estimate_adapter_rank(config=cfg,available_vram_bytes=80*1024**3)
    assert low['rank']<=high['rank']
    assert high['rank'] in (4,8,16,32,64)
    assert low['rank']<=low['rank_capacity'] or low['rank_capacity']==0
    assert estimate_adapter_rank(config=None,available_ram_bytes=8*1024**3)['rank']==16
    with pytest.raises(ValueError):estimate_adapter_rank(config={},kind='unsafe')


def test_pinned_adapter_is_checked_by_manifest():
    sources=[{'repo':'roneneldan/TinyStories','revision':'main','config':'default','split':'train','weight':1.0,'columns':{'text':'text'}}]
    m=make_hf_manifest('roneneldan/TinyStories-1M','main',sources)
    m['adapter']={'kind':'loha','rank':32,'wide_lora_rank':4}
    assert validate_manifest(m)['adapter']['rank']==32
    m['adapter']['rank']=31
    with pytest.raises(Exception):validate_manifest(m)


def test_long_history_range_and_partial_tail(tmp_path):
    p=tmp_path/'metrics.jsonl'
    with p.open('w',encoding='utf8') as f:
        for i in range(14000):
            f.write(json.dumps({'steps':i,'elapsed_s':i*2.0,'loss':6.0-i/3000,
                                'accuracy':i/14000,'state':'training'})+'\n')
        f.write('{"uncommitted":true')
    short=history_samples(p,window_seconds=300,max_points=125)
    long=history_samples(p,window_seconds=21600,max_points=125)
    all_rows=history_samples(p,window_seconds=0,max_points=125)
    assert short[-1]['steps']==long[-1]['steps']==all_rows[-1]['steps']==13999
    assert short[0]['elapsed_s']>=long[0]['elapsed_s']>=all_rows[0]['elapsed_s']
    assert all(len(rows)<=125 for rows in [short,long,all_rows])
    assert all(row['elapsed_s']>=27998-300 for row in short)
    with pytest.raises(ValueError):history_samples(p,window_seconds=-123)
    with pytest.raises(ValueError):history_samples(p,max_points=10000)


def test_ui_has_unique_metric_ids_square_edges_and_range_control():
    html=compose_html()
    assert html.count('id="modelCurrentLoss"')==1
    assert html.count('id="modelCurrentAccuracy"')==1
    assert html.count('id="chartWindow"')==1
    assert '604800' in html and 'Entire run' in html
    assert 'border-radius:0!important' in html
    assert '--jp:"Noto Sans JP"' in html
    assert 'trainAutoRank' in html and 'id="rank"' not in html
    assert "training_history" in html
    assert "historyWindow:'表示期間'" in html
    assert 'chartAxisNumber(val,field)' in html
