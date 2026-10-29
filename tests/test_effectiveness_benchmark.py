import csv
import json
import math
import time
from pathlib import Path

import pytest

from patchlib.effectiveness_benchmark import run_effectiveness_benchmark
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge, compose_html


def test_real_loha_causal_lm_equal_data_budget_and_files(tmp_path):
    data=run_effectiveness_benchmark(peers=2,rounds=2,local_steps=1,
                                     seed=103,output_dir=tmp_path)
    assert data['status']=='OFFLINE_LOHA_EFFECTIVENESS'
    assert data['configuration']['kind']=='loha'
    assert len(data['trace'])==2
    assert data['trace'][-1]['tokens_seen']==2*2*1*4*12
    assert abs(data['compute']['p2p_serial_s']/2 -
               data['compute']['p2p_divided_by_peers_s'])<1e-12
    assert data['compute']['p2p_max_peer_critical_path_s'] >= data['compute']['p2p_divided_by_peers_s']
    assert data['fairness']['identical_training_samples']
    for key in ('initial','central','p2p'):
        assert math.isfinite(data['loss'][key])
    assert (tmp_path/'effectiveness-result.json').exists()
    assert (tmp_path/'effectiveness-trace.csv').exists()
    assert data['fairness']['peer_factors_seeded_identically_at_initialization']
    if data['chart_created']:
        assert (tmp_path/'effectiveness-loss.png').exists()
        assert (tmp_path/'effectiveness-time.png').exists()
    assert json.loads((tmp_path/'effectiveness-result.json').read_text())['configuration']['seed']==103
    with (tmp_path/'effectiveness-trace.csv').open() as fh:
        assert len(list(csv.DictReader(fh)))==2


def test_invalid_benchmark_dimensions_and_gui_input(tmp_path):
    with pytest.raises(ValueError):
        run_effectiveness_benchmark(peers=1,rounds=1)
    with pytest.raises(ValueError):
        run_effectiveness_benchmark(peers=2,rounds=1,rank=30,width=24)
    ctl=WebviewController(tmp_path)
    with pytest.raises(ValueError):
        ctl.start_benchmark({'peers':'four'})
    with pytest.raises(ValueError):
        ctl.start_benchmark({'peers':4,'rank':100})
    assert not ctl.benchmark_status()['running']
    ctl.close()


def test_benchmark_gui_bridge_is_functional(tmp_path):
    html=compose_html()
    # Kagami's six fixed tabs intentionally omit the old benchmark controls.
    # Benchmark RPC remains supported for compatibility and scripting.
    assert 'id="panel-training"' in html
    assert 'id="panel-new"' in html
    ctl=WebviewController(tmp_path)
    api=FedparaBridge(ctl)
    # Keep the CI GUI-process smoke test offline. User GUI defaults to TinyStories-1M.
    response=api.start_benchmark({'model':'toy','dataset':'auto','peers':2,'rounds':1,'local_steps':1,'kind':'loha','rank':4})
    assert response['pid']>0
    assert Path(response['output_dir']).is_dir()
    try:
        for _ in range(150):
            status=api.benchmark_status()
            if not status['running']:
                break
            time.sleep(.1)
        assert status['exit_code']==0, status['logs']
        assert status['result']['configuration']['peers']==2
        assert 'effectiveness-result.json' in status['result']['artifacts']
    finally:
        ctl.close()


def test_benchmark_seed_reproducible():
    options={'peers':2,'rounds':2,'local_steps':1,'seed':419}
    first=run_effectiveness_benchmark(**options)
    second=run_effectiveness_benchmark(**options)
    assert first['loss']==second['loss']
    assert first['trace'][0]['tokens_seen']==2*1*4*12


def test_local_byte_corpus_runs(tmp_path):
    text=tmp_path/'corpus.txt'
    text.write_text(('Once upon a time: a curious fox jumped.\n' * 100),encoding='utf-8')
    result=run_effectiveness_benchmark(peers=2,rounds=1,local_steps=1,
        vocab=256,rank=4,text_file=text,output_dir=tmp_path/'run')
    assert result['configuration']['dataset']['type']=='local-utf8-bytes'
    assert math.isfinite(result['loss']['p2p'])
    assert (tmp_path/'run'/'effectiveness-trace.csv').exists()
