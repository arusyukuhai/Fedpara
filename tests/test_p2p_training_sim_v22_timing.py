"""Regression: evaluate each round; plot P2P compute/peers, never serial total."""
import csv
import json
import math

import pytest

from patchlib.training_sim import run_training_sim


def _config(tmp_path, n=4):
    return dict(model='toy', peers=n, rounds=4, local_steps=2,
                reinit_every=2, batch_size=2, context=8, width=24,
                sequences=32, validation_sequences=12, threads=1,
                loss_probability=0, churn_fraction=0,
                output_dir=tmp_path)


def test_every_round_by_default_and_divided_x_axis(tmp_path):
    result = run_training_sim(**_config(tmp_path))
    trace = result['trace']
    assert [r['round'] for r in trace] == [0, 1, 2, 3, 4]
    assert trace[0]['central_compute_s'] == 0
    assert trace[0]['p2p_compute_div_peers_s'] == 0
    for row in trace:
        assert row['p2p_compute_div_peers_s'] == pytest.approx(
            row['p2p_compute_sum_s'] / 4, abs=1e-6)
        assert row['p2p_compute_max_peer_s'] >= 0
        assert math.isfinite(row['central_valid_loss'])
        assert math.isfinite(row['p2p_mean_valid_loss'])
    assert all(trace[j]['central_compute_s'] > trace[j-1]['central_compute_s']
               for j in range(1, len(trace)))
    assert all(trace[j]['p2p_compute_div_peers_s'] > trace[j-1]['p2p_compute_div_peers_s']
               for j in range(1, len(trace)))
    assert result['result']['peer_training_ideal_parallel_wall_s'] == pytest.approx(
        result['result']['peer_training_serial_wall_s']/4, abs=1e-4)
    assert (tmp_path/'p2p-training-loss.png').is_file()
    assert (tmp_path/'p2p-training-loss-tokens.png').is_file()
    with (tmp_path/'p2p-training-trace.csv').open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 5
    assert 'p2p_compute_div_peers_s' in records[0]
    assert 'central_compute_s' in records[0]


def test_16_peers_ideal_time_is_sum_divided_by_16(tmp_path):
    result = run_training_sim(**_config(tmp_path, n=16))
    last = result['trace'][-1]
    assert last['p2p_compute_div_peers_s'] == pytest.approx(
        last['p2p_compute_sum_s']/16, abs=1e-6)
    assert [row['round'] for row in result['trace']] == [0,1,2,3,4]


def test_opt_in_sparse_validation_remains_available(tmp_path):
    result = run_training_sim(**_config(tmp_path), eval_every=2)
    assert [row['round'] for row in result['trace']] == [0,2,4]


def test_plot_uses_two_distinct_time_axes(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    original = Axes.plot
    observed = {}

    def capture(self, *args, **kwargs):
        label = kwargs.get('label', '')
        if label.startswith('Centralized:') or label.startswith('P2P mean:'):
            observed[label] = list(args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', capture)
    out = run_training_sim(**_config(tmp_path, n=3))
    rows = out['trace']
    assert observed['Centralized: measured compute'] == [r['central_compute_s'] for r in rows]
    assert observed['P2P mean: sum(compute)/3'] == [r['p2p_compute_div_peers_s'] for r in rows]
    assert observed['P2P mean: sum(compute)/3'] != [r['p2p_compute_sum_s'] for r in rows]
