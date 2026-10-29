import csv
import json
from pathlib import Path

from scripts.sweep_p2p_training import main


def test_peer_sweep_outputs_and_graphs(tmp_path):
    out = tmp_path / 'sweep'
    main([
        '--model', 'toy',
        '--dataset', 'builtin',
        '--peers-list', '2,4',
        '--rounds', '3',
        '--local-steps', '2',
        '--reinit-every', '2',
        '--batch-size', '2',
        '--context', '8',
        '--width', '24',
        '--vocab', '32',
        '--sequences', '32',
        '--validation-sequences', '12',
        '--threads', '1',
        '--eval-every', '1',
        '--output-dir', str(out),
    ])
    assert (out / 'peer-sweep-loss.png').is_file()
    assert (out / 'peer-sweep-vs-peers.png').is_file()
    assert (out / 'peer-sweep-summary.csv').is_file()
    assert (out / 'peer-sweep-trace.csv').is_file()
    payload = json.loads((out / 'peer-sweep-result.json').read_text())
    assert payload['peers_list'] == [2, 4]
    with (out / 'peer-sweep-summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['peers']) for r in rows] == [2, 4]
    with (out / 'peer-sweep-trace.csv').open() as f:
        trace = list(csv.DictReader(f))
    peers_seen = sorted({int(r['peers']) for r in trace})
    assert peers_seen == [2, 4]
    assert (out / 'peers-0002' / 'p2p-training-result.json').is_file()
    assert (out / 'peers-0004' / 'p2p-training-result.json').is_file()
