"""Self-only, bounded local live graph tests; no network or Torch required."""
import json
import os
import time

from patchlib.live_telemetry import LocalTelemetry, recent_samples


def test_private_telemetry_is_independent_between_two_windows(tmp_path):
    a, b = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    first = LocalTelemetry(a, sample_seconds=0)
    second = LocalTelemetry(b, sample_seconds=0)
    first.record(steps=4,loss=3.1,emitted=1,inbound_applied=0,inbox=0,channels=1)
    second.record(steps=2,loss=8.9,emitted=0,inbound_applied=5,inbox=2,channels=3)
    first.close();second.close()
    assert recent_samples(a)[0]['loss'] == 3.1
    assert recent_samples(a)[0]['inbound_applied'] == 0
    assert recent_samples(b)[0]['loss'] == 8.9
    assert os.stat(a).st_mode & 0o777 == 0o600


def test_throttle_nan_and_unique_path(tmp_path):
    path=tmp_path/'telemetry.jsonl'
    writer=LocalTelemetry(path,sample_seconds=99999)
    assert writer.record(steps=0,loss=float('nan'),emitted=0,inbound_applied=0,inbox=0,channels=0,state='waiting')
    assert not writer.record(steps=1,loss=1.0,emitted=0,inbound_applied=0,inbox=0,channels=0)
    assert writer.record(steps=2,loss=1.0,emitted=0,inbound_applied=0,inbox=0,channels=0,force=True)
    writer.close()
    assert recent_samples(path)[0]['loss'] is None
    assert [r['steps'] for r in recent_samples(path)] == [0,2]
    try:
        LocalTelemetry(path)
        assert False, 'should reject pre-existing path'
    except FileExistsError:
        pass


def test_reader_ignores_partial_lines_and_keeps_recent_samples(tmp_path):
    path=tmp_path/'trace.jsonl'
    with path.open('wb') as f:
        for i in range(500):
            f.write((json.dumps({'steps':i,'elapsed_s':i*1.5,'loss':10.0-i/1000,'state':'training'})+'\n').encode())
        f.write(b'{"steps":501')
    results=recent_samples(path,max_samples=20)
    assert len(results)==20
    assert results[0]['steps']==480 and results[-1]['steps']==499
    with path.open('ab') as f:
        f.write(b'\n')
    # Malformed trailing line ignored.
    assert recent_samples(path,max_samples=1)==[]


def test_empty_and_short_reads(tmp_path):
    assert recent_samples(tmp_path/'missing')==[]
    writer=LocalTelemetry(None)
    assert not writer.record(steps=1,loss=1.,emitted=0,inbound_applied=0,inbox=0,channels=0)
    writer.close()
