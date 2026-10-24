"""Large-swarm bootstrap barriers prevent all newcomers sharing one moving tip."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from patchlib.swarm_sim import SwarmRunner, SimulationError


def test_sixteen_peer_bootstraps_in_verified_waves(tmp_path,monkeypatch):
    async def scenario():
        args=SimpleNamespace(output=tmp_path/'run',peers=16,seed=5,
                             bootstrap_batch_size=0,join_interval=0,
                             startup_timeout=1)
        sim=SwarmRunner(args)
        sim.root.mkdir()
        launched=[]
        pending=[]
        class Alive:
            returncode=None
        async def spawn(i):
            sim.workers[i].process=Alive()
            sim.workers[i].starts=1
            launched.append(i)
            wave=len(launched)//4
            # A previous wave must be verified before the fifth, ninth, or
            # thirteenth process can start.
            if i in (5,9,13):
                assert all(sim.workers[j].snapshot and sim.workers[j].joined
                           for j in range(1,i))
            async def finish():
                await asyncio.sleep(.015)
                sim.workers[i].joined=True
                sim.workers[i].snapshot=True
            pending.append(asyncio.create_task(finish()))
        monkeypatch.setattr(sim,'spawn',spawn)
        await sim.join_newcomers()
        await asyncio.gather(*pending)
        assert launched==list(range(1,16))
        starts=[e for e in sim.events if e['event']=='bootstrap-wave-start']
        completed=[e for e in sim.events if e['event']=='bootstrap-wave-complete']
        assert [e['batch_size'] for e in starts]==[4,4,4,3]
        assert [e['verified_snapshots'] for e in completed]==[4,8,12,15]
    asyncio.run(scenario())


def test_startup_snapshot_failure_saves_peer_diagnostics(tmp_path,monkeypatch):
    async def scenario():
        args=SimpleNamespace(output=tmp_path/'swarm',peers=16,seed=5,
                             bootstrap_batch_size=2,join_interval=0,
                             startup_timeout=.05)
        sim=SwarmRunner(args)
        sim.root.mkdir()
        class Alive:
            returncode=None
        async def spawn(i):
            w=sim.workers[i]
            w.starts=1
            w.process=Alive()
            w.log_tail=["JOIN WAIT: snapshot_progress={'source_stage': 'waiting-state-channel'}"]
            if i==1:
                w.joined=w.snapshot=True
        monkeypatch.setattr(sim,'spawn',spawn)
        with pytest.raises(SimulationError,match='startup-diagnostic.json'):
            await sim.join_newcomers()
        report=json.loads((sim.root/'startup-diagnostic.json').read_text())
        assert report['wave']==1
        assert report['wave_peer_numbers']==[1,2]
        assert report['join_completed']==1
        assert report['workers'][1]['snapshot_verified'] is False
        assert report['workers'][1]['recent_relevant_logs']
    asyncio.run(scenario())


def test_bootstrap_flag_is_distinct_from_epoch_commit_trigger():
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'patchlib'/'elasticnet.py'
    source=p.read_text(encoding='utf-8')
    assert "startswith('stop optimization;')" in source
    assert 'if epoch_quiesce and not epoch_probe_started' in source
