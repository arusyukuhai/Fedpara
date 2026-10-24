"""Regression for 8-peer crash/rejoin stalled on a moving gossip inventory."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json

from patchlib.join_guard import established_member_decision, initial_join_decision
from patchlib.swarm_sim import Worker, SwarmRunner


def test_verified_member_rejoins_even_while_neighbors_are_catching_up():
    active = {'p' + format(i, '032x'): False for i in range(7)}
    # Before the fix, restart was treated like a brand-new peer and waited
    # indefinitely for every live neighbor to report inventory-ready.
    cold = initial_join_decision(initial_complete=False, lobby_online=True,
        lobby_members=set(active), neighbors=active,
        durable_history=True, signaling_seen=True)
    assert not cold.can_train
    resumed = established_member_decision(persisted_ready=True,
        fresh_join=False, lobby_online=True, signaling_seen=True)
    assert resumed is not None and resumed.can_train
    assert 'rejoining-' in resumed.reason


def test_first_time_peer_is_still_blocked_on_incomplete_snapshot():
    for fresh, persisted in ((True, False), (True, True), (False, False)):
        assert established_member_decision(persisted_ready=persisted,
            fresh_join=fresh,lobby_online=True,signaling_seen=False) is None


def test_returning_member_can_resume_without_signaling_if_previously_confirmed():
    assert established_member_decision(persisted_ready=True,fresh_join=False,
        lobby_online=False,signaling_seen=True).can_train
    assert established_member_decision(persisted_ready=True,fresh_join=False,
        lobby_online=False,signaling_seen=False) is None


def test_monitor_captures_rejoin_and_new_hello_only(tmp_path):
    async def run():
        worker=Worker(6,tmp_path/'peer-06')
        worker.root.mkdir(parents=True)
        # subprocess is just a controlled output producer; no fake WebRTC
        # success is claimed by this harness test.
        script=("print('elastic peer=pa genesis=abc recovered=24 events', flush=True);"
                "print('REJOIN READY: rejoining-with-durable-model-and-background-anti-entropy', flush=True);"
                "print('SIGNED HELLO OK peer=pb', flush=True)")
        import sys
        proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',script,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        with (worker.root/'run-02.log').open('w') as fp:
            runner=SwarmRunner(SimpleNamespace(output=tmp_path/'sim',peers=8,seed=1))
            await runner.monitor(worker,proc,fp)
        assert worker.joined and worker.rejoin_ready
        assert worker.run_signed_hello==1
    asyncio.run(run())


def test_rejoin_diagnostic_retains_last_logs(tmp_path):
    runner=SwarmRunner(SimpleNamespace(output=tmp_path/'sim',peers=8,seed=1))
    worker=runner.workers[6]
    worker.root.mkdir(parents=True)
    worker.starts=2
    worker.log_tail=['JOIN WAIT: reason=recover', 'error example']
    runner.report_rejoin_failure(6)
    data=json.loads((worker.root/'rejoin-diagnostic.json').read_text())
    assert data['peer']==6 and data['starts']==2
    assert data['last_logs'][-1]=='error example'
    assert 'run-02.log' in data['log_file']
