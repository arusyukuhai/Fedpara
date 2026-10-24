"""HF request spam is not a P2P failure; verify responsiveness & telemetry."""
import asyncio
import time
from types import SimpleNamespace
from patchlib.elasticnet import run_step_offloop


def test_slow_dataset_emits_progress_but_does_not_restart_fetch(capsys):
    hits=[]
    class OneBatch:
        def __iter__(self): return self
        def __next__(self):
            hits.append(1)
            time.sleep(.115)
            return 'batch'
    async def go():
        ticks=0
        async def timer():
            nonlocal ticks
            for _ in range(8):
                await asyncio.sleep(.017)
                ticks+=1
        task=asyncio.create_task(timer())
        answer=await run_step_offloop(lambda m,a,o,b,args: b, None,None,None,
                                      iter(OneBatch()),SimpleNamespace(peer_id='abc',data_wait_report_seconds=.025))
        await task
        return answer,ticks
    answer,ticks=asyncio.run(go())
    out=capsys.readouterr().out
    assert answer=='batch' and hits==[1]
    assert ticks>=6
    assert 'DATA WAIT stage=data-fetch' in out


def test_slow_training_is_reported_as_model_step(capsys):
    def slow_step(*_):
        time.sleep(.09)
        return .5
    async def go():
        return await run_step_offloop(slow_step,None,None,None,iter([1]),
                                      SimpleNamespace(peer_id='abc',data_wait_report_seconds=.02))
    assert asyncio.run(go())==.5
    assert 'DATA WAIT stage=model-step' in capsys.readouterr().out


def test_fast_data_has_no_wait_logs(capsys):
    async def go():
        return await run_step_offloop(lambda m,a,o,b,args:b,None,None,None,
                     iter([2]),SimpleNamespace(peer_id='abc',data_wait_report_seconds=.1))
    assert asyncio.run(go())==2
    assert not capsys.readouterr().out
