"""Streaming dataset iteration must not block the aiortc event loop."""
import asyncio
import time
from patchlib.elasticnet import run_step_offloop


def test_slow_streaming_batch_does_not_block_heartbeats():
    def batches():
        time.sleep(.3)  # Stand-in for slow Hugging Face streaming read
        yield 7
    async def scenario():
        task=asyncio.create_task(run_step_offloop(
            lambda model,adapter,optimizer,batch,args: batch*2,
            None,None,None,iter(batches()),None))
        start=time.monotonic()
        await asyncio.sleep(.05)
        assert time.monotonic()-start<.2
        assert await asyncio.wait_for(task,3)==14
    asyncio.run(scenario())
