"""Regression tests for connection storms and heavyweight model export stalls."""
import asyncio
import time
from types import SimpleNamespace

from patchlib.elasticnet import pump_heartbeats
from patchlib.p2p import PatchNode


def test_keepalive_independent_of_slow_checkpoint():
    async def scenario():
        sent=[]
        class Node:
            connected=asyncio.Event()
            def control(self, name):
                sent.append(name)
        node=Node();node.connected.set()
        peer=SimpleNamespace(node=node,disabled=False)
        stop=asyncio.Event()
        task=asyncio.create_task(pump_heartbeats({'peer':peer},stop,
                interval=.02,peer_timeout=.6))
        try:
            # Model export/checkpoint runs off the training coroutine.
            await asyncio.to_thread(time.sleep,.47)
            assert sent.count('heartbeat')>=2, sent
        finally:
            stop.set();await task
    asyncio.run(scenario())


def test_phase_change_is_observable():
    node=PatchNode('ws://127.0.0.1:8080/ws','a'*24)
    assert node.phase=='created'
    node.role='initiator'
    node.set_phase('gathering-offer')
    assert node.phase=='gathering-offer'
    assert node.role=='initiator'
