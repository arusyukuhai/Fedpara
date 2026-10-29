import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from patchlib.core import PatchError
from patchlib.online_generation import _digest
from patchlib.epoch_multihop_v56 import CertifiedChain,MultiEpochCatchup,DurableOfflineOutbox
from patchlib.epoch_sharded_rcu_v56 import ShardedEpochRCU,ShardedGeneration


class Ledger:
    genesis='0'*64
    def __init__(self,count=4):
        self.commits={}
        parent=self.genesis
        for epoch in range(1,count+1):
            p={'epoch':epoch,'parent':parent,'manifest':f'{epoch:064x}'}
            self.commits[epoch]={'proposal':p}
            parent=_digest(p)
    def certified_epoch(self,e):return self.commits.get(e)
    def latest(self):return self.commits[max(self.commits)] if self.commits else None


class Router:
    def __init__(self):self.current=SimpleNamespace(epoch=0)


class Service:
    def __init__(self,ledger):
        self.ledger=ledger;self.router=Router();self.fail_at=None;self.calls=[]
    async def activate_next(self):
        n=self.router.current.epoch+1
        self.calls.append(n)
        if self.fail_at==n:raise ConnectionError('missing checkpoint shard')
        self.router.current.epoch=n
        return True


@pytest.mark.asyncio
async def test_three_epoch_offline_recovery_adjacent_only():
    ledger=Ledger(4);service=Service(ledger)
    recovery=MultiEpochCatchup(service,max_hops_per_poll=2)
    assert await recovery.poll()==2
    assert service.router.current.epoch==2
    service.fail_at=3
    with pytest.raises(ConnectionError):await recovery.poll()
    assert service.router.current.epoch==2
    service.fail_at=None
    assert await recovery.poll()==2
    assert service.calls==[1,2,3,3,4]


@pytest.mark.asyncio
async def test_missing_or_broken_certificate_blocks_recovery():
    ledger=Ledger(3);service=Service(ledger)
    ledger.commits[2]['proposal']['parent']='f'*64
    with pytest.raises(PatchError,match='broken parent'):
        await MultiEpochCatchup(service).poll()
    assert not service.calls
    ledger=Ledger(3);service=Service(ledger)
    del ledger.commits[2]
    with pytest.raises(PatchError,match='missing authenticated'):
        await MultiEpochCatchup(service).poll()
    assert not service.calls


def test_quarantine_never_drops_unacknowledged_old_wire(tmp_path):
    q=DurableOfflineOutbox(tmp_path,max_bytes=1024)
    assert q.hold(epoch=0,origin='a',seq=1,wire=b'a'*600)
    assert not q.hold(epoch=0,origin='a',seq=1,wire=b'a'*600)
    with pytest.raises(PatchError,match='equivocating'):
        q.hold(epoch=0,origin='a',seq=1,wire=b'wrong')
    with pytest.raises(PatchError,match='capacity'):
        q.hold(epoch=0,origin='a',seq=2,wire=b'b'*600)
    with pytest.raises(PatchError):q.acknowledge(epoch=0,origin='a',seq=1,verified_archive=True)
    assert list(q.iter_epoch(0))==[('a',1,b'a'*600)]
    q.close()
    q=DurableOfflineOutbox(tmp_path,max_bytes=1024)
    assert list(q.iter_epoch(0))==[('a',1,b'a'*600)]
    q.close()


def generation(tmp_path,epoch):
    c=tmp_path/f'c{epoch}';o=tmp_path/f'o{epoch}'
    c.mkdir();o.mkdir()
    (c/'shard-0000').write_bytes(b'weights')
    (o/'moment-0000').write_bytes(b'moments')
    return ShardedGeneration(epoch,f'{epoch:064x}',f'{epoch+1:064x}',c,o)


@pytest.mark.asyncio
async def test_sharded_epoch_rcu_reader_survives_switch_and_writer_fences(tmp_path):
    generations={i:generation(tmp_path,i) for i in (1,2)}
    rcu=ShardedEpochRCU(tmp_path/'rcu',verify_generation=lambda g:
        (g.checkpoint_dir/'shard-0000').read_bytes()==b'weights' and
        (g.optimizer_dir/'moment-0000').read_bytes()==b'moments')
    await rcu.stage(generations[1])
    await rcu.publish(certified_step_boundary=lambda old,new:True)
    await rcu.stage(generations[2])
    started=asyncio.Event();release=asyncio.Event()
    async def old_writer():
        async with rcu.step() as active:
            assert active.epoch==1
            started.set();await release.wait()
    writer=asyncio.create_task(old_writer());await started.wait()
    fenced=asyncio.Event()
    async def fence(old,new):fenced.set();return True
    activation=asyncio.create_task(rcu.publish(certified_step_boundary=fence))
    await asyncio.sleep(0.02)
    assert not fenced.is_set()
    release.set();await writer
    assert (await activation).epoch==2
    async with rcu.step() as active:assert active.epoch==2
    assert rcu.pinned_epochs()==frozenset({2})
    recovered=ShardedEpochRCU(tmp_path/'rcu',verify_generation=rcu.verify_generation)
    out=recovered.recover(lambda e,p,m:generations[e])
    assert out.epoch==2


@pytest.mark.asyncio
async def test_sharded_bad_fence_and_shard_modification_fail_closed(tmp_path):
    rcu=ShardedEpochRCU(tmp_path/'rcu',verify_generation=lambda g:
        (g.checkpoint_dir/'shard-0000').read_bytes()==b'weights')
    await rcu.stage(generation(tmp_path,1))
    with pytest.raises(PatchError,match='fence'):
        await rcu.publish(certified_step_boundary=lambda old,new:False)
    # A rejected unknown local handoff cannot silently reopen writer admission.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(rcu.step().__aenter__(),timeout=.05)
    rcu2=ShardedEpochRCU(tmp_path/'other',verify_generation=rcu.verify_generation)
    g=generation(tmp_path,2)
    await rcu2.stage(g)
    (g.checkpoint_dir/'shard-0000').write_bytes(b'tampered')
    with pytest.raises(PatchError,match='verification failed'):
        await rcu2.publish(certified_step_boundary=lambda old,new:True)
    assert not rcu2.pointer.exists()
