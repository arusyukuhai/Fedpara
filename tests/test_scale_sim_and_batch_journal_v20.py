"""V20 signed bundle durability, NEED fairness and bounded scale simulator."""
import asyncio
import hashlib
import os

import pytest

from patchlib.elastic import EventJournal
from patchlib.core import PatchError
from patchlib import elastic as archive
from patchlib.elasticnet import ElasticNode, NEED_IN_FLIGHT_RETRY_SECONDS
from patchlib.scale_sim import Parameters, NetworkSimulation, publication_interval


def record(i):
    return (f'p{i:032x}',i+1, (f'signed-frame-{i}'*40).encode())


def test_group_append_one_fsync_exact_replay(tmp_path,monkeypatch):
    archive_path=tmp_path/'batch.sqlite3'
    journal=EventJournal(archive_path,'0'*64,sync='FULL')
    original_fsync=archive.os.fsync
    calls=[]
    def counted(fd):
        calls.append(fd)
        return original_fsync(fd)
    monkeypatch.setattr(archive.os,'fsync',counted)
    batch=[record(i) for i in range(3)]
    try:
        assert journal.append_many(batch)==[True,True,True]
        # Batch log requires only one explicit fsync instead of three.
        assert calls==[journal._log_fd]
        assert journal.append_many(batch)==[False,False,False]
        assert len(calls)==1
        assert journal.append_many([batch[0],record(3)])==[False,True]
        assert len(calls)==2
        assert journal.count()==4
        assert list(journal.events())==[row[2] for row in (*batch,record(3))]
    finally:
        journal.close()
    journal=EventJournal(archive_path,'0'*64,sync='FULL')
    try:
        assert list(journal.events())==[row[2] for row in (*batch,record(3))]
    finally:
        journal.close()


def test_batch_conflict_and_invalid_index_are_atomic(tmp_path):
    journal=EventJournal(tmp_path/'journal.sqlite3','0'*64,sync='FULL')
    try:
        prior=record(0)
        journal.append(*prior)
        with pytest.raises(PatchError,match='equivocation'):
            journal.append_many([record(1),(prior[0],prior[1],b'malicious event')])
        assert journal.count()==1
        with pytest.raises(PatchError,match='duplicate'):
            journal.append_many([record(2),record(2)])
        assert journal.count()==1
    finally:
        journal.close()


def test_partial_archive_failure_recovers_orphan_tail(tmp_path,monkeypatch):
    journal=EventJournal(tmp_path/'archive.sqlite3','0'*64,sync='FULL')
    before=journal._committed_end
    real_append=archive.append_frame
    counter=[0]
    def fail_once(*args,**kwargs):
        counter[0]+=1
        if counter[0]==2:
            raise OSError('simulated disk failure between frames')
        return real_append(*args,**kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(archive,'append_frame',fail_once)
            with pytest.raises(OSError,match='simulated'):
                journal.append_many([record(0),record(1)])
        assert journal.count()==0
        assert os.fstat(journal._log_fd).st_size>before
        # The next signed event append first trims the orphan, and archive
        # replay never applies any half-committed aggregate.
        assert journal.append_many([record(2),record(3)])==[True,True]
        assert journal.count()==2
    finally:
        journal.close()
    j=EventJournal(tmp_path/'archive.sqlite3','0'*64,sync='FULL')
    try:
        assert list(j.events())==[record(2)[2],record(3)[2]]
    finally:
        j.close()


def test_batch_async_worker_reuses_archive_index(tmp_path):
    journal=EventJournal(tmp_path/'async.sqlite3','0'*64,sync='FULL')
    try:
        async def run():
            first=await journal.append_many_async([record(0),record(1)])
            second=await journal.append_many_async([record(1),record(2)])
            return first,second
        assert asyncio.run(run())==([True,True],[False,True])
        assert journal.count()==3
    finally:
        journal.close()


def test_need_does_not_resend_inflight_within_cooldown():
    n=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,origin='p'+'a'*32,
                  remote='p'+'b'*32,genesis='0'*64,unsafe=True)
    n.connected.set()
    n.pending={('p'+'b'*32,1)}
    sent=[]
    n.control=lambda name,**kw:sent.append(kw['events']) or True
    assert n.retry_missing(10.)
    for t in (10.31,11.,12.,13.,15.):
        assert not n.retry_missing(t)
    assert len(sent)==1
    assert n.retry_missing(10+NEED_IN_FLIGHT_RETRY_SECONDS+.3)
    assert len(sent)==2
    n.pending.clear()
    n.retry_missing(18.)
    assert n.need_in_flight.get(('p'+'b'*32,1)) is not None  # cleaned on next nonempty need
    n.pending={('p'+'b'*32,2)}
    assert n.retry_missing(20.)
    assert ('p'+'b'*32,1) not in n.need_in_flight


def test_simulation_reproducible_churn_and_exact_dedup():
    config=Parameters(peers=20,events_per_peer=3,duration_s=8.,
        catchup_s=40.,event_kib=64,uplink_mbps=500.,churn_fraction=.25,
        loss_probability=.03)
    a=NetworkSimulation(config).run()
    b=NetworkSimulation(config).run()
    assert a==b
    assert a['published_events']>0
    assert a['event_copies_missing']==0
    assert a['exact_fedsum_peers']==20
    assert a['converged']
    assert a['sent_mib']>0


def test_network_sim_upper_scale_is_bounded():
    assert publication_interval(4,5)==4
    assert publication_interval(4,20)==8
    assert publication_interval(4,80)==16
    assert publication_interval(4,320)==32
    assert publication_interval(4,1280)==64
    config=Parameters(peers=256,events_per_peer=1,duration_s=3.,
        catchup_s=6.,event_kib=4,uplink_mbps=600.,churn_fraction=.1)
    s=NetworkSimulation(config)
    assert max(map(len,s.neighbors))<=8
    report=s.run()
    assert report['sent_packets']>=report['published_events']
    assert 0<=report['coverage']<=1
    assert report['publish_steps']==16


def test_input_validation():
    with pytest.raises(ValueError):
        Parameters(peers=1).validate()
    with pytest.raises(ValueError):
        Parameters(bundle_size=9).validate()
    with pytest.raises(ValueError):
        Parameters(compression_ratio=0.).validate()


def test_need_repair_coalesces_three_source_events(monkeypatch):
    from patchlib import elasticnet as net
    from collections import OrderedDict
    from patchlib.aggregate_wire import is_aggregate, unpack_aggregate
    n=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,origin='p'+'a'*32,
                  remote='p'+'b'*32,genesis='0'*64,unsafe=True)
    n.remote_supports_aggregate=True
    n.hello_received=True
    n._channel=object()
    sources=[f'p{i:032x}' for i in range(3)]
    requested=[(p,1) for p in sources]
    n.requested_events=OrderedDict((x,None) for x in requested)
    wires={(p,1):b'FPEL1\0'+(p.encode()+b'0'*2000) for p in sources}
    n.lookup_requested_event=lambda origin,seq:wires[(origin,seq)]
    sent=[]
    async def fake_transmit(channel, data, **kwargs):
        sent.append(data)
    monkeypatch.setattr(net,'transmit',fake_transmit)

    async def run():
        task=asyncio.create_task(n._pump())
        try:
            for _ in range(30):
                if sent: break
                await asyncio.sleep(.01)
            assert len(sent)==1
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    asyncio.run(run())
    assert is_aggregate(sent[0])
    assert set(unpack_aggregate(sent[0],max_bytes=n.max_bytes))==set(wires.values())
    assert n.repair_sent_events==3
    assert not n.requested_events


def test_async_need_lookup_is_not_on_event_loop(tmp_path):
    journal=EventJournal(tmp_path/'need.sqlite3','0'*64,sync='FULL')
    try:
        journal.append_many([record(0),record(1)])
        async def read():
            return await journal.get_async(record(0)[0],record(0)[1])
        assert asyncio.run(read())==record(0)[2]
    finally:
        journal.close()


def test_failed_aggregate_repair_requeues_all_and_does_not_count_sent(monkeypatch):
    from patchlib import elasticnet as net
    from collections import OrderedDict
    n=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,origin='p'+'a'*32,
                  remote='p'+'b'*32,genesis='0'*64,unsafe=True)
    n.remote_supports_aggregate=True
    n.hello_received=True
    n._channel=object()
    ids=[(f'p{i:032x}',1) for i in range(3)]
    n.requested_events=OrderedDict((x,None) for x in ids)
    n.lookup_requested_event=lambda origin,seq:b'FPEL1\0'+(origin.encode()+b'0'*2000)
    async def fail_transmit(*args,**kwargs):
        raise ConnectionError('simulated disconnected datachannel')
    monkeypatch.setattr(net,'transmit',fail_transmit)
    asyncio.run(n._pump())
    assert list(n.requested_events)==ids
    assert n.aggregate_sent_batches==0
    assert n.aggregate_bytes_avoided==0
    assert n.repair_sent_events==0
    assert n.protocol_failure.is_set()
