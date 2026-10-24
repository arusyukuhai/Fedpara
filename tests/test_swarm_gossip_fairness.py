"""Regression cases grounded in 8-peer report: thousands missing + disconnect storm."""
import asyncio
from types import SimpleNamespace

from patchlib.elasticnet import ElasticNode, REQUEST_PAGE
from patchlib.swarm_sim import add_parser


class FakeChannel:
    def __init__(self):
        self.readyState='open'
        self.bufferedAmount=0
        self.callbacks={}
    def on(self, name):
        def add(func):self.callbacks[name]=func;return func
        return add
    def send(self, payload):pass
    def close(self):
        self.readyState='closed'
        self.callbacks['close']()


def test_need_pagination_does_not_starve_later_1400_missing_updates():
    node=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                     origin='p'+'a'*32,remote='p'+'b'*32,
                     genesis='0'*64,unsafe=True)
    node.connected.set()
    node.pending={('p'+'b'*32,i) for i in range(1,1401)}
    pages=[]
    def send(kind,**kwargs):
        assert kind=='need'
        pages.append(tuple(tuple(v) for v in kwargs['events']))
        return True
    node.control=send
    now=1.0
    for _ in range(50):
        assert node.retry_missing(now)
        now+=.35
    seen={seq for page in pages for _,seq in page}
    assert seen==set(range(1,1401)),len(seen)
    assert all(1<=len(page)<=REQUEST_PAGE for page in pages)


def test_failed_need_send_does_not_advance_cursor():
    node=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                     origin='p'+'a'*32,remote='p'+'b'*32,
                     genesis='0'*64,unsafe=True)
    node.connected.set()
    node.pending={('p'+'b'*32,i) for i in range(1,72)}
    seq=[]
    def send(kind,**kwargs):
        seq.append(kwargs['events'])
        return len(seq)>1
    node.control=send
    assert not node.retry_missing(1)
    assert node.retry_missing(1.3)
    assert seq[0]==seq[1]


def test_saturated_inventory_queue_does_not_close_good_sctp():
    # An authenticated event is received while the model trainer is busy.
    async def scenario():
        node=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                         origin='p'+'a'*32,remote='p'+'b'*32,
                         genesis='0'*64,unsafe=True)
        ch=FakeChannel();node._attach_channel(ch)
        node.hello_received=True
        while not node.controls.full():
            node.controls.put_nowait({'type':'inventory'})
        node.counter=0
        from patchlib.elasticnet import encode_control
        wire=encode_control({'type':'inventory','peer':node.remote,
                             'genesis':node.genesis,'counter':1,
                             'events':[], 'done':False})
        ch.callbacks['message'](wire)
        assert node.connected.is_set()
        assert not node.protocol_failure.is_set()
        assert node.backpressure_drops==1
        ch.close()
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_swarm_cli_default_does_not_hide_unlimited_pressure():
    import argparse
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='cmd')
    add_parser(subs)
    assert p.parse_args(['swarm-sim','--output','x']).max_events_per_peer==0
    assert p.parse_args(['swarm-sim','--output','x','--max-events-per-peer','256']).max_events_per_peer==256


def test_diagnostics_identify_missing_origins_after_churn(tmp_path):
    import sqlite3
    from patchlib.swarm_sim import detailed_gossip_report
    p=[]
    for i in range(3):
        path=tmp_path/f'peer-{i}.sqlite'
        conn=sqlite3.connect(path)
        conn.execute('CREATE TABLE events(origin TEXT,seq INT,digest TEXT, wire BLOB, wire_len INT)')
        for origin,seq in ((('p'+'a'*32),1),(('p'+'b'*32),1),(('p'+'c'*32),1))[:i+1]:
            conn.execute('INSERT INTO events VALUES (?,?,?,?,?)',(origin,seq,'a'*64,b'blob',4))
        conn.commit();conn.close()
        p.append(path)
    report=detailed_gossip_report(p)
    assert report['unique_events']==3
    assert [x['missing'] for x in report['peers']]==[2,1,0]
    assert report['peers'][0]['missing_by_origin']=={'p'+'b'*32:1,'p'+'c'*32:1}
