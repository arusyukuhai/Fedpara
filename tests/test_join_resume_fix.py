"""Regression coverage for 2-window join and restart after complete outage."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch

from patchlib.profiles import allocate_profile
from patchlib.roomid import auto_identity, RoomDiscovery, create_room_id
from patchlib.elastic import EventJournal, ElasticEngine
from patchlib.localprogress import save_working, restore_working, snapshot_path
from patchlib.core import UniversalAdapter, PatchError
from patchlib.webview_controller import WebviewController
from test_livefed import Tiny


def test_two_gui_instances_get_distinct_reusable_identity_and_journal(tmp_path,monkeypatch):
    room=create_room_id()
    payload={'room':room,'model':'org/model','revision':'main','task':'causal-lm',
            'sources':[{'repo':'org/data','revision':'main','config':None,'split':'train',
                        'columns':{'text':'text'},'weight':1.0}]}
    a=WebviewController(tmp_path);b=WebviewController(tmp_path)
    a.save(payload)
    for ctrl in (a,b):
        monkeypatch.setattr(ctrl.train,'start',lambda cmd,log: 12345)
    kw={'room':room,'ws':'ws://127.0.0.1:8080/ws','kind':'loha','rank':32}
    aa=a.start(kw);bb=b.start(kw)
    assert aa['base']==bb['base']  # immutable genesis shared
    assert aa['journal']!=bb['journal']  # events/ID/optimizer never shared
    assert aa['profile']!=bb['profile']
    aid,*_=auto_identity(Path(aa['journal']+'.identity.pem'))
    bid,*_=auto_identity(Path(bb['journal']+'.identity.pem'))
    assert aid!=bid
    # Both identities show up as peers in a room membership exchange.
    disc_a=RoomDiscovery('ws://127.0.0.1:8080/ws',room,aid)
    disc_b=RoomDiscovery('ws://127.0.0.1:8080/ws',room,bid)
    disc_a.accept({'type':'group-members','peers':[bid]})
    disc_b.accept({'type':'group-members','peers':[aid]})
    assert len(disc_a.members)==len(disc_b.members)==1
    a.close();b.close()
    c=WebviewController(tmp_path);monkeypatch.setattr(c.train,'start',lambda cmd,log: 12345)
    cc=c.start(kw)
    assert cc['journal'] in (aa['journal'],bb['journal'])
    cid,*_=auto_identity(Path(cc['journal']+'.identity.pem'))
    assert cid in (aid,bid)
    c.close()


def test_legacy_history_retained_and_locked(tmp_path):
    root=tmp_path/'room'
    root.mkdir()
    legacy=root/'events.sqlite3'
    legacy.write_bytes(b'old')
    a=allocate_profile(root)
    assert a.name=='legacy'
    assert a.journal==legacy
    b=allocate_profile(root)
    assert b.journal!=legacy
    a.close();b.close()


def test_journal_protects_against_parallel_trainers(tmp_path):
    genesis='a'*64
    journal=tmp_path/'alice.sqlite3'
    j=EventJournal(journal,genesis)
    with pytest.raises(PatchError,match='already used'):
        EventJournal(journal,genesis)
    j.close()
    again=EventJournal(journal,genesis)
    again.close()


def test_unsealed_lora_and_adamw_survive_crash(tmp_path):
    torch.manual_seed(93)
    base=Tiny()
    gen=hashlib.sha256(b'consistent start').hexdigest()
    first=UniversalAdapter(deepcopy(base),'model','v',rank=3)
    journal=EventJournal(tmp_path/'alice.sqlite3',gen)
    opt=torch.optim.AdamW(first.trainable_parameters(),lr=0.002)
    loss=first.model.forward()
    loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    original=[p.detach().cpu().clone() for p in first.trainable_parameters()]
    orig_moments=[{k:v.detach().cpu().clone() for k,v in opt.state[p].items()} for p in first.trainable_parameters()]
    save_working(journal,first,opt,1)
    journal.close()
    # Crash and cold reboot using identical genesis model & local journal.
    restored=UniversalAdapter(deepcopy(base),'model','v',rank=3)
    journal2=EventJournal(tmp_path/'alice.sqlite3',gen)
    engine=ElasticEngine(restored,'alice',gen,journal2,unsafe=True)
    engine.replay()
    opt2=torch.optim.AdamW(restored.trainable_parameters(),lr=0.002)
    assert restore_working(journal2,restored,opt2)==1
    for before,after in zip(original,restored.trainable_parameters()):
        assert torch.equal(before,after.cpu())
    for old,p in zip(orig_moments,restored.trainable_parameters()):
        for key,value in old.items():
            assert torch.equal(value,opt2.state[p][key].cpu())
    # Once an event is committed, the obsolete working state cannot be replayed.
    engine.seal_local(1)
    assert restore_working(journal2,restored,opt2)==0
    journal2.close()
