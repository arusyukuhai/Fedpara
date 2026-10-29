"""Critical mixed-rank and non-barriered asynchronous live training regressions."""
import asyncio
from copy import deepcopy
import hashlib
import types

import pytest
import torch
from torch import nn

from patchlib.core import UniversalAdapter, PatchError
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.async_ingest import AsyncInboundPipeline
from test_livefed import Tiny


class MiniNeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=nn.Module()
        self.transformer.wte=nn.Embedding(512,64)
        self.transformer.wpe=nn.Embedding(256,64)
        self.transformer.h=nn.ModuleList([nn.Module()])
        layer=self.transformer.h[0]
        layer.attn=nn.Module();layer.attn.attention=nn.Module()
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            setattr(layer.attn.attention,name,nn.Linear(64,64))
        layer.mlp=nn.Module();layer.mlp.c_fc=nn.Linear(64,256);layer.mlp.c_proj=nn.Linear(256,64)
        self.lm_head=nn.Linear(64,512,bias=False)
        self.lm_head.weight=self.transformer.wte.weight
    def forward(self,ids):
        x=self.transformer.wte(ids)+self.transformer.wpe(torch.arange(ids.shape[1]))
        a=self.transformer.h[0]
        x=a.attn.attention.q_proj(x)+a.attn.attention.v_proj(x)
        x=a.mlp.c_proj(torch.relu(a.mlp.c_fc(x)))
        return self.lm_head(x)


def test_mixed_wide_rank4_tied_head_gradients_and_wire_shrink():
    torch.manual_seed(31)
    initial=MiniNeo()
    old=UniversalAdapter(deepcopy(initial),'m','r',rank=16,kind='loha',target_scope='gpt_neo_full')
    wide=[g for g in old.groups if g.canonical in ('transformer.wte.weight','transformer.wpe.weight')]
    assert len(wide)==2
    assert all(g.overlay.local.kind=='lora' and g.overlay.local.rank==4 for g in wide)
    assert any('lm_head.weight' in g.aliases for g in wide)
    assert sum('lm_head.weight' in g.aliases for g in old.groups)==1
    assert all(g.overlay.local.kind=='loha' for g in old.groups if g not in wide)
    assert old.counts()['wide_lora_groups']==2
    inp=torch.tensor([[2,5,7,3]])
    baseline=old.model(inp).detach().clone()
    assert torch.isfinite(baseline).all()
    loss=old.model(inp).square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for g in wide for p in g.overlay.local.parts)
    assert all(not p.requires_grad for p in old.model.parameters() if id(p) not in {id(t) for t in old.trainable_parameters()})
    with torch.no_grad():
        next(g for g in wide if g.canonical=='transformer.wte.weight').overlay.local.parts[-1].add_(.01)
    assert not torch.equal(baseline,old.model(inp).detach())
    assert all(not g.overlay.local.dense for g in old.groups)
    assert old.counts()['patch'] < 4 * old.counts()['original']
    # A pure LoHa embedding is substantially larger than the selected LoRA.
    saved=old.counts()['patch']
    pure=UniversalAdapter(deepcopy(initial),'m','r',rank=16,kind='loha',target_scope='all')
    assert saved < pure.counts()['patch']
    # Layout includes per-group rank/kind: old v5 events are incompatible.
    assert old.layout != pure.layout


def test_async_signed_inbound_does_not_seal_local_or_reset_optimizer(tmp_path):
    async def exercise():
        torch.manual_seed(121)
        base=Tiny()
        genesis=hashlib.sha256(b'test-async-v6').hexdigest()
        a=ContinualElasticEngine(UniversalAdapter(deepcopy(base),'mini','r',rank=3),
                'alice',genesis,EventJournal(tmp_path/'a.sqlite3',genesis),unsafe=True)
        b=ContinualElasticEngine(UniversalAdapter(deepcopy(base),'mini','r',rank=3),
                'bob',genesis,EventJournal(tmp_path/'b.sqlite3',genesis),unsafe=True)
        opt=torch.optim.AdamW(list(b.adapter.trainable_parameters()),lr=.001)
        x=b.adapter.model()
        x.backward(); opt.step();opt.zero_grad()
        moments=[opt.state[t]['exp_avg'].clone() for t in b.adapter.trainable_parameters()]
        before_local=[t.detach().clone() for t in b.adapter.trainable_parameters()]
        with a.adapter.lock:
            x=a.adapter.model();x.backward()
            torch.optim.SGD(list(a.adapter.trainable_parameters()),lr=.005).step()
        wire=await a.seal_local_responsive(1)
        seen=[]
        async def on_commit(info,data,source,fresh):seen.append((info.origin,source,fresh))
        pipe=AsyncInboundPipeline(b,on_commit,max_queue=1)
        pipe.start()
        assert pipe.submit('alice',wire)
        assert not pipe.submit('alice',wire)
        await pipe.drain()
        assert seen==[('alice','alice',True)]
        assert b.journal.count()==1
        assert all(torch.equal(x,p) for x,p in zip(before_local,b.adapter.trainable_parameters()))
        assert all(torch.equal(x,opt.state[p]['exp_avg']) for x,p in zip(moments,b.adapter.trainable_parameters()))
        assert pipe.submit('alice',wire)
        await pipe.drain()
        assert pipe.duplicates==1 and b.journal.count()==1
        assert seen[-1]==('alice','alice',False)
        # Crash/replay has exactly one committed update.
        replay=ContinualElasticEngine(UniversalAdapter(deepcopy(base),'mini','r',rank=3),
                'bob',genesis,EventJournal(tmp_path/'c.sqlite3',genesis),unsafe=True)
        assert replay.ingest(wire)
        for gg,rr in zip(b.adapter.groups,replay.adapter.groups):
            module,_,attr=gg.aliases[0].rpartition('.')
            assert torch.allclose(b.adapter.model.get_submodule(module).parametrizations[attr].original,
                replay.adapter.model.get_submodule(module).parametrizations[attr].original,atol=1e-6)
        await pipe.stop()
        a.journal.close();b.journal.close();replay.journal.close()
    asyncio.run(exercise())


def test_pipeline_propagates_bad_events_and_does_not_ack(tmp_path):
    async def exercise():
        genesis=hashlib.sha256(b'bad-event').hexdigest()
        engine=ContinualElasticEngine(UniversalAdapter(Tiny(),'mini','r',rank=3),
                'me',genesis,EventJournal(tmp_path/'bad.sqlite3',genesis),unsafe=True)
        acknowledged=[]
        async def on_commit(*args):acknowledged.append(args)
        pipe=AsyncInboundPipeline(engine,on_commit,max_queue=1)
        pipe.start()
        assert pipe.submit('evil',b'not-an-event')
        for _ in range(100):
            await asyncio.sleep(.001)
            if pipe.task.done():break
        with pytest.raises(PatchError):pipe.check()
        assert not acknowledged and engine.journal.count()==0
        await pipe.stop()
        engine.journal.close()
    asyncio.run(exercise())


def test_gui_live_real_command_async_defaults(tmp_path,monkeypatch):
    from patchlib.webview_controller import WebviewController
    from patchlib.roomid import create_room_id
    controller=WebviewController(tmp_path)
    cfg=controller.settings()
    assert cfg['kind']=='loha' and cfg['rank']==16
    assert cfg['wide_lora_rank']==4
    assert cfg['local_steps']==4 and cfg['rounds']==0 and 'peers' not in cfg
    assert cfg['reinit_every']==20 and cfg['lr']==.0001
    controller.save(cfg)
    calls=[]
    monkeypatch.setattr(controller.train,'start',lambda command,log:calls.append(command) or 98765)
    try:
        controller.start({'room':cfg['room'],'ws':cfg['ws']})
        cmd=calls[0]
        pairs={'--local-steps':'4','--rank':'16','--wide-lora-rank':'4',
               '--reinit-every':'20',
               '--rounds':'0','--lr':'0.0001'}
        for flag,value in pairs.items():
            assert cmd[cmd.index(flag)+1]==value, (flag,cmd)
        assert '--peers' not in cmd  # Elastic no fixed cohort barrier.
    finally:
        controller.close()
