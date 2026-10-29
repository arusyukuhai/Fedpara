import asyncio
from copy import deepcopy
import torch
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.async_ingest import AsyncInboundPipeline
from tests.test_epoch_carry_v46 import Tiny,make_engine


def test_live_inbound_pipeline_signed_old_genesis_carry(tmp_path):
    async def scenario():
        torch.manual_seed(7)
        base=Tiny();old_model=deepcopy(base);new_model=deepcopy(base)
        prv=tmp_path/'a.pem';pub=tmp_path/'a.pub';generate_keypair(prv,pub)
        keys={'alice':verifier_from_pem(pub)}
        old,_=make_engine(old_model,tmp_path/'old.db','a'*64,'alice',keys,signer_from_pem(prv))
        adapter=old.adapter
        optimizer=torch.optim.AdamW(adapter.trainable_parameters(),lr=.005)
        old_model().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        wire=old.seal_local(1)
        new,_=make_engine(new_model,tmp_path/'new.db','b'*64,'bob',keys)
        archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis='a'*64,
            target_genesis='b'*64,certified_manifest='c'*64,epoch=1,cutoffs={})
        assert archive.replay(new)==0
        observations=[]
        async def committed(info,wire,peer,fresh):
            observations.append((info.origin,info.seq,peer,fresh))
        pipe=AsyncInboundPipeline(new,committed,carry_archive=archive)
        pipe.start()
        try:
            assert pipe.submit('neighbor',wire)
            await asyncio.wait_for(pipe.drain(),timeout=10)
            assert pipe.submit('neighbor',wire)
            await asyncio.wait_for(pipe.drain(),timeout=10)
            assert observations==[('alice',1,'neighbor',True),('alice',1,'neighbor',False)]
            assert pipe.applied==1
        finally:
            await pipe.stop()
            old.journal.close();new.journal.close();archive.close()
    asyncio.run(scenario())
