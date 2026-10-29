from copy import deepcopy
import hashlib
import pytest
import torch
from patchlib.core import PatchError,UniversalAdapter
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(4,4,bias=False)

    def forward(self):
        return self.layer(torch.ones(4)).square().sum()


def make_engine(model,journal,genesis,self_id,trusted,signer=None):
    adapter=UniversalAdapter(model,'tiny','v1',kind='loha',rank=2)
    engine=ContinualElasticEngine(adapter,self_id,genesis,
        EventJournal(journal,genesis,max_bytes=0),unsafe=False,trusted=trusted,signer=signer)
    return engine,adapter


def test_recover_signed_carry_and_prevent_double_apply(tmp_path):
    torch.manual_seed(123)
    base=Tiny()
    a=deepcopy(base); b=deepcopy(base)
    private=tmp_path/'a.key';public=tmp_path/'a.pub'
    generate_keypair(private,public)
    signed=signer_from_pem(private); trusted={'alice':verifier_from_pem(public)}
    old_genesis='a'*64; new_genesis='b'*64
    original,adapt=make_engine(a,tmp_path/'source.db',old_genesis,'alice',trusted,signed)
    opt=torch.optim.AdamW(adapt.trainable_parameters(),lr=.005)
    original.adapter.model().backward();opt.step();opt.zero_grad(set_to_none=True)
    wire=original.seal_local(1)
    recipient,_=make_engine(b,tmp_path/'recipient.db',new_genesis,'bob',trusted)
    archive=CrossEpochCarryArchive(tmp_path/'carries.db',source_genesis=old_genesis,
        target_genesis=new_genesis,certified_manifest='c'*64,epoch=1,cutoffs={'alice':0})
    assert archive.replay(recipient)==0
    initial=b.layer.parametrizations.weight.original.detach().clone()
    assert archive.accept(recipient,wire)
    changed=b.layer.parametrizations.weight.original.detach().clone()
    assert not torch.equal(changed,initial)
    assert archive.accept(recipient,wire) is False
    assert torch.equal(b.layer.parametrizations.weight.original,changed)
    archive.close();recipient.journal.close()
    rebuilt=deepcopy(base)
    recovery,_=make_engine(rebuilt,tmp_path/'replay.db',new_genesis,'bob',trusted)
    archive=CrossEpochCarryArchive(tmp_path/'carries.db',source_genesis=old_genesis,
        target_genesis=new_genesis,certified_manifest='c'*64,epoch=1,cutoffs={'alice':0})
    assert archive.replay(recovery)==1
    assert torch.allclose(rebuilt.layer.parametrizations.weight.original,changed,atol=1e-6)
    with pytest.raises(PatchError,match='checkpoint'):
        archive.accept(recovery,wire) if False else CrossEpochCarryArchive(
            tmp_path/'different.db',source_genesis=old_genesis,target_genesis=new_genesis,
            certified_manifest='c'*64,epoch=1,cutoffs={'alice':1})._check(recovery,wire)
    archive.close();recovery.journal.close();original.journal.close()


def test_carry_rejects_unsafe_engine_and_bad_source(tmp_path):
    import types
    archive=CrossEpochCarryArchive(tmp_path/'carries.db',source_genesis='a'*64,
        target_genesis='b'*64,certified_manifest='c'*64,epoch=1,cutoffs={})
    with pytest.raises(PatchError):archive.replay(types.SimpleNamespace(genesis='b'*64,unsafe=True))
    with pytest.raises(PatchError):CrossEpochCarryArchive(tmp_path/'bad.db',source_genesis='a'*64,
        target_genesis='a'*64,certified_manifest='c'*64,epoch=1,cutoffs={})
    archive.close()
