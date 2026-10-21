import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from patchlib import cli
from patchlib.core import PatchError, decode_patch
from patchlib.state import inspect_state


def test_custom_guard_blocks_host_execution(tmp_path, monkeypatch):
    p=tmp_path/'user.py'
    sentinel=tmp_path/'marker'
    p.write_text(f"open({str(sentinel)!r}, 'w').write('EXECUTED')\ndef build_model():\n return None\n")
    args=types.SimpleNamespace(trust_remote_code=False,factory_file=str(p),model='x',revision='a'*40,
            allow_legacy_bin=False,cmd='train')
    monkeypatch.delenv('FEDPARA_ISOLATED_RUNTIME',raising=False)
    with pytest.raises(PermissionError, match='Docker'):
        cli._untrusted_guard(args)
    assert not sentinel.exists()


def test_custom_factory_training_state_and_hash_binding(tmp_path, monkeypatch):
    p=tmp_path/'my_model.py'
    p.write_text('''import torch
from torch import nn
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(10,8)
        self.lin=nn.Linear(8,10)
        self.register_buffer('counter',torch.tensor(11))
    def forward(self, input_ids, labels):
        logits=self.lin(self.emb(input_ids))
        loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
        return {'loss':loss}
def build_model():
    torch.manual_seed(12)
    return Toy()
''')
    samples=tmp_path/'batches'; samples.mkdir()
    ids=torch.tensor([[1,2,3]],dtype=torch.long)
    save_file({'input_ids':ids,'labels':ids.clone()},str(samples/'example.safetensors'))
    out=tmp_path/'a.fpatch'; state=tmp_path/'a.fstate'
    monkeypatch.setenv('FEDPARA_ISOLATED_RUNTIME','1')
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    monkeypatch.setattr(sys,'argv',['cli','train','--model','/model','--factory-file',str(p),
             '--batches-dir',str(samples),'--peer-id','alpha','--rank','3',
             '--steps','2','--output',str(out),'--full-state-output',str(state)])
    cli.main()
    meta,_=decode_patch(out.read_bytes())
    assert meta['revision'].startswith('factory-sha256:')
    assert inspect_state(state)['revision']==meta['revision']
    assert len(inspect_state(state)['tensors'])>=4
    # Changed source code must produce a different revision identity.
    p.write_text(p.read_text()+'\n# new code\n')
    assert cli.load_model.__name__=='load_model'
