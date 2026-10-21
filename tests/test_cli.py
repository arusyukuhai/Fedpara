import sys
import types
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file
from patchlib import cli
from patchlib.core import decode_patch

class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=True, _commit_hash='a' * 40)
        self.emb = nn.Embedding(10, 6)
        self.head = nn.Linear(6, 10, bias=False)
        self.head.weight = self.emb.weight

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        torch.manual_seed(3)
        return cls()

    def forward(self, input_ids, labels):
        out = self.head(self.emb(input_ids))
        loss = nn.functional.cross_entropy(out.reshape(-1, 10), labels.reshape(-1))
        return types.SimpleNamespace(loss=loss)

    def save_pretrained(self, folder, safe_serialization=True):
        Path(folder).mkdir(parents=True, exist_ok=True)
        # In this smoke-test the tied model is saved only once.
        save_file({'head.weight': self.head.weight.detach().cpu().contiguous()},
                  Path(folder)/'model.safetensors')


def test_full_training_and_merging_cli(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoModelForCausalLM=ToyLM))
    samples = tmp_path / 'batches'
    samples.mkdir()
    inputs = torch.tensor([[0,2,3],[1,4,5]])
    save_file({'input_ids':inputs, 'labels':inputs.clone()}, samples/'batch.safetensors')
    output = tmp_path/'trained.fpatch'
    monkeypatch.setattr(sys, 'argv', ['patchlib.cli','train','--model','fake/repo',
                         '--revision','a'*40, '--peer-id','worker1', '--rank','4',
                         '--kind','loha','--batches-dir',str(samples),'--steps','3',
                         '--log-steps','1','--output',str(output)])
    cli.main()
    assert output.exists()
    meta, tensors = decode_patch(output.read_bytes())
    assert meta['seq']==3 and meta['kind']=='loha'
    assert meta['revision']=='a'*40
    assert len(meta['targets'])==1 and len(meta['targets'][0]['aliases']) == 2
    out_dir = tmp_path / 'merged'
    monkeypatch.setattr(sys, 'argv', ['patchlib.cli','merge','--model','fake/repo',
                         '--revision','a'*40,'--rank','4','--kind','loha',
                         '--local-patch',str(output),'--output-dir',str(out_dir)])
    cli.main()
    assert (out_dir / 'model.safetensors').exists()
