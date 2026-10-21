import pytest
import torch
from torch import nn
from patchlib.core import UniversalAdapter, PatchError, decode_patch


class Little(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 7)
        self.linear = nn.Linear(7, 13, bias=True)
        self.linear.weight = self.embedding.weight
        self.conv = nn.Conv2d(2, 4, kernel_size=(3, 3))
        self.t = nn.Parameter(torch.randn(4, 5, 6, 4))
        self.small = nn.Parameter(torch.randn(2, 9))
        self.norm = nn.LayerNorm(7)

    def forward(self, x):
        e = self.embedding(x)
        y = self.linear(e)
        return y.square().mean() + self.conv(torch.randn(1, 2, 6, 6)).square().mean() + self.t.square().mean() + self.small.square().mean() + self.norm(e).square().mean()


@pytest.mark.parametrize('kind', ['lora', 'loha'])
def test_all_tensors_training_and_exchange(kind):
    torch.manual_seed(10)
    m1 = Little()
    m2 = Little()
    m2.load_state_dict(m1.state_dict())
    a = UniversalAdapter(m1, 'model-X', 'fixedsha', kind=kind, rank=4)
    b = UniversalAdapter(m2, 'model-X', 'fixedsha', kind=kind, rank=4)
    assert len(a.groups) == 8  # tied embedding/linear.weight once
    assert a.counts()['dense_fallback'] >= 2  # 1D/bias and first dim size 2
    assert any(len(g.shape)==4 and not g.overlay.local.dense for g in a.groups)
    assert any(len(g.aliases)==2 for g in a.groups)
    loss = m1(torch.tensor([[0, 1, 2, 3]]))
    loss.backward()
    nonzero = [p.grad for p in a.trainable_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert nonzero, 'no trainable gradients'
    opt = torch.optim.AdamW(a.trainable_parameters(), lr=0.05)
    opt.step()
    wire = a.export('peer-A', 1)
    meta, tensors = decode_patch(wire)
    assert meta['kind']==kind and tensors
    assert b.import_remote(wire) == 'peer-A'
    assert len(b.remote_versions) == 1
    assert all(g.overlay.remotes for g in b.groups)
    with pytest.raises(PatchError):
        b.import_remote(wire)
    assert b.import_remote(a.export('peer-A', 2)) == 'peer-A'
    assert len(b.remote_versions) == 1
    b.remove_remote('peer-A')
    assert not b.remote_versions
    # Version mismatch and layout mismatch detected.
    c = UniversalAdapter(Little(), 'other', 'fixedsha', kind=kind, rank=4)
    with pytest.raises(PatchError):
        c.import_remote(wire)


def test_cp_order_three_exact_shape():
    torch.manual_seed(11)
    layer = nn.Module()
    layer.register_parameter('array', nn.Parameter(torch.ones(4, 5, 6)))
    a = UniversalAdapter(layer, 'm', 'r', kind='lora', rank=3)
    f = a.groups[0].overlay.local
    for param in f.parts:
        with torch.no_grad():
            param.fill_(1.)
    assert f.delta().shape == (4, 5, 6)
    assert torch.allclose(f.delta(), torch.full((4,5,6), 3.))


def test_merge_preserves_tied_weights():
    torch.manual_seed(12)
    m = Little()
    a = UniversalAdapter(m, 'model-X', 'fixedsha', rank=4,kind='lora')
    for g in a.groups:
        with torch.no_grad():
            for p in g.overlay.local.parts:
                p.fill_(0.05)
    pre = m.embedding.weight.clone().detach()
    a.merge_into_base()
    assert not hasattr(m.embedding, 'parametrizations')
    assert torch.allclose(m.embedding.weight, pre)
    assert m.embedding.weight is m.linear.weight
