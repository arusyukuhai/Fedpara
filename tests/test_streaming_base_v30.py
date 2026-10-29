from pathlib import Path
import json
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from patchlib.streaming_base import materialize_meta_from_safetensors, StreamingModelError

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(16,8)
        self.linear=nn.Linear(8,16,bias=True)
        self.linear.weight=self.embed.weight
        self.norm=nn.LayerNorm(8)
    def forward(self,x):
        return self.norm(self.embed(x)).sum()

def save_toy(tmp_path):
    base=Toy()
    save_file({'embed.weight':base.embed.weight.detach().contiguous(),
                'linear.bias':base.linear.bias.detach().contiguous(),
                'norm.weight':base.norm.weight.detach().contiguous(),
                'norm.bias':base.norm.bias.detach().contiguous()},str(tmp_path/'model.safetensors'))
    return base

@pytest.mark.parametrize('writable',[False,True])
def test_meta_load_and_tied_writes(tmp_path,writable):
    base=save_toy(tmp_path)
    with torch.device('meta'):
        mdl=Toy()
    materialize_meta_from_safetensors(mdl,tmp_path,writable=writable,chunk_mib=1)
    assert mdl.embed.weight is mdl.linear.weight
    assert all(p.device.type=='cpu' for p in mdl.parameters())
    ids=torch.tensor([[1,2]])
    assert torch.allclose(mdl(ids),base(ids))
    assert mdl._fedpara_streaming_base_info['tensors']==4
    if writable:
        mdl.embed.weight.data[1,0] += 13
        assert mdl.embed.weight.data[1,0]!=base.embed.weight.data[1,0]
    else:
        assert mdl.embed.weight.data_ptr()!=base.embed.weight.data_ptr()

def test_shard_index_and_bad_index(tmp_path):
    base=save_toy(tmp_path)
    p=tmp_path/'model.safetensors'
    p.rename(tmp_path/'model-00001-of-00001.safetensors')
    contents=['embed.weight','linear.bias','norm.weight','norm.bias']
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {x:'model-00001-of-00001.safetensors' for x in contents}}))
    with torch.device('meta'):m=Toy()
    materialize_meta_from_safetensors(m,tmp_path)
    assert torch.allclose(m.embed.weight,base.embed.weight)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'embed.weight':'../escape.safetensors'}}))
    with pytest.raises(Exception):materialize_meta_from_safetensors(Toy().to('meta'),tmp_path)

def test_missing_parameter_fails_without_model_mutation(tmp_path):
    save_file({'embed.weight':torch.rand(16,8)},str(tmp_path/'model.safetensors'))
    with torch.device('meta'):m=Toy()
    with pytest.raises(StreamingModelError,match='missing parameter'):
        materialize_meta_from_safetensors(m,tmp_path)
    assert m.embed.weight.device.type=='meta'

def test_disk_embedding_and_tiled_linear_backward(tmp_path):
    from patchlib.core import UniversalAdapter
    from patchlib.layer_offload import install_disk_linear_offload
    torch.manual_seed(3)
    base=save_toy(tmp_path)
    with torch.device('meta'):streamed=Toy()
    materialize_meta_from_safetensors(streamed,tmp_path)
    normal=Toy()
    normal.load_state_dict(base.state_dict())
    a=UniversalAdapter(streamed,'tiny','rev',kind='lora',rank=2,target_scope='auto',max_peers=2)
    b=UniversalAdapter(normal,'tiny','rev',kind='lora',rank=2,target_scope='auto',max_peers=2)
    # Factors seeded independently during model construction; equalize for comparison.
    for ga,gb in zip(a.groups,b.groups):
        for pa,pb in zip(ga.overlay.local.parts,gb.overlay.local.parts):
            pb.data.copy_(pa.data)
            if pa.ndim>1:
                pa.data.uniform_(-.07,.07)
                pb.data.copy_(pa.data)
    installed=install_disk_linear_offload(streamed,a,tmp_path/'pages',include_embeddings=True,tile_mib=1)
    assert installed['embeddings']==1
    assert installed['layers']==1
    assert not list((tmp_path/'pages').glob('*.weight')) # no safetensors duplicate
    assert streamed.embed.weight is not streamed.linear.weight # parametrizations are wrappers
    ids=torch.tensor([[2,4,5],[0,7,8]])
    assert torch.allclose(streamed(ids),normal(ids),atol=1e-5)
    streamed(ids).backward();normal(ids).backward()
    for ga,gb in zip(a.groups,b.groups):
        for pa,pb in zip(ga.overlay.local.parts,gb.overlay.local.parts):
            assert torch.allclose(pa.grad,pb.grad,atol=1e-5) if pa.grad is not None else pb.grad is None


def test_multipass_tiled_matmul_gradients(tmp_path):
    from patchlib.layer_offload import _FrozenDiskLinear,DiskWeightCache
    weight=torch.randn(23,29)
    bias=torch.randn(23,requires_grad=True)
    f=tmp_path/'raw'
    f.write_bytes(weight.contiguous().view(torch.uint8).numpy().tobytes())
    x=torch.randn(5,2,29,requires_grad=True)
    cache=DiskWeightCache('cpu',tile_mib=1)
    cache.tile_bytes=52  # force 2-D tiling with multiple in/out steps
    y=_FrozenDiskLinear.apply(x,bias,str(f),tuple(weight.shape),weight.dtype,cache)
    assert torch.allclose(y,torch.nn.functional.linear(x,weight,bias),atol=1e-5)
    y.square().mean().backward()
    expected_x=x.detach().clone().requires_grad_(True)
    expected_bias=bias.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(expected_x,weight,expected_bias).square().mean().backward()
    assert torch.allclose(x.grad,expected_x.grad,atol=1e-5)
    assert torch.allclose(bias.grad,expected_bias.grad,atol=1e-5)

def test_gui_streaming_new_room_and_existing_genesis_guard(tmp_path,monkeypatch):
    from patchlib.roomid import create_room_id
    from patchlib.webview_controller import WebviewController
    ctl=WebviewController(tmp_path)
    room=create_room_id()
    ctl.save(dict(room=room,model='org/model',revision='abc',task='causal-lm',
        initialization='pretrained',sources=[dict(repo='org/data',revision='main',
        config=None,split='train',columns={'text':'text'},weight=1)]))
    commands=[]
    monkeypatch.setattr(ctl.train,'start',lambda cmd,log:commands.append(cmd) or 100)
    r=ctl.start(dict(room=room,base='',ws='ws://127.0.0.1:8080/ws',
          kind='loha',rank=16,streaming_base=True,disk_cache_fraction=.8))
    cmd=commands[0]
    assert '--streaming-base' in cmd
    assert '--state-base' not in cmd
    assert '--auto-state' not in cmd
    assert cmd[cmd.index('--disk-cache-fraction')+1]=='0.8'
    assert r['auto_state'] is False
    with pytest.raises(ValueError):
        ctl.start(dict(room=room,base='/tmp/old.fstate',
                      ws='ws://127.0.0.1:8080/ws',streaming_base=True))


def test_genesis_hash_is_deterministic_for_identical_safetensors(tmp_path):
    save_toy(tmp_path)
    with torch.device('meta'):m=Toy()
    materialize_meta_from_safetensors(m,tmp_path,hash_genesis=True)
    key=m._fedpara_streaming_base_info['genesis_digest']
    with torch.device('meta'):other=Toy()
    materialize_meta_from_safetensors(other,tmp_path,hash_genesis=True)
    assert other._fedpara_streaming_base_info['genesis_digest']==key
    assert len(key)==64

def test_scalar_buffer_roundtrip(tmp_path):
    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('running',torch.zeros(()))
            self.linear=nn.Linear(4,4,bias=False)
    with torch.device('meta'):m=Scalar()
    save_file({'running':torch.tensor(9.5), 'linear.weight':torch.arange(16,dtype=torch.float32).reshape(4,4)},str(tmp_path/'model.safetensors'))
    materialize_meta_from_safetensors(m,tmp_path,writable=True)
    assert m.running.shape==() and m.running.item()==9.5
    assert torch.equal(m.linear.weight.detach(),torch.arange(16,dtype=torch.float32).reshape(4,4))


def test_dtype_mismatch_is_fail_closed(tmp_path):
    with torch.device('meta'):model=Toy()
    save_file({'embed.weight':torch.zeros(16,8,dtype=torch.float16),
               'linear.bias':torch.zeros(16),
               'norm.weight':torch.ones(8),
               'norm.bias':torch.zeros(8)},str(tmp_path/'model.safetensors'))
    with pytest.raises(StreamingModelError,match='dtype mismatch'):
        materialize_meta_from_safetensors(model,tmp_path,writable=True)
    assert model.embed.weight.device.type=='meta'
