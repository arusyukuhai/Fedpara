"""v60 local CPU boot cost; NOT a 300B / GPU / distributed benchmark.

Run: PYTHONPATH=. python benchmarks/bench_v60_lazy_boot.py --output BENCHMARK_V60.json
"""
from __future__ import annotations
import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from patchlib.streaming_base import materialize_meta_from_safetensors
from patchlib.core import UniversalAdapter
from patchlib.lazy_mutable_weights_v60 import attach_lazy_mutable


class FourLayers(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(4)])


def benchmark(width=1536):
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='kagami-v60-boot-bench-') as folder:
        root=Path(folder)
        weights=FourLayers(width)
        source=root/'model.safetensors'
        save_file({k:p.detach().contiguous() for k,p in weights.state_dict().items()},str(source))
        del weights
        gc.collect()
        with torch.device('meta'):
            eager=FourLayers(width)
        start=time.perf_counter()
        materialize_meta_from_safetensors(eager,root,writable=True,chunk_mib=8)
        eager_elapsed=time.perf_counter()-start
        old_scratch=Path(eager._fedpara_ephemeral_weights.name)
        eager_bytes=sum(f.stat().st_size for f in old_scratch.glob('*.raw'))
        del eager
        gc.collect()
        with torch.device('meta'):
            lazy=FourLayers(width)
        start=time.perf_counter()
        materialize_meta_from_safetensors(lazy,root,writable=False,chunk_mib=8)
        adapter=UniversalAdapter(lazy,'benchmark','v1',kind='lora',rank=2,target_scope='auto')
        manager=attach_lazy_mutable(lazy,adapter,scratch_dir=root/'scratch')
        lazy_elapsed=time.perf_counter()-start
        lazy_initial=manager.used_bytes
        materialization_start=time.perf_counter()
        manager.materialize_group(adapter.groups[0])
        first_update=time.perf_counter()-materialization_start
        first_update_bytes=manager.used_bytes
        return {'disclaimer':'local CPU synthetic startup only; does not cover 300B, P2P, GPU, or full optimizer',
            'width':width,'layers':4,'base_size_bytes':source.stat().st_size,
            'eager_startup_s':round(eager_elapsed,6),
            'eager_initial_mutable_scratch_bytes':eager_bytes,
            'lazy_startup_and_adapter_s':round(lazy_elapsed,6),
            'lazy_initial_mutable_scratch_bytes':lazy_initial,
            'lazy_first_tensor_materialization_s':round(first_update,6),
            'lazy_first_tensor_materialized_bytes':first_update_bytes,
            'note':'Times not apples-to-apples: lazy path includes adapter construction; first mutation cost is deferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('BENCHMARK_V60.json'))
    p.add_argument('--width',type=int,default=1536)
    args=p.parse_args()
    result=benchmark(args.width)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
