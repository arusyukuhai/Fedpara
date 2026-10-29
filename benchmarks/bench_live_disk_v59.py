"""Reproducible local CPU AdamW microbenchmark; NOT 300B or WebRTC.

PYTHONPATH=. python benchmarks/bench_live_disk_v59.py --output BENCHMARK_V59.json
"""
from __future__ import annotations
import argparse, json, tempfile, time, statistics
from pathlib import Path
import torch
from patchlib.disk_adamw import DiskAdamW


def run(d, n, steps):
    torch.manual_seed(9)
    p0=[torch.nn.Parameter(torch.randn(d,d)) for _ in range(n)]
    p1=[torch.nn.Parameter(x.detach().clone()) for x in p0]
    with tempfile.TemporaryDirectory() as root:
        a=torch.optim.AdamW(p0,lr=1e-3,weight_decay=.001)
        b=DiskAdamW([(f'w{i}',p) for i,p in enumerate(p1)],root,lr=1e-3,weight_decay=.001)
        grads=[torch.randn(d,d) for _ in range(n)]
        def bench(opt,params):
            timings=[]
            for _ in range(steps):
                for p,g in zip(params,grads):p.grad=g
                start=time.perf_counter()
                opt.step()
                timings.append((time.perf_counter()-start)*1000)
            return timings
        # Baseline first, disk second, both have identical values and gradients.
        timings_a=bench(a,p0)
        timings_b=bench(b,p1)
        b.flush()
        err=max((x-y).abs().max().item() for x,y in zip(p0,p1))
        resident_adam=sum(v.numel()*v.element_size() for state in a.state.values() for v in state.values() if isinstance(v,torch.Tensor))
        resident_disk=sum(v.numel()*v.element_size() for state in b.state.values() for v in state.values() if isinstance(v,torch.Tensor))
        disk_bytes=sum((Path(root)/f).stat().st_size for f in __import__('os').listdir(root) if f.endswith('.f32'))
    return {
        'task':'synthetic dense CPU adapter-like tensors; no model/P2P/checkpoint',
        'parameter_count':n*d*d,'optimizer_steps':steps,
        'adamw_step_median_ms':statistics.median(timings_a),
        'disk_adamw_step_median_ms':statistics.median(timings_b),
        'max_abs_parameter_difference':err,
        'pytorch_adamw_moment_ram_bytes':resident_adam,
        'disk_adamw_moment_ram_bytes':resident_disk,
        'disk_adamw_moment_file_bytes':disk_bytes,
        'note':'disk RAM excludes OS mmap page cache; disk uses staged whole-parameter moments; no SSD performance guarantee',
    }

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('BENCHMARK_V59.json'))
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--matrices',type=int,default=4)
    parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=run(args.dim,args.matrices,args.steps)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
