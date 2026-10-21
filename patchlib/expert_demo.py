"""An actual aiortc DataChannel expert forward/backward demonstration.

Two or more terminals join with the same room ID during a short enrollment
window. Placement freezes for the demo. This is NOT FedAvg/LoHA sync.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import torch
from torch import nn
from .roomid import auto_identity,RoomDiscovery
from .expert_parallel import discover_moe
from .expert_plan import plan_experts
from .expert_runtime import ExpertRuntime


class ToyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(8,4)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))
                                    for _ in range(4)])
    def forward(self,x):
        ids=self.gate(x).topk(1,dim=-1).indices.flatten()
        return torch.stack([self.experts[int(ids[i])](x[i]) for i in range(len(ids))])


async def run_expert_demo(args):
    room=args.room_id
    peer,*_=auto_identity(Path(str(args.journal)+'.identity.pem'))
    disco=RoomDiscovery(args.ws,room,peer,token=args.token)
    stop=asyncio.Event()
    task=asyncio.create_task(disco.run(stop))
    try:
        try:
            await asyncio.wait_for(disco.updated.wait(),20)
        except asyncio.TimeoutError as e:
            raise ConnectionError('could not discover room participants') from e
        await asyncio.sleep(args.enroll_seconds)
        participants=sorted(disco.members|{peer})
        print('pinned expert demo peers:',participants,flush=True)
        torch.manual_seed(4107)
        model=ToyExperts()
        if args.quant4:
            from .expert_memory import quantize_expert_linears
            for expert in model.experts:
                quantize_expert_linears(expert,rank=args.rank,kind=args.kind)
        layers=discover_moe(model)
        plan=plan_experts(room,layers,{p:1 for p in participants},replicas=1,
                          generation='toy-demo-v1')
        async with ExpertRuntime(model,room,plan,ws=args.ws,journal=args.journal,token=args.token) as runtime:
            await runtime.wait_ready(timeout=120)
            def step():
                torch.manual_seed(777)
                x=torch.randn(64,8,requires_grad=True)
                loss=model(x).square().mean();loss.backward()
                return {'loss':float(loss.item()),'grad_input':bool(x.grad is not None),
                        'local_experts':len(runtime.owner.experts),
                        'remote_experts':sum(len(m) for m in plan['placements'].values())-len(runtime.owner.experts)}
            print('expert network autograd result:',json.dumps(await asyncio.to_thread(step)),flush=True)
    finally:
        stop.set();task.cancel()
        await asyncio.gather(task,return_exceptions=True)
