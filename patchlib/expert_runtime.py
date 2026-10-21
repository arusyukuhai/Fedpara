"""P2P expert execution session using Mummy WebRTC signaling, aiortc and signed peer hello.

Use from an ASYNC host process, run model train steps in asyncio.to_thread.
The plan is a fixed, versioned placement; peer membership changes require
verified resharding outside this class. No central model/activation relay.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import torch
from torch import nn
from .roomid import validate_room, auto_identity, pair_room
from .expert_plan import plan_digest
from .expert_parallel import (discover_moe, ExpertOwner, RemoteExpert,
                              ExpertRPCError, install_remote_experts)
from .expert_channel import ExpertDataChannelBridge
from .elasticnet import ElasticNode

LOG=logging.getLogger(__name__)


class ExpertRuntime:
    def __init__(self,model:nn.Module,room_id:str,plan:dict,*,ws='ws://127.0.0.1:8080/ws',
                 journal:Path=Path('expert-session'),token='',timeout=30.0):
        self.model,self.room= model,validate_room(room_id)
        self.plan=plan;self.ws=ws;self.token=token;self.timeout=timeout
        if not isinstance(plan,dict) or plan.get('version')!=1:
            raise ValueError('expert runtime requires placement plan v1')
        if hashlib.sha256(room_id.encode()).hexdigest()!=plan.get('room_sha256'):
            raise ValueError('expert plan belongs to another room')
        if not ws.startswith(('wss://','ws://')):raise ValueError('invalid signaling address')
        if ws.startswith('ws://') and ws.split('/')[2].split(':')[0] not in ('localhost','127.0.0.1','[::1]'):
            raise PermissionError('internet signaling must use WSS')
        self.peer,self.signer,self.pub,self.verifier=auto_identity(Path(str(journal)+'.identity.pem'))
        if self.peer not in plan['capacity']:
            raise ValueError('this peer is not a member of the pinned placement epoch; start a new generation')
        self.generation=plan_digest(plan)
        self.nodes={};self.bridges={};self.tasks={};self._stop=False
        self.owner=None

    async def start(self):
        """Install remote autograd proxies and connect to every primary owner.

        Direct-owner links can exceed the normal 4-neighbor gossip fanout;
        activation RPC cannot be routed via intermediate gossip peers.
        """
        layer_desc={x['experts']:x for x in discover_moe(self.model) if x.get('supported')}
        offered={}
        targets={}
        for path,mapping in self.plan['placements'].items():
            if path not in layer_desc:raise ValueError('model lacks supported MoE router '+path)
            experts=self.model.get_submodule(path)
            if len(experts)!=len(mapping):raise ValueError('expert count differs from epoch plan')
            target={}
            for number,owners in mapping.items():
                i=int(number); primary=owners[0]
                key=str(i) if isinstance(experts,nn.ModuleList) else 'expert_'+str(i)
                expert_id=f'{path}.{key}'
                if primary==self.peer:
                    offered[expert_id]=experts[i] if isinstance(experts,nn.ModuleList) else experts[key]
                else:target[i]=primary
            targets[path]=target
        self.owner=ExpertOwner(offered,self.generation,ttl=self.timeout*4)
        # Every owner must accept incoming execution RPCs from all peers that
        # could route to its experts, not only establish its outgoing links.
        others=set()
        for mapping in self.plan['placements'].values():
            for owners in mapping.values():
                primary=owners[0]
                if primary==self.peer:
                    others.update(p for p in self.plan['capacity'] if p!=self.peer)
                else:
                    others.add(primary)
        if len(others)>31:raise ValueError('too many owner links')
        # Secondary replicas may have a copy, but there is no automatic failover
        # until the replacement proves the exact checkpoint and optimizer epoch.
        loop=asyncio.get_running_loop()
        for peer in others:
            if peer==self.peer:raise ValueError('self owner mapping inconsistent')
            bridge=ExpertDataChannelBridge(loop,owner=self.owner,timeout=self.timeout)
            self.bridges[peer]=bridge
            await self._connect(peer,bridge)
        for path,mapping in targets.items():
            install_remote_experts(self.model,path,
                {i:self.bridges[peer] for i,peer in mapping.items()},self.generation)
        return self

    async def _connect(self,peer,bridge):
        node=ElasticNode(self.ws,pair_room(self.room,self.peer,peer),self.token,
            origin=self.peer,remote=peer,genesis=self.generation,
            signer=self.signer,public_key=self.pub,unsafe=False)
        bridge.attach(node)
        self.nodes[peer]=node
        self.tasks[peer]=asyncio.create_task(node.run(),name='expert-peer-'+peer)

    async def wait_ready(self,timeout:float=60.0):
        end=asyncio.get_running_loop().time()+timeout
        while asyncio.get_running_loop().time()<end:
            if all(node.connected.is_set() and node.verifier is not None
                   for node in self.nodes.values()):return True
            for task in self.tasks.values():
                if task.done() and task.exception() is not None:raise task.exception()
            await asyncio.sleep(.1)
        raise TimeoutError('expert owner links were not authenticated before deadline')

    async def close(self):
        self._stop=True
        for task in self.tasks.values():task.cancel()
        await asyncio.gather(*self.tasks.values(),return_exceptions=True)
        self.tasks.clear()

    async def __aenter__(self):return await self.start()
    async def __aexit__(self,*exc):await self.close()

    def primary_expert_parameters(self):
        """Local owner expert weights; caller controls optimizer lifecycle.

        DO NOT step while asynchronous backwards are executing on this owner.
        Only restart/migrate on a quiescent checkpoint boundary.
        """
        if self.owner is None:raise RuntimeError('runtime not started')
        seen=set()
        for module in self.owner.experts.values():
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p));yield p
