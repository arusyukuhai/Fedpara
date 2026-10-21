"""Bounded DataChannel request/reply bridge for ExpertOwner, integrated with ElasticNode.

The owner receives activations and backprop gradients, NOT Python code or pickle.
The ElasticNode's signed hello authenticates the key-derived peer before dispatch.
Run training on a worker thread while the asyncio loop pumps WebRTC frames.
"""
from __future__ import annotations
import asyncio
import base64
import json
import concurrent.futures
import threading
from .expert_parallel import ExpertRPCError,MAX_RPC_BYTES

PREFIX='FP-EXPERT1:'
MAX_WIRE=2*MAX_RPC_BYTES+8192


def encode_rpc(message):
    copy=dict(message)
    if 'tensor' in copy:
        blob=copy['tensor']
        if not isinstance(blob,bytes) or len(blob)>MAX_RPC_BYTES:raise ExpertRPCError('invalid tensor payload')
        copy['tensor']=base64.b64encode(blob).decode('ascii')
    wire=PREFIX+json.dumps(copy,sort_keys=True,separators=(',',':'))
    if len(wire)>MAX_WIRE:raise ExpertRPCError('expert RPC too large')
    return wire


def decode_rpc(wire):
    if not isinstance(wire,str) or not wire.startswith(PREFIX) or len(wire)>MAX_WIRE:
        raise ExpertRPCError('malformed expert envelope')
    obj=json.loads(wire[len(PREFIX):])
    if not isinstance(obj,dict) or obj.get('version')!=1:raise ExpertRPCError('bad RPC version')
    if 'tensor' in obj:
        if not isinstance(obj['tensor'],str) or len(obj['tensor'])>2*MAX_RPC_BYTES:
            raise ExpertRPCError('oversized RPC base64')
        obj['tensor']=base64.b64decode(obj['tensor'],validate=True)
    return obj


class ExpertDataChannelBridge:
    """A per-neighbor RPC endpoint. All transport callbacks run on `loop`.

    request() is sync for torch.autograd: only call from another thread.
    """
    def __init__(self,loop:asyncio.AbstractEventLoop,owner=None,timeout:float=30.0):
        if not 0<timeout<=120:raise ValueError('timeout invalid')
        self.loop,self.owner,self.timeout=loop,owner,timeout
        self.node=None
        self._pending:dict[str,asyncio.Future]={}
        self._max_pending=64
    def attach(self,elastic_node):
        self.node=elastic_node
        elastic_node.expert_bridge=self
        return self
    async def _send(self,body):
        if not self.node or not self.node.connected.is_set() or not self.node.verifier:
            raise ExpertRPCError('remote peer not authenticated/connected')
        channel=self.node._channel
        if not channel or channel.readyState!='open' or channel.bufferedAmount>4*1024*1024:
            raise ExpertRPCError('DataChannel backpressure/disconnected')
        channel.send(encode_rpc(body))
    async def _request(self,req):
        key=req.get('id')
        if not isinstance(key,str) or key in self._pending or len(self._pending)>=self._max_pending:
            raise ExpertRPCError('too many/duplicate pending RPC requests')
        fut=self.loop.create_future();self._pending[key]=fut
        try:
            await self._send(req)
            return await asyncio.wait_for(fut,self.timeout)
        finally:self._pending.pop(key,None)
    def request(self,req):
        if threading.get_ident()==getattr(self.loop,'_thread_id',None):
            raise ExpertRPCError('RPC from event-loop thread would deadlock')
        task=asyncio.run_coroutine_threadsafe(self._request(req),self.loop)
        try:return task.result(timeout=self.timeout+2)
        except concurrent.futures.TimeoutError:
            task.cancel();raise ExpertRPCError('expert RPC deadline exceeded')
    def accept(self,wire):
        obj=decode_rpc(wire)
        if obj.get('op') in ('forward','backward'):
            if self.owner is None:raise ExpertRPCError('not an expert owner')
            asyncio.create_task(self._serve(obj))
        elif obj.get('op')=='reply':
            key=obj.get('id');fut=self._pending.get(key)
            if fut is not None and not fut.done():
                if obj.get('error'):fut.set_exception(ExpertRPCError(str(obj['error'])[:200]))
                else:fut.set_result(obj)
        else:raise ExpertRPCError('invalid expert RPC operation')
    async def _serve(self,obj):
        try:
            response=await asyncio.to_thread(self.owner.handle,self.node.remote,obj)
            response['op']='reply'
        except Exception as exc:
            response={'version':1,'id':obj.get('id'),'op':'reply','error':type(exc).__name__}
        try:await self._send(response)
        except (ExpertRPCError,ConnectionError):pass
