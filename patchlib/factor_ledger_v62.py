"""Bounded, exact (up to ordinary floating arithmetic) factor-separated FedSum base.

Unmodified safetensors stays immutable.  Authenticated before/after factor
pairs are held as a *bounded* virtual base rather than written as dense COW
pages.  The signed journal, not this RAM-only ledger, is authoritative after
restarts.  Reaching the cap compacts to a private dense scratch tensor once;
subsequent writes use the existing v60 implementation.  It is not a magic
constant-storage representation of indefinitely many independent updates.
"""
from __future__ import annotations
import math
import threading
from contextlib import contextmanager
import torch
from .core import PatchError
from .lazy_mutable_weights_v60 import LazyMutableWeights, LazyWeightError
from .livefed import _delta_block


def _factor_tile(spec, parts, rs, cs):
    if spec.dense:
        return parts[0][rs,cs]
    a,b=parts[0],parts[1]
    if spec.kind=='lora':
        return (a[rs] @ b[cs].T)*(spec.alpha/spec.rank)
    c,d=parts[2],parts[3]
    return ((a[rs] @ b[cs].T)*(c[rs] @ d[cs].T))*(spec.alpha/spec.rank)


class FactorLedgerSource:
    """CPU tile source consumed by existing disk Linear/Conv1D/Embedding kernels."""
    _fedpara_is_paged=True  # tells the disk kernel to tile, never cache full weight
    def __init__(self,base,spec,*,max_entries=32):
        if base.ndim!=2 or base.device.type!='cpu' or not base.is_contiguous():
            raise LazyWeightError('factor ledger requires contiguous CPU 2-D base')
        if spec.dense or spec.kind not in ('lora','loha'):
            raise LazyWeightError('factor ledger requires actual low-rank factors')
        self.base=base.detach()
        self.spec=spec
        self.shape=tuple(base.shape)
        self.dtype=base.dtype
        self.device=torch.device('cpu')
        self.max_entries=max_entries
        self.entries=[]
        self.lock=threading.RLock()
        self.poisoned=False

    def _healthy(self):
        if self.poisoned:raise LazyWeightError('factor ledger poisoned; signed journal replay required')

    def tile(self,rs,cs):
        if not isinstance(rs,slice) or not isinstance(cs,slice):
            raise TypeError('factor ledger requires bounded slice tiles')
        lo,hi,s1=rs.indices(self.shape[0]);cl,ch,s2=cs.indices(self.shape[1])
        if s1!=1 or s2!=1:raise ValueError('factor ledger requires unit stride')
        with self.lock,torch.no_grad():
            self._healthy()
            base=self.base[lo:hi,cl:ch].clone()
            for now,old in self.entries:
                # Float32 accumulation matches continual FedSum's update order.
                diff=_factor_tile(self.spec,now,slice(lo,hi),slice(cl,ch))-\
                     _factor_tile(self.spec,old,slice(lo,hi),slice(cl,ch))
                base=(base.float()+diff).to(self.dtype)
            return base

    def __getitem__(self,index):
        if not isinstance(index,tuple) or len(index)!=2:
            raise TypeError('factor ledger requires two-dimensional slices')
        return self.tile(*index)

    @property
    def T(self):return _TransposeLedger(self)

    def gather_rows(self,ids):
        cpu=ids.to(device='cpu',dtype=torch.long)
        if cpu.numel() and ((cpu<0).any() or (cpu>=self.shape[0]).any()):
            raise IndexError('out of range embedding ID')
        uniq,inv=torch.unique(cpu.reshape(-1),sorted=True,return_inverse=True)
        result=torch.empty((len(uniq),self.shape[1]),dtype=self.dtype)
        for i,k in enumerate(uniq.tolist()):
            result[i]=self.tile(slice(k,k+1),slice(0,self.shape[1]))[0]
        return result[inv].view(*ids.shape,self.shape[1])

    def rows(self,lo,hi):return self.tile(slice(lo,hi),slice(0,self.shape[1]))

    def snapshot(self):
        """Pin the current immutable factor tuples for a concurrent exporter."""
        with self.lock:
            self._healthy()
            copy=FactorLedgerSource.__new__(FactorLedgerSource)
            copy.base=self.base
            copy.spec=self.spec
            copy.shape=self.shape
            copy.dtype=self.dtype
            copy.device=self.device
            copy.max_entries=self.max_entries
            copy.entries=list(self.entries)  # append never mutates existing entries
            copy.lock=threading.RLock()
            copy.poisoned=False
            return copy

    def append(self,now,old):
        with self.lock:
            self._healthy()
            if len(self.entries)>=self.max_entries:
                raise LazyWeightError('factor ledger full; compaction required')
            self.entries.append((tuple(t.detach().to('cpu',dtype=torch.float32).contiguous().clone() for t in now),
                                 tuple(t.detach().to('cpu',dtype=torch.float32).contiguous().clone() for t in old)))

    def stats(self):
        return {'entries':len(self.entries),
                'factor_bytes':sum(t.numel()*t.element_size() for pair in self.entries for p in pair for t in p)}


class _TransposeLedger:
    _fedpara_is_paged=True
    def __init__(self,source):
        self.source=source;self.shape=source.shape[::-1];self.dtype=source.dtype
    def __getitem__(self,index):return self.source[index[1],index[0]].T.contiguous()


class FactorLedgerManager:
    _fedpara_is_factor_ledger=True
    def __init__(self,model,adapter,*,scratch_dir=None,max_entries=32,limit_gib=0):
        if type(max_entries) is not int or not 1<=max_entries<=256:
            raise ValueError('factor-ledger max_entries must be 1..256')
        self.model=model;self.adapter=adapter
        self.fallback=LazyMutableWeights(model,scratch_dir=scratch_dir,limit_gib=limit_gib)
        self.directory=self.fallback.directory
        self.max_entries=max_entries
        self.sources={}
        self.compacted=set()
        self.poisoned=False
        self.limit_bytes=int(limit_gib*1024**3)
        # This limit covers factors; fallback has its own independent quota.
        self.factor_bytes=0
        self._export_context=threading.local()

    def _healthy(self):
        if self.poisoned or any(s.poisoned for s in self.sources.values()):
            raise LazyWeightError('factor ledger poisoned; restart from authenticated journal')

    def _original(self,group):
        path,_,attr=group.aliases[0].rpartition('.')
        mod=self.model.get_submodule(path) if path else self.model
        return mod.parametrizations[attr].original

    def _members(self,group):
        modules=[]
        for alias in group.aliases:
            path,_,attr=alias.rpartition('.')
            mod=self.model.get_submodule(path) if path else self.model
            if mod.parametrizations[attr].original is not self._original(group):
                raise LazyWeightError('tied group mismatch')
            modules.append(mod)
        return modules

    def materialize_group(self,group):
        """Called by legacy signed gate; do NOT force a dense copy here."""
        self._healthy()
        key=group.canonical
        if key in self.compacted or key in self.sources:return False
        base=self._original(group)
        members=self._members(group)
        spec=group.overlay.local
        if (base.ndim!=2 or base.device.type!='cpu' or not base.is_contiguous()
            or spec.dense or any(not hasattr(m,'_fedpara_disk_source') for m in members)):
            self.fallback.materialize_group(group)
            self.compacted.add(key)
            return False
        source=FactorLedgerSource(base,spec,max_entries=self.max_entries)
        self.sources[key]=source
        for mod in members:
            cache=getattr(mod,'_fedpara_disk_cache',None)
            if cache is not None:cache.invalidate()
            mod._fedpara_disk_source=source
            mod._fedpara_disk_path='<factor-separated-journal-overlay>'
        return True

    def _append(self,group,now,old):
        key=group.canonical
        source=self.sources.get(key)
        if source is None:
            raise LazyWeightError('non-ledger groups require legacy signed apply')
        if len(source.entries)>=self.max_entries:
            self._compact(group)
            return False
        size=sum(t.numel()*4 for parts in (now,old) for t in parts)
        if self.limit_bytes and self.factor_bytes+size+self.fallback.used_bytes>self.limit_bytes:
            self._compact(group)
            return False
        source.append(now,old)
        self.factor_bytes+=size
        return True

    def _compact(self,group):
        """One-time expensive exact materialization; no lossily truncated SVD."""
        key=group.canonical
        source=self.sources[key]
        original=self._original(group)
        remaining=self.factor_bytes-source.stats()['factor_bytes']
        dense_bytes=original.numel()*original.element_size()
        if self.limit_bytes and self.fallback.used_bytes+remaining+dense_bytes>self.limit_bytes:
            raise LazyWeightError('exact ledger compaction exceeds combined capacity quota')
        # The source references original genesis; v60 replaces original.data
        # and switches disk kernels, so retain it while reconstructing.
        self.fallback.materialize_group(group)
        with torch.no_grad():
            flat=original.data
            width=original.shape[1]
            row_step=max(1,min(64,65536//max(1,width)))
            for lo in range(0,original.shape[0],row_step):
                end=min(original.shape[0],lo+row_step)
                col_step=max(1,65536//(end-lo))
                for col in range(0,width,col_step):
                    ce=min(width,col+col_step)
                    flat[lo:end,col:ce].copy_(
                        source[lo:end,col:ce].to(flat.dtype))
        self.factor_bytes-=source.stats()['factor_bytes']
        self.sources.pop(key)
        self.compacted.add(key)

    @staticmethod
    def _tiles(shape,budget):
        """Yield bounded 2-D tiles, including rows wider than the byte budget."""
        if len(shape)!=2:
            width=math.prod(shape[1:]) if len(shape)>1 else 1
            for lo in range(0,shape[0] if shape else 1,max(1,budget//width)):
                hi=min(shape[0] if shape else 1,lo+max(1,budget//width))
                yield lo,hi,None,None
            return
        height,width=shape
        rows=max(1,budget//max(width,1))
        for lo in range(0,height,rows):
            hi=min(height,lo+rows)
            cols=max(1,budget//(hi-lo))
            for cl in range(0,width,cols):
                yield lo,hi,cl,min(width,cl+cols)

    @staticmethod
    def _piece(spec,prepared,gi,lo,hi,cl,ce):
        dense=prepared.deltas[gi]
        if dense is not None:
            value=dense[lo:hi] if spec.shape else dense
            return value if cl is None else value[:,cl:ce]
        a,b=prepared.now_parts[gi],prepared.old_parts[gi]
        if cl is not None:
            rs,cs=slice(lo,hi),slice(cl,ce)
            return _factor_tile(spec,a,rs,cs)-_factor_tile(spec,b,rs,cs)
        return _delta_block(spec,a,lo,hi)-_delta_block(spec,b,lo,hi)

    def apply_verified(self,engine,updates):
        """Consume independently authenticated/validated event factors only.

        All current candidates are checked before any append. Commit failures
        poison the trainer for clean signed-journal replay. The record journal
        is durable before the engine invokes this method.
        """
        self._healthy()
        adapter=self.adapter
        for group in adapter.groups:
            self.materialize_group(group)
        try:
            with torch.no_grad():
                for gi,group in enumerate(adapter.groups):
                    orig=self._original(group)
                    source=self.sources.get(group.canonical)
                    spec=group.overlay.local
                    for lo,hi,cl,ce in self._tiles(group.shape,engine.block_elements):
                        if source is not None:
                            base=(source.rows(lo,hi) if cl is None else
                                  source[lo:hi,cl:ce])
                        else:
                            base=orig[lo:hi] if cl is None and group.shape else (orig if cl is None else orig[lo:hi,cl:ce])
                        candidate=base.float()
                        for _,prepared in updates:
                            value=self._piece(spec,prepared,gi,lo,hi,cl,ce)
                            candidate=candidate+value.to(candidate.device)
                        if not bool(torch.isfinite(candidate).all().item()):
                            raise PatchError('nonfinite factor-separated effective base')
            for gi,group in enumerate(adapter.groups):
                key=group.canonical
                if key in self.sources and len(self.sources[key].entries)+len(updates)>self.max_entries:
                    self._compact(group)
                orig=self._original(group)
                spec=group.overlay.local
                for _,prepared in updates:
                    now,old=prepared.now_parts[gi],prepared.old_parts[gi]
                    if key in self.sources and self._append(group,now,old):
                        continue
                    for lo,hi,cl,ce in self._tiles(group.shape,engine.block_elements):
                        delta=self._piece(spec,prepared,gi,lo,hi,cl,ce)
                        dest=(orig[lo:hi] if group.shape else orig) if cl is None else orig[lo:hi,cl:ce]
                        dest.copy_((dest.float()+delta.to(dest.device)).to(dest.dtype))
        except BaseException:
            self.poisoned=True
            raise

    def frozen_sources(self):
        self._healthy()
        return {alias:source.snapshot() for group in self.adapter.groups
                for alias in group.aliases
                if (source:=self.sources.get(group.canonical)) is not None}

    @contextmanager
    def export_context(self):
        # Only trusted, bounded sharded serialization may see stale original
        # tensors; the exporter explicitly supplies our virtual base instead.
        previous=getattr(self._export_context,'active',False)
        self._export_context.active=True
        try:yield
        finally:self._export_context.active=previous

    def export_allowed(self):return getattr(self._export_context,'active',False)

    def materialize_all(self,adapter):
        self._healthy()
        try:
            for group in adapter.groups:
                if group.canonical in self.sources:self._compact(group)
                elif group.canonical not in self.compacted:
                    self.fallback.materialize_group(group)
                    self.compacted.add(group.canonical)
        except BaseException:
            self.poisoned=True
            raise

    def stats(self):
        return {'ledger_groups':len(self.sources),'compacted_groups':len(self.compacted),
                'factor_bytes':self.factor_bytes,'fallback_bytes':self.fallback.used_bytes,
                'scratch':str(self.directory)}


def attach_factor_ledger(model,adapter,*,scratch_dir=None,max_entries=32,limit_gib=0):
    if getattr(model,'_fedpara_streaming_base_info',{}).get('mode')!='read-only-mmap':
        raise LazyWeightError('factor ledger requires immutable safetensors genesis')
    manager=FactorLedgerManager(model,adapter,scratch_dir=scratch_dir,
                                max_entries=max_entries,limit_gib=limit_gib)
    object.__setattr__(model,'_fedpara_lazy_mutable_weights',manager)
    def forbid_stale_state(module,prefix,keep_vars):
        if manager.sources and not manager.export_allowed():
            raise LazyWeightError('virtual low-rank ledger: use sharded export, not model.state_dict')
    model.register_state_dict_pre_hook(forbid_stale_state)
    return manager
