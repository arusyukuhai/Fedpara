"""Experimental disk sparse-page overlay for signed, out-of-core FedSum bases.

The original CPU safetensors mapping is immutable. Each changed PAGE is held in
an epoch-private sparse file, indexed by an in-process page set. Scratch is
NEVER trusted on restart; signed EventJournal replay recreates it. This is a
bounded-I/O mechanism, not a compression guarantee for dense LoHA changes.

All writes must arrive through ContinualElasticEngine's signed update gate.
The overlay MUST be materialized before legacy snapshot/whole-model export.
"""
from __future__ import annotations
import hashlib
import math
from collections import OrderedDict
import os
import threading
from pathlib import Path

import torch

from .lazy_mutable_weights_v60 import LazyMutableWeights, LazyWeightError


class SparsePageStore:
    """Writable sparse-page virtual 2-D tensor with immutable mapped backing."""
    _fedpara_is_paged=True

    def __init__(self, base, dest:Path, *, page_bytes=65536, limit_callback=None, on_fault=None):
        if base.ndim!=2 or base.device.type!='cpu' or not base.is_contiguous():
            raise LazyWeightError('page COW supports contiguous CPU matrix only')
        if not base.is_floating_point() or base.element_size() not in (2,4,8):
            raise LazyWeightError('page COW requires floating numeric matrix')
        if page_bytes < 4096 or page_bytes > 4*1024*1024 or page_bytes%base.element_size():
            raise ValueError('invalid COW page size')
        self.base=base.detach()
        self.shape=tuple(base.shape)
        self.dtype=base.dtype
        self.device=torch.device('cpu')
        self.page_bytes=page_bytes
        self.page_elems=page_bytes//base.element_size()
        self.n=base.numel()
        self.path=Path(dest)
        self.lock=threading.RLock()
        self.pages=set()
        self.limit_callback=limit_callback
        self.on_fault=on_fault
        self.fd=os.open(self.path,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
        self.closed=False
        self.poisoned=False

    def close(self):
        with self.lock:
            if not self.closed:os.close(self.fd);self.closed=True

    def _healthy(self):
        if self.closed or self.poisoned:
            raise LazyWeightError("paged COW store closed or poisoned; clean signed-journal replay required")

    def _bounds(self,page):
        a=page*self.page_elems
        return a,min(self.n,a+self.page_elems)

    def _page(self,page):
        a,b=self._bounds(page)
        if page not in self.pages:
            return self.base.reshape(-1)[a:b].clone()
        length=(b-a)*self.base.element_size()
        data=os.pread(self.fd,length,a*self.base.element_size())
        if len(data)!=length:raise LazyWeightError('short COW page read')
        # clone: never allow caller to mutate the returned bytes or original base.
        return torch.frombuffer(bytearray(data),dtype=self.dtype).clone()

    def _read_flat(self,a,b):
        values=self.base.reshape(-1)[a:b].clone()
        for page in range(a//self.page_elems,(b-1)//self.page_elems+1):
            if page not in self.pages:continue
            pa,pb=self._bounds(page)
            left=max(a,pa);right=min(b,pb)
            values[left-a:right-a].copy_(self._page(page)[left-pa:right-pa])
        return values

    def rows(self,lo,hi):
        if not 0<=lo<=hi<=self.shape[0]:raise IndexError('bad COW row range')
        with self.lock:
            self._healthy()
            a=lo*self.shape[1];b=hi*self.shape[1]
            return self._read_flat(a,b).view(hi-lo,self.shape[1])

    def tile(self,rs,cs):
        if not isinstance(rs,slice) or not isinstance(cs,slice):
            raise TypeError('paged source permits only bounded 2D slice tiles')
        lo,hi,step=rs.indices(self.shape[0]);l2,h2,step2=cs.indices(self.shape[1])
        if step!=1 or step2!=1:raise ValueError('paged source requires unit stride')
        with self.lock:
            self._healthy()
            # read exactly requested columns, never widen to entire long row
            out=self.base[lo:hi,l2:h2].clone()
            if not self.pages:return out
            # Tiny bounded LRU prevents rereading the same on-disk page
            # once per row when a page spans several narrow rows.
            cache=OrderedDict()
            for i,r in enumerate(range(lo,hi)):
                a=r*self.shape[1]+l2;b=r*self.shape[1]+h2
                for page in range(a//self.page_elems,(b-1)//self.page_elems+1) if b>a else ():
                    if page not in self.pages:continue
                    pa,pb=self._bounds(page)
                    left=max(a,pa);right=min(b,pb)
                    if page in cache:
                        cached=cache[page];cache.move_to_end(page)
                    else:
                        cached=self._page(page);cache[page]=cached
                        if len(cache)>16:cache.popitem(last=False)
                    out[i,left-a:right-a].copy_(cached[left-pa:right-pa])
            return out

    def __getitem__(self,item):
        if not isinstance(item,tuple) or len(item)!=2:
            raise TypeError('paged source requires 2D tile')
        return self.tile(*item)

    @property
    def T(self):return TransposedPages(self)

    def gather_rows(self,idx):
        cpu=idx.to(device='cpu',dtype=torch.long)
        if cpu.numel() and ((cpu<0).any() or (cpu>=self.shape[0]).any()):
            raise IndexError('embedding index outside COW source')
        unique,inverse=torch.unique(cpu.reshape(-1),sorted=True,return_inverse=True)
        result=torch.empty((len(unique),self.shape[1]),dtype=self.dtype)
        with self.lock:
            self._healthy()
            for n,row in enumerate(unique.tolist()):
                result[n].copy_(self._read_flat(row*self.shape[1],(row+1)*self.shape[1]))
        return result[inverse].view(*idx.shape,self.shape[1])

    def write_rows(self,lo,values):
        """Called under signed gate. Fail closed on *any* partial update error."""
        try:
            return self._write_rows(lo,values)
        except BaseException:
            self.poisoned=True
            if self.on_fault is not None:self.on_fault()
            raise

    def _write_rows(self,lo,values):
        if values.device.type!='cpu' or values.dtype!=self.dtype:
            values=values.to(device='cpu',dtype=self.dtype)
        if values.ndim!=2 or values.shape[1]!=self.shape[1]:
            raise LazyWeightError('COW update shape mismatch')
        hi=lo+values.shape[0]
        if not 0<=lo<=hi<=self.shape[0]:raise LazyWeightError('COW update range')
        if not torch.isfinite(values).all():raise LazyWeightError('nonfinite COW values')
        with self.lock:
            self._healthy()
            a=lo*self.shape[1];b=hi*self.shape[1]
            newflat=values.contiguous().view(-1)
            # Validate disk capacity for ALL new pages before writing any page.
            page_ids=list(range(a//self.page_elems,(b-1)//self.page_elems+1)) if b>a else []
            new_pages=[p for p in page_ids if p not in self.pages]
            if self.limit_callback:self.limit_callback(len(new_pages)*self.page_bytes)
            for page in page_ids:
                pa,pb=self._bounds(page)
                l=max(a,pa);r=min(b,pb)
                old=self._page(page)
                replacement=newflat[l-a:r-a]
                # Deltas can be sparse: do not allocate untouched pages.
                if torch.equal(old[l-pa:r-pa],replacement):continue
                old[l-pa:r-pa].copy_(replacement)
                offset=pa*self.base.element_size()
                data=old.contiguous().view(torch.uint8).numpy().tobytes()
                if page not in self.pages and hasattr(os,'posix_fallocate'):
                    # Disk-full is an exception rather than mmap SIGBUS.
                    try:os.posix_fallocate(self.fd,offset,len(data))
                    except OSError as exc:
                        import errno
                        if exc.errno not in (errno.EOPNOTSUPP,errno.ENOSYS,errno.EINVAL):raise
                if os.pwrite(self.fd,data,offset)!=len(data):
                    raise LazyWeightError('short COW page write')
                if page not in self.pages:
                    self.pages.add(page)
                    if self.limit_callback:self.limit_callback(-len(data))

    def stats(self):
        return {'changed_pages':len(self.pages),
                'allocated_page_bytes':sum((self._bounds(i)[1]-self._bounds(i)[0])*self.base.element_size()
                                          for i in self.pages)}


class TransposedPages:
    _fedpara_is_paged=True
    def __init__(self,source):
        self.source=source;self.shape=(source.shape[1],source.shape[0]);self.dtype=source.dtype
    def __getitem__(self,index):
        return self.source[index[1],index[0]].T.contiguous()


class PageDestination:
    device=torch.device('cpu')
    def __init__(self,store,lo,hi):
        self.store=store;self.lo=lo;self.hi=hi;self.dtype=store.dtype
    def float(self):return self.store.rows(self.lo,self.hi).float()
    def copy_(self,value):self.store.write_rows(self.lo,value)


class PagedMutableWeights:
    """Fallback to v60 for unsupported groups, safely page eligible disk bases."""
    def __init__(self,model,adapter,*,scratch_dir=None,page_kib=64,limit_gib=0):
        self.model=model
        self.fallback=LazyMutableWeights(model,scratch_dir=scratch_dir,limit_gib=limit_gib)
        self.directory=self.fallback.directory
        self.stores={}
        if page_kib not in (4,16,64,256,1024):
            raise ValueError("page size must be 4,16,64,256,1024 KiB")
        self.page_bytes=page_kib*1024
        self.limit_bytes=int(limit_gib*1024**3)
        self._reserved=0
        self.poisoned=False

    def _mark_poison(self):
        self.poisoned=True

    def _healthy(self):
        if self.poisoned:
            raise LazyWeightError('paged COW manager poisoned; restart from signed journal')

    def _fallback_group(self,group):
        parent,sep,attr=group.aliases[0].rpartition('.')
        module=self.model.get_submodule(parent) if sep else self.model
        original=module.parametrizations[attr].original
        added=original.numel()*original.element_size()
        if self.limit_bytes and self._reserved+self.fallback.used_bytes+added>self.limit_bytes:
            raise LazyWeightError('COW fallback would exceed combined disk quota')
        return self.fallback.materialize_group(group)

    def _quota(self,amount):
        if amount>0 and self.limit_bytes and self._reserved+amount+self.fallback.used_bytes>self.limit_bytes:
            raise LazyWeightError('paged COW capacity exceeds disk quota')
        if amount<0:self._reserved+=(-amount)

    def materialize_group(self,group):
        self._healthy()
        if group.canonical in self.stores:
            self.stores[group.canonical]._healthy()
            return False
        if group.canonical in self.fallback.materialized:return False
        modules=[]
        for alias in group.aliases:
            parent,sep,leaf=alias.rpartition('.')
            module=self.model.get_submodule(parent) if sep else self.model
            # Existing offload kernels must consume the virtual source. Never
            # expose stale original tensors to a non-intercepted forward.
            if not hasattr(module,'_fedpara_disk_source'):
                return self._fallback_group(group)
            modules.append(module)
        parent,sep,attr=group.aliases[0].rpartition('.')
        orig=(self.model.get_submodule(parent) if sep else self.model).parametrizations[attr].original
        if orig.ndim!=2 or orig.device.type!='cpu' or not orig.is_contiguous():
            return self._fallback_group(group)
        digest=hashlib.sha256(group.canonical.encode()).hexdigest()
        store=SparsePageStore(orig,self.directory/(digest+'.pages'),page_bytes=self.page_bytes,
                              limit_callback=self._quota,on_fault=self._mark_poison)
        self.stores[group.canonical]=store
        for mod in modules:
            cache=getattr(mod,'_fedpara_disk_cache',None)
            if cache is not None:cache.invalidate()
            mod._fedpara_disk_source=store
            mod._fedpara_disk_path='<epoch-private-sparse-pages>'
        return True

    def destination(self,group,lo,hi,original):
        self._healthy()
        store=self.stores.get(group.canonical)
        if store is not None:return PageDestination(store,lo,hi)
        return original[lo:hi]

    def materialize_all(self,adapter):
        self._healthy()
        # Legacy exporters need ordinary effective torch.Tensor Parameters.
        # This operation is explicitly O(model size); don't do it each event.
        for group in adapter.groups:
            store=self.stores.get(group.canonical)
            if store is not None:store._healthy()
            if store is None:
                self._fallback_group(group)
                continue
            # The source has not been modified, so v60's immutable copy is safe
            self._fallback_group(group)
            parent,_,attr=group.aliases[0].rpartition('.')
            orig=self.model.get_submodule(parent).parametrizations[attr].original
            flat=orig.data.reshape(-1)
            with torch.no_grad():
                for i in sorted(store.pages):
                    a,b=store._bounds(i)
                    flat[a:b].copy_(store._page(i))
            store.close()
            self.stores.pop(group.canonical,None)

    def stats(self):
        return {'changed_tensors':len(self.stores),
                'allocated_page_bytes':sum(s.stats()['allocated_page_bytes'] for s in self.stores.values()),
                'fallback_bytes':self.fallback.used_bytes,'scratch':str(self.directory)}


def attach_paged_mutable(model,adapter,*,scratch_dir=None,page_kib=64,limit_gib=0):
    if getattr(model,'_fedpara_streaming_base_info',{}).get('mode')!='read-only-mmap':
        raise LazyWeightError('paged COW requires immutable safetensors genesis')
    manager=PagedMutableWeights(model,adapter,scratch_dir=scratch_dir,page_kib=page_kib,limit_gib=limit_gib)
    object.__setattr__(model,'_fedpara_lazy_mutable_weights',manager)
    def forbid_stale_full_state(module,prefix,keep_vars):
        if manager.stores:
            raise LazyWeightError('paged COW base is virtual: materialize all before torch state_dict')
    # A direct torch.save(model.state_dict()) would otherwise export stale
    # immutable originals despite current, signed virtual page updates.
    model.register_state_dict_pre_hook(forbid_stale_full_state)
    return manager


def paged_destination(adapter,group,lo,hi,original):
    manager=getattr(adapter.model,'_fedpara_lazy_mutable_weights',None)
    if manager is not None and hasattr(manager,'destination'):
        return manager.destination(group,lo,hi,original)
    return original[lo:hi] if group.shape else original
