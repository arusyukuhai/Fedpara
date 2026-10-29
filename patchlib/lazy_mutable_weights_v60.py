"""On-first-signed-update writable backing for a frozen safetensors genesis.

This is NOT sparse delta storage: each *modified tensor* is copied once on first
update, so the eventual disk footprint can approach the full model.  It avoids
up-front O(model-size) disk copies and, critically, never mutates the genesis.

Only the ContinualElasticEngine write gate can materialize a group.  This module
must not be used for arbitrary third-party in-place writes to base Parameters.
Ephemeral scratch is reconstructed from the signed journal after restart.
"""
from __future__ import annotations

import hashlib
import errno
import os
import tempfile
import weakref
from pathlib import Path

import torch


class LazyWeightError(RuntimeError):
    pass


class LazyMutableWeights:
    def __init__(self, model, *, scratch_dir=None, copy_mib=4, limit_gib=0):
        if not 1 <= copy_mib <= 128:
            raise ValueError('copy_mib must be 1..128')
        if limit_gib < 0:
            raise ValueError('limit_gib must be nonnegative')
        self.model=model
        from .live_linear_workspace_v59 import allocate, _release
        # v59's locked lease collector is crash-safe. Use a dedicated root so
        # another offload subsystem can never reclaim our active scratch.
        scratch_root=Path(scratch_dir or tempfile.gettempdir())/'kagami-lazy-v60'
        self.directory,lease=allocate(scratch_root)
        self._cleanup=weakref.finalize(model,_release,lease,self.directory)
        self.block_bytes=int(copy_mib*1024**2)
        self.limit_bytes=int(limit_gib*1024**3)
        self.used_bytes=0
        self.materialized={}

    def materialize_group(self, group):
        """Called with adapter.lock held before the first potential base write."""
        if group.canonical in self.materialized:
            return False
        members=[]
        for alias in group.aliases:
            owner,sep,attr=alias.rpartition('.')
            if not sep:
                module=self.model
                attr=alias
            else:
                module=self.model.get_submodule(owner)
            original=module.parametrizations[attr].original
            members.append((module,original))
        if not members or any(p is not members[0][1] for _,p in members):
            raise LazyWeightError('tied base group is inconsistent')
        original=members[0][1]
        if original.device.type!='cpu' or not original.is_contiguous() or original.layout!=torch.strided:
            raise LazyWeightError('lazy mutable base requires contiguous CPU safetensors')
        count=original.numel()
        byte_size=count*original.element_size()
        if self.limit_bytes and self.used_bytes+byte_size>self.limit_bytes:
            raise LazyWeightError('lazy writable tensor would exceed configured disk quota')
        digest=hashlib.sha256(group.canonical.encode('utf-8')).hexdigest()
        dest=self.directory/(digest+'.raw')
        # A different process or previous epoch must never reuse this scratch.
        if dest.exists():raise LazyWeightError('writable scratch name already exists')
        try:
            with dest.open('xb') as fp:
                # Reserve blocks before mapping. A sparse truncate followed
                # by mmap writes can SIGBUS the whole trainer on disk-full.
                # posix_fallocate detects ENOSPC as an ordinary exception.
                if byte_size:
                    allocated=False
                    if hasattr(os,'posix_fallocate'):
                        try:
                            os.posix_fallocate(fp.fileno(),0,byte_size)
                            allocated=True
                        except OSError as e:
                            if e.errno not in (errno.EOPNOTSUPP,errno.ENOSYS,errno.EINVAL):
                                raise
                    if not allocated:
                        # Fallback for filesystems/platforms without fallocate.
                        # Never allocate a multi-GB temporary Python bytestring.
                        block=b'\0' * min(self.block_bytes,1024*1024)
                        left=byte_size
                        while left:
                            n=min(left,len(block))
                            fp.write(block[:n]);left-=n
                fp.truncate(byte_size)
            mapped=torch.from_file(str(dest),shared=True,size=count,dtype=original.dtype).view(original.shape)
            with torch.no_grad():
                src=original.detach().reshape(-1)
                target=mapped.reshape(-1)
                step=max(1,self.block_bytes//original.element_size())
                for i in range(0,count,step):
                    target[i:i+step].copy_(src[i:i+step])
            # Only publish the new tensor after the entire source has been copied.
            # Disk kernels cache the source separately from the Parameter.
            old_data=original.data
            old_raw=getattr(original,'_fedpara_raw_path',None)
            previous=[]
            seen=set()
            try:
                for module,_ in members:
                    if id(module) in seen:continue
                    seen.add(id(module))
                    if hasattr(module,'_fedpara_disk_source'):
                        previous.append((module,module._fedpara_disk_source,
                                         module._fedpara_disk_path))
                original.data=mapped
                original._fedpara_raw_path=str(dest)
                for module,_,_ in previous:
                    cache=getattr(module,'_fedpara_disk_cache',None)
                    if cache is not None:cache.invalidate()
                    module._fedpara_disk_source=str(dest)
                    module._fedpara_disk_path=str(dest)
                self.materialized[group.canonical]=str(dest)
                self.used_bytes+=byte_size
            except BaseException:
                original.data=old_data
                if old_raw is None and hasattr(original,'_fedpara_raw_path'):
                    delattr(original,'_fedpara_raw_path')
                elif old_raw is not None:
                    original._fedpara_raw_path=old_raw
                for module,old_source,old_path in previous:
                    module._fedpara_disk_source=old_source
                    module._fedpara_disk_path=old_path
                raise
            return True
        except BaseException:
            # Fail closed: do not allow an incomplete snapshot to masquerade
            # as the mutable base. The engine must poison on apply failure.
            if group.canonical not in self.materialized:
                dest.unlink(missing_ok=True)
            raise

    def stats(self):
        return {'materialized_tensors':len(self.materialized),
                'mutable_disk_bytes':self.used_bytes,
                'scratch':str(self.directory)}


def attach_lazy_mutable(model,adapter,*,scratch_dir=None,copy_mib=4,limit_gib=0):
    if getattr(model,'_fedpara_streaming_base_info',{}).get('mode')!='read-only-mmap':
        raise LazyWeightError('lazy mutable weights require immutable safetensors base')
    manager=LazyMutableWeights(model,scratch_dir=scratch_dir,copy_mib=copy_mib,limit_gib=limit_gib)
    object.__setattr__(model,'_fedpara_lazy_mutable_weights',manager)
    return manager


def prepare_base_group(adapter,group):
    manager=getattr(adapter.model,'_fedpara_lazy_mutable_weights',None)
    if manager is not None:
        manager.materialize_group(group)


def materialize_all(adapter):
    manager=getattr(adapter.model,'_fedpara_lazy_mutable_weights',None)
    if manager is not None:
        if hasattr(manager,"materialize_all"):
            return manager.materialize_all(adapter)
        for group in adapter.groups:
            manager.materialize_group(group)
