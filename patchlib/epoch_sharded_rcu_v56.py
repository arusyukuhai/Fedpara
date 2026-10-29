"""RCU-style metadata switch for independently verified *disk-sharded* epochs.

Not an in-memory `nn.Module` replacement: each step must acquire a generation
lease and use only its immutable checkpoint + per-epoch optimizer store.
Background prewarming and atomic pointer publication are O(1) in model size.
Old leases finish with the old epoch; new leases receive the new one. No shard
is deleted automatically. A real 300B train-live backend must implement the
per-layer computation and optimizer writer fence separately.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from .core import PatchError,_canonical_json
from .online_generation import _sha


@dataclass(frozen=True)
class ShardedGeneration:
    epoch:int
    proposal_hash:str
    manifest_hash:str
    checkpoint_dir:Path
    optimizer_dir:Path


class ShardedEpochRCU:
    def __init__(self,root:Path,*, verify_generation):
        if not callable(verify_generation):raise ValueError('required independent verifier')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.pointer=self.root/'sharded-active.json'
        self.verify_generation=verify_generation
        self._active=None
        self._lock=asyncio.Lock()
        self._transition_lock=asyncio.Lock()
        self._admission_open=asyncio.Event();self._admission_open.set()
        self._writers=0
        self._readers={}
        self._drained=asyncio.Event();self._drained.set()
        self._staged=None
        if self.pointer.is_symlink():raise PatchError('unsafe sharded active pointer')

    def recover(self,resolver):
        """Reopen a committed pointer by rechecking **all** hashes and proofs."""
        if not self.pointer.exists():return None
        if not self.pointer.is_file() or self.pointer.stat().st_size>4096:
            raise PatchError('invalid sharded pointer length')
        try:data=json.loads(self.pointer.read_bytes())
        except (ValueError,UnicodeError) as exc:raise PatchError('corrupted sharded pointer') from exc
        if (not isinstance(data,dict) or set(data)!={'epoch','proposal','manifest'} or
                type(data['epoch']) is not int or data['epoch']<1 or
                not _sha(data['proposal']) or not _sha(data['manifest'])):
            raise PatchError('invalid durable sharded epoch pointer')
        generation=resolver(data['epoch'],data['proposal'],data['manifest'])
        self._verify(generation)
        self._active=generation
        return generation

    def _verify(self,generation):
        if (not isinstance(generation,ShardedGeneration) or
                type(generation.epoch) is not int or generation.epoch<1 or
                not _sha(generation.proposal_hash) or not _sha(generation.manifest_hash)
                or not isinstance(generation.checkpoint_dir,Path)
                or not isinstance(generation.optimizer_dir,Path)
                or generation.checkpoint_dir.is_symlink() or
                generation.optimizer_dir.is_symlink() or
                not generation.checkpoint_dir.is_dir() or
                not generation.optimizer_dir.is_dir()):
            raise PatchError('invalid sharded generation')
        if not self.verify_generation(generation):
            raise PatchError('sharded checkpoint/optimizer certificate verification failed')

    async def stage(self,generation):
        """Expensive verification is off the training event loop."""
        await asyncio.to_thread(self._verify,generation)
        async with self._transition_lock:
            async with self._lock:
                if self._active is not None and generation.epoch!=self._active.epoch+1:
                    raise PatchError('sharded adoption must be adjacent')
                self._staged=generation

    @asynccontextmanager
    async def step(self, *, writable=True):
        """Steps use a pinned generation; writing steps prevent rekey until done."""
        while True:
            await self._admission_open.wait()
            async with self._lock:
                if not self._admission_open.is_set():continue
                if self._active is None:raise PatchError('no sharded epoch active')
                generation=self._active
                self._readers[generation.epoch]=self._readers.get(generation.epoch,0)+1
                if writable:
                    self._writers+=1;self._drained.clear()
                break
        try:yield generation
        finally:
            async with self._lock:
                self._readers[generation.epoch]-=1
                if self._readers[generation.epoch]==0:del self._readers[generation.epoch]
                if writable:
                    self._writers-=1
                    if not self._writers:self._drained.set()

    async def publish(self, *, certified_step_boundary):
        """Small local fence only; no global training stop or full model copy.

        certified_step_boundary must be a LOCAL callback implementing final
        carry+network drain+optimizer flush, not an untrusted remote flag.
        """
        if not callable(certified_step_boundary):raise PatchError('writer fence callback required')
        async with self._transition_lock:
            async with self._lock:
                next_gen=self._staged
                if next_gen is None:raise PatchError('no verified sharded generation staged')
                if self._active is not None and next_gen.epoch!=self._active.epoch+1:
                    raise PatchError('non-adjacent sharded generation')
                # Close writer admission, then release the lock so old writers
                # can finish; otherwise waiting for _drained would deadlock.
                self._admission_open.clear()
            await self._drained.wait()
            result=certified_step_boundary(self._active,next_gen)
            if asyncio.iscoroutine(result):result=await result
            if result is not True:
                raise PatchError('epoch writer/carry fence was not certified')
            # Revalidate immutable contents after staging; no TOCTOU from user
            # supplied paths. Readers keep old files pinned until externally retired.
            await asyncio.to_thread(self._verify,next_gen)
            data={'epoch':next_gen.epoch,'proposal':next_gen.proposal_hash,'manifest':next_gen.manifest_hash}
            raw=_canonical_json(data)
            stage=self.pointer.with_suffix('.next')
            flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
            if stage.exists():raise PatchError('interrupted sharded pointer publication; recover first')
            fd=os.open(stage,flags,0o600)
            try:
                with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                os.replace(stage,self.pointer)
                dfd=os.open(self.root,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
                try:os.fsync(dfd)
                finally:os.close(dfd)
            finally:stage.unlink(missing_ok=True)
            async with self._lock:
                self._active=next_gen;self._staged=None
                self._admission_open.set()
            return next_gen

    def pinned_epochs(self):
        return frozenset(self._readers) | ({self._active.epoch} if self._active else set())
