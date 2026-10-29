import asyncio
import hashlib
import os
import subprocess
import sys

import pytest

from patchlib.epoch_objects_v50 import (
    EpochObjectStore, CacheError, CapacityError, ByteRateLimiter, ResumableEpochFetch,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_resume_after_abrupt_process_exit(tmp_path):
    content = b'abcdefgh'*100
    digest = sha(content)
    source = '''
import os, sys
from patchlib.epoch_objects_v50 import EpochObjectStore
s=EpochObjectStore(sys.argv[1],quota_bytes=1024)
s.pin('candidate',{sys.argv[2]:800})
s.append(sys.argv[2],0,b'abcdefgh'*40)
os._exit(13)
'''
    result = subprocess.run([sys.executable,'-c',source,str(tmp_path),digest])
    assert result.returncode == 13
    with EpochObjectStore(tmp_path,quota_bytes=1024) as s:
        assert s.status(digest) == (800,320,False)
        assert s.append(digest,320,content[320:])
        assert s.path(digest).read_bytes() == content


def test_quota_never_evicts_pins_and_deduplicates(tmp_path):
    a,b = b'a'*600,b'b'*600
    with EpochObjectStore(tmp_path,quota_bytes=1024) as s:
        s.pin('active',{sha(a):len(a)})
        s.append(sha(a),0,a)
        s.pin('shadow',{sha(a):len(a)})
        assert s.reserved_bytes == 600
        with pytest.raises(CapacityError):
            s.pin('new',{sha(b):len(b)})
        assert s.path(sha(a)).read_bytes() == a
        s.release('active')
        assert s.collect(all_unpinned=True) == 0
        s.release('shadow')
        s.pin('new',{sha(b):len(b)})
        assert s.count == 1 and s.reserved_bytes == 600


def test_corrupt_payload_cannot_publish(tmp_path):
    with EpochObjectStore(tmp_path,quota_bytes=100) as s:
        digest=sha(b'good')
        s.pin('candidate',{digest:4})
        with pytest.raises(CacheError,match='hash mismatch'):
            s.append(digest,0,b'evil')
        assert s.status(digest) == (4,0,False)
        with pytest.raises(CacheError):
            s.path(digest)
        s.append(digest,0,b'good')
        assert s.path(digest).read_bytes() == b'good'


def test_recover_rename_before_index_commit(tmp_path):
    data=b'durable'
    d=sha(data)
    with EpochObjectStore(tmp_path) as s:
        s.pin('active',{d:len(data)})
        s.append(d,0,data)
        with s.db:
            s.db.execute('UPDATE objects SET complete=0,received=0 WHERE digest=?',(d,))
    with EpochObjectStore(tmp_path) as s:
        assert s.status(d) == (len(data),len(data),True)


def test_partial_uncommitted_tail_is_truncated(tmp_path):
    data=b'abcdefgh'
    d=sha(data)
    with EpochObjectStore(tmp_path) as s:
        s.pin('shadow',{d:8})
        s.append(d,0,data[:3])
        with s._path(d,partial=True).open('ab') as f:
            f.write(b'crash-tail')
    with EpochObjectStore(tmp_path) as s:
        assert s._path(d,partial=True).read_bytes() == b'abc'
        s.append(d,3,data[3:])
        assert s.path(d).read_bytes() == data


def test_cached_corruption_is_redownloaded_on_restart(tmp_path):
    d=sha(b'good')
    with EpochObjectStore(tmp_path) as s:
        s.pin('active',{d:4})
        s.append(d,0,b'good')
        s.path(d).write_bytes(b'evil')
    with EpochObjectStore(tmp_path) as s:
        assert s.status(d) == (4,0,False)
        assert s.collect(all_unpinned=True) == 0


def test_second_owner_rejected(tmp_path):
    with EpochObjectStore(tmp_path):
        with pytest.raises(CacheError,match='already has an owner'):
            EpochObjectStore(tmp_path)


def test_object_count_and_range_limits(tmp_path):
    with EpochObjectStore(tmp_path,max_objects=1) as s:
        s.pin('active',{sha(b'a'):1})
        with pytest.raises(CapacityError):
            s.pin('next',{sha(b'b'):1})
        with pytest.raises(CacheError):
            s.append(sha(b'a'),1,b'a')
        with pytest.raises(CacheError):
            s.pin('next',{'../../bad':3})


@pytest.mark.asyncio
async def test_failover_only_requests_missing_bytes(tmp_path):
    data = b'x'*200_000
    digest=sha(data)
    calls=[]
    async def request(peer,d,offset,length):
        calls.append((peer,offset,length))
        if peer=='disconnected':
            raise ConnectionError('gone')
        return data[offset:offset+length]
    with EpochObjectStore(tmp_path) as s:
        s.pin('shadow',{digest:len(data)})
        s.append(digest,0,data[:65000])
        downloader=ResumableEpochFetch(s,request,limiter=ByteRateLimiter(1e9))
        result=await downloader.download(digest,['disconnected','live'])
        assert result.read_bytes()==data
        assert calls[0][1]==65000
        assert all(n<=65536 for _,_,n in calls)
        assert sum(n for p,_,n in calls if p=='live')==len(data)-65000


@pytest.mark.asyncio
async def test_cancel_fetch_and_resume(tmp_path):
    data=b'x'*100_000
    d=sha(data)
    started=asyncio.Event()
    async def blocked(*_):
        started.set()
        await asyncio.Event().wait()
    with EpochObjectStore(tmp_path) as s:
        s.pin('shadow',{d:len(data)})
        fetch=ResumableEpochFetch(s,blocked,limiter=ByteRateLimiter(1e9))
        task=asyncio.create_task(fetch.download(d,['neighbor']))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert s.status(d)==(len(data),0,False)
        async def good(p,k,o,n):return data[o:o+n]
        fetch.fetch=good
        await asyncio.wait_for(fetch.download(d,['neighbor']),1)
        assert s.path(d).read_bytes()==data


@pytest.mark.asyncio
async def test_bad_source_rotated_and_retry_bounded(tmp_path):
    content=b'12345678'
    d=sha(content)
    calls=[]
    async def fetch(peer,d,o,n):
        calls.append(peer)
        return b'!'*n if peer=='bad' else content[o:o+n]
    with EpochObjectStore(tmp_path) as s:
        s.pin('shadow',{d:len(content)})
        job=ResumableEpochFetch(s,fetch,limiter=ByteRateLimiter(1e9),max_neighbors=2,max_attempts=2)
        await job.download(d,['bad','good','never-contact'])
        assert calls==['bad','good']


@pytest.mark.asyncio
async def test_global_bandwidth_limiter_is_shared():
    limiter=ByteRateLimiter(1000,burst=100)
    start=asyncio.get_running_loop().time()
    await asyncio.gather(*(limiter.take(100) for _ in range(4)))
    elapsed=asyncio.get_running_loop().time()-start
    assert elapsed>=.29


def test_long_run_storage_plateau(tmp_path):
    # Actual filesystem and SQLite, 300 successive bases. Only active and
    # preparing generations are pinned. Shared bytes are deduplicated.
    shared=b's'*200
    dshared=sha(shared)
    with EpochObjectStore(tmp_path,quota_bytes=2000,max_objects=10) as s:
        last=None
        for epoch in range(300):
            data=epoch.to_bytes(4,'big')*100
            d=sha(data)
            owner=f'epoch-{epoch}'
            s.pin(owner,{d:len(data),dshared:len(shared)})
            if not s.status(dshared)[2]:s.append(dshared,0,shared)
            s.append(d,0,data)
            if last is not None:s.release(last)
            last=owner
            s.collect(all_unpinned=True)
            assert s.reserved_bytes==600
            assert s.count==2
        assert sum(p.stat().st_size for p in (tmp_path/'objects').iterdir())==600
        # Metadata must not grow linearly with generations after page reuse.
        assert (tmp_path/'cache.sqlite3').stat().st_size < 128*1024

@pytest.mark.parametrize('corrupt',[False,True])
def test_crash_after_final_block_before_publication(tmp_path,corrupt):
    data=b'verified'
    digest=sha(data)
    with EpochObjectStore(tmp_path) as s:
        s.pin('active',{digest:len(data)})
        s._path(digest,partial=True).write_bytes(b'corrupt!' if corrupt else data)
        with s.db:
            s.db.execute('UPDATE objects SET received=size,complete=0 WHERE digest=?',(digest,))
    with EpochObjectStore(tmp_path) as s:
        if corrupt:
            assert s.status(digest)==(len(data),0,False)
        else:
            assert s.path(digest).read_bytes()==data
