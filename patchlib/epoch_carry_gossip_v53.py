"""Bounded, authenticated, reconnect-safe cross-epoch signed update repair.

The outer messages travel ONLY on an Ed25519-authenticated generation-root
WebRTC control link. Event bodies retain their independent original signatures:
the new-generation AsyncInboundPipeline verifies them against the certified
previous source genesis before journaling/applying. This helper NEVER applies
or deletes model state. Each peer has at most one streamed need in flight.

It is intentional that this is a one-hop *adjacent-generation* repair lane;
peers offline for multiple epochs still need verified checkpoint chain recovery.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time

from .core import PatchError

_SHA=re.compile(r'^[0-9a-f]{64}$')
_ORIGIN=re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
CHUNK=8192
INVENTORY_PAGE=16
RECENT_PAGE=8  # reserve half of every offer for newly sealed signed updates
MAX_EVENT=16*1024*1024


def _valid_id(x):
    return isinstance(x,str) and _ORIGIN.fullmatch(x) is not None


def validate_carry_control(kind,fields):
    if not isinstance(fields,dict):raise PatchError('invalid carry fields')
    specs={
        'generation-carry-offer':{'source','events'},
        'generation-carry-need':{'source','origin','seq','digest'},
        'generation-carry-chunk':{'source','origin','seq','digest','length','index','data'},
    }
    if kind not in specs or set(fields)!=specs[kind]:
        raise PatchError('invalid cross-generation control shape')
    if not isinstance(fields['source'],str) or not _SHA.fullmatch(fields['source']):
        raise PatchError('invalid source generation')
    if kind=='generation-carry-offer':
        events=fields['events']
        if not isinstance(events,list) or len(events)>INVENTORY_PAGE:
            raise PatchError('carry inventory exceeds page bound')
        for e in events:
            if (not isinstance(e,list) or len(e)!=4 or not _valid_id(e[0]) or
                type(e[1]) is not int or e[1]<1 or e[1]>=2**63 or
                not isinstance(e[2],str) or not _SHA.fullmatch(e[2]) or
                type(e[3]) is not int or not 0<e[3]<=MAX_EVENT):
                raise PatchError('invalid carry inventory descriptor')
    else:
        if (not _valid_id(fields['origin']) or type(fields['seq']) is not int or
            not 0<fields['seq']<2**63 or not isinstance(fields['digest'],str) or
            not _SHA.fullmatch(fields['digest'])):
            raise PatchError('invalid carried event identity')
        if kind=='generation-carry-chunk':
            if (type(fields['length']) is not int or not 0<fields['length']<=MAX_EVENT or
                type(fields['index']) is not int or not 0<=fields['index']<=MAX_EVENT//CHUNK or
                not isinstance(fields['data'],str) or len(fields['data'])>4*((CHUNK+2)//3)):
                raise PatchError('invalid carry frame length')
            try: data=base64.b64decode(fields['data'],validate=True)
            except (ValueError,base64.binascii.Error) as exc:raise PatchError('invalid carry base64') from exc
            if not data or len(data)>CHUNK or fields['index']*CHUNK+len(data)>fields['length']:
                raise PatchError('invalid carry frame alignment')
            if len(data)!=min(CHUNK,fields['length']-fields['index']*CHUNK):
                raise PatchError('short nonterminal carry frame')
    return True


class _Inbox:
    def __init__(self,root,peer,fields):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        if root.is_symlink():raise PatchError('unsafe carry spool root')
        fd,name=tempfile.mkstemp(prefix='carry-',suffix='.part',dir=root)
        self.path=Path(name);self.writer=os.fdopen(fd,'wb')
        self.peer=peer;self.ident=tuple(fields[k] for k in ('source','origin','seq','digest','length'))
        self.next_index=0;self.started=time.monotonic()
        self._verified_wire=None
        self._verified=False
        self._verify_task=None

    def write(self,fields):
        if tuple(fields[k] for k in ('source','origin','seq','digest','length'))!=self.ident:
            raise PatchError('interleaved carry frames')
        if fields['index']!=self.next_index:raise PatchError('out-of-order carry frame')
        data=base64.b64decode(fields['data'],validate=True)
        self.writer.write(data);self.next_index+=1
        if self.next_index*CHUNK+0 >= self.ident[4]:
            # Temporary spool contents are untrusted and discarded on crash.
            # The signed ingestion path fsyncs the authenticated wire *after*
            # verification. Avoid a 16-MiB fsync + SHA read in RTC callback.
            self.writer.close()
            return True
        return False

    def verify(self):
        if not self.writer.closed:raise PatchError('carry not complete')
        with self.path.open('rb') as f:
            return hashlib.file_digest(f,'sha256').hexdigest()==self.ident[3]

    def read(self):
        if not self.writer.closed or not self._verified:
            raise PatchError('carry not verified')
        # Queue backpressure can retry this submission hundreds of times;
        # do not reread and allocate a whole multi-MiB update every 3 ms.
        if self._verified_wire is None:
            self._verified_wire=self.path.read_bytes()
        return self._verified_wire

    def close(self):
        if not self.writer.closed:self.writer.close()
        if self._verify_task is not None:self._verify_task.cancel()
        self.path.unlink(missing_ok=True)
        self._verified_wire=None


class AdjacentEpochCarryGossip:
    """Single-event bounded repair over signed control frames, no full mesh.

    The old generation offers its original EventJournal; a newer generation
    offers the previous generation's durable carry archive. A receiver requests
    only events after its certified cut, and the regular signed inbound pipeline
    remains the sole model writer. No per-room peer tracking is retained.
    """
    def __init__(self,*,source_genesis,source_journal,spool,carry_archive=None,
                 max_event=MAX_EVENT,offer_interval=1.0):
        if (not isinstance(source_genesis,str) or not _SHA.fullmatch(source_genesis) or
            not 1<=max_event<=MAX_EVENT or not 0.1<=offer_interval<=60):
            raise ValueError('invalid carry gossip limits')
        if source_journal.genesis != source_genesis:
            raise PatchError('carry gossip source journal genesis mismatch')
        self.source=source_genesis;self.journal=source_journal
        self.archive=carry_archive;self.spool=Path(spool)
        self.spool.mkdir(parents=True,exist_ok=True)
        if self.spool.is_symlink():raise PatchError('unsafe carry spool directory')
        # Uncommitted fragments are NEVER authority for model state. A restart
        # rediscovers missing IDs from signed inventories; reclaim abandoned
        # partial files instead of leaking disk across many restarts.
        for stale in self.spool.glob('carry-*.part'):
            if stale.is_symlink() or not stale.is_file():
                raise PatchError('unsafe abandoned carry fragment')
            stale.unlink()
        self.max_event=max_event;self.interval=offer_interval
        self._peers={}  # bounded to live direct neighbors; cleared on disconnect
        self._inbox={};self._completed={}
        self._next_offer={};self._cursor={};self._sending={}
        self._last_need={};self._recent_sent={}
        self._send_requests={}   # one pending (origin, seq, digest) per neighbor
        self._fetch_tasks={}     # bounded disk-worker reads, NEVER in RTC callback
        self._submit_retry={}    # failed queue submissions are rate-limited
        self.bytes_sent=0;self.events_received=0
        self.bytes_read_for_delivery=0
        self.max_cached_complete_bytes=0

    def disconnect(self,peer):
        self._peers.pop(peer,None);self._next_offer.pop(peer,None)
        self._cursor.pop(peer,None);self._sending.pop(peer,None)
        self._last_need.pop(peer,None);self._recent_sent.pop(peer,None)
        self._send_requests.pop(peer,None);self._submit_retry.pop(peer,None)
        task=self._fetch_tasks.pop(peer,None)
        if task is not None:
            task.cancel()
        if peer in self._inbox:self._inbox.pop(peer).close()
        if peer in self._completed:self._completed.pop(peer).close()

    def _requestable(self,origin,seq,digest):
        if self.archive is None:return False
        if seq<=self.archive.cutoffs.get(origin,0):return False
        existing=self.archive.journal.has(origin,seq)
        if existing is not None:
            if existing[0]!=digest:raise PatchError('conflicting signed carry advertisement')
            return False
        return True

    def receive(self,peer,kind,fields):
        validate_carry_control(kind,fields)
        if fields['source']!=self.source:
            return # wrong adjacent generation; never reinterpret as current model updates
        if kind=='generation-carry-offer':
            if self.archive is None:return
            # Limit pending per peer to exactly one request. Each later offer
            # rotates the inventory; eventual repair survives reconnects.
            if peer in self._inbox or peer in self._completed:return
            # AsyncInboundPipeline may not have committed a completed event
            # when another OFFER arrived. Re-evaluate its durable index once
            # it does, or the peer can starve behind a fulfilled request.
            previous=self._peers.get(peer)
            if previous is not None:
                if self._requestable(*previous):return
                self._peers.pop(peer,None)
            for origin,seq,digest,length in fields['events']:
                if length<=self.max_event and self._requestable(origin,seq,digest):
                    self._peers[peer]=(origin,seq,digest)
                    return
        elif kind=='generation-carry-need':
            event=self.journal.has(fields['origin'],fields['seq'])
            if event is None or event[0]!=fields['digest']:
                return
            key=(fields['origin'],fields['seq'],fields['digest'])
            # Enqueue identity only; 16-MiB get()+hash MUST NOT block the
            # WebRTC control event loop. flush_completed() resolves requests
            # with EventJournal.get_async() on its dedicated I/O worker.
            if (peer not in self._sending and peer not in self._fetch_tasks and
                    not (self._recent_sent.get(peer,(None,0))[0]==key and
                         time.monotonic()-self._recent_sent[peer][1]<20.0)):
                self._send_requests[peer]=key
        elif kind=='generation-carry-chunk':
            if self.archive is None or fields['length']>self.max_event:return
            if not self._requestable(fields['origin'],fields['seq'],fields['digest']):return
            requested=self._peers.get(peer)
            if requested!=(fields['origin'],fields['seq'],fields['digest']):return
            obj=self._inbox.get(peer)
            if obj is None:
                if fields['index']!=0:return
                obj=_Inbox(self.spool,peer,fields);self._inbox[peer]=obj
            try:
                finished=obj.write(fields)
                if finished:
                    self._inbox.pop(peer)
                    self._completed[peer]=obj
            except BaseException:
                self.disconnect(peer)
                raise

    async def _fetch_wire(self,origin,seq):
        fetch=getattr(self.journal,'get_async',None)
        if fetch is not None:
            return await fetch(origin,seq)
        # Legacy wrappers must expose get_async when backed by SQLite:
        # sqlite3 connections cannot be moved to arbitrary worker threads.
        return self.journal.get(origin,seq)

    async def flush_completed(self,submit):
        """Pump bounded disk reads and delivered bodies outside RTC callbacks.

        A pending NEED never issues an unbounded read on the control path.
        Completed spool data is read only once, and admission retries are
        throttled if the signed inbound pipeline is saturated.
        """
        for peer,key in list(self._send_requests.items()):
            if peer not in self._fetch_tasks and peer not in self._sending:
                origin,seq,_=key
                self._fetch_tasks[peer]=(key,asyncio.create_task(
                    self._fetch_wire(origin,seq),name='carry-disk-read'))
                self._send_requests.pop(peer,None)
        for peer,(key,task) in list(self._fetch_tasks.items()):
            if not task.done():continue
            self._fetch_tasks.pop(peer,None)
            try:wire=task.result()
            except asyncio.CancelledError:continue
            if (wire is not None and len(wire)<=self.max_event and
                    hashlib.sha256(wire).hexdigest()==key[2]):
                # A completed fetch is ignored if the peer disconnected.
                self._sending[peer]=[key[0],key[1],key[2],0,wire]
        now=time.monotonic()
        for peer,obj in list(self._completed.items()):
            if not obj._verified:
                if obj._verify_task is None:
                    obj._verify_task=asyncio.create_task(asyncio.to_thread(obj.verify),
                        name='carry-spool-sha256')
                if not obj._verify_task.done():continue
                try:
                    valid=obj._verify_task.result()
                except BaseException:
                    self.disconnect(peer)
                    raise
                if not valid:
                    self.disconnect(peer)
                    raise PatchError('carry body SHA-256 mismatch')
                obj._verified=True
            if now<self._submit_retry.get(peer,0):continue
            cached_before=obj._verified_wire is not None
            wire=obj.read()
            if not cached_before:
                self.bytes_read_for_delivery+=len(wire)
            self.max_cached_complete_bytes=max(self.max_cached_complete_bytes,
                sum(len(o._verified_wire or b'') for o in self._completed.values()))
            if submit(peer,wire):
                self.events_received+=1
                obj.close();del self._completed[peer]
                self._submit_retry.pop(peer,None)
                self._peers.pop(peer,None)
            else:
                self._submit_retry[peer]=now+0.05

    def tick(self,neighbors,*,now=None):
        if now is None:now=time.monotonic()
        active=set()
        for peer,node in neighbors:
            if (node is None or not getattr(node,'hello_received',False) or
                not getattr(node,'remote_supports_generation',False) or
                not (getattr(node,'bridge_only',False) or self.archive is not None)):
                continue
            active.add(peer)
            if (peer in self._peers and peer not in self._completed and
                    peer not in self._inbox and now>=self._last_need.get(peer,0)):
                origin,seq,digest=self._peers[peer]
                if node.control('generation-carry-need',source=self.source,
                                origin=origin,seq=seq,digest=digest):
                    self._last_need[peer]=now+3.0
            active_send=self._sending.get(peer)
            if active_send is not None:
                origin,seq,digest,index,wire=active_send
                # The original signed bytes are held only while the requested
                # event is in flight. Bound: degree(<=8)*max_event bytes.
                # At most four small frames per tick and neighbor.
                for _ in range(4):
                    if index*CHUNK>=len(wire):
                        self._sending.pop(peer,None);break
                    raw=wire[index*CHUNK:(index+1)*CHUNK]
                    if not node.control('generation-carry-chunk',source=self.source,
                            origin=origin,seq=seq,digest=digest,length=len(wire),
                            index=index,data=base64.b64encode(raw).decode('ascii')):break
                    self.bytes_sent+=len(raw);index+=1
                    active_send[3]=index
                if index*CHUNK>=len(wire):
                    self._recent_sent[peer]=((origin,seq,digest),time.monotonic())
                    self._sending.pop(peer,None)
            if now>=self._next_offer.get(peer,0):
                cursor=self._cursor.get(peer,0)
                # Recent signed suffixes must not spend days behind a
                # multi-million-row archive. Reserve half the wire page for
                # newest events; the other half round-robins the COMPLETE
                # historical rowid space. Both queries use the rowid index,
                # never OFFSET or a full events materialization.
                old_rows=self.journal.db.execute(
                    'SELECT rowid,origin,seq,digest,wire_len FROM events WHERE rowid>? '
                    'ORDER BY rowid LIMIT ?', (cursor,INVENTORY_PAGE-RECENT_PAGE)).fetchall()
                if not old_rows and cursor:
                    old_rows=self.journal.db.execute(
                        'SELECT rowid,origin,seq,digest,wire_len FROM events '
                        'ORDER BY rowid LIMIT ?', (INVENTORY_PAGE-RECENT_PAGE,)).fetchall()
                new_rows=self.journal.db.execute(
                    'SELECT rowid,origin,seq,digest,wire_len FROM events '
                    'ORDER BY rowid DESC LIMIT ?', (RECENT_PAGE,)).fetchall()
                seen=set();offered=[]
                for row in list(reversed(new_rows))+old_rows:
                    rowid,origin,seq,digest,length=row
                    if rowid not in seen and 0<length<=self.max_event:
                        offered.append([origin,seq,digest,length]);seen.add(rowid)
                if node.control('generation-carry-offer',source=self.source,events=offered):
                    if old_rows:self._cursor[peer]=old_rows[-1][0]
                    self._next_offer[peer]=now+self.interval
        for peer in (set(self._next_offer)|set(self._fetch_tasks)|
                     set(self._send_requests)|set(self._sending)|
                     set(self._inbox)|set(self._completed))-active:
            self.disconnect(peer)

    def close(self):
        for peer in (set(self._inbox)|set(self._completed)|
                     set(self._fetch_tasks)|set(self._send_requests)|
                     set(self._sending)):self.disconnect(peer)
