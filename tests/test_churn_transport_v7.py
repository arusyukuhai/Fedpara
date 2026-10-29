"""Deterministic failure injection for stale signaling, async SCTP, and churn.

These tests intentionally cover stop/start while another peer is negotiating.
No ICE service, GPU, or Hugging Face download is required.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from patchlib.negotiation_guard import NegotiationGuard
from patchlib.p2p import PatchNode, wait_send_window, WINDOW
from patchlib.elasticnet import PeerSlot


def test_sdp_answer_offer_and_burst_of_candidates_are_idempotent():
    initiator=NegotiationGuard('initiator',max_candidates=3)
    # Late answer from a prior connection, while current negotiation is stable.
    assert not initiator.accept_answer('stable')
    assert initiator.accept_answer('have-local-offer')
    assert not initiator.accept_answer('have-local-offer')
    responder=NegotiationGuard('answerer',max_candidates=3)
    assert not responder.accept_offer('have-remote-offer')
    for i in range(100):
        responder.push_candidate(str(i),'cand'+str(i),False)
    assert len(responder.pending)==3
    assert responder.ignored_candidates>=97
    assert not responder.push_candidate('1','duplicate',False)
    assert responder.accept_offer('stable')
    assert not responder.accept_offer('stable')
    assert responder.release()==['cand0','cand1','cand2']
    assert responder.release()==[]
    assert responder.push_candidate('new','immediate',True) is True


def test_sctp_uses_low_water_notifications_and_unsubscribes():
    class FakeSCTP:
        readyState='open'
        bufferedAmount=WINDOW+65536
        bufferedAmountLowThreshold=0
        def __init__(self): self.callbacks={};self.wired=0;self.cleaned=0
        def once(self,name,callback):self.callbacks[name]=callback;self.wired+=1
        def remove_listener(self,name,callback):
            if self.callbacks.get(name) is callback:self.callbacks.pop(name)
            self.cleaned+=1
        def drain(self,value):
            self.bufferedAmount=value
            if value<=self.bufferedAmountLowThreshold and 'bufferedamountlow' in self.callbacks:
                self.callbacks['bufferedamountlow']()
    async def exercise():
        ch=FakeSCTP()
        async def drain():
            await asyncio.sleep(.002)
            ch.drain(0)
        started=time.monotonic()
        await asyncio.gather(wait_send_window(ch,stall_seconds=.1),drain())
        assert time.monotonic()-started<.08
        assert ch.wired>=1 and ch.cleaned==ch.wired and not ch.callbacks
    asyncio.run(exercise())


def test_sctp_close_during_buffered_wait_is_not_misclassified_as_success():
    class Closed:
        readyState='open'
        bufferedAmount=WINDOW+1
    async def exercise():
        c=Closed()
        async def close():
            await asyncio.sleep(.005)
            c.readyState='closed'
        with pytest.raises(ConnectionError,match='closed'):
            await asyncio.gather(wait_send_window(c,stall_seconds=.1),close())
    asyncio.run(exercise())


def test_cancelled_midflight_send_requeues_if_room_available():
    class Channel:
        readyState='open'
        bufferedAmount=WINDOW+1
        def __init__(self):self.handlers={};self.sent=[]
        def on(self,name):
            def wrap(cb):self.handlers[name]=cb;return cb
            return wrap
        def send(self,value):self.sent.append(value)
    async def exercise():
        n=PatchNode('ws://localhost/ws','a'*32)
        ch=Channel()
        n._attach_channel(ch)
        n.publish(b'x'*32768)
        await asyncio.sleep(.01)
        assert n._send_task is not None
        n._send_task.cancel()
        await asyncio.gather(n._send_task,return_exceptions=True)
        assert n.outgoing.qsize()==1  # retained for new connected generation
    asyncio.run(exercise())


def test_pause_resume_and_simultaneous_join_reject_late_offer(monkeypatch):
    """A responder starts while remote is paused; duplicate offer races resume.

    After a good answer the RTC channel remains live despite transient
    signaling departure. Once the live link closes, node.run exits cleanly.
    """
    import sys
    import types
    class Channel:
        readyState='open';bufferedAmount=0;label='fedpara-patch'
        def __init__(self):self.callbacks={}
        def on(self,name):
            def wrap(cb):self.callbacks[name]=cb;return cb
            return wrap
        def close(self):
            self.readyState='closed'
            self.callbacks['close']()
        def send(self,value):pass
    class PC:
        instance=None
        def __init__(self,configuration):
            PC.instance=self
            self.events={};self.signalingState='stable';self.remoteDescription=None
            self.localDescription=None;self.connectionState='connected'
            self.channel=Channel();self.answers=0
        def on(self,name):
            def wrap(cb):self.events[name]=cb
            return wrap
        async def setRemoteDescription(self,v):
            self.remoteDescription=v;self.signalingState='have-remote-offer'
        async def setLocalDescription(self,v):
            self.localDescription=v;self.signalingState='stable'
        async def createAnswer(self):
            self.answers+=1
            return SimpleNamespace(sdp='our-answer')
        async def addIceCandidate(self,c):pass
        async def close(self):pass
    class WS:
        def __init__(self):
            self.queue=asyncio.Queue()
            for m in ({'type':'joined','role':'answerer','iceServers':[]},
                      {'type':'candidate','candidate':{'candidate':'candidate:abc','sdpMid':'0','sdpMLineIndex':0}},
                      {'type':'offer','sdp':'first'},
                      {'type':'offer','sdp':'first'},
                      {'type':'peer-left'}):
                self.queue.put_nowait(json.dumps(m))
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def send(self,value):
            if json.loads(value).get('type')=='answer':
                PC.instance.events['datachannel'](PC.instance.channel)
                asyncio.get_running_loop().call_later(.02,PC.instance.channel.close)
        async def recv(self):return await self.queue.get()
    aiortc=types.ModuleType('aiortc')
    aiortc.RTCPeerConnection=PC
    aiortc.RTCSessionDescription=lambda **kw:SimpleNamespace(**kw)
    aiortc.RTCConfiguration=lambda **kw:SimpleNamespace(**kw)
    aiortc.RTCIceServer=lambda **kw:SimpleNamespace(**kw)
    sdp=types.ModuleType('aiortc.sdp');sdp.candidate_from_sdp=lambda value:SimpleNamespace(candidate=value)
    monkeypatch.setitem(sys.modules,'aiortc',aiortc)
    monkeypatch.setitem(sys.modules,'aiortc.sdp',sdp)
    import websockets
    monkeypatch.setattr(websockets,'connect',lambda *args,**kwargs:WS())
    async def exercise():
        n=PatchNode('ws://localhost/ws','a'*32)
        # A graceful return is acceptable after the DataChannel closes during
        # signaling outage; the slot supervisor treats task completion as a
        # recoverable reconnect just like ConnectionError.
        try:
            await asyncio.wait_for(n.run(),timeout=1.5)
        except ConnectionError:
            pass
        assert PC.instance.answers==1
        assert n.stopped.is_set()
    asyncio.run(exercise())


def test_peer_slot_has_separate_retiring_generation():
    slot=PeerSlot('p'+'1'*32,'r'*32,None)
    assert slot.retiring is None


def test_gui_stop_restart_does_not_open_journal_during_previous_termination(monkeypatch,tmp_path):
    from patchlib.gui import DesktopController
    import subprocess
    created=[]
    class FakeProc:
        def __init__(self):
            self.pid=1234+len(created);self.running=True
            self.terminates=0;self.finishes_stop=True
        def poll(self):return None if self.running else 0
        def terminate(self):self.terminates+=1
        def wait(self,timeout):
            if not self.finishes_stop:
                raise subprocess.TimeoutExpired('old',timeout)
            self.running=False
            return 0
    def fake_spawn(*args,**kwargs):
        v=FakeProc();created.append(v);return v
    monkeypatch.setattr(subprocess,'Popen',fake_spawn)
    ctrl=DesktopController()
    first=ctrl.start(['python','some-script'],tmp_path/'one.log')
    assert first==1234
    ctrl.stop(wait=False)
    ctrl.stop(wait=False)
    assert created[0].terminates==1
    second=ctrl.start(['python','some-script'],tmp_path/'two.log')
    assert second!=first and len(created)==2
    assert not created[0].running
    assert not ctrl.stop_requested
    ctrl.stop(wait=True)
    if ctrl.log_handle:ctrl.log_handle.close()


def test_gui_refuses_second_generation_while_old_peer_is_stuck(monkeypatch,tmp_path):
    from patchlib.gui import DesktopController
    import subprocess
    class FakeProc:
        pid=4711
        def poll(self):return None
        def terminate(self):pass
        def wait(self,timeout):raise subprocess.TimeoutExpired('peer',timeout)
    spawned=[]
    def popen(*args,**kwargs):
        spawned.append(1)
        return FakeProc()
    monkeypatch.setattr(subprocess,'Popen',popen)
    ctrl=DesktopController()
    ctrl.start(['worker'],tmp_path/'one.log')
    ctrl.stop()
    with pytest.raises(RuntimeError,match='still stopping'):
        ctrl.start(['worker'],tmp_path/'two.log')
    assert len(spawned)==1  # no second process with the same profile/journal
    if ctrl.log_handle:ctrl.log_handle.close()
