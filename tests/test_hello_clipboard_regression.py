"""Handshake order and a copyable diagnostic log are non-optional."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.roomid import auto_identity
from patchlib.elasticnet import ElasticNode, encode_control
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html


class Channel:
    readyState='open'
    bufferedAmount=0
    def __init__(self, fail_first=False):
        self.handlers={};self.sent=[];self.fail_first=fail_first
    def on(self,event):
        def decorator(f):self.handlers[event]=f;return f
        return decorator
    def send(self,msg):
        if self.fail_first:
            self.fail_first=False
            raise ConnectionError('temporary buffer unavailable')
        self.sent.append(msg)
    def close(self):
        self.readyState='closed'
        if 'close' in self.handlers:self.handlers['close']()


def test_signed_hello_before_local_open_event(tmp_path):
    async def run():
        alice,sa,pa,_=auto_identity(tmp_path/'alice.pem')
        bob,sb,pb,_=auto_identity(tmp_path/'bob.pem')
        n=ElasticNode('ws://localhost/ws','x'*32,origin=alice,remote=bob,genesis='0'*64,
                      public_key=pa,signer=sa)
        ch=Channel()
        n._attach_channel(ch)
        # Remote hello arrives before own open (valid and common race).
        remote=encode_control({'type':'hello','peer':bob,'genesis':'0'*64,'counter':1,'pub':pb},sb)
        ch.handlers['message'](remote)
        assert n.hello_received and n.verified_at and n.verifier is not None
        ch.handlers['open']()
        assert n.hello_received and n.verified_at, 'local on_open must not erase the remote hello'
        assert n.control('heartbeat')
        assert [json.loads(s)['type'] for s in ch.sent] == ['hello','heartbeat']
        ch.close()
    asyncio.run(run())


def test_first_hello_send_failure_retries_before_heartbeat(tmp_path):
    async def run():
        a,sign,pub,_=auto_identity(tmp_path/'alice.pem')
        b,_,_,_=auto_identity(tmp_path/'bob.pem')
        n=ElasticNode('ws://localhost/ws','x'*32,origin=a,remote=b,genesis='0'*64,
                      public_key=pub,signer=sign)
        ch=Channel(fail_first=True)
        n._attach_channel(ch)
        ch.handlers['open']()
        assert not n.hello_sent
        assert n.control('heartbeat')
        assert [json.loads(s)['type'] for s in ch.sent] == ['hello','heartbeat']
        ch.close()
    asyncio.run(run())


def test_stale_channel_close_cannot_cancel_current_connection(tmp_path):
    async def run():
        a,sign,pub,_=auto_identity(tmp_path/'a.pem')
        b,_,_,_=auto_identity(tmp_path/'b.pem')
        node=ElasticNode('ws://localhost/ws','x'*32,origin=a,remote=b,genesis='0'*64,
                         public_key=pub,signer=sign)
        old=Channel();new=Channel()
        node._attach_channel(old);old.handlers['open']()
        node._attach_channel(new);new.handlers['open']()
        old.close()
        assert node.connected.is_set() and node._channel is new
        new.close()
    asyncio.run(run())


def test_logs_native_copy_and_safe_export(tmp_path,monkeypatch):
    import patchlib.webview_controller as vc
    controller=WebviewController(tmp_path)
    p=tmp_path/'trainer.log';p.write_text('SIGNED HELLO FAILED: diagnostic example\n')
    controller.train.logfile=p
    args=[]
    monkeypatch.setattr(vc.sys,'platform','darwin')
    monkeypatch.setattr(vc.subprocess,'run',lambda *a,**kw: args.append((a,kw)))
    result=controller.copy_logs()
    assert result['chars']>0 and args[0][0][0]==['pbcopy']
    assert 'SIGNED HELLO FAILED' in args[0][1]['input']
    saved=controller.save_logs()
    assert Path(saved['path']).read_text()==p.read_text()
    assert Path(saved['path']).stat().st_mode & 0o077 == 0
    html=compose_html()
    assert 'id="copyLogs"' in html and 'id="saveLogs"' in html


def test_genesis_mismatch_is_explicit_and_aborts_instead_of_looping(tmp_path):
    from patchlib.elasticnet import HandshakeError
    async def scenario():
        local,sign,pub,_=auto_identity(tmp_path/'a.pem')
        remote,remote_sign,remote_pub,_=auto_identity(tmp_path/'b.pem')
        n=ElasticNode('ws://localhost/ws','x'*32,origin=local,remote=remote,
                      genesis='a'*64,public_key=pub,signer=sign)
        ch=Channel()
        n._attach_channel(ch)
        bad_hello=encode_control({'type':'hello','peer':remote,'genesis':'b'*64,
                                  'counter':1,'pub':remote_pub},remote_sign)
        ch.handlers['message'](bad_hello)
        assert isinstance(n.last_error,HandshakeError)
        assert 'GENESIS MISMATCH' in str(n.last_error)
        assert n.protocol_failure.is_set()
        assert ch.readyState=='closed'
        await asyncio.sleep(0)
    asyncio.run(scenario())
