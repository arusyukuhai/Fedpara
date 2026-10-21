import hashlib
import json
import pytest
from patchlib.core import PatchError
from patchlib.p2p import FrameReceiver, CHUNK


def test_receive_chunks_and_integrity():
    data = b'fpatch-test' * 11987
    fr = FrameReceiver()
    fr.feed(json.dumps({'v':1,'type':'start','id':'abc','size':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
    for i in range(0, len(data), CHUNK):
        assert fr.feed(data[i:i+CHUNK]) is None
    assert fr.feed(json.dumps({'v':1,'type':'end','id':'abc'})) == data


def test_reject_oversized_mismatched_and_overlap():
    fr = FrameReceiver(max_bytes=10)
    with pytest.raises(PatchError):
        fr.feed(json.dumps({'v':1,'type':'start','id':'a','size':11,'sha256':'0'*64}))
    fr.feed(json.dumps({'v':1,'type':'start','id':'a','size':3,'sha256':'0'*64}))
    with pytest.raises(PatchError):
        fr.feed(b'1234')
    with pytest.raises(PatchError):
        fr.feed(json.dumps({'v':1,'type':'end','id':'a'}))
