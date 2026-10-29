import json
import pytest
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat,load_pem_public_key
from patchlib.identity import generate_keypair
from patchlib.roomid import id_from_raw_key
from patchlib.core import PatchError
from patchlib.generation_committee_v43 import read_generation_committee


def test_offline_committee_keys_pinned(tmp_path):
    keys={}
    for n in range(3):
        priv=tmp_path/f'{n}.key';pub=tmp_path/f'{n}.pub';generate_keypair(priv,pub)
        pem=pub.read_bytes();key=load_pem_public_key(pem)
        peer=id_from_raw_key(key.public_bytes(Encoding.Raw,PublicFormat.Raw))
        keys[peer]=pem.decode('ascii')
    path=tmp_path/'committee.json'
    path.write_text(json.dumps({'keys':keys}))
    members,pins=read_generation_committee(path)
    assert len(members)==3 and all(pins[k] for k in members)
    fake={'not-real':next(iter(keys.values()))}
    path.write_text(json.dumps({'keys':fake}))
    with pytest.raises(PatchError):read_generation_committee(path)


def test_legacy_list_supported(tmp_path):
    path=tmp_path/'committee.json';path.write_text(json.dumps(['peer-a','peer-b','peer-c']))
    assert read_generation_committee(path)==(['peer-a','peer-b','peer-c'],{})
