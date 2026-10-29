"""Pinned committee configuration usable while members are disconnected.

`{"keys": {"<peer id>": "-----BEGIN PUBLIC KEY-----..."}}` lets an
ordinary peer verify a certified chain without ever having seen each committee
member connect. Old arrays of IDs still work but require learned/pinned keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from .core import PatchError,PEER_ID


def read_generation_committee(path:Path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    pins={}
    if isinstance(obj,dict) and set(obj)=={'keys'}:
        payload=obj['keys']
        if not isinstance(payload,dict):raise PatchError('committee keys must be a map')
        from cryptography.hazmat.primitives.serialization import load_pem_public_key,Encoding,PublicFormat
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from .roomid import id_from_raw_key
        for peer,pem in payload.items():
            if not isinstance(peer,str) or not PEER_ID.fullmatch(peer) or not isinstance(pem,str) or len(pem)>4096:
                raise PatchError('invalid committee peer ID or public key')
            try:public=load_pem_public_key(pem.encode('ascii'))
            except (ValueError,UnicodeError,TypeError) as exc:raise PatchError('invalid public key PEM') from exc
            if not isinstance(public,Ed25519PublicKey):raise PatchError('only Ed25519 committee keys are accepted')
            raw=public.public_bytes(Encoding.Raw,PublicFormat.Raw)
            if id_from_raw_key(raw)!=peer:raise PatchError('committee key does not match cryptographic peer ID')
            pins[peer]=public.public_bytes(Encoding.PEM,PublicFormat.SubjectPublicKeyInfo)
        members=list(payload)
    elif isinstance(obj,list):
        members=obj
    else:
        raise PatchError('expected legacy member array or pinned key map')
    if (len(members)!=len(set(members)) or not 3<=len(members)<=31
        or any(not isinstance(k,str) or not PEER_ID.fullmatch(k) for k in members)):
        raise PatchError('generation committee must contain 3..31 unique identities')
    return sorted(members),pins
