#!/usr/bin/env python3
"""Durable Kagami generation staging and fail-closed local GC.

Use only with a pinned public committee and a *separately implemented* trainer
handoff. It does not install weights or silently switch an active v1 trainer.
"""
from __future__ import annotations
import argparse
import base64
from pathlib import Path
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from patchlib.core import PatchError
from patchlib.identity import _message
from patchlib.generation_committee_v43 import read_generation_committee
from patchlib.online_generation import GenerationLedger
from patchlib.epoch_runtime_v43 import EpochRuntimeV43


def _trusted(pins):
    def verifier(pem):
        key=load_pem_public_key(pem)
        def verify(data):
            try:
                sig=base64.b64decode(data['sig'],validate=True)
                key.verify(sig,_message(data['size'],data['sha256']))
            except (KeyError,ValueError,TypeError,InvalidSignature) as exc:
                raise PatchError('generation signature verification failed') from exc
        return verify
    return {p:verifier(raw) for p,raw in pins.items()}


def open_controller(*,ledger_path,root,room,genesis,committee):
    members,pins=read_generation_committee(committee)
    if set(members)!=set(pins):
        raise PatchError('GC daemon requires complete pinned Ed25519 public committee keys')
    ledger=GenerationLedger(Path(ledger_path),room=room,peer='local-gc-daemon',
                            committee=members,signer=None,trusted=_trusted(pins),genesis=genesis)
    return ledger,EpochRuntimeV43(Path(root),ledger)


def run_once(manager,old_journal):
    cert=manager.ledger.latest()
    if cert is None:return {'state':'waiting-for-certified-generation'}
    epoch=cert['proposal']['epoch']
    state=manager.status(epoch)
    if state['state']=='missing' and Path(old_journal).exists():
        state=manager.stage(old_journal,epoch=epoch)
    retired=manager.poll()
    return {'epoch':epoch,'state':manager.status(epoch), 'retired_epochs':retired}


def main():
    p=argparse.ArgumentParser(description='Crash-recoverable certified generation GC, not a model installer')
    for name in ('ledger','root','room','genesis','committee','old-journal'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--interval',type=float,default=15.)
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if not 1<=a.interval<=3600:p.error('--interval must be 1..3600')
    ledger,manager=open_controller(ledger_path=a.ledger,root=a.root,room=a.room,
                                    genesis=a.genesis,committee=Path(a.committee))
    last=None
    try:
        while True:
            try:
                result=run_once(manager,Path(a.old_journal))
                summary=str(result)
                if summary!=last:
                    print(summary,flush=True);last=summary
            except (PatchError,OSError,ValueError) as exc:
                print(f'GC deferred safely: {type(exc).__name__}: {exc}',flush=True)
            if a.once:break
            time.sleep(a.interval)
    finally:
        manager.close();ledger.close()

if __name__=='__main__':main()
