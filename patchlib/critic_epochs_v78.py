"""Verified Actor epoch ancestry admitted by the independent v78 Critic lane.

No bytes supplied by the peer/network are treated as an epoch certificate.
The caller passes the already trusted, disk-backed quorum certificate ledger.
"""
from .core import PatchError
from .online_generation import _digest
from .live_epoch_switch_v48 import certified_genesis


def verified_epochs(ledger, root_genesis, active_epoch, active_genesis, *, limit=65536):
    """Verify a *contiguous* certified ancestry, returning known Actor genesi.

    This repeats certificate validation rather than trusting file/directory names.
    A missing link or fork fails closed, before Critic journal replay.
    """
    if type(active_epoch) is not int or not 0 <= active_epoch <= limit:
        raise PatchError('Critic certified epoch history exceeds safety bound')
    known={root_genesis}
    parent=root_genesis
    for epoch in range(1,active_epoch+1):
        certificate=ledger.certified_epoch(epoch)
        if certificate is None or not isinstance(certificate,dict):
            raise PatchError('missing certified Actor epoch in Critic lineage')
        proposal=certificate.get('proposal')
        if (not isinstance(proposal,dict) or proposal.get('epoch')!=epoch or
                proposal.get('parent')!=parent):
            raise PatchError('broken certified Actor ancestry for Critic')
        known.add(certified_genesis(proposal))
        parent=_digest(proposal)
    if (active_genesis not in known or
            (active_epoch==0 and active_genesis!=root_genesis)):
        raise PatchError('Actor active epoch does not match verified Critic ancestry')
    return known
