"""Fail-closed projection of a *verified* paged-v2 epoch into the v52 trainer.

The projection is local ONLY, never signed or sent as a quorum proof. Every
projected cut is reconstructed from the complete Merkle page set whose header
is bound to a persisted, cryptographically verified compact quorum certificate.
Missing even one page => not certified *for the local trainer*.
"""
from __future__ import annotations
from .core import PatchError
from .online_generation import _digest
from .frontier_commitment_v44 import verify_full_frontier

class CertifiedCompactTrainerLedger:
    def __init__(self, compact_ledger, paged_store, *, max_epochs=16384):
        self.compact=compact_ledger
        self.store=paged_store
        self.genesis=compact_ledger.genesis
        self.max_epochs=max_epochs
        self._projection={}

    def certified_epoch(self, epoch):
        if type(epoch) is not int or epoch<1 or epoch>self.max_epochs:
            return None
        # Consult the pinned cryptographic ledger even for cached projections.
        base=self.compact.certified_epoch(epoch)
        if base is None:return None
        signed=base['proposal']
        if epoch in self._projection:
            cached=self._projection[epoch]
            if cached['compact_proposal_digest']!=_digest(signed):
                raise PatchError('certified compact proposal changed after projection')
            return {'proposal':cached, 'compact_certificate':base}
        # The signed certificate is NOT a frontier: the FULL set of individually
        # verified pages is required, including for historical restart.
        pages=self.store.certified_pages(signed)
        if pages is None:return None
        cutoffs=verify_full_frontier(signed['frontier_header'], pages)
        if len(cutoffs)>4096:raise PatchError('oversized projected frontier')
        predecessor=(self.genesis if epoch==1 else self.certified_epoch(epoch-1))
        if predecessor is None:return None
        parent=(predecessor if isinstance(predecessor,str)
                else _digest(predecessor['proposal']))
        projected=dict(signed)
        projected.update({'protocol':'kagami-local-compact-trainer-projection-v57',
            'parent':parent, 'cutoffs':cutoffs, 'cut':signed['inventory'],
            'compact_proposal_digest':_digest(signed)})
        self._projection[epoch]=projected
        return {'proposal':projected,'compact_certificate':base}

    def latest(self):
        base=self.compact.latest()
        if base is None:return None
        # Never skip a missing page, predecessor or verified certificate.
        return self.certified_epoch(base['proposal']['epoch'])

    def head(self):
        latest=self.latest()
        return _digest(latest['proposal']) if latest else self.genesis
