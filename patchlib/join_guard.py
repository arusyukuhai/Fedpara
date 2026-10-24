"""Cold-join guard: do not train on genesis while reachable members have history.

This guard deliberately does not claim that a disconnected network can reach
consensus. A first-time joiner must establish a signaling roster and finish
anti-entropy with every selected live neighbor before training. An existing
participant with a durable local journal may keep training during an outage.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class JoinDecision:
    can_train: bool
    reason: str


def initial_join_decision(*, initial_complete: bool, lobby_online: bool,
                          lobby_members: set[str], neighbors: dict[str, bool],
                          durable_history: bool, signaling_seen: bool) -> JoinDecision:
    if initial_complete:
        return JoinDecision(True, 'synchronized')
    if not lobby_online:
        # This exception is for a peer that really existed before the outage.
        if durable_history and signaling_seen:
            return JoinDecision(True, 'recovered-offline-history')
        return JoinDecision(False, 'waiting-for-room-signaling: check Mummy /ws URL, version and port')
    if not lobby_members:
        return JoinDecision(True, 'room-confirmed-empty')
    missing = [peer for peer, ready in neighbors.items() if not ready]
    if missing or not neighbors:
        return JoinDecision(False,
            f'waiting-for-initial-P2P-sync: {len(missing)} neighbor(s) not caught up; '
            f'known room peers={len(lobby_members)}')
    return JoinDecision(True, 'initial-inventory-verified')


def established_member_decision(*, persisted_ready: bool, fresh_join: bool,
                                lobby_online: bool, signaling_seen: bool) -> JoinDecision | None:
    """Do not cold-join a crash-recovered member with a verified local baseline.

    A durable initial_ready bit was written only AFTER founding the room or
    installing a verified snapshot.  On restart, the engine replays its signed
    event journal and restores its local working checkpoint before this check.
    Its remote event gaps are repaired by asynchronous anti-entropy rather
    than blocking on a moving inventory-ready barrier.

    This cannot authorize a brand-new peer: its persisted_ready is False.
    """
    if persisted_ready and not fresh_join and (lobby_online or signaling_seen):
        return JoinDecision(True, 'rejoining-with-durable-model-and-background-anti-entropy')
    return None


@dataclass
class BootstrapState:
    """Mutable, per-process cold-join state.

    A first member who receives an *authoritative empty-room roster* is the
    genesis founder: its locally initialized model is the room baseline.
    After that transition it must NEVER request a full snapshot from later
    arrivals. Otherwise an existing trainer stops as soon as a second user
    enters, while that second user can appear to run normally.

    The transition is intentionally stricter than merely receiving a roster
    event, and is not taken during a signaling outage or when any peer exists.
    """
    fresh_join: bool
    snapshot_installed: bool

    @classmethod
    def restored(cls, persisted_ready: bool) -> 'BootstrapState':
        return cls(fresh_join=not persisted_ready,
                   snapshot_installed=bool(persisted_ready))

    def confirm_genesis_founder(self, *, decision: JoinDecision,
                                lobby_online: bool, lobby_members: set[str],
                                first_roster_was_empty: bool = False) -> bool:
        # Only the FIRST server-authored roster can establish genesis
        # ownership. A newcomer who saw an existing peer must NOT assume
        # ownership if that peer subsequently disconnects; its model may be
        # behind the room's last checkpoint.
        if not (self.fresh_join and lobby_online and first_roster_was_empty):
            return False
        self.fresh_join = False
        self.snapshot_installed = True
        return True

    def confirm_verified_snapshot(self) -> None:
        if not self.fresh_join:
            raise RuntimeError('A room founder/recovered peer must not replace its state with a newcomer snapshot')
        self.snapshot_installed = True
        self.fresh_join = False  # one-way transition: now an established member


def should_request_snapshot(*, fresh_join: bool, snapshot_installed: bool,
                            signaling_has_members: bool, authenticated_transport: bool) -> bool:
    """A live authenticated DataChannel is enough, even if Mummy has gone down.

    This predicate does NOT authorize training: that only happens after the
    signed manifest, its shards, and journal boundary have been verified.
    """
    return (fresh_join and not snapshot_installed and
            (signaling_has_members or authenticated_transport))
