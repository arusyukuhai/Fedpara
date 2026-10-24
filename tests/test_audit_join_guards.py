"""Audited cold-join state transitions and emergency snapshot sources."""
from patchlib.join_guard import BootstrapState, initial_join_decision, should_request_snapshot


def test_newcomer_still_requests_snapshot_if_signaling_dies_after_webrtc_auth():
    assert should_request_snapshot(fresh_join=True,snapshot_installed=False,
                                   signaling_has_members=False,authenticated_transport=True)
    assert not should_request_snapshot(fresh_join=True,snapshot_installed=False,
                                       signaling_has_members=False,authenticated_transport=False)
    assert not should_request_snapshot(fresh_join=False,snapshot_installed=True,
                                       signaling_has_members=True,authenticated_transport=True)


def test_completed_verified_snapshot_can_release_cold_join_without_neighbor_inventory():
    state=BootstrapState.restored(False)
    state.confirm_verified_snapshot()
    assert not state.fresh_join and state.snapshot_installed
    # The snapshot installer sets initial_complete at the signed history boundary.
    d=initial_join_decision(initial_complete=True,lobby_online=True,
        lobby_members={'p'+('a'*32)},neighbors={'p'+('a'*32):False},
        durable_history=True,signaling_seen=True)
    assert d.can_train
