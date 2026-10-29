from patchlib.elasticnet import (frontier_advertisement_delay,
                                 inventory_cycle_delay)


def test_synced_large_room_slows_redundant_control_not_loss_repair():
    assert frontier_advertisement_delay(4096)==1.5
    assert frontier_advertisement_delay(4096,catchup=True)==.3
    assert frontier_advertisement_delay(8)==.3
    assert inventory_cycle_delay(5.0,4096,peer_synced=True,has_frontier=True)==120.0
    assert inventory_cycle_delay(5.0,4096,peer_synced=False,has_frontier=True)==5.0
    assert inventory_cycle_delay(5.0,4096,peer_synced=True,has_frontier=False)==5.0
