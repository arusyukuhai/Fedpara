"""New multi-ring topology: bounded connections, logarithmic dissemination, churn."""
from collections import deque
import pytest
from patchlib.topology import neighbors, topology_metrics


def peers(n):
    return tuple(f'p{i:032x}' for i in range(n))


def distance_distribution(ids):
    index={p:i for i,p in enumerate(ids)}
    adj=[tuple(index[q] for q in neighbors(p,ids)) for p in ids]
    distance={0:0};q=deque([0]);
    while q:
        i=q.popleft()
        for j in adj[i]:
            if j not in distance:
                distance[j]=distance[i]+1;q.append(j)
    return distance,adj


@pytest.mark.parametrize('n', [33, 97, 1000, 4096])
def test_spread_without_complete_graph(n):
    ids=peers(n)
    metrics=topology_metrics(ids)
    assert metrics['undirected']
    assert metrics['max_degree']<=8
    assert metrics['links']<=4*n
    d,adj=distance_distribution(ids)
    assert len(d)==n
    if n>=1000:
        assert max(d.values()) <= 12
    if n>=1000:
        assert sum(d.values())/n < 8


@pytest.mark.parametrize('n',[96,2048,4095])
def test_joins_change_at_most_16_old_neighbor_lists(n):
    # Sequential-looking IDs are an adversarial test for weak hash mixing.
    before=peers(n)
    joining='p'+f'{2*n+1:032x}'
    after=(*before,joining)
    different=[p for p in before if neighbors(p,before)!=neighbors(p,after)]
    assert len(different)<=16
    for p in before:
        assert p in neighbors(joining,after) if joining in neighbors(p,after) else True


def test_overlay_deterministic_for_distinct_roster_orders():
    original=peers(555)
    for p in original[::20]:
        assert neighbors(p,original)==neighbors(p,reversed(original))


def test_empty_room_and_founder():
    assert topology_metrics([])['links'] == 0
    assert neighbors('p'+f'{1:032x}', ['p'+f'{1:032x}']) == set()
