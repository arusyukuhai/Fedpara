from tests.test_v44_low_cost_generation import bridge, DummyNode


def test_proposal_broadcast_to_all_authenticated_neighbors(tmp_path):
    b, ledger, _ = bridge(tmp_path)
    from patchlib.online_generation import make_proposal
    proposal = make_proposal(room='room', epoch=1, parent='0'*64,
        manifest='a'*64, semantic='b'*64, cut='c'*64,
        cutoffs={'peer01':5}, replicas=['a','b','c'])
    b.control.propose(proposal)
    peers = [('n'+str(i), DummyNode()) for i in range(8)]
    b.tick(peers, now=10.)
    for _, node in peers:
        assert any(t == 'generation-wire' and x['payload']['kind'] == 'proposal'
                   for t,x in node.wires)
    ledger.close()


def test_failed_send_not_consume_other_neighbors_or_late_join(tmp_path):
    b, ledger, _ = bridge(tmp_path)
    from patchlib.online_generation import make_proposal
    proposal = make_proposal(room='room', epoch=1, parent='0'*64,
        manifest='a'*64, semantic='b'*64, cut='c'*64,
        cutoffs={'peer01':5}, replicas=['a','b','c'])
    b.control.propose(proposal)
    a, c = DummyNode(), DummyNode()
    a.ready = False
    b.tick([('offline',a),('online',c)], now=10.)
    assert any(t == 'generation-wire' for t,_ in c.wires)
    a.ready = True
    b.tick([('offline',a),('online',c)], now=10.5)
    assert any(t == 'generation-wire' for t,_ in a.wires)
    assert sum(t=='generation-wire' and x['payload']['kind']=='proposal' for t,x in c.wires)==1
    ledger.close()


def test_no_neighbors_does_not_drain_durably_needed_proposal(tmp_path):
    b, ledger, _ = bridge(tmp_path)
    from patchlib.online_generation import make_proposal
    proposal = make_proposal(room='room', epoch=1, parent='0'*64,
        manifest='a'*64, semantic='b'*64, cut='c'*64,
        cutoffs={'peer01':5}, replicas=['a','b','c'])
    b.control.propose(proposal)
    b.tick([], now=1.)
    n = DummyNode()
    b.tick([('reconnect',n)], now=2.)
    assert any(t == 'generation-wire' for t,_ in n.wires)
    ledger.close()
