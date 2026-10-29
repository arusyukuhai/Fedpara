"""Authenticated replay, tampering, duplicate receipt, crash recovery, actor IS."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import pytest
import torch
from patchlib.core import PatchError
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, spec, validate_args, MixedLearningError
from patchlib.trajectory_gossip_v80 import TrajectoryGossip, validate_rollout, make_wire, parse_wire
from test_mixed_learning_v65 import make_args, setup_model, write, ToyTokenizer


def sample():
    return {'prompt':[3,4], 'tokens':[5,6], 'behavior_logp':[-1.,-2.],
            'reward':0.5,'reference':[5,7], 'reward_mode':'token_match'}

def crypt():
    def sign(length,digest):return 's:'+digest
    def verify(data):
        if data['sig']!='s:'+data['sha256']:raise PatchError('wrong signature')
    return sign,verify


def peers(tmp_path):
    data=write(tmp_path,'train.jsonl',[{'prompt':'Hi','reference':'Ag'}])
    args=make_args(tmp_path,ppo_jsonl=data,distributed_critic=True,
                   critic_rollout_gossip=True,ppo_reward='token_match',critic_offpolicy='trajectory_is')
    model,adapter,opt,_=setup_model()
    mix_a=MixedObjectiveTrainer(args,ToyTokenizer())
    model_b,adapter_b,opt_b,_=setup_model()
    args_b=make_args(tmp_path,ppo_jsonl=data,distributed_critic=True,
                   critic_rollout_gossip=True,ppo_reward='token_match',critic_offpolicy='trajectory_is')
    mix_b=MixedObjectiveTrainer(args_b,ToyTokenizer())
    sign,verify=crypt()
    a=TrajectoryGossip(tmp_path/'a.db','a'*64,mix_a,'alpha',sign,trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    b=TrajectoryGossip(tmp_path/'b.db','a'*64,mix_b,'beta',sign,trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    return a,b,model,model_b,mix_a,mix_b


def test_protocol_rejects_bad_reward_and_bad_probs():
    bad=sample();bad['reward']=.4
    with pytest.raises(PatchError,match='reward'):validate_rollout(bad)
    bad=sample();bad['behavior_logp']=[0.2,-2.]
    with pytest.raises(PatchError,match='probabilities'):validate_rollout(bad)


def test_wire_signed_payload_and_certified_actor_epoch():
    sign,verify=crypt();x=sample()
    w=make_wire('alpha',1,'a'*64,'b'*64,x,sign)
    with pytest.raises(PatchError,match='lineage'):
        parse_wire(w,'a'*64,{'a'*64},{'alpha':verify},reward_mode='token_match')
    assert parse_wire(w,'a'*64,{'b'*64},{'alpha':verify},reward_mode='token_match')[2]==x
    bad=bytearray(w);bad[-1]=bad[-1]^1
    with pytest.raises(PatchError):parse_wire(bytes(bad),'a'*64,{'b'*64},{'alpha':verify},reward_mode='token_match')


def test_real_receiver_recomputes_actor_and_persists_exactly_once(tmp_path):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        torch.manual_seed(23)
        actor_loss,closs,_=mixa.ppo_loss(ma,torch.device('cpu'))
        wire=a.local(mixa.export_rollout_v80())
        assert b.ingest(wire)
        assert not b.ingest(wire)
        assert b.pending() and len(b.pending())==1
        old=[p.detach().clone() for p in mixb.critic.parameters()]
        aparams=[p.detach().clone() for p in mb.parameters()]
        n=b.apply_pending(mb,torch.device('cpu'))
        assert n==1
        assert b.apply_pending(mb,torch.device('cpu'))==0
        assert any(not torch.equal(x,y) for x,y in zip(old,mixb.critic.parameters()))
        assert all(torch.equal(x,y) for x,y in zip(aparams,mb.parameters()))
        saved=[p.detach().clone() for p in mixb.critic.parameters()]
        step=mixb.critic_steps
        b.close()
        torch.manual_seed(555)
        # Simulate process death: old local objects discarded, a new
        # Critic starts at DIFFERENT random parameters and is restored.
        new_mixer=MixedObjectiveTrainer(mixb.args,ToyTokenizer())
        b=TrajectoryGossip(tmp_path/'b.db','a'*64,new_mixer,'beta',crypt()[0],
            trusted={'alpha':crypt()[1]},allowed_epochs={'a'*64},actor_epoch='a'*64)
        assert new_mixer.critic_steps==step
        assert all(torch.equal(x,y) for x,y in zip(saved,new_mixer.critic.parameters()))
        assert b.apply_pending(mb,torch.device('cpu'))==0
        new_mixer.close()
    finally:
        a.close();b.close();mixa.close();mixb.close()


def test_crash_after_event_before_value_step_recovers_pending(tmp_path):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        w=a.local(sample())
        assert b.ingest(w)
        b.close()
        b=TrajectoryGossip(tmp_path/'b.db','a'*64,mixb,'beta',crypt()[0],
            trusted={'alpha':crypt()[1]},allowed_epochs={'a'*64},actor_epoch='a'*64)
        assert len(b.pending())==1
        b.apply_pending(mb,torch.device('cpu'))
        assert not b.pending()
    finally:
        a.close();b.close();mixa.close();mixb.close()


def test_gap_repair_and_equivocation(tmp_path):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        first=a.local(sample());second=a.local(sample())
        b.ingest(second)
        assert b.need(a.frontiers())==[('alpha',1)]
        assert b.ingest(first)
        assert b.need(a.frontiers())==[]
        conflict=make_wire('alpha',1,a.genesis,'a'*64,
            dict(sample(),behavior_logp=[-2.,-2.]),crypt()[0])
        with pytest.raises(PatchError,match='equivocation'):b.ingest(conflict)
    finally:
        a.close();b.close();mixa.close();mixb.close()


def test_policy_and_gui_pin_v80(tmp_path):
    p=write(tmp_path,'ex.jsonl',[{'prompt':'abc','reference':'yes'}])
    args=make_args(tmp_path,ppo_jsonl=p,distributed_critic=True,critic_rollout_gossip=True)
    validate_args(args)
    assert spec(args)['critic_distribution']=='loha-complete-checkpoint-verified-trajectory-is-v82'
    assert spec(args)['version']=='pt-sft-clipped-ppo-v82'
    args.ppo_reward='hf_model'
    with pytest.raises(MixedLearningError,match='reference rewards'):validate_args(args)
    from patchlib.mix_gui_v66 import normalize, command_options
    data=normalize({'enabled':True,'ppo_jsonl':str(p),'distributed_critic':True,'critic_rollout_gossip':True,'sft_weight':0.0},task='causal-lm')
    assert '--critic-rollout-gossip' in command_options(data,state_dir=tmp_path)


def test_sqlite_failure_rolls_back_critic_and_leaves_unsigned_receipt_pending(tmp_path,monkeypatch):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        assert b.ingest(a.local(sample()))
        weights=[p.detach().clone() for p in mixb.critic.parameters()]
        old_persist=b.persist
        def fail_once(receipt=None):
            if receipt is not None:raise OSError('disk out of space')
            return old_persist(receipt)
        monkeypatch.setattr(b,'persist',fail_once)
        with pytest.raises(OSError,match='disk out of space'):
            b.apply_pending(mb,torch.device('cpu'))
        assert b.pending()
        assert all(torch.equal(x,y) for x,y in zip(weights,mixb.critic.parameters()))
        monkeypatch.setattr(b,'persist',old_persist)
        assert b.apply_pending(mb,torch.device('cpu'))==1
        assert not b.pending()
    finally:
        a.close();b.close();mixa.close();mixb.close()


def test_corrupt_checkpoint_blocks_restart(tmp_path):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        b.journal.db.execute("UPDATE trajectory_state SET digest=? WHERE id=1",('0'*64,))
        b.journal.db.commit()
        b.close()
        with pytest.raises(PatchError,match='corrupt local trajectory replay checkpoint'):
            TrajectoryGossip(tmp_path/'b.db','a'*64,mixb,'beta',crypt()[0],
                trusted={'alpha':crypt()[1]},allowed_epochs={'a'*64},actor_epoch='a'*64)
    finally:
        a.close();mixa.close();mixb.close()


def test_receiver_recomputes_probability_after_its_actor_changes(tmp_path):
    a,b,ma,mb,mixa,mixb=peers(tmp_path)
    try:
        d=sample()
        first=mixb.train_imported_rollout(mb,torch.device('cpu'),d)
        with torch.no_grad():
            # The source's old log probabilities remain unchanged; only the
            # RECEIVER Actor changes. The next Critic step must see that.
            mb.head.bias[5].add_(8.)
        second=mixb.train_imported_rollout(mb,torch.device('cpu'),d)
        assert abs(first['trajectory_ratio']-second['trajectory_ratio'])>1e-6
    finally:
        a.close();b.close();mixa.close();mixb.close()
