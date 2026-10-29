"""End-to-end source checkpoint replay, adversarial likelihoods, crashes/churn."""
import pytest
import torch
from patchlib.core import PatchError
from patchlib.behavior_checkpoint_v81 import capture, replay_logps, digest
from patchlib.trajectory_gossip_v81 import (VerifiedTrajectoryGossip, parse_rollout,
                                            rollout_wire, page_wire, parse_page)
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, spec
from test_mixed_learning_v65 import make_args, setup_model, write, ToyTokenizer
from test_trajectory_gossip_v80 import crypt


def peers(tmp_path):
    dataset=write(tmp_path,'ppo.jsonl',[{'prompt':'Hi','reference':'Ag'}])
    args=make_args(tmp_path,ppo_jsonl=dataset,distributed_critic=True,
                   critic_rollout_gossip=True,critic_offpolicy='trajectory_is')
    aa,bb=setup_model(),setup_model()
    model_a,adapter_a,_,_=aa
    model_b,adapter_b,_,_=bb
    mixer_a=MixedObjectiveTrainer(args,ToyTokenizer())
    args2=make_args(tmp_path,ppo_jsonl=dataset,distributed_critic=True,
                   critic_rollout_gossip=True,critic_offpolicy='trajectory_is')
    mixer_b=MixedObjectiveTrainer(args2,ToyTokenizer())
    sign,verify=crypt()
    a=VerifiedTrajectoryGossip(tmp_path/'a.db','a'*64,mixer_a,'alpha',sign,
        trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    b=VerifiedTrajectoryGossip(tmp_path/'b.db','a'*64,mixer_b,'beta',sign,
        trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    return a,b,model_a,model_b,mixer_a,mixer_b


def rollout(model,mix):
    torch.manual_seed(23)
    mix.ppo_loss(model,torch.device('cpu'))
    # Explicitly exercise the legacy v81 codec, which is no longer the live v82 protocol.
    return mix.export_rollout_v80(),capture(model)


def finish(a,b,ma,mb):
    a.close();b.close();ma.close();mb.close()


def test_checkpoint_reproduces_source_after_actor_update(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        expected=replay_logps(dst,ck,data,ma.args,torch.device('cpu'))
        assert max(abs(x-y) for x,y in zip(expected,data['behavior_logp']))<2e-3
        before=[p.clone() for p in dst.parameters()]
        with torch.no_grad():
            # The sampled source has already moved on; its immutable snapshot
            # must still replay the OLD probabilities at the receiver.
            src.head.bias[5].add_(0.4)
        frames=a.local(data,checkpoint=ck)
        assert frames[0].startswith(b'KCP81\0')
        assert frames[-1].startswith(b'KTR81\0')
        # Unchanged receiver weights even when old behavior Actor must be loaded.
        for wire in frames:b.ingest(wire)
        assert b.apply_pending(dst,torch.device('cpu'))==1
        assert all(torch.equal(p,x) for p,x in zip(dst.parameters(),before))
        assert b.apply_pending(dst,torch.device('cpu'))==0
    finally:finish(a,b,ma,mb)


def test_forged_claim_with_valid_signature_fails_before_value_update(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        malicious=dict(data,behavior_logp=[x-.65 for x in data['behavior_logp']])
        frames=a.local(malicious,checkpoint=ck)
        for wire in frames:b.ingest(wire)
        before=[p.clone() for p in mb.critic.parameters()]
        with pytest.raises(PatchError,match='probabilities disagree'):
            b.apply_pending(dst,torch.device('cpu'))
        assert b.pending()
        assert all(torch.equal(p,x) for p,x in zip(before,mb.critic.parameters()))
    finally:finish(a,b,ma,mb)


def test_missing_checkpoint_waits_and_restart_resumes_without_duplicate(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        frames=a.local(data,checkpoint=ck)
        b.ingest(frames[-1])
        assert b.apply_pending(dst,torch.device('cpu'))==0
        assert b.pending()
        b.close()
        b=VerifiedTrajectoryGossip(tmp_path/'b.db','a'*64,mb,'beta',crypt()[0],
            trusted={'alpha':crypt()[1]},allowed_epochs={'a'*64},actor_epoch='a'*64)
        for wire in reversed(frames[:-1]):b.ingest(wire)
        assert b.apply_pending(dst,torch.device('cpu'))==1
        assert not b.pending()
        assert b.apply_pending(dst,torch.device('cpu'))==0
    finally:finish(a,b,ma,mb)


def test_wrong_frozen_base_fails_and_restores_receiver_actor(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        frames=a.local(data,checkpoint=ck)
        for wire in frames:b.ingest(wire)
        with torch.no_grad():
            # source/recipient frozen base must match, not just trainable LoRA.
            dst.emb.weight[0,0].add_(2.0)
        before=[p.clone() for p in dst.parameters()]
        with pytest.raises(PatchError,match='frozen base differs'):
            b.apply_pending(dst,torch.device('cpu'))
        assert all(torch.equal(p,x) for p,x in zip(before,dst.parameters()))
        assert b.pending()
    finally:finish(a,b,ma,mb)


def test_forged_checkpoint_page_and_old_protocol_are_rejected(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        frames=a.local(data,checkpoint=ck)
        broken=bytearray(frames[0]);broken[-1]^=1
        with pytest.raises(PatchError):b.ingest(bytes(broken))
        with pytest.raises(PatchError,match='legacy'):
            b.ingest(b'KTR80\0junk')
        assert b.journal.count()==0
        # A signed but invalid checkpoint digest cannot impersonate an
        # uploaded snapshot of the source policy.
        fake=rollout_wire('alpha',99,a.genesis,'a'*64,data,'f'*64,crypt()[0])
        b.ingest(fake)
        assert b.apply_pending(dst,torch.device('cpu'))==0
    finally:finish(a,b,ma,mb)


def test_oversized_model_checkpoint_rejected_and_policy_segregated(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        from patchlib.behavior_checkpoint_v81 import MAX_CHECKPOINT_BYTES
        assert spec(ma.args)['version']=='pt-sft-clipped-ppo-v82'
        assert spec(ma.args)['behavior_policy_proof'].startswith('content-addressed')
        with pytest.raises(PatchError,match='checkpoint'):
            a.local({'prompt':[3,4],'tokens':[5], 'behavior_logp':[-1],
                     'reward':0.,'reference':[7], 'reward_mode':'token_match'},checkpoint=None)
        # Any new v81 rollout frame must contain a checkpoint digest.
        data,_=rollout(src,ma)
        with pytest.raises(PatchError,match='missing behavior'):
            rollout_wire('alpha',1,a.genesis,'a'*64,data,'invalid',crypt()[0])
    finally:finish(a,b,ma,mb)


def test_multiframe_checkpoint_out_of_order_and_rebuilt_index(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        # Add a trainable tensor to both identical models so snapshot spans
        # multiple SCTP-safe pages. Its output need not depend on this tensor.
        from torch import nn
        src.extra=nn.Parameter(torch.zeros(13000))
        dst.extra=nn.Parameter(torch.zeros(13000))
        data,ck=rollout(src,ma)
        frames=a.local(data,checkpoint=ck)
        assert len(frames)>=3
        b.ingest(frames[-1])
        b.ingest(frames[-2])
        assert b.apply_pending(dst,torch.device('cpu'))==0
        for f in frames[:-2]:b.ingest(f)
        # Deliberately lose SQLite's secondary index but retain EventJournal.
        with b.journal.db:
            b.journal.db.execute('DELETE FROM v81_pages')
        b.close()
        b=VerifiedTrajectoryGossip(tmp_path/'b.db','a'*64,mb,'beta',crypt()[0],
            trusted={'alpha':crypt()[1]},allowed_epochs={'a'*64},actor_epoch='a'*64)
        assert b.apply_pending(dst,torch.device('cpu'))==1
    finally:finish(a,b,ma,mb)


def test_checkpoint_page_conflict_and_actor_epoch_fence(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        frames=a.local(data,checkpoint=ck)
        duplicate=page_wire('alpha',100,a.genesis,'a'*64,ck,0,crypt()[0])
        b.ingest(frames[0])
        with pytest.raises(PatchError,match='conflicting'):
            b.ingest(duplicate)
        invalid_epoch=page_wire('alpha',101,a.genesis,'f'*64,ck,0,crypt()[0])
        with pytest.raises(PatchError,match='certified'):
            b.ingest(invalid_epoch)
    finally:finish(a,b,ma,mb)


def test_repeat_policy_content_addressed_no_retransmission(tmp_path):
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        data,ck=rollout(src,ma)
        first=a.local(data,checkpoint=ck)
        again=a.local(data,checkpoint=ck)
        assert len(first)>=2
        assert len(again)==1 and again[0].startswith(b'KTR81\0')
        for f in first+again:b.ingest(f)
        assert b.apply_pending(dst,torch.device('cpu'),limit=2)==2
        assert not b.pending()
    finally:finish(a,b,ma,mb)


def test_elastic_training_step_exports_checkpoint_from_sampling_time(tmp_path):
    from patchlib.cli import run_step
    a,b,src,dst,ma,mb=peers(tmp_path)
    try:
        _,adapter,optimizer,batch=setup_model()
        # Need the same model+adapter pair for run_step and checkpoint capture.
        model,adapter,optimizer,batch=setup_model()
        ma.args.mode='elastic'
        ma.args._v65_mixer=ma
        run_step(model,adapter,optimizer,batch,ma.args)
        result=getattr(ma.args,'_v80_rollout_pending',None)
        assert result is not None
        sample,blob=result
        from patchlib.behavior_checkpoint_v82 import replay_logps as replay_v82
        replay_v82(model,blob,sample,ma.args,torch.device('cpu'))
    finally:finish(a,b,ma,mb)
