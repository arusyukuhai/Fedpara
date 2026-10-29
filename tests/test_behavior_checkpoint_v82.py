"""LoHA-specific behavior distributions across diverged peer overlay states."""
import pytest
import torch
from patchlib.core import PatchError, UniversalAdapter, FrozenDelta
from patchlib.behavior_checkpoint_v82 import capture, replay_logps, temporarily_loaded
from patchlib.trajectory_gossip_v82 import LohaVerifiedTrajectoryGossip
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, spec
from test_mixed_learning_v65 import SmallCausalLM, ToyTokenizer, make_args, write
from test_trajectory_gossip_v80 import crypt


def pair(tmp_path):
    data=write(tmp_path,'s.jsonl',[{'prompt':'Hi','reference':'Ag'}])
    args=make_args(tmp_path,ppo_jsonl=data,distributed_critic=True,critic_rollout_gossip=True,
                   critic_offpolicy='trajectory_is')
    def build():
        torch.manual_seed(214)
        model=SmallCausalLM()
        adapter=UniversalAdapter(model,'toy','rev',kind='loha',rank=2,target_scope='linear')
        return model,adapter
    src,asrc=build();dst,adst=build()
    ma=MixedObjectiveTrainer(args,ToyTokenizer())
    mb=MixedObjectiveTrainer(args,ToyTokenizer())
    sign,verify=crypt()
    a=LohaVerifiedTrajectoryGossip(tmp_path/'a.db','a'*64,ma,'alpha',sign,
        trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    b=LohaVerifiedTrajectoryGossip(tmp_path/'b.db','a'*64,mb,'beta',sign,
        trusted={'alpha':verify,'beta':verify},allowed_epochs={'a'*64},actor_epoch='a'*64)
    return src,asrc,dst,adst,ma,mb,a,b


def attach_history(adapter,peer,scale):
    for g in adapter.groups:
        parts=[torch.full_like(p,scale) for p in g.overlay.local.parts]
        g.overlay.install(peer,parts)
        g.overlay.anchor=FrozenDelta(g.overlay.local,[p.detach().clone()*scale for p in g.overlay.local.parts])


def test_loha_old_actor_replays_across_peer_remote_and_anchor_divergence(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        attach_history(asrc,'alpha-before',.5)
        attach_history(adst,'unrelated',.05)
        torch.manual_seed(20)
        ma.ppo_loss(src,torch.device('cpu'))
        data=ma.export_rollout_v80()
        blob=ma.cached['behavior_checkpoint_v82']
        assert spec(ma.args)['version']=='pt-sft-clipped-ppo-v82'
        original_params=[p.clone() for p in dst.parameters()]
        original_overlay=[(g.overlay.anchor,g.overlay.remotes) for g in adst.groups]
        # Sender can discard live old adapter immediately after taking rollout.
        with torch.no_grad():
            for p in asrc.trainable_parameters():p.add_(.1)
        verified=replay_logps(dst,blob,data,ma.args,torch.device('cpu'))
        assert max(abs(x-y) for x,y in zip(verified,data['behavior_logp']))<2e-3
        assert all(torch.equal(p,was) for p,was in zip(dst.parameters(),original_params))
        assert all((g.overlay.anchor is anchor and g.overlay.remotes is remotes)
                   for g,(anchor,remotes) in zip(adst.groups,original_overlay))
        frames=a.local(data,checkpoint=blob)
        assert frames[0].startswith(b'KCP82\0')
        assert frames[-1].startswith(b'KTR82\0')
        for frame in reversed(frames):b.ingest(frame)
        assert b.apply_pending(dst,torch.device('cpu'))==1
        assert b.apply_pending(dst,torch.device('cpu'))==0
    finally:
        a.close();b.close();ma.close();mb.close()


def test_loha_factor_forgery_and_missing_page_rejected(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        attach_history(asrc,'owner',.4)
        torch.manual_seed(8);ma.ppo_loss(src,torch.device('cpu'))
        data=ma.export_rollout_v80();blob=ma.cached['behavior_checkpoint_v82']
        modified=dict(data,behavior_logp=[x-.9 for x in data['behavior_logp']])
        frames=a.local(modified,checkpoint=blob)
        b.ingest(frames[-1]);assert b.apply_pending(dst,torch.device('cpu'))==0
        for frame in frames[:-1]:b.ingest(frame)
        before=[p.clone() for p in mb.critic.parameters()]
        with pytest.raises(PatchError,match='probabilities disagree'):
            b.apply_pending(dst,torch.device('cpu'))
        assert all(torch.equal(p,x) for p,x in zip(mb.critic.parameters(),before))
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_cross_epoch_base_change_fails_closed(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        attach_history(asrc,'first',.2)
        torch.manual_seed(8);ma.ppo_loss(src,torch.device('cpu'))
        data=ma.export_rollout_v80();blob=ma.cached['behavior_checkpoint_v82']
        with torch.no_grad():dst.emb.weight[0,0].add_(.2)
        with pytest.raises(PatchError,match='frozen base differs'):
            replay_logps(dst,blob,data,ma.args,torch.device('cpu'))
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_replay_restores_state_after_exception(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        attach_history(asrc,'first',.1)
        blob=capture(src)
        old_params=[p.detach().clone() for p in dst.parameters()]
        old_overlay=[(g.overlay.anchor if hasattr(g.overlay,'anchor') else None,
                      g.overlay.remotes) for g in adst.groups]
        with pytest.raises(ZeroDivisionError):
            with temporarily_loaded(dst,blob):
                assert all(hasattr(g.overlay,'anchor') for g in adst.groups)
                assert all(len(g.overlay.remotes)==1 for g in adst.groups)
                1/0
        assert all(torch.equal(p,old) for p,old in zip(dst.parameters(),old_params))
        assert all((getattr(g.overlay,'anchor',None) is anc and g.overlay.remotes is rem)
                   for g,(anc,rem) in zip(adst.groups,old_overlay))
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_checkpoint_rejects_any_old_checkpoint_layout(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        from patchlib.behavior_checkpoint_v81 import capture as legacy
        # Old replay proof does not freeze remote contributions, so mixed versions
        # are refused rather than trusted as equivalent.
        blob=legacy(src)
        with pytest.raises(PatchError,match='metadata'):
            with temporarily_loaded(dst,blob):pass
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_cross_epoch_replay_with_pinned_frozen_base_cas(tmp_path):
    from patchlib.behavior_checkpoint_v82 import capture_frozen_base, FrozenBaseCAS
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        attach_history(asrc,'older-source',.1)
        torch.manual_seed(3);ma.ppo_loss(src,torch.device('cpu'))
        data=ma.export_rollout_v80()
        blob=ma.cached['behavior_checkpoint_v82']
        hash_before,base_blob=capture_frozen_base(src)
        store=FrozenBaseCAS(tmp_path/'historical')
        store.put(hash_before,base_blob)
        # Receiver has newer frozen weights and different LoHA history.
        with torch.no_grad():
            dst.emb.weight[0,0].add_(.23)
            dst.head.bias[0].add_(.21)
        attach_history(adst,'different-peer',.02)
        before_params=[p.clone() for p in dst.parameters()]
        before_buffers=[p.clone() for p in dst.buffers()]
        logps=replay_logps(dst,blob,data,ma.args,torch.device('cpu'),base_resolver=store)
        assert max(abs(x-y) for x,y in zip(logps,data['behavior_logp']))<2e-3
        assert all(torch.equal(p,q) for p,q in zip(dst.parameters(),before_params))
        assert all(torch.equal(p,q) for p,q in zip(dst.buffers(),before_buffers))
        frames=a.local(data,checkpoint=blob)
        b.base_resolver=store
        for frame in frames:b.ingest(frame)
        assert b.apply_pending(dst,torch.device('cpu'))==1
        assert b.apply_pending(dst,torch.device('cpu'))==0
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_historical_base_wrong_commitment_is_rejected(tmp_path):
    from patchlib.behavior_checkpoint_v82 import capture_frozen_base, FrozenBaseCAS
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        sha,blob=capture_frozen_base(src)
        store=FrozenBaseCAS(tmp_path/'history')
        with pytest.raises(PatchError,match='digest mismatch'):
            store.put('a'*64,blob)
        store.put(sha,blob)
        with pytest.raises(PatchError,match='digest mismatch'):
            store.put(sha,blob[:-1]+bytes([blob[-1]^1]))
    finally:a.close();b.close();ma.close();mb.close()


def test_loha_disk_virtual_base_requires_separate_certified_snapshot(tmp_path):
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        src.linear._fedpara_disk_path='private-layer-cow'
        with pytest.raises(PatchError,match='certified virtual/disk base'):
            capture(src)
    finally:a.close();b.close();ma.close();mb.close()


def test_mixed_loha_lora_auto_scopes_and_heterogeneous_eval_flags(tmp_path):
    torch.manual_seed(29)
    model=SmallCausalLM()
    adapter=UniversalAdapter(model,'toy','replay',kind='loha',rank=2,
                             target_scope='auto',wide_lora_rank=2)
    assert any(g.overlay.local.kind=='loha' for g in adapter.groups)
    assert any(g.overlay.local.kind=='lora' for g in adapter.groups)
    attach_history(adapter,'alpha',.2)
    model.train();model.emb.eval()
    old_modes=[m.training for m in model.modules()]
    blob=capture(model)
    # Different original model states must be restored after replay, not all train().
    with temporarily_loaded(model,blob):
        assert not model.training
        assert all(not m.training for m in model.modules())
    assert [m.training for m in model.modules()]==old_modes


def test_loha_cli_live_mixed_training_exports_replayable_v82_checkpoint(tmp_path):
    from patchlib.cli import run_step
    src,asrc,dst,adst,ma,mb,a,b=pair(tmp_path)
    try:
        ma.args.mode='elastic';ma.args._v65_mixer=ma
        optimizer=torch.optim.AdamW(list(asrc.trainable_parameters()),lr=.005)
        batch={'input_ids':torch.tensor([[3,4,5]],dtype=torch.long),
               'labels':torch.tensor([[3,4,5]],dtype=torch.long)}
        run_step(src,asrc,optimizer,batch,ma.args)
        data,blob=ma.args._v80_rollout_pending
        from patchlib.behavior_checkpoint_v82 import replay_logps as verify
        verify(dst,blob,data,ma.args,torch.device('cpu'))
    finally:a.close();b.close();ma.close();mb.close()
