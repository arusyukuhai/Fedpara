"""Regression and exact finite-MDP enumeration of full-trajectory IS Critic."""
import itertools
import math
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from patchlib.offpolicy_critic_v79 import (
    exact_trajectory_value_surrogate as value_is, OffPolicyCorrectionError, resolved_mode)
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, spec, MixedLearningError
from test_mixed_learning_v65 import make_args, setup_model, write, ToyTokenizer


def test_importance_weighted_critic_gradient_matches_target_exact_enumeration():
    # 2-step binary autoregressive policy. Target is VERY different from
    # behavior and rewards depend on both actions. Compute the true on-target
    # expected gradient over all four complete trajectories, not MC estimates.
    mu={(0,):.9,(1,):.1,(0,0):.8,(0,1):.2,(1,0):.3,(1,1):.7}
    pi={(0,):.2,(1,):.8,(0,0):.4,(0,1):.6,(1,0):.6,(1,1):.4}
    gamma=.93
    total_is=torch.zeros(3,dtype=torch.double)
    total_true=torch.zeros(3,dtype=torch.double)
    for a,b in itertools.product((0,1),repeat=2):
        p_mu=mu[(a,)]*mu[(a,b)]
        p_pi=pi[(a,)]*pi[(a,b)]
        old=torch.log(torch.tensor([mu[(a,)],mu[(a,b)]],dtype=torch.double))
        new=torch.log(torch.tensor([pi[(a,)],pi[(a,b)]],dtype=torch.double))
        reward=float(2*a+3*b-1)
        theta=torch.tensor([.4,-.5,.7],dtype=torch.double,requires_grad=True)
        values=torch.stack((theta[0],theta[1+a]))
        surrogate,details=value_is(values,old,new,reward,gamma)
        (p_mu*surrogate).backward()
        total_is+=theta.grad.detach()
        direct=torch.tensor([.4,-.5,.7],dtype=torch.double,requires_grad=True)
        target=.5*torch.stack([((direct[0]-gamma*reward)**2),
                                ((direct[1+a]-reward)**2)]).mean()
        (p_pi*target).backward()
        total_true+=direct.grad.detach()
        assert math.isclose(details['trajectory_ratio'],p_pi/p_mu,rel_tol=1e-12)
    torch.testing.assert_close(total_is,total_true,rtol=0,atol=1e-12)


def test_unshifted_policy_is_gradient_identical_to_onpolicy_monte_carlo():
    values=torch.tensor([.2,-.4,.7],dtype=torch.double,requires_grad=True)
    old=torch.tensor([-.4,-1.7,-.3],dtype=torch.double)
    surrogate,_=value_is(values,old,old.clone(),.8,.95)
    reference=.5*((values-torch.tensor([.95**2*.8,.95*.8,.8],dtype=torch.double))**2).mean()
    torch.testing.assert_close(surrogate,reference,atol=1e-12,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(surrogate,values)[0],
                               torch.autograd.grad(reference,values)[0])


def test_samplewise_suffix_only_is_not_sufficient_for_target_state_distribution():
    old=torch.log(torch.tensor([.9,.8],dtype=torch.double))
    new=torch.log(torch.tensor([.2,.4],dtype=torch.double))
    x=torch.tensor([.5,.6],dtype=torch.double,requires_grad=True)
    _,metrics=value_is(x,old,new,1.,.9)
    assert math.isclose(metrics['max_occupancy_ratio'],1.0)  # t=0 has occupancy weight 1
    assert math.isclose(metrics['trajectory_ratio'],(.2/.9)*(.4/.8))


def test_exact_trajectory_weights_never_silently_clipped_and_do_not_update_actor():
    v=torch.tensor([.3,.7],requires_grad=True)
    actor=torch.tensor([2.0,0.0],requires_grad=True)
    old=torch.tensor([-6.0,-1.0])
    surrogate,diag=value_is(v,old,actor,.5,.9)
    assert diag['trajectory_ratio']>1000
    surrogate.backward()
    assert v.grad is not None and v.grad.isfinite().all()
    assert actor.grad is None  # this is a Critic, NOT an Actor objective


def test_nonfinite_and_overflow_rejected_without_weight_clipping():
    x=torch.ones(2,requires_grad=True)
    a=torch.tensor([0.,0.])
    with pytest.raises(OffPolicyCorrectionError,match='overflow'):
        value_is(x,a,torch.tensor([400.,400.]),1.,1.)
    with pytest.raises(OffPolicyCorrectionError,match='nonfinite'):
        value_is(x,a,torch.tensor([0.,float('nan')]),1.,1.)
    with pytest.raises(OffPolicyCorrectionError,match='finite terminal reward'):
        value_is(x,a,a,float('inf'),1.)
    with pytest.raises(OffPolicyCorrectionError,match='different lengths'):
        value_is(x,a,a[:1],1.,1.)


def test_distributed_defaults_to_unclipped_is_and_policy_version_pinned(tmp_path):
    p=write(tmp_path,'ppo.jsonl',[{'prompt':'hi','reference':'there'}])
    args=make_args(tmp_path,ppo_jsonl=p,distributed_critic=True)
    assert resolved_mode(args)=='trajectory_is'
    policy=spec(args)
    assert policy['version']=='pt-sft-clipped-ppo-v79'
    assert policy['critic_offpolicy']=='unclipped-trajectory-is-mc-v79'
    assert spec(make_args(tmp_path,ppo_jsonl=p,distributed_critic=True,critic_offpolicy='gae')) != policy
    assert resolved_mode(make_args(tmp_path,ppo_jsonl=p))=='gae'
    with pytest.raises(MixedLearningError,match='critic_offpolicy'):
        from patchlib.mixed_learning_v65 import validate_args
        validate_args(make_args(tmp_path,ppo_jsonl=p,critic_offpolicy='invented'))


def test_actual_ppo_critic_target_recomputes_on_actor_update(tmp_path):
    p=write(tmp_path,'ppo.jsonl',[{'prompt':'hi','reference':'there'}])
    args=make_args(tmp_path,ppo_jsonl=p,ppo_reuse=2,distributed_critic=True)
    model,adapter,optim,_=setup_model()
    mix=MixedObjectiveTrainer(args,ToyTokenizer())
    try:
        torch.manual_seed(48)
        actor_loss,critic_loss,metrics=mix.ppo_loss(model,torch.device('cpu'))
        assert actor_loss.requires_grad and critic_loss.requires_grad
        assert metrics['offpolicy_mode']=='trajectory_is'
        assert math.isclose(metrics['trajectory_ratio'],1.,rel_tol=1e-5,abs_tol=1e-5)
        before=mix.cached
        critic_loss.backward(retain_graph=True)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in mix.critic.parameters())
        optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        optim.step()
        with torch.no_grad():
            for p in adapter.trainable_parameters():p.add_(.02)
        _,value_again,new_metrics=mix.ppo_loss(model,torch.device('cpu'))
        assert mix.cached is before
        assert math.isfinite(new_metrics['trajectory_ratio'])
        assert value_again.requires_grad
    finally:
        mix.close()


def test_gui_cli_config_roundtrip(tmp_path):
    from patchlib.mix_gui_v66 import normalize,command_options
    p=write(tmp_path,'ppo.jsonl',[{'prompt':'Hi','reference':'yes'}])
    data=normalize({'enabled':True,'ppo_jsonl':str(p),'distributed_critic':True,
                    'critic_offpolicy':'trajectory_is','sft_weight':0.0},task='causal-lm')
    opts=command_options(data,state_dir=tmp_path)
    assert '--critic-offpolicy' in opts
    assert opts[opts.index('--critic-offpolicy')+1]=='trajectory_is'
    assert '--distributed-critic' in opts


def test_hf_value_head_trains_with_trajectory_is_and_frozen_backbone(tmp_path,monkeypatch):
    from test_hf_value_critic_v69 import settings,hf_mock,Tokenizer,Actor
    hf_mock(monkeypatch)
    (tmp_path/'ppo.jsonl').write_text('{"prompt":"abc","reference":"abcd"}\n')
    args=settings(tmp_path,distributed_critic=True,critic_offpolicy='trajectory_is')
    actor=Actor()
    mixer=MixedObjectiveTrainer(args,Tokenizer())
    try:
        _,v,metrics=mixer.ppo_loss(actor,torch.device('cpu'))
        assert metrics['offpolicy_mode']=='trajectory_is'
        v.backward()
        assert mixer.critic.head.weight.grad is not None
        assert torch.isfinite(mixer.critic.head.weight.grad).all()
        assert all(p.grad is None for p in mixer.critic.model.body.parameters())
    finally:
        mixer.close()
