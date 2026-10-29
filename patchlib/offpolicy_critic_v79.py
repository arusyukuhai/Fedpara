"""Unclipped trajectory importance-sampling surrogate for a changing Actor.

Assumptions for unbiasedness IN EXPECTATION:
- Behavior log-probabilities correspond to the entire sampled action sequence.
- The target Actor's action probabilities are evaluated on the same prefixes.
- Both policies share the same action space / tokenizer and target has support
  wherever the behavior policy places positive mass (autoregressive softmax).
- Prompt distribution, terminal reward semantics, and environment dynamics
  remain fixed across behavior/target policies.
- Critic features are functions of state at the fixed target-policy snapshot.

For a trajectory tau=(a_0,...a_{T-1}) sampled from mu and terminal reward R,
state occupancy correction is w_{<t}=prod_{j<t} pi(a_j)/mu(a_j).
The whole-trajectory correction is W=prod_{j<T} pi(a_j)/mu(a_j).
Let G_t=gamma**(T-1-t)*R. Then this differentiable surrogate

  L_hat = (1/(2T)) sum_t [w_{<t} V(s_t)^2 - 2 W V(s_t) G_t + W G_t^2]

has expectation equal to 0.5 * E_pi[mean_t (V(s_t)-G_t)^2] and its
Critic gradient is unbiased under the assumptions. Ratio weights are detached.
It is NOT an exact finite-sample correction, nor an unbiased replay-based
PPO actor update. Zero clipping means potentially enormous variance; we fail
closed on numeric overflow rather than silently bias the estimator.
"""
from __future__ import annotations
import math
import torch


class OffPolicyCorrectionError(ValueError):
    pass


def resolved_mode(args):
    mode=getattr(args,'critic_offpolicy','auto')
    if mode not in ('auto','gae','trajectory_is'):
        raise OffPolicyCorrectionError('critic_offpolicy must be auto, gae or trajectory_is')
    return ('trajectory_is' if bool(getattr(args,'distributed_critic',False)) else 'gae') if mode=='auto' else mode


def exact_trajectory_value_surrogate(values, behavior_logp, target_logp, terminal_reward, gamma):
    if not isinstance(values,torch.Tensor) or values.ndim!=1 or values.numel()<1 or values.numel()>128:
        raise OffPolicyCorrectionError('Critic IS requires 1..128 state values')
    if not isinstance(behavior_logp,torch.Tensor) or not isinstance(target_logp,torch.Tensor):
        raise OffPolicyCorrectionError('IS requires behavior and target per-action log probabilities')
    if behavior_logp.shape!=values.shape or target_logp.shape!=values.shape:
        raise OffPolicyCorrectionError('IS actions and values have different lengths')
    if not (isinstance(terminal_reward,(float,int)) and math.isfinite(terminal_reward)):
        raise OffPolicyCorrectionError('IS requires a finite terminal reward')
    if not (isinstance(gamma,(int,float)) and math.isfinite(gamma) and 0<float(gamma)<=1):
        raise OffPolicyCorrectionError('IS requires gamma in (0,1]')
    with torch.no_grad():
        # Float64 reduces catastrophic cancellation of long log-likelihood
        # products. No ratio clipping or fake normalization can preserve the
        # exact target-policy gradient expectation.
        old=behavior_logp.detach().to(dtype=torch.float64)
        new=target_logp.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(old).all() and torch.isfinite(new).all()):
            raise OffPolicyCorrectionError('nonfinite behavior/target policy probabilities')
        log_ratios=new-old
        cumulative=torch.cumsum(log_ratios,dim=0)
        prefix=torch.cat((torch.zeros_like(cumulative[:1]),cumulative[:-1]))
        total=cumulative[-1]
        # The maximum exponent for finite float64 is ~709.78; float32 output
        # is validated below before entering the Critic optimizer.
        if bool((prefix.abs()>700).any()) or bool(total.abs()>700):
            raise OffPolicyCorrectionError('trajectory importance ratio overflow; resample on policy')
        occupancy=prefix.exp()
        all_actions=total.exp()
        powers=torch.arange(values.numel()-1,-1,-1,device=values.device,dtype=torch.float64)
        discounted=(float(gamma)**powers)*float(terminal_reward)
        square=occupancy * values.detach().double().square()
        cross=2.*all_actions*values.detach().double()*discounted
        constant=all_actions*discounted.square()
        if not bool(torch.isfinite(square-cross+constant).all()):
            raise OffPolicyCorrectionError('nonfinite importance-weighted value objective')
    # Keep weights as constants (no Actor gradients through the Critic
    # objective). Evaluate the Critic expression in double precision to avoid
    # overflow before validating its value and backward scale.
    v=values.double()
    result=(occupancy*v.square() - 2.*all_actions*discounted*v +
            all_actions*discounted.square()).mean()*0.5
    if not bool(torch.isfinite(result)):
        raise OffPolicyCorrectionError('nonfinite off-policy Critic objective')
    # Guard the actual *per-state* analytic derivative as well, so that a
    # finite scalar after cancellation cannot hide an exploding value update.
    grad=(occupancy*v.detach()-all_actions*discounted)/values.numel()
    if not bool(torch.isfinite(grad).all()):
        raise OffPolicyCorrectionError('nonfinite off-policy Critic gradient')
    if values.dtype in (torch.float16,torch.bfloat16,torch.float32):
        # Overflow when cast into the trainable head dtype would poison AdamW.
        limit=torch.finfo(values.dtype).max
        if bool((grad.abs()>limit).any()):
            raise OffPolicyCorrectionError('off-policy gradient exceeds value tensor dtype')
    return result, {'trajectory_ratio':float(all_actions),
                    'max_log_ratio':float(log_ratios.abs().max()),
                    'max_occupancy_ratio':float(occupancy.max()),
                    'offpolicy_mode':'trajectory_is'}
