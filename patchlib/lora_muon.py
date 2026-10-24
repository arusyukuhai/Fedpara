"""LoRA-Muon (Cesista et al., arXiv:2606.12921, Algorithm 1).

Matrix factors A:[m,r], B:[n,r], effective update = (alpha/r) A B.T.
The QR-free spectral step is taken in factor-tangent coordinates and requires
only first-moment optimizer state. Gram inverse square roots use small (r x r)
PSD eigendecompositions for numerical robustness. Matrix signs default to the
paper's 8-stage Polar Express Newton-Schulz polynomial; no large SVD/QR.

LoRA-Muon is defined for LoRA, NOT the Hadamard product of two LoRA updates
(LoHa). Do not represent the latter as the paper's algorithm.
"""
from __future__ import annotations

import math
import torch

POLAR_EXPRESS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
)


def matrix_sign(x: torch.Tensor, *, method: str = 'newton-schulz') -> torch.Tensor:
    """Return UV.T for rectangular matrix x (paper Algorithm 3)."""
    if x.ndim != 2:
        raise ValueError('matrix_sign expects a matrix')
    z=x.float()
    if not torch.isfinite(z).all().item():
        raise FloatingPointError('nonfinite LoRA-Muon momentum')
    norm=torch.linalg.vector_norm(z)
    if norm.item() <= 1e-20:
        return torch.zeros_like(z)
    if method == 'svd':
        u, _, vh = torch.linalg.svd(z, full_matrices=False)
        return u @ vh
    if method != 'newton-schulz':
        raise ValueError('unsupported matrix-sign algorithm')
    transposed=z.shape[0] > z.shape[1]
    if transposed:
        z=z.T
    z=z / (norm + 1e-20)
    for a,b,c in POLAR_EXPRESS:
        gram=z @ z.T
        z=a*z + (b*gram + c*(gram@gram))@z
    return z.T if transposed else z


def gram_inverse_root(x: torch.Tensor, *, relative_epsilon: float=1e-5) -> torch.Tensor:
    """Stable PSD inverse square root with a *rank-sized* symmetric eigh.

    Equivalent inverse-root operation to Algorithm 1, but unlike Appendix B.4
    this implementation uses an exact eigensolver rather than its NS kernel.
    Damping makes a rank-deficient/zero-initialized LoRA factor well-defined.
    """
    g=x.float().T @ x.float()
    g=(g+g.T)*.5
    scale=torch.linalg.vector_norm(g).clamp_min(1e-20)
    eig, q = torch.linalg.eigh(g / scale)
    inv=(eig.clamp_min(0) + relative_epsilon).rsqrt()
    result=(q * inv.unsqueeze(0)) @ q.T / torch.sqrt(scale)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError('nonfinite inverse Gram root')
    return result


class LoRAMuon(torch.optim.Optimizer):
    """Paired low-rank optimizer implementing paper Algorithm 1.

    All model parameters supplied must belong to rank-r 2D LoRA factor pairs.
    Dense fallback / LoHa parts are rejected rather than silently updating
    them under a different algorithm. Parameter order derives from adapter's
    group.local.parts in the same order as the adapter itself.
    """
    def __init__(self, adapter, lr=0.01, momentum=.9, weight_decay=0.,
                 orthogonalizer='newton-schulz', inverse_root_eps=1e-5):
        if adapter.kind != 'lora':
            raise ValueError('LoRA-Muon requires --kind lora; LoHa is a different manifold')
        if not (0 < lr <= 1 and 0 <= momentum < 1 and 0 <= weight_decay and
                lr*weight_decay < 1 and 0 < inverse_root_eps < 1):
            raise ValueError('invalid LoRA-Muon optimizer parameters')
        if orthogonalizer not in ('newton-schulz','svd'):
            raise ValueError('invalid orthogonalizer')
        self.pairs=[]
        parameters=[]
        for group in adapter.groups:
            delta=group.overlay.local
            if delta.dense or len(delta.shape)!=2 or len(delta.parts)!=2:
                raise ValueError(f'LoRA-Muon only supports rank-r matrix factors: {group.canonical}; '
                                 'select --target-scope gpt_neo_linear')
            a,b=delta.parts
            if a.shape[1] != b.shape[1] or a.shape[1] != adapter.rank:
                raise ValueError('invalid LoRA factors')
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                raise ValueError('LoRA-Muon needs float32 factors')
            parameters.extend([a,b]);self.pairs.append((a,b))
        if not self.pairs:
            raise ValueError('no LoRA matrix pairs for Muon')
        super().__init__(parameters, defaults=dict(lr=lr, momentum=momentum,
            weight_decay=weight_decay, orthogonalizer=orthogonalizer,
            inverse_root_eps=inverse_root_eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        settings=self.param_groups[0]
        lr=settings['lr'];momentum=settings['momentum'];decay=settings['weight_decay']
        if not 0 < lr <= 1 or lr*decay >= 1:
            raise ValueError('invalid scheduled Muon learning rate')
        scale=math.sqrt(1-lr*decay)
        for a,b in self.pairs:
            if a.grad is None and b.grad is None:
                continue
            ga=torch.zeros_like(a) if a.grad is None else a.grad.detach().float()
            gb=torch.zeros_like(b) if b.grad is None else b.grad.detach().float()
            if not torch.isfinite(ga).all().item() or not torch.isfinite(gb).all().item():
                raise FloatingPointError('nonfinite LoRA-Muon gradients')
            state_a=self.state[a];state_b=self.state[b]
            ma=state_a.setdefault('momentum',torch.zeros_like(a))
            mb=state_b.setdefault('momentum',torch.zeros_like(b))
            ma.mul_(momentum).add_(ga,alpha=1-momentum)
            mb.mul_(momentum).add_(gb,alpha=1-momentum)
            # Same-step old factor values: do not update one factor before
            # computing the other's tangent-space spectral direction.
            ra=gram_inverse_root(a,relative_epsilon=settings['inverse_root_eps'])
            rb=gram_inverse_root(b,relative_epsilon=settings['inverse_root_eps'])
            da=(-.5*lr)*(matrix_sign(ma@rb,method=settings['orthogonalizer'])@rb)
            db=(-.5*lr)*(matrix_sign(mb@ra,method=settings['orthogonalizer'])@ra)
            if not torch.isfinite(da).all().item() or not torch.isfinite(db).all().item():
                raise FloatingPointError('nonfinite LoRA-Muon update')
            a.mul_(scale).add_(da,alpha=1/scale)
            b.mul_(scale).add_(db,alpha=1/scale)
            state_a['step']=state_a.get('step',0)+1
            state_b['step']=state_b.get('step',0)+1
        return loss


def make_adapter_optimizer(adapter, *, optimizer='adamw', lr=1e-4,
                           weight_decay=0., muon_momentum=.9):
    if optimizer=='adamw':
        return torch.optim.AdamW(list(adapter.trainable_parameters()),lr=lr,weight_decay=weight_decay)
    if optimizer=='lora-muon':
        return LoRAMuon(adapter,lr=lr,weight_decay=weight_decay,momentum=muon_momentum)
    raise ValueError('unknown adapter optimizer')
