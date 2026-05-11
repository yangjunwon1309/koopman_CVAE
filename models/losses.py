"""
losses.py — KODAQ v5 Loss Functions
=====================================

v5 additions on top of v4:
  L_reward : Categorical CE with Two-Hot encoding over [0,4], B=16 bins
  L_Q      : Categorical CE with Two-Hot target = Bellman TD target
             y_t = r_t + γ · E[Q̄(o_{t+1}, π(o_{t+1}))]
  L_pi     : Policy prior loss (TD-MPC2 style)
             L_π = ((entropy_coef · log_π - Q) * rho).mean()
  rho      : Moving 5th/95th percentile normalization of Q values (DreamerV3/TD-MPC2)

Bin design (accumulated reward, no symlog):
  [0, 4] → 16 bins, width = 4/15 ≈ 0.267
  bins[i] = i * 4.0 / (B-1)  for i=0..B-1
  Two-Hot interpolates between adjacent bins for non-integer values.

Existing v4 losses unchanged.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# v4 utilities (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def matrix_inv(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "inv"):
        return torch.linalg.inv(x)
    return torch.inverse(x)


def blend_koopman(log_lambdas, thetas, G_k, U, w):
    r_bar = torch.einsum('bk,km->bm', w, log_lambdas)
    t_bar = torch.einsum('bk,km->bm', w, thetas)
    r_exp = torch.exp(r_bar)
    lambda_real = r_exp * torch.cos(t_bar)
    U_inv = matrix_inv(U)
    Lam   = torch.diag_embed(lambda_real)
    A_bar = U.unsqueeze(0) @ Lam @ U_inv.unsqueeze(0)
    G_mix = torch.einsum('bk,kmd->bmd', w, G_k)
    B_bar = U.unsqueeze(0) @ G_mix
    return A_bar, B_bar, r_bar, t_bar


def koopman_step(o, u, A_bar, B_bar):
    return (A_bar @ o.unsqueeze(-1)).squeeze(-1) + \
           (B_bar @ u.unsqueeze(-1)).squeeze(-1)


def reconstruction_loss(preds, targets, weights):
    per_head = {}
    total    = torch.tensor(0.0, device=next(iter(preds.values())).device)
    for key in preds:
        if key not in targets:
            continue
        p = preds[key]
        t = targets[key]
        w = weights.get(key, 1.0)
        if key == 'reward':
            loss = F.binary_cross_entropy_with_logits(p, t.float())
        else:
            loss = F.mse_loss(p, t)
        per_head[key] = loss
        total = total + w * loss
    return total, per_head


def koopman_consistency_loss(mu_next, o_pred):
    return F.mse_loss(mu_next, o_pred)


def multistep_koopman_consistency_loss(
    mu_seq, o_seq, A_bar_seq, B_bar_seq, u_seq, H=4, alpha=0.95
):
    B, T, m = mu_seq.shape
    T_seq   = T - 1
    if H > T_seq:
        H = T_seq
    total_loss   = torch.tensor(0.0, device=mu_seq.device)
    total_weight = 0.0
    for t in range(T_seq - H + 1):
        z_hat = o_seq[:, t]
        for k in range(1, H + 1):
            step = t + k - 1
            if step >= T_seq:
                break
            A_t   = A_bar_seq[:, step]
            B_t   = B_bar_seq[:, step]
            u_t   = u_seq[:, step]
            z_hat = (A_t @ z_hat.unsqueeze(-1)).squeeze(-1) \
                  + (B_t @ u_t.unsqueeze(-1)).squeeze(-1)
            mu_target    = mu_seq[:, t + k]
            weight       = alpha ** k
            step_loss    = F.mse_loss(z_hat, mu_target)
            total_loss   = total_loss + weight * step_loss
            total_weight += weight
    if total_weight > 0:
        total_loss = total_loss / total_weight
    return total_loss


def skill_classification_loss(logits, labels, mask=None):
    B, T, K = logits.shape
    logits_flat = logits.reshape(B * T, K)
    labels_flat = labels.reshape(B * T).long()
    if mask is not None:
        valid       = mask.reshape(B * T)
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
    return F.cross_entropy(logits_flat, labels_flat)


def posterior_regularization_loss(mu_t, o_pred):
    target = o_pred.detach()
    return F.mse_loss(mu_t, target)


def eigenvalue_stability_loss(log_lambdas, margin=0.0):
    excess = F.relu(log_lambdas - margin)
    return (excess ** 2).mean()


def compute_total_loss(
    loss_rec, loss_dyn, loss_skill, loss_reg, loss_stab,
    lambda1, lambda2, lambda3, lambda4, phase,
):
    total = loss_rec
    if phase >= 2:
        total = total + lambda1 * loss_dyn + lambda2 * loss_skill
    if phase >= 3:
        total = total + lambda3 * loss_reg
    total = total + lambda4 * loss_stab
    return total, {
        'rec':   1.0,
        'dyn':   lambda1 if phase >= 2 else 0.0,
        'skill': lambda2 if phase >= 2 else 0.0,
        'reg':   lambda3 if phase >= 3 else 0.0,
        'stab':  lambda4,
    }


# ──────────────────────────────────────────────────────────────────────────────
# v5: Two-Hot Encoding / Decoding  [0, 4], B bins
# ──────────────────────────────────────────────────────────────────────────────

def get_bins(num_bins: int, v_min: float = 0.0, v_max: float = 4.0,
             device=None) -> torch.Tensor:
    """
    Bin centers uniformly spaced in [v_min, v_max].
    bins[i] = v_min + i * (v_max - v_min) / (num_bins - 1)
    """
    return torch.linspace(v_min, v_max, num_bins, device=device)


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Two-Hot encoding of scalar(s) x into B-dim soft label.

    x:    (...,)        scalar targets
    bins: (B,)          bin centers, monotonically increasing

    Returns: (..., B)   soft labels, sum=1 (mostly sparse, at most 2 non-zero)

    Algorithm:
      lower_idx = clamp(searchsorted(bins, x) - 1, 0, B-2)
      upper_idx = lower_idx + 1
      upper_w   = (x - bins[lower_idx]) / (bins[upper_idx] - bins[lower_idx])
      lower_w   = 1 - upper_w
      label[lower_idx] = lower_w,  label[upper_idx] = upper_w
    """
    B   = bins.shape[0]
    dev = x.device

    # searchsorted returns index where x would be inserted to keep sorted order
    # shape: same as x
    idx = torch.searchsorted(bins.contiguous(), x.contiguous())
    idx = idx.clamp(1, B - 1)          # upper bin index, in [1, B-1]
    lower_idx = idx - 1                 # lower bin index, in [0, B-2]

    b_lo = bins[lower_idx]              # (...,)
    b_hi = bins[idx]                    # (...,)
    span = (b_hi - b_lo).clamp(min=1e-8)

    upper_w = ((x - b_lo) / span).clamp(0.0, 1.0)  # (...,)
    lower_w = 1.0 - upper_w                          # (...,)

    # Scatter into B-dim vector
    label = torch.zeros(*x.shape, B, device=dev, dtype=x.dtype)
    label.scatter_(-1, lower_idx.unsqueeze(-1), lower_w.unsqueeze(-1))
    label.scatter_(-1, idx.unsqueeze(-1),       upper_w.unsqueeze(-1))
    return label                        # (..., B)


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Decode categorical logits → scalar via E[bin_center].

    logits: (..., B)  raw (unnormalized) or probabilities
    bins:   (B,)

    Returns: (...,)  scalar expected value
    """
    probs = torch.softmax(logits, dim=-1)           # (..., B)
    return (probs * bins.to(logits.device)).sum(-1) # (...,)


# ──────────────────────────────────────────────────────────────────────────────
# v5: Categorical Reward Loss  (Two-Hot CE)
# ──────────────────────────────────────────────────────────────────────────────

def reward_categorical_loss(
    logits:  torch.Tensor,   # (..., B)  reward head output
    targets: torch.Tensor,   # (...,)    step reward ∈ {0, 1}
    bins:    torch.Tensor,   # (B,)      covers [v_min, v_max]
) -> torch.Tensor:
    """
    L_R = CE(reward_logits, TwoHot(r_t))

    targets are step rewards ∈ {0, 1}.
    bins cover [0, v_max] where v_max ≥ 1, so no clamping needed:
    two_hot_encode handles boundary values via searchsorted clamp internally.
    """
    label = two_hot_encode(targets, bins)   # (..., B)
    log_p = F.log_softmax(logits, dim=-1)
    return -(label * log_p).sum(-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# v5: Categorical Q Loss  (Bellman TD, Two-Hot CE)
# ──────────────────────────────────────────────────────────────────────────────

def q_categorical_loss(
    q_logits:        torch.Tensor,   # (B_batch, T-1, num_q, num_bins)
    reward_seq:      torch.Tensor,   # (B_batch, T-1)   step reward (used in 1-step mode)
    q_target_scalar: torch.Tensor,   # (B_batch, T-1)   pre-computed TD target scalar
    bins:            torch.Tensor,   # (num_bins,)       covers [v_min, v_max]
    gamma:           float = 0.99,
) -> torch.Tensor:
    """
    TD target = reward_seq + gamma * q_target_scalar  (element-wise)

    1-step mode:  reward_seq = r_t,  q_target_scalar = Q_bar(o_{t+1}, pi)
                  gamma = cfg.gamma  (e.g. 0.99)

    H-step mode:  reward_seq = zeros,  q_target_scalar = pre-computed H-step return
                  gamma = 1.0  (discount already folded into the return)

    No clamp: caller is responsible for ensuring target is within [v_min, v_max].
    """
    with torch.no_grad():
        td_target = reward_seq + gamma * q_target_scalar      # (B, T-1)
        label     = two_hot_encode(td_target, bins.to(td_target.device))  # (B, T-1, B)

    # q_logits: (B, T-1, num_q, num_bins)
    num_q = q_logits.shape[2]
    label_expanded = label.unsqueeze(2).expand_as(q_logits)   # (B, T-1, num_q, B)

    log_p = F.log_softmax(q_logits, dim=-1)
    loss  = -(label_expanded * log_p).sum(-1).mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# v5: Moving Percentile Scale  (DreamerV3 / TD-MPC2 rho)
# ──────────────────────────────────────────────────────────────────────────────

class MovingPercentileScale:
    """
    Tracks EMA of 5th and 95th percentile of Q values.
    Used to normalize policy loss magnitude so that entropy_coef
    remains task-invariant regardless of Q scale.

    rho = 1 / max(1, S)   where S = EMA(95th) - EMA(5th)

    Usage:
        scale_tracker = MovingPercentileScale()
        ...
        scale_tracker.update(q_values.detach())
        rho = scale_tracker.rho
        pi_loss = ((entropy_coef * log_pi - q_pi) * rho).mean()
    """

    def __init__(self, decay: float = 0.99, lo: float = 0.05, hi: float = 0.95):
        self.decay = decay
        self.lo    = lo
        self.hi    = hi
        self._ema_lo: Optional[float] = None
        self._ema_hi: Optional[float] = None

    def update(self, values: torch.Tensor):
        """values: any shape tensor of Q values."""
        v = values.detach().float().flatten()
        if v.numel() < 2:
            return
        lo_val = torch.quantile(v, self.lo).item()
        hi_val = torch.quantile(v, self.hi).item()
        if self._ema_lo is None:
            self._ema_lo = lo_val
            self._ema_hi = hi_val
        else:
            self._ema_lo = self.decay * self._ema_lo + (1 - self.decay) * lo_val
            self._ema_hi = self.decay * self._ema_hi + (1 - self.decay) * hi_val

    @property
    def scale(self) -> float:
        if self._ema_lo is None:
            return 1.0
        return max(1.0, self._ema_hi - self._ema_lo)

    @property
    def rho(self) -> float:
        return 1.0 / self.scale

    def state_dict(self):
        return {'ema_lo': self._ema_lo, 'ema_hi': self._ema_hi,
                'decay': self.decay, 'lo': self.lo, 'hi': self.hi}

    def load_state_dict(self, d):
        self._ema_lo = d['ema_lo']
        self._ema_hi = d['ema_hi']


# ──────────────────────────────────────────────────────────────────────────────
# v5: Policy Prior Loss  (TD-MPC2 style, entropy + Q maximization)
# ──────────────────────────────────────────────────────────────────────────────

def policy_prior_loss(
    log_pi:       torch.Tensor,   # (B, T, action_latent)  log prob of sampled action
    q_pi:         torch.Tensor,   # (B, T)  Q(o_t, π(o_t)) scalar
    rho:          float,          # moving-percentile normalization
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    L_π = E[(entropy_coef · log_π(u|o) - Q(o, u)) * rho]

    Minimizing this:
      - pushes log_π down (maximize entropy)
      - pushes Q(o, π(o)) up (maximize Q)
      - rho normalizes loss magnitude to Q's current scale

    log_pi:  summed log prob of action dimensions, shape (B, T)
    q_pi:    scalar Q estimate for sampled action, shape (B, T)
    """
    # log_pi: sum over action dims if passed per-dim
    if log_pi.dim() > q_pi.dim():
        log_pi = log_pi.sum(-1)   # (B, T)

    loss = ((entropy_coef * log_pi - q_pi) * rho).mean()
    return loss
