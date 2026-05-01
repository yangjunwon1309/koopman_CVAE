"""
koopman_cvae.py — KODAQ v5
============================

v5 changes over v4:
  1. RewardCategoricalHead : step reward {0,1}, Two-Hot CE, bins [0, 5], B=16
  2. QHead                 : Bellman TD target, Two-Hot CE, bins [0, 5], ensemble
  3. PolicyPrior           : Gaussian pi(u|o), TD-MPC2 policy loss
  4. Target Q network      : EMA copy of QHead (tau=0.005)
  5. MovingPercentileScale : rho normalization for entropy/Q balance
  6. Zero-init last layer  : reward head & Q head (uniform initial distribution)
  7. Joint loss            : L_total = L_wm + lam_R*L_R + lam_Q*L_Q + lam_pi*L_pi

Bin design [v_min=0, v_max=5], B=16 bins:
  - reward head: predicts step r_t in {0,1} — always within [0,5], no clamp
  - Q head     : predicts TD target y_t = r_t + gamma*Q_bar
                 worst case: 1 + 0.99*4 = 4.96 < 5.0 — no clamp needed

rewards input = step reward {0, 1} per timestep (NOT accumulated).

World model components (v4, unchanged):
  ActionEncoder, PosteriorEncoder, RecurrentTransition,
  SkillPrior, SkillKoopmanOperator, MultiHeadDecoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.losses import (
    # v4
    symlog, symexp,
    blend_koopman, koopman_step,
    reconstruction_loss,
    koopman_consistency_loss,
    multistep_koopman_consistency_loss,
    skill_classification_loss,
    posterior_regularization_loss,
    eigenvalue_stability_loss,
    compute_total_loss,
    # v5
    get_bins,
    two_hot_encode,
    two_hot_decode,
    reward_categorical_loss,
    q_categorical_loss,
    MovingPercentileScale,
    policy_prior_loss,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # ── Input dimensions ──────────────────────────────────────────────────────
    dim_delta_e:   int   = 2048
    dim_delta_p:   int   = 42
    dim_q:         int   = 9
    dim_qdot:      int   = 9
    action_dim:    int   = 9
    state_dim:     int   = 60

    # ── Latent dimensions ─────────────────────────────────────────────────────
    koopman_dim:   int   = 128
    gru_hidden:    int   = 256
    action_latent: int   = 64
    num_skills:    int   = 8

    # ── Architecture ──────────────────────────────────────────────────────────
    mlp_hidden:    int   = 256
    enc_layers:    int   = 3
    dec_layers:    int   = 3
    dropout:       float = 0.1

    # ── v4 Loss weights ───────────────────────────────────────────────────────
    lambda1:       float = 1.0    # L_dyn
    lambda2:       float = 0.5    # L_skill
    lambda3:       float = 0.1    # L_reg
    lambda4:       float = 0.01   # L_stab

    alpha_delta_e: float = 1.0
    alpha_delta_p: float = 2.0
    alpha_q:       float = 1.0
    alpha_qdot:    float = 0.5
    dt_control:    float = 0.08

    # ── v5 Q / Reward head ────────────────────────────────────────────────────
    num_bins:      int   = 16     # B: bin count for Two-Hot distribution
    v_min:         float = 0.0    # bin range min
    v_max:         float = 5.0    # bin range max — covers H-step TD target
    num_q:         int   = 2      # Q ensemble size

    # ── v5.1 Reward Ensemble Head (MOPO-style) ────────────────────────────────
    reward_ensemble_n:   int   = 5      # N: reward ensemble members
    td_horizon:          int   = 4      # H: rollout steps for H-step TD (4 or 8)
    mopo_beta:           float = 1.0    # beta: MOPO penalty (mean - beta*std)
    use_ensemble_reward: bool  = False  # False=single head, True=ensemble
    tau:           float = 0.005  # EMA rate for target Q update
    gamma:         float = 0.99   # discount

    # ── v5 Loss weights ───────────────────────────────────────────────────────
    lambda_reward: float = 1.0    # L_R categorical reward head
    lambda_q:      float = 1.0    # L_Q Bellman TD
    lambda_pi:     float = 0.1    # L_π policy prior

    # ── v5 Policy prior ───────────────────────────────────────────────────────
    entropy_coef:  float = 0.01   # entropy regularization (α)
    log_std_min:   float = -5.0
    log_std_max:   float = 2.0

    # ── Multi-step L_dyn ──────────────────────────────────────────────────────
    multistep_dyn: bool  = True
    dyn_horizon:   int   = 8
    dyn_alpha:     float = 0.95

    # ── Training phase ────────────────────────────────────────────────────────
    phase:         int   = 1      # 1/2/3

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def x_dim(self) -> int:
        return self.dim_delta_e + self.dim_delta_p + self.dim_q + self.dim_qdot

    @property
    def rec_weights(self) -> dict:
        return {
            'delta_e': self.alpha_delta_e,
            'delta_p': self.alpha_delta_p,
            'q':       self.alpha_q,
            'qdot':    self.alpha_qdot,
        }

    @property
    def x_slices(self) -> dict:
        i0 = 0;  i1 = i0 + self.dim_delta_e
        i2 = i1 + self.dim_delta_p
        i3 = i2 + self.dim_q
        i4 = i3 + self.dim_qdot
        return {
            'delta_e': slice(i0, i1),
            'delta_p': slice(i1, i2),
            'q':       slice(i2, i3),
            'qdot':    slice(i3, i4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utility: MLP builder (v4 unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim, out_dim, hidden, n_layers, dropout=0.1,
             activate_last=False) -> nn.Sequential:
    layers = []
    d = in_dim
    for i in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                   nn.SiLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if activate_last:
        layers += [nn.LayerNorm(out_dim), nn.SiLU()]
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# v4 modules (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = make_mlp(cfg.action_dim, cfg.action_latent,
                            cfg.mlp_hidden, 2, cfg.dropout)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class PosteriorEncoder(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim = cfg.x_dim + cfg.gru_hidden
        self.trunk      = make_mlp(in_dim, cfg.mlp_hidden, cfg.mlp_hidden,
                                   cfg.enc_layers, cfg.dropout, activate_last=True)
        self.mu_head    = nn.Linear(cfg.mlp_hidden, cfg.koopman_dim)
        self.logvar_head = nn.Linear(cfg.mlp_hidden, cfg.koopman_dim)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.logvar_head.weight, gain=0.01)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x, h):
        feat   = self.trunk(torch.cat([x, h], dim=-1))
        mu     = self.mu_head(feat)
        logvar = self.logvar_head(feat).clamp(-10, 2)
        sigma2 = torch.exp(logvar)
        return mu, sigma2

    def sample(self, x, h):
        mu, sigma2 = self.forward(x, h)
        eps = torch.randn_like(mu)
        o   = mu + eps * torch.sqrt(sigma2 + 1e-8)
        return o, mu, sigma2


class RecurrentTransition(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        gru_in = cfg.koopman_dim + cfg.action_dim
        self.input_proj = nn.Linear(gru_in, cfg.gru_hidden)
        self.gru_cell   = nn.GRUCell(cfg.gru_hidden, cfg.gru_hidden)

    def forward(self, h, o, a):
        x_in = F.silu(self.input_proj(torch.cat([o, a], dim=-1)))
        return self.gru_cell(x_in, h)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.gru_cell.hidden_size, device=device)


class SkillPrior(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.W_c = nn.Linear(cfg.gru_hidden, cfg.num_skills)

    def forward(self, h):
        return self.W_c(h)

    def soft_weights(self, h):
        return torch.softmax(self.W_c(h), dim=-1)


class SkillKoopmanOperator(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        K, m, d_u = cfg.num_skills, cfg.koopman_dim, cfg.action_latent
        self.U       = nn.Parameter(torch.eye(m) + 0.01 * torch.randn(m, m))
        self.r_k     = nn.Parameter(3.0 * torch.ones(K, m))
        self.theta_k = nn.Parameter(0.01 * torch.randn(K, m))
        self.G_k     = nn.Parameter(0.01 * torch.randn(K, m, d_u))
        self.K, self.m, self.d_u = K, m, d_u

    def get_log_lambdas(self):
        return torch.log(torch.tanh(self.r_k.clamp(min=0.01)) + 1e-8)

    def forward(self, o, u, w):
        log_lam              = self.get_log_lambdas()
        A_bar, B_bar, _, _  = blend_koopman(log_lam, self.theta_k, self.G_k, self.U, w)
        o_next               = koopman_step(o, u, A_bar, B_bar)
        return o_next, A_bar, B_bar

    def get_A_k(self):
        log_lam = self.get_log_lambdas()
        U_c = self.U.to(dtype=torch.complex64)
        U_inv = torch.linalg.inv(U_c)
        r_exp = torch.exp(log_lam)
        lam_c = torch.complex(r_exp * torch.cos(self.theta_k),
                              r_exp * torch.sin(self.theta_k))
        Lam   = torch.diag_embed(lam_c)
        A_c   = U_c.unsqueeze(0) @ Lam @ U_inv.unsqueeze(0)
        return A_c.real

    def get_B_k(self):
        return self.U.unsqueeze(0) @ self.G_k


class MultiHeadDecoder(nn.Module):
    """v4 decoder: Δe, Δp, q, qdot (no reward — moved to dedicated head)."""
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        m, h, n, d = cfg.koopman_dim, cfg.mlp_hidden, cfg.dec_layers, cfg.dropout
        self.dt           = cfg.dt_control
        self.head_delta_e = make_mlp(m, cfg.dim_delta_e, h, n, d)
        self.head_delta_p = make_mlp(m, cfg.dim_delta_p, h, n, d)
        self.head_q       = make_mlp(m, cfg.dim_q, h, n, d)

    def forward(self, o: torch.Tensor) -> dict:
        q_hat = self.head_q(o)
        if o.dim() >= 3:
            dq = torch.zeros_like(q_hat)
            dq[..., 1:, :] = (q_hat[..., 1:, :] - q_hat[..., :-1, :]) / self.dt
        else:
            dq = torch.zeros_like(q_hat)
        return {
            'delta_e': self.head_delta_e(o),
            'delta_p': self.head_delta_p(o),
            'q':       q_hat,
            'qdot':    dq,
        }


# ──────────────────────────────────────────────────────────────────────────────
# v5 NEW: Reward Categorical Head
# ──────────────────────────────────────────────────────────────────────────────

class RewardCategoricalHead(nn.Module):
    """
    R̂(o_t, u_t) → categorical distribution over B bins covering [v_min, v_max].

    Predicts the step reward r_t ∈ {0, 1} obtained at timestep t.
    Using Two-Hot CE instead of BCE gives a consistent interface with the Q head
    and naturally extends if reward becomes continuous.

    Bin grid [0, 5]: step reward {0,1} is always within range.
    No clamping needed since targets ∈ {0, 1} ⊂ [0, 5].

    Input: concat(o_t, u_t) — state + action conditioned.
    Last linear layer zero-initialized → uniform initial output (no loss spikes).
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim  = cfg.koopman_dim + cfg.action_latent   # concat(o, u)
        hidden  = cfg.mlp_hidden
        B       = cfg.num_bins

        self.net = make_mlp(in_dim, B, hidden, 2, cfg.dropout)

        # Zero-init last layer → uniform initial distribution
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Register bin centers as buffer (moves with .to(device))
        self.register_buffer(
            'bins',
            torch.linspace(cfg.v_min, cfg.v_max, B)
        )

    def forward(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        o: (..., koopman_dim)
        u: (..., action_latent)
        → logits: (..., B)
        """
        return self.net(torch.cat([o, u], dim=-1))

    def expected_value(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Decode to scalar E[bin_center]."""
        logits = self.forward(o, u)
        return two_hot_decode(logits, self.bins)


# ──────────────────────────────────────────────────────────────────────────────
# v5.1 NEW: Reward Ensemble Head  (MOPO-style pessimistic reward)
# ──────────────────────────────────────────────────────────────────────────────

class RewardEnsembleHead(nn.Module):
    """
    N independent reward heads, each predicting step reward ∈ {0, 1}
    via sigmoid + BCE.

    Why sigmoid instead of Two-Hot for step reward:
      - Step reward is sparse impulse {0,1}, not a continuous distribution.
      - Two-Hot assumes the target lives on a continuous bin grid [0,5].
        With >96% r=0 transitions, the model learns "always predict bin 0"
        → collapse to a constant output.
      - Sigmoid BCE directly models P(r=1|o,u) ∈ (0,1), matching the
        binary nature of Kitchen subtask completion.
      - Two-Hot is appropriate for Q values (TD target is continuous [0,5]).

    MOPO-style penalized reward:
        R̂_pen(o, u) = mean_i[σ(r̂_i)] - beta * std_i[σ(r̂_i)]

    Each member outputs a scalar logit → sigmoid → probability of r=1.
    """

    def __init__(self, cfg: 'KoopmanCVAEConfig'):
        super().__init__()
        in_dim = cfg.koopman_dim + cfg.action_latent
        N      = cfg.reward_ensemble_n

        self.N    = N
        self.beta = cfg.mopo_beta

        # Each member: MLP → scalar logit (single output)
        self.nets = nn.ModuleList([
            make_mlp(in_dim, 1, cfg.mlp_hidden, 2, cfg.dropout)
            for _ in range(N)
        ])
        # Zero-init last layer → initial sigmoid output = 0.5 (neutral)
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward_logits(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Raw logits (before sigmoid).
        → (N, ..., 1)
        """
        x = torch.cat([o, u], dim=-1)
        return torch.stack([net(x) for net in self.nets], dim=0)  # (N, ..., 1)

    def member_probs(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        P(r=1|o,u) per member → (N, ...) after squeezing last dim.
        """
        return torch.sigmoid(self.forward_logits(o, u)).squeeze(-1)  # (N, ...)

    # Alias for analyze_v5.py compatibility
    def member_values(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.member_probs(o, u)

    def penalized_reward(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        MOPO-style: mean_i[P(r=1)] - beta * std_i[P(r=1)]
        → (...,) scalar ∈ (-beta, 1)
        Clamp to [0,1] for use as reward signal.
        """
        probs = self.member_probs(o, u)          # (N, ...)
        mu    = probs.mean(0)                    # (...,)
        sigma = probs.std(0)                     # (...,)
        return (mu - self.beta * sigma).clamp(0.0, 1.0)

    def ensemble_loss(self, o: torch.Tensor, u: torch.Tensor,
                      r_targets: torch.Tensor) -> torch.Tensor:
        """
        BCE loss averaged over all N members.
        Each member trained independently on the same binary target.

        r_targets: (...,) float, step reward ∈ {0.0, 1.0}
        """
        logits = self.forward_logits(o, u)              # (N, ..., 1)
        logits = logits.squeeze(-1)                     # (N, ...)
        # expand target to match N
        t_exp  = r_targets.unsqueeze(0).expand_as(logits)  # (N, ...)
        return F.binary_cross_entropy_with_logits(logits, t_exp)


# ──────────────────────────────────────────────────────────────────────────────
# v5 NEW: Q Head (Ensemble)
# ──────────────────────────────────────────────────────────────────────────────

class QHead(nn.Module):
    """
    Ensemble of num_q Q networks, each outputting B logits.

    Input:  concat(o_t, u_t)
    Output: (num_q, ..., B)  logits per network

    Architecture mirrors RewardCategoricalHead but with dropout
    (TD-MPC2 uses dropout in Q ensemble).

    Zero-init on each head's last layer.

    Pessimism via random subsampling: during loss computation,
    pick 2 random Q networks from the ensemble and take the min.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim = cfg.koopman_dim + cfg.action_latent
        B      = cfg.num_bins

        # Build ensemble
        self.nets = nn.ModuleList([
            make_mlp(in_dim, B, cfg.mlp_hidden, 2, cfg.dropout)
            for _ in range(cfg.num_q)
        ])
        # Zero-init each head's last linear
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        self.register_buffer(
            'bins',
            torch.linspace(cfg.v_min, cfg.v_max, B)
        )
        self.num_q = cfg.num_q

    def forward(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        o: (..., koopman_dim)
        u: (..., action_latent)
        → (num_q, ..., B)  logits
        """
        x = torch.cat([o, u], dim=-1)
        return torch.stack([net(x) for net in self.nets], dim=0)  # (num_q, ..., B)

    def expected_value(
        self,
        o: torch.Tensor,
        u: torch.Tensor,
        return_type: str = 'min',
    ) -> torch.Tensor:
        """
        Decode to scalar, with pessimism strategy.

        return_type:
          'min'  — min of 2 randomly sampled networks (offline conservative)
          'avg'  — average of 2 randomly sampled networks
          'all'  — all num_q networks, shape (num_q, ...)
        """
        logits = self.forward(o, u)                    # (num_q, ..., B)
        values = two_hot_decode(logits, self.bins)     # (num_q, ...)

        if return_type == 'all':
            return values

        idx = torch.randperm(self.num_q, device=o.device)[:2]
        Q2  = values[idx]                              # (2, ...)
        if return_type == 'min':
            return Q2.min(0).values
        return Q2.mean(0)


# ──────────────────────────────────────────────────────────────────────────────
# v5 NEW: Policy Prior  π(u | o)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyPrior(nn.Module):
    """
    π_θ(u | o_t)  — Gaussian policy in Koopman latent action space.

    Input:  o_t ∈ ℝ^{d_o}   (posterior latent)
    Output: mean ∈ ℝ^{d_u},  log_std ∈ ℝ^{d_u}
            action u ~ N(mean, exp(log_std)²)  then tanh-squashed

    Identical structure to TD-MPC2's _pi:
      MLP → split last dim → (mean, log_std)
      log_std clamped to [log_std_min, log_std_max]

    Training signal:
      L_π = E[(entropy_coef · log_π(u|o) - Q(o, u)) · rho]
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = make_mlp(
            cfg.koopman_dim,
            2 * cfg.action_latent,   # → split into mean, log_std
            cfg.mlp_hidden, 2, cfg.dropout
        )
        self.log_std_min = cfg.log_std_min
        self.log_std_dif = cfg.log_std_max - cfg.log_std_min

    def forward(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        """
        o: (..., d_o)

        Returns:
          action   (..., d_u)  tanh-squashed reparameterized sample
          log_prob (...,)      log probability of action (summed over d_u)
          mean     (..., d_u)
          log_std  (..., d_u)
        """
        out     = self.net(o)
        mean, log_std_raw = out.chunk(2, dim=-1)

        # Clamp log_std into [log_std_min, log_std_max]  (TD-MPC2 §H)
        log_std = torch.sigmoid(log_std_raw) * self.log_std_dif + self.log_std_min

        eps     = torch.randn_like(mean)
        u_raw   = mean + eps * log_std.exp()

        # log_prob of Gaussian (pre-tanh)  — identical to TD-MPC2 math.gaussian_logprob
        # = sum_i [ -0.5*eps_i^2 - log_std_i - 0.5*log(2π) ]
        log_prob_gauss = -0.5 * (
            eps.pow(2) + 2 * log_std + math.log(2 * math.pi)
        ).sum(-1)   # (...,)

        # Tanh squash + log_prob correction  — TD-MPC2 math.squash
        # log π(u|o) = log N(u_raw) - sum log(1 - tanh²(u_raw))
        # Use F.relu before log to guard against float32 rounding:
        # tanh² can numerically exceed 1 by ~1e-7, making (1-tanh²) slightly negative.
        action   = torch.tanh(u_raw)
        log_prob = log_prob_gauss - torch.log(
            F.relu(1 - action.pow(2)) + 1e-6
        ).sum(-1)   # (...,)

        return action, log_prob, mean, log_std


# ──────────────────────────────────────────────────────────────────────────────
# Main Model: KODAQ v5
# ──────────────────────────────────────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    KODAQ v5 — RSSM-Koopman + TD-MPC2-style Q/Reward/Policy heads.

    New heads (v5):
      reward_head  : RewardCategoricalHead  (Two-Hot CE, step reward {0,1}, bins [0,5])
      q_head       : QHead                  (Two-Hot CE TD target, ensemble, bins [0,5])
      q_head_target: EMA copy of q_head     (stop-gradient Bellman target)
      policy_prior : PolicyPrior            (Gaussian, tanh-squash)
      scale_tracker: MovingPercentileScale  (rho for pi loss normalization)

    Forward input:
      x_batch      : (B, T, x_dim)
      actions      : (B, T, da)
      skill_labels : (B, T) int64
      rewards      : (B, T) float   <- step reward {0, 1} per timestep
      mask         : (B, T) bool    (optional)

    Bin grid [v_min=0, v_max=5] shared by reward_head and q_head:
      - step reward {0,1} is always in [0,5]
      - TD target = r_t + gamma*Q_bar: worst case 1 + 0.99*4 = 4.96 < 5  no clamp needed
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

        # ── v4 modules (unchanged) ────────────────────────────────────────
        self.action_encoder = ActionEncoder(cfg)
        self.posterior      = PosteriorEncoder(cfg)
        self.recurrent      = RecurrentTransition(cfg)
        self.skill_prior    = SkillPrior(cfg)
        self.koopman        = SkillKoopmanOperator(cfg)
        self.decoder        = MultiHeadDecoder(cfg)

        # ── v5 NEW modules ────────────────────────────────────────────────
        self.reward_head         = RewardCategoricalHead(cfg)
        self.reward_ensemble_head = RewardEnsembleHead(cfg)   # v5.1
        self.q_head              = QHead(cfg)
        self.policy_prior        = PolicyPrior(cfg)

        # ── Q network: three versions (TD-MPC2 §H pattern) ───────────────
        #
        # q_head         : online Q — updated by Q loss gradient
        # q_head_target  : EMA copy — used for Bellman TD target (no grad)
        # _detach_q_head : shares q_head's parameters but blocks gradient
        #                  back through Q weights when computing policy loss.
        #                  Gradient flows only through u_pi → policy_prior.
        #
        # Why _detach_q_head?
        #   policy loss = E[entropy_coef*log_pi - Q(o, pi(o))]
        #   We WANT gradient of Q w.r.t. u_pi (to teach policy to increase Q).
        #   We do NOT want policy loss to update Q's weights — that's the
        #   Q loss's job. Without detach, policy loss would corrupt Q training.

        # target Q: deepcopy, no grad, always eval
        self.q_head_target = deepcopy(self.q_head)
        for p in self.q_head_target.parameters():
            p.requires_grad_(False)
        self.q_head_target.eval()

        # detach Q: same architecture as q_head, params linked via _sync_detach_q
        # We keep a separate module so .train()/.eval() calls don't cross-contaminate.
        self._detach_q_head = deepcopy(self.q_head)
        # params are synced manually before policy loss (no EMA — exact copy each step)
        # requires_grad=False so gradient does NOT flow into Q weights from policy loss
        for p in self._detach_q_head.parameters():
            p.requires_grad_(False)

        # Moving percentile tracker (non-parameter, saved in checkpoint manually)
        self.scale_tracker = MovingPercentileScale(decay=0.99)

    # ── Phase control ────────────────────────────────────────────────────────

    def set_phase(self, phase: int):
        assert phase in (1, 2, 3)
        self.cfg.phase = phase
        print(f"[KoopmanCVAE v5] Phase → {phase}")

    def init_skill_centroids(self, centroids: torch.Tensor):
        K, m = centroids.shape
        if K == self.cfg.num_skills and m == self.cfg.koopman_dim:
            U_init, _, _ = torch.linalg.svd(centroids.T, full_matrices=True)
            with torch.no_grad():
                self.koopman.U.copy_(
                    U_init[:m, :m] if U_init.shape[0] >= m else torch.eye(m)
                )
            print("[KoopmanCVAE v5] Initialized U from centroid SVD.")

    # ── Target Q soft update (call after each optimizer step) ────────────────

    @torch.no_grad()
    def soft_update_target_Q(self):
        """
        Called by trainer after every optimizer step.

        1. EMA update of q_head_target  (Bellman target, τ=0.005)
        2. Exact copy  of _detach_q_head (policy loss Q evaluator, no grad)

        TD-MPC2 equivalent:
          target_Qs  ← lerp(detach_Qs, tau)   # EMA
          detach_Qs  shares data with online Qs via TensorDictParams
          Here we replicate that with an explicit copy each step.
        """
        tau = self.cfg.tau
        for p_online, p_target in zip(
            self.q_head.parameters(), self.q_head_target.parameters()
        ):
            p_target.data.lerp_(p_online.data, tau)

        # _detach_q_head: exact copy of online weights (no EMA).
        # Always reflects the latest q_head so policy loss uses current Q values,
        # but no gradient flows back into q_head parameters.
        for p_online, p_detach in zip(
            self.q_head.parameters(), self._detach_q_head.parameters()
        ):
            p_detach.data.copy_(p_online.data)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x_batch:      torch.Tensor,
        actions:      torch.Tensor,
        skill_labels: Optional[torch.Tensor] = None,
        mask:         Optional[torch.Tensor] = None,
        rewards:      Optional[torch.Tensor] = None,  # (B, T) step reward {0, 1}
    ) -> Dict[str, torch.Tensor]:

        B, T, _ = x_batch.shape
        device   = x_batch.device
        cfg      = self.cfg

        # ── Encode actions ────────────────────────────────────────────────
        u_seq = self.action_encoder(actions)   # (B, T, d_u)

        # ── Unroll RSSM ───────────────────────────────────────────────────
        h = self.recurrent.init_hidden(B, device)
        h_list, o_list, mu_list, sigma2_list = [], [], [], []
        skill_logits_list  = []
        koopman_pred_list  = []
        A_bar_list, B_bar_list = [], []

        for t in range(T):
            x_t = x_batch[:, t]
            a_t = actions[:, t]
            u_t = u_seq[:, t]

            w_t = self.skill_prior.soft_weights(h)
            skill_logits_list.append(self.skill_prior(h))

            o_t, mu_t, sig2_t = self.posterior.sample(x_t, h)
            o_list.append(o_t)
            mu_list.append(mu_t)
            sigma2_list.append(sig2_t)

            if t < T - 1:
                _, A_bar, B_bar = self.koopman(o_t, u_t, w_t)
                o_next_pred = koopman_step(o_t, u_t, A_bar, B_bar)
                koopman_pred_list.append(o_next_pred)
                A_bar_list.append(A_bar)
                B_bar_list.append(B_bar)

            h = self.recurrent(h, o_t, a_t)
            h_list.append(h)

        # ── Stack ─────────────────────────────────────────────────────────
        h_seq        = torch.stack(h_list,             dim=1)  # (B, T, d_h)
        o_seq        = torch.stack(o_list,             dim=1)  # (B, T, d_o)
        mu_seq       = torch.stack(mu_list,            dim=1)  # (B, T, d_o)
        sigma2_seq   = torch.stack(sigma2_list,        dim=1)  # (B, T, d_o)
        skill_logits = torch.stack(skill_logits_list,  dim=1)  # (B, T, K)
        koopman_pred = torch.stack(koopman_pred_list,  dim=1)  # (B, T-1, d_o)
        A_bar_seq    = torch.stack(A_bar_list,         dim=1)  # (B, T-1, m, m)
        B_bar_seq    = torch.stack(B_bar_list,         dim=1)  # (B, T-1, m, d_u)

        # ── v4 Decoder ────────────────────────────────────────────────────
        recon = self.decoder(o_seq)   # dict, each (B, T, dim)

        # ── Compute all losses ────────────────────────────────────────────
        losses = self._compute_losses(
            x_batch=x_batch,
            recon=recon,
            o_seq=o_seq,
            mu_seq=mu_seq,
            u_seq=u_seq,
            koopman_pred=koopman_pred,
            A_bar_seq=A_bar_seq,
            B_bar_seq=B_bar_seq,
            skill_logits=skill_logits,
            skill_labels=skill_labels,
            mask=mask,
            rewards=rewards,
        )

        return {
            **losses,
            'z_seq':        o_seq,
            'mu_seq':       mu_seq,
            'sigma2_seq':   sigma2_seq,
            'h_seq':        h_seq,
            'skill_logits': skill_logits,
            'recon':        recon,
        }

    # ── Loss computation ─────────────────────────────────────────────────────

    def _compute_losses(
        self,
        x_batch:      torch.Tensor,          # (B, T, x_dim)
        recon:        dict,
        o_seq:        torch.Tensor,          # (B, T, d_o)
        mu_seq:       torch.Tensor,          # (B, T, d_o)
        u_seq:        torch.Tensor,          # (B, T, d_u)
        koopman_pred: torch.Tensor,          # (B, T-1, d_o)
        A_bar_seq:    torch.Tensor,          # (B, T-1, m, m)
        B_bar_seq:    torch.Tensor,          # (B, T-1, m, d_u)
        skill_logits: torch.Tensor,          # (B, T, K)
        skill_labels: Optional[torch.Tensor],
        mask:         Optional[torch.Tensor],
        rewards:      Optional[torch.Tensor],  # (B, T) accumulated
    ) -> Dict[str, torch.Tensor]:

        cfg    = self.cfg
        slices = cfg.x_slices
        device = x_batch.device

        # ── L_rec: v4 reconstruction (MSE + symlog) ───────────────────────
        targets = {
            'delta_e': symlog(x_batch[..., slices['delta_e']]),
            'delta_p': symlog(x_batch[..., slices['delta_p']]),
            'q':       symlog(x_batch[..., slices['q']]),
            'qdot':    symlog(x_batch[..., slices['qdot']]),
        }
        loss_rec, rec_per_head = reconstruction_loss(recon, targets, cfg.rec_weights)

        # ── L_dyn: Koopman consistency ────────────────────────────────────
        if cfg.multistep_dyn:
            loss_dyn = multistep_koopman_consistency_loss(
                mu_seq=mu_seq, o_seq=o_seq,
                A_bar_seq=A_bar_seq, B_bar_seq=B_bar_seq,
                u_seq=u_seq[:, :o_seq.shape[1]-1],
                H=cfg.dyn_horizon, alpha=cfg.dyn_alpha,
            )
        else:
            loss_dyn = koopman_consistency_loss(mu_seq[:, 1:], koopman_pred)

        # ── L_skill ───────────────────────────────────────────────────────
        if skill_labels is not None:
            loss_skill = skill_classification_loss(skill_logits, skill_labels, mask)
        else:
            loss_skill = torch.tensor(0.0, device=device)

        # ── L_reg ─────────────────────────────────────────────────────────
        loss_reg = posterior_regularization_loss(mu_seq[:, 1:], koopman_pred)

        # ── L_stab ────────────────────────────────────────────────────────
        loss_stab = eigenvalue_stability_loss(self.koopman.get_log_lambdas())

        # ── v4 total (phase-gated) ────────────────────────────────────────
        loss_wm, _ = compute_total_loss(
            loss_rec, loss_dyn, loss_skill, loss_reg, loss_stab,
            cfg.lambda1, cfg.lambda2, cfg.lambda3, cfg.lambda4, cfg.phase,
        )

        # ── v5 Reward Head: Categorical CE (Two-Hot, step reward) ────────
        # Transition: (o_t, u_t) → predicts step reward r_t (received at t)
        # We pair o_t = o_seq[:, :-1], u_t = u_seq[:, :-1] with r_t = rewards[:, :-1]
        # (rewards[:, t] is the step reward obtained at timestep t, before transition)
        loss_reward = torch.tensor(0.0, device=device)
        if rewards is not None:
            o_in      = o_seq[:, :-1]      # (B, T-1, d_o)
            u_in      = u_seq[:, :-1]      # (B, T-1, d_u)
            r_targets = rewards[:, :-1]    # (B, T-1)  step reward {0,1} at t

            if cfg.use_ensemble_reward:
                # sigmoid BCE ensemble loss
                loss_reward = self.reward_ensemble_head.ensemble_loss(
                    o_in, u_in, r_targets,
                )
            else:
                # original single categorical head (Two-Hot)
                reward_logits = self.reward_head(o_in, u_in)   # (B, T-1, B_bins)
                loss_reward   = reward_categorical_loss(
                    reward_logits, r_targets,
                    self.reward_head.bins,
                )

        # ── v5 Q Head + H-step Bellman TD Loss ──────────────────────────────
        #
        # Mode A (use_ensemble_reward=False): original 1-step TD
        #   y_t = r_t + gamma * Q_bar(o_{t+1}, pi(o_{t+1}))
        #
        # Mode B (use_ensemble_reward=True):  H-step MOPO-penalized TD
        #   ẑ_{t+0} = o_t
        #   û_{t+k} = pi(ẑ_{t+k})                  (policy prior, no grad)
        #   ẑ_{t+k+1} = Ā(w)·ẑ_{t+k} + B̄(w)·û_{t+k}  (Koopman linear rollout)
        #   R̂_pen_k  = mean_i[r̂_i(ẑ_{t+k}, û_{t+k})] - beta*std_i[...]
        #   y_t = Σ_{k=0}^{H-1} gamma^k · R̂_pen_k
        #       + gamma^H · Q̄(ẑ_{t+H}, pi(ẑ_{t+H}))
        #
        # Koopman rollout is used (not learned MLP dynamics):
        #   - linear, no compounding nonlinear error
        #   - world model freeze-safe: A_bar/B_bar are from frozen koopman
        #
        loss_q = torch.tensor(0.0, device=device)
        if rewards is not None:
            o_t = o_seq[:, :-1]    # (B, T-1, d_o)
            u_t = u_seq[:, :-1]    # (B, T-1, d_u)
            r_t = rewards[:, :-1]  # (B, T-1)  step reward at t

            with torch.no_grad():
                if cfg.use_ensemble_reward and cfg.td_horizon > 1:
                    # ── Mode B: H-step MOPO rollout ───────────────────────
                    H   = cfg.td_horizon
                    gm  = cfg.gamma
                    B_b = o_t.shape[0]
                    T1  = o_t.shape[1]

                    # Skill weights for Koopman: use current h (from forward pass)
                    # We re-derive A_bar/B_bar at t=0 positions using stored seqs
                    # A_bar_seq: (B, T-1, m, m),  B_bar_seq: (B, T-1, m, d_u)
                    # For rollout we use the A_bar at each starting t — then
                    # subsequent steps use the same A_bar (frozen world model).
                    # Simpler: use a single A_bar/B_bar per sequence from t
                    # (sufficient since world model is frozen in resume mode).

                    G      = torch.zeros(B_b, T1, device=device)
                    z_roll = o_t.clone()   # (B, T-1, d_o)  — starting latent

                    for k in range(H):
                        # Policy action at current rollout latent
                        u_roll, _, _, _ = self.policy_prior(z_roll)  # (B, T-1, d_u)

                        # MOPO penalized reward from ensemble
                        R_pen = self.reward_ensemble_head.penalized_reward(
                            z_roll, u_roll
                        )  # (B, T-1)
                        G = G + (gm ** k) * R_pen

                        # Koopman linear step: ẑ_{k+1} = Ā·ẑ_k + B̄·û_k
                        # Use A_bar_seq[:, 0] as representative (frozen WM)
                        # For a proper rollout use the t-indexed A_bar
                        if k < H - 1:
                            # A_bar_seq: (B, T-1, m, m)
                            A_k = A_bar_seq[:, :T1]   # (B, T-1, m, m)
                            B_k = B_bar_seq[:, :T1]   # (B, T-1, m, d_u)
                            # z_{k+1} = A·z_k + B·u_k
                            z_roll = (
                                (A_k @ z_roll.unsqueeze(-1)).squeeze(-1)
                                + (B_k @ u_roll.unsqueeze(-1)).squeeze(-1)
                            )   # (B, T-1, d_o)

                    # Terminal bootstrap: Q_bar(ẑ_H, pi(ẑ_H))
                    u_term, _, _, _ = self.policy_prior(z_roll)
                    q_terminal = self.q_head_target.expected_value(
                        z_roll, u_term, return_type='min'
                    )   # (B, T-1)
                    q_target_scalar = G + (gm ** H) * q_terminal

                else:
                    # ── Mode A: original 1-step TD ────────────────────────
                    o_tp1 = o_seq[:, 1:]          # (B, T-1, d_o)
                    u_pi_next, _, _, _ = self.policy_prior(o_tp1)
                    q_target_scalar = (
                        r_t + cfg.gamma
                        * self.q_head_target.expected_value(
                            o_tp1, u_pi_next, return_type='min'
                        )
                    )   # (B, T-1)

            # Online Q: all num_q networks
            q_logits_all = self.q_head(o_t, u_t)            # (num_q, B, T-1, B)
            q_logits_bt  = q_logits_all.permute(1, 2, 0, 3) # (B, T-1, num_q, B)

            loss_q = q_categorical_loss(
                q_logits=q_logits_bt,
                reward_seq=torch.zeros_like(r_t),  # reward already folded into target
                q_target_scalar=q_target_scalar,
                bins=self.q_head.bins,
                gamma=1.0,   # discount already applied above
            )

            # Update moving percentile scale
            with torch.no_grad():
                q_scalar_all = two_hot_decode(
                    q_logits_all.mean(0), self.q_head.bins,
                )   # (B, T-1)
                self.scale_tracker.update(q_scalar_all)

        # ── v5 Policy Prior Loss ──────────────────────────────────────────
        # L_π = E[ (entropy_coef · log_π(u|o) - Q(o, π(o))) · rho ]
        #
        # Gradient flow (TD-MPC2 §3.2):
        #   - log_π gradient → policy_prior parameters  (maximize entropy)
        #   - Q(o, π(o)) gradient w.r.t. u_π → policy_prior  (maximize Q)
        #   - Q(o, π(o)) gradient w.r.t. Q weights → BLOCKED via _detach_q_head
        #
        # rho warmup guard: skip policy update until scale_tracker has seen
        # enough Q values (scale > 0.1) to give a meaningful normalization.
        # This prevents a collapsed rho=1.0 from over-scaling early policy loss.
        loss_pi = torch.tensor(0.0, device=device)
        if cfg.phase >= 2 and self.scale_tracker.scale > 0.1:
            o_in_pi = o_seq[:, :-1].detach()            # (B, T-1, d_o)
            u_pi, log_pi, _, _ = self.policy_prior(o_in_pi)  # (B, T-1, d_u)

            # Use _detach_q_head: same weights as q_head but requires_grad=False.
            # Gradient flows through u_pi into policy_prior ONLY —
            # q_head parameters receive no gradient from policy loss.
            q_logits_pi = self._detach_q_head(o_in_pi, u_pi)  # (num_q, B, T-1, B)
            q_vals_pi   = two_hot_decode(q_logits_pi, self.q_head.bins)  # (num_q, B, T-1)

            # Pessimistic Q estimate: min of 2 randomly subsampled networks
            idx    = torch.randperm(self.cfg.num_q, device=device)[:2]
            q_pi   = q_vals_pi[idx].min(0).values   # (B, T-1)

            rho = self.scale_tracker.rho

            loss_pi = policy_prior_loss(
                log_pi=log_pi,
                q_pi=q_pi,
                rho=rho,
                entropy_coef=cfg.entropy_coef,
            )

        # ── Total loss ────────────────────────────────────────────────────
        loss_total = (
            loss_wm
            + cfg.lambda_reward * loss_reward
            + cfg.lambda_q      * loss_q
            + cfg.lambda_pi     * loss_pi
        )

        return {
            # v4
            'loss':             loss_total,
            'loss_wm':          loss_wm,
            'loss_rec':         loss_rec,
            'loss_dyn':         loss_dyn,
            'loss_skill':       loss_skill,
            'loss_reg':         loss_reg,
            'loss_stab':        loss_stab,
            'loss_rec_delta_e': rec_per_head['delta_e'],
            'loss_rec_delta_p': rec_per_head['delta_p'],
            'loss_rec_q':       rec_per_head['q'],
            'loss_rec_qdot':    rec_per_head['qdot'],
            # v5
            'loss_reward':      loss_reward,
            'loss_q':           loss_q,
            'loss_pi':          loss_pi,
            'rho':              torch.tensor(self.scale_tracker.rho, device=device),
            'q_scale':          torch.tensor(self.scale_tracker.scale, device=device),
        }

    # ── Inference utilities (v4 + v5 extensions) ─────────────────────────────

    @torch.no_grad()
    def encode_sequence(self, x_batch, actions):
        B, T, _ = x_batch.shape
        device   = x_batch.device
        h = self.recurrent.init_hidden(B, device)
        h_list, o_list, w_list = [], [], []
        for t in range(T):
            w_t = self.skill_prior.soft_weights(h)
            o_t, mu_t, _ = self.posterior.sample(x_batch[:, t], h)
            h = self.recurrent(h, o_t, actions[:, t])
            h_list.append(h)
            o_list.append(o_t)
            w_list.append(w_t)
        return {
            'o_seq': torch.stack(o_list, dim=1),
            'h_seq': torch.stack(h_list, dim=1),
            'w_seq': torch.stack(w_list, dim=1),
            'A_k':   self.koopman.get_A_k(),
            'B_k':   self.koopman.get_B_k(),
            'U':     self.koopman.U,
        }

    @torch.no_grad()
    def predict_q(self, o: torch.Tensor, u: torch.Tensor,
                  return_type: str = 'min') -> torch.Tensor:
        """Scalar Q estimate for a single (o, u) pair."""
        return self.q_head.expected_value(o, u, return_type=return_type)

    @torch.no_grad()
    def predict_reward(self, o: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Scalar reward estimate."""
        return self.reward_head.expected_value(o, u)

    @torch.no_grad()
    def act(self, o: torch.Tensor) -> torch.Tensor:
        """
        Sample action from policy prior (tanh-squashed Gaussian).
        o: (B, d_o) or (d_o,)
        Returns u: same batch shape, (d_u,)
        """
        if o.dim() == 1:
            o = o.unsqueeze(0)
        action, _, _, _ = self.policy_prior(o)
        return action.squeeze(0)

    @torch.no_grad()
    def rollout(self, x_cond, a_cond, a_plan, h_init=None, o_init=None):
        """
        Koopman rollout from a conditioning window.

        h_init, o_init: pre-computed hidden state and latent at the START of
        x_cond. If provided (recommended for mid-episode rollout), x_cond
        warm-up begins from these states instead of h=0.

        If None: h starts from zeros (correct only at episode start t=0).
        """
        B      = x_cond.shape[0]
        device = x_cond.device
        h      = h_init if h_init is not None else self.recurrent.init_hidden(B, device)
        o      = o_init  # may be None, will be set on first posterior sample

        for t in range(x_cond.shape[1]):
            o, _, _ = self.posterior.sample(x_cond[:, t], h)
            h = self.recurrent(h, o, a_cond[:, t])

        o_preds, recon_preds = [], []
        w = self.skill_prior.soft_weights(h)
        for t in range(a_plan.shape[1]):
            u      = self.action_encoder(a_plan[:, t])
            o_next, A_bar, B_bar = self.koopman(o, u, w)
            o      = o_next
            h      = self.recurrent(h, o, a_plan[:, t])
            w      = self.skill_prior.soft_weights(h)
            o_preds.append(o)
            recon_preds.append(self.decoder(o))

        result = {'o_preds': torch.stack(o_preds, dim=1)}
        for key in ['delta_e', 'delta_p', 'q', 'qdot']:
            result[key] = symexp(torch.stack([r[key] for r in recon_preds], dim=1))
        return result