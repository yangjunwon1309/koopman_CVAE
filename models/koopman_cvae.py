"""
koopman_cvae.py — KODAQ Full RSSM-Koopman
==========================================

Document → Implementation mapping (KODAQ §3):

  Input x_t = [Δe_t (2048), Δp_t (42), q_t (9), q̇_t (9)] ∈ ℝ^{2108}
    Δe_t = R3M(s_t) - R3M(s_1)   episode-first difference
    Δp_t = p^obj_t - p^obj_1     object state difference

  State Variables:
    o_t ∈ ℝ^{d_o}  : lifted (Koopman) state
    h_t ∈ ℝ^{d_h}  : GRU hidden (temporal history)
    c_t ∈ {1..K}    : discrete skill label (from EXTRACT)
    u_t ∈ ℝ^{d_u}  : encoded action

  Generative model (§3.2):
    h_{t+1}  = GRU(h_t, o_t, a_t)
    p(c_t|h_t) = Cat(softmax(W_c h_t))
    p(o_{t+1}|o_t,a_t,h_t) = N(Ā(w)·o_t + B̄(w)·u_t, σ²I)
    Ā(w) = U · exp(Σ_k w_k log Λ_k) · U⁻¹   [log-space interpolation]
    B̄(w) = U · (Σ_k w_k G_k)

  Recognition model (§3.3):
    q_φ(o_t|x_t,h_t) = N(μ_φ(x_t,h_t), diag(σ_φ²(x_t,h_t)))

  Decoder (§3.4):
    p(x_t|o_t) = p(Δe_t|o_t) · p(Δp_t|o_t) · p(q_t|o_t) · p(q̇_t|o_t)
    4 independent MLP heads, MSE loss

  Action encoder:
    u_t = ψ_θ(a_t)   (MLP)

  Loss (§4):
    L = L_rec - λ1·L_dyn - λ2·L_skill - λ3·L_reg
    Phase 1: L_rec only
    Phase 2: + λ1·L_dyn + λ2·L_skill
    Phase 3: + λ3·L_reg

Architecture notes:
  - A_k, B_k initialized near identity (A_k = I + ε, B_k = ε) per §6
  - μ_k initialized from EXTRACT cluster centroids (external call)
  - U shared across skills (Assumption 2)
  - Stability: tanh(r^(k)_i)·e^{iθ^(k)_i} guarantees |λ^(k)_i| ≤ 1 (Assumption 3)

Module separation:
  losses.py   : all loss functions (pure functions, no nn.Module state)
  koopman_cvae.py : nn.Module classes + forward/loss delegation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.losses import (
    symlog, symexp,
    blend_koopman, koopman_step,
    reconstruction_loss,
    koopman_consistency_loss,
    multistep_koopman_consistency_loss,
    skill_classification_loss,
    posterior_regularization_loss,
    eigenvalue_stability_loss,
    compute_total_loss,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # ── Input dimensions (KODAQ §1.1) ─────────────────────────────────────────
    # x_t = [Δe_t(2048), Δp_t(42), q_t(9), q̇_t(9)]
    dim_delta_e:  int   = 2048   # R3M embedding diff
    dim_delta_p:  int   = 42     # object state diff
    dim_q:        int   = 9      # joint positions
    dim_qdot:     int   = 9      # joint velocities
    action_dim:   int   = 9      # robot action a_t

    # Raw D4RL observation (60-dim) needed for GRU input fallback
    state_dim:    int   = 60     # used in env_configs, not for x_t construction

    # ── Latent dimensions ─────────────────────────────────────────────────────
    koopman_dim:  int   = 128    # d_o: lifted state dimension
    gru_hidden:   int   = 256    # d_h: GRU hidden state
    action_latent: int  = 64     # d_u: encoded action dimension
    num_skills:   int   = 8      # K: number of skill components

    # ── Architecture ──────────────────────────────────────────────────────────
    mlp_hidden:   int   = 256
    enc_layers:   int   = 3      # encoder MLP depth
    dec_layers:   int   = 3      # decoder MLP depth
    dropout:      float = 0.1

    # ── Loss weights (§4) ─────────────────────────────────────────────────────
    lambda1:      float = 1.0    # L_dyn   (Koopman consistency)
    lambda2:      float = 0.5    # L_skill (cross-entropy)
    lambda3:      float = 0.1    # L_reg   (posterior regularization)
    lambda4:      float = 0.01   # L_stab  (eigenvalue stability monitoring)

    # Reconstruction head weights α_j (§4)
    alpha_delta_e: float = 1.0
    alpha_delta_p: float = 2.0   # object states are key for skill separation
    alpha_q:       float = 1.0
    alpha_qdot:    float = 0.5   # velocities noisier → lower weight

    # ── Reward head ───────────────────────────────────────────────────────────
    use_reward_head: bool  = True   # binary reward prediction head (BCE)
    alpha_reward:    float = 1.0    # BCE weight in L_rec
    dt_control:      float = 0.08   # Kitchen dt=12.5Hz for qdot finite diff

    # ── Multi-step L_dyn (RWM-style) ─────────────────────────────────────────
    multistep_dyn:  bool  = True    # use multi-step Koopman consistency loss
    dyn_horizon:    int   = 8       # H: rollout steps (≤ seq_len-1)
    dyn_alpha:      float = 0.95    # α: per-step decay

    # ── Training phase ────────────────────────────────────────────────────────
    # Updated externally by trainer
    phase:        int   = 1      # 1: rec, 2: +dyn+skill, 3: +reg

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def x_dim(self) -> int:
        """Total input dimension."""
        return self.dim_delta_e + self.dim_delta_p + self.dim_q + self.dim_qdot

    @property
    def rec_weights(self) -> dict:
        d = {
            'delta_e': self.alpha_delta_e,
            'delta_p': self.alpha_delta_p,
            'q':       self.alpha_q,
            'qdot':    self.alpha_qdot,
        }
        if self.use_reward_head:
            d['reward'] = self.alpha_reward
        return d

    @property
    def x_slices(self) -> dict:
        """Index slices for each x_t component."""
        i0 = 0
        i1 = self.dim_delta_e
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
# Utility: MLP builder
# ──────────────────────────────────────────────────────────────────────────────

def make_mlp(
    in_dim:   int,
    out_dim:  int,
    hidden:   int,
    n_layers: int,
    dropout:  float = 0.1,
    activate_last: bool = False,
) -> nn.Sequential:
    """Standard MLP with LayerNorm + SiLU activations."""
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
    # Orthogonal init
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# Action Encoder: ψ_θ(a_t) → u_t
# ──────────────────────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """
    u_t = ψ_θ(a_t)  (§3.2)
    MLP: a_t ∈ ℝ^{da} → u_t ∈ ℝ^{d_u}
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = make_mlp(
            cfg.action_dim, cfg.action_latent,
            cfg.mlp_hidden, 2, cfg.dropout
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """a: (..., da) → u: (..., d_u)"""
        return self.net(a)


# ──────────────────────────────────────────────────────────────────────────────
# Recognition Model: q_φ(o_t | x_t, h_t)
# ──────────────────────────────────────────────────────────────────────────────

class PosteriorEncoder(nn.Module):
    """
    q_φ(o_t | x_t, h_t) = N(μ_φ(x_t, h_t), diag(σ_φ²(x_t, h_t)))  (§3.3)

    MLP conditioned on (x_t, h_t).
    h_t resolves temporal ambiguity (same object position, different skill phase).
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim = cfg.x_dim + cfg.gru_hidden
        self.trunk = make_mlp(
            in_dim, cfg.mlp_hidden, cfg.mlp_hidden,
            cfg.enc_layers, cfg.dropout, activate_last=True
        )
        self.mu_head    = nn.Linear(cfg.mlp_hidden, cfg.koopman_dim)
        self.logvar_head = nn.Linear(cfg.mlp_hidden, cfg.koopman_dim)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.logvar_head.weight, gain=0.01)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # start with small σ

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., x_dim)
        h: (..., d_h)
        Returns μ: (..., d_o), σ²: (..., d_o) [clamped]
        """
        feat    = self.trunk(torch.cat([x, h], dim=-1))
        mu      = self.mu_head(feat)
        logvar  = self.logvar_head(feat).clamp(-10, 2)
        sigma2  = torch.exp(logvar)
        return mu, sigma2

    def sample(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample + return μ, σ².
        Returns (o_sample, μ, σ²)
        """
        mu, sigma2 = self.forward(x, h)
        eps  = torch.randn_like(mu)
        o    = mu + eps * torch.sqrt(sigma2 + 1e-8)
        return o, mu, sigma2


# ──────────────────────────────────────────────────────────────────────────────
# GRU Recurrent Transition: h_{t+1} = f_θ(h_t, o_t, a_t)
# ──────────────────────────────────────────────────────────────────────────────

class RecurrentTransition(nn.Module):
    """
    h_{t+1} = GRU(h_t, [o_t, a_t])  (§3.2)

    Projects (o_t, a_t) → GRU input, then runs GRU cell.
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        gru_in = cfg.koopman_dim + cfg.action_dim
        self.input_proj = nn.Linear(gru_in, cfg.gru_hidden)
        self.gru_cell   = nn.GRUCell(cfg.gru_hidden, cfg.gru_hidden)

    def forward(
        self,
        h: torch.Tensor,   # (B, d_h)
        o: torch.Tensor,   # (B, d_o)
        a: torch.Tensor,   # (B, da)
    ) -> torch.Tensor:     # (B, d_h)
        x_in = F.silu(self.input_proj(torch.cat([o, a], dim=-1)))
        return self.gru_cell(x_in, h)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru_cell.hidden_size, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Skill Prior: p_θ(c_t | h_t)
# ──────────────────────────────────────────────────────────────────────────────

class SkillPrior(nn.Module):
    """
    p_θ(c_t | h_t) = Cat(softmax(W_c h_t))  (§3.2)
    Returns logits for cross-entropy loss.
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.W_c = nn.Linear(cfg.gru_hidden, cfg.num_skills)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_h) → logits: (..., K)"""
        return self.W_c(h)

    def soft_weights(self, h: torch.Tensor) -> torch.Tensor:
        """w_k = softmax(W_c h_t): (..., K)"""
        return torch.softmax(self.W_c(h), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Skill-conditioned Koopman: A_k, B_k, U (§3.2)
# ──────────────────────────────────────────────────────────────────────────────

class SkillKoopmanOperator(nn.Module):
    """
    Shared eigenbasis U ∈ ℝ^{m×m}
    Skill-specific eigenvalues: λ^(k)_i = tanh(r^(k)_i) · e^{iθ^(k)_i}
      → |λ^(k)_i| = tanh(r^(k)_i) ≤ 1 always (Assumption 3)
    Skill-specific input coupling: G_k ∈ ℝ^{m×d_u}
      A_k = U Λ_k U⁻¹,   B_k = U G_k
    Log-space interpolation (§3.2):
      Ā(w) = U · exp(Σ_k w_k log Λ_k) · U⁻¹
      B̄(w) = U · (Σ_k w_k G_k)
    A_k initialized near identity (§6): r^(k) → -∞ (tanh≈0 → |λ|≈0 → A≈0),
      so we init r small negative and θ near uniform.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        K = cfg.num_skills
        m = cfg.koopman_dim
        d_u = cfg.action_latent

        # Shared eigenbasis U (orthogonal init)
        self.U = nn.Parameter(torch.eye(m) + 0.01 * torch.randn(m, m))

        # Skill-specific eigenvalue parameters
        # r^(k): log-magnitude before tanh, init small → tanh(r) ~ r → small |λ|
        # Near-identity: A_k ≈ I → need large r (tanh(r)→1) with θ≈0
        # §6: A_k = I + ε → start with tanh(r)~1 → r=3, θ~0
        self.r_k = nn.Parameter(3.0 * torch.ones(K, m))     # magnitude
        self.theta_k = nn.Parameter(0.01 * torch.randn(K, m))  # phase

        # Skill-specific input coupling G_k ∈ ℝ^{K × m × d_u}
        # B_k = U G_k,  init small (§6: B_k = ε)
        self.G_k = nn.Parameter(0.01 * torch.randn(K, m, d_u))

        self.K   = K
        self.m   = m
        self.d_u = d_u

    def get_log_lambdas(self) -> torch.Tensor:
        """
        log|λ^(k)_i| = log(tanh(r^(k)_i))
        Since tanh(r) ∈ (0,1), log(tanh(r)) ≤ 0 → guaranteed stable.
        Returns: (K, m)
        """
        return torch.log(torch.tanh(self.r_k.clamp(min=0.01)) + 1e-8)

    def forward(
        self,
        o: torch.Tensor,   # (B, m)
        u: torch.Tensor,   # (B, d_u)
        w: torch.Tensor,   # (B, K) soft skill weights
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step Koopman prediction.
        Returns (o_next_pred, A_bar, B_bar)
        """
        log_lam = self.get_log_lambdas()   # (K, m)
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, self.theta_k, self.G_k, self.U, w
        )
        o_next = koopman_step(o, u, A_bar, B_bar)
        return o_next, A_bar, B_bar

    def get_A_k(self) -> torch.Tensor:
        """Returns A_k for each skill: (K, m, m) — for LQR."""
        log_lam = self.get_log_lambdas()   # (K, m)
        U = self.U
        U_c = U.to(dtype=torch.complex64)
        U_inv = torch.linalg.inv(U_c)

        r_exp = torch.exp(log_lam)         # (K, m)
        lam_c = torch.complex(
            r_exp * torch.cos(self.theta_k),
            r_exp * torch.sin(self.theta_k),
        )  # (K, m)
        Lam = torch.diag_embed(lam_c)     # (K, m, m)
        A_c = U_c.unsqueeze(0) @ Lam @ U_inv.unsqueeze(0)
        return A_c.real                    # (K, m, m)

    def get_B_k(self) -> torch.Tensor:
        """Returns B_k = U G_k for each skill: (K, m, d_u)."""
        return self.U.unsqueeze(0) @ self.G_k   # (K, m, d_u)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder: p_θ(x_t | o_t) — 4 independent MLP heads
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadDecoder(nn.Module):
    """
    Decoder heads:
      Δê_t  = D_e(o_t)        (2048-dim, symlog MSE)
      Δp̂_t  = D_p(o_t)       (42-dim,   symlog MSE)
      q̂_t   = D_q(o_t)       (9-dim,    symlog MSE)
      q̇̂_t   = finite_diff(q̂) (9-dim,    symlog MSE)
                                — physically: q̇ = (q_t - q_{t-1}) / dt
      r̂_t   = D_r(o_t)       (1-dim,    BCE logit)  optional reward head

    qdot head is removed and derived from q via finite difference.
    For single-step input (no time dim) qdot is zero.
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        m  = cfg.koopman_dim
        h  = cfg.mlp_hidden
        n  = cfg.dec_layers
        d  = cfg.dropout
        self.dt              = cfg.dt_control
        self.use_reward_head = cfg.use_reward_head

        self.head_delta_e = make_mlp(m, cfg.dim_delta_e, h, n, d)
        self.head_delta_p = make_mlp(m, cfg.dim_delta_p, h, n, d)
        self.head_q       = make_mlp(m, cfg.dim_q,       h, n, d)
        # head_qdot removed — derived from head_q via finite difference
        if self.use_reward_head:
            self.head_reward = make_mlp(m, 1, h, max(n - 1, 1), d)

    def forward(self, o: torch.Tensor) -> dict:
        """
        o: (..., m) or (B, T, m)

        qdot:
          ndim >= 3 (B,T,m): qdot[...,1:,:] = (q[...,1:,:] - q[...,:-1,:]) / dt
          otherwise:         qdot = zeros_like(q)
        """
        q_hat = self.head_q(o)

        if o.dim() >= 3:
            dq = torch.zeros_like(q_hat)
            dq[..., 1:, :] = (q_hat[..., 1:, :] - q_hat[..., :-1, :]) / self.dt
        else:
            dq = torch.zeros_like(q_hat)

        out = {
            'delta_e': self.head_delta_e(o),
            'delta_p': self.head_delta_p(o),
            'q':       q_hat,
            'qdot':    dq,
        }
        if self.use_reward_head:
            out['reward'] = self.head_reward(o)   # (..., 1) raw logit for BCE
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Main Model: KODAQ Full RSSM-Koopman
# ──────────────────────────────────────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    KODAQ Full RSSM-Koopman (§3).

    Forward input contract:
        x_batch: (B, T, x_dim)  where x_dim = dim_delta_e+dim_delta_p+dim_q+dim_qdot
        actions: (B, T, da)
        skill_labels: (B, T) int64  — EXTRACT cluster assignments ĉ_t

    Forward output:
        loss, loss_rec, loss_dyn, loss_skill, loss_reg, loss_stab
        z_seq: (B, T, d_o)          posterior samples o_t
        mu_seq: (B, T, d_o)         posterior means
        h_seq: (B, T, d_h)          GRU hidden states
        skill_logits: (B, T, K)     log p(c_t|h_t)
        recon: dict of (B, T, dim)  per-head reconstructions

    Data preparation (outside this file):
        x_t is built from cached R3M embeddings + D4RL observations.
        See data/extract_skill_label.py::build_x_sequence() and
        data/dataset_utils.py::KODAQDataset.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

        # ── Modules ────────────────────────────────────────────────────────
        self.action_encoder    = ActionEncoder(cfg)
        self.posterior         = PosteriorEncoder(cfg)
        self.recurrent         = RecurrentTransition(cfg)
        self.skill_prior       = SkillPrior(cfg)
        self.koopman           = SkillKoopmanOperator(cfg)
        self.decoder           = MultiHeadDecoder(cfg)

    # ── Phase control ───────────────────────────────────────────────────────

    def set_phase(self, phase: int):
        """Update training phase (1→2→3). Called externally by trainer."""
        assert phase in (1, 2, 3), f"Phase must be 1, 2, or 3, got {phase}"
        self.cfg.phase = phase
        print(f"[KoopmanCVAE] Phase → {phase}")

    def init_skill_centroids(self, centroids: torch.Tensor):
        """
        §6: μ_k initialized from EXTRACT cluster centroids projected to lifted space.
        centroids: (K, m) — centroids in Koopman space, e.g. from k-means on z.
        This sets r_k, theta_k such that the prior starts near centroid distribution.
        (Optional: can also be used to warm-start U via SVD of centroid matrix.)
        """
        # Warm-start: project centroids to find reasonable U initialization
        K, m = centroids.shape
        if K == self.cfg.num_skills and m == self.cfg.koopman_dim:
            # SVD of centroid matrix as initial U estimate
            U_init, _, _ = torch.linalg.svd(centroids.T, full_matrices=True)
            with torch.no_grad():
                self.koopman.U.copy_(U_init[:m, :m] if U_init.shape[0] >= m
                                     else torch.eye(m))
            print(f"[KoopmanCVAE] Initialized U from centroid SVD.")

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x_batch:      torch.Tensor,              # (B, T, x_dim)
        actions:      torch.Tensor,              # (B, T, da)
        skill_labels: Optional[torch.Tensor] = None,  # (B, T) int64
        mask:         Optional[torch.Tensor] = None,  # (B, T) bool
        rewards:      Optional[torch.Tensor] = None,  # (B, T) float — reward diff (0/1)
    ) -> Dict[str, torch.Tensor]:

        B, T, _ = x_batch.shape
        device   = x_batch.device
        cfg      = self.cfg

        # ── Encode actions ──────────────────────────────────────────────────
        # u_t = ψ_θ(a_t),  shape (B, T, d_u)
        u_seq = self.action_encoder(actions)

        # ── Unroll RSSM ─────────────────────────────────────────────────────
        # h_t: GRU hidden, o_t: posterior sample, mu_t: posterior mean
        h = self.recurrent.init_hidden(B, device)    # (B, d_h)
        h_list        = []
        o_list        = []
        mu_list       = []
        sigma2_list   = []
        skill_logits_list = []
        koopman_pred_list = []  # Ā·o_{t-1} + B̄·u_{t-1}  for t=1..T-1
        A_bar_list    = []      # Ā(w_t) per step  — for multi-step L_dyn
        B_bar_list    = []      # B̄(w_t) per step

        for t in range(T):
            x_t = x_batch[:, t]      # (B, x_dim)
            a_t = actions[:, t]      # (B, da)
            u_t = u_seq[:, t]        # (B, d_u)

            # Skill weights from current h_t
            w_t       = self.skill_prior.soft_weights(h)         # (B, K)
            skill_logits_list.append(self.skill_prior(h))        # (B, K)

            # Posterior: o_t ~ q_φ(o_t | x_t, h_t)
            o_t, mu_t, sig2_t = self.posterior.sample(x_t, h)
            o_list.append(o_t)
            mu_list.append(mu_t)
            sigma2_list.append(sig2_t)

            # Koopman prior prediction for NEXT step (t+1)
            # Collect A_bar, B_bar for multi-step L_dyn
            if t < T - 1:
                _, A_bar, B_bar = self.koopman(o_t, u_t, w_t)
                o_next_pred = koopman_step(o_t, u_t, A_bar, B_bar)
                koopman_pred_list.append(o_next_pred)   # (B, m)
                A_bar_list.append(A_bar)                # (B, m, m)
                B_bar_list.append(B_bar)                # (B, m, d_u)

            # GRU update: h_{t+1} = f(h_t, o_t, a_t)
            h = self.recurrent(h, o_t, a_t)
            h_list.append(h)

        # ── Stack sequences ──────────────────────────────────────────────────
        h_seq           = torch.stack(h_list,         dim=1)   # (B, T, d_h)
        o_seq           = torch.stack(o_list,         dim=1)   # (B, T, m)
        mu_seq          = torch.stack(mu_list,        dim=1)   # (B, T, m)
        sigma2_seq      = torch.stack(sigma2_list,    dim=1)   # (B, T, m)
        skill_logits    = torch.stack(skill_logits_list, dim=1)  # (B, T, K)
        koopman_pred    = torch.stack(koopman_pred_list, dim=1)  # (B, T-1, m)
        A_bar_seq       = torch.stack(A_bar_list, dim=1)         # (B, T-1, m, m)
        B_bar_seq       = torch.stack(B_bar_list, dim=1)         # (B, T-1, m, d_u)

        # ── Decode ───────────────────────────────────────────────────────────
        recon = self.decoder(o_seq)                             # dict of (B, T, dim)

        # ── Losses ───────────────────────────────────────────────────────────
        losses = self._compute_losses(
            x_batch=x_batch,
            recon=recon,
            mu_seq=mu_seq,
            koopman_pred=koopman_pred,
            A_bar_seq=A_bar_seq,
            B_bar_seq=B_bar_seq,
            u_seq=u_seq[:, :T-1],
            skill_logits=skill_logits,
            skill_labels=skill_labels,
            o_seq=o_seq,
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
        x_batch:      torch.Tensor,
        recon:        dict,
        mu_seq:       torch.Tensor,              # (B, T, m)
        koopman_pred: torch.Tensor,              # (B, T-1, m)  single-step pred
        A_bar_seq:    torch.Tensor,              # (B, T-1, m, m)
        B_bar_seq:    torch.Tensor,              # (B, T-1, m, d_u)
        u_seq:        torch.Tensor,              # (B, T-1, d_u)
        skill_logits: torch.Tensor,
        skill_labels: Optional[torch.Tensor],
        o_seq:        torch.Tensor,              # (B, T, m)
        mask:         Optional[torch.Tensor],
        rewards:      Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        cfg    = self.cfg
        slices = cfg.x_slices

        # ── L_rec: multi-head reconstruction ────────────────────────────────
        targets = {
            'delta_e': symlog(x_batch[..., slices['delta_e']]),
            'delta_p': symlog(x_batch[..., slices['delta_p']]),
            'q':       symlog(x_batch[..., slices['q']]),
            'qdot':    symlog(x_batch[..., slices['qdot']]),
        }
        if cfg.use_reward_head and rewards is not None and 'reward' in recon:
            targets['reward'] = rewards.unsqueeze(-1).float()
        loss_rec, rec_per_head = reconstruction_loss(recon, targets, cfg.rec_weights)

        # ── L_dyn: Koopman consistency ────────────────────────────────────
        if cfg.multistep_dyn:
            # Multi-step RWM-style: (1/N) Σ α^k ||μ_{t+k} - ẑ_{t+k}||²
            loss_dyn = multistep_koopman_consistency_loss(
                mu_seq    = mu_seq,
                o_seq     = o_seq,
                A_bar_seq = A_bar_seq,
                B_bar_seq = B_bar_seq,
                u_seq     = u_seq,
                H         = cfg.dyn_horizon,
                alpha     = cfg.dyn_alpha,
            )
        else:
            # Single-step fallback
            loss_dyn = koopman_consistency_loss(
                mu_next = mu_seq[:, 1:],
                o_pred  = koopman_pred,
            )

        # ── L_skill: cross-entropy vs EXTRACT labels ─────────────────────────
        if skill_labels is not None:
            loss_skill = skill_classification_loss(
                skill_logits, skill_labels, mask
            )
        else:
            loss_skill = torch.tensor(0.0, device=x_batch.device)

        # ── L_reg: posterior-prior alignment with stop_gradient ─────────────
        # μ_φ(x_t, h_t) vs sg(Ā(w_{t-1})·o_{t-1} + B̄(w_{t-1})·u_{t-1})
        # = mu_seq[:, 1:] vs sg(koopman_pred[:, :])
        loss_reg = posterior_regularization_loss(
            mu_t   = mu_seq[:, 1:],        # (B, T-1, m)
            o_pred = koopman_pred,         # (B, T-1, m)  stop_grad inside fn
        )

        # ── L_stab: eigenvalue stability (monitoring + light push) ───────────
        log_lam   = self.koopman.get_log_lambdas()   # (K, m), should be ≤ 0
        loss_stab = eigenvalue_stability_loss(log_lam)

        # ── Total (phase-gated) ───────────────────────────────────────────────
        loss, weights = compute_total_loss(
            loss_rec, loss_dyn, loss_skill, loss_reg, loss_stab,
            cfg.lambda1, cfg.lambda2, cfg.lambda3, cfg.lambda4,
            cfg.phase,
        )

        return {
            'loss':       loss,
            'loss_rec':   loss_rec,
            'loss_dyn':   loss_dyn,
            'loss_skill': loss_skill,
            'loss_reg':   loss_reg,
            'loss_stab':  loss_stab,
            'loss_rec_delta_e': rec_per_head['delta_e'],
            'loss_rec_delta_p': rec_per_head['delta_p'],
            'loss_rec_q':       rec_per_head['q'],
            'loss_rec_qdot':    rec_per_head['qdot'],
            'loss_rec_reward':  rec_per_head.get('reward',
                                    torch.tensor(0.0, device=x_batch.device)),
        }

    # ── Inference utilities ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode_sequence(
        self,
        x_batch: torch.Tensor,   # (B, T, x_dim)
        actions: torch.Tensor,   # (B, T, da)
    ) -> Dict[str, torch.Tensor]:
        """Encode without loss computation. Returns latents + skill weights."""
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
            'o_seq': torch.stack(o_list, dim=1),    # (B, T, m)
            'h_seq': torch.stack(h_list, dim=1),    # (B, T, d_h)
            'w_seq': torch.stack(w_list, dim=1),    # (B, T, K) skill weights
            'A_k':   self.koopman.get_A_k(),         # (K, m, m)
            'B_k':   self.koopman.get_B_k(),         # (K, m, d_u)
            'U':     self.koopman.U,                 # (m, m)
        }

    @torch.no_grad()
    def rollout(
        self,
        x_cond:  torch.Tensor,   # (B, T_cond, x_dim)  conditioning context
        a_cond:  torch.Tensor,   # (B, T_cond, da)
        a_plan:  torch.Tensor,   # (B, H, da)           planned actions
    ) -> Dict[str, torch.Tensor]:
        """
        Closed-loop rollout in Koopman space.
        Conditions on x_cond/a_cond, then rolls out for H steps.
        Returns predicted x_t components (decoded from o_t).
        """
        B = x_cond.shape[0]
        device = x_cond.device

        # Condition
        h = self.recurrent.init_hidden(B, device)
        o = None
        for t in range(x_cond.shape[1]):
            o, mu, _ = self.posterior.sample(x_cond[:, t], h)
            h = self.recurrent(h, o, a_cond[:, t])

        # Rollout
        o_preds, recon_preds = [], []
        w = self.skill_prior.soft_weights(h)
        for t in range(a_plan.shape[1]):
            u = self.action_encoder(a_plan[:, t])
            o_next, A_bar, B_bar = self.koopman(o, u, w)
            o = o_next
            h = self.recurrent(h, o, a_plan[:, t])
            w = self.skill_prior.soft_weights(h)
            o_preds.append(o)
            recon_preds.append(self.decoder(o))

        # Aggregate decoder outputs
        result = {'o_preds': torch.stack(o_preds, dim=1)}
        for key in ['delta_e', 'delta_p', 'q', 'qdot']:
            result[key] = symexp(torch.stack([r[key] for r in recon_preds], dim=1))
        if self.cfg.use_reward_head and 'reward' in recon_preds[0]:
            result['reward'] = torch.sigmoid(
                torch.stack([r['reward'] for r in recon_preds], dim=1))

        return result