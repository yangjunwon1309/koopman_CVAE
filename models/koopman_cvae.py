"""
koopman_cvae.py — KODAC (Koopman Diagonal Matrix Prior CVAE)
=============================================================

Theoretical basis:
    ZOH scalar sum formula (diagonal B, Real Schur form):
        g(x_{t+Δt}) = Σ_k v_k · exp[(λ_k + Σ_l β_k^(l) u_l) Δt] · z_k(t)

Latent separation (KEY design):
─────────────────────────────────────────────────────────────
    State stream:   z_s = Φ_s(s_t) ∈ R^{2m}
        Transition: z_s_{t+1} = A · z_s_t           (diagonal ZOH, no u)
        Decode:     ŝ_t = D_s(z_s_t)                (MLP, no skill)

    Action stream:  z_a = Φ_a(a_t) ∈ R^{2m}
        Transition: z_a_{t+1} = (A + B(u)) · z_a_t  (diagonal+LoRA ZOH)
        Decode:     â_t = D_a(z_a_t, v_eff, β_eff)  (MLP conditioned on skill)

    Both streams share eigenvalues {λ_k = μ_k + iω_k}.
    Skills parametrize only the action stream: v_i, β_i.
    State latent transitions are autonomous and skill-agnostic.

Rollout loop (closed):
    z_s_t, z_a_t  →  ŝ_t, â_t
         ↓ ZOH (A for state, A+B·â_t for action)
    z_s_{t+1}, z_a_{t+1}  →  ŝ_{t+1}, â_{t+1}  → ...
─────────────────────────────────────────────────────────────

All loss functions imported from models/losses.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from models.losses import (
    schur_block_propagate,
    schur_block_rollout,
    kl_koopman_prior,
    kl_standard_prior,
    kodac_multistep_prediction_loss,
    reconstruction_loss,
    kodac_contrastive_loss,
    eigenvalue_frequency_repulsion,
    posterior_entropy_regularization,
    mode_diversity_loss,
    decorrelation_loss,
)


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # Environment
    action_dim: int   = 6
    state_dim: int    = 24
    patch_size: int   = 5        # for dt: dt_patch = patch_size * dt_control
    dt_control: float = 0.02

    # Architecture
    mlp_hidden_dim: int   = 256
    gru_hidden_dim: int   = 256
    embed_dim: int        = 128
    koopman_dim: int      = 64   # m: eigenfunction pairs per stream

    # Skill structure
    num_skills: int = 8          # S
    lora_rank: int  = 4          # r: rank of LoRA residual in β

    # Eigenvalue (Real Schur form), shared across both streams
    mu_fixed: float   = -0.2    # fixed decay (buffer, not trained)
    omega_max: float  = math.pi # init grid max frequency

    # Loss weights (pred dominant)
    alpha_pred:    float = 1.0
    alpha_recon_s: float = 0.5
    alpha_recon_a: float = 0.5
    beta_kl:       float = 0.05
    gamma_eig:     float = 0.05
    delta_cst:     float = 0.1
    delta_div:     float = 0.1
    delta_ent:     float = 0.05
    delta_decorr:  float = 0.05

    # Skill mode stability constraints
    # div_margin: cosine similarity threshold below which diversity penalty = 0
    #   (modes are "diverse enough" once cos_sim < margin)
    div_margin:  float = 0.2
    # v_max: max L2 norm of each skill mode vector v_i
    #   prevents norm explosion while allowing direction learning
    v_max:       float = 1.0
    # beta_max: element-wise clamp on beta (input coupling)
    #   |mu_fixed| = 0.2, so |beta| << 1/dt keeps exponent stable
    beta_max:    float = 0.5

    # KL prior
    kl_prior: str = 'koopman'

    # Multi-step prediction horizon
    pred_steps: int = 5

    # Contrastive temperature
    temp_contrastive: float = 0.1

    # Frequency repulsion bandwidth
    freq_repulsion_sigma: float = 0.3

    dropout: float = 0.1


# ─────────────────────────────────────────────────────────────
# Shared Eigenvalue Parameters
# ─────────────────────────────────────────────────────────────

class KoopmanEigenvalues(nn.Module):
    """
    Shared across both state and action streams, and across all skills.

    Real Schur 2x2 block: lam_k = mu_k + i*omega_k
      mu_k    : fixed buffer in (-1, 0)  -> stability by construction
      omega_k : learnable, log-uniform init -> spectral diversity

    Exact ZOH state transition (no Taylor approx, A is diagonal):
        [z_re]         = e^{mu_k dt} [cos w dt  -sin w dt] [z_re]
        [z_im]_{t+dt}                [sin w dt   cos w dt] [z_im]_t
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m  = cfg.koopman_dim
        self.dt = cfg.patch_size * cfg.dt_control

        self.register_buffer('mu', torch.full((self.m,), cfg.mu_fixed))

        omega_init = torch.tensor([
            math.pi * cfg.omega_max / (self.m + 1 - i)
            for i in range(1, self.m + 1)
        ])
        self.omega      = nn.Parameter(omega_init)
        self.log_sigma0 = nn.Parameter(torch.zeros(self.m))

    @property
    def sigma0(self) -> torch.Tensor:
        return F.softplus(self.log_sigma0) + 1e-6

    def get_discrete(self) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.exp(self.mu * self.dt)
        return decay * torch.cos(self.omega * self.dt), \
               decay * torch.sin(self.omega * self.dt)

    def propagate(
        self, z_re: torch.Tensor, z_im: torch.Tensor, steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """State stream: autonomous exact ZOH. (..., m) -> (..., m)"""
        return schur_block_propagate(
            z_re, z_im, self.mu, self.omega, self.dt, steps
        )

    def propagate_with_input(
        self,
        z_re: torch.Tensor,     # (B, m)
        z_im: torch.Tensor,
        beta_eff: torch.Tensor, # (B, m, da)
        u: torch.Tensor,        # (B, da)
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Action stream: ZOH with diagonal+LoRA B.

        Effective exponent per mode k:
            (mu_k + sum_l beta_k^(l) * u_l) * dt

        Diagonal B assumption keeps the block structure intact,
        so the 2x2 Schur block ZOH remains exact per mode.
        """
        beta_u   = torch.bmm(beta_eff, u.unsqueeze(-1)).squeeze(-1)   # (B, m)
        eff_mu   = self.mu.unsqueeze(0) + beta_u                       # (B, m)
        dt       = self.dt * steps
        decay    = torch.exp(eff_mu * dt)                              # (B, m)
        angle    = self.omega.unsqueeze(0) * dt                        # (1, m)
        cos_a    = torch.cos(angle)
        sin_a    = torch.sin(angle)
        z_re_nxt = decay * (cos_a * z_re - sin_a * z_im)
        z_im_nxt = decay * (sin_a * z_re + cos_a * z_im)
        return z_re_nxt, z_im_nxt

    def rollout(
        self, z_re: torch.Tensor, z_im: torch.Tensor, steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autonomous rollout (state stream). Returns (B, steps, m) each."""
        return schur_block_rollout(
            z_re, z_im, self.mu, self.omega, self.dt, steps
        )


# ─────────────────────────────────────────────────────────────
# Stream Encoders (separate MLP per stream)
# ─────────────────────────────────────────────────────────────

class StreamEncoder(nn.Module):
    """
    Generic MLP encoder for one stream.
    input_dim -> [z_re | z_im] in R^{2m}

    Used as:
      phi_s : state_dim  -> 2m   (state eigenfunction)
      phi_a : action_dim -> 2m   (action eigenfunction)
    """

    def __init__(self, input_dim: int, koopman_dim: int,
                 hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.m = koopman_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * koopman_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (..., input_dim) -> z_re, z_im: (..., m)"""
        out = self.net(symlog(x))
        return out.chunk(2, dim=-1)


# ─────────────────────────────────────────────────────────────
# Variational Head (one per stream)
# ─────────────────────────────────────────────────────────────

class VariationalHead(nn.Module):
    """VAE stochasticity on top of deterministic encoder output."""

    def __init__(self, koopman_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * koopman_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * koopman_dim),
        )

    def forward(
        self, z_re: torch.Tensor, z_im: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out = self.net(torch.cat([z_re, z_im], dim=-1))
        mu_re, mu_im, log_sigma = out.chunk(3, dim=-1)
        log_sigma = log_sigma.clamp(-6, 2)
        sigma = log_sigma.exp() + 1e-6
        z_re_s = mu_re + sigma * torch.randn_like(mu_re)
        z_im_s = mu_im + sigma * torch.randn_like(mu_im)
        return {
            'mu_re': mu_re, 'mu_im': mu_im,
            'log_sigma': log_sigma,
            'z_re': z_re_s, 'z_im': z_im_s,
        }


# ─────────────────────────────────────────────────────────────
# Stream Decoders
# ─────────────────────────────────────────────────────────────

class StateDecoder(nn.Module):
    """
    D_s : (z_s_re, z_s_im) -> s_hat in R^{state_dim}

    No skill conditioning. State dynamics are autonomous and
    skill-agnostic, so D_s only receives z_s.

    Output is in symlog space; apply symexp for true state.
    Used in training (recon loss) and rollout (next state).
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.koopman_dim, cfg.mlp_hidden_dim),
            nn.LayerNorm(cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.state_dim),
        )

    def forward(
        self, z_s_re: torch.Tensor, z_s_im: torch.Tensor
    ) -> torch.Tensor:
        """(..., m), (..., m) -> (..., state_dim) [symlog]"""
        return self.net(torch.cat([z_s_re, z_s_im], dim=-1))


class ActionDecoder(nn.Module):
    """
    D_a : (z_a_re, z_a_im, v_eff, beta_eff) -> a_hat in R^{action_dim}

    Skill identity is injected via v_eff and beta_eff:
      - v_eff  in R^m      : Koopman mode (scalar readout per eigenfunction)
      - beta_eff in R^{m x da}: input coupling (flattened and concatenated)

    Same z_a with different (v_eff, beta_eff) -> different action,
    reflecting that two skills can share eigenfunction space but produce
    different observable outputs.

    Output is in symlog space; apply symexp for true action.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m  = cfg.koopman_dim
        self.da = cfg.action_dim
        # Input: [z_a_re | z_a_im | v_eff | beta_eff_flat]
        in_dim = (2 * cfg.koopman_dim
                  + cfg.koopman_dim
                  + cfg.koopman_dim * cfg.action_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden_dim),
            nn.LayerNorm(cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.action_dim),
        )

    def forward(
        self,
        z_a_re: torch.Tensor,   # (..., m)
        z_a_im: torch.Tensor,   # (..., m)
        v_eff: torch.Tensor,    # (B, m)
        beta_eff: torch.Tensor, # (B, m, da)
    ) -> torch.Tensor:
        """Returns a_hat: (..., action_dim) [symlog]"""
        shape = z_a_re.shape[:-1]

        # Broadcast v_eff and beta_eff to match time dimension
        if v_eff.dim() < z_a_re.dim():
            for _ in range(z_a_re.dim() - v_eff.dim()):
                v_eff    = v_eff.unsqueeze(-2)
                beta_eff = beta_eff.unsqueeze(-3)
            v_eff    = v_eff.expand(shape + (self.m,))
            beta_eff = beta_eff.expand(shape + (self.m, self.da))

        beta_flat = beta_eff.reshape(shape + (self.m * self.da,))
        x = torch.cat([z_a_re, z_a_im, v_eff, beta_flat], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Skill-Specific Parameters (action stream only)
# ─────────────────────────────────────────────────────────────

class SkillParameters(nn.Module):
    """
    Per-skill parameters for the ACTION stream only.
    State stream has no skill-specific parameters.

      V in R^{S x m}      : Koopman mode vectors
      beta in R^{S x m x da}: Input coupling (diagonal + low-rank LoRA)
          beta_i = diag(beta_bar_i) + U_i @ VT_i

    Stability constraints:
      - V: max-norm constraint (||v_i||_2 <= v_max) applied at each forward call.
        Prevents norm explosion while allowing direction learning.
        Mode diversity loss (cosine-based) then only affects direction.
      - beta: component-wise clamp to prevent input coupling from
        overwhelming the eigenvalue decay term.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        S, m, da, r = cfg.num_skills, cfg.koopman_dim, cfg.action_dim, cfg.lora_rank

        self.V         = nn.Parameter(torch.randn(S, m) * 0.01)
        self.beta_diag = nn.Parameter(torch.zeros(S, m, da))
        self.beta_U    = nn.Parameter(torch.randn(S, m, r) * 0.01)
        self.beta_VT   = nn.Parameter(torch.randn(S, r, da) * 0.01)
        self.S, self.m, self.da, self.r = S, m, da, r

        # Max-norm for V: ||v_i||_2 <= v_max
        # Set relative to expected z_a norm (~1 after decorrelation constraint)
        self.v_max    = cfg.v_max
        # Max absolute value for beta components
        # |beta_k^(l)| controls effective eigenvalue shift;
        # clamped so |mu_k + beta_k*u| stays in a stable range
        self.beta_max = cfg.beta_max

    def _constrain_V(self) -> torch.Tensor:
        """
        Project V rows onto L2 ball of radius v_max.
        Applied at interpolation time (not in-place on parameter).
        """
        norms = self.V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (norms / self.v_max).clamp(min=1.0)   # only shrink, never grow
        return self.V / scale                           # (S, m)

    def get_beta(self) -> torch.Tensor:
        """
        beta in R^{S x m x da}: diag + low-rank, clamped.
        Clamp prevents beta from overwhelming mu in the exponent:
            exp((mu_k + beta_k * u) * dt)
        """
        beta = self.beta_diag + torch.bmm(self.beta_U, self.beta_VT)
        return beta.clamp(-self.beta_max, self.beta_max)

    def interpolate(
        self, P_hat: torch.Tensor  # (B, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v_eff  = sum_i P_i * v_i   in R^{B x m}   (uses norm-constrained V)
        beta_eff = sum_i P_i * beta_i  in R^{B x m x da}  (uses clamped beta)
        """
        V_constrained = self._constrain_V()                              # (S, m)
        v_eff         = torch.mm(P_hat, V_constrained)                   # (B, m)
        beta_eff      = torch.mm(
            P_hat, self.get_beta().reshape(self.S, -1)
        ).reshape(-1, self.m, self.da)                                   # (B, m, da)
        return v_eff, beta_eff


# ─────────────────────────────────────────────────────────────
# Skill Posterior GRU
# ─────────────────────────────────────────────────────────────

class SkillPosteriorGRU(nn.Module):
    """
    {(s_t, a_t)}_{t<=T} -> P_hat in Delta^{S-1}

    Role: TEMPORAL SKILL IDENTITY.
    Separate from eigenfunction encoders (instantaneous latent state).
    Role separation maintained by contrastive loss against
    skill-conditioned z_a summary (not raw z_a_t).
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim = cfg.state_dim + cfg.action_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.ln            = nn.LayerNorm(cfg.gru_hidden_dim)
        self.skill_head    = nn.Linear(cfg.gru_hidden_dim, cfg.num_skills)
        self.contrast_proj = nn.Linear(cfg.gru_hidden_dim, cfg.koopman_dim)

    def forward(
        self,
        states: torch.Tensor,   # (B, T, ds)
        actions: torch.Tensor,  # (B, T, da)
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(
            torch.cat([symlog(states), symlog(actions)], dim=-1)
        )
        h_seq, _ = self.gru(x)
        h_seq    = self.ln(h_seq)

        P_hat_seq = F.softmax(self.skill_head(h_seq), dim=-1)  # (B, T, S)
        P_hat     = P_hat_seq[:, -1, :]                         # (B, S)
        h_T       = h_seq[:, -1, :]                             # (B, d_h)
        h_T_proj  = self.contrast_proj(h_T)                     # (B, m)

        return {
            'P_hat': P_hat, 'P_hat_seq': P_hat_seq,
            'h_T': h_T, 'h_T_proj': h_T_proj,
        }

    def compute_skill_conditioned_summary(
        self,
        P_hat_seq: torch.Tensor,  # (B, T, S)
        za_re_det: torch.Tensor,  # (B, T, m)  ACTION stream (not state)
    ) -> torch.Tensor:
        """z_bar_{s*} = sum_t P_{s*}^t * za_re_t  (B, m)"""
        s_star  = P_hat_seq[:, -1, :].argmax(dim=-1)
        weights = P_hat_seq.gather(
            2, s_star.unsqueeze(1).unsqueeze(2).expand(-1, P_hat_seq.shape[1], 1)
        ).squeeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return (weights.unsqueeze(-1) * za_re_det).sum(dim=1)


# ─────────────────────────────────────────────────────────────
# Main Model: KODAC
# ─────────────────────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    KODAC — Koopman Diagonal Matrix Prior CVAE

    Two explicit latent streams:
    +----------------------------------------------------------+
    |  STATE STREAM (autonomous, skill-agnostic)               |
    |    phi_s(s_t) -> z_s = (z_s_re, z_s_im) in R^{2m}       |
    |    Transition: z_s_{t+1} = A * z_s_t  (diagonal ZOH)    |
    |    Decode:     D_s(z_s_t) -> s_hat in R^{ds}             |
    +----------------------------------------------------------+
    |  ACTION STREAM (skill-conditioned)                       |
    |    phi_a(a_t) -> z_a = (z_a_re, z_a_im) in R^{2m}       |
    |    Transition: z_a_{t+1} = (A + B(u)) * z_a_t           |
    |                B = diag(beta_bar) + U*VT  (LoRA)         |
    |    Decode:     D_a(z_a_t, v_eff, beta_eff) -> a_hat      |
    +----------------------------------------------------------+
    Shared: eigenvalues {lam_k = mu_k + i*omega_k}
    Skill params: V, beta (action stream only)
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

        # State stream
        self.phi_s = StreamEncoder(cfg.state_dim,  cfg.koopman_dim,
                                   cfg.mlp_hidden_dim, cfg.dropout)
        self.var_s = VariationalHead(cfg.koopman_dim, cfg.mlp_hidden_dim)
        self.dec_s = StateDecoder(cfg)

        # Action stream
        self.phi_a = StreamEncoder(cfg.action_dim, cfg.koopman_dim,
                                   cfg.mlp_hidden_dim, cfg.dropout)
        self.var_a = VariationalHead(cfg.koopman_dim, cfg.mlp_hidden_dim)
        self.dec_a = ActionDecoder(cfg)

        # Shared eigenvalues
        self.koopman = KoopmanEigenvalues(cfg)

        # Skill parameters (action stream only)
        self.skill_params = SkillParameters(cfg)

        # Skill posterior GRU
        self.skill_gru = SkillPosteriorGRU(cfg)

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,  # (B, T, da)
        states: torch.Tensor,   # (B, T, ds)
    ) -> Dict[str, torch.Tensor]:

        # State stream
        zs_re_det, zs_im_det = self.phi_s(states)
        enc_s = self.var_s(zs_re_det, zs_im_det)
        zs_re, zs_im = enc_s['z_re'], enc_s['z_im']

        # Action stream
        za_re_det, za_im_det = self.phi_a(actions)
        enc_a = self.var_a(za_re_det, za_im_det)
        za_re, za_im = enc_a['z_re'], enc_a['z_im']

        # Skill posterior
        skill_out       = self.skill_gru(states, actions)
        P_hat           = skill_out['P_hat']
        v_eff, beta_eff = self.skill_params.interpolate(P_hat)

        # Decode
        s_hat = self.dec_s(zs_re, zs_im)                       # (B, T, ds) symlog
        a_hat = self.dec_a(za_re, za_im, v_eff, beta_eff)       # (B, T, da) symlog

        # Skill-conditioned action-stream summary for contrastive
        z_a_summary = self.skill_gru.compute_skill_conditioned_summary(
            skill_out['P_hat_seq'], za_re_det
        )

        losses = self._compute_losses(
            states=states, actions=actions,
            s_hat=s_hat, a_hat=a_hat,
            zs_re=zs_re, zs_im=zs_im,
            za_re=za_re, za_im=za_im,
            zs_re_det=zs_re_det, za_re_det=za_re_det,
            enc_s=enc_s, enc_a=enc_a,
            v_eff=v_eff, beta_eff=beta_eff,
            P_hat=P_hat,
            h_T_proj=skill_out['h_T_proj'],
            z_a_summary=z_a_summary,
        )

        return {
            **losses,
            's_hat': s_hat, 'a_hat': a_hat,
            'zs_re': zs_re, 'zs_im': zs_im,
            'za_re': za_re, 'za_im': za_im,
            'P_hat': P_hat,
            'v_eff': v_eff, 'beta_eff': beta_eff,
        }

    # ─────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────

    def _compute_losses(
        self,
        states, actions,
        s_hat, a_hat,
        zs_re, zs_im,
        za_re, za_im,
        zs_re_det, za_re_det,
        enc_s, enc_a,
        v_eff, beta_eff,
        P_hat, h_T_proj, z_a_summary,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        # 1. Prediction loss (dominant)
        #    ZOH multi-step in action-stream latent space.
        #    Supervises: phi_a, phi_s (via state info), v_i, beta_i, omega_k, GRU
        loss_pred = kodac_multistep_prediction_loss(
            zs_re=zs_re, zs_im=zs_im,
            za_re=za_re, za_im=za_im,
            v_eff=v_eff, beta_eff=beta_eff,
            actions=actions,
            mu=self.koopman.mu, omega=self.koopman.omega,
            dt=self.koopman.dt, H=cfg.pred_steps,
        )

        # 2. State reconstruction: D_s must recover s from z_s alone
        loss_recon_s = reconstruction_loss(s_hat, symlog(states))

        # 3. Action reconstruction: D_a must recover a from z_a + skill params
        loss_recon_a = reconstruction_loss(a_hat, symlog(actions))

        # 4. KL: state stream (prior = autonomous Koopman or standard)
        if cfg.kl_prior == 'koopman':
            loss_kl_s = kl_koopman_prior(
                mu_re=enc_s['mu_re'], mu_im=enc_s['mu_im'],
                log_sigma=enc_s['log_sigma'],
                z_re=enc_s['z_re'], z_im=enc_s['z_im'],
                mu_k=self.koopman.mu, omega=self.koopman.omega,
                dt=self.koopman.dt, log_sigma0=self.koopman.log_sigma0,
            )
        else:
            loss_kl_s = kl_standard_prior(
                mu_re=enc_s['mu_re'], mu_im=enc_s['mu_im'],
                log_sigma=enc_s['log_sigma'],
                log_sigma0=self.koopman.log_sigma0,
            )

        # 5. KL: action stream (autonomous prior as approximation)
        if cfg.kl_prior == 'koopman':
            loss_kl_a = kl_koopman_prior(
                mu_re=enc_a['mu_re'], mu_im=enc_a['mu_im'],
                log_sigma=enc_a['log_sigma'],
                z_re=enc_a['z_re'], z_im=enc_a['z_im'],
                mu_k=self.koopman.mu, omega=self.koopman.omega,
                dt=self.koopman.dt, log_sigma0=self.koopman.log_sigma0,
            )
        else:
            loss_kl_a = kl_standard_prior(
                mu_re=enc_a['mu_re'], mu_im=enc_a['mu_im'],
                log_sigma=enc_a['log_sigma'],
                log_sigma0=self.koopman.log_sigma0,
            )

        loss_kl = loss_kl_s + loss_kl_a

        # 6. Eigenvalue frequency repulsion
        loss_eig = eigenvalue_frequency_repulsion(
            self.koopman.omega, sigma=cfg.freq_repulsion_sigma
        )

        # 7. Contrastive: GRU h_T vs skill-conditioned z_a summary
        loss_cst = kodac_contrastive_loss(
            gru_hidden=h_T_proj, skill_z_summary=z_a_summary,
            tau=cfg.temp_contrastive,
        )

        # 8. Skill mode diversity (cosine-based, bounded, with margin)
        loss_div = mode_diversity_loss(
            self.skill_params.V,
            margin=cfg.div_margin,
        )

        # 9. Posterior entropy (anti-collapse)
        loss_ent = posterior_entropy_regularization(P_hat)

        # 10. Decorrelation (gauge uniqueness, both streams)
        loss_decorr = 0.5 * (
            decorrelation_loss(zs_re_det.reshape(-1, cfg.koopman_dim))
            + decorrelation_loss(za_re_det.reshape(-1, cfg.koopman_dim))
        )

        loss = (
            cfg.alpha_pred    * loss_pred
            + cfg.alpha_recon_s * loss_recon_s
            + cfg.alpha_recon_a * loss_recon_a
            + cfg.beta_kl     * loss_kl
            + cfg.gamma_eig   * loss_eig
            + cfg.delta_cst   * loss_cst
            + cfg.delta_div   * loss_div
            + cfg.delta_ent   * loss_ent
            + cfg.delta_decorr * loss_decorr
        )

        return {
            'loss':          loss,
            'loss_pred':     loss_pred,
            'loss_recon_s':  loss_recon_s,
            'loss_recon_a':  loss_recon_a,
            'loss_kl':       loss_kl,
            'loss_eig':      loss_eig,
            'loss_cst':      loss_cst,
            'loss_div':      loss_div,
            'loss_ent':      loss_ent,
            'loss_decorr':   loss_decorr,
        }

    # ─────────────────────────────────────────────────────────
    # Rollout (closed-loop)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        states: torch.Tensor,   # (B, T_cond, ds)
        actions: torch.Tensor,  # (B, T_cond, da)
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Closed-loop rollout for `horizon` steps.

        State stream:   z_s_{t+1} = A * z_s_t         (no u, exact ZOH)
        Action stream:  z_a_{t+1} = (A + B(a_hat_t)) * z_a_t
                        a_hat_t = D_a(z_a_t, v_eff, beta_eff)  [self-consistent]

        Returns:
          s_preds: (B, horizon, ds)
          a_preds: (B, horizon, da)
        """
        self.eval()
        B = states.shape[0]

        # Get skill posterior from conditioning window
        skill_out       = self.skill_gru(states, actions)
        P_hat           = skill_out['P_hat']
        v_eff, beta_eff = self.skill_params.interpolate(P_hat)

        # Initialize from last conditioning step
        zs_re_seq, zs_im_seq = self.phi_s(states)
        za_re_seq, za_im_seq = self.phi_a(actions)
        zs_re = zs_re_seq[:, -1]  # (B, m)
        zs_im = zs_im_seq[:, -1]
        za_re = za_re_seq[:, -1]  # (B, m)
        za_im = za_im_seq[:, -1]

        s_preds, a_preds = [], []

        for _ in range(horizon):
            # Decode current latent states
            s_t = symexp(self.dec_s(zs_re, zs_im))                    # (B, ds)
            a_t = symexp(self.dec_a(za_re, za_im, v_eff, beta_eff))   # (B, da)
            s_preds.append(s_t)
            a_preds.append(a_t)

            # Propagate state stream (autonomous)
            zs_re, zs_im = self.koopman.propagate(zs_re, zs_im)

            # Propagate action stream (with predicted action as input u)
            za_re, za_im = self.koopman.propagate_with_input(
                za_re, za_im, beta_eff, a_t
            )

        return {
            's_preds': torch.stack(s_preds, dim=1),  # (B, horizon, ds)
            'a_preds': torch.stack(a_preds, dim=1),  # (B, horizon, da)
        }

    @torch.no_grad()
    def predict_skill(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        return self.skill_gru(states, actions)['P_hat']

    @torch.no_grad()
    def encode_trajectory(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        zs_re, zs_im = self.phi_s(states)
        za_re, za_im = self.phi_a(actions)
        P_hat        = self.skill_gru(states, actions)['P_hat']
        v_eff, beta_eff = self.skill_params.interpolate(P_hat)
        return {
            'zs_re': zs_re, 'zs_im': zs_im,
            'za_re': za_re, 'za_im': za_im,
            'P_hat': P_hat, 'v_eff': v_eff, 'beta_eff': beta_eff,
            'omega': self.koopman.omega, 'mu': self.koopman.mu,
        }