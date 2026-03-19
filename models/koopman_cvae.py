"""
Diagonal Koopman Prior CVAE — v2
=====================================
Changes from v1:
1. Polar latent: z_k^i = A_k^i * exp(j * theta_k^i)
2. Temporal contrastive on patch embeddings (upstream of latent)
3. Multi-step prediction loss (KoVAE-style)
4. KL prior: 'koopman' or 'standard'
5. All loss computation delegated to models/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# ── Import loss functions from losses.py ──────────────────
from models.losses import (
    complex_mul,
    kl_koopman_prior,
    kl_standard_prior,
    multistep_prediction_loss,
    temporal_contrastive_loss,
)

# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # Environment
    action_dim: int   = 6
    state_dim: int    = 24
    patch_size: int   = 5
    dt_control: float = 0.02

    # Architecture
    embed_dim: int       = 128
    state_embed_dim: int = 64
    gru_hidden_dim: int  = 256
    mlp_hidden_dim: int  = 256
    koopman_dim: int     = 64

    # Koopman eigenvalue
    mu_fixed: float  = -0.2
    omega_max: float = math.pi
    mu_min: float    = -0.5
    mu_max: float    = -0.01

    # Loss weights
    beta_kl: float    = 0.1
    alpha_pred: float = 1.0
    gamma_eig: float  = 0.1
    delta_cst: float  = 1.0

    # Multi-step prediction
    pred_steps: int = 5

    # KL prior mode: 'koopman' or 'standard'
    kl_prior: str = 'koopman'

    # Contrastive
    temp_contrastive: float = 0.1
    delta_pos: int = 2
    delta_neg: int = 4

    # Regularization
    dropout: float    = 0.1
    layer_norm: bool  = True


# ─────────────────────────────────────────────
# Koopman Eigenvalues
# ─────────────────────────────────────────────

class KoopmanEigenvalues(nn.Module):
    """
    Diagonal Koopman: lambda_i = mu_fixed + j*omega_i
    mu_fixed: fixed decay,  omega_i: learnable frequency

    Discrete via ZOH: lambda_bar_i = exp(lambda_i * dt_patch)
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m  = cfg.koopman_dim
        self.dt = cfg.patch_size * cfg.dt_control

        self.register_buffer('mu_fixed',
                             torch.full((self.m,), cfg.mu_fixed))

        # Ascending frequency init: omega_i = pi * omega_max / (m+1-i)
        omega_init = torch.tensor([
            math.pi * cfg.omega_max / (self.m + 1 - i)
            for i in range(1, self.m + 1)
        ])
        self.omega     = nn.Parameter(omega_init)
        self.log_sigma = nn.Parameter(torch.zeros(self.m))

    @property
    def sigma_sq(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + 1e-6

    def get_discrete(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (lambda_bar_re, lambda_bar_im), shape (m,)"""
        decay = torch.exp(self.mu_fixed * self.dt)
        return (decay * torch.cos(self.omega * self.dt),
                decay * torch.sin(self.omega * self.dt))

    def get_modulus_angle(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (|lambda_bar|, angle(lambda_bar)), shape (m,)"""
        lb_re, lb_im = self.get_discrete()
        return torch.sqrt(lb_re ** 2 + lb_im ** 2), torch.atan2(lb_im, lb_re)

    def propagate_polar(
        self,
        A: torch.Tensor,      # (..., m)
        theta: torch.Tensor,  # (..., m)
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mod, ang = self.get_modulus_angle()
        return A * (mod ** steps), theta + steps * ang

    def rollout_polar(
        self,
        A: torch.Tensor,      # (B, m)
        theta: torch.Tensor,  # (B, m)
        steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (A_seq, theta_seq), each (B, steps, m)"""
        mod, ang = self.get_modulus_angle()
        hs = torch.arange(1, steps + 1, device=A.device).float()  # (steps,)
        A_seq     = A.unsqueeze(1) * (mod.unsqueeze(0) ** hs.unsqueeze(-1))
        theta_seq = theta.unsqueeze(1) + hs.unsqueeze(-1) * ang.unsqueeze(0)
        return A_seq, theta_seq


# ─────────────────────────────────────────────
# Patch / State Embedding
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.patch_size * cfg.action_dim, cfg.embed_dim)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (B, Np, n*da) → (B, Np, embed_dim)"""
        return self.proj(p)


class StateEmbedding(nn.Module):
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.state_embed_dim, cfg.state_embed_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ─────────────────────────────────────────────
# Encoder — Polar Latent
# ─────────────────────────────────────────────

class KoopmanEncoder(nn.Module):
    """
    GRU input = [patch_emb ; state_emb]
    Output: polar posterior (A, theta, sigma) per patch
      A     = softplus(log_A_net) + eps   > 0
      theta = theta_net                   unbounded
      z     = A*exp(j*theta) + sigma*eps  (reparameterize)
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m   = cfg.koopman_dim
        gru_in   = cfg.embed_dim + cfg.state_embed_dim

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
        )
        self.ln  = nn.LayerNorm(cfg.gru_hidden_dim) if cfg.layer_norm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, 3 * cfg.koopman_dim),
        )

    def forward(
        self,
        p_emb: torch.Tensor,  # (B, Np, embed_dim)
        s_emb: torch.Tensor,  # (B, Np, state_embed_dim)
    ) -> Dict[str, torch.Tensor]:
        x        = torch.cat([p_emb, s_emb], dim=-1)
        h, _     = self.gru(x)
        h        = self.ln(h)
        out      = self.mlp(h)                      # (B, Np, 3m)
        log_A, theta, log_sigma = out.chunk(3, dim=-1)

        A     = F.softplus(log_A)    + 1e-4         # amplitude  > 0
        sigma = F.softplus(log_sigma) + 1e-4        # std        > 0

        eps_re = torch.randn_like(A)
        eps_im = torch.randn_like(A)
        z_re   = A * torch.cos(theta) + (sigma / math.sqrt(2)) * eps_re
        z_im   = A * torch.sin(theta) + (sigma / math.sqrt(2)) * eps_im

        return {
            'A':     A,
            'theta': theta,
            'sigma': sigma,
            'z_re':  z_re,
            'z_im':  z_im,
            'mu_re': A * torch.cos(theta),
            'mu_im': A * torch.sin(theta),
        }


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class KoopmanDecoder(nn.Module):
    """
    Initial hidden = proj([Re(z), Im(z), state_emb])
    GRU input      = state_emb repeated n times
    Output         = action patch in symlog space
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.n  = cfg.patch_size
        self.da = cfg.action_dim

        self.proj = nn.Sequential(
            nn.Linear(2 * cfg.koopman_dim + cfg.state_embed_dim,
                      cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.gru_hidden_dim * 2),
        )
        self.gru = nn.GRU(
            input_size=cfg.state_embed_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
        )
        self.out = nn.Linear(cfg.gru_hidden_dim, cfg.action_dim)

    def forward(
        self,
        z_re: torch.Tensor,  # (B, Np, m)
        z_im: torch.Tensor,
        s_emb: torch.Tensor, # (B, Np, state_embed_dim)
    ) -> torch.Tensor:
        """Returns (B, Np, n, da) in symlog space"""
        B, Np, _ = s_emb.shape

        z_re_f = z_re.reshape(B * Np, -1)
        z_im_f = z_im.reshape(B * Np, -1)
        s_f    = s_emb.reshape(B * Np, -1)

        h_init = self.proj(torch.cat([z_re_f, z_im_f, s_f], dim=-1))
        h_init = h_init.reshape(B * Np, 2, -1).permute(1, 0, 2).contiguous()

        s_seq  = s_f.unsqueeze(1).expand(-1, self.n, -1)
        out, _ = self.gru(s_seq, h_init)
        acts   = self.out(out)                      # (B*Np, n, da)
        return acts.reshape(B, Np, self.n, self.da)


# ─────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    Diagonal Koopman Prior CVAE v2
    - Polar latent: z_k = A_k * exp(j*theta_k)
    - Temporal contrastive on patch embeddings (upstream of latent)
    - Multi-step Koopman prediction loss
    - KL with koopman or standard prior
    - All losses delegated to models/losses.py
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg         = cfg
        self.patch_embed = PatchEmbedding(cfg)
        self.state_embed = StateEmbedding(cfg)
        self.encoder     = KoopmanEncoder(cfg)
        self.decoder     = KoopmanDecoder(cfg)
        self.koopman     = KoopmanEigenvalues(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    # ── Preprocessing ─────────────────────────────────────────

    def preprocess(
        self,
        actions: torch.Tensor,  # (B, T, da)
        states: torch.Tensor,   # (B, T, ds)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n  = self.cfg.patch_size
        B, T, da = actions.shape
        T_crop   = (T // n) * n
        Np       = T_crop // n

        a_norm   = symlog(actions[:, :T_crop, :])
        s_norm   = symlog(states)

        patches  = a_norm.reshape(B, Np, n, da)          # (B, Np, n, da)
        s_patch  = s_norm[:, ::n, :][:, :Np, :]          # (B, Np, ds)

        p_flat   = patches.reshape(B, Np, n * da)
        p_emb    = self.patch_embed(p_flat)               # (B, Np, embed_dim)
        s_emb    = self.state_embed(s_patch)              # (B, Np, state_embed_dim)

        return patches, p_emb, s_emb

    # ── Encode / Decode ───────────────────────────────────────

    def encode(self, p_emb, s_emb):
        return self.encoder(p_emb, s_emb)

    def decode(self, z_re, z_im, s_emb):
        return self.decoder(z_re, z_im, s_emb)

    # ── Forward ───────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        patches, p_emb, s_emb = self.preprocess(actions, states)
        enc   = self.encode(p_emb, s_emb)
        p_hat = self.decode(enc['z_re'], enc['z_im'], s_emb)
        losses = self.compute_losses(patches, p_hat, enc, p_emb)
        return {**losses, 'p_hat': p_hat, **enc}

    # ── Loss: Reconstruction ──────────────────────────────────

    def loss_reconstruction(
        self,
        patches: torch.Tensor,
        p_hat: torch.Tensor,
    ) -> torch.Tensor:
        """MSE in symlog space"""
        return F.mse_loss(p_hat.reshape(*patches.shape), patches)

    # ── Loss: KL — delegates to losses.py ────────────────────

    def loss_kl(self, enc: Dict) -> torch.Tensor:
        """
        Delegates to kl_koopman_prior or kl_standard_prior in losses.py.
        """
        lb_re, lb_im = self.koopman.get_discrete()
        s0_sq        = self.koopman.sigma_sq

        if self.cfg.kl_prior == 'koopman':
            return kl_koopman_prior(
                mu_re=enc['mu_re'],
                mu_im=enc['mu_im'],
                sigma=enc['sigma'],
                z_re=enc['z_re'],
                z_im=enc['z_im'],
                lb_re=lb_re,
                lb_im=lb_im,
                sigma0_sq=s0_sq,
            )
        else:  # 'standard'
            return kl_standard_prior(
                mu_re=enc['mu_re'],
                mu_im=enc['mu_im'],
                sigma=enc['sigma'],
                sigma0_sq=s0_sq,
            )

    # ── Loss: Prediction — delegates to losses.py ─────────────

    def loss_prediction(self, enc: Dict) -> torch.Tensor:
        """
        Delegates to multistep_prediction_loss in losses.py.
        """
        mod, ang = self.koopman.get_modulus_angle()
        return multistep_prediction_loss(
            A=enc['A'],
            theta=enc['theta'],
            mu_re=enc['mu_re'],
            mu_im=enc['mu_im'],
            mod=mod,
            ang=ang,
            H=self.cfg.pred_steps,
        )

    # ── Loss: Contrastive — delegates to losses.py ────────────

    def loss_contrastive(
        self,
        p_emb: torch.Tensor,  # (B, Np, embed_dim) patch embeddings
    ) -> torch.Tensor:
        """
        Delegates to temporal_contrastive_loss in losses.py.
        Applied on patch embeddings (upstream of latent) to prevent
        upstream collapse.
        """
        return temporal_contrastive_loss(
            p_emb=p_emb,
            tau=self.cfg.temp_contrastive,
            delta_pos=self.cfg.delta_pos,
            delta_neg=self.cfg.delta_neg,
        )

    # ── Loss: Eigenvalue Regularization ───────────────────────

    def loss_eigenvalue(self) -> torch.Tensor:
        """Penalize omega drifting far from ascending-frequency init."""
        omega_init = torch.tensor([
            math.pi * self.cfg.omega_max / (self.cfg.koopman_dim + 1 - i)
            for i in range(1, self.cfg.koopman_dim + 1)
        ], device=self.koopman.omega.device)
        drift = (self.koopman.omega - omega_init) / (omega_init.abs() + 1e-6)
        return drift.pow(2).mean()

    # ── Aggregate ─────────────────────────────────────────────

    def compute_losses(
        self,
        patches: torch.Tensor,
        p_hat: torch.Tensor,
        enc: Dict,
        p_emb: torch.Tensor,  # patch embeddings for contrastive
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        loss_recon = self.loss_reconstruction(patches, p_hat)
        loss_kl    = self.loss_kl(enc)
        loss_pred  = self.loss_prediction(enc)
        loss_eig   = self.loss_eigenvalue()
        loss_cst   = self.loss_contrastive(p_emb)   # ← patch embedding 단

        loss_total = (
            loss_recon
            + cfg.beta_kl    * loss_kl
            + cfg.alpha_pred * loss_pred
            + cfg.gamma_eig  * loss_eig
            + cfg.delta_cst  * loss_cst
        )

        return {
            'loss':       loss_total,
            'loss_recon': loss_recon,
            'loss_kl':    loss_kl,
            'loss_pred':  loss_pred,
            'loss_eig':   loss_eig,
            'loss_cst':   loss_cst,
        }

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,            # (B, T_in, ds)
        horizon: Optional[int] = None,
        z0_re: Optional[torch.Tensor] = None,
        z0_im: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        B, T_in, _ = states.shape
        device = states.device
        n  = self.cfg.patch_size
        Np = math.ceil((horizon or T_in) / n)

        s_norm = symlog(states)
        if T_in >= Np * n:
            s_patch = s_norm[:, ::n, :][:, :Np, :]
        else:
            s_avail = s_norm[:, ::n, :]
            pad     = Np - s_avail.shape[1]
            s_last  = s_norm[:, -1:, :].expand(-1, pad, -1)
            s_patch = torch.cat([s_avail, s_last], dim=1)

        s_emb = self.state_embed(s_patch)

        if z0_re is None:
            A0     = torch.abs(torch.randn(B, self.cfg.koopman_dim, device=device))
            theta0 = torch.rand(B, self.cfg.koopman_dim, device=device) * 2 * math.pi
        else:
            A0     = torch.sqrt(z0_re ** 2 + z0_im ** 2 + 1e-8)
            theta0 = torch.atan2(z0_im, z0_re)

        A_seq, theta_seq = self.koopman.rollout_polar(A0, theta0, Np)
        z_re_all = A_seq * torch.cos(theta_seq)
        z_im_all = A_seq * torch.sin(theta_seq)

        p_hat = self.decode(z_re_all, z_im_all, s_emb)
        acts  = symexp(p_hat.reshape(B, Np * n, self.cfg.action_dim))

        return acts[:, :horizon, :] if horizon else acts

    @torch.no_grad()
    def encode_trajectory(self, actions, states):
        patches, p_emb, s_emb = self.preprocess(actions, states)
        enc = self.encode(p_emb, s_emb)
        return enc['z_re'], enc['z_im'], enc['A'], enc['theta']