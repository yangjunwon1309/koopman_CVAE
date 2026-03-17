"""
Diagonal Koopman Prior CVAE — v2
=====================================
Changes from v1:
  1. Polar latent: z_k^i = A_k^i * exp(j * theta_k^i)
     - Encoder outputs amplitude A and phase theta separately
     - More natural for Koopman rotation: exp(lambda*dt) = |lambda|*exp(j*angle)

  2. Temporal contrastive loss (time-series aware):
     - Positive: patches within |j-k| <= delta_pos (temporally adjacent)
     - Negative: patches with |j-k| >= delta_neg
     - No augmentation needed

  3. Multi-step prediction loss (KoVAE-style):
     - For each anchor patch k: predict z_{k+1}, z_{k+2}, ..., z_{k+H}
       using z_k * lambda_bar^1, lambda_bar^2, ..., lambda_bar^H
     - MSE between predicted and true latents

  4. KL prior options:
     - 'koopman': prior = CN(lambda_bar * z_{k-1}, Sigma)  [dynamic prior]
     - 'standard': prior = CN(0, Sigma)                    [simple regularization]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def complex_mul(a_re, a_im, b_re, b_im):
    """Element-wise complex multiply: (a_re+j*a_im)(b_re+j*b_im)"""
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # Environment
    action_dim: int = 6
    state_dim: int = 24
    patch_size: int = 5
    dt_control: float = 0.02

    # Architecture
    embed_dim: int = 128
    state_embed_dim: int = 64
    gru_hidden_dim: int = 256
    mlp_hidden_dim: int = 256
    koopman_dim: int = 64

    # Koopman eigenvalue
    mu_fixed: float = -0.2
    omega_max: float = math.pi
    mu_min: float = -0.5
    mu_max: float = -0.01

    # Loss weights
    beta_kl: float = 0.1
    alpha_pred: float = 1.0
    gamma_eig: float = 0.1
    delta_cst: float = 1.0

    # Multi-step prediction
    pred_steps: int = 5          # H: predict up to H steps ahead

    # KL prior mode: 'koopman' or 'standard'
    kl_prior: str = 'koopman'

    # Contrastive
    temp_contrastive: float = 0.1
    delta_pos: int = 2           # |j-k| <= delta_pos → positive
    delta_neg: int = 4           # |j-k| >= delta_neg → negative

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True


# ─────────────────────────────────────────────
#  Koopman Eigenvalues
# ─────────────────────────────────────────────

class KoopmanEigenvalues(nn.Module):
    """
    Diagonal Koopman: lambda_i = mu_i + j*omega_i
    mu_i: fixed,  omega_i: learnable
    Discrete via ZOH: lambda_bar_i = exp(lambda_i * dt)
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m  = cfg.koopman_dim
        self.dt = cfg.patch_size * cfg.dt_control

        self.register_buffer('mu_fixed',
            torch.full((self.m,), cfg.mu_fixed))

        # Ascending frequency initialization: omega_i = pi*omega_max / (m+1-i)
        omega_init = torch.tensor([
            math.pi * cfg.omega_max / (self.m + 1 - i)
            for i in range(1, self.m + 1)
        ])
        self.omega = nn.Parameter(omega_init)

        # Learnable process noise
        self.log_sigma = nn.Parameter(torch.zeros(self.m))

    @property
    def sigma_sq(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + 1e-6

    def get_discrete(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (lambda_bar_re, lambda_bar_im), shape (m,)"""
        decay = torch.exp(self.mu_fixed * self.dt)
        return decay * torch.cos(self.omega * self.dt), \
               decay * torch.sin(self.omega * self.dt)

    def get_modulus_angle(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (|lambda_bar|, angle), shape (m,)"""
        lb_re, lb_im = self.get_discrete()
        return torch.sqrt(lb_re**2 + lb_im**2), torch.atan2(lb_im, lb_re)

    def propagate_polar(
        self,
        A: torch.Tensor,      # (..., m) amplitude
        theta: torch.Tensor,  # (..., m) phase
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Koopman propagation in polar:
          A_{k+h} = |lambda_bar|^h * A_k
          theta_{k+h} = theta_k + h * angle(lambda_bar)
        Returns (A_new, theta_new) same shape as input.
        """
        mod, ang = self.get_modulus_angle()  # (m,)
        A_new     = A     * (mod ** steps)
        theta_new = theta + steps * ang
        return A_new, theta_new

    def rollout_polar(
        self,
        A: torch.Tensor,      # (B, m)
        theta: torch.Tensor,  # (B, m)
        steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (A_seq, theta_seq), each (B, steps, m)
        z_{k+h} = |lambda_bar|^h * A_k * exp(j*(theta_k + h*ang))
        """
        mod, ang = self.get_modulus_angle()       # (m,)
        hs = torch.arange(1, steps+1,
                          device=A.device).float() # (steps,)

        # (B, steps, m)
        A_seq     = A.unsqueeze(1) * (mod.unsqueeze(0) ** hs.unsqueeze(-1))
        theta_seq = theta.unsqueeze(1) + hs.unsqueeze(-1) * ang.unsqueeze(0)
        return A_seq, theta_seq


# ─────────────────────────────────────────────
#  Patch Embedding
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
#  Encoder — Polar Latent
# ─────────────────────────────────────────────

class KoopmanEncoder(nn.Module):
    """
    Symmetric encoder: GRU input = [patch_emb ; state_emb]
    Outputs polar posterior: (log_A, theta, log_sigma) per patch

    Reparameterization:
      A     = softplus(log_A_net) + eps    (amplitude > 0)
      theta = theta_net                    (phase, unbounded)
      z_re  = A * cos(theta)
      z_im  = A * sin(theta)
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m = cfg.koopman_dim
        gru_in = cfg.embed_dim + cfg.state_embed_dim

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
        )
        self.ln = nn.LayerNorm(cfg.gru_hidden_dim) if cfg.layer_norm else nn.Identity()

        # Output: log_A (m), theta (m), log_sigma (m)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, 3 * cfg.koopman_dim),
        )

    def forward(
        self,
        p_emb: torch.Tensor,   # (B, Np, embed_dim)
        s_emb: torch.Tensor,   # (B, Np, state_embed_dim)
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([p_emb, s_emb], dim=-1)
        h, _ = self.gru(x)
        h = self.ln(h)
        out = self.mlp(h)                              # (B, Np, 3m)
        log_A, theta, log_sigma = out.chunk(3, dim=-1)

        # Amplitude: always positive
        A     = F.softplus(log_A) + 1e-4              # (B, Np, m)
        sigma = F.softplus(log_sigma) + 1e-4          # (B, Np, m)

        # Reparameterize: z = A*exp(j*theta) + sigma*eps
        eps_re = torch.randn_like(A)
        eps_im = torch.randn_like(A)

        z_re = A * torch.cos(theta) + (sigma / math.sqrt(2)) * eps_re
        z_im = A * torch.sin(theta) + (sigma / math.sqrt(2)) * eps_im

        return {
            'A': A,           # amplitude (B, Np, m)
            'theta': theta,   # phase     (B, Np, m)
            'sigma': sigma,   # std       (B, Np, m)
            'z_re': z_re,     # (B, Np, m)
            'z_im': z_im,     # (B, Np, m)
            # posterior mean in re/im for losses
            'mu_re': A * torch.cos(theta),
            'mu_im': A * torch.sin(theta),
        }


# ─────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────

class KoopmanDecoder(nn.Module):
    """
    Symmetric decoder: GRU input = [zeros ; state_emb]
    Initial hidden = projected (Re(z), Im(z), state_emb)
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
        z_re: torch.Tensor,     # (B, Np, m)
        z_im: torch.Tensor,     # (B, Np, m)
        s_emb: torch.Tensor,    # (B, Np, state_embed_dim)
    ) -> torch.Tensor:
        """Returns (B, Np, n, da) in symlog space"""
        B, Np, _ = s_emb.shape

        z_re_f = z_re.reshape(B * Np, -1)
        z_im_f = z_im.reshape(B * Np, -1)
        s_f    = s_emb.reshape(B * Np, -1)

        h_init = self.proj(torch.cat([z_re_f, z_im_f, s_f], dim=-1))
        h_init = h_init.reshape(B * Np, 2, -1).permute(1, 0, 2).contiguous()

        s_seq = s_f.unsqueeze(1).expand(-1, self.n, -1)
        out, _ = self.gru(s_seq, h_init)
        acts = self.out(out)                   # (B*Np, n, da)
        return acts.reshape(B, Np, self.n, self.da)


# ─────────────────────────────────────────────
#  Main Model
# ─────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    Diagonal Koopman Prior CVAE v2
    - Polar latent: z_k = A_k * exp(j*theta_k)
    - Temporal contrastive (time-proximity as positive)
    - Multi-step Koopman prediction loss
    - KL with koopman or standard prior
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

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

    # ── Preprocessing ────────────────────────────────────────

    def preprocess(
        self,
        actions: torch.Tensor,   # (B, T, da)
        states: torch.Tensor,    # (B, T, ds)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.cfg.patch_size
        B, T, da = actions.shape

        T_crop = (T // n) * n
        Np     = T_crop // n

        a_norm = symlog(actions[:, :T_crop, :])
        s_norm = symlog(states)

        patches = a_norm.reshape(B, Np, n, da)       # (B, Np, n, da)
        s_patch = s_norm[:, ::n, :][:, :Np, :]       # (B, Np, ds)

        p_flat   = patches.reshape(B, Np, n * da)
        p_emb    = self.patch_embed(p_flat)           # (B, Np, embed_dim)
        s_emb    = self.state_embed(s_patch)          # (B, Np, state_embed_dim)

        return patches, p_emb, s_emb

    # ── Encode / Decode ──────────────────────────────────────

    def encode(self, p_emb, s_emb):
        return self.encoder(p_emb, s_emb)

    def decode(self, z_re, z_im, s_emb):
        return self.decoder(z_re, z_im, s_emb)

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        patches, p_emb, s_emb = self.preprocess(actions, states)
        enc    = self.encode(p_emb, s_emb)
        p_hat  = self.decode(enc['z_re'], enc['z_im'], s_emb)
        losses = self.compute_losses(patches, p_hat, enc, p_emb, s_emb)
        return {**losses, 'p_hat': p_hat, **enc}

    # ── Loss: Reconstruction ─────────────────────────────────

    def loss_reconstruction(self, patches, p_hat):
        """MSE in symlog space"""
        return F.mse_loss(
            p_hat.reshape(*patches.shape),
            patches
        )

    # ── Loss: KL Divergence ──────────────────────────────────

    def loss_kl(self, enc: Dict) -> torch.Tensor:
        """
        Two modes (cfg.kl_prior):
          'koopman' : prior = CN(lambda_bar * z_{k-1}, Sigma)
          'standard': prior = CN(0+0j, Sigma)

        KL(CN(mu_hat, sigma_hat^2) || CN(mu0, sigma0^2))
          = (||mu_hat_re - mu0_re||^2 + ||mu_hat_im - mu0_im||^2) / sigma0^2
          + sigma_hat^2 / sigma0^2 - log(sigma_hat^2/sigma0^2) - 1
        """
        mu_re = enc['mu_re']   # (B, Np, m)
        mu_im = enc['mu_im']
        sigma = enc['sigma']   # (B, Np, m)
        z_re  = enc['z_re']
        z_im  = enc['z_im']

        s0_sq = self.koopman.sigma_sq.unsqueeze(0).unsqueeze(0)  # (1,1,m)
        sk_sq = sigma ** 2

        if self.cfg.kl_prior == 'koopman':
            # Prior mean: lambda_bar * z_{k-1}  for k=2,...,Np
            lb_re, lb_im = self.koopman.get_discrete()
            mu0_re, mu0_im = complex_mul(
                lb_re, lb_im,
                z_re[:, :-1, :], z_im[:, :-1, :]
            )
            mu_re_k = mu_re[:, 1:, :]
            mu_im_k = mu_im[:, 1:, :]
            sk_sq_k = sk_sq[:, 1:, :]

            kl = (
                ((mu_re_k - mu0_re)**2 + (mu_im_k - mu0_im)**2) / (s0_sq + 1e-8)
                + sk_sq_k / (s0_sq + 1e-8)
                - torch.log(sk_sq_k / (s0_sq + 1e-8) + 1e-8)
                - 1.0
            )

        else:  # 'standard': CN(0, Sigma)
            mu0_re = torch.zeros_like(mu_re)
            mu0_im = torch.zeros_like(mu_im)

            kl = (
                (mu_re**2 + mu_im**2) / (s0_sq + 1e-8)
                + sk_sq / (s0_sq + 1e-8)
                - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
                - 1.0
            )

        return kl.mean()

    # ── Loss: Multi-step Prediction ──────────────────────────

    def loss_prediction(self, enc: Dict) -> torch.Tensor:
        """
        KoVAE-style multi-step prediction loss:
        For each anchor k and step h=1,...,H:
          predicted: z_k * lambda_bar^h   (in polar: A_k*|lambda|^h, theta_k + h*angle)
          target:    z_{k+h}
          loss:      MSE(pred_re - true_re)^2 + MSE(pred_im - true_im)^2

        Total: mean over all valid (k, h) pairs
        """
        A      = enc['A']      # (B, Np, m)
        theta  = enc['theta']  # (B, Np, m)
        mu_re  = enc['mu_re']  # (B, Np, m)  = A*cos(theta)
        mu_im  = enc['mu_im']  # (B, Np, m)  = A*sin(theta)

        B, Np, m = A.shape
        H = min(self.cfg.pred_steps, Np - 1)

        mod, ang = self.koopman.get_modulus_angle()  # (m,)

        total_loss = torch.tensor(0.0, device=A.device)
        count = 0

        for h in range(1, H + 1):
            # Predicted: Koopman propagation h steps from each anchor k
            # anchor range: k = 0, ..., Np-h-1
            # target range: k+h = h, ..., Np-1

            A_anchor     = A[:, :Np-h, :]        # (B, Np-h, m)
            theta_anchor = theta[:, :Np-h, :]

            # Polar propagation: |lambda|^h * amplitude, theta + h*angle
            A_pred     = A_anchor     * (mod ** h)              # (B, Np-h, m)
            theta_pred = theta_anchor + h * ang                  # (B, Np-h, m)

            pred_re = A_pred * torch.cos(theta_pred)
            pred_im = A_pred * torch.sin(theta_pred)

            true_re = mu_re[:, h:, :]             # (B, Np-h, m)
            true_im = mu_im[:, h:, :]

            loss_h = (pred_re - true_re)**2 + (pred_im - true_im)**2
            # Normalize by Frobenius norm (as in KoVAE)
            norm_h = torch.sqrt((pred_re**2 + pred_im**2).sum() + 1e-8)
            total_loss += loss_h.mean() / norm_h
            count += 1

        return total_loss / max(count, 1)

    # ── Loss: Temporal Contrastive ───────────────────────────

    def loss_contrastive(
        self,
        enc: Dict,
    ) -> torch.Tensor:
        """
        Time-series aware InfoNCE:
          Positive pairs: patches with |j-k| <= delta_pos
          Negative pairs: patches with |j-k| >= delta_neg

        Uses posterior means (mu_re, mu_im) as representations.
        No augmentation — temporal proximity is the positive signal.
        """
        mu_re = enc['mu_re']   # (B, Np, m)
        mu_im = enc['mu_im']

        B, Np, m = mu_re.shape
        tau      = self.cfg.temp_contrastive
        dp       = self.cfg.delta_pos
        dn       = self.cfg.delta_neg

        # L2-normalize [Re; Im] as feature vector
        z_flat = torch.cat([mu_re, mu_im], dim=-1)        # (B, Np, 2m)
        z_norm = F.normalize(z_flat, dim=-1)              # (B, Np, 2m)

        total_loss = torch.tensor(0.0, device=mu_re.device)
        count = 0

        for k in range(Np):
            pos_idx = [j for j in range(Np) if 0 < abs(j - k) <= dp]
            neg_idx = [j for j in range(Np) if abs(j - k) >= dn]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            q    = z_norm[:, k, :]                        # (B, 2m)
            pos  = z_norm[:, pos_idx, :]                  # (B, n_pos, 2m)
            negs = z_norm[:, neg_idx, :]                  # (B, n_neg, 2m)

            # Similarities
            sim_pos = torch.bmm(pos,  q.unsqueeze(-1)).squeeze(-1) / tau  # (B, n_pos)
            sim_neg = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / tau  # (B, n_neg)

            # InfoNCE: label = mean over positives
            # Use logsumexp formulation
            log_pos = torch.logsumexp(sim_pos, dim=-1)              # (B,)
            log_all = torch.logsumexp(
                torch.cat([sim_pos, sim_neg], dim=-1), dim=-1       # (B,)
            )
            total_loss += (log_all - log_pos).mean()
            count += 1

        return total_loss / max(count, 1)

    # ── Loss: Eigenvalue Regularization ──────────────────────

    def loss_eigenvalue(self) -> torch.Tensor:
        """Regularize omega away from drifting too far"""
        omega_init = torch.tensor([
            math.pi * self.cfg.omega_max / (self.cfg.koopman_dim + 1 - i)
            for i in range(1, self.cfg.koopman_dim + 1)
        ], device=self.koopman.omega.device)
        drift = (self.koopman.omega - omega_init) / (omega_init.abs() + 1e-6)
        return drift.pow(2).mean()

    # ── Aggregate Losses ─────────────────────────────────────

    def compute_losses(
        self,
        patches:  torch.Tensor,
        p_hat:    torch.Tensor,
        enc:      Dict,
        p_emb:    torch.Tensor,
        s_emb:    torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        loss_recon = self.loss_reconstruction(patches, p_hat)
        loss_kl    = self.loss_kl(enc)
        loss_pred  = self.loss_prediction(enc)
        loss_eig   = self.loss_eigenvalue()
        loss_cst   = self.loss_contrastive(enc)

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

    # ── Inference ────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,    # (B, T_in, ds)
        horizon: Optional[int] = None,
        z0_re: Optional[torch.Tensor] = None,
        z0_im: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate action sequence from initial state.
        If z0 not provided, samples from prior (A~Exp(1), theta~Uniform).
        Returns actions in original space: (B, T_out, da)
        """
        self.eval()
        B, T_in, _ = states.shape
        device = states.device
        n  = self.cfg.patch_size
        Np = math.ceil((horizon or T_in) / n)

        # State embedding
        s_norm = symlog(states)
        if T_in >= Np * n:
            s_patch = s_norm[:, ::n, :][:, :Np, :]
        else:
            s_last = s_norm[:, -1:, :].expand(-1, Np, -1)
            s_avail = s_norm[:, ::n, :]
            pad = Np - s_avail.shape[1]
            s_patch = torch.cat([s_avail, s_last[:, :pad, :]], dim=1)
        s_emb = self.state_embed(s_patch)  # (B, Np, state_embed_dim)

        # Initial latent
        if z0_re is None:
            # Sample from prior: A ~ Rayleigh(1), theta ~ Uniform(0, 2pi)
            A0    = torch.abs(torch.randn(B, self.cfg.koopman_dim, device=device))
            theta0 = torch.rand(B, self.cfg.koopman_dim, device=device) * 2 * math.pi
            z0_re = A0 * torch.cos(theta0)
            z0_im = A0 * torch.sin(theta0)

        # Koopman rollout
        A0 = torch.sqrt(z0_re**2 + z0_im**2 + 1e-8)
        theta0 = torch.atan2(z0_im, z0_re)
        A_seq, theta_seq = self.koopman.rollout_polar(A0, theta0, Np)

        z_re_all = A_seq * torch.cos(theta_seq)   # (B, Np, m)
        z_im_all = A_seq * torch.sin(theta_seq)

        # Decode
        p_hat = self.decode(z_re_all, z_im_all, s_emb)  # (B, Np, n, da)
        actions_symlog = p_hat.reshape(B, Np * n, self.cfg.action_dim)
        if horizon is not None:
            actions_symlog = actions_symlog[:, :horizon, :]

        return symexp(actions_symlog)

    @torch.no_grad()
    def encode_trajectory(self, actions, states):
        patches, p_emb, s_emb = self.preprocess(actions, states)
        enc = self.encode(p_emb, s_emb)
        return enc['z_re'], enc['z_im'], enc['A'], enc['theta']