"""
Diagonal Koopman Prior CVAE
============================
Action sequence modeling conditioned on robot states.
Compatible with: DMControl, Isaac Gym, D4RL (Adroit), HumanoidBench

Architecture:
  - Symlog normalization for actions and states
  - Action patch tokenization
  - Symmetric GRU encoder/decoder (state as GRU input, not just init)
  - Diagonal Koopman prior: CN(diag(lambda) z_{k-1}, Sigma)
  - Complex-valued latent z_k in C^m
  - Learnable imaginary eigenvalues, fixed real eigenvalues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class KoopmanCVAEConfig:
    # Environment dimensions
    action_dim: int = 6              # d_a
    state_dim: int = 24              # d_s

    # Patch settings
    patch_size: int = 5              # n (timesteps per patch)
    dt_control: float = 0.02         # control period in seconds (1/Hz)

    # Architecture
    embed_dim: int = 128             # d: patch embedding dim
    state_embed_dim: int = 64        # state embedding dim
    gru_hidden_dim: int = 256        # GRU hidden size
    mlp_hidden_dim: int = 256        # MLP hidden size
    koopman_dim: int = 64            # m: complex latent dim

    # Koopman eigenvalue settings
    mu_fixed: float = -0.2           # fixed real part of continuous eigenvalue
    omega_max: float = math.pi       # max frequency for init
    mu_min: float = -0.5             # stability constraint lower bound
    mu_max: float = -0.01            # stability constraint upper bound

    # Loss weights
    beta_kl: float = 1.0
    alpha_pred: float = 1.0
    gamma_eig: float = 0.1
    delta_cst: float = 1.0

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True

    temp_contrastive: float = 0.1    # temperature
    delta_min: int = 3   


# ─────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def complex_mul(a_re, a_im, b_re, b_im):
    """Element-wise complex multiplication: (a_re + j*a_im)(b_re + j*b_im)"""
    re = a_re * b_re - a_im * b_im
    im = a_re * b_im + a_im * b_re
    return re, im


# ─────────────────────────────────────────────
#  Koopman Eigenvalue Module
# ─────────────────────────────────────────────

class KoopmanEigenvalues(nn.Module):
    """
    Diagonal Koopman eigenvalues: lambda_i = mu_i + j * omega_i
    - mu_i: fixed constant in [-0.5, -0.01]
    - omega_i: learnable, initialized with ascending frequencies
    - Discretized via ZOH: lambda_bar_i = exp(lambda_i * dt)
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m = cfg.koopman_dim
        self.mu = cfg.mu_fixed
        self.dt = cfg.patch_size * cfg.dt_control   # Delta t per patch

        # Fixed real part (no gradient)
        mu_tensor = torch.full((self.m,), cfg.mu_fixed)
        self.register_buffer('mu_fixed', mu_tensor)

        # Learnable imaginary part: omega_i = pi * omega_max / (m+1-i)
        omega_init = torch.tensor([
            math.pi * cfg.omega_max / (self.m + 1 - i)
            for i in range(1, self.m + 1)
        ])
        self.omega = nn.Parameter(omega_init)

        # Learnable process noise: Sigma = diag(sigma_1^2, ..., sigma_m^2)
        self.log_sigma = nn.Parameter(torch.zeros(self.m))

    @property
    def sigma_sq(self) -> torch.Tensor:
        """Process noise variance (always positive)"""
        return F.softplus(self.log_sigma) + 1e-6

    def get_discrete_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (lambda_bar_re, lambda_bar_im) via ZOH discretization.
        lambda_bar_i = exp(mu_i * dt) * (cos(omega_i * dt) + j*sin(omega_i * dt))
        """
        dt = self.dt
        decay = torch.exp(self.mu_fixed * dt)          # e^{mu * dt}, shape (m,)
        lambda_bar_re = decay * torch.cos(self.omega * dt)
        lambda_bar_im = decay * torch.sin(self.omega * dt)
        return lambda_bar_re, lambda_bar_im

    def propagate(
        self,
        z_re: torch.Tensor,  # (B, m)
        z_im: torch.Tensor,  # (B, m)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prior mean: mu_0 = diag(lambda_bar) z_{k-1}
        Returns (mu0_re, mu0_im), each (B, m)
        """
        lb_re, lb_im = self.get_discrete_eigenvalues()  # (m,)
        mu0_re, mu0_im = complex_mul(lb_re, lb_im, z_re, z_im)
        return mu0_re, mu0_im

    def rollout(
        self,
        z_re: torch.Tensor,   # (B, m)
        z_im: torch.Tensor,   # (B, m)
        steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step deterministic rollout: z_{k+tau} = diag(lambda_bar)^tau * z_k
        Returns (z_re_seq, z_im_seq), each (B, steps, m)
        O(m * steps), fully parallel via Vandermonde
        """
        lb_re, lb_im = self.get_discrete_eigenvalues()  # (m,)
        # tau = 1, ..., steps
        taus = torch.arange(1, steps + 1, device=z_re.device).float()  # (steps,)

        # |lambda_bar|^tau and angle * tau
        decay = torch.exp(self.mu_fixed * self.dt)           # (m,)
        decay_tau = decay.unsqueeze(0) ** taus.unsqueeze(1)  # (steps, m)
        angle = torch.atan2(lb_im, lb_re)                    # (m,)
        angle_tau = angle.unsqueeze(0) * taus.unsqueeze(1)   # (steps, m)

        lb_re_tau = decay_tau * torch.cos(angle_tau)  # (steps, m)
        lb_im_tau = decay_tau * torch.sin(angle_tau)  # (steps, m)

        # z_re: (B, m) -> broadcast with (steps, m)
        z_re_seq, z_im_seq = complex_mul(
            lb_re_tau.unsqueeze(0),   # (1, steps, m)
            lb_im_tau.unsqueeze(0),
            z_re.unsqueeze(1),        # (B, 1, m)
            z_im.unsqueeze(1),
        )
        return z_re_seq, z_im_seq  # (B, steps, m)


# ─────────────────────────────────────────────
#  Patch Embedding
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Flatten action patch and project to embedding dim"""
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.action_dim = cfg.action_dim
        patch_flat_dim = cfg.patch_size * cfg.action_dim
        self.proj = nn.Linear(patch_flat_dim, cfg.embed_dim)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: (B, Np, patch_size * action_dim) or (B, Np, patch_size, action_dim)
        Returns: (B, Np, embed_dim)
        """
        if p.dim() == 4:
            B, Np, n, da = p.shape
            p = p.reshape(B, Np, n * da)
        return self.proj(p)


# ─────────────────────────────────────────────
#  State Embedding
# ─────────────────────────────────────────────

class StateEmbedding(nn.Module):
    """Embed symlog-normalized state into fixed-size vector"""
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.state_embed_dim, cfg.state_embed_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, Np, state_dim) -> (B, Np, state_embed_dim)"""
        return self.net(s)


# ─────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────

class KoopmanEncoder(nn.Module):
    """
    Symmetric encoder:
      GRU input = [patch_embed ; state_embed]  (action + state together)
    Outputs posterior parameters: mu_re, mu_im, log_sigma per patch
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.m = cfg.koopman_dim
        gru_input_dim = cfg.embed_dim + cfg.state_embed_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
        )

        if cfg.layer_norm:
            self.ln = nn.LayerNorm(cfg.gru_hidden_dim)
        else:
            self.ln = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, 3 * cfg.koopman_dim),  # mu_re, mu_im, log_sigma
        )

    def forward(
        self,
        p_emb: torch.Tensor,   # (B, Np, embed_dim)
        s_emb: torch.Tensor,   # (B, Np, state_embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mu_re:    (B, Np, m)
          mu_im:    (B, Np, m)
          sigma:    (B, Np, m)  -- shared for re and im
        """
        x = torch.cat([p_emb, s_emb], dim=-1)   # (B, Np, embed+state_embed)
        h, _ = self.gru(x)                        # (B, Np, gru_hidden)
        h = self.ln(h)
        out = self.mlp(h)                          # (B, Np, 3m)

        mu_re, mu_im, log_sigma = out.chunk(3, dim=-1)   # each (B, Np, m)
        sigma = F.softplus(log_sigma) + 1e-4

        return mu_re, mu_im, sigma


# ─────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────

class KoopmanDecoder(nn.Module):
    """
    Symmetric decoder:
      GRU input = [zero_action ; state_embed]  (zero + state together)
      GRU initial hidden state = projected latent z_k
    Outputs reconstructed action patch
    """
    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.action_dim = cfg.action_dim
        self.m = cfg.koopman_dim

        latent_real_dim = 2 * cfg.koopman_dim  # Re(z) + Im(z)
        gru_input_dim = cfg.state_embed_dim     # zero action + state_embed
        # (zero action part is just zeros, same dim as state_embed for symmetry)

        # Project latent + state to GRU init hidden state
        self.proj = nn.Sequential(
            nn.Linear(latent_real_dim + cfg.state_embed_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.gru_hidden_dim * 2),  # for 2-layer GRU
        )

        # GRU: unroll patch_size steps
        # Input at each step: state_embed (zeros for action part, symmetric with encoder)
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout if cfg.dropout > 0 else 0,
        )

        # Output: action per step
        self.out = nn.Linear(cfg.gru_hidden_dim, cfg.action_dim)

    def forward(
        self,
        z_re: torch.Tensor,   # (B, Np, m)
        z_im: torch.Tensor,   # (B, Np, m)
        s_emb: torch.Tensor,  # (B, Np, state_embed_dim)
    ) -> torch.Tensor:
        """
        Returns reconstructed action patches: (B, Np, patch_size, action_dim)
        in symlog space
        """
        B, Np, _ = s_emb.shape

        # Flatten patches into batch for parallel processing
        z_re_flat = z_re.reshape(B * Np, self.m)
        z_im_flat = z_im.reshape(B * Np, self.m)
        s_flat = s_emb.reshape(B * Np, -1)

        # Project latent to GRU initial hidden state
        latent = torch.cat([z_re_flat, z_im_flat, s_flat], dim=-1)  # (B*Np, 2m+state_embed)
        h_init = self.proj(latent)  # (B*Np, gru_hidden*2)
        # Split into 2 layers: (2, B*Np, gru_hidden)
        h_init = h_init.reshape(B * Np, 2, -1).permute(1, 0, 2).contiguous()

        # GRU input: repeat state embedding for each step, zero action (symmetric)
        # Encoder uses [action_patch ; state_embed], decoder uses [zeros ; state_embed]
        s_seq = s_flat.unsqueeze(1).expand(-1, self.patch_size, -1)  # (B*Np, n, state_embed)

        out, _ = self.gru(s_seq, h_init)  # (B*Np, n, gru_hidden)
        actions = self.out(out)            # (B*Np, n, action_dim)

        return actions.reshape(B, Np, self.patch_size, self.action_dim)


# ─────────────────────────────────────────────
#  Main Model
# ─────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    Diagonal Koopman Prior CVAE for robot action sequence modeling.

    P(a_{1:T} | s_{1:T}) modeled via structured latent dynamics:
      - z_k in C^m follows diagonal Koopman prior
      - State s_k conditions encoder and decoder symmetrically

    Compatible environments:
      - DMControl: action_dim=1~21, state_dim=6~67, dt=0.02s
      - Isaac Gym: action_dim=6~23, state_dim=32~200, dt=0.0167s
      - D4RL Adroit: action_dim=24~30, state_dim=39~46, dt=0.04s
      - HumanoidBench: action_dim=19~56, state_dim=76~350, dt=0.01s
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg)
        self.state_embed = StateEmbedding(cfg)
        self.encoder = KoopmanEncoder(cfg)
        self.decoder = KoopmanDecoder(cfg)
        self.koopman = KoopmanEigenvalues(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def preprocess(
        self,
        actions: torch.Tensor,   # (B, T, action_dim)
        states: torch.Tensor,    # (B, T, state_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Symlog normalization
        2. Patch tokenization for actions
        3. State selection per patch (start of each patch)
        Returns:
          patches: (B, Np, patch_size, action_dim) -- symlog space
          patch_emb: (B, Np, embed_dim)
          state_emb: (B, Np, state_embed_dim)
        """
        n = self.cfg.patch_size
        B, T, da = actions.shape

        # Symlog normalize
        a_norm = symlog(actions)
        s_norm = symlog(states)

        # Truncate to multiple of patch_size
        T_crop = (T // n) * n
        a_norm = a_norm[:, :T_crop, :]     # (B, T_crop, da)
        Np = T_crop // n

        # Reshape to patches: (B, Np, n, da)
        patches = a_norm.reshape(B, Np, n, da)

        # Select state at start of each patch
        s_patch = s_norm[:, ::n, :][:, :Np, :]   # (B, Np, ds)

        # Embeddings
        p_flat = patches.reshape(B, Np, n * da)
        patch_emb = self.patch_embed.proj(p_flat)   # (B, Np, embed_dim)
        state_emb = self.state_embed(s_patch)        # (B, Np, state_embed_dim)

        return patches, patch_emb, state_emb

    def encode(
        self,
        patch_emb: torch.Tensor,  # (B, Np, embed_dim)
        state_emb: torch.Tensor,  # (B, Np, state_embed_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns posterior parameters and sampled latents.
        """
        mu_re, mu_im, sigma = self.encoder(patch_emb, state_emb)
        # (B, Np, m) each

        # Reparameterize: z = mu + (1/sqrt(2)) * eps * sigma
        eps_re = torch.randn_like(mu_re)
        eps_im = torch.randn_like(mu_im)
        z_re = mu_re + (1.0 / math.sqrt(2)) * eps_re * sigma
        z_im = mu_im + (1.0 / math.sqrt(2)) * eps_im * sigma

        return {
            'mu_re': mu_re,     # (B, Np, m)
            'mu_im': mu_im,     # (B, Np, m)
            'sigma': sigma,     # (B, Np, m)
            'z_re': z_re,       # (B, Np, m)
            'z_im': z_im,       # (B, Np, m)
        }

    def decode(
        self,
        z_re: torch.Tensor,    # (B, Np, m)
        z_im: torch.Tensor,    # (B, Np, m)
        state_emb: torch.Tensor,  # (B, Np, state_embed_dim)
    ) -> torch.Tensor:
        """
        Returns reconstructed action patches in symlog space.
        Shape: (B, Np, patch_size, action_dim)
        """
        return self.decoder(z_re, z_im, state_emb)

    def forward(
        self,
        actions: torch.Tensor,   # (B, T, action_dim)
        states: torch.Tensor,    # (B, T, state_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass. Returns dict with losses and intermediates.
        """
        patches, patch_emb, state_emb = self.preprocess(actions, states)
        enc = self.encode(patch_emb, state_emb)

        # Decode
        p_hat = self.decode(enc['z_re'], enc['z_im'], state_emb)
        # p_hat: (B, Np, n, da)  -- symlog space

        # Losses
        losses = self.compute_losses(patches, p_hat, enc, patch_emb, state_emb)

        return {**losses, 'p_hat': p_hat, **enc}

    def compute_losses(
        self,
        patches: torch.Tensor,   # (B, Np, n, da)  symlog space
        p_hat: torch.Tensor,     # (B, Np, n, da)  symlog space
        enc: Dict[str, torch.Tensor],
        patch_emb: torch.Tensor,
        state_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        B, Np, n, da = patches.shape
        mu_re  = enc['mu_re']    # (B, Np, m)
        mu_im  = enc['mu_im']
        sigma  = enc['sigma']
        z_re   = enc['z_re']
        z_im   = enc['z_im']

        # ── 1. Reconstruction Loss ──────────────────────────────────────
        p_flat = patches.reshape(B, Np, -1)
        p_hat_flat = p_hat.reshape(B, Np, -1)
        loss_recon = F.mse_loss(p_hat_flat, p_flat, reduction='mean')

        # ── 2. KL Divergence ────────────────────────────────────────────
        # Prior mean: mu_0_{k,i} = lambda_bar_i * z_{k-1,i}
        # Use sampled z for prior mean (as in original ELBO derivation)
        lb_re, lb_im = self.koopman.get_discrete_eigenvalues()  # (m,)
        sigma_sq = self.koopman.sigma_sq                         # (m,)

        # Shift: z_{k-1} for k=2,...,Np
        z_prev_re = z_re[:, :-1, :]   # (B, Np-1, m)
        z_prev_im = z_im[:, :-1, :]

        mu0_re, mu0_im = complex_mul(
            lb_re, lb_im,
            z_prev_re, z_prev_im,
        )  # (B, Np-1, m)

        # Posterior params for k=2,...,Np
        mu_re_k = mu_re[:, 1:, :]    # (B, Np-1, m)
        mu_im_k = mu_im[:, 1:, :]
        sigma_k = sigma[:, 1:, :]    # (B, Np-1, m)

        # KL = ||mu_re - mu0_re||^2 / sigma0^2
        #    + ||mu_im - mu0_im||^2 / sigma0^2
        #    + sigma_k^2 / sigma0^2
        #    - ln(sigma_k^2 / sigma0^2) - 1
        diff_re_sq = (mu_re_k - mu0_re).pow(2)   # (B, Np-1, m)
        diff_im_sq = (mu_im_k - mu0_im).pow(2)

        # sigma0^2 broadcast: (m,) -> (1, 1, m)
        s0_sq = sigma_sq.unsqueeze(0).unsqueeze(0)
        sk_sq = sigma_k.pow(2)

        kl_per = (
            (diff_re_sq + diff_im_sq) / s0_sq
            + sk_sq / s0_sq
            - torch.log(sk_sq / s0_sq + 1e-8)
            - 1.0
        )  # (B, Np-1, m)
        loss_kl = kl_per.mean()

        # ── 3. Linearity Promotion Loss ─────────────────────────────────
        # L_pred = sum_{k=2} sum_i |mu_{k,i} - lambda_bar_i * mu_{k-1,i}|^2
        mu_prev_re = mu_re[:, :-1, :]
        mu_prev_im = mu_im[:, :-1, :]

        mu0_re_det, mu0_im_det = complex_mul(
            lb_re, lb_im,
            mu_prev_re, mu_prev_im,
        )  # (B, Np-1, m)

        loss_pred = (
            (mu_re_k - mu0_re_det).pow(2) +
            (mu_im_k - mu0_im_det).pow(2)
        ).mean()

        # ── 4. Eigenvalue Regularization ────────────────────────────────
        # Regularize omega away from drifting too far from initialization
        # (mu is fixed so only omega matters)
        omega_init = torch.tensor([
            math.pi * self.cfg.omega_max / (self.cfg.koopman_dim + 1 - i)
            for i in range(1, self.cfg.koopman_dim + 1)
        ], device=mu_re.device)
        omega_drift = (self.koopman.omega - omega_init) / (omega_init + 1e-6)
        loss_eig = omega_drift.pow(2).mean()

        # ── 5. Constrastive Regularization ────────────────────────────────
        loss_cst = self.compute_contrastive_loss(patch_emb, state_emb, enc)
        # ── Total Loss ───────────────────────────────────────────────────

        cfg = self.cfg
        loss_total = (
            loss_recon
            + cfg.beta_kl * loss_kl
            + cfg.alpha_pred * loss_pred
            + cfg.gamma_eig * loss_eig
            + cfg.delta_cst * loss_cst
        )

        return {
            'loss': loss_total,
            'loss_recon': loss_recon,
            'loss_kl': loss_kl,
            'loss_pred': loss_pred,
            'loss_eig': loss_eig,
            'loss_cst' : loss_cst
        }
    
    def compute_contrastive_loss(
        self,
        patch_emb: torch.Tensor,   # (B, Np, embed_dim)
        state_emb: torch.Tensor,   # (B, Np, state_embed_dim)
        enc: dict,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss on latent z.
        Positive: temporal jitter augmentation of same patch
        Negative: patches with |j-k| > delta_min
        """
        B, Np, _ = patch_emb.shape
        tau = self.cfg.temp_contrastive
        delta = self.cfg.delta_min
        m = self.cfg.koopman_dim

        mu_re = enc['mu_re']   # (B, Np, m)
        mu_im = enc['mu_im']   # (B, Np, m)

        # Flatten z_k as query: concat Re+Im, L2 normalize
        z_flat = torch.cat([mu_re, mu_im], dim=-1)          # (B, Np, 2m)
        z_norm = F.normalize(z_flat, dim=-1)                 # (B, Np, 2m)

        # Positive: same patch with Gaussian noise augmentation (temporal jitter)
        noise = torch.randn_like(patch_emb) * 0.05
        p_aug = patch_emb + noise
        mu_re_aug, mu_im_aug, _ = self.encoder(p_aug, state_emb)
        z_aug = torch.cat([mu_re_aug, mu_im_aug], dim=-1)
        z_aug_norm = F.normalize(z_aug, dim=-1)              # (B, Np, 2m)

        loss_cst = torch.tensor(0.0, device=patch_emb.device)
        count = 0

        for k in range(Np):
            q = z_norm[:, k, :]          # (B, 2m) query
            pos = z_aug_norm[:, k, :]    # (B, 2m) positive

            # Negatives: patches far enough in time
            neg_idx = [j for j in range(Np) if abs(j - k) > delta]
            if len(neg_idx) < 2:
                continue

            negs = z_norm[:, neg_idx, :]   # (B, N_neg, 2m)

            # Similarity: (B, 1+N_neg)
            sim_pos = (q * pos).sum(dim=-1, keepdim=True) / tau      # (B, 1)
            sim_neg = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / tau  # (B, N_neg)

            logits = torch.cat([sim_pos, sim_neg], dim=-1)   # (B, 1+N_neg)
            labels = torch.zeros(B, dtype=torch.long, device=q.device)
            loss_cst += F.cross_entropy(logits, labels)
            count += 1

        return loss_cst / max(count, 1)

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,   # (B, T, state_dim) -- full horizon or just s_1
        horizon: Optional[int] = None,
        z1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Generate action sequence given states.
        If only s_1 is provided (T=1), uses Koopman rollout for z_k.
        Returns actions in original (symexp) space: (B, T_out, action_dim)
        """
        self.eval()
        B = states.shape[0]
        device = states.device

        s_norm = symlog(states)
        T_in = s_norm.shape[1]
        n = self.cfg.patch_size

        # Determine number of patches
        if horizon is None:
            Np = T_in // n
        else:
            Np = math.ceil(horizon / n)

        # State embedding per patch
        if T_in >= Np * n:
            s_patch = s_norm[:, ::n, :][:, :Np, :]
        else:
            # Repeat last state
            s_last = s_norm[:, -1:, :].expand(-1, Np, -1)
            s_patch_avail = s_norm[:, ::n, :]
            pad = Np - s_patch_avail.shape[1]
            s_patch = torch.cat([
                s_patch_avail,
                s_last[:, :pad, :]
            ], dim=1)

        state_emb = self.state_embed(s_patch)  # (B, Np, state_embed_dim)

        # Sample or use provided z_1
        if z1 is None:
            # Sample z_1 from prior CN(0, I)
            z_re = torch.randn(B, self.cfg.koopman_dim, device=device) / math.sqrt(2)
            z_im = torch.randn(B, self.cfg.koopman_dim, device=device) / math.sqrt(2)
        else:
            z_re, z_im = z1

        # Koopman rollout: z_1, ..., z_Np
        z_re_seq, z_im_seq = self.koopman.rollout(z_re, z_im, Np)
        # Prepend z_1: (B, Np, m)
        z_re_all = torch.cat([z_re.unsqueeze(1), z_re_seq[:, :-1, :]], dim=1)
        z_im_all = torch.cat([z_im.unsqueeze(1), z_im_seq[:, :-1, :]], dim=1)

        # Decode
        p_hat = self.decode(z_re_all, z_im_all, state_emb)
        # (B, Np, n, da)

        # Reshape and symexp
        actions_symlog = p_hat.reshape(B, Np * n, self.cfg.action_dim)
        if horizon is not None:
            actions_symlog = actions_symlog[:, :horizon, :]

        return symexp(actions_symlog)

    @torch.no_grad()
    def encode_trajectory(
        self,
        actions: torch.Tensor,   # (B, T, action_dim)
        states: torch.Tensor,    # (B, T, state_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory to latent sequence.
        Returns (z_re, z_im): (B, Np, m) each
        """
        patches, patch_emb, state_emb = self.preprocess(actions, states)
        enc = self.encode(patch_emb, state_emb)
        return enc['z_re'], enc['z_im']
