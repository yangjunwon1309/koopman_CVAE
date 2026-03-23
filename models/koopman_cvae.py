"""
koopman_cvae.py — KODAC-S
==========================

Simplified KODAC with Full A + Low-rank B.

Removed from previous version:
  - Diagonal A / Real Schur 2x2 block structure
  - Action stream encoder + decoder (action is given control input only)
  - VAE posterior (mu, sigma, KL, reparameterization)
  - Skill-specific V_i, beta_i, discrete P_hat
  - GRU skill posterior
  - loss_kl, loss_div, loss_ent, loss_cst, loss_recon_a

New structure:
  State stream:   z_t = tanh(MLP_Phi(s_t)) in R^m
  Transition:     z_{t+1} = F(a_t) z_t,  F = I + (A + sum_l B^(l) a_l) * dt
                  A in R^{m x m}: full learnable
                  B^(l) = U^(l) @ V^(l).T: low-rank per action dim
  Multi-head:     v^(n)_t = W^(n) c_t,  c_t = TCN({s,a}_{1:t})
                  g^(n)_t = v^(n)_t · z_t  (observable readout)
  Decoder:        s_hat = MLP_Ds(z_t)
  Loss:           L = alpha*L_pred + beta*L_recon + gamma*L_eig + delta*L_decorr
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from models.losses import (
    propagate,
    propagate_h_steps,
    multistep_prediction_loss,
    reconstruction_loss,
    eigenvalue_stability_loss,
    eigenvalue_diversity_loss,
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
    patch_size: int   = 5        # dt = patch_size * dt_control
    dt_control: float = 0.02

    # Architecture
    mlp_hidden_dim: int  = 256
    tcn_hidden_dim: int  = 256   # TCN channel width
    tcn_n_layers:   int  = 4     # TCN depth (number of residual blocks)
    tcn_kernel_size: int = 3     # TCN causal kernel size
    koopman_dim:    int  = 64    # m: z dimension
    num_heads:      int  = 8     # Nh: number of observable heads

    # Full A + Low-rank B
    lora_rank: int   = 8         # r: rank of B^(l) = U^(l) @ V^(l).T
    # B magnitude clamp: |B^(l)| <= b_max element-wise
    b_max: float     = 0.3

    # Eigenvalue stability
    eig_target_radius: float = 0.99   # target spectral radius of A
    eig_margin:        float = 0.01   # soft margin
    eig_div_sigma:     float = 0.1    # diversity penalty bandwidth

    # Loss weights
    alpha_pred:   float = 1.0
    alpha_recon:  float = 0.5
    gamma_eig:    float = 0.1    # stability loss
    gamma_div:    float = 0.05   # eigenvalue diversity loss
    delta_decorr: float = 0.1

    # Multi-step prediction horizon
    pred_steps: int = 5

    dropout: float = 0.1


# ─────────────────────────────────────────────────────────────
# Koopman Operator: Full A + Low-rank B
# ─────────────────────────────────────────────────────────────

class KoopmanOperator(nn.Module):
    """
    Full transition matrix A in R^{m x m}
    Input coupling B^(l) = U^(l) @ V^(l).T in R^{m x m}, l=1,...,da
    (low-rank, da separate matrices)

    ZOH (1st-order):  z_{t+1} = (I + (A + sum_l B^(l)*a_l) * dt) z_t
                               = F(a_t) z_t

    Stability enforced via eigenvalue_stability_loss on A (not hard constraint).
    A is NOT constrained during forward — loss penalizes unstable eigenvalues.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        m, da, r = cfg.koopman_dim, cfg.action_dim, cfg.lora_rank

        self.m   = m
        self.da  = da
        self.r   = r
        self.dt  = cfg.patch_size * cfg.dt_control
        self.b_max = cfg.b_max

        # Full A: initialized near zero to start close to identity transition
        self.A = nn.Parameter(torch.randn(m, m) * 0.01)

        # Low-rank B: B^(l) = B_U[l] @ B_V[l].T
        self.B_U = nn.Parameter(torch.randn(da, m, r) * 0.01)
        self.B_V = nn.Parameter(torch.randn(da, m, r) * 0.01)

    def get_B_clamped(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return B_U, B_V with magnitude soft-clamped via tanh."""
        # tanh keeps values bounded without hard clamp discontinuity
        B_U = torch.tanh(self.B_U) * self.b_max
        B_V = torch.tanh(self.B_V) * self.b_max
        return B_U, B_V

    def get_B_full(self) -> torch.Tensor:
        """Returns B: (da, m, m) — full low-rank matrices."""
        B_U, B_V = self.get_B_clamped()
        return torch.bmm(B_U, B_V.transpose(-1, -2))              # (da, m, m)

    def forward(
        self,
        z: torch.Tensor,    # (batch, m) or (B, T, m)
        a: torch.Tensor,    # (batch, da) or (B, T, da)
        steps: int = 1,
    ) -> torch.Tensor:
        """Single or multi-step ZOH propagation."""
        B_U, B_V = self.get_B_clamped()
        shape = z.shape[:-1]

        # Flatten leading dims
        z_flat = z.reshape(-1, self.m)
        a_flat = a.reshape(-1, self.da)

        if steps == 1:
            z_next = propagate(z_flat, self.A, B_U, B_V, a_flat, self.dt)
        else:
            z_next = propagate_h_steps(z_flat, self.A, B_U, B_V,
                                       a_flat, self.dt, steps)
        return z_next.reshape(shape + (self.m,))

    def rollout(
        self,
        z0: torch.Tensor,       # (B, m)
        actions: torch.Tensor,  # (B, T, da)
        horizon: int,
    ) -> torch.Tensor:
        """
        Closed-loop rollout: z_{t+1} = F(a_t) z_t
        Returns z_seq: (B, horizon, m)
        """
        B_U, B_V = self.get_B_clamped()
        z_list = []
        z = z0
        for t in range(horizon):
            a = actions[:, t] if t < actions.shape[1] else actions[:, -1]
            z = propagate(z, self.A, B_U, B_V, a, self.dt)
            z_list.append(z)
        return torch.stack(z_list, dim=1)                          # (B, horizon, m)

    def get_eigenvalues(self) -> torch.Tensor:
        """Returns complex eigenvalues of A, shape (m,)."""
        return torch.linalg.eigvals(self.A)


# ─────────────────────────────────────────────────────────────
# State Eigenfunction Encoder
# ─────────────────────────────────────────────────────────────

class EigenfunctionEncoder(nn.Module):
    """
    MLP: s_t -> z_t in R^m
    tanh output keeps z bounded in (-1, 1) per element.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.mlp_hidden_dim),
            nn.LayerNorm(cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.LayerNorm(cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.koopman_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (..., ds) -> z: (..., m)"""
        return torch.tanh(self.net(symlog(s)))


# ─────────────────────────────────────────────────────────────
# State Decoder
# ─────────────────────────────────────────────────────────────

class StateDecoder(nn.Module):
    """z_t -> s_hat in symlog space."""

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.koopman_dim, cfg.mlp_hidden_dim),
            nn.LayerNorm(cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., m) -> s_hat: (..., ds) [symlog]"""
        return self.net(z)


# ─────────────────────────────────────────────────────────────
# TCN Skill Encoder → Multi-head Observable
# ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Single causal (left-padded) 1D convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C_out, T)"""
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        # LayerNorm over channel dim
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        return self.drop(x)


class TCNSkillEncoder(nn.Module):
    """
    Causal TCN: {(s_t, a_t)} -> c_t in R^{d_c} for each t.
    Multi-head readout: v^(n)_t = W^(n) c_t in R^m, n=1,...,Nh.

    c_t encodes temporal skill context at time t.
    v^(n)_t is a time-varying observable direction in z-space.
    Different heads see z from different directions simultaneously,
    providing richer supervision of the Koopman structure.
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        in_dim  = cfg.state_dim + cfg.action_dim
        d_c     = cfg.tcn_hidden_dim
        Nh      = cfg.num_heads
        m       = cfg.koopman_dim

        # Input projection
        self.input_proj = nn.Conv1d(in_dim, d_c, kernel_size=1)

        # Stacked causal convolutions with exponentially growing dilation
        self.layers = nn.ModuleList([
            CausalConv1d(d_c, d_c,
                         kernel_size=cfg.tcn_kernel_size,
                         dilation=2**i,
                         dropout=cfg.dropout)
            for i in range(cfg.tcn_n_layers)
        ])

        # Residual projections (if channel mismatch; here same channel)
        # Multi-head readout: Nh separate linear maps
        # W^(n) c_t -> v^(n)_t in R^m
        self.head_projs = nn.Linear(d_c, Nh * m)

        self.d_c = d_c
        self.Nh  = Nh
        self.m   = m

    def forward(
        self,
        states:  torch.Tensor,  # (B, T, ds)
        actions: torch.Tensor,  # (B, T, da)
    ) -> torch.Tensor:
        """
        Returns v_heads: (B, T, Nh, m)
        v_heads[b, t, n, :] = v^(n)_t = W^(n) c_t
        """
        # (B, T, ds+da) -> (B, ds+da, T) for Conv1d
        x = torch.cat([symlog(states), symlog(actions)], dim=-1)
        x = x.transpose(1, 2)                                     # (B, in_dim, T)
        x = F.silu(self.input_proj(x))                            # (B, d_c, T)

        for layer in self.layers:
            x = x + layer(x)                                       # residual

        # (B, d_c, T) -> (B, T, d_c)
        c = x.transpose(1, 2)                                      # (B, T, d_c)

        # Multi-head projection
        v_flat = self.head_projs(c)                                # (B, T, Nh*m)
        v_heads = v_flat.reshape(c.shape[0], c.shape[1],
                                 self.Nh, self.m)                  # (B, T, Nh, m)
        return v_heads


# ─────────────────────────────────────────────────────────────
# Main Model: KODAC-S
# ─────────────────────────────────────────────────────────────

class KoopmanCVAE(nn.Module):
    """
    KODAC-S — Simplified Koopman CVAE

    z_t = Phi(s_t)                        state eigenfunction
    z_{t+1} = F(a_t) z_t                  full A + low-rank B ZOH
    v^(n)_t = W^(n) TCN({s,a}_{1:t})     multi-head observable
    g^(n)_t = v^(n)_t · z_t              observable readout
    s_hat_t = D_s(z_t)                    state decoder

    Loss:
        L_pred   : MSE(v^(n) · F^h z_t, v^(n) · z_{t+h})  multi-head
        L_recon  : MSE(D_s(z_t), symlog(s_t))
        L_eig    : eigenvalue stability of A
        L_div    : eigenvalue diversity of A
        L_decorr : off-diagonal cosine similarity of z
    """

    def __init__(self, cfg: KoopmanCVAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder  = EigenfunctionEncoder(cfg)
        self.decoder  = StateDecoder(cfg)
        self.koopman  = KoopmanOperator(cfg)
        self.tcn      = TCNSkillEncoder(cfg)

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,  # (B, T, da)
        states:  torch.Tensor,  # (B, T, ds)
    ) -> Dict[str, torch.Tensor]:

        # Eigenfunction encoding
        z = self.encoder(states)                                   # (B, T, m)

        # Multi-head observable readout from TCN
        v_heads = self.tcn(states, actions)                        # (B, T, Nh, m)

        # State reconstruction
        s_hat = self.decoder(z)                                    # (B, T, ds)

        # Losses
        losses = self._compute_losses(
            states=states, z=z, s_hat=s_hat,
            v_heads=v_heads, actions=actions,
        )

        return {**losses, 'z': z, 's_hat': s_hat, 'v_heads': v_heads}

    # ── Loss ─────────────────────────────────────────────────

    def _compute_losses(
        self,
        states:   torch.Tensor,   # (B, T, ds)
        z:        torch.Tensor,   # (B, T, m)
        s_hat:    torch.Tensor,   # (B, T, ds)
        v_heads:  torch.Tensor,   # (B, T, Nh, m)
        actions:  torch.Tensor,   # (B, T, da)
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B_U, B_V = self.koopman.get_B_clamped()

        # 1. Prediction loss (dominant)
        #    Jointly supervises: Phi, A, B, TCN heads
        loss_pred = multistep_prediction_loss(
            z=z,
            v_heads=v_heads,
            actions=actions,
            A=self.koopman.A,
            B_U=B_U,
            B_V=B_V,
            dt=self.koopman.dt,
            H=cfg.pred_steps,
        )

        # 2. State reconstruction
        loss_recon = reconstruction_loss(s_hat, symlog(states))

        # 3. Eigenvalue stability: penalize |lambda_k(A)| > target_radius
        loss_eig_stab = eigenvalue_stability_loss(
            self.koopman.A,
            target_radius=cfg.eig_target_radius,
            margin=cfg.eig_margin,
        )

        # 4. Eigenvalue diversity: penalize clustering of eigenvalues
        loss_eig_div = eigenvalue_diversity_loss(
            self.koopman.A,
            sigma=cfg.eig_div_sigma,
        )

        loss_eig = loss_eig_stab + cfg.gamma_div / cfg.gamma_eig * loss_eig_div

        # 5. Decorrelation: z columns should be uncorrelated
        loss_decorr = decorrelation_loss(z.reshape(-1, cfg.koopman_dim))

        loss = (
            cfg.alpha_pred  * loss_pred
            + cfg.alpha_recon * loss_recon
            + cfg.gamma_eig   * loss_eig
            + cfg.delta_decorr * loss_decorr
        )

        return {
            'loss':           loss,
            'loss_pred':      loss_pred,
            'loss_recon':     loss_recon,
            'loss_eig_stab':  loss_eig_stab,
            'loss_eig_div':   loss_eig_div,
            'loss_decorr':    loss_decorr,
        }

    # ── Rollout ───────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        states:   torch.Tensor,  # (B, T_cond, ds)
        actions:  torch.Tensor,  # (B, T_cond + horizon, da)
        horizon:  int,
    ) -> Dict[str, torch.Tensor]:
        """
        Closed-loop state prediction.
        Uses given actions for ZOH transition.
        Returns s_preds: (B, horizon, ds)
        """
        self.eval()
        B = states.shape[0]
        T_cond = states.shape[1]

        # Encode last conditioning step
        z = self.encoder(states)                                   # (B, T_cond, m)
        z_cur = z[:, -1]                                           # (B, m)

        s_preds = []
        for t in range(horizon):
            a_t = actions[:, T_cond + t] if (T_cond + t) < actions.shape[1] \
                  else actions[:, -1]
            # Propagate
            B_U, B_V = self.koopman.get_B_clamped()
            z_cur = propagate(z_cur, self.koopman.A, B_U, B_V, a_t,
                              self.koopman.dt)
            s_hat = symexp(self.decoder(z_cur))                    # (B, ds)
            s_preds.append(s_hat)

        return {'s_preds': torch.stack(s_preds, dim=1)}

    @torch.no_grad()
    def encode_trajectory(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z       = self.encoder(states)
        v_heads = self.tcn(states, actions)
        eigvals = self.koopman.get_eigenvalues()
        return {
            'z':       z,
            'v_heads': v_heads,
            'A':       self.koopman.A,
            'B_full':  self.koopman.get_B_full(),
            'eigvals': eigvals,
        }