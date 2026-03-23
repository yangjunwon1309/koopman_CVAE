"""
losses.py — KODAC-S loss functions
===================================

Architecture change from diagonal to Full A + Low-rank B:

  REMOVED assumptions:
    - A = diag(lambda_k)  [exact eigenfunction basis]
    - B = diag(beta_k^(l)) [no cross-mode coupling]
    - Real Schur 2x2 block structure
    - schur_block_propagate / schur_block_rollout
    - KL divergence, reparameterization

  NEW structure:
    - A in R^{m x m}: full learnable transition matrix
    - B^(l) = U^(l) @ V^(l).T in R^{m x m}: low-rank per action dim
    - ZOH: z_{t+1} = (I + M(a_t)*dt) z_t,  M = A + sum_l B^(l) a_l
                     (1st-order Taylor; exact when dt is small)
    - eigenvalue_stability_loss: penalize eigenvalues of A outside unit disk
    - decorrelation_loss: off-diagonal cosine similarity (scale-invariant)
    - reconstruction_loss: MSE in symlog space
    - multistep_prediction_loss: multi-head observable h-step prediction

All functions are self-contained (no import from koopman_cvae.py).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# ZOH propagation: Full A + Low-rank B
# ─────────────────────────────────────────────────────────────

def get_transition_matrix(
    A: torch.Tensor,         # (m, m)
    B_U: torch.Tensor,       # (da, m, r)
    B_V: torch.Tensor,       # (da, m, r)
    a: torch.Tensor,         # (batch, da)
    dt: float,
) -> torch.Tensor:
    """
    Compute F = I + M*dt  where  M = A + sum_l B^(l) a_l
    B^(l) = B_U[l] @ B_V[l].T  (low-rank decomposition)

    Returns F: (batch, m, m)
    """
    m = A.shape[0]
    device = A.device

    # B^(l) = U^(l) @ V^(l).T -> (da, m, m)
    B_full = torch.bmm(B_U, B_V.transpose(-1, -2))           # (da, m, m)

    # sum_l B^(l) * a_l -> (batch, m, m)
    Ba = torch.einsum('bl,lij->bij', a, B_full)               # (batch, m, m)

    # M = A + Ba
    M = A.unsqueeze(0) + Ba                                    # (batch, m, m)

    # F = I + M*dt  (1st-order ZOH)
    I = torch.eye(m, device=device).unsqueeze(0)              # (1, m, m)
    F = I + M * dt                                             # (batch, m, m)
    return F


def propagate(
    z: torch.Tensor,         # (batch, m)
    A: torch.Tensor,         # (m, m)
    B_U: torch.Tensor,       # (da, m, r)
    B_V: torch.Tensor,       # (da, m, r)
    a: torch.Tensor,         # (batch, da)
    dt: float,
) -> torch.Tensor:
    """Single-step ZOH: z_{t+1} = F(a) z_t"""
    F = get_transition_matrix(A, B_U, B_V, a, dt)             # (batch, m, m)
    return torch.bmm(F, z.unsqueeze(-1)).squeeze(-1)           # (batch, m)


def propagate_h_steps(
    z: torch.Tensor,         # (BT, m)  flattened batch x time
    A: torch.Tensor,         # (m, m)
    B_U: torch.Tensor,       # (da, m, r)
    B_V: torch.Tensor,       # (da, m, r)
    a: torch.Tensor,         # (BT, da)  action at anchor time
    dt: float,
    h: int,
) -> torch.Tensor:
    """
    h-step ZOH assuming constant action over [t, t+h*dt) (ZOH assumption).
    F^h = (I + M*dt)^h via matrix power.

    Returns z_pred: (BT, m)
    """
    F = get_transition_matrix(A, B_U, B_V, a, dt)             # (BT, m, m)

    # F^h via repeated matmul (h is small, typically 1-5)
    Fh = F
    for _ in range(h - 1):
        Fh = torch.bmm(Fh, F)

    return torch.bmm(Fh, z.unsqueeze(-1)).squeeze(-1)         # (BT, m)


# ─────────────────────────────────────────────────────────────
# Eigenvalue stability loss (on full A)
# ─────────────────────────────────────────────────────────────

def eigenvalue_stability_loss(
    A: torch.Tensor,           # (m, m)
    target_radius: float = 0.99,
    margin: float = 0.01,
) -> torch.Tensor:
    """
    Penalize eigenvalues of A whose modulus exceeds target_radius.

    For full A (not necessarily symmetric), eigenvalues are complex.
    We use torch.linalg.eigvals which returns complex eigenvalues.

    Loss = mean over eigenvalues of ReLU(|lambda_k| - target_radius + margin)^2

    This replaces the diagonal mu_k fixed constraint.
    gradient flows through the real eigenvalue magnitudes.

    Note: torch.linalg.eigvals is differentiable w.r.t. A.
    """
    # eigvals: complex (m,)
    eigvals = torch.linalg.eigvals(A)
    modulus = eigvals.abs()                                    # (m,) real

    # Penalize moduli that exceed target_radius (soft margin)
    excess = F.relu(modulus - (target_radius - margin))       # (m,)
    return (excess ** 2).mean()


def eigenvalue_diversity_loss(
    A: torch.Tensor,           # (m, m)
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Penalize eigenvalues of A that are too close together (frequency collapse).
    Replaces frequency repulsion loss on omega_k.

    Loss = mean_{i != j} exp(-|lambda_i - lambda_j|^2 / sigma^2)

    Uses complex eigenvalues directly.
    """
    eigvals = torch.linalg.eigvals(A)                         # (m,) complex
    diff    = eigvals.unsqueeze(0) - eigvals.unsqueeze(1)     # (m, m) complex
    dist_sq = diff.real**2 + diff.imag**2                     # (m, m) real
    repul   = torch.exp(-dist_sq / (sigma**2 + 1e-8))
    mask    = 1.0 - torch.eye(A.shape[0], device=A.device)
    m       = A.shape[0]
    return (repul * mask).sum() / max(m * (m - 1), 1)


# ─────────────────────────────────────────────────────────────
# Multi-head prediction loss
# ─────────────────────────────────────────────────────────────

def multistep_prediction_loss(
    z: torch.Tensor,           # (B, T, m)  encoder output
    v_heads: torch.Tensor,     # (B, T, Nh, m)  multi-head readout vectors
    actions: torch.Tensor,     # (B, T, da)
    A: torch.Tensor,           # (m, m)
    B_U: torch.Tensor,         # (da, m, r)
    B_V: torch.Tensor,         # (da, m, r)
    dt: float,
    H: int,
) -> torch.Tensor:
    """
    Multi-head Koopman prediction loss.

    For each anchor t, horizon h, head n:
        g_pred^(n)(t+h) = v^(n)(t) · z_pred(t+h)
                        = v^(n)(t) · F^h(a_t) z(t)

        g_true^(n)(t+h) = v^(n)(t) · z(t+h)   [stop_gradient on z(t+h)]

    Loss = mean over h, n of MSE(g_pred, g_true)

    Jointly supervises: Phi (encoder), A, B_U, B_V, TCN heads W^(n).
    """
    B, T, m  = z.shape
    Nh       = v_heads.shape[2]
    H        = min(H, T - 1)

    total = torch.tensor(0.0, device=z.device)

    for h in range(1, H + 1):
        T_anc = T - h
        BT    = B * T_anc

        # Flatten batch x time for vectorized matmul
        z_anc   = z[:, :T_anc].reshape(BT, m)                 # (BT, m)
        a_anc   = actions[:, :T_anc].reshape(BT, -1)          # (BT, da)
        vh_anc  = v_heads[:, :T_anc].reshape(BT, Nh, m)       # (BT, Nh, m)

        # h-step ZOH prediction
        z_pred  = propagate_h_steps(z_anc, A, B_U, B_V,
                                    a_anc, dt, h)              # (BT, m)

        # Observable: g^(n) = v^(n) · z
        # g_pred: (BT, Nh)
        g_pred  = torch.bmm(vh_anc,
                            z_pred.unsqueeze(-1)).squeeze(-1)  # (BT, Nh)

        # Target: v^(n)(t) · z(t+h),  stop_gradient on z(t+h)
        z_true  = z[:, h:].reshape(BT, m).detach()            # (BT, m)
        g_true  = torch.bmm(vh_anc,
                            z_true.unsqueeze(-1)).squeeze(-1)  # (BT, Nh)

        total = total + F.mse_loss(g_pred, g_true)

    return total / max(H, 1)


# ─────────────────────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────────────────────

def reconstruction_loss(
    pred: torch.Tensor,        # (..., dim)  symlog space
    target: torch.Tensor,      # (..., dim)  symlog space
) -> torch.Tensor:
    return F.mse_loss(pred, target)


# ─────────────────────────────────────────────────────────────
# Decorrelation (scale-invariant, off-diagonal only)
# ─────────────────────────────────────────────────────────────

def decorrelation_loss(
    z: torch.Tensor,           # (N, m)
) -> torch.Tensor:
    """
    Off-diagonal cosine similarity penalty.
    Bounded [0, 1], scale-invariant.
    Prevents arbitrary linear transforms of z (gauge uniqueness).
    """
    N, m   = z.shape
    z_norm = F.normalize(z, p=2, dim=0)                       # (N, m)
    corr   = z_norm.T @ z_norm                                # (m, m)
    mask   = 1.0 - torch.eye(m, device=z.device)
    return (corr ** 2 * mask).sum() / max(m * (m - 1), 1)