"""
losses.py — standalone loss functions for KODAC

Theoretical basis:
    State stream:   z_s_{t+1} = A * z_s_t           (diagonal ZOH, no u)
    Action stream:  z_a_{t+1} = (A + B(u)) * z_a_t  (diagonal+LoRA ZOH)

    Observable readout (action stream):
        g(a_{t+h}) = sum_k v_k * exp[(lam_k + sum_l beta_k^(l) u_l) dt] z_a_k(t)

    Reconstruction:
        s_hat = D_s(z_s_re, z_s_im)             -- state decoder
        a_hat = D_a(z_a_re, z_a_im, v_eff, beta) -- action decoder

All functions are self-contained (no import from koopman_cvae.py).
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Real Schur 2x2 block propagation (exact ZOH)
# ─────────────────────────────────────────────────────────────

def schur_block_propagate(
    z_re: torch.Tensor,   # (..., m)
    z_im: torch.Tensor,   # (..., m)
    mu: torch.Tensor,     # (m,)   fixed decay
    omega: torch.Tensor,  # (m,)   learnable frequency
    dt: float,
    steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact ZOH for the autonomous state stream using Real Schur 2x2 blocks.
    No Taylor approximation -- exact because A is diagonal.

    Each pair (z_re_k, z_im_k) evolves as:
        [z_re]_{t+dt} = e^{mu_k dt} [cos w dt  -sin w dt] [z_re]_t
        [z_im]_{t+dt}               [sin w dt   cos w dt] [z_im]_t
    """
    decay = torch.exp(mu * dt * steps)     # (m,)
    angle = omega * dt * steps             # (m,)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    z_re_next = decay * (cos_a * z_re - sin_a * z_im)
    z_im_next = decay * (sin_a * z_re + cos_a * z_im)
    return z_re_next, z_im_next


def schur_block_rollout(
    z_re: torch.Tensor,   # (B, m)
    z_im: torch.Tensor,   # (B, m)
    mu: torch.Tensor,     # (m,)
    omega: torch.Tensor,  # (m,)
    dt: float,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized multi-step autonomous rollout.
    Returns (z_re_seq, z_im_seq) each (B, steps, m).
    """
    hs    = torch.arange(1, steps + 1, device=z_re.device, dtype=z_re.dtype)
    decay = torch.exp(mu.unsqueeze(0) * dt * hs.unsqueeze(1))  # (steps, m)
    angle = omega.unsqueeze(0) * dt * hs.unsqueeze(1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    z_re_0 = z_re.unsqueeze(1)  # (B, 1, m)
    z_im_0 = z_im.unsqueeze(1)

    z_re_seq = decay * (cos_a * z_re_0 - sin_a * z_im_0)  # (B, steps, m)
    z_im_seq = decay * (sin_a * z_re_0 + cos_a * z_im_0)
    return z_re_seq, z_im_seq


# ─────────────────────────────────────────────────────────────
# KL divergences
# ─────────────────────────────────────────────────────────────

def kl_koopman_prior(
    mu_re: torch.Tensor,      # (B, T, m)
    mu_im: torch.Tensor,
    log_sigma: torch.Tensor,  # (B, T, m)
    z_re: torch.Tensor,       # (B, T, m)  sampled
    z_im: torch.Tensor,
    mu_k: torch.Tensor,       # (m,)  fixed decay
    omega: torch.Tensor,      # (m,)
    dt: float,
    log_sigma0: torch.Tensor, # (m,)
) -> torch.Tensor:
    """
    KL with Koopman dynamic prior p(z_t | z_{t-1}) = CN(A*z_{t-1}, Sigma_0).
    Prior mean = schur_block_propagate(z_{t-1}).
    Applied for t=1,...,T-1.
    """
    prior_re, prior_im = schur_block_propagate(
        z_re[:, :-1], z_im[:, :-1], mu_k, omega, dt, steps=1
    )
    mu_re_q  = mu_re[:, 1:]
    mu_im_q  = mu_im[:, 1:]
    sigma_q  = log_sigma[:, 1:].exp() + 1e-6
    sigma_0  = log_sigma0.exp() + 1e-6

    sigma_q_sq = sigma_q ** 2
    sigma_0_sq = sigma_0.unsqueeze(0).unsqueeze(0) ** 2

    kl = (
        ((mu_re_q - prior_re) ** 2 + (mu_im_q - prior_im) ** 2) / sigma_0_sq
        + sigma_q_sq / sigma_0_sq
        - torch.log(sigma_q_sq / (sigma_0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return 0.5 * kl.mean()


def kl_standard_prior(
    mu_re: torch.Tensor,
    mu_im: torch.Tensor,
    log_sigma: torch.Tensor,
    log_sigma0: torch.Tensor,
) -> torch.Tensor:
    """KL(CN(mu_q, sigma_q^2) || CN(0, sigma_0^2))"""
    sigma_q_sq = (log_sigma.exp() + 1e-6) ** 2
    sigma_0_sq = (log_sigma0.exp() + 1e-6).unsqueeze(0).unsqueeze(0) ** 2
    kl = (
        (mu_re ** 2 + mu_im ** 2) / sigma_0_sq
        + sigma_q_sq / sigma_0_sq
        - torch.log(sigma_q_sq / (sigma_0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return 0.5 * kl.mean()


# ─────────────────────────────────────────────────────────────
# Reconstruction losses
# ─────────────────────────────────────────────────────────────

def reconstruction_loss(
    pred: torch.Tensor,   # (..., dim)  decoder output (symlog space)
    target: torch.Tensor, # (..., dim)  ground truth (symlog space)
) -> torch.Tensor:
    """
    MSE in symlog space.
    Used for both:
      - state recon:  D_s(z_s) vs symlog(s)
      - action recon: D_a(z_a, v_eff, beta_eff) vs symlog(a)
    """
    return F.mse_loss(pred, target)


# ─────────────────────────────────────────────────────────────
# Multi-step prediction loss (dominant, KODAC-specific)
# ─────────────────────────────────────────────────────────────

def kodac_multistep_prediction_loss(
    zs_re: torch.Tensor,        # (B, T, m)  state stream latent
    zs_im: torch.Tensor,
    za_re: torch.Tensor,        # (B, T, m)  action stream latent
    za_im: torch.Tensor,
    v_eff: torch.Tensor,        # (B, m)     interpolated Koopman mode
    beta_eff: torch.Tensor,     # (B, m, da) interpolated input coupling
    actions: torch.Tensor,      # (B, T, da) ground-truth actions
    mu: torch.Tensor,           # (m,)
    omega: torch.Tensor,        # (m,)
    dt: float,
    H: int,
) -> torch.Tensor:
    """
    ZOH multi-step prediction loss for BOTH streams.

    For the ACTION stream, the predicted h-step ZOH observable is:
        g_hat(t+h) = sum_k v_k * exp[(mu_k + beta_k*u_t) h*dt] * za_re_k(t)
        g_tgt(t+h) = sum_k v_k * za_re_k(t+h)   [from encoder posterior]

    For the STATE stream, the predicted h-step ZOH state is:
        zs_hat(t+h) = A^h * zs(t)  [pure autonomous rollout]
        zs_tgt(t+h) = zs_re(t+h)   [from encoder posterior]

    Both losses are summed. This jointly supervises:
        phi_s, phi_a, v_i, beta_i, omega_k, GRU posterior (via P_hat in v_eff)
    """
    B, T, m = za_re.shape
    H = min(H, T - 1)

    loss_action = torch.tensor(0.0, device=za_re.device)
    loss_state  = torch.tensor(0.0, device=zs_re.device)

    for h in range(1, H + 1):
        T_anc = T - h
        BT    = B * T_anc

        # ── Action stream prediction ─────────────────────────
        za_re_anc = za_re[:, :T_anc].reshape(BT, m)
        za_im_anc = za_im[:, :T_anc].reshape(BT, m)
        u_anc     = actions[:, :T_anc].reshape(BT, -1)   # (BT, da)

        ve_f  = v_eff.unsqueeze(1).expand(-1, T_anc, -1).reshape(BT, m)
        be_f  = beta_eff.unsqueeze(1).expand(-1, T_anc, -1, -1).reshape(BT, m, -1)

        # Effective decay with input coupling: exp[(mu + beta*u) h*dt]
        beta_u   = torch.bmm(be_f, u_anc.unsqueeze(-1)).squeeze(-1)   # (BT, m)
        eff_mu   = mu.unsqueeze(0) + beta_u                             # (BT, m)
        decay_h  = torch.exp(eff_mu * dt * h)                          # (BT, m)
        angle_h  = omega.unsqueeze(0) * dt * h                         # (1, m)
        cos_h    = torch.cos(angle_h)
        sin_h    = torch.sin(angle_h)

        za_re_pred = decay_h * (cos_h * za_re_anc - sin_h * za_im_anc)
        # Observable: Re(v * z) = v * za_re  (v is real, z is complex)
        g_pred = (ve_f * za_re_pred).sum(dim=-1).reshape(B, T_anc)     # (B, T_anc)

        # Target: Re(v * za(t+h)) from encoder
        za_re_tgt = za_re[:, h:].reshape(BT, m)
        g_tgt = (ve_f * za_re_tgt).sum(dim=-1).reshape(B, T_anc)

        loss_action = loss_action + F.mse_loss(g_pred, g_tgt.detach())

        # ── State stream prediction ──────────────────────────
        # Autonomous ZOH: no u, just eigenvalue decay + rotation
        zs_re_anc = zs_re[:, :T_anc].reshape(BT, m)
        zs_im_anc = zs_im[:, :T_anc].reshape(BT, m)

        decay_s  = torch.exp(mu.unsqueeze(0) * dt * h)                 # (1, m)
        zs_re_pred = decay_s * (cos_h * zs_re_anc - sin_h * zs_im_anc)

        # Target: zs_re(t+h) from encoder
        zs_re_tgt = zs_re[:, h:].reshape(BT, m)

        loss_state = loss_state + F.mse_loss(zs_re_pred, zs_re_tgt.detach())

    n = max(H, 1)
    return (loss_action + loss_state) / n


# ─────────────────────────────────────────────────────────────
# Contrastive loss
# ─────────────────────────────────────────────────────────────

def kodac_contrastive_loss(
    gru_hidden: torch.Tensor,      # (B, m)  GRU h_T projected
    skill_z_summary: torch.Tensor, # (B, m)  skill-conditioned za summary
    tau: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE: aligns temporal skill identity (GRU) with action eigenfunction
    summary (skill-conditioned mean of za_re over trajectory).

    Query: h_T_proj   (temporal context — WHICH skill)
    Key:   z_a_summary (eigenfunction state summary — WHERE in latent)

    NOT aligned with raw za_t to preserve GRU/encoder role separation.
    """
    q = F.normalize(gru_hidden,      dim=-1)
    k = F.normalize(skill_z_summary, dim=-1)
    logits = torch.mm(q, k.T) / tau
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────
# Eigenvalue regularization
# ─────────────────────────────────────────────────────────────

def eigenvalue_frequency_repulsion(
    omega: torch.Tensor,  # (m,)
    sigma: float = 0.3,
) -> torch.Tensor:
    """
    Penalize nearby frequencies to prevent mode collapse.
    L_eig = mean_{k != k'} exp(-(omega_k - omega_k')^2 / sigma^2)
    """
    diff = omega.unsqueeze(0) - omega.unsqueeze(1)   # (m, m)
    repulsion = torch.exp(-(diff ** 2) / (sigma ** 2))
    mask = 1.0 - torch.eye(omega.shape[0], device=omega.device)
    return (repulsion * mask).sum() / (omega.shape[0] * (omega.shape[0] - 1) + 1e-8)


# ─────────────────────────────────────────────────────────────
# Skill posterior regularizers
# ─────────────────────────────────────────────────────────────

def posterior_entropy_regularization(
    P_hat: torch.Tensor,  # (B, S)
) -> torch.Tensor:
    """
    Entropy regularization: prevents posterior collapse to single skill.
    R_ent = -H(P_hat) = sum_i P_i * log(P_i)  [positive; minimize to spread]
    """
    return (P_hat * (P_hat + 1e-8).log()).sum(dim=-1).mean()


def mode_diversity_loss(
    V: torch.Tensor,        # (S, m)  all skill mode vectors
    margin: float = 0.2,    # stop penalizing once cosine sim drops below margin
) -> torch.Tensor:
    """
    Penalize skill modes that are too similar in DIRECTION (cosine similarity).

    Replaces the previous ||v_i - v_j||^2 formulation which caused norm
    divergence: minimizing -||v_i - v_j||^2 is satisfied by growing ||v_i||
    indefinitely without actually separating directions.

    New formulation:
        R_div = mean_{i != j} ReLU(cos_sim(v_i, v_j) - margin)^2

    Properties:
      - Norm-invariant: only direction matters
      - Margin: gradient is zero once cos_sim(v_i, v_j) < margin (diverse enough)
      - Bounded: always in [0, (1 - margin)^2]
      - No divergence: V norm is not rewarded
    """
    S = V.shape[0]
    V_norm = F.normalize(V, p=2, dim=-1)               # (S, m), unit vectors
    cos_sim = torch.mm(V_norm, V_norm.T)               # (S, S)
    mask    = 1.0 - torch.eye(S, device=V.device)
    # Penalize pairs that are still too similar (above margin)
    excess  = F.relu(cos_sim - margin) ** 2            # zero when already diverse
    return (excess * mask).sum() / max(S * (S - 1), 1)


def decorrelation_loss(
    z: torch.Tensor,  # (N, m)  eigenfunction output, flattened over batch/time
) -> torch.Tensor:
    """
    Gauge uniqueness: E[z z^T] approx I prevents arbitrary linear transforms.
    R_decorr = ||E[z z^T] - I||_F^2
    """
    N, m = z.shape
    cov  = (z.T @ z) / (N + 1e-8)
    return F.mse_loss(cov, torch.eye(m, device=z.device))