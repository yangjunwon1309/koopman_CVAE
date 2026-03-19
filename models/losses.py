"""
losses.py — standalone loss functions for Koopman CVAE v2

All functions are self-contained (no import from koopman_cvae.py).
KoopmanCVAE imports and calls these directly.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


def complex_mul(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Element-wise complex multiply: (a_re + j*a_im)(b_re + j*b_im)"""
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


def kl_koopman_prior(
    mu_re: torch.Tensor,    # (B, Np, m)
    mu_im: torch.Tensor,
    sigma: torch.Tensor,    # (B, Np, m)
    z_re: torch.Tensor,     # (B, Np, m) sampled latent
    z_im: torch.Tensor,
    lb_re: torch.Tensor,    # (m,)  discrete eigenvalue real part
    lb_im: torch.Tensor,    # (m,)  discrete eigenvalue imag part
    sigma0_sq: torch.Tensor,  # (m,) process noise variance
) -> torch.Tensor:
    """
    KL with Koopman dynamic prior:
        p(z_k | z_{k-1}) = CN(lambda_bar * z_{k-1}, Sigma)

    KL(CN(mu_k, sk^2) || CN(mu0_k, s0^2))
        = (||mu_re_k - mu0_re||^2 + ||mu_im_k - mu0_im||^2) / s0^2
          + sk^2/s0^2 - log(sk^2/s0^2) - 1

    Applied to patches k=1,...,Np-1 (needs z_{k-1} as prior anchor).
    Returns scalar mean KL.
    """
    # Prior mean: lambda_bar * z_{k-1}  for k = 1, ..., Np-1
    mu0_re, mu0_im = complex_mul(lb_re, lb_im, z_re[:, :-1], z_im[:, :-1])

    mu_re_k  = mu_re[:, 1:]        # (B, Np-1, m)
    mu_im_k  = mu_im[:, 1:]
    sk_sq    = sigma[:, 1:] ** 2   # (B, Np-1, m)
    s0_sq    = sigma0_sq.unsqueeze(0).unsqueeze(0)  # (1, 1, m)

    kl = (
        ((mu_re_k - mu0_re) ** 2 + (mu_im_k - mu0_im) ** 2) / (s0_sq + 1e-8)
        + sk_sq / (s0_sq + 1e-8)
        - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return kl.mean()


def kl_standard_prior(
    mu_re: torch.Tensor,      # (B, Np, m)
    mu_im: torch.Tensor,
    sigma: torch.Tensor,      # (B, Np, m)
    sigma0_sq: torch.Tensor,  # (m,)
) -> torch.Tensor:
    """
    KL with isotropic standard prior: p(z) = CN(0, Sigma)

    KL(CN(mu, sk^2) || CN(0, s0^2))
        = (mu_re^2 + mu_im^2) / s0^2
          + sk^2/s0^2 - log(sk^2/s0^2) - 1
    """
    s0_sq = sigma0_sq.unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    sk_sq = sigma ** 2

    kl = (
        (mu_re ** 2 + mu_im ** 2) / (s0_sq + 1e-8)
        + sk_sq / (s0_sq + 1e-8)
        - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return kl.mean()


def multistep_prediction_loss(
    A: torch.Tensor,      # (B, Np, m)  amplitude
    theta: torch.Tensor,  # (B, Np, m)  phase
    mu_re: torch.Tensor,  # (B, Np, m)  posterior mean real
    mu_im: torch.Tensor,  # (B, Np, m)  posterior mean imag
    mod: torch.Tensor,    # (m,)  |lambda_bar|
    ang: torch.Tensor,    # (m,)  angle(lambda_bar)
    H: int,
) -> torch.Tensor:
    """
    Multi-step Koopman prediction loss (KoVAE-style).

    For each anchor patch k and horizon h = 1, ..., H:
        predicted:  A_k * |lambda|^h  *  exp(j*(theta_k + h*angle))
        target:     mu_{k+h}  (posterior mean of next patch)

    Loss per (k,h): MSE in Re/Im, normalized by pred Frobenius norm.
    Returns scalar mean over all valid (k, h) pairs.
    """
    B, Np, m = A.shape
    H = min(H, Np - 1)

    total = torch.tensor(0.0, device=A.device)
    count = 0

    for h in range(1, H + 1):
        A_anchor     = A[:, :Np - h]          # (B, Np-h, m)
        theta_anchor = theta[:, :Np - h]

        A_pred     = A_anchor * (mod ** h)
        theta_pred = theta_anchor + h * ang

        pred_re = A_pred * torch.cos(theta_pred)
        pred_im = A_pred * torch.sin(theta_pred)

        true_re = mu_re[:, h:]                 # (B, Np-h, m)
        true_im = mu_im[:, h:]

        loss_h  = (pred_re - true_re) ** 2 + (pred_im - true_im) ** 2
        norm_h  = torch.sqrt((pred_re ** 2 + pred_im ** 2).sum() + 1e-8)

        total  += loss_h.mean() / norm_h
        count  += 1

    return total / max(count, 1)


def temporal_contrastive_loss(
    p_emb: torch.Tensor,   # (B, Np, embed_dim)  patch embeddings (pre-latent)
    tau: float = 0.1,
    delta_pos: int = 2,
    delta_neg: int = 4,
) -> torch.Tensor:
    """
    Time-series aware InfoNCE applied to patch embeddings (upstream of latent).

    Positive pairs: |j - k| <= delta_pos  (temporally adjacent)
    Negative pairs: |j - k| >= delta_neg  (temporally distant)

    Using patch embeddings (not latent means) prevents the contrastive signal
    from collapsing together with the latent — it regularizes the encoder
    at an earlier stage.

    Returns scalar InfoNCE loss.
    """
    B, Np, _ = p_emb.shape
    z_norm = F.normalize(p_emb, dim=-1)   # (B, Np, D)

    total = torch.tensor(0.0, device=p_emb.device)
    count = 0

    for k in range(Np):
        pos_idx = [j for j in range(Np) if 0 < abs(j - k) <= delta_pos]
        neg_idx = [j for j in range(Np) if abs(j - k) >= delta_neg]

        if not pos_idx or not neg_idx:
            continue

        q    = z_norm[:, k]                   # (B, D)
        pos  = z_norm[:, pos_idx]             # (B, n_pos, D)
        negs = z_norm[:, neg_idx]             # (B, n_neg, D)

        sim_pos = torch.bmm(pos,  q.unsqueeze(-1)).squeeze(-1) / tau   # (B, n_pos)
        sim_neg = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / tau   # (B, n_neg)

        log_pos = torch.logsumexp(sim_pos, dim=-1)
        log_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=-1), dim=-1)

        total += (log_all - log_pos).mean()
        count += 1

    return total / max(count, 1)