"""
losses.py — standalone loss functions for Koopman CVAE v2
For unit testing and external use.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


def kl_koopman_prior(
    mu_re: torch.Tensor,    # (B, Np, m)
    mu_im: torch.Tensor,
    sigma: torch.Tensor,    # (B, Np, m)
    z_re:  torch.Tensor,    # (B, Np, m)  sampled
    z_im:  torch.Tensor,
    lb_re: torch.Tensor,    # (m,)
    lb_im: torch.Tensor,
    sigma0_sq: torch.Tensor,  # (m,)
) -> torch.Tensor:
    """
    KL with Koopman prior: p(z_k | z_{k-1}) = CN(lambda_bar * z_{k-1}, Sigma)

    KL = (||mu_re_k - mu0_re||^2 + ||mu_im_k - mu0_im||^2) / sigma0^2
       + sigma_k^2 / sigma0^2 - log(sigma_k^2/sigma0^2) - 1
    """
    from models.koopman_cvae import complex_mul

    mu0_re, mu0_im = complex_mul(lb_re, lb_im, z_re[:, :-1], z_im[:, :-1])

    mu_re_k = mu_re[:, 1:]
    mu_im_k = mu_im[:, 1:]
    sk_sq   = sigma[:, 1:] ** 2
    s0_sq   = sigma0_sq.unsqueeze(0).unsqueeze(0)

    kl = (
        ((mu_re_k - mu0_re)**2 + (mu_im_k - mu0_im)**2) / (s0_sq + 1e-8)
        + sk_sq / (s0_sq + 1e-8)
        - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return kl.mean()


def kl_standard_prior(
    mu_re: torch.Tensor,
    mu_im: torch.Tensor,
    sigma: torch.Tensor,
    sigma0_sq: torch.Tensor,
) -> torch.Tensor:
    """
    KL with standard prior: p(z) = CN(0, Sigma)
    """
    s0_sq = sigma0_sq.unsqueeze(0).unsqueeze(0)
    sk_sq = sigma ** 2

    kl = (
        (mu_re**2 + mu_im**2) / (s0_sq + 1e-8)
        + sk_sq / (s0_sq + 1e-8)
        - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return kl.mean()


def multistep_prediction_loss(
    A: torch.Tensor,        # (B, Np, m)
    theta: torch.Tensor,    # (B, Np, m)
    mu_re: torch.Tensor,    # (B, Np, m)
    mu_im: torch.Tensor,
    mod: torch.Tensor,      # (m,) |lambda_bar|
    ang: torch.Tensor,      # (m,) angle(lambda_bar)
    H: int,
) -> torch.Tensor:
    """
    Multi-step prediction loss over H steps.
    For each (k, h): predict z_{k+h} from z_k using Koopman propagation.
    Loss: MSE(pred_re - true_re)^2 + MSE(pred_im - true_im)^2
    Normalized by Frobenius norm of prediction.
    """
    B, Np, m = A.shape
    total = torch.tensor(0.0, device=A.device)
    count = 0

    for h in range(1, min(H, Np-1) + 1):
        A_anchor     = A[:, :Np-h]
        theta_anchor = theta[:, :Np-h]

        A_pred     = A_anchor     * (mod ** h)
        theta_pred = theta_anchor + h * ang

        pred_re = A_pred * torch.cos(theta_pred)
        pred_im = A_pred * torch.sin(theta_pred)

        true_re = mu_re[:, h:]
        true_im = mu_im[:, h:]

        loss_h = (pred_re - true_re)**2 + (pred_im - true_im)**2
        norm_h = torch.sqrt((pred_re**2 + pred_im**2).sum() + 1e-8)
        total += loss_h.mean() / norm_h
        count += 1

    return total / max(count, 1)


def temporal_contrastive_loss(
    mu_re: torch.Tensor,    # (B, Np, m)
    mu_im: torch.Tensor,
    tau: float = 0.1,
    delta_pos: int = 2,
    delta_neg: int = 4,
) -> torch.Tensor:
    """
    Time-series InfoNCE with temporal proximity as positive signal.
    Positive: |j-k| <= delta_pos
    Negative: |j-k| >= delta_neg
    """
    B, Np, m = mu_re.shape
    z_flat = torch.cat([mu_re, mu_im], dim=-1)
    z_norm = F.normalize(z_flat, dim=-1)

    total = torch.tensor(0.0, device=mu_re.device)
    count = 0

    for k in range(Np):
        pos_idx = [j for j in range(Np) if 0 < abs(j-k) <= delta_pos]
        neg_idx = [j for j in range(Np) if abs(j-k) >= delta_neg]
        if not pos_idx or not neg_idx:
            continue

        q    = z_norm[:, k]
        pos  = z_norm[:, pos_idx]
        negs = z_norm[:, neg_idx]

        sim_pos = torch.bmm(pos,  q.unsqueeze(-1)).squeeze(-1) / tau
        sim_neg = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / tau

        log_pos = torch.logsumexp(sim_pos, dim=-1)
        log_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=-1), dim=-1)
        total += (log_all - log_pos).mean()
        count += 1

    return total / max(count, 1)