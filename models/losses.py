"""
Standalone loss functions for Koopman CVAE.
Separated for unit testing and reuse.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


def kl_complex_gaussian(
    mu_re: torch.Tensor,    # (B, Np, m)  posterior mean re
    mu_im: torch.Tensor,    # (B, Np, m)  posterior mean im
    sigma: torch.Tensor,    # (B, Np, m)  posterior std (shared re/im)
    mu0_re: torch.Tensor,   # (B, Np-1, m) prior mean re
    mu0_im: torch.Tensor,   # (B, Np-1, m) prior mean im
    sigma0_sq: torch.Tensor,  # (m,)  prior variance (learnable)
) -> torch.Tensor:
    """
    KL(CN(mu_hat, sigma_hat^2) || CN(mu0, sigma0^2))
    Decomposed as sum of two real Gaussian KLs:
      KL(N(mu_re, sigma^2/2) || N(mu0_re, sigma0^2/2))
    + KL(N(mu_im, sigma^2/2) || N(mu0_im, sigma0^2/2))

    Closed form:
      = (||mu_re - mu0_re||^2 + ||mu_im - mu0_im||^2) / sigma0^2
      + sigma^2 / sigma0^2
      - ln(sigma^2 / sigma0^2)
      - 1

    Returns scalar (mean over B, Np-1, m)
    """
    # Posterior for k=2,...,Np
    mu_re_k = mu_re[:, 1:, :]    # (B, Np-1, m)
    mu_im_k = mu_im[:, 1:, :]
    sigma_k  = sigma[:, 1:, :]

    s0_sq = sigma0_sq.unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    sk_sq = sigma_k.pow(2)

    diff_re_sq = (mu_re_k - mu0_re).pow(2)
    diff_im_sq = (mu_im_k - mu0_im).pow(2)

    kl = (
        (diff_re_sq + diff_im_sq) / (s0_sq + 1e-8)
        + sk_sq / (s0_sq + 1e-8)
        - torch.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
        - 1.0
    )
    return kl.mean()


def linearity_loss(
    mu_re: torch.Tensor,      # (B, Np, m)
    mu_im: torch.Tensor,      # (B, Np, m)
    lb_re: torch.Tensor,      # (m,)
    lb_im: torch.Tensor,      # (m,)
) -> torch.Tensor:
    """
    L_pred = mean_{k=2,...,Np} sum_i |mu_{k,i} - lambda_bar_i * mu_{k-1,i}|^2
    Uses posterior means (not samples) to enforce encoder linearity.
    """
    from models.koopman_cvae import complex_mul

    mu_prev_re = mu_re[:, :-1, :]
    mu_prev_im = mu_im[:, :-1, :]

    target_re, target_im = complex_mul(lb_re, lb_im, mu_prev_re, mu_prev_im)

    loss = (
        (mu_re[:, 1:, :] - target_re).pow(2) +
        (mu_im[:, 1:, :] - target_im).pow(2)
    )
    return loss.mean()


def reconstruction_loss(
    p_hat: torch.Tensor,   # (B, Np, n, da) symlog space
    p_true: torch.Tensor,  # (B, Np, n, da) symlog space
) -> torch.Tensor:
    """MSE in symlog space (DreamerV3-style)"""
    return F.mse_loss(p_hat, p_true)

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