"""
losses.py — KODAQ Full RSSM-Koopman Loss Functions
====================================================

Document → Implementation mapping:

  L_rec   : Reconstruction of x_t = [Δe_t, Δp_t, q_t, q̇_t]
             4 independent MLP heads, weighted by signal magnitude (α_j)
  L_dyn   : Koopman Consistency — L2 regression (NOT KL)
             ||μ_φ(x_{t+1}, h_{t+1}) - (Ā(w)·o_t + B̄(w)·u_t)||²
  L_skill : Cross-entropy vs EXTRACT labels ĉ_t
             -log p_θ(c_t = ĉ_t | h_t)
  L_reg   : Posterior-prior alignment (stop-gradient on prior)
             ||μ_φ(x_t, h_t) - sg(Ā(w_{t-1})·o_{t-1} + B̄(w_{t-1})·u_{t-1})||²

All functions are pure (no nn.Module state).
koopman_cvae.py delegates all loss computation here.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Symlog / Symexp  (scale compression for wide-range signals)
# ──────────────────────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Skill-interpolated Koopman: log-eigenvalue space blending
# ──────────────────────────────────────────────────────────────────────────────

def blend_koopman(
    log_lambdas: torch.Tensor,   # (K, m)  log-magnitude of eigenvalues per skill
    thetas:      torch.Tensor,   # (K, m)  phase angles per skill
    G_k:         torch.Tensor,   # (K, m, da)  skill-specific input coupling
    U:           torch.Tensor,   # (m, m)  shared eigenbasis
    w:           torch.Tensor,   # (B, K)  soft skill weights (sum=1)
) -> tuple:
    """
    Skill-interpolated Koopman matrices in log-eigenvalue space.
    Guarantees |λ̄_i| ≤ 1 when all |λ^(k)_i| ≤ 1.

    Ā(w) = U · diag(exp(Σ_k w_k · log_λ_k) · e^{iθ̄}) · U⁻¹
    B̄(w) = U · (Σ_k w_k · G_k)

    Returns:
        A_bar: (B, m, m)  blended transition
        B_bar: (B, m, da) blended input coupling
        r_bar: (B, m)     blended log-magnitudes
        t_bar: (B, m)     blended phases
    """
    # Interpolate in log-eigenvalue space: (B, m)
    r_bar = torch.einsum('bk,km->bm', w, log_lambdas)  # (B, m)
    t_bar = torch.einsum('bk,km->bm', w, thetas)        # (B, m)

    # Build diagonal complex eigenvalues in real form
    # λ̄_i = exp(r̄_i) · e^{iθ̄_i}  → 2x2 rotation-scaling blocks
    # We work in real-valued block-diagonal representation
    # Λ̄ = diag(..., [exp(r)*cos(θ), -exp(r)*sin(θ);
    #                  exp(r)*sin(θ),  exp(r)*cos(θ)], ...)
    # For simplicity with full U: Ā = U Λ̄ U⁻¹ using complex representation
    r_exp = torch.exp(r_bar)                            # (B, m) magnitudes

    # Build Λ̄ as complex: (B, m) complex
    lambdas_c = torch.complex(
        r_exp * torch.cos(t_bar),
        r_exp * torch.sin(t_bar),
    )  # (B, m)

    # Ā = U · diag(λ̄) · U⁻¹
    # U: (m, m) real — treat as complex
    U_c    = U.to(dtype=torch.complex64)                       # (m, m)
    U_inv  = torch.linalg.inv(U_c)                            # (m, m)

    # (B, m, m): batch diagonal
    Lam    = torch.diag_embed(lambdas_c)                       # (B, m, m)
    A_c    = U_c.unsqueeze(0) @ Lam @ U_inv.unsqueeze(0)      # (B, m, m)
    A_bar  = A_c.real                                          # (B, m, m)

    # B̄ = U · (Σ_k w_k G_k): (B, m, da)
    G_mix  = torch.einsum('bk,kmd->bmd', w, G_k)              # (B, m, da)
    B_bar  = U.unsqueeze(0) @ G_mix                           # (B, m, da)

    return A_bar, B_bar, r_bar, t_bar


def koopman_step(
    o:     torch.Tensor,   # (B, m)  current lifted state
    u:     torch.Tensor,   # (B, da) encoded action
    A_bar: torch.Tensor,   # (B, m, m)
    B_bar: torch.Tensor,   # (B, m, da)
) -> torch.Tensor:
    """
    o_{t+1} = Ā(w) · o_t + B̄(w) · u_t
    Returns: (B, m)
    """
    return (A_bar @ o.unsqueeze(-1)).squeeze(-1) + \
           (B_bar @ u.unsqueeze(-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# L_rec: Weighted multi-head reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def reconstruction_loss(
    preds:   dict,          # {'delta_e': (B,T,2048), 'delta_p': (B,T,42),
                            #  'q': (B,T,9), 'qdot': (B,T,9)}
    targets: dict,          # same keys, same shapes
    weights: dict,          # {'delta_e': α_e, 'delta_p': α_p, 'q': α_q, 'qdot': α_qd}
) -> tuple:
    """
    L_rec = Σ_j α_j · MSE(x̂^(j), x^(j))

    Returns (total_loss, {key: per_head_loss})
    """
    per_head = {}
    total    = torch.tensor(0.0, device=next(iter(preds.values())).device)

    for key in preds:
        p = preds[key]
        t = targets[key]
        w = weights.get(key, 1.0)
        loss = F.mse_loss(p, t)
        per_head[key] = loss
        total = total + w * loss

    return total, per_head


# ──────────────────────────────────────────────────────────────────────────────
# L_dyn: Koopman Consistency (L2, NOT KL — avoids posterior collapse)
# ──────────────────────────────────────────────────────────────────────────────

def koopman_consistency_loss(
    mu_next:   torch.Tensor,   # (B, T-1, m)  posterior mean at t+1
    o_pred:    torch.Tensor,   # (B, T-1, m)  Koopman prediction Ā·o_t + B̄·u_t
) -> torch.Tensor:
    """
    L_dyn = ||μ_φ(x_{t+1}, h_{t+1}) - (Ā·o_t + B̄·u_t)||²

    Regresses the posterior mean at t+1 onto the Koopman prediction.
    Enforces linear dynamics structure without KL-induced collapse.
    No stop_gradient here: gradient flows to both encoder and A/B matrices.
    """
    return F.mse_loss(mu_next, o_pred)


# ──────────────────────────────────────────────────────────────────────────────
# L_skill: Cross-entropy vs EXTRACT labels
# ──────────────────────────────────────────────────────────────────────────────

def skill_classification_loss(
    logits: torch.Tensor,   # (B, T, K)  log p_θ(c_t | h_t)
    labels: torch.Tensor,   # (B, T)     int64, EXTRACT cluster assignments
    mask:   Optional[torch.Tensor] = None,  # (B, T) bool, valid timesteps
) -> torch.Tensor:
    """
    L_skill = -log p_θ(c_t = ĉ_t | h_t)
            = CrossEntropy(logits, labels)

    mask: optionally exclude padding timesteps.
    """
    B, T, K = logits.shape
    logits_flat = logits.reshape(B * T, K)
    labels_flat = labels.reshape(B * T).long()

    if mask is not None:
        valid = mask.reshape(B * T)
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]

    return F.cross_entropy(logits_flat, labels_flat)


# ──────────────────────────────────────────────────────────────────────────────
# L_reg: Posterior-prior alignment (stop-gradient on prior)
# ──────────────────────────────────────────────────────────────────────────────

def posterior_regularization_loss(
    mu_t:   torch.Tensor,   # (B, T-1, m)  posterior mean at t (t=1..T-1)
    o_pred: torch.Tensor,   # (B, T-1, m)  Koopman prior prediction at t
                            #              = Ā(w_{t-1})·o_{t-1} + B̄(w_{t-1})·u_{t-1}
) -> torch.Tensor:
    """
    L_reg = ||μ_φ(x_t, h_t) - sg(Ā(w_{t-1})·o_{t-1} + B̄(w_{t-1})·u_{t-1})||²

    stop_gradient on the prior prediction prevents h_t drift.
    Gradient flows only to the posterior encoder φ.
    """
    target = o_pred.detach()
    return F.mse_loss(mu_t, target)


# ──────────────────────────────────────────────────────────────────────────────
# Stability: |λ^(k)_i| ≤ 1 via tanh parameterization (no additional loss needed)
# But we add a soft penalty for monitoring and robustness
# ──────────────────────────────────────────────────────────────────────────────

def eigenvalue_stability_loss(
    log_lambdas: torch.Tensor,   # (K, m)  log-magnitudes (should be ≤ 0 for stable)
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Soft penalty for log_lambdas > margin (i.e., |λ| > exp(margin)).
    Since we use tanh(r)·e^{iθ}, |λ| = tanh(r) < 1 is guaranteed.
    This loss monitors for near-boundary values and adds a soft push.

    With tanh parameterization this is primarily a diagnostic / light regularizer.
    """
    # tanh(r) → log_lambda should be <= 0
    # Penalize if log_lambda > margin (exp(margin) = target_radius > 1: forbidden)
    excess = F.relu(log_lambdas - margin)
    return (excess ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss (called from KoopmanCVAE._compute_losses)
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    loss_rec:   torch.Tensor,
    loss_dyn:   torch.Tensor,
    loss_skill: torch.Tensor,
    loss_reg:   torch.Tensor,
    loss_stab:  torch.Tensor,
    lambda1: float,   # Koopman weight
    lambda2: float,   # Skill supervision weight
    lambda3: float,   # Posterior regularization weight
    lambda4: float,   # Stability weight
    phase:   int,     # 1: rec only, 2: +dyn+skill, 3: +reg
) -> tuple:
    """
    Phase-gated loss aggregation.

    Phase 1 (warm-up) : L_rec only
    Phase 2 (Koopman) : L_rec + λ1·L_dyn + λ2·L_skill
    Phase 3 (full)    : L_rec + λ1·L_dyn + λ2·L_skill + λ3·L_reg

    Returns (total, weights_used_dict)
    """
    total = loss_rec

    if phase >= 2:
        total = total + lambda1 * loss_dyn + lambda2 * loss_skill
    if phase >= 3:
        total = total + lambda3 * loss_reg
    # Stability always active (lightweight)
    total = total + lambda4 * loss_stab

    return total, {
        'rec':   1.0,
        'dyn':   lambda1 if phase >= 2 else 0.0,
        'skill': lambda2 if phase >= 2 else 0.0,
        'reg':   lambda3 if phase >= 3 else 0.0,
        'stab':  lambda4,
    }