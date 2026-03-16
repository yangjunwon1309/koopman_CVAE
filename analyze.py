"""
Koopman CVAE Analysis Script
Analyzes: reconstruction, eigenvalues, prediction, latent distribution
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import os

# ── path setup ────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig, symlog, symexp
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset
from envs.env_configs import D4RL_ENV_MAP
from torch.utils.data import DataLoader
import math

# ── config ────────────────────────────────────────────────
CKPT_PATH  = os.path.expanduser('~/koopman_CVAE/checkpoints/adroit_pen_human/best.pt')
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES  = 8       # trajectories to visualize
SAVE_DIR   = Path(os.path.expanduser('~/koopman_CVAE/analysis'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
#  Load model & data
# ─────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg  = ckpt['cfg']
    model = KoopmanCVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded: koopman_dim={cfg.koopman_dim}, "
          f"patch_size={cfg.patch_size}, dt={cfg.dt_control}")
    return model, cfg


def load_data(cfg, quality='human', seq_len=50):
    try:
        dataset = load_d4rl_trajectories('adroit_pen', seq_len=seq_len,
                                          quality=quality, min_episode_len=30)
        print(f"D4RL dataset: {len(dataset)} samples")
    except Exception as e:
        print(f"D4RL load failed ({e}), using synthetic")
        dataset = make_synthetic_dataset(cfg.action_dim, cfg.state_dim,
                                          n_samples=200, seq_len=seq_len)
    return dataset


# ─────────────────────────────────────────────────────────
#  1. Reconstruction Analysis
# ─────────────────────────────────────────────────────────

def analyze_reconstruction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[1] Reconstruction Analysis")
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        out = model(actions, states)
        patches, patch_emb, state_emb = model.preprocess(actions, states)

        # A. Encode real data -> decode
        p_hat_symlog = out['p_hat']           # (B, Np, n, da) symlog space
        p_true_symlog = patches               # (B, Np, n, da)

        # Convert back to original space
        p_hat  = symexp(p_hat_symlog).cpu().numpy()
        p_true = symexp(p_true_symlog).cpu().numpy()

        # B. Random z -> decode
        B, Np, patch_n, da = patches.shape
        z_rand_re = torch.randn(B, Np, cfg.koopman_dim, device=DEVICE) / math.sqrt(2)
        z_rand_im = torch.randn(B, Np, cfg.koopman_dim, device=DEVICE) / math.sqrt(2)
        p_rand_symlog = model.decode(z_rand_re, z_rand_im, state_emb)
        p_rand = symexp(p_rand_symlog).cpu().numpy()

    # Reshape to (B, T, da)
    T = Np * patch_n
    p_hat_flat  = p_hat.reshape(n, T, da)
    p_true_flat = p_true.reshape(n, T, da)
    p_rand_flat = p_rand.reshape(n, T, da)

    # MSE metrics
    mse_recon = np.mean((p_hat_flat - p_true_flat)**2)
    mse_rand  = np.mean((p_rand_flat - p_true_flat)**2)
    print(f"  Recon MSE (encoded z): {mse_recon:.4f}")
    print(f"  Recon MSE (random z):  {mse_rand:.4f}")

    # Plot: action dimensions over time for first sample
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    dims_to_plot = [0, 5, 10, 15]  # sample 4 action dims
    colors = {'true': '#2196F3', 'recon': '#F44336', 'rand': '#4CAF50'}

    for ax, d in zip(axes, dims_to_plot):
        t = np.arange(T)
        ax.plot(t, p_true_flat[0, :, d], color=colors['true'],
                lw=2, label=f'True a[{d}]')
        ax.plot(t, p_hat_flat[0, :, d],  color=colors['recon'],
                lw=1.5, linestyle='--', label=f'Recon (encoded z)')
        ax.plot(t, p_rand_flat[0, :, d], color=colors['rand'],
                lw=1, linestyle=':', alpha=0.7, label=f'Recon (random z)')
        ax.set_ylabel(f'a[{d}]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark patch boundaries
        for k in range(1, Np):
            ax.axvline(x=k*patch_n, color='gray', alpha=0.3, linestyle='-')

    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'Reconstruction Analysis\n'
                 f'MSE(encoded)={mse_recon:.4f}  MSE(random)={mse_rand:.4f}',
                 fontsize=12)
    plt.tight_layout()
    path = SAVE_DIR / '1_reconstruction.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Distribution comparison (all dims, all samples)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes,
        [p_true_flat, p_hat_flat, p_rand_flat],
        ['True actions', 'Recon (encoded z)', 'Recon (random z)']):
        ax.hist(data.flatten(), bins=80, density=True, alpha=0.7)
        ax.set_title(f'{title}\nμ={data.mean():.3f}, σ={data.std():.3f}')
        ax.set_xlabel('Action value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Action Value Distribution Comparison', fontsize=12)
    plt.tight_layout()
    path = SAVE_DIR / '1b_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return mse_recon, mse_rand


# ─────────────────────────────────────────────────────────
#  2. Eigenvalue Analysis
# ─────────────────────────────────────────────────────────

def analyze_eigenvalues(model, cfg):
    print("\n[2] Eigenvalue Analysis")
    with torch.no_grad():
        omega   = model.koopman.omega.cpu().numpy()         # (m,)
        mu      = model.koopman.mu_fixed.cpu().numpy()      # (m,)
        lb_re, lb_im = model.koopman.get_discrete_eigenvalues()
        lb_re   = lb_re.cpu().numpy()
        lb_im   = lb_im.cpu().numpy()
        sigma_sq = model.koopman.sigma_sq.cpu().numpy()     # (m,)

    modulus = np.sqrt(lb_re**2 + lb_im**2)
    phase   = np.arctan2(lb_im, lb_re)
    dt      = cfg.patch_size * cfg.dt_control

    # Init omega for comparison
    m = cfg.koopman_dim
    omega_init = np.array([math.pi * 1.0 / (m + 1 - i) for i in range(1, m+1)])

    print(f"  dt_patch = {dt*1000:.1f}ms")
    print(f"  |lambda| range: [{modulus.min():.4f}, {modulus.max():.4f}]")
    print(f"  Expected |lambda| = {np.exp(-0.2 * dt):.4f} (all same, mu fixed)")
    print(f"  omega range: [{omega.min():.3f}, {omega.max():.3f}] rad")
    print(f"  sigma_sq range: [{sigma_sq.min():.4f}, {sigma_sq.max():.4f}]")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # A. Unit circle plot
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=1, label='Unit circle')
    sc = ax1.scatter(lb_re, lb_im, c=np.arange(m), cmap='viridis', s=60, zorder=5)
    plt.colorbar(sc, ax=ax1, label='Dimension index')
    ax1.set_aspect('equal')
    ax1.axhline(0, color='gray', alpha=0.3)
    ax1.axvline(0, color='gray', alpha=0.3)
    ax1.set_xlabel('Re(λ̄)')
    ax1.set_ylabel('Im(λ̄)')
    ax1.set_title('Discrete Eigenvalues in Complex Plane')
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    # B. omega: learned vs init
    ax2 = fig.add_subplot(gs[0, 1])
    idx = np.arange(m)
    ax2.bar(idx - 0.2, omega_init, width=0.4, label='Init ω', alpha=0.7, color='#2196F3')
    ax2.bar(idx + 0.2, omega,      width=0.4, label='Learned ω', alpha=0.7, color='#F44336')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('ω (rad)')
    ax2.set_title('Imaginary Eigenvalue: Init vs Learned')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # C. omega drift
    ax3 = fig.add_subplot(gs[0, 2])
    drift = omega - omega_init
    ax3.bar(idx, drift, color=np.where(drift > 0, '#F44336', '#2196F3'), alpha=0.7)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Δω = learned - init')
    ax3.set_title('Omega Drift from Initialization')
    ax3.grid(True, alpha=0.3)

    # D. Modulus per dimension
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(idx, modulus, color='#4CAF50', alpha=0.7)
    ax4.axhline(np.exp(-0.2 * dt), color='red', linestyle='--',
                label=f'Expected = {np.exp(-0.2*dt):.4f}')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('|λ̄|')
    ax4.set_title('Eigenvalue Modulus per Dimension')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # E. Phase per dimension
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(idx, phase, color='#FF9800', alpha=0.7)
    ax5.set_xlabel('Dimension')
    ax5.set_ylabel('∠λ̄ (rad)')
    ax5.set_title('Eigenvalue Phase per Dimension')
    ax5.grid(True, alpha=0.3)

    # F. Process noise sigma_sq
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(idx, sigma_sq, color='#9C27B0', alpha=0.7)
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('σᵢ² (process noise)')
    ax6.set_title('Learned Process Noise Σ')
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Koopman Eigenvalue Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = SAVE_DIR / '2_eigenvalues.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
#  3. Prediction Analysis
# ─────────────────────────────────────────────────────────

def analyze_prediction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[3] Prediction Analysis")
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, patch_emb, state_emb = model.preprocess(actions, states)
        enc = model.encode(patch_emb, state_emb)

        mu_re = enc['mu_re']  # (B, Np, m)
        mu_im = enc['mu_im']
        z_re  = enc['z_re']
        z_im  = enc['z_im']

        lb_re, lb_im = model.koopman.get_discrete_eigenvalues()  # (m,)

        # 1-step Koopman prediction of posterior mean
        from models.koopman_cvae import complex_mul
        pred_mu_re, pred_mu_im = complex_mul(
            lb_re, lb_im,
            mu_re[:, :-1, :],  # (B, Np-1, m)
            mu_im[:, :-1, :],
        )
        # actual next mu
        true_mu_re = mu_re[:, 1:, :]  # (B, Np-1, m)
        true_mu_im = mu_im[:, 1:, :]

        # Prediction error per patch
        pred_err = ((pred_mu_re - true_mu_re)**2 +
                    (pred_mu_im - true_mu_im)**2).mean(dim=-1)  # (B, Np-1)
        pred_err = pred_err.cpu().numpy()

        # Decode predicted z to action
        # Use predicted latent for k=1,...,Np-1 patches
        p_hat_pred = model.decode(pred_mu_re, pred_mu_im, state_emb[:, 1:, :])
        p_hat_pred = symexp(p_hat_pred).cpu().numpy()
        p_true     = symexp(patches[:, 1:, :, :]).cpu().numpy()

    Np_m1 = patches.shape[1] - 1
    patch_n = cfg.patch_size
    da = cfg.action_dim

    pred_action_mse = np.mean((p_hat_pred - p_true)**2)
    print(f"  Prediction MSE (latent):  {pred_err.mean():.6f}")
    print(f"  Prediction MSE (action):  {pred_action_mse:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # A. Prediction error over patches
    ax = axes[0]
    mean_err = pred_err.mean(axis=0)  # (Np-1,)
    std_err  = pred_err.std(axis=0)
    x = np.arange(len(mean_err))
    ax.fill_between(x, mean_err-std_err, mean_err+std_err, alpha=0.3, color='#F44336')
    ax.plot(x, mean_err, color='#F44336', lw=2, label='Mean prediction error')
    ax.set_xlabel('Patch index k')
    ax.set_ylabel('|μ̂_k - λ̄μ̂_{k-1}|²')
    ax.set_title(f'Koopman 1-step Prediction Error in Latent Space\n(mean={pred_err.mean():.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # B. Predicted vs true action (1 sample, 2 dims)
    ax = axes[1]
    T_pred = Np_m1 * patch_n
    t = np.arange(T_pred)
    p_pred_flat = p_hat_pred[0].reshape(T_pred, da)
    p_true_flat = p_true[0].reshape(T_pred, da)
    for d, c in [(0, '#2196F3'), (5, '#F44336')]:
        ax.plot(t, p_true_flat[:, d], color=c, lw=2, label=f'True a[{d}]')
        ax.plot(t, p_pred_flat[:, d], color=c, lw=1.5, linestyle='--',
                alpha=0.8, label=f'Predicted a[{d}]')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action value')
    ax.set_title(f'Koopman Predicted vs True Next Patch (action space)\nMSE={pred_action_mse:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # C. Per-dimension prediction error in latent
    ax = axes[2]
    dim_err = ((pred_mu_re.cpu() - true_mu_re.cpu())**2 +
               (pred_mu_im.cpu() - true_mu_im.cpu())**2).mean(dim=[0,1]).numpy()
    ax.bar(np.arange(cfg.koopman_dim), dim_err, color='#9C27B0', alpha=0.7)
    ax.set_xlabel('Koopman dimension i')
    ax.set_ylabel('Mean |μ̂_k - λ̄_i μ̂_{k-1}|²')
    ax.set_title('Prediction Error per Koopman Dimension')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / '3_prediction.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
#  4. Latent Distribution Analysis
# ─────────────────────────────────────────────────────────

def analyze_latent(model, dataset, cfg):
    print("\n[4] Latent Distribution Analysis")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, patch_emb, state_emb = model.preprocess(actions, states)
        enc = model.encode(patch_emb, state_emb)

        mu_re = enc['mu_re'].cpu().numpy()   # (B, Np, m)
        mu_im = enc['mu_im'].cpu().numpy()
        sigma = enc['sigma'].cpu().numpy()   # (B, Np, m)
        z_re  = enc['z_re'].cpu().numpy()
        z_im  = enc['z_im'].cpu().numpy()

        lb_re, lb_im = model.koopman.get_discrete_eigenvalues()
        from models.koopman_cvae import complex_mul
        sigma_sq = model.koopman.sigma_sq.cpu().numpy()  # (m,)

        # KL per dimension per patch
        z_re_t  = enc['z_re'][:, :-1, :]
        z_im_t  = enc['z_im'][:, :-1, :]
        mu0_re, mu0_im = complex_mul(lb_re, lb_im, z_re_t, z_im_t)
        mu0_re = mu0_re.cpu().numpy()
        mu0_im = mu0_im.cpu().numpy()

        mu_re_k = mu_re[:, 1:, :]
        mu_im_k = mu_im[:, 1:, :]
        sigma_k = sigma[:, 1:, :]

        s0_sq = sigma_sq[np.newaxis, np.newaxis, :]  # (1,1,m)
        sk_sq = sigma_k**2

        kl_per = (
            ((mu_re_k - mu0_re)**2 + (mu_im_k - mu0_im)**2) / (s0_sq + 1e-8)
            + sk_sq / (s0_sq + 1e-8)
            - np.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
            - 1.0
        )  # (B, Np-1, m)

    kl_per_dim  = kl_per.mean(axis=(0, 1))   # (m,)
    kl_per_patch = kl_per.mean(axis=(0, 2))  # (Np-1,)

    print(f"  Total KL (mean): {kl_per.mean():.4f}")
    print(f"  KL per dim (mean): {kl_per_dim.mean():.4f}")
    print(f"  σ_posterior (mean): {sigma.mean():.4f}")
    print(f"  σ_prior (mean): {np.sqrt(sigma_sq.mean()):.4f}")
    print(f"  |μ_re| (mean): {np.abs(mu_re).mean():.4f}")
    print(f"  |μ_im| (mean): {np.abs(mu_im).mean():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # A. KL per dimension
    ax = axes[0, 0]
    ax.bar(np.arange(cfg.koopman_dim), kl_per_dim, color='#E91E63', alpha=0.7)
    ax.set_xlabel('Koopman dimension')
    ax.set_ylabel('Mean KL')
    ax.set_title(f'KL Divergence per Dimension\n(total mean={kl_per.mean():.3f})')
    ax.grid(True, alpha=0.3)

    # B. KL over patches
    ax = axes[0, 1]
    ax.plot(np.arange(len(kl_per_patch)), kl_per_patch,
            color='#E91E63', lw=2, marker='o', markersize=4)
    ax.set_xlabel('Patch index k')
    ax.set_ylabel('Mean KL')
    ax.set_title('KL Divergence over Patches')
    ax.grid(True, alpha=0.3)

    # C. Posterior sigma distribution
    ax = axes[0, 2]
    ax.hist(sigma.flatten(), bins=60, density=True, alpha=0.7,
            color='#2196F3', label='Posterior σ')
    ax.axvline(np.sqrt(sigma_sq.mean()), color='red', linestyle='--',
               lw=2, label=f'Prior σ (mean={np.sqrt(sigma_sq.mean()):.3f})')
    ax.set_xlabel('σ value')
    ax.set_ylabel('Density')
    ax.set_title('Posterior vs Prior Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # D. Latent z distribution (Re vs Im)
    ax = axes[1, 0]
    ax.scatter(z_re[:, :, 0].flatten(), z_im[:, :, 0].flatten(),
               alpha=0.3, s=5, color='#4CAF50')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='Unit circle')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('Latent z distribution (dim 0)\nRe vs Im')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # E. Posterior mean norm over all dims
    ax = axes[1, 1]
    mu_norm = np.sqrt(mu_re**2 + mu_im**2)  # (B, Np, m)
    ax.hist(mu_norm.flatten(), bins=60, density=True, alpha=0.7, color='#FF9800')
    ax.set_xlabel('|μ̂| = √(μ_re² + μ_im²)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Mean Magnitude Distribution')
    ax.grid(True, alpha=0.3)

    # F. KL heatmap (patch x dim)
    ax = axes[1, 2]
    kl_map = kl_per.mean(axis=0)  # (Np-1, m)
    im = ax.imshow(kl_map.T, aspect='auto', cmap='hot_r', origin='lower')
    plt.colorbar(im, ax=ax, label='KL value')
    ax.set_xlabel('Patch index k')
    ax.set_ylabel('Koopman dimension')
    ax.set_title('KL Heatmap (patch × dimension)')

    fig.suptitle('Latent Distribution Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = SAVE_DIR / '4_latent.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
#  5. Loss 0 문제 진단
# ─────────────────────────────────────────────────────────

def diagnose_zero_losses(model, dataset, cfg):
    print("\n[5] Zero Loss Diagnosis")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, patch_emb, state_emb = model.preprocess(actions, states)
        enc = model.encode(patch_emb, state_emb)

        mu_re = enc['mu_re']
        mu_im = enc['mu_im']
        lb_re, lb_im = model.koopman.get_discrete_eigenvalues()

        from models.koopman_cvae import complex_mul
        mu_prev_re = mu_re[:, :-1, :]
        mu_prev_im = mu_im[:, :-1, :]
        target_re, target_im = complex_mul(lb_re, lb_im, mu_prev_re, mu_prev_im)
        diff_re = (mu_re[:, 1:, :] - target_re)
        diff_im = (mu_im[:, 1:, :] - target_im)
        pred_loss_raw = (diff_re**2 + diff_im**2).mean()

        omega_init = torch.tensor([
            math.pi * 1.0 / (cfg.koopman_dim + 1 - i)
            for i in range(1, cfg.koopman_dim + 1)
        ], device=DEVICE)
        omega_drift = (model.koopman.omega - omega_init) / (omega_init + 1e-6)
        eig_loss_raw = omega_drift.pow(2).mean()

    print(f"  L_pred (raw):     {pred_loss_raw.item():.6e}")
    print(f"  L_eig  (raw):     {eig_loss_raw.item():.6e}")
    print(f"  omega grad:       {model.koopman.omega.grad}")
    print(f"  mu_re max:        {mu_re.abs().max().item():.6f}")
    print(f"  mu_im max:        {mu_im.abs().max().item():.6f}")
    print(f"  lb_re range:      [{lb_re.min().item():.4f}, {lb_re.max().item():.4f}]")
    print(f"  lb_im range:      [{lb_im.min().item():.4f}, {lb_im.max().item():.4f}]")
    print(f"\n  → If L_pred ~ 0: encoder already learns linear z (good!)")
    print(f"    or alpha_pred weight too small, or loss not propagating")
    print(f"  → If L_eig ~ 0: omega hasn't moved from init (check gamma_eig)")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Koopman CVAE Analysis")
    print("=" * 60)

    model, cfg = load_model(CKPT_PATH)
    dataset = load_data(cfg, quality='human', seq_len=50)

    analyze_reconstruction(model, dataset, cfg)
    analyze_eigenvalues(model, cfg)
    analyze_prediction(model, dataset, cfg)
    analyze_latent(model, dataset, cfg)
    diagnose_zero_losses(model, dataset, cfg)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {SAVE_DIR}")
    print(f"{'='*60}")