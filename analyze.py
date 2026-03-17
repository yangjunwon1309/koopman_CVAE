"""
Koopman CVAE v2 Analysis Script
Polar latent (A, theta), temporal contrastive, multi-step prediction
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys, os, math

sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig, symlog, symexp, complex_mul
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset
from torch.utils.data import DataLoader

# ── config ────────────────────────────────────────────────
CKPT_PATH = os.path.expanduser('~/koopman_CVAE/checkpoints/v2_human/best.pt')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 8
SAVE_DIR  = Path(os.path.expanduser('~/koopman_CVAE/analysis'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
#  Load model & data
# ─────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    cfg   = ckpt['cfg']
    model = KoopmanCVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded | koopman_dim={cfg.koopman_dim} "
          f"patch_size={cfg.patch_size} dt={cfg.dt_control} "
          f"kl_prior={cfg.kl_prior}")
    return model, cfg


def load_data(cfg, quality='human', seq_len=50):
    try:
        ds = load_d4rl_trajectories('adroit_pen', seq_len=seq_len,
                                     quality=quality, min_episode_len=30)
        print(f"D4RL dataset: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"D4RL load failed ({e}), using synthetic")
        return make_synthetic_dataset(cfg.action_dim, cfg.state_dim,
                                       n_samples=200, seq_len=seq_len)


# ─────────────────────────────────────────────────────────
#  1. Reconstruction
# ─────────────────────────────────────────────────────────

def analyze_reconstruction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[1] Reconstruction Analysis")
    actions, states = next(iter(DataLoader(dataset, batch_size=n)))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        out = model(actions, states)
        patches, p_emb, s_emb = model.preprocess(actions, states)
        B, Np, patch_n, da = patches.shape

        p_hat  = symexp(out['p_hat']).cpu().numpy()
        p_true = symexp(patches).cpu().numpy()

        # Random z (polar: A~|N(0,1)|, theta~U(0,2pi))
        A_rand     = torch.abs(torch.randn(B, Np, cfg.koopman_dim, device=DEVICE))
        theta_rand = torch.rand(B, Np, cfg.koopman_dim, device=DEVICE) * 2 * math.pi
        z_rand_re  = A_rand * torch.cos(theta_rand)
        z_rand_im  = A_rand * torch.sin(theta_rand)
        p_rand = symexp(model.decode(z_rand_re, z_rand_im, s_emb)).cpu().numpy()

    T = Np * patch_n
    p_hat_f  = p_hat.reshape(n, T, da)
    p_true_f = p_true.reshape(n, T, da)
    p_rand_f = p_rand.reshape(n, T, da)

    mse_enc  = np.mean((p_hat_f  - p_true_f)**2)
    mse_rand = np.mean((p_rand_f - p_true_f)**2)
    print(f"  MSE (encoded z): {mse_enc:.4f}")
    print(f"  MSE (random z):  {mse_rand:.4f}")

    # Time-series plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for ax, d in zip(axes, [0, 5, 10, 15]):
        t = np.arange(T)
        ax.plot(t, p_true_f[0, :, d], '#2196F3', lw=2,   label=f'True a[{d}]')
        ax.plot(t, p_hat_f[0, :, d],  '#F44336', lw=1.5,
                linestyle='--', label='Recon (encoded z)')
        ax.plot(t, p_rand_f[0, :, d], '#4CAF50', lw=1,
                linestyle=':', alpha=0.7, label='Recon (random z)')
        for k in range(1, Np):
            ax.axvline(k * patch_n, color='gray', alpha=0.3)
        ax.set_ylabel(f'a[{d}]')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'Reconstruction  MSE(enc)={mse_enc:.4f}  MSE(rand)={mse_rand:.4f}')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / '1_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes,
            [p_true_f, p_hat_f, p_rand_f],
            ['True', 'Recon (encoded)', 'Recon (random)']):
        ax.hist(data.flatten(), bins=80, density=True, alpha=0.7)
        ax.set_title(f'{title}\nμ={data.mean():.3f} σ={data.std():.3f}')
        ax.set_xlabel('Action value'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / '1b_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_reconstruction.png, 1b_distribution.png")
    return mse_enc, mse_rand


# ─────────────────────────────────────────────────────────
#  2. Eigenvalue Analysis
# ─────────────────────────────────────────────────────────

def analyze_eigenvalues(model, cfg):
    print("\n[2] Eigenvalue Analysis")
    with torch.no_grad():
        omega    = model.koopman.omega.cpu().numpy()
        lb_re, lb_im = model.koopman.get_discrete()
        lb_re    = lb_re.cpu().numpy()
        lb_im    = lb_im.cpu().numpy()
        sigma_sq = model.koopman.sigma_sq.cpu().numpy()

    m       = cfg.koopman_dim
    dt      = cfg.patch_size * cfg.dt_control
    modulus = np.sqrt(lb_re**2 + lb_im**2)
    phase   = np.arctan2(lb_im, lb_re)
    omega_init = np.array([math.pi * cfg.omega_max / (m + 1 - i)
                            for i in range(1, m + 1)])

    print(f"  dt_patch={dt*1000:.0f}ms  |λ̄|={modulus.mean():.4f} "
          f"(expected {math.exp(-0.2*dt):.4f})")
    print(f"  ω range [{omega.min():.3f}, {omega.max():.3f}] rad")
    print(f"  σ² range [{sigma_sq.min():.4f}, {sigma_sq.max():.4f}]")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    idx = np.arange(m)

    # Unit circle
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k--', alpha=0.3, lw=1)
    sc = ax.scatter(lb_re, lb_im, c=idx, cmap='viridis', s=60, zorder=5)
    plt.colorbar(sc, ax=ax, label='Dim index')
    ax.set_aspect('equal')
    ax.set_title('Discrete Eigenvalues in ℂ')
    ax.set_xlabel('Re(λ̄)'); ax.set_ylabel('Im(λ̄)')
    ax.grid(alpha=0.2)

    # omega init vs learned
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(idx - 0.2, omega_init, 0.4, label='Init ω',    alpha=0.7, color='#2196F3')
    ax.bar(idx + 0.2, omega,      0.4, label='Learned ω', alpha=0.7, color='#F44336')
    ax.set_title('ω Init vs Learned'); ax.legend(); ax.grid(alpha=0.3)

    # omega drift
    ax = fig.add_subplot(gs[0, 2])
    drift = omega - omega_init
    ax.bar(idx, drift,
           color=np.where(drift > 0, '#F44336', '#2196F3'), alpha=0.7)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Δω = learned − init'); ax.grid(alpha=0.3)

    # Modulus
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(idx, modulus, color='#4CAF50', alpha=0.7)
    ax.axhline(math.exp(-0.2 * dt), color='red', linestyle='--',
               label=f'Expected={math.exp(-0.2*dt):.4f}')
    ax.set_title('|λ̄| per Dim'); ax.legend(); ax.grid(alpha=0.3)

    # Phase
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(idx, phase, color='#FF9800', alpha=0.7)
    ax.set_title('∠λ̄ per Dim'); ax.grid(alpha=0.3)

    # Process noise
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(idx, sigma_sq, color='#9C27B0', alpha=0.7)
    ax.set_title('Process Noise σᵢ²'); ax.grid(alpha=0.3)

    fig.suptitle('Koopman Eigenvalue Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / '2_eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_eigenvalues.png")


# ─────────────────────────────────────────────────────────
#  3. Prediction Analysis (multi-step, polar)
# ─────────────────────────────────────────────────────────

def analyze_prediction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[3] Multi-step Prediction Analysis")
    actions, states = next(iter(DataLoader(dataset, batch_size=n)))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    H = min(cfg.pred_steps, 5)

    with torch.no_grad():
        patches, p_emb, s_emb = model.preprocess(actions, states)
        enc = model.encode(p_emb, s_emb)

        A      = enc['A']       # (B, Np, m)
        theta  = enc['theta']
        mu_re  = enc['mu_re']
        mu_im  = enc['mu_im']

        mod, ang = model.koopman.get_modulus_angle()  # (m,)

        B, Np, m = A.shape
        da = cfg.action_dim
        pn = cfg.patch_size

    # Multi-step prediction error (latent)
    errs = {}
    with torch.no_grad():
        for h in range(1, H + 1):
            A_pred     = A[:, :Np-h] * (mod ** h)
            theta_pred = theta[:, :Np-h] + h * ang
            pred_re = A_pred * torch.cos(theta_pred)
            pred_im = A_pred * torch.sin(theta_pred)
            true_re = mu_re[:, h:]
            true_im = mu_im[:, h:]
            err = ((pred_re - true_re)**2 +
                   (pred_im - true_im)**2).mean(dim=-1).cpu().numpy()  # (B, Np-h)
            errs[h] = err

    # 1-step action prediction
    with torch.no_grad():
        A_pred1     = A[:, :-1] * mod
        theta_pred1 = theta[:, :-1] + ang
        z_re1 = A_pred1 * torch.cos(theta_pred1)
        z_im1 = A_pred1 * torch.sin(theta_pred1)
        p_pred1 = symexp(model.decode(z_re1, z_im1, s_emb[:, 1:])).cpu().numpy()
        p_true1 = symexp(patches[:, 1:]).cpu().numpy()
    mse_action = np.mean((p_pred1 - p_true1)**2)

    print(f"  1-step latent err:  {errs[1].mean():.4e}")
    print(f"  {H}-step latent err: {errs[H].mean():.4e}")
    print(f"  1-step action MSE:  {mse_action:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # A. Multi-step latent error
    ax = axes[0]
    colors_h = plt.cm.Reds(np.linspace(0.3, 1.0, H))
    for h, c in zip(range(1, H+1), colors_h):
        e = errs[h]
        mean_e = e.mean(axis=0)
        x = np.arange(len(mean_e))
        ax.plot(x, mean_e, color=c, lw=2, label=f'h={h}')
    ax.set_xlabel('Patch anchor k')
    ax.set_ylabel('|Â_{k+h} - μ_{k+h}|²')
    ax.set_title('Multi-step Koopman Prediction Error (latent)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # B. 1-step action prediction
    ax = axes[1]
    T1 = (Np - 1) * pn
    t  = np.arange(T1)
    for d, c in [(0, '#2196F3'), (5, '#F44336')]:
        ax.plot(t, p_true1[0].reshape(T1, da)[:, d], c, lw=2, label=f'True a[{d}]')
        ax.plot(t, p_pred1[0].reshape(T1, da)[:, d], c,
                lw=1.5, linestyle='--', alpha=0.8, label=f'Pred a[{d}]')
    ax.set_title(f'1-step Prediction in Action Space (MSE={mse_action:.4f})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # C. Per-dim 1-step latent error
    ax = axes[2]
    with torch.no_grad():
        dim_err = ((A[:, :-1]*mod*torch.cos(theta[:, :-1]+ang) - mu_re[:, 1:])**2 +
                   (A[:, :-1]*mod*torch.sin(theta[:, :-1]+ang) - mu_im[:, 1:])**2
                   ).mean(dim=[0, 1]).cpu().numpy()
    ax.bar(np.arange(m), dim_err, color='#9C27B0', alpha=0.7)
    ax.set_title('1-step Prediction Error per Koopman Dim')
    ax.set_xlabel('Dim i'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / '3_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_prediction.png")


# ─────────────────────────────────────────────────────────
#  4. Latent Distribution (polar)
# ─────────────────────────────────────────────────────────

def analyze_latent(model, dataset, cfg):
    print("\n[4] Latent Distribution Analysis")
    loader  = DataLoader(dataset, batch_size=len(dataset))
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, p_emb, s_emb = model.preprocess(actions, states)
        enc = model.encode(p_emb, s_emb)

        A      = enc['A'].cpu().numpy()       # (B, Np, m)
        theta  = enc['theta'].cpu().numpy()
        sigma  = enc['sigma'].cpu().numpy()
        mu_re  = enc['mu_re'].cpu().numpy()
        mu_im  = enc['mu_im'].cpu().numpy()
        z_re   = enc['z_re'].cpu().numpy()
        z_im   = enc['z_im'].cpu().numpy()

        sigma_sq = model.koopman.sigma_sq.cpu().numpy()   # (m,)
        lb_re_t, lb_im_t = model.koopman.get_discrete()

        # KL computation (koopman prior)
        z_re_t = enc['z_re'][:, :-1]
        z_im_t = enc['z_im'][:, :-1]
        mu0_re, mu0_im = complex_mul(lb_re_t, lb_im_t, z_re_t, z_im_t)
        mu0_re = mu0_re.cpu().numpy()
        mu0_im = mu0_im.cpu().numpy()

        mu_re_k = mu_re[:, 1:]
        mu_im_k = mu_im[:, 1:]
        sk_sq   = sigma[:, 1:] ** 2
        s0_sq   = sigma_sq[np.newaxis, np.newaxis, :]

        kl_per = (
            ((mu_re_k - mu0_re)**2 + (mu_im_k - mu0_im)**2) / (s0_sq + 1e-8)
            + sk_sq / (s0_sq + 1e-8)
            - np.log(sk_sq / (s0_sq + 1e-8) + 1e-8)
            - 1.0
        )

    kl_per_dim   = kl_per.mean(axis=(0, 1))
    kl_per_patch = kl_per.mean(axis=(0, 2))
    mu_norm = np.sqrt(mu_re**2 + mu_im**2)   # amplitude = A

    print(f"  KL total: {kl_per.mean():.4f}")
    print(f"  A (amplitude) mean: {A.mean():.4f}  std: {A.std():.4f}")
    print(f"  theta mean: {theta.mean():.4f}  std: {theta.std():.4f}")
    print(f"  σ_post mean: {sigma.mean():.4f}  σ_prior mean: {np.sqrt(sigma_sq.mean()):.4f}")
    print(f"  |μ| (=A) mean: {mu_norm.mean():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # KL per dim
    ax = axes[0, 0]
    ax.bar(np.arange(cfg.koopman_dim), kl_per_dim, color='#E91E63', alpha=0.7)
    ax.set_title(f'KL per Dimension (mean={kl_per.mean():.3f})')
    ax.grid(alpha=0.3)

    # KL over patches
    ax = axes[0, 1]
    ax.plot(kl_per_patch, '#E91E63', lw=2, marker='o', ms=4)
    ax.set_title('KL over Patches'); ax.grid(alpha=0.3)

    # Posterior sigma
    ax = axes[0, 2]
    ax.hist(sigma.flatten(), bins=60, density=True, alpha=0.7,
            color='#2196F3', label='Post σ')
    ax.axvline(np.sqrt(sigma_sq.mean()), color='red', linestyle='--',
               lw=2, label=f'Prior σ={np.sqrt(sigma_sq.mean()):.3f}')
    ax.set_title('Posterior vs Prior σ'); ax.legend(); ax.grid(alpha=0.3)

    # Amplitude distribution (polar)
    ax = axes[1, 0]
    ax.hist(A.flatten(), bins=60, density=True, alpha=0.7, color='#FF9800')
    ax.set_title('Amplitude A distribution\n(polar latent)')
    ax.set_xlabel('A = |z|'); ax.grid(alpha=0.3)

    # Phase distribution
    ax = axes[1, 1]
    ax.hist(theta.flatten() % (2*math.pi), bins=60, density=True,
            alpha=0.7, color='#4CAF50')
    ax.set_title('Phase θ distribution\n(polar latent, mod 2π)')
    ax.set_xlabel('θ (rad)'); ax.grid(alpha=0.3)

    # KL heatmap
    ax = axes[1, 2]
    im = ax.imshow(kl_per.mean(axis=0).T, aspect='auto',
                   cmap='hot_r', origin='lower')
    plt.colorbar(im, ax=ax, label='KL')
    ax.set_xlabel('Patch k'); ax.set_ylabel('Dim')
    ax.set_title('KL Heatmap (patch × dim)')

    fig.suptitle('Latent Distribution Analysis (Polar)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / '4_latent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_latent.png")


# ─────────────────────────────────────────────────────────
#  5. Polar Latent Trajectory
# ─────────────────────────────────────────────────────────

def analyze_polar_trajectory(model, dataset, cfg, n=4):
    print("\n[5] Polar Trajectory Analysis")
    actions, states = next(iter(DataLoader(dataset, batch_size=n)))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, p_emb, s_emb = model.preprocess(actions, states)
        enc = model.encode(p_emb, s_emb)
        A     = enc['A'].cpu().numpy()      # (B, Np, m)
        theta = enc['theta'].cpu().numpy()

        mod, ang = model.koopman.get_modulus_angle()
        mod = mod.cpu().numpy()
        ang = ang.cpu().numpy()

    Np = A.shape[1]
    # Expected trajectory from Koopman: A_0 * |lambda|^k, theta_0 + k*angle
    ks = np.arange(Np)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    dims_show = [0, 1, 2, 3]

    for col, d in enumerate(dims_show):
        # Amplitude trajectory
        ax = axes[0, col]
        for b in range(n):
            ax.plot(ks, A[b, :, d], alpha=0.6, lw=1.5, label=f'Sample {b}')
        # Expected Koopman decay from sample 0
        A0_d = A[0, 0, d]
        ax.plot(ks, A0_d * (mod[d] ** ks), 'k--', lw=2, label='Koopman pred')
        ax.set_title(f'Dim {d}: Amplitude A')
        ax.set_xlabel('Patch k'); ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)

        # Phase trajectory
        ax = axes[1, col]
        for b in range(n):
            ax.plot(ks, theta[b, :, d] % (2*math.pi), alpha=0.6, lw=1.5)
        # Expected phase: theta_0 + k*angle
        theta0_d = theta[0, 0, d]
        ax.plot(ks, (theta0_d + ks * ang[d]) % (2*math.pi),
                'k--', lw=2, label='Koopman pred')
        ax.set_title(f'Dim {d}: Phase θ')
        ax.set_xlabel('Patch k'); ax.grid(alpha=0.3)

    fig.suptitle('Polar Latent Trajectory vs Koopman Prediction\n'
                 '(black dashed = Koopman propagation from z_0)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / '5_polar_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_polar_trajectory.png")


# ─────────────────────────────────────────────────────────
#  6. Diagnosis
# ─────────────────────────────────────────────────────────

def diagnose(model, dataset, cfg):
    print("\n[6] Diagnosis")
    actions, states = next(iter(DataLoader(dataset, batch_size=8)))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        patches, p_emb, s_emb = model.preprocess(actions, states)
        enc = model.encode(p_emb, s_emb)

        A, theta = enc['A'], enc['theta']
        mod, ang = model.koopman.get_modulus_angle()

        A_pred     = A[:, :-1] * mod
        theta_pred = theta[:, :-1] + ang
        pred_re    = A_pred * torch.cos(theta_pred)
        pred_im    = A_pred * torch.sin(theta_pred)
        pred_loss_raw = ((pred_re - enc['mu_re'][:, 1:])**2 +
                         (pred_im - enc['mu_im'][:, 1:])**2).mean()

        omega_init = torch.tensor([
            math.pi * cfg.omega_max / (cfg.koopman_dim + 1 - i)
            for i in range(1, cfg.koopman_dim + 1)
        ], device=DEVICE)
        drift = (model.koopman.omega - omega_init) / (omega_init.abs() + 1e-6)
        eig_loss_raw = drift.pow(2).mean()

    print(f"  L_pred (raw):  {pred_loss_raw.item():.4e}")
    print(f"  L_eig  (raw):  {eig_loss_raw.item():.4e}")
    print(f"  A mean/std:    {A.mean().item():.4f} / {A.std().item():.4f}")
    print(f"  theta std:     {theta.std().item():.4f}")
    print(f"  mod range:     [{mod.min().item():.4f}, {mod.max().item():.4f}]")
    print(f"  omega grad:    {model.koopman.omega.grad}")
    print(f"  sigma_sq mean: {model.koopman.sigma_sq.mean().item():.4f}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    type=str, default=CKPT_PATH)
    parser.add_argument('--quality', type=str, default='human')
    parser.add_argument('--seq_len', type=int, default=50)
    args = parser.parse_args()

    CKPT_PATH = args.ckpt

    print("=" * 60)
    print("Koopman CVAE v2 Analysis")
    print("=" * 60)

    model, cfg = load_model(CKPT_PATH)
    dataset    = load_data(cfg, quality=args.quality, seq_len=args.seq_len)

    analyze_reconstruction(model, dataset, cfg)
    analyze_eigenvalues(model, cfg)
    analyze_prediction(model, dataset, cfg)
    analyze_latent(model, dataset, cfg)
    analyze_polar_trajectory(model, dataset, cfg)
    diagnose(model, dataset, cfg)

    print(f"\nAll plots → {SAVE_DIR}")