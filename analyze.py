"""
KODAC-S Analysis Script
========================
Full A + Low-rank B, TCN multi-head observable, state-only prediction.

Sections:
  1. Reconstruction     — z -> s_hat vs ground truth
  2. Eigenvalue         — A spectrum, stability, diversity
  3. Prediction         — h-step ZOH latent prediction error
  4. Rollout            — closed-loop s prediction vs ground truth
  5. Latent             — z distribution, decorrelation, PCA
  6. TCN Heads          — v^(n) diversity, temporal patterns
  7. B Matrix           — input coupling structure
  8. Diagnosis          — gradient check, collapse detection
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

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig, symlog, symexp
from models.losses import propagate, propagate_h_steps
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset
from torch.utils.data import DataLoader

CKPT_PATH = os.path.expanduser('~/koopman_CVAE/checkpoints/kitchen_partial/best.pt')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 8
SAVE_DIR  = Path(os.path.expanduser('~/koopman_CVAE/analysis/KODAC'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

C_TRUE   = '#4CAF50'
C_PRED   = '#F44336'
C_LATENT = '#2196F3'
C_EIG    = '#FF9800'


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    cfg   = ckpt['cfg']
    model = KoopmanCVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    print(f"Loaded | m={cfg.koopman_dim}  Nh={cfg.num_heads}  "
          f"r={cfg.lora_rank}  dt={cfg.patch_size*cfg.dt_control*1000:.0f}ms")
    return model, cfg


def load_data(cfg, seq_len=200):
    for env_name in ['kitchen_partial', 'kitchen_mixed', 'kitchen_complete']:
        try:
            ds = load_d4rl_trajectories(env_name, seq_len=seq_len,
                                         min_episode_len=100)
            print(f"Loaded {env_name}: {len(ds)} samples")
            return ds
        except Exception as e:
            print(f"{env_name} failed: {e}")
    print("Falling back to synthetic")
    return make_synthetic_dataset(cfg.action_dim, cfg.state_dim,
                                   n_samples=200, seq_len=seq_len)


def get_batch(dataset, n):
    acts, states = next(iter(DataLoader(dataset, batch_size=n, shuffle=True)))
    return acts.to(DEVICE), states.to(DEVICE)


def savefig(name):
    plt.tight_layout()
    plt.savefig(SAVE_DIR / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {name}")


# ─────────────────────────────────────────────────────────────
#  1. Reconstruction
# ─────────────────────────────────────────────────────────────

def analyze_reconstruction(model, dataset, cfg):
    print("\n[1] Reconstruction")
    actions, states = get_batch(dataset, N_SAMPLES)
    B, T, ds = states.shape

    with torch.no_grad():
        out    = model(actions, states)
        s_hat  = symexp(out['s_hat']).cpu().numpy()
        s_true = states.cpu().numpy()
        z_rand = torch.tanh(torch.randn(B, T, cfg.koopman_dim, device=DEVICE))
        s_rand = symexp(model.decoder(z_rand)).cpu().numpy()

    s_hat  = np.clip(s_hat,  -1e3, 1e3)
    s_rand = np.clip(s_rand, -1e3, 1e3)

    mse_enc  = np.mean((s_hat  - s_true)**2)
    mse_rand = np.mean((s_rand - s_true)**2)
    ratio    = mse_rand / max(mse_enc, 1e-9)
    print(f"  MSE encoded={mse_enc:.6f}  random={mse_rand:.4f}  "
          f"ratio={ratio:.1f}x")

    mse_per_dim = np.mean((s_hat - s_true)**2, axis=(0, 1))

    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    t = np.arange(T)
    for row, d in enumerate([0, 10, 20, 30]):
        ax = axes[row % 3, row // 3]
        ax.plot(t, s_true[0, :, d % ds], C_TRUE,   lw=2,   label='True')
        ax.plot(t, s_hat[0, :, d % ds],  C_PRED,   lw=1.5, ls='--', label='Recon')
        ax.plot(t, s_rand[0, :, d % ds], 'gray',   lw=1,   ls=':', alpha=0.5,
                label='Random z')
        ax.set_title(f's[{d%ds}]'); ax.grid(alpha=0.3)
        if row == 0: ax.legend(fontsize=7)

    ax = axes[2, 0]
    ax.bar(range(ds), mse_per_dim, color=C_LATENT, alpha=0.7)
    ax.set_title(f'Per-dim recon MSE (total={mse_enc:.6f})')
    ax.set_xlabel('State dim'); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.hist(s_true.flatten(), bins=60, density=True, alpha=0.6,
            color=C_TRUE, label='True')
    ax.hist(s_hat.flatten(),  bins=60, density=True, alpha=0.6,
            color=C_PRED, label=f'Recon MSE={mse_enc:.5f}')
    ax.legend(); ax.set_title('State value distribution'); ax.grid(alpha=0.3)

    fig.suptitle(f'Reconstruction  MSE={mse_enc:.6f}  '
                 f'({ratio:.0f}x better than random)',
                 fontsize=12, fontweight='bold')
    savefig('1_reconstruction.png')


# ─────────────────────────────────────────────────────────────
#  2. Eigenvalue Analysis (Full A)
# ─────────────────────────────────────────────────────────────

def analyze_eigenvalues(model, cfg):
    print("\n[2] Eigenvalue Analysis (Full A)")
    with torch.no_grad():
        A       = model.koopman.A.cpu()
        eigvals = torch.linalg.eigvals(A)
        eig_re  = eigvals.real.numpy()
        eig_im  = eigvals.imag.numpy()
        modulus = eigvals.abs().numpy()
        phase   = torch.atan2(eigvals.imag, eigvals.real).numpy()

    m  = cfg.koopman_dim
    dt = cfg.patch_size * cfg.dt_control

    print(f"  Spectral radius: {modulus.max():.4f}  "
          f"(target <= {cfg.eig_target_radius})")
    print(f"  Modulus: mean={modulus.mean():.4f}  "
          f"min={modulus.min():.4f}  max={modulus.max():.4f}")
    print(f"  Unstable (|λ|>1): {(modulus>1).sum()}/{m}")

    with np.errstate(divide='ignore', invalid='ignore'):
        lam_cont  = np.log(eigvals.numpy()) / dt
    mu_eff    = lam_cont.real
    omega_eff = lam_cont.imag
    print(f"  Effective μ (decay):  [{mu_eff.min():.3f}, {mu_eff.max():.3f}]")
    print(f"  Effective ω (freq):   [{omega_eff.min():.3f}, {omega_eff.max():.3f}]")

    diff = eigvals.unsqueeze(0) - eigvals.unsqueeze(1)
    dist = (diff.real**2 + diff.imag**2).sqrt()
    dist.fill_diagonal_(float('inf'))
    print(f"  Min eigenvalue gap: {dist.min().item():.4f}")

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    # Unit circle
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'k--', alpha=0.3, lw=1)
    sc = ax.scatter(eig_re, eig_im, c=modulus, cmap='RdYlGn_r',
                    vmin=0.85, vmax=1.05, s=70, zorder=5)
    plt.colorbar(sc, ax=ax, label='|λ|')
    ax.set_aspect('equal')
    ax.set_title('Eigenvalues of A\n(green=stable, red=unstable)')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(alpha=0.2)

    # Modulus histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(modulus, bins=25, color=C_EIG, alpha=0.8, edgecolor='white')
    ax.axvline(1.0,                    color='red',    ls='--', lw=2, label='|λ|=1')
    ax.axvline(cfg.eig_target_radius,  color='orange', ls='--', lw=1.5,
               label=f'target={cfg.eig_target_radius}')
    ax.set_title('Eigenvalue Modulus'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Effective decay μ_eff
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(mu_eff, bins=25, color=C_LATENT, alpha=0.8, edgecolor='white')
    ax.axvline(0, color='red', ls='--', lw=2, label='stability boundary')
    ax.set_title('Effective Decay μ_eff = Re(log λ / dt)')
    ax.set_xlabel('μ_eff'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Effective frequency ω_eff
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(omega_eff, bins=25, color=C_PRED, alpha=0.8, edgecolor='white')
    ax.set_title('Effective Frequency ω_eff = Im(log λ / dt)')
    ax.set_xlabel('ω_eff (rad/s)'); ax.grid(alpha=0.3)

    # A matrix heatmap
    ax = fig.add_subplot(gs[1, 0])
    A_np = A.numpy()
    vmax = np.abs(A_np).max()
    im = ax.imshow(A_np, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'A matrix  ||A||_F={np.linalg.norm(A_np):.3f}')

    # Polar plot
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(modulus, phase, c=range(m), cmap='viridis', s=40, alpha=0.7)
    ax.axvline(1.0, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('|λ|'); ax.set_ylabel('∠λ (rad)')
    ax.set_title('Eigenvalue Polar (modulus vs phase)'); ax.grid(alpha=0.3)

    # Pairwise distance (first 32)
    ax = fig.add_subplot(gs[1, 2])
    dist_np = dist.numpy()[:32, :32]
    dist_np[np.isinf(dist_np)] = 0
    im2 = ax.imshow(dist_np, cmap='Blues', aspect='auto')
    plt.colorbar(im2, ax=ax, label='|λ_i - λ_j|')
    ax.set_title('Pairwise Eigenvalue Distance\n(first 32 dims)')

    # Sorted modulus
    ax = fig.add_subplot(gs[1, 3])
    sorted_mod = np.sort(modulus)[::-1]
    colors = [C_PRED if v > 1.0 else C_TRUE for v in sorted_mod]
    ax.bar(range(m), sorted_mod, color=colors, alpha=0.8)
    ax.axhline(1.0,                   color='red',    ls='--', lw=1.5, label='|λ|=1')
    ax.axhline(cfg.eig_target_radius, color='orange', ls='--', lw=1,
               label='target')
    ax.set_title('Sorted |λ_k|  (red=unstable)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f'Full A Eigenvalue Analysis  m={m}  '
                 f'spectral_radius={modulus.max():.4f}  '
                 f'unstable={int((modulus>1).sum())}/{m}',
                 fontsize=12, fontweight='bold')
    savefig('2_eigenvalues.png')


# ─────────────────────────────────────────────────────────────
#  3. Multi-step Prediction
# ─────────────────────────────────────────────────────────────

def analyze_prediction(model, dataset, cfg):
    print("\n[3] Multi-step Prediction")
    actions, states = get_batch(dataset, N_SAMPLES)
    B, T, da = actions.shape
    H  = min(cfg.pred_steps, 10)
    BU, BV = model.koopman.get_B_clamped()

    with torch.no_grad():
        z = model.encoder(states)

    errs = {}
    with torch.no_grad():
        for h in range(1, H + 1):
            T_a   = T - h
            BT    = B * T_a
            z_a   = z[:, :T_a].reshape(BT, cfg.koopman_dim)
            a_a   = actions[:, :T_a].reshape(BT, da)
            z_p   = propagate_h_steps(z_a, model.koopman.A, BU, BV, a_a,
                                      model.koopman.dt, h)
            z_t   = z[:, h:].reshape(BT, cfg.koopman_dim).detach()
            err   = ((z_p - z_t)**2).mean(dim=-1).reshape(B, T_a)
            errs[h] = err.cpu().numpy()

    print(f"  1-step MSE: {errs[1].mean():.6f}")
    print(f"  {H}-step MSE: {errs[H].mean():.6f}  "
          f"ratio: {errs[H].mean()/max(errs[1].mean(),1e-9):.1f}x")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    colors_h = plt.cm.Reds(np.linspace(0.3, 1.0, H))
    for h, c in zip(range(1, H+1), colors_h):
        ax.plot(errs[h].mean(axis=0), color=c, lw=2, label=f'h={h}')
    ax.set_title('Prediction Error vs Anchor Time')
    ax.set_xlabel('Anchor t'); ax.set_ylabel('MSE')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    mean_errs = [errs[h].mean() for h in range(1, H+1)]
    ax.plot(range(1, H+1), mean_errs, 'o-', color=C_PRED, lw=2, ms=6)
    ax.set_title('Mean Prediction Error vs Horizon')
    ax.set_xlabel('h'); ax.set_ylabel('Mean MSE'); ax.grid(alpha=0.3)
    for i, e in enumerate(mean_errs):
        if e > 2 * mean_errs[0]:
            ax.axvline(i+1, color='orange', ls='--', alpha=0.8,
                       label=f'2× error at h={i+1}')
            ax.legend(); break

    with torch.no_grad():
        BT1   = B * (T - 1)
        z_p1  = propagate_h_steps(
            z[:, :-1].reshape(BT1, cfg.koopman_dim),
            model.koopman.A, BU, BV,
            actions[:, :-1].reshape(BT1, da),
            model.koopman.dt, 1,
        )
        z_t1  = z[:, 1:].reshape(BT1, cfg.koopman_dim)
        dim_e = ((z_p1 - z_t1)**2).mean(dim=0).cpu().numpy()

    ax = axes[1, 0]
    ax.bar(range(cfg.koopman_dim), dim_e, color=C_LATENT, alpha=0.8)
    ax.set_title('1-step Error per Latent Dim')
    ax.set_xlabel('Dim k'); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    with torch.no_grad():
        z_roll = [z[0, 0].cpu().numpy()]
        z_cur  = z[0, 0:1]
        for t in range(min(T-1, 60)):
            z_cur = propagate(z_cur, model.koopman.A, BU, BV,
                              actions[0:1, t], model.koopman.dt)
            z_roll.append(z_cur[0].cpu().numpy())
    z_roll = np.array(z_roll)
    z_enc  = z[0].cpu().numpy()
    t_ax   = np.arange(len(z_roll))
    for d, c in enumerate(['#E53935','#1E88E5','#43A047','#FB8C00']):
        ax.plot(t_ax, z_enc[:len(t_ax), d], color=c, lw=2, alpha=0.8)
        ax.plot(t_ax, z_roll[:, d],          color=c, lw=1.5, ls='--', alpha=0.6)
    ax.set_title('Latent Trajectory: Encoder (solid) vs ZOH Rollout (dashed)')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Multi-step ZOH Prediction  H={H}',
                 fontsize=12, fontweight='bold')
    savefig('3_prediction.png')


# ─────────────────────────────────────────────────────────────
#  4. Closed-loop Rollout
# ─────────────────────────────────────────────────────────────

def analyze_rollout(model, dataset, cfg, horizon=50):
    print(f"\n[4] Rollout (horizon={horizon})")
    actions, states = get_batch(dataset, 4)
    B, T, ds = states.shape
    cond_len = min(20, T // 4)

    with torch.no_grad():
        out    = model.rollout(states[:, :cond_len], actions, horizon=horizon)
        s_pred = out['s_preds'].cpu().numpy()

    horizon_clamp = min(horizon, T - cond_len)
    s_true_r = states[:, cond_len:cond_len+horizon_clamp].cpu().numpy()
    s_pred_c = np.clip(s_pred[:, :horizon_clamp], -1e3, 1e3)

    mse_roll = np.mean((s_pred_c - s_true_r)**2)
    n_clip   = int(np.sum(np.abs(s_pred[:, :horizon_clamp]) > 1e3))
    print(f"  Rollout MSE={mse_roll:.4f}  clipped={n_clip}")

    err_over_t  = np.mean((s_pred_c - s_true_r)**2, axis=(0, 2))

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    t_cond = np.arange(cond_len)
    t_pred = np.arange(cond_len, cond_len + horizon)
    t_true = np.arange(cond_len, cond_len + horizon_clamp)

    dims_show = [0, 1, 2, 3, 9, 10, 20, 30]
    for idx, d in enumerate(dims_show):
        ax = axes[idx // 4, idx % 4]
        d  = d % ds
        ax.plot(t_cond, states[0, :cond_len, d].cpu(),
                C_TRUE, lw=2.5, label='Conditioning')
        ax.plot(t_pred, s_pred[0, :, d],
                C_PRED, lw=2, ls='--', label='Rollout')
        if horizon_clamp > 0:
            ax.plot(t_true, s_true_r[0, :, d],
                    'gray', lw=1.5, ls=':', alpha=0.8, label='GT')
        ax.axvline(cond_len, color='k', ls='--', alpha=0.4, lw=1)
        ax.set_title(f's[{d}]', fontsize=9); ax.grid(alpha=0.3)
        if idx == 0: ax.legend(fontsize=7)

    fig.suptitle(f'Closed-loop Rollout  MSE={mse_roll:.4f}  '
                 f'cond={cond_len}  horizon={horizon}',
                 fontsize=12, fontweight='bold')
    savefig('4_rollout.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(horizon_clamp), err_over_t, 'o-', color=C_PRED, lw=2, ms=4)
    axes[0].set_title('Rollout MSE vs Step')
    axes[0].set_xlabel('Step after conditioning end'); axes[0].grid(alpha=0.3)

    per_dim_e = np.mean((s_pred_c - s_true_r)**2, axis=(0, 1))
    axes[1].bar(range(ds), per_dim_e, color=C_PRED, alpha=0.8)
    axes[1].set_title('Per-dim Rollout MSE'); axes[1].grid(alpha=0.3)

    fig.suptitle('Rollout Error Analysis', fontsize=11, fontweight='bold')
    savefig('4b_rollout_error.png')


# ─────────────────────────────────────────────────────────────
#  5. Latent Space
# ─────────────────────────────────────────────────────────────

def analyze_latent(model, dataset, cfg):
    print("\n[5] Latent Space")
    loader = DataLoader(dataset, batch_size=min(len(dataset), 512))
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)

    with torch.no_grad():
        z = model.encoder(states)

    z_np   = z.reshape(-1, cfg.koopman_dim).cpu().numpy()
    z_norm = z_np / (np.linalg.norm(z_np, axis=0, keepdims=True) + 1e-8)
    corr   = z_norm.T @ z_norm
    off    = np.abs(corr - np.diag(np.diag(corr)))

    print(f"  z range [{z_np.min():.3f}, {z_np.max():.3f}]")
    print(f"  Decorrelation off-diag: mean={off.mean():.4f}  max={off.max():.4f}")
    sat = (np.abs(z_np) > 0.95).mean()
    print(f"  tanh saturation: {sat*100:.1f}%")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(z_np.flatten(), bins=80, density=True,
                    color=C_LATENT, alpha=0.8)
    axes[0, 0].set_title(f'z Distribution  μ={z_np.mean():.3f} σ={z_np.std():.3f}')
    axes[0, 0].axvline(-1, color='red', ls='--', alpha=0.5)
    axes[0, 0].axvline(1,  color='red', ls='--', alpha=0.5, label='tanh bounds')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(range(cfg.koopman_dim),
                   np.abs(z_np).mean(axis=0), color=C_LATENT, alpha=0.8)
    axes[0, 1].set_title('Mean |z_k| per Dim'); axes[0, 1].grid(alpha=0.3)

    im = axes[0, 2].imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[0, 2], label='cosine corr')
    axes[0, 2].set_title(f'z Correlation  off-diag={off.mean():.4f}')

    z1 = z[0].cpu().numpy()
    t_ax = np.arange(z1.shape[0])
    for d in range(min(8, cfg.koopman_dim)):
        axes[1, 0].plot(t_ax, z1[:, d], alpha=0.6, lw=1.2)
    axes[1, 0].set_title('z Trajectory (sample 0, first 8 dims)')
    axes[1, 0].grid(alpha=0.3)

    sat_per_dim = (np.abs(z_np) > 0.95).mean(axis=0)
    axes[1, 1].bar(range(cfg.koopman_dim), sat_per_dim,
                   color=np.where(sat_per_dim > 0.5, C_PRED, C_LATENT), alpha=0.8)
    axes[1, 1].axhline(0.5, color='red', ls='--', lw=1.5)
    axes[1, 1].set_title('tanh Saturation per Dim (red > 50%)')
    axes[1, 1].grid(alpha=0.3)

    U, S, Vt = np.linalg.svd(z_np - z_np.mean(0), full_matrices=False)
    var = (S**2 / (S**2).sum())[:20]
    axes[1, 2].bar(range(len(var)), var * 100, color=C_LATENT, alpha=0.8)
    axes[1, 2].set_title(f'PCA Variance  top1={var[0]*100:.1f}%  '
                          f'top5={var[:5].sum()*100:.1f}%')
    axes[1, 2].set_xlabel('PC'); axes[1, 2].set_ylabel('%'); axes[1, 2].grid(alpha=0.3)

    fig.suptitle('Latent Space Analysis', fontsize=13, fontweight='bold')
    savefig('5_latent.png')


# ─────────────────────────────────────────────────────────────
#  6. TCN Heads
# ─────────────────────────────────────────────────────────────

def analyze_heads(model, dataset, cfg):
    print("\n[6] TCN Multi-head Analysis")
    actions, states = get_batch(dataset, N_SAMPLES)
    B, T, _ = states.shape
    Nh, m   = cfg.num_heads, cfg.koopman_dim

    with torch.no_grad():
        v_heads = model.tcn(states, actions)          # (B, T, Nh, m)
        z       = model.encoder(states)               # (B, T, m)

    v_np = v_heads.cpu().numpy()
    z_np = z.cpu().numpy()

    v_last    = v_np[:, -1, :, :]                     # (B, Nh, m)
    v_mean    = v_last.mean(0)                        # (Nh, m)
    nrm       = np.linalg.norm(v_mean, axis=-1, keepdims=True)
    v_unit    = v_mean / (nrm + 1e-8)
    cos_sim   = v_unit @ v_unit.T
    off_cos   = np.abs(cos_sim - np.eye(Nh)).mean()
    print(f"  Head cosine sim off-diag: {off_cos:.4f}  "
          f"(want < 0.3)")

    # g^(n)_t = v^(n)_t · z_t
    g_heads = (v_np * z_np[:, :, np.newaxis, :]).sum(-1)  # (B, T, Nh)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im = axes[0, 0].imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=axes[0, 0], label='cos sim')
    axes[0, 0].set_title(f'Head Cosine Similarity  off-diag={off_cos:.3f}')

    t_ax = np.arange(T)
    for n in range(Nh):
        axes[0, 1].plot(t_ax,
                        np.linalg.norm(v_np[0, :, n, :], axis=-1),
                        alpha=0.7, lw=1.5, label=f'h{n}')
    axes[0, 1].set_title('||v^(n)_t|| over Time (sample 0)')
    axes[0, 1].legend(fontsize=6, ncol=2); axes[0, 1].grid(alpha=0.3)

    for n in range(Nh):
        axes[0, 2].plot(t_ax, g_heads[0, :, n], alpha=0.7, lw=1.5, label=f'g^({n})')
    axes[0, 2].set_title('Observable g^(n) = v^(n)·z  (sample 0)')
    axes[0, 2].legend(fontsize=6, ncol=2); axes[0, 2].grid(alpha=0.3)

    head_var = v_np.var(axis=1).mean(axis=(0, 2))    # (Nh,) temporal var
    axes[1, 0].bar(range(Nh), head_var, color=C_LATENT, alpha=0.8)
    axes[1, 0].set_title('Temporal Variance of v^(n)\n'
                          '(near-zero = TCN ignoring dynamics)')
    axes[1, 0].grid(alpha=0.3)

    v_std = v_np.std(axis=1).mean(axis=(0, 2))
    axes[1, 1].bar(range(Nh), v_std, color=C_EIG, alpha=0.8)
    axes[1, 1].set_title('Temporal Std of v^(n) per Head')
    axes[1, 1].grid(alpha=0.3)

    g_corr   = np.corrcoef(g_heads[0].T)
    off_g    = np.abs(g_corr - np.eye(Nh)).mean()
    im2 = axes[1, 2].imshow(g_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im2, ax=axes[1, 2], label='corr')
    axes[1, 2].set_title(f'Observable Correlation  off-diag={off_g:.3f}')

    fig.suptitle(f'TCN Multi-head Analysis  Nh={Nh}',
                 fontsize=12, fontweight='bold')
    savefig('6_heads.png')


# ─────────────────────────────────────────────────────────────
#  7. B Matrix
# ─────────────────────────────────────────────────────────────

def analyze_B(model, cfg):
    print("\n[7] B Matrix (Input Coupling)")
    with torch.no_grad():
        BU, BV = model.koopman.get_B_clamped()
        B_full = torch.bmm(BU, BV.transpose(-1, -2)).cpu().numpy()  # (da, m, m)

    da, m   = B_full.shape[0], B_full.shape[1]
    B_norms = np.linalg.norm(B_full.reshape(da, -1), axis=1)
    print(f"  ||B^(l)||_F: mean={B_norms.mean():.4f}  "
          f"max={B_norms.max():.4f}  argmax={B_norms.argmax()}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].bar(range(da), B_norms, color=C_EIG, alpha=0.8)
    axes[0, 0].set_title('||B^(l)||_F per Action Dim')
    axes[0, 0].set_xlabel('Action dim (joint)'); axes[0, 0].grid(alpha=0.3)

    top3 = np.argsort(B_norms)[::-1][:3]
    for i, l in enumerate(top3):
        ax  = [axes[0,1], axes[0,2], axes[1,0]][i]
        vm  = np.abs(B_full[l]).max()
        im  = ax.imshow(B_full[l], cmap='RdBu_r', aspect='auto', vmin=-vm, vmax=vm)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'B^({l})  ||B||={B_norms[l]:.4f}  (joint {l})')

    B_mean = B_full.mean(axis=0)
    vm = np.abs(B_mean).max()
    im = axes[1, 1].imshow(B_mean, cmap='RdBu_r', aspect='auto', vmin=-vm, vmax=vm)
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title('Mean B^(l) over action dims')

    diag_n    = np.array([np.abs(np.diag(B_full[l])).mean() for l in range(da)])
    offdiag_n = np.array([
        np.abs(B_full[l] - np.diag(np.diag(B_full[l]))).mean()
        for l in range(da)
    ])
    x = np.arange(da)
    axes[1, 2].bar(x-0.2, diag_n,    0.4, color=C_LATENT, alpha=0.8, label='Diag')
    axes[1, 2].bar(x+0.2, offdiag_n, 0.4, color=C_PRED,   alpha=0.8, label='Off-diag')
    axes[1, 2].set_title('B diagonal vs off-diagonal strength')
    axes[1, 2].set_xlabel('Action dim'); axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

    fig.suptitle('Input Coupling B  (B^(l) = U^(l)@V^(l).T  low-rank)',
                 fontsize=12, fontweight='bold')
    savefig('7_B_matrix.png')


# ─────────────────────────────────────────────────────────────
#  8. Diagnosis
# ─────────────────────────────────────────────────────────────

def diagnose(model, dataset, cfg):
    print("\n[8] Diagnosis")
    actions, states = get_batch(dataset, 8)
    B, T, da = actions.shape
    m  = cfg.koopman_dim
    BT = B * (T - 1)
    BU, BV = model.koopman.get_B_clamped()

    with torch.no_grad():
        z      = model.encoder(states)
        z_p1   = propagate_h_steps(
            z[:, :-1].reshape(BT, m), model.koopman.A, BU, BV,
            actions[:, :-1].reshape(BT, da), model.koopman.dt, 1)
        z_t1   = z[:, 1:].reshape(BT, m)
        pred_e = ((z_p1 - z_t1)**2).mean().item()

        eigv   = model.koopman.get_eigenvalues()
        mod    = eigv.abs()
        n_uns  = int((mod > 1.0).sum().item())

        z_flat = z.reshape(-1, m)
        zn     = F_normalize(z_flat)
        corr   = zn.T @ zn
        mask   = 1 - torch.eye(m, device=DEVICE)
        decorr = ((corr**2 * mask).sum() / (m*(m-1))).item()
        sat    = (z_flat.abs() > 0.95).float().mean().item()

    print(f"  1-step latent MSE: {pred_e:.6f}")
    print(f"  Spectral radius:   {mod.max().item():.4f}  "
          f"Unstable: {n_uns}/{m}")
    print(f"  Decorrelation:     {decorr:.4f}  (want ≈ 0)")
    print(f"  tanh saturation:   {sat*100:.1f}%")

    if n_uns > 0:
        print(f"  [WARNING] {n_uns} unstable eigenvalues — increase gamma_eig")
    if sat > 0.5:
        print(f"  [WARNING] tanh saturating {sat*100:.0f}% — grad vanishing risk")
    if decorr > 0.2:
        print(f"  [WARNING] high decorrelation {decorr:.4f} — increase delta_decorr")

    model.train()
    model.zero_grad()
    a2, s2 = get_batch(dataset, 4)
    out = model(a2, s2)
    out['loss'].backward()
    model.eval()

    A_g  = model.koopman.A.grad
    BU_g = model.koopman.B_U.grad
    enc_g = next((p.grad for p in model.encoder.parameters()
                  if p.grad is not None), None)
    tcn_g = next((p.grad for p in model.tcn.parameters()
                  if p.grad is not None), None)

    for name, g in [('A', A_g), ('B_U', BU_g),
                    ('Encoder', enc_g), ('TCN', tcn_g)]:
        if g is not None:
            print(f"  {name:8s} grad norm: {g.norm().item():.4e}")
        else:
            print(f"  {name:8s} grad: None — gradient not flowing!")


def F_normalize(x):
    return x / (x.norm(dim=0, keepdim=True) + 1e-8)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     type=str, default=CKPT_PATH)
    p.add_argument('--seq_len',  type=int, default=200)
    p.add_argument('--horizon',  type=int, default=50)
    p.add_argument('--sections', type=str, default='all',
                   help='e.g. 1,2,4 or all')
    args = p.parse_args()

    CKPT_PATH = args.ckpt
    secs = None if args.sections == 'all' \
           else set(args.sections.split(','))

    def run(sid, fn, *a, **kw):
        if secs is None or str(sid) in secs:
            fn(*a, **kw)

    print("=" * 60)
    print("KODAC-S Analysis")
    print("=" * 60)

    model, cfg = load_model(CKPT_PATH)
    dataset    = load_data(cfg, seq_len=args.seq_len)

    run(1, analyze_reconstruction, model, dataset, cfg)
    run(2, analyze_eigenvalues,    model, cfg)
    run(3, analyze_prediction,     model, dataset, cfg)
    run(4, analyze_rollout,        model, dataset, cfg, horizon=args.horizon)
    run(5, analyze_latent,         model, dataset, cfg)
    run(6, analyze_heads,          model, dataset, cfg)
    run(7, analyze_B,              model, cfg)
    run(8, diagnose,               model, dataset, cfg)

    print(f"\nAll plots saved to: {SAVE_DIR}")