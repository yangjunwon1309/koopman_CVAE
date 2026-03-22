"""
KODAC Analysis Script
=====================
Two-stream Koopman CVAE: separate state/action latent streams.

Sections:
  1. Reconstruction       — D_s: z_s -> s_hat,  D_a: z_a -> a_hat
  2. Eigenvalue           — shared omega/mu, unit circle, frequency grid
  3. Prediction           — ZOH multi-step in both streams
  4. Latent Distribution  — z_s vs z_a stats, KL, decorrelation
  5. Stream Trajectories  — z_s autonomous vs z_a input-driven
  6. Skill Posterior      — P_hat distribution, mode diversity, v_eff
  7. Rollout              — closed-loop s/a prediction vs ground truth
  8. Diagnosis            — gradient check, collapse detection
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
from models.losses import schur_block_propagate
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset
from torch.utils.data import DataLoader

# ── config ────────────────────────────────────────────────────
CKPT_PATH = os.path.expanduser('~/koopman_CVAE/checkpoints/best.pt')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 8
SAVE_DIR  = Path(os.path.expanduser('~/koopman_CVAE/analysis'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# color palette
C_STATE  = '#2196F3'   # blue  — state stream
C_ACTION = '#F44336'   # red   — action stream
C_PRED   = '#FF9800'   # orange — prediction
C_SKILL  = '#9C27B0'   # purple — skill
C_TRUE   = '#4CAF50'   # green  — ground truth


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    cfg   = ckpt['cfg']
    model = KoopmanCVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    print(f"Model loaded | m={cfg.koopman_dim}  S={cfg.num_skills}  "
          f"r={cfg.lora_rank}  kl={cfg.kl_prior}  "
          f"dt={cfg.patch_size*cfg.dt_control*1000:.0f}ms")
    return model, cfg


def load_data(cfg, quality='human', seq_len=50):
    try:
        ds = load_d4rl_trajectories('adroit_pen', seq_len=seq_len,
                                     quality=quality, min_episode_len=30)
        print(f"D4RL loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"D4RL failed ({e}), using synthetic")
        return make_synthetic_dataset(cfg.action_dim, cfg.state_dim,
                                       n_samples=200, seq_len=seq_len)


def batch(dataset, n):
    actions, states = next(iter(DataLoader(dataset, batch_size=n, shuffle=True)))
    return actions.to(DEVICE), states.to(DEVICE)


def savefig(name):
    path = SAVE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


# ─────────────────────────────────────────────────────────────
#  1. Reconstruction
# ─────────────────────────────────────────────────────────────

def analyze_reconstruction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[1] Reconstruction Analysis")
    actions, states = batch(dataset, n)
    B, T, da = actions.shape
    ds = states.shape[-1]

    with torch.no_grad():
        out = model(actions, states)
        # s_hat / a_hat are in symlog space; decode to original scale
        s_hat = symexp(out['s_hat']).cpu().numpy()   # (B, T, ds)
        a_hat = symexp(out['a_hat']).cpu().numpy()   # (B, T, da)
        s_true = states.cpu().numpy()
        a_true = actions.cpu().numpy()

        # Random latent baseline
        zs_re_rand = torch.randn(B, T, cfg.koopman_dim, device=DEVICE)
        zs_im_rand = torch.randn(B, T, cfg.koopman_dim, device=DEVICE)
        za_re_rand = torch.randn(B, T, cfg.koopman_dim, device=DEVICE)
        za_im_rand = torch.randn(B, T, cfg.koopman_dim, device=DEVICE)
        v_eff, beta_eff = model.skill_params.interpolate(out['P_hat'])
        s_rand = symexp(model.dec_s(zs_re_rand, zs_im_rand)).cpu().numpy()
        a_rand = symexp(model.dec_a(za_re_rand, za_im_rand,
                                    v_eff, beta_eff)).cpu().numpy()

    mse_s_enc  = np.mean((s_hat  - s_true)**2)
    mse_s_rand = np.mean((s_rand - s_true)**2)
    mse_a_enc  = np.mean((a_hat  - a_true)**2)
    mse_a_rand = np.mean((a_rand - a_true)**2)
    print(f"  State  MSE  encoded={mse_s_enc:.4f}  random={mse_s_rand:.4f}")
    print(f"  Action MSE  encoded={mse_a_enc:.4f}  random={mse_a_rand:.4f}")

    t = np.arange(T)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))

    # State reconstruction (left column)
    for row, d in enumerate([0, 5, 10, 15]):
        ax = axes[row, 0]
        ax.plot(t, s_true[0, :, d % ds], C_TRUE,   lw=2,   label='True')
        ax.plot(t, s_hat[0, :, d % ds],  C_STATE,  lw=1.5, ls='--', label='Recon')
        ax.plot(t, s_rand[0, :, d % ds], 'gray',   lw=1,   ls=':',  alpha=0.6, label='Random z')
        ax.set_ylabel(f's[{d % ds}]')
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f'State Recon  MSE={mse_s_enc:.4f}')
            ax.legend(fontsize=7, loc='upper right')

    # Action reconstruction (right column)
    for row, d in enumerate([0, 5, 10, 15]):
        ax = axes[row, 1]
        ax.plot(t, a_true[0, :, d % da], C_TRUE,   lw=2,   label='True')
        ax.plot(t, a_hat[0, :, d % da],  C_ACTION, lw=1.5, ls='--', label='Recon')
        ax.plot(t, a_rand[0, :, d % da], 'gray',   lw=1,   ls=':',  alpha=0.6, label='Random z')
        ax.set_ylabel(f'a[{d % da}]')
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f'Action Recon  MSE={mse_a_enc:.4f}')
            ax.legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('Timestep')
    axes[-1, 1].set_xlabel('Timestep')
    fig.suptitle('Reconstruction: State Stream (blue) vs Action Stream (red)',
                 fontsize=12, fontweight='bold')
    savefig('1_reconstruction.png')

    # Distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for col, (true, enc, rnd, name) in enumerate([
        (s_true, s_hat,  s_rand, 'State'),
        (a_true, a_hat,  a_rand, 'Action'),
    ]):
        for row, (data, label, color) in enumerate([
            (true, 'True',    C_TRUE),
            (enc,  'Encoded', C_STATE if col==0 else C_ACTION),
        ]):
            ax = axes[row, col]
            ax.hist(data.flatten(), bins=80, density=True, alpha=0.7, color=color)
            ax.set_title(f'{name} {label}  μ={data.mean():.3f} σ={data.std():.3f}')
            ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.bar(['s_enc', 's_rand', 'a_enc', 'a_rand'],
           [mse_s_enc, mse_s_rand, mse_a_enc, mse_a_rand],
           color=[C_STATE, 'gray', C_ACTION, 'gray'], alpha=0.8)
    ax.set_title('MSE Comparison'); ax.grid(alpha=0.3)
    axes[1, 2].axis('off')
    savefig('1b_distribution.png')


# ─────────────────────────────────────────────────────────────
#  2. Eigenvalue Analysis
# ─────────────────────────────────────────────────────────────

def analyze_eigenvalues(model, cfg):
    print("\n[2] Eigenvalue Analysis")
    with torch.no_grad():
        omega    = model.koopman.omega.cpu().numpy()
        mu       = model.koopman.mu.cpu().numpy()
        lb_re, lb_im = model.koopman.get_discrete()
        lb_re    = lb_re.cpu().numpy()
        lb_im    = lb_im.cpu().numpy()
        sigma0   = model.koopman.sigma0.cpu().numpy()

    m        = cfg.koopman_dim
    dt       = cfg.patch_size * cfg.dt_control
    modulus  = np.sqrt(lb_re**2 + lb_im**2)
    phase    = np.arctan2(lb_im, lb_re)
    omega_init = np.array([math.pi * cfg.omega_max / (m + 1 - i)
                            for i in range(1, m + 1)])
    expected_mod = math.exp(cfg.mu_fixed * dt)

    print(f"  dt={dt*1000:.0f}ms  |lambda_bar| mean={modulus.mean():.4f} "
          f"(expected {expected_mod:.4f})")
    print(f"  omega range [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"  sigma0 mean={sigma0.mean():.4f}")
    # Frequency repulsion check
    diffs = np.abs(omega[:, None] - omega[None, :])
    np.fill_diagonal(diffs, np.inf)
    print(f"  Min freq gap: {diffs.min():.4f} rad  "
          f"(collapse if << {omega_init[1]-omega_init[0]:.4f})")

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig)
    idx = np.arange(m)

    # Unit circle
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k--', alpha=0.3, lw=1)
    sc = ax.scatter(lb_re, lb_im, c=idx, cmap='viridis', s=70, zorder=5)
    plt.colorbar(sc, ax=ax, label='dim')
    ax.set_aspect('equal')
    ax.set_title('Discrete Eigenvalues in C')
    ax.set_xlabel('Re'); ax.set_ylabel('Im'); ax.grid(alpha=0.2)

    # omega: init vs learned
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(idx - 0.2, omega_init, 0.4, label='Init',   alpha=0.7, color='#90CAF9')
    ax.bar(idx + 0.2, omega,      0.4, label='Learned', alpha=0.7, color=C_STATE)
    ax.set_title('omega: init vs learned'); ax.legend(); ax.grid(alpha=0.3)

    # omega drift
    ax = fig.add_subplot(gs[0, 2])
    drift = omega - omega_init
    ax.bar(idx, drift, color=np.where(drift > 0, C_ACTION, C_STATE), alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Δω = learned − init'); ax.grid(alpha=0.3)

    # Frequency gap heatmap (repulsion check)
    ax = fig.add_subplot(gs[0, 3])
    gap_mat = np.abs(omega[:, None] - omega[None, :])
    np.fill_diagonal(gap_mat, 0)
    im = ax.imshow(gap_mat, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='|ω_i − ω_j|')
    ax.set_title('Frequency Gap Matrix\n(want: non-zero everywhere)')

    # Modulus per dim
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(idx, modulus, color=C_TRUE, alpha=0.8)
    ax.axhline(expected_mod, color='red', ls='--',
               label=f'Expected={expected_mod:.4f}')
    ax.set_title('|lambda_bar| per dim'); ax.legend(); ax.grid(alpha=0.3)

    # Phase per dim
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(idx, phase, color=C_PRED, alpha=0.8)
    ax.set_title('angle(lambda_bar) per dim'); ax.grid(alpha=0.3)

    # Prior sigma0
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(idx, sigma0, color=C_SKILL, alpha=0.8)
    ax.set_title('Prior sigma_0 (learnable)'); ax.grid(alpha=0.3)

    # mu (fixed)
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(idx, mu, color='#607D8B', alpha=0.8)
    ax.axhline(cfg.mu_fixed, color='red', ls='--', label=f'mu_fixed={cfg.mu_fixed}')
    ax.set_title('mu_k (fixed decay)'); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f'Eigenvalue Analysis  m={m}  dt={dt*1000:.0f}ms',
                 fontsize=13, fontweight='bold')
    savefig('2_eigenvalues.png')


# ─────────────────────────────────────────────────────────────
#  3. Multi-step Prediction
# ─────────────────────────────────────────────────────────────

def analyze_prediction(model, dataset, cfg, n=N_SAMPLES):
    print("\n[3] Multi-step Prediction Analysis")
    actions, states = batch(dataset, n)
    B, T, da = actions.shape
    ds = states.shape[-1]
    H  = min(cfg.pred_steps, 5)
    dt = cfg.patch_size * cfg.dt_control

    with torch.no_grad():
        # Get both stream latents
        zs_re, zs_im = model.phi_s(states)   # (B, T, m)
        za_re, za_im = model.phi_a(actions)
        P_hat        = model.skill_gru(states, actions)['P_hat']
        v_eff, beta_eff = model.skill_params.interpolate(P_hat)

        mu    = model.koopman.mu
        omega = model.koopman.omega
        dt_val = model.koopman.dt

        # ── State stream prediction (autonomous ZOH) ──────────
        errs_s = {}
        for h in range(1, H + 1):
            T_a = T - h
            BT  = B * T_a
            zs_re_anc = zs_re[:, :T_a].reshape(BT, cfg.koopman_dim)
            zs_im_anc = zs_im[:, :T_a].reshape(BT, cfg.koopman_dim)
            zs_re_pred, _ = schur_block_propagate(
                zs_re_anc, zs_im_anc, mu, omega, dt_val, steps=h
            )
            zs_re_tgt = zs_re[:, h:].reshape(BT, cfg.koopman_dim)
            err = ((zs_re_pred - zs_re_tgt)**2).mean(dim=-1).reshape(B, T_a)
            errs_s[h] = err.cpu().numpy()

        # ── Action stream prediction (ZOH with beta*u) ────────
        errs_a = {}
        for h in range(1, H + 1):
            T_a   = T - h
            BT    = B * T_a
            za_re_anc = za_re[:, :T_a].reshape(BT, cfg.koopman_dim)
            za_im_anc = za_im[:, :T_a].reshape(BT, cfg.koopman_dim)
            u_anc     = actions[:, :T_a].reshape(BT, da)
            be_f      = beta_eff.unsqueeze(1).expand(-1, T_a, -1, -1).reshape(BT, cfg.koopman_dim, da)
            ve_f      = v_eff.unsqueeze(1).expand(-1, T_a, -1).reshape(BT, cfg.koopman_dim)

            beta_u  = torch.bmm(be_f, u_anc.unsqueeze(-1)).squeeze(-1)
            eff_mu  = mu.unsqueeze(0) + beta_u
            decay   = torch.exp(eff_mu * dt_val * h)
            angle   = omega.unsqueeze(0) * dt_val * h
            za_re_pred = decay * (torch.cos(angle) * za_re_anc - torch.sin(angle) * za_im_anc)

            # Observable: g = v_eff * za_re
            g_pred = (ve_f * za_re_pred).sum(dim=-1).reshape(B, T_a)
            g_tgt  = (ve_f.reshape(B, T_a, cfg.koopman_dim) *
                      za_re[:, h:]).sum(dim=-1)
            err = (g_pred - g_tgt)**2
            errs_a[h] = err.cpu().numpy()

        # ── 1-step decoded prediction ─────────────────────────
        # State: propagate z_s then decode
        zs_re_1, zs_im_1 = schur_block_propagate(
            zs_re[:, :-1], zs_im[:, :-1], mu, omega, dt_val, steps=1
        )
        s_pred_1 = symexp(model.dec_s(zs_re_1, zs_im_1)).cpu().numpy()
        s_true_1 = states[:, 1:].cpu().numpy()

        # Action: propagate z_a with input then decode
        za_re_1, za_im_1 = model.koopman.propagate_with_input(
            za_re[:, :-1].reshape((B*(T-1), cfg.koopman_dim)),
            za_im[:, :-1].reshape((B*(T-1), cfg.koopman_dim)),
            beta_eff.unsqueeze(1).expand(-1, T-1, -1, -1).reshape(B*(T-1), cfg.koopman_dim, da),
            actions[:, :-1].reshape(B*(T-1), da),
        )
        a_pred_1 = symexp(model.dec_a(
            za_re_1.reshape(B, T-1, cfg.koopman_dim),
            za_im_1.reshape(B, T-1, cfg.koopman_dim),
            v_eff, beta_eff,
        )).cpu().numpy()
        a_true_1 = actions[:, 1:].cpu().numpy()

    mse_s1 = np.mean((s_pred_1 - s_true_1)**2)
    mse_a1 = np.mean((a_pred_1 - a_true_1)**2)
    print(f"  State  stream: 1-step latent err={errs_s[1].mean():.4e}  "
          f"{H}-step={errs_s[H].mean():.4e}  decoded MSE={mse_s1:.4f}")
    print(f"  Action stream: 1-step obs err={errs_a[1].mean():.4e}    "
          f"{H}-step={errs_a[H].mean():.4e}  decoded MSE={mse_a1:.4f}")

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    colors_h = plt.cm.Reds(np.linspace(0.3, 1.0, H))

    # State stream prediction error
    ax = axes[0, 0]
    for h, c in zip(range(1, H+1), colors_h):
        ax.plot(errs_s[h].mean(axis=0), color=c, lw=2, label=f'h={h}')
    ax.set_title('State stream: ZOH prediction error (latent)'); ax.legend(fontsize=8)
    ax.set_xlabel('Timestep anchor'); ax.set_ylabel('||z_s_pred - z_s_true||²')
    ax.grid(alpha=0.3)

    # Action stream prediction error
    ax = axes[0, 1]
    colors_h2 = plt.cm.Blues(np.linspace(0.3, 1.0, H))
    for h, c in zip(range(1, H+1), colors_h2):
        ax.plot(errs_a[h].mean(axis=0), color=c, lw=2, label=f'h={h}')
    ax.set_title('Action stream: ZOH observable prediction error'); ax.legend(fontsize=8)
    ax.set_xlabel('Timestep anchor'); ax.set_ylabel('(g_pred - g_true)²')
    ax.grid(alpha=0.3)

    # 1-step state decode
    ax = axes[1, 0]
    t = np.arange(T-1)
    for d, alpha in [(0, 1.0), (3, 0.7)]:
        ax.plot(t, s_true_1[0, :, d % ds], C_TRUE,  lw=2, alpha=alpha, label=f'True s[{d%ds}]')
        ax.plot(t, s_pred_1[0, :, d % ds], C_STATE, lw=1.5, ls='--', alpha=alpha, label=f'Pred s[{d%ds}]')
    ax.set_title(f'1-step State Prediction  MSE={mse_s1:.4f}')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 1-step action decode
    ax = axes[1, 1]
    for d, alpha in [(0, 1.0), (3, 0.7)]:
        ax.plot(t, a_true_1[0, :, d % da], C_TRUE,   lw=2, alpha=alpha, label=f'True a[{d%da}]')
        ax.plot(t, a_pred_1[0, :, d % da], C_ACTION, lw=1.5, ls='--', alpha=alpha, label=f'Pred a[{d%da}]')
    ax.set_title(f'1-step Action Prediction  MSE={mse_a1:.4f}')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Per-dim latent error (state stream)
    ax = axes[2, 0]
    dim_err_s = np.stack([errs_s[h].mean() for h in range(1, H+1)])
    ax.bar(range(1, H+1), dim_err_s, color=C_STATE, alpha=0.8)
    ax.set_title('State stream: mean err vs horizon')
    ax.set_xlabel('Horizon h'); ax.grid(alpha=0.3)

    # Per-dim observable error (action stream)
    ax = axes[2, 1]
    dim_err_a = np.stack([errs_a[h].mean() for h in range(1, H+1)])
    ax.bar(range(1, H+1), dim_err_a, color=C_ACTION, alpha=0.8)
    ax.set_title('Action stream: mean observable err vs horizon')
    ax.set_xlabel('Horizon h'); ax.grid(alpha=0.3)

    fig.suptitle('Multi-step ZOH Prediction — State (blue) vs Action (red) Streams',
                 fontsize=12, fontweight='bold')
    savefig('3_prediction.png')


# ─────────────────────────────────────────────────────────────
#  4. Latent Distribution
# ─────────────────────────────────────────────────────────────

def analyze_latent(model, dataset, cfg):
    print("\n[4] Latent Distribution Analysis")
    loader = DataLoader(dataset, batch_size=min(len(dataset), 256))
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)
    B, T, _ = actions.shape

    with torch.no_grad():
        zs_re, zs_im = model.phi_s(states)   # (B, T, m)
        za_re, za_im = model.phi_a(actions)
        enc_s = model.var_s(zs_re, zs_im)
        enc_a = model.var_a(za_re, za_im)
        mu    = model.koopman.mu
        omega = model.koopman.omega
        dt    = model.koopman.dt

        # KL: state stream
        prior_re_s, prior_im_s = schur_block_propagate(
            enc_s['z_re'][:, :-1], enc_s['z_im'][:, :-1], mu, omega, dt
        )
        sigma_s = enc_s['log_sigma'][:, 1:].exp() + 1e-6
        s0      = model.koopman.sigma0
        kl_s = 0.5 * (
            ((enc_s['mu_re'][:, 1:] - prior_re_s)**2 +
             (enc_s['mu_im'][:, 1:] - prior_im_s)**2) / (s0**2 + 1e-8)
            + sigma_s**2 / (s0**2 + 1e-8) - 1.0
        ).mean(dim=0)   # (T-1, m)

        # KL: action stream
        prior_re_a, prior_im_a = schur_block_propagate(
            enc_a['z_re'][:, :-1], enc_a['z_im'][:, :-1], mu, omega, dt
        )
        sigma_a = enc_a['log_sigma'][:, 1:].exp() + 1e-6
        kl_a = 0.5 * (
            ((enc_a['mu_re'][:, 1:] - prior_re_a)**2 +
             (enc_a['mu_im'][:, 1:] - prior_im_a)**2) / (s0**2 + 1e-8)
            + sigma_a**2 / (s0**2 + 1e-8) - 1.0
        ).mean(dim=0)   # (T-1, m)

        zs_re_np = zs_re.cpu().numpy()
        za_re_np = za_re.cpu().numpy()
        zs_im_np = zs_im.cpu().numpy()
        za_im_np = za_im.cpu().numpy()
        sigma_s_np = enc_s['log_sigma'].exp().cpu().numpy()
        sigma_a_np = enc_a['log_sigma'].exp().cpu().numpy()
        kl_s_np = kl_s.cpu().numpy()   # (T-1, m)
        kl_a_np = kl_a.cpu().numpy()
        sigma0_np = s0.cpu().numpy()

    print(f"  KL state  total={kl_s_np.mean():.4f}  per-dim max={kl_s_np.mean(0).max():.4f}")
    print(f"  KL action total={kl_a_np.mean():.4f}  per-dim max={kl_a_np.mean(0).max():.4f}")
    # Decorrelation check
    zs_flat = zs_re.reshape(-1, cfg.koopman_dim)
    za_flat = za_re.reshape(-1, cfg.koopman_dim)
    cov_s = (zs_flat.T @ zs_flat / zs_flat.shape[0]).cpu().numpy()
    cov_a = (za_flat.T @ za_flat / za_flat.shape[0]).cpu().numpy()
    off_s = np.abs(cov_s - np.diag(np.diag(cov_s))).mean()
    off_a = np.abs(cov_a - np.diag(np.diag(cov_a))).mean()
    print(f"  Decorrelation: state off-diag mean={off_s:.4f}  "
          f"action off-diag mean={off_a:.4f}  (want ≈ 0)")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # z_s distribution
    ax = axes[0, 0]
    ax.hist(zs_re_np.flatten(), bins=60, density=True, alpha=0.7,
            color=C_STATE, label='z_s_re')
    ax.hist(zs_im_np.flatten(), bins=60, density=True, alpha=0.5,
            color='#90CAF9', label='z_s_im')
    ax.set_title(f'State latent z_s  μ={zs_re_np.mean():.3f} σ={zs_re_np.std():.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    # z_a distribution
    ax = axes[0, 1]
    ax.hist(za_re_np.flatten(), bins=60, density=True, alpha=0.7,
            color=C_ACTION, label='z_a_re')
    ax.hist(za_im_np.flatten(), bins=60, density=True, alpha=0.5,
            color='#FFCDD2', label='z_a_im')
    ax.set_title(f'Action latent z_a  μ={za_re_np.mean():.3f} σ={za_re_np.std():.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    # Posterior sigma comparison
    ax = axes[0, 2]
    ax.hist(sigma_s_np.flatten(), bins=60, density=True, alpha=0.7,
            color=C_STATE, label='sigma_q (state)')
    ax.hist(sigma_a_np.flatten(), bins=60, density=True, alpha=0.7,
            color=C_ACTION, label='sigma_q (action)')
    ax.axvline(sigma0_np.mean(), color='k', ls='--', lw=2,
               label=f'sigma_0={sigma0_np.mean():.3f}')
    ax.set_title('Posterior sigma vs Prior sigma_0'); ax.legend(); ax.grid(alpha=0.3)

    # KL per dim: state
    ax = axes[1, 0]
    ax.bar(range(cfg.koopman_dim), kl_s_np.mean(0), color=C_STATE, alpha=0.8)
    ax.set_title(f'KL per dim — state stream (total={kl_s_np.mean():.3f})')
    ax.grid(alpha=0.3)

    # KL per dim: action
    ax = axes[1, 1]
    ax.bar(range(cfg.koopman_dim), kl_a_np.mean(0), color=C_ACTION, alpha=0.8)
    ax.set_title(f'KL per dim — action stream (total={kl_a_np.mean():.3f})')
    ax.grid(alpha=0.3)

    # KL over time
    ax = axes[1, 2]
    ax.plot(kl_s_np.mean(1), C_STATE,  lw=2, label='State stream')
    ax.plot(kl_a_np.mean(1), C_ACTION, lw=2, label='Action stream')
    ax.set_title('KL over timesteps'); ax.legend(); ax.grid(alpha=0.3)

    # Covariance: state stream
    ax = axes[2, 0]
    im = ax.imshow(cov_s, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(cov_s).max(), vmax=np.abs(cov_s).max())
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Cov(z_s_re)  off-diag={off_s:.4f}')

    # Covariance: action stream
    ax = axes[2, 1]
    im = ax.imshow(cov_a, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(cov_a).max(), vmax=np.abs(cov_a).max())
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Cov(z_a_re)  off-diag={off_a:.4f}')

    # Norm per dim comparison
    ax = axes[2, 2]
    idx = np.arange(cfg.koopman_dim)
    ax.bar(idx - 0.2, np.abs(zs_re_np).mean(axis=(0,1)), 0.4,
           color=C_STATE,  alpha=0.8, label='|z_s_re| mean')
    ax.bar(idx + 0.2, np.abs(za_re_np).mean(axis=(0,1)), 0.4,
           color=C_ACTION, alpha=0.8, label='|z_a_re| mean')
    ax.set_title('Mean |z_re| per dim'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Latent Distribution — State (blue) vs Action (red) Streams',
                 fontsize=13, fontweight='bold')
    savefig('4_latent.png')


# ─────────────────────────────────────────────────────────────
#  5. Stream Trajectories
# ─────────────────────────────────────────────────────────────

def analyze_stream_trajectories(model, dataset, cfg, n=4):
    print("\n[5] Stream Trajectory Analysis")
    actions, states = batch(dataset, n)
    B, T, da = actions.shape

    with torch.no_grad():
        zs_re, zs_im = model.phi_s(states)
        za_re, za_im = model.phi_a(actions)
        P_hat        = model.skill_gru(states, actions)['P_hat']
        v_eff, beta_eff = model.skill_params.interpolate(P_hat)

        mu    = model.koopman.mu
        omega = model.koopman.omega
        dt    = model.koopman.dt

    zs_re_np = zs_re.cpu().numpy()
    za_re_np = za_re.cpu().numpy()

    # State: predict from t=0 using autonomous ZOH
    with torch.no_grad():
        z0_s_re = zs_re[:, 0]   # (B, m)
        z0_s_im = zs_im[:, 0]
        zs_re_roll, zs_im_roll = model.koopman.rollout(z0_s_re, z0_s_im, T-1)
        # (B, T-1, m)

    # Action: predict from t=0 using ZOH with actions
    with torch.no_grad():
        z0_a_re = za_re[:, 0]
        z0_a_im = za_im[:, 0]
        za_re_pred_list = []
        z_re_cur = z0_a_re
        z_im_cur = z0_a_im
        for t in range(T - 1):
            z_re_cur, z_im_cur = model.koopman.propagate_with_input(
                z_re_cur, z_im_cur, beta_eff, actions[:, t]
            )
            za_re_pred_list.append(z_re_cur.cpu().numpy())
        za_re_pred = np.stack(za_re_pred_list, axis=1)   # (B, T-1, m)

    zs_re_roll_np = zs_re_roll.cpu().numpy()

    dims = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for col, d in enumerate(dims):
        t_ax = np.arange(T)

        # State stream: encoder vs Koopman rollout
        ax = axes[0, col]
        for b in range(n):
            ax.plot(t_ax, zs_re_np[b, :, d],
                    color=C_STATE, alpha=0.5, lw=1.5)
        # Koopman prediction (from t=0)
        t_pred = np.arange(1, T)
        for b in range(1):
            ax.plot(t_pred, zs_re_roll_np[b, :, d],
                    'k--', lw=2, label='Koopman pred (auto)')
        ax.set_title(f'State z_s_re[{d}]')
        ax.set_xlabel('t'); ax.grid(alpha=0.3)
        if col == 0:
            ax.plot([], [], C_STATE, lw=1.5, label='Encoder')
            ax.legend(fontsize=7)

        # Action stream: encoder vs ZOH with u
        ax = axes[1, col]
        for b in range(n):
            ax.plot(t_ax, za_re_np[b, :, d],
                    color=C_ACTION, alpha=0.5, lw=1.5)
        for b in range(1):
            ax.plot(t_pred, za_re_pred[b, :, d],
                    'k--', lw=2, label='ZOH+u pred')
        ax.set_title(f'Action z_a_re[{d}]')
        ax.set_xlabel('t'); ax.grid(alpha=0.3)
        if col == 0:
            ax.plot([], [], C_ACTION, lw=1.5, label='Encoder')
            ax.legend(fontsize=7)

    fig.suptitle('Stream Trajectories: Encoder vs Koopman Prediction from t=0\n'
                 'Top: State (autonomous A·z_s)  Bottom: Action ((A+Bu)·z_a)',
                 fontsize=12, fontweight='bold')
    savefig('5_trajectories.png')


# ─────────────────────────────────────────────────────────────
#  6. Skill Posterior Analysis
# ─────────────────────────────────────────────────────────────

def analyze_skills(model, dataset, cfg):
    print("\n[6] Skill Posterior Analysis")
    loader  = DataLoader(dataset, batch_size=min(len(dataset), 256))
    actions, states = next(iter(loader))
    actions, states = actions.to(DEVICE), states.to(DEVICE)
    B = actions.shape[0]

    with torch.no_grad():
        skill_out = model.skill_gru(states, actions)
        P_hat     = skill_out['P_hat'].cpu().numpy()           # (B, S)
        P_seq     = skill_out['P_hat_seq'].cpu().numpy()       # (B, T, S)
        V         = model.skill_params.V.cpu().numpy()         # (S, m)
        beta_all  = model.skill_params.get_beta().cpu().numpy()# (S, m, da)

    S  = cfg.num_skills
    m  = cfg.koopman_dim
    da = cfg.action_dim

    # Effective skill distribution
    dominant = P_hat.argmax(axis=1)          # (B,)
    mean_P   = P_hat.mean(axis=0)            # (S,)
    entropy  = -(P_hat * np.log(P_hat + 1e-8)).sum(axis=1).mean()
    max_prob = P_hat.max(axis=1).mean()

    print(f"  Mean P_hat: {np.array2string(mean_P, precision=3)}")
    print(f"  Mean entropy: {entropy:.4f}  "
          f"(max={math.log(S):.2f} uniform, 0=collapsed)")
    print(f"  Mean max prob: {max_prob:.4f}")

    # V mode diversity
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    cosine_sim = V_norm @ V_norm.T   # (S, S)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # P_hat distribution per skill
    ax = axes[0, 0]
    for s in range(S):
        ax.hist(P_hat[:, s], bins=30, alpha=0.5, density=True,
                label=f'skill {s}')
    ax.set_title(f'P_hat per skill  H={entropy:.3f}')
    ax.set_xlabel('P(skill)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # Mean P_hat bar
    ax = axes[0, 1]
    colors_s = plt.cm.Set2(np.linspace(0, 1, S))
    ax.bar(range(S), mean_P, color=colors_s, alpha=0.85)
    ax.set_title('Mean skill posterior P_hat')
    ax.set_xlabel('Skill index'); ax.grid(alpha=0.3)

    # P_hat over time (first sample)
    ax = axes[0, 2]
    for s in range(S):
        ax.plot(P_seq[0, :, s], alpha=0.8, lw=1.5, label=f's={s}')
    ax.set_title('Skill posterior over time (sample 0)')
    ax.set_xlabel('t'); ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # V cosine similarity (mode diversity)
    ax = axes[1, 0]
    im = ax.imshow(cosine_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='cosine sim')
    off_diag = np.abs(cosine_sim - np.eye(S)).mean()
    ax.set_title(f'V cosine similarity  off-diag={off_diag:.3f}\n(want ≈ 0 = diverse)')
    ax.set_xlabel('Skill i'); ax.set_ylabel('Skill j')

    # V magnitude per skill per dim
    ax = axes[1, 1]
    im = ax.imshow(np.abs(V), cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='|v_k|')
    ax.set_title('|V| (S x m)  Koopman modes')
    ax.set_xlabel('Dim k'); ax.set_ylabel('Skill')

    # beta_eff norm per skill
    ax = axes[1, 2]
    beta_norms = np.linalg.norm(beta_all.reshape(S, -1), axis=1)   # (S,)
    ax.bar(range(S), beta_norms, color=C_SKILL, alpha=0.8)
    ax.set_title('||beta_i||_F per skill (input coupling magnitude)')
    ax.set_xlabel('Skill'); ax.grid(alpha=0.3)

    fig.suptitle(f'Skill Posterior Analysis  S={S}  entropy={entropy:.3f}',
                 fontsize=13, fontweight='bold')
    savefig('6_skills.png')


# ─────────────────────────────────────────────────────────────
#  7. Closed-loop Rollout
# ─────────────────────────────────────────────────────────────

def analyze_rollout(model, dataset, cfg, n=4, horizon=20):
    print(f"\n[7] Closed-loop Rollout (horizon={horizon})")
    actions, states = batch(dataset, n)
    B, T, da = actions.shape
    ds = states.shape[-1]
    cond_len = min(10, T // 2)

    with torch.no_grad():
        out = model.rollout(
            states[:, :cond_len],
            actions[:, :cond_len],
            horizon=horizon,
        )
        s_pred = out['s_preds'].cpu().numpy()   # (B, horizon, ds)
        a_pred = out['a_preds'].cpu().numpy()   # (B, horizon, da)

    # Ground truth for the same window
    horizon_clamp = min(horizon, T - cond_len)
    s_true = states[:, cond_len:cond_len + horizon_clamp].cpu().numpy()
    a_true = actions[:, cond_len:cond_len + horizon_clamp].cpu().numpy()

    mse_s = np.mean((s_pred[:, :horizon_clamp] - s_true)**2)
    mse_a = np.mean((a_pred[:, :horizon_clamp] - a_true)**2)
    print(f"  Rollout MSE  state={mse_s:.4f}  action={mse_a:.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    t_cond = np.arange(cond_len)
    t_pred = np.arange(cond_len, cond_len + horizon)
    t_true = np.arange(cond_len, cond_len + horizon_clamp)

    for col, d in enumerate([0, 3, 6, 9]):
        # State rollout
        ax = axes[0, col]
        ax.plot(t_cond, states[0, :cond_len, d % ds].cpu(),
                C_STATE, lw=2, label='Conditioning')
        ax.plot(t_pred, s_pred[0, :, d % ds],
                C_PRED, lw=2, ls='--', label='Rollout')
        if horizon_clamp > 0:
            ax.plot(t_true, s_true[0, :, d % ds],
                    C_TRUE, lw=1.5, ls=':', label='GT')
        ax.axvline(cond_len, color='gray', ls='--', alpha=0.5)
        ax.set_title(f's[{d % ds}]')
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel('State')

        # Action rollout
        ax = axes[1, col]
        ax.plot(t_cond, actions[0, :cond_len, d % da].cpu(),
                C_ACTION, lw=2, label='Conditioning')
        ax.plot(t_pred, a_pred[0, :, d % da],
                C_PRED, lw=2, ls='--', label='Rollout')
        if horizon_clamp > 0:
            ax.plot(t_true, a_true[0, :, d % da],
                    C_TRUE, lw=1.5, ls=':', label='GT')
        ax.axvline(cond_len, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'a[{d % da}]')
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel('Action')
            ax.set_xlabel('Timestep')

    fig.suptitle(f'Closed-loop Rollout (cond={cond_len} steps, horizon={horizon})\n'
                 f'MSE: state={mse_s:.4f}  action={mse_a:.4f}',
                 fontsize=12, fontweight='bold')
    savefig('7_rollout.png')


# ─────────────────────────────────────────────────────────────
#  8. Diagnosis
# ─────────────────────────────────────────────────────────────

def diagnose(model, dataset, cfg):
    print("\n[8] Diagnosis")
    actions, states = batch(dataset, 8)

    with torch.no_grad():
        zs_re, zs_im = model.phi_s(states)
        za_re, za_im = model.phi_a(actions)
        mu    = model.koopman.mu
        omega = model.koopman.omega
        dt    = model.koopman.dt

        # State stream: 1-step prediction error
        zs_pred_re, _ = schur_block_propagate(zs_re[:, :-1], zs_im[:, :-1],
                                               mu, omega, dt, steps=1)
        pred_err_s = ((zs_pred_re - zs_re[:, 1:])**2).mean().item()

        # Action stream: observable prediction
        P_hat = model.skill_gru(states, actions)['P_hat']
        v_eff, beta_eff = model.skill_params.interpolate(P_hat)
        B, T, da = actions.shape
        m = cfg.koopman_dim
        BT = B * (T - 1)
        beta_u = torch.bmm(
            beta_eff.unsqueeze(1).expand(-1, T-1, -1, -1).reshape(BT, m, da),
            actions[:, :-1].reshape(BT, da).unsqueeze(-1)
        ).squeeze(-1)
        eff_mu  = mu.unsqueeze(0) + beta_u
        decay   = torch.exp(eff_mu * dt)
        angle   = omega.unsqueeze(0) * dt
        za_pred_re = decay * (torch.cos(angle) * za_re[:, :-1].reshape(BT, m)
                              - torch.sin(angle) * za_im[:, :-1].reshape(BT, m))
        ve_f = v_eff.unsqueeze(1).expand(-1, T-1, -1).reshape(BT, m)
        g_pred = (ve_f * za_pred_re).sum(-1)
        g_true = (ve_f * za_re[:, 1:].reshape(BT, m)).sum(-1)
        pred_err_a = ((g_pred - g_true)**2).mean().item()

        # Skill collapse check
        skill_out = model.skill_gru(states, actions)
        P = skill_out['P_hat']
        entropy = -(P * (P + 1e-8).log()).sum(-1).mean().item()
        max_P   = P.max(-1).values.mean().item()

        # Eigenvalue stability
        lb_re, lb_im = model.koopman.get_discrete()
        mod = (lb_re**2 + lb_im**2).sqrt()

        # Decorrelation
        zs_flat = zs_re.reshape(-1, m)
        za_flat = za_re.reshape(-1, m)
        cov_s = (zs_flat.T @ zs_flat / zs_flat.shape[0])
        cov_a = (za_flat.T @ za_flat / za_flat.shape[0])
        decorr_s = ((cov_s - torch.eye(m, device=DEVICE))**2).mean().item()
        decorr_a = ((cov_a - torch.eye(m, device=DEVICE))**2).mean().item()

    print(f"  State  stream: 1-step latent err = {pred_err_s:.4e}")
    print(f"  Action stream: 1-step obs err    = {pred_err_a:.4e}")
    print(f"  Skill posterior: entropy={entropy:.4f}  max_P={max_P:.4f}")
    if max_P > 0.95:
        print("  [WARNING] Skill posterior collapsed to single skill")
    if entropy < 0.1:
        print("  [WARNING] Posterior entropy very low — consider increasing delta_ent")
    print(f"  |lambda_bar|: min={mod.min().item():.4f}  max={mod.max().item():.4f}  "
          f"mean={mod.mean().item():.4f}")
    if mod.max().item() > 1.0:
        print("  [WARNING] Some eigenvalues outside unit circle — unstable!")
    print(f"  Decorrelation: state={decorr_s:.4f}  action={decorr_a:.4f}  "
          f"(want ≈ 0)")
    if decorr_s > 0.5 or decorr_a > 0.5:
        print("  [WARNING] High off-diagonal covariance — consider increasing delta_decorr")

    # omega gradient (require one backward pass)
    actions2, states2 = batch(dataset, 4)
    out = model(actions2, states2)
    out['loss'].backward()
    omega_grad = model.koopman.omega.grad
    V_grad     = model.skill_params.V.grad
    print(f"  omega grad norm: {omega_grad.norm().item():.4e}  "
          f"(None = no gradient flow)" if omega_grad is not None
          else "  omega grad: None — check prediction loss path")
    print(f"  V     grad norm: {V_grad.norm().item():.4e}"
          if V_grad is not None else "  V grad: None")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    type=str,  default=CKPT_PATH)
    p.add_argument('--quality', type=str,  default='human')
    p.add_argument('--seq_len', type=int,  default=50)
    p.add_argument('--horizon', type=int,  default=20,
                   help='Rollout horizon for section 7')
    p.add_argument('--sections', type=str, default='all',
                   help='Comma-separated sections to run, e.g. 1,2,3 or all')
    args = p.parse_args()

    CKPT_PATH = args.ckpt
    sections  = set(args.sections.split(',')) if args.sections != 'all' else None

    def run(sec_id, fn, *a, **kw):
        if sections is None or str(sec_id) in sections:
            fn(*a, **kw)

    print("=" * 60)
    print("KODAC Analysis")
    print("=" * 60)

    model, cfg = load_model(CKPT_PATH)
    dataset    = load_data(cfg, quality=args.quality, seq_len=args.seq_len)

    run(1, analyze_reconstruction,      model, dataset, cfg)
    run(2, analyze_eigenvalues,         model, cfg)
    run(3, analyze_prediction,          model, dataset, cfg)
    run(4, analyze_latent,              model, dataset, cfg)
    run(5, analyze_stream_trajectories, model, dataset, cfg)
    run(6, analyze_skills,              model, dataset, cfg)
    run(7, analyze_rollout,             model, dataset, cfg, horizon=args.horizon)
    run(8, diagnose,                    model, dataset, cfg)

    print(f"\nAll plots saved to: {SAVE_DIR}")