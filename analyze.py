"""
analyze_v5.py — KODAQ v5 Resume Checkpoint 분석
=================================================

기존 analyze.py 기능 + v5 전용 분석:

  Fig A. Rollout quality (Δq, q̇, Δp)  — 기존과 동일
  Fig B. Ensemble reward head 분석
         - 각 member별 예측 + mean + ±std 밴드 (GT sparse reward 오버레이)
         - MOPO penalized reward = mean - beta*std
         - member간 variance over time
  Fig C. Policy prior H-step rollout vs GT action
         - GT action vs π(z_t) action: per-dim deviation
         - policy prior rollout에서의 ensemble reward 예측
         - Koopman rollout latent vs posterior latent (z deviation)

Usage:
    python analyze_v5.py \\
        --ckpt   checkpoints/kodaq_v5_resume/final.pt \\
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \\
        --skill_h5 checkpoints/skill_pretrain/cluster_data.h5 \\
        --out_dir checkpoints/kodaq_v5_resume/analysis \\
        --device cuda

    # 특정 에피소드 지정:
    python analyze_v5.py --ckpt ... --ep_indices 0,3,7
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional, Dict

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from models.losses import symexp, two_hot_decode
from data.extract_skill_label import load_x_sequences, load_cluster_data


PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
       '#8E24AA','#00ACC1','#FFB300','#6D4C41','#546E7A','#D81B60']


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers  (analyze.py와 동일)
# ──────────────────────────────────────────────────────────────────────────────

def load_model_v5(ckpt_path: str, device: str) -> KoopmanCVAE:
    """v4 또는 v5 checkpoint 모두 로드 가능."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt['cfg']

    # v5 필드 없으면 기본값 주입 (v4 ckpt 호환)
    v5_defaults = dict(
        num_bins=101, v_min=0.0, v_max=5.0, num_q=2, tau=0.005,
        gamma=0.99, entropy_coef=0.01, log_std_min=-5.0, log_std_max=2.0,
        lambda_reward=1.0, lambda_q=1.0, lambda_pi=0.1,
        reward_ensemble_n=5, td_horizon=4, mopo_beta=1.0,
        use_ensemble_reward=True,
    )
    for k, v in v5_defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    model = KoopmanCVAE(cfg)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval().to(device)
    print(f"Loaded: {ckpt_path}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (first 5): {missing[:5]}")
    return model


def load_rewards(env_name: str = 'kitchen-mixed-v0') -> Optional[np.ndarray]:
    try:
        import d4rl, gym
        ds    = gym.make(env_name).get_dataset()
        r     = ds['rewards'].astype(np.float32)
        r_diff = np.clip(np.diff(r, prepend=r[0]), 0, 1)
        print(f"Rewards: shape={r_diff.shape}  nonzero={int((r_diff>0).sum())}")
        return r_diff
    except Exception as e:
        print(f"Rewards unavailable ({e}).")
        return None


def sample_episodes(x_seq, actions, terminals, assignments,
                    rewards=None, n_ep=5, device='cuda', ep_indices=None):
    ends   = list(np.where(terminals)[0])
    starts = [0] + [e + 1 for e in ends[:-1]]
    eps    = list(zip(starts, ends))
    n_total = len(eps)

    if ep_indices is not None:
        selected = [(i, eps[i]) for i in ep_indices if 0 <= i < n_total]
    else:
        sorted_eps = sorted(enumerate(eps),
                            key=lambda ie: ie[1][1]-ie[1][0], reverse=True)
        selected   = sorted_eps[:n_ep]

    samples = []
    for ep_idx, (s, e) in selected:
        L = e - s + 1
        samp = {
            'x':      torch.FloatTensor(x_seq[s:e+1]).unsqueeze(0).to(device),
            'a':      torch.FloatTensor(actions[s:e+1]).unsqueeze(0).to(device),
            'labels': assignments[s:e+1],
            'length': L, 'start': s, 'ep_idx': ep_idx,
            'rewards': (rewards[s:e+1] if rewards is not None
                        else np.zeros(L, dtype=np.float32)),
        }
        samples.append(samp)
    print(f"Sampled {len(samples)} episodes")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Fig A. Rollout Quality  (Δq, q̇, Δp overview — analyze.py 호환)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_rollout_quality(model: KoopmanCVAE, samples: list, out_dir: Path,
                         cond_len: int = 16, horizon: int = 32):
    cfg   = model.cfg
    dq_sl = slice(cfg.dim_delta_e + cfg.dim_delta_p,
                  cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q)
    qd_sl = slice(cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q,
                  cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q + cfg.dim_qdot)
    dp_sl = slice(cfg.dim_delta_e, cfg.dim_delta_e + cfg.dim_delta_p)

    n = len(samples)
    results, rmse_dq_all, rmse_dp_all = [], [], []
    ts = np.arange(horizon)

    rng = np.random.default_rng(seed=42)

    for samp in samples:
        x, a = samp['x'], samp['a']
        L = samp['length']
        if L < cond_len + horizon + 2:
            results.append(None); continue

        # Random start: pick a conditioning window anywhere in the episode
        # (not always from t=0) so we sample different dynamics regions.
        max_start = L - cond_len - horizon - 1
        t_start   = int(rng.integers(0, max(1, max_start)))
        t_cond_e  = t_start + cond_len
        t_pred_e  = t_cond_e + horizon

        x_cond = x[:, t_start:t_cond_e]
        a_cond = a[:, t_start:t_cond_e]
        a_plan = a[:, t_cond_e:t_pred_e]
        x_true = x[0, t_cond_e:t_pred_e].cpu().numpy()

        pred    = model.rollout(x_cond, a_cond, a_plan)
        dq_pred = pred['q'][0].cpu().numpy()
        dq_true = x_true[:, dq_sl]
        dp_pred = pred['delta_p'][0].cpu().numpy()
        dp_true = x_true[:, dp_sl]
        samp['_t_start']  = t_start
        samp['_t_cond_e'] = t_cond_e

        rmse_dq = np.sqrt(((dq_pred - dq_true)**2).mean(axis=0))
        rmse_dp = np.sqrt(((dp_pred - dp_true)**2).mean(axis=0))
        rmse_dq_all.append(rmse_dq)
        rmse_dp_all.append(rmse_dp)
        results.append({
            'dq_pred': dq_pred, 'dq_true': dq_true,
            'dp_pred': dp_pred, 'dp_true': dp_true,
            'rmse_dq': rmse_dq, 'rmse_dp': rmse_dp,
        })

    # Overview: N_ep × 3 (Δq, Δp top5, RMSE bar)
    cmap9 = plt.get_cmap('tab10')
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.5*n), squeeze=False)
    for i, res in enumerate(results):
        if res is None:
            for ax in axes[i]: ax.set_visible(False)
            continue
        t_s = samp.get('_t_cond_e', cond_len)
        ax = axes[i, 0]
        for d in range(9):
            ax.plot(ts, res['dq_true'][:,d], '-',  color=cmap9(d), lw=1.2, alpha=0.8)
            ax.plot(ts, res['dq_pred'][:,d], '--', color=cmap9(d), lw=1.2, alpha=0.8)
        ax.set_title(f'Ep {samp["ep_idx"]} @t={t_s}  Δq  RMSE={res["rmse_dq"].mean():.4f}',
                     fontsize=9)
        ax.set_ylabel('Δq [rad]', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        ax = axes[i, 1]
        top5 = np.argsort(res['dp_true'].var(0))[-5:]
        for j, d in enumerate(top5):
            c = PAL[j]
            ax.plot(ts, res['dp_true'][:,d], '-',  color=c, lw=1.5, alpha=0.85)
            ax.plot(ts, res['dp_pred'][:,d], '--', color=c, lw=1.5, alpha=0.85)
        ax.set_title(f'Δp top-5  RMSE={res["rmse_dp"][top5].mean():.4f}', fontsize=9)
        ax.set_ylabel('Δp', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        ax = axes[i, 2]
        ax.bar(range(9), res['rmse_dq'], color=[cmap9(d) for d in range(9)],
               alpha=0.8, label='Δq per joint')
        ax.set_title('RMSE per joint', fontsize=9)
        ax.set_xlabel('joint'); ax.set_ylabel('RMSE')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Rollout Quality (solid=GT, dashed=pred)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = str(out_dir / 'rollout_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig B. Ensemble Reward Head 분석
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_ensemble_reward(model: KoopmanCVAE, samples: list, out_dir: Path,
                         cond_len: int = 16, horizon: int = 64):
    """
    각 에피소드에 대해:
      Row 0: 각 member r̂_i(z_t, u_t) + mean + ±std 밴드  vs GT
      Row 1: MOPO penalized = mean - beta*std  vs GT
      Row 2: member간 std over time (uncertainty)
    """
    cfg  = model.cfg
    has_ens = hasattr(model, 'reward_ensemble_head')
    if not has_ens:
        print("No reward_ensemble_head found — skipping Fig B.")
        return

    beta = cfg.mopo_beta
    N    = cfg.reward_ensemble_n
    bins = model.reward_ensemble_head.bins

    n = len(samples)
    ts = np.arange(horizon)

    fig, axes = plt.subplots(n, 3, figsize=(18, 4*n), squeeze=False)

    for i, samp in enumerate(samples):
        x, a = samp['x'], samp['a']
        L    = samp['length']
        r_gt = samp['rewards']

        if L < cond_len + horizon + 2:
            for ax in axes[i]: ax.text(0.5, 0.5, 'Episode too short',
                                        ha='center', transform=ax.transAxes)
            continue

        # encode full episode → z_seq, u_seq
        enc   = model.encode_sequence(x, a)
        z_seq = enc['o_seq'][0]                      # (L, m)
        u_seq = model.action_encoder(a[0])            # (L, d_u)

        # 분석 구간: 랜덤 시작점 (rollout_quality와 동일하게 맞추거나 독립적으로)
        rng_r = np.random.default_rng(seed=samp['ep_idx'])
        max_t0 = max(1, L - cond_len - horizon - 1)
        t0 = cond_len + int(rng_r.integers(0, max_t0))
        t1 = min(t0 + horizon, L - 1)
        H  = t1 - t0

        z_w = z_seq[t0:t1]    # (H, m)
        u_w = u_seq[t0:t1]    # (H, d_u)
        r_true = r_gt[t0:t1]  # (H,)
        ts_ep  = np.arange(H)

        # Ensemble predictions: (N, H) scalar values
        vals = model.reward_ensemble_head.member_values(z_w, u_w)  # (N, H)
        vals_np = vals.cpu().numpy()                                 # (N, H)
        mu_np   = vals_np.mean(0)                                    # (H,)
        std_np  = vals_np.std(0)                                     # (H,)
        pen_np  = mu_np - beta * std_np                              # (H,) MOPO

        # ── Row 0: member별 + mean ± std + GT ─────────────────────────────
        ax = axes[i, 0]
        # GT sparse reward
        ax.bar(ts_ep, r_true, color='#43A047', alpha=0.25, width=1.0,
               label='GT sparse', zorder=1)
        for rs in np.where(r_true > 0)[0]:
            ax.axvline(rs, color='#43A047', lw=1.5, ls='--', alpha=0.5)
        # Each member (thin, transparent)
        for ni in range(N):
            ax.plot(ts_ep, vals_np[ni], color='#90CAF9', lw=0.8, alpha=0.5,
                    label='Members' if ni == 0 else '_')
        # Mean ± std band
        ax.fill_between(ts_ep, mu_np - std_np, mu_np + std_np,
                        color='#1E88E5', alpha=0.25, label='±std')
        ax.plot(ts_ep, mu_np, color='#1E88E5', lw=2.0, label='Mean')
        ax.set_title(f'Ep {samp["ep_idx"]}  Ensemble members (N={N})', fontsize=9)
        ax.set_ylabel('Reward prediction', fontsize=8)
        ax.set_xlabel('step', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines[['top','right']].set_visible(False)

        # ── Row 1: MOPO penalized reward vs GT ────────────────────────────
        ax = axes[i, 1]
        ax.bar(ts_ep, r_true, color='#43A047', alpha=0.25, width=1.0,
               label='GT sparse')
        for rs in np.where(r_true > 0)[0]:
            ax.axvline(rs, color='#43A047', lw=1.5, ls='--', alpha=0.5)
        ax.plot(ts_ep, mu_np,  color='#1E88E5', lw=1.5, ls='--', label='Mean', alpha=0.7)
        ax.plot(ts_ep, pen_np, color='#E53935', lw=2.0, label=f'Penalized (β={beta})')
        ax.fill_between(ts_ep, pen_np, mu_np,
                        color='#E53935', alpha=0.15, label='Penalty region')
        ax.set_title(f'MOPO penalized: mean - β·std', fontsize=9)
        ax.set_ylabel('Penalized reward', fontsize=8)
        ax.set_xlabel('step', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines[['top','right']].set_visible(False)

        # ── Row 2: std (uncertainty) over time ────────────────────────────
        ax = axes[i, 2]
        ax.bar(ts_ep, r_true, color='#43A047', alpha=0.2, width=1.0,
               label='GT sparse')
        ax2 = ax.twinx()
        ax2.plot(ts_ep, std_np, color='#FB8C00', lw=2.0, label='Ensemble std')
        ax2.fill_between(ts_ep, 0, std_np, color='#FB8C00', alpha=0.2)
        ax.set_title(f'Ensemble uncertainty (std)', fontsize=9)
        ax.set_ylabel('GT reward', fontsize=8, color='#43A047')
        ax2.set_ylabel('Std', fontsize=8, color='#FB8C00')
        ax.set_xlabel('step', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # stats
        mean_std = std_np.mean()
        peak_std_t = ts_ep[std_np.argmax()]
        ax.text(0.02, 0.92, f'mean_std={mean_std:.4f}  peak@t={peak_std_t}',
                transform=ax.transAxes, fontsize=7, color='#FB8C00')

    fig.suptitle('Ensemble Reward Head Analysis (v5.1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = str(out_dir / 'ensemble_reward.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig C. Policy Prior H-step Rollout  vs GT action
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_policy_prior_rollout(model: KoopmanCVAE, samples: list, out_dir: Path,
                              cond_len: int = 16, H: int = 16):
    """
    각 에피소드의 cond_len 이후 H step에 대해:

    Panel A (상단): GT action  vs  policy prior π(z_t) action  — per action dim
                    deviation = |π(z_t) - a_t_GT|  per dim + mean

    Panel B (중간): Koopman latent rollout (using π actions) vs posterior z_t
                    latent deviation = ||ẑ_{t+k} - z_{t+k}||₂

    Panel C (하단): Ensemble reward prediction along policy rollout
                    mean ± std,  MOPO penalized,  GT sparse reward
    """
    cfg  = model.cfg
    has_ens = hasattr(model, 'reward_ensemble_head')
    has_pi  = hasattr(model, 'policy_prior')
    if not has_pi:
        print("No policy_prior found — skipping Fig C.")
        return

    action_dim = cfg.action_dim  # 9
    n = len(samples)

    fig = plt.figure(figsize=(20, 9*n))

    for i, samp in enumerate(samples):
        x, a = samp['x'], samp['a']
        L    = samp['length']
        r_gt = samp['rewards']

        if L < cond_len + H + 2:
            continue

        # ── Encode full episode ────────────────────────────────────────────
        enc   = model.encode_sequence(x, a)
        z_seq = enc['o_seq'][0]      # (L, m)  posterior latents
        u_seq = model.action_encoder(a[0])  # (L, d_u)

        # Random start within episode
        rng_p = np.random.default_rng(seed=samp['ep_idx'] + 1000)
        max_t0p = max(1, L - cond_len - H - 1)
        t0 = cond_len + int(rng_p.integers(0, max_t0p))
        t1 = t0 + H

        z_start   = z_seq[t0]                    # (m,)  starting latent
        a_gt      = a[0, t0:t1].cpu().numpy()    # (H, 9) GT actions
        z_post    = z_seq[t0:t1].cpu().numpy()   # (H, m) posterior latents
        r_true    = r_gt[t0:t1]                  # (H,)

        ts = np.arange(H)

        # ── Policy prior rollout ──────────────────────────────────────────
        # ẑ_k, û_k = π(ẑ_k)  →  ẑ_{k+1} = Ā·ẑ_k + B̄·û_k
        z_roll     = z_start.unsqueeze(0)        # (1, m)
        pi_actions = []                           # list of (1, d_u) latent actions
        pi_a_raw   = []                           # raw action in action space? No —
        # policy prior operates in latent action space (d_u=64)
        # GT actions are in raw action space (9-dim)
        # We compare them via action_encoder: u_gt vs u_pi
        z_roll_list = [z_roll.squeeze(0).cpu().numpy()]
        reward_members_list = []   # (N,) per step

        # skill weights for Koopman  (use first step's h as proxy)
        # For simplicity, use soft_weights from the encoded h at t0
        # (skill weights change slowly)
        h_proxy = enc['h_seq'][0, t0].unsqueeze(0)   # (1, d_h)
        w_proxy = model.skill_prior.soft_weights(h_proxy)   # (1, K)

        for k in range(H):
            # Policy prior action in latent space
            u_pi, _, _, _ = model.policy_prior(z_roll)     # (1, d_u)
            pi_actions.append(u_pi.squeeze(0).cpu().numpy())

            # Ensemble reward at current latent
            if has_ens:
                vals = model.reward_ensemble_head.member_values(
                    z_roll, u_pi
                ).squeeze(1).cpu().numpy()   # (N,)
                reward_members_list.append(vals)

            # Koopman step
            log_lam  = model.koopman.get_log_lambdas()
            from models.losses import blend_koopman, koopman_step
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, model.koopman.theta_k,
                model.koopman.G_k, model.koopman.U, w_proxy
            )
            z_next = koopman_step(z_roll.squeeze(0), u_pi.squeeze(0),
                                  A_bar.squeeze(0), B_bar.squeeze(0))
            z_roll = z_next.unsqueeze(0)
            z_roll_list.append(z_next.cpu().numpy())

        pi_actions_np = np.array(pi_actions)        # (H, d_u)
        z_roll_np     = np.array(z_roll_list[:-1])  # (H, m)  rollout latents
        # GT encoded actions in latent space
        u_gt_np = u_seq[t0:t1].cpu().numpy()        # (H, d_u)

        # ── Latent deviation: ||ẑ_k - z_k^post||₂ ────────────────────────
        z_dev = np.linalg.norm(z_roll_np - z_post, axis=1)    # (H,)

        # ── Action deviation: ||u_pi - u_gt||  per latent dim (d_u) ──────
        a_dev_latent = np.abs(pi_actions_np - u_gt_np)         # (H, d_u)
        a_dev_mean   = a_dev_latent.mean(axis=1)               # (H,)

        # ── Reward along policy rollout ────────────────────────────────────
        if has_ens and reward_members_list:
            reward_members = np.array(reward_members_list)     # (H, N)
            r_mu   = reward_members.mean(axis=1)               # (H,)
            r_std  = reward_members.std(axis=1)                # (H,)
            r_pen  = r_mu - cfg.mopo_beta * r_std              # (H,)
        else:
            r_mu = r_std = r_pen = None

        # ── Subplot layout: 3 rows ─────────────────────────────────────────
        gs_top = gridspec.GridSpec(
            n, 3,
            figure=fig,
            top=1 - i/n + 0.02/n,
            bottom=1 - (i+1)/n + 0.04/n,
            hspace=0.5, wspace=0.35,
        )

        # Panel A: action deviation (latent space)
        ax_a = fig.add_subplot(gs_top[i, 0])
        # mean deviation
        ax_a.plot(ts, a_dev_mean, color='#E53935', lw=2.0,
                  label='mean |u_π - u_GT|')
        # top-3 dims by deviation
        top3_dims = np.argsort(a_dev_latent.mean(0))[-3:]
        cmap_d = plt.get_cmap('tab10')
        for rank, d in enumerate(top3_dims):
            ax_a.plot(ts, a_dev_latent[:, d], lw=1.0, alpha=0.7,
                      color=cmap_d(rank), ls='--',
                      label=f'dim{d}')
        ax_a.set_title(f'Ep {samp["ep_idx"]}  Action deviation\n'
                       f'(policy prior vs GT, latent space d_u={cfg.action_latent})',
                       fontsize=9)
        ax_a.set_xlabel('rollout step', fontsize=8)
        ax_a.set_ylabel('|u_π - u_GT|', fontsize=8)
        ax_a.legend(fontsize=7, loc='upper left')
        ax_a.spines[['top','right']].set_visible(False)

        # Panel B: latent rollout deviation
        ax_z = fig.add_subplot(gs_top[i, 1])
        ax_z.plot(ts, z_dev, color='#8E24AA', lw=2.0,
                  label='||ẑ_k - z_k^post||₂')
        ax_z.fill_between(ts, 0, z_dev, color='#8E24AA', alpha=0.15)
        ax_z.set_title(f'Koopman rollout latent deviation\n'
                       f'(π-action rollout vs posterior)',
                       fontsize=9)
        ax_z.set_xlabel('rollout step', fontsize=8)
        ax_z.set_ylabel('L2 deviation', fontsize=8)
        ax_z.legend(fontsize=7)
        ax_z.spines[['top','right']].set_visible(False)
        # annotate final deviation
        ax_z.text(H-1, z_dev[-1],
                  f'{z_dev[-1]:.3f}', ha='right', va='bottom', fontsize=7)

        # Panel C: reward prediction along policy rollout
        ax_r = fig.add_subplot(gs_top[i, 2])
        ax_r.bar(ts, r_true, color='#43A047', alpha=0.25, width=1.0,
                 label='GT sparse')
        for rs in np.where(r_true > 0)[0]:
            ax_r.axvline(rs, color='#43A047', lw=1.5, ls='--', alpha=0.5)
        if r_mu is not None:
            ax_r.fill_between(ts, r_mu - r_std, r_mu + r_std,
                              color='#1E88E5', alpha=0.2, label='±std')
            ax_r.plot(ts, r_mu,  color='#1E88E5', lw=1.8, label='Ens. mean')
            ax_r.plot(ts, r_pen, color='#E53935', lw=1.8, ls='--',
                      label=f'Penalized (β={cfg.mopo_beta:.1f})')
        ax_r.set_title(f'Ensemble reward along π rollout\n'
                       f'(H={H} steps)', fontsize=9)
        ax_r.set_xlabel('rollout step', fontsize=8)
        ax_r.set_ylabel('Reward prediction', fontsize=8)
        ax_r.legend(fontsize=7, loc='upper right')
        ax_r.spines[['top','right']].set_visible(False)

        # console summary
        print(f"  Ep {samp['ep_idx']:3d}: "
              f"mean_a_dev={a_dev_mean.mean():.4f}  "
              f"final_z_dev={z_dev[-1]:.4f}  "
              f"r_mu_mean={r_mu.mean():.4f}" if r_mu is not None else
              f"  Ep {samp['ep_idx']:3d}: "
              f"mean_a_dev={a_dev_mean.mean():.4f}  "
              f"final_z_dev={z_dev[-1]:.4f}")

    fig.suptitle('Policy Prior H-step Rollout Analysis (v5)',
                 fontsize=14, fontweight='bold', y=1.01)
    path = str(out_dir / 'policy_prior_rollout.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--skill_h5',   default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--env',        default='kitchen-mixed-v0')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v5_resume/analysis')
    p.add_argument('--n_ep',       type=int, default=5)
    p.add_argument('--ep_indices', type=str, default=None,
                   help='Comma-separated episode indices, e.g. "0,3,7"')
    p.add_argument('--list_eps',   action='store_true')
    p.add_argument('--cond_len',   type=int, default=16)
    p.add_argument('--horizon',    type=int, default=48,
                   help='Rollout/reward analysis horizon')
    p.add_argument('--pi_horizon', type=int, default=16,
                   help='Policy prior H-step rollout steps')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = args.device
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model   = load_model_v5(args.ckpt, device)
    x_seq, actions, terminals, assignments, K = \
        load_x_sequences(args.x_cache), None, None, None, None

    # load_x_sequences returns (x, actions, terminals)
    data = load_x_sequences(args.x_cache)
    if len(data) == 3:
        x_seq, actions, terminals = data
    else:
        x_seq, actions = data[:2]
        terminals = None

    assignments, _ = load_cluster_data(args.skill_h5)
    rewards        = load_rewards(args.env)

    if args.list_eps:
        ends   = list(np.where(terminals)[0])
        starts = [0] + [e+1 for e in ends[:-1]]
        print(f"\n{'Idx':>5}  {'Start':>7}  {'End':>7}  {'Len':>5}  {'Reward':>7}")
        print('-'*42)
        for i,(s,e) in enumerate(zip(starts,ends)):
            rs = float(rewards[s:e+1].sum()) if rewards is not None else 0.0
            print(f"{i:>5}  {s:>7}  {e:>7}  {e-s+1:>5}  {rs:>7.1f}")
        print(f"\nTotal: {len(starts)} episodes")
        exit(0)

    ep_indices = None
    if args.ep_indices:
        ep_indices = [int(x) for x in args.ep_indices.split(',')]

    samples = sample_episodes(
        x_seq, actions, terminals, assignments,
        rewards=rewards, n_ep=args.n_ep,
        device=device, ep_indices=ep_indices,
    )

    print("\n=== Fig A. Rollout Quality ===")
    plot_rollout_quality(model, samples, out,
                         cond_len=args.cond_len, horizon=args.horizon)

    print("\n=== Fig B. Ensemble Reward Head ===")
    plot_ensemble_reward(model, samples, out,
                         cond_len=args.cond_len, horizon=args.horizon)

    print("\n=== Fig C. Policy Prior H-step Rollout ===")
    plot_policy_prior_rollout(model, samples, out,
                              cond_len=args.cond_len, H=args.pi_horizon)

    print(f"\nAll outputs → {out}/")