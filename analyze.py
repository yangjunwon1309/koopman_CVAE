"""
analyze.py — KODAQ 학습 결과 분석
===================================
1. Koopman 고유값 분포 + A_k 시각화
2. 스킬별 trajectory (skill weight w_t over time)
3. Rollout 예측 품질
   - Δq_t  (9 joints)
   - q̇_t   (9 joint velocities, finite-diff head)
   - Δp_t  (object state top-5 active dims)
   - r_t   (reward prediction vs ground-truth sparse signal)

Usage:
    python analyze.py --ckpt checkpoints/kodaq/best.pt \
                      --x_cache checkpoints/skill_pretrain/x_sequences.npz \
                      --skill_h5 checkpoints/skill_pretrain/cluster_data.h5 \
                      --out_dir checkpoints/kodaq/analysis
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from data.extract_skill_label import load_x_sequences, load_cluster_data
from models.losses import symexp


PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
       '#8E24AA','#00ACC1','#FFB300','#6D4C41','#546E7A','#D81B60']


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> KoopmanCVAE:
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model.to(device)
    print(f"Loaded: {ckpt_path}  phase={ckpt.get('phase',3)}")
    return model


def load_data(x_cache: str, skill_h5: str):
    x_seq, actions, terminals = load_x_sequences(x_cache)
    assignments, logprobs     = load_cluster_data(skill_h5)
    K = int(assignments.max()) + 1
    print(f"Data: x={x_seq.shape}  K={K}  terminals={terminals.sum()}")
    return x_seq, actions, terminals, assignments, K


def load_rewards(env_name: str = 'kitchen-mixed-v0') -> np.ndarray:
    """D4RL에서 reward 로드 → diff (sparse 0/1)."""
    try:
        import d4rl, gym
        ds = gym.make(env_name).get_dataset()
        r  = ds['rewards'].astype(np.float32)
        r_diff = np.clip(np.diff(r, prepend=r[0]), 0, 1)
        print(f"Rewards loaded: shape={r_diff.shape}  "
              f"nonzero={int((r_diff > 0).sum())}")
        return r_diff
    except Exception as e:
        print(f"Rewards not available ({e}). Using zeros.")
        return None


def sample_episodes(x_seq, actions, terminals, assignments,
                    rewards=None, n_ep=5, device='cuda'):
    """에피소드 경계 기준으로 n_ep개 샘플링 (긴 에피소드 우선)."""
    ends   = list(np.where(terminals)[0])
    starts = [0] + [e + 1 for e in ends[:-1]]
    eps    = list(zip(starts, ends))
    eps_sorted = sorted(eps, key=lambda se: se[1]-se[0], reverse=True)[:n_ep]

    samples = []
    for s, e in eps_sorted:
        L = e - s + 1
        samp = {
            'x':      torch.FloatTensor(x_seq[s:e+1]).unsqueeze(0).to(device),
            'a':      torch.FloatTensor(actions[s:e+1]).unsqueeze(0).to(device),
            'labels': assignments[s:e+1],
            'length': L,
            'start':  s,
        }
        if rewards is not None:
            samp['rewards'] = rewards[s:e+1]   # (L,) float32 diff
        else:
            samp['rewards'] = np.zeros(L, dtype=np.float32)
        samples.append(samp)
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 1. Koopman 고유값 분포 + A_k 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_eigenvalues(model: KoopmanCVAE, out_path: str):
    K   = model.cfg.num_skills
    A_k = model.koopman.get_A_k().detach().cpu()

    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 3.5))
    if K == 1: axes = [axes]

    all_moduli = []
    for k in range(K):
        ax      = axes[k]
        eigvals = torch.linalg.eigvals(A_k[k])
        re      = eigvals.real.numpy()
        im      = eigvals.imag.numpy()
        mod     = np.sqrt(re**2 + im**2)
        all_moduli.append(mod)

        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.4)
        ax.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)
        ax.scatter(re, im, color=PAL[k % len(PAL)], s=25, alpha=0.8, zorder=3)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal')
        ax.set_title(f'Skill {k}\n|λ| mean={mod.mean():.3f}', fontsize=9)
        ax.set_xlabel('Re', fontsize=8)
        if k == 0: ax.set_ylabel('Im', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Koopman Eigenvalues per Skill  (unit circle = stability boundary)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print("\n=== Eigenvalue moduli per skill ===")
    for k, mod in enumerate(all_moduli):
        print(f"  Skill {k}: mean={mod.mean():.4f}  max={mod.max():.4f}  "
              f"min={mod.min():.4f}  stable={( mod <= 1.0).all()}")


def plot_A_heatmap(model: KoopmanCVAE, out_path: str):
    K   = model.cfg.num_skills
    m   = model.cfg.koopman_dim
    A_k = model.koopman.get_A_k().detach().cpu()

    cols = min(K, 4)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()
    vmax = A_k.abs().quantile(0.98).item()

    for k in range(K):
        ax = axes[k]
        im = ax.imshow(A_k[k].numpy(), cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'A_{k}', fontsize=9)
        ax.set_xlabel(f'm={m}', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for k in range(K, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Koopman Transition Matrices A_k', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. 스킬별 trajectory
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_skill_trajectories(model: KoopmanCVAE, samples: list, out_path: str):
    K  = model.cfg.num_skills
    n  = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 2.5 * n),
                             gridspec_kw={'width_ratios': [1, 1]})
    if n == 1: axes = axes.reshape(1, 2)

    for i, samp in enumerate(samples):
        x, a, gt_labels = samp['x'], samp['a'], samp['labels']
        L = samp['length']

        enc   = model.encode_sequence(x, a)
        w_seq = enc['w_seq'][0].cpu().numpy()

        ax_gt = axes[i, 0]
        for t in range(L):
            ax_gt.axvspan(t, t+1, color=PAL[gt_labels[t] % len(PAL)],
                          alpha=0.85, linewidth=0)
        ax_gt.set_xlim(0, L); ax_gt.set_yticks([])
        ax_gt.set_ylabel(f'Ep {i}', rotation=0, labelpad=35, fontsize=8)
        ax_gt.set_title('EXTRACT GT labels' if i == 0 else '', fontsize=9)
        ax_gt.set_xlabel('timestep', fontsize=8)

        ax_w  = axes[i, 1]
        ts    = np.arange(min(L, w_seq.shape[0]))
        w_    = w_seq[:len(ts)]
        bottom = np.zeros(len(ts))
        for k in range(K):
            ax_w.fill_between(ts, bottom, bottom + w_[:, k],
                              color=PAL[k % len(PAL)], alpha=0.75,
                              label=f'Skill {k}' if i == 0 else '_')
            bottom += w_[:, k]
        ax_w.set_xlim(0, len(ts)); ax_w.set_ylim(0, 1)
        ax_w.set_yticks([0, 0.5, 1])
        ax_w.set_title('Predicted skill weights w_t' if i == 0 else '', fontsize=9)
        ax_w.set_xlabel('timestep', fontsize=8); ax_w.set_ylabel('weight', fontsize=8)

    from matplotlib.patches import Patch
    handles = [Patch(color=PAL[k % len(PAL)], label=f'Skill {k}') for k in range(K)]
    fig.legend(handles=handles, loc='upper right', ncol=min(K, 4),
               fontsize=8, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle('Skill Trajectories: GT vs Predicted Weights',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rollout 예측 품질 (Δq, q̇, Δp, reward)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_rollout_quality(model: KoopmanCVAE, samples: list, out_path: str,
                         cond_len: int = 16, horizon: int = 32):
    """
    Per-episode rollout visualization:
      Col 0: Δq_t  — 9 joint position diffs
      Col 1: q̇_t   — 9 joint velocities (finite-diff head)
      Col 2: Δp_t  — top-5 active object state dims
      Col 3: r_t   — reward prediction (sigmoid) vs ground-truth sparse signal
    """
    cfg  = model.cfg
    dq_sl = slice(cfg.dim_delta_e + cfg.dim_delta_p,
                  cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q)
    qd_sl = slice(cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q,
                  cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q + cfg.dim_qdot)
    dp_sl = slice(cfg.dim_delta_e,
                  cfg.dim_delta_e + cfg.dim_delta_p)

    has_reward = cfg.use_reward_head
    n_cols = 4 if has_reward else 3
    n      = len(samples)

    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 3 * n))
    if n == 1: axes = axes.reshape(1, n_cols)

    cmap = plt.get_cmap('tab10')

    rmse_dq_all, rmse_qd_all, rmse_dp_all = [], [], []
    roc_auc_all = []

    for i, samp in enumerate(samples):
        x, a = samp['x'], samp['a']
        L    = samp['length']
        if L < cond_len + horizon + 2:
            for ax in axes[i]: ax.set_visible(False)
            continue

        x_cond = x[:, :cond_len]
        a_cond = a[:, :cond_len]
        a_plan = a[:, cond_len:cond_len + horizon]
        x_true = x[0, cond_len:cond_len + horizon].cpu().numpy()   # (H, x_dim)

        # Ground-truth reward segment
        r_true = samp['rewards'][cond_len:cond_len + horizon]       # (H,)

        pred   = model.rollout(x_cond, a_cond, a_plan)

        dq_pred = pred['q'][0].cpu().numpy()         # (H, 9)
        dq_true = x_true[:, dq_sl]
        qd_pred = pred['qdot'][0].cpu().numpy()      # (H, 9)
        qd_true = x_true[:, qd_sl]
        dp_pred = pred['delta_p'][0].cpu().numpy()   # (H, 42)
        dp_true = x_true[:, dp_sl]

        rmse_dq = np.sqrt(((dq_pred - dq_true)**2).mean(axis=0))   # (9,)
        rmse_qd = np.sqrt(((qd_pred - qd_true)**2).mean(axis=0))   # (9,)
        rmse_dp = np.sqrt(((dp_pred - dp_true)**2).mean(axis=0))   # (42,)
        rmse_dq_all.append(rmse_dq)
        rmse_qd_all.append(rmse_qd)
        rmse_dp_all.append(rmse_dp)

        ts = np.arange(horizon)

        # ── Col 0: Δq_t (9 joints) ────────────────────────────────────────
        ax = axes[i, 0]
        for d in range(9):
            c = cmap(d)
            ax.plot(ts, dq_true[:, d], '-',  color=c, lw=1.2, alpha=0.8)
            ax.plot(ts, dq_pred[:, d], '--', color=c, lw=1.2, alpha=0.8)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.set_title(f'Ep {i}  Δq (9 joints)\nMean RMSE={rmse_dq.mean():.4f}',
                     fontsize=9)
        ax.set_xlabel('step'); ax.set_ylabel('Δq [rad]', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # ── Col 1: q̇_t (9 joints) ─────────────────────────────────────────
        ax = axes[i, 1]
        for d in range(9):
            c = cmap(d)
            ax.plot(ts, qd_true[:, d], '-',  color=c, lw=1.2, alpha=0.8)
            ax.plot(ts, qd_pred[:, d], '--', color=c, lw=1.2, alpha=0.8)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.set_title(f'Ep {i}  q̇ (finite-diff)\nMean RMSE={rmse_qd.mean():.4f}',
                     fontsize=9)
        ax.set_xlabel('step'); ax.set_ylabel('q̇ [rad/s]', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # ── Col 2: Δp_t (top-5 active object dims) ────────────────────────
        ax = axes[i, 2]
        top5  = np.argsort(dp_true.var(axis=0))[-5:]
        for j, d in enumerate(top5):
            c = PAL[j % len(PAL)]
            ax.plot(ts, dp_true[:, d], '-',  color=c, lw=1.5, alpha=0.85,
                    label=f'dim{d}')
            ax.plot(ts, dp_pred[:, d], '--', color=c, lw=1.5, alpha=0.85)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        top5_rmse = rmse_dp[top5].mean()
        ax.set_title(f'Ep {i}  Δp (top-5 active)\nTop-5 RMSE={top5_rmse:.4f}',
                     fontsize=9)
        ax.set_xlabel('step'); ax.set_ylabel('Δp', fontsize=8)
        if i == 0: ax.legend(fontsize=7, loc='upper left')
        ax.spines[['top','right']].set_visible(False)

        # ── Col 3: r_t reward prediction ──────────────────────────────────
        if has_reward and 'reward' in pred:
            ax  = axes[i, 3]
            r_pred = pred['reward'][0, :, 0].cpu().numpy()  # (H,) sigmoid prob

            # Ground-truth: sparse step bar
            reward_steps = np.where(r_true > 0)[0]
            ax.bar(ts, r_true, color='#43A047', alpha=0.35, width=1.0,
                   label='GT reward (diff)')
            ax.plot(ts, r_pred, color='#E53935', lw=1.8,
                    label='Pred prob (sigmoid)')
            for rs in reward_steps:
                ax.axvline(rs, color='#43A047', lw=1.5, ls='--', alpha=0.7)

            # Binary cross-entropy for this segment
            eps   = 1e-7
            r_p   = np.clip(r_pred, eps, 1 - eps)
            bce   = -(r_true * np.log(r_p) + (1 - r_true) * np.log(1 - r_p)).mean()

            # AUC-ROC if both classes exist
            auc_str = ''
            if r_true.sum() > 0 and r_true.sum() < len(r_true):
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(r_true, r_pred)
                    roc_auc_all.append(auc)
                    auc_str = f'  AUC={auc:.3f}'
                except Exception:
                    pass

            ax.set_title(f'Ep {i}  Reward prediction\nBCE={bce:.4f}{auc_str}',
                         fontsize=9)
            ax.set_xlabel('step')
            ax.set_ylabel('prob / signal', fontsize=8)
            ax.set_ylim(-0.05, 1.15)
            ax.legend(fontsize=7, loc='upper left')
            ax.spines[['top','right']].set_visible(False)
        elif has_reward:
            axes[i, 3].set_title('reward head: no output', fontsize=8)
            axes[i, 3].set_visible(False)

    # ── Add legend for joint colors (col 0) ───────────────────────────────
    from matplotlib.lines import Line2D
    joint_handles = [
        Line2D([0],[0], color=cmap(d), lw=1.5, label=f'Joint {d}')
        for d in range(9)
    ]
    fig.legend(handles=joint_handles, loc='lower center',
               ncol=9, fontsize=7, bbox_to_anchor=(0.4, -0.01))

    if has_reward:
        leg_r = [
            Line2D([0],[0], color='#43A047', lw=2, label='GT reward'),
            Line2D([0],[0], color='#E53935', lw=2, label='Pred reward'),
        ]
        fig.legend(handles=leg_r, loc='lower right', fontsize=8,
                   bbox_to_anchor=(1.0, -0.01))

    fig.suptitle(
        f'Rollout Quality  cond={cond_len} → pred {horizon} steps  '
        f'(solid=true, dashed=pred)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # ── Console summary ────────────────────────────────────────────────────
    dq_avg = np.mean(rmse_dq_all, axis=0)
    qd_avg = np.mean(rmse_qd_all, axis=0)
    dp_avg = np.mean(rmse_dp_all, axis=0)

    print(f"\n=== Rollout RMSE (horizon={horizon}) ===")
    print(" [Δq_t — 9 Joints]")
    for d in range(9):
        print(f"   Joint {d}: {dq_avg[d]:.4f}")
    print(f"   Average: {dq_avg.mean():.4f}")

    print("\n [q̇_t — 9 Joint Vels (finite-diff head)]")
    for d in range(9):
        print(f"   Vel   {d}: {qd_avg[d]:.4f}")
    print(f"   Average: {qd_avg.mean():.4f}")

    print(f"\n [Δp_t — Object state (42 dims)]")
    print(f"   Overall mean RMSE: {dp_avg.mean():.4f}")
    print(f"   Top-3 highest err dims: {np.argsort(dp_avg)[-3:][::-1]}")

    if has_reward and roc_auc_all:
        print(f"\n [r_t — Reward prediction]")
        print(f"   Mean AUC-ROC: {np.mean(roc_auc_all):.4f} "
              f"(over {len(roc_auc_all)} episodes with both reward classes)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      default='checkpoints/kodaq/best.pt')
    p.add_argument('--x_cache',   default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--skill_h5',  default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--env',       default='kitchen-mixed-v0',
                   help='D4RL env name for reward loading')
    p.add_argument('--out_dir',   default='checkpoints/kodaq/analysis')
    p.add_argument('--n_ep',      type=int, default=5)
    p.add_argument('--cond_len',  type=int, default=16)
    p.add_argument('--horizon',   type=int, default=32)
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = args.device
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model   = load_model(args.ckpt, device)
    x_seq, actions, terminals, assignments, K = load_data(args.x_cache, args.skill_h5)
    rewards = load_rewards(args.env)
    samples = sample_episodes(x_seq, actions, terminals, assignments,
                              rewards=rewards, n_ep=args.n_ep, device=device)
    print(f"Sampled {len(samples)} episodes\n")

    print("=== 1. Koopman Eigenvalues ===")
    plot_eigenvalues(model, str(out / 'eigenvalues.png'))
    plot_A_heatmap(model,   str(out / 'A_heatmap.png'))

    print("\n=== 2. Skill Trajectories ===")
    plot_skill_trajectories(model, samples, str(out / 'skill_trajectories.png'))

    print("\n=== 3. Rollout Quality ===")
    plot_rollout_quality(model, samples, str(out / 'rollout_quality.png'),
                         cond_len=args.cond_len, horizon=args.horizon)

    print(f"\nAll outputs → {out}/")