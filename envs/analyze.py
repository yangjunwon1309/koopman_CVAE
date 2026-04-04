"""
analyze.py — KODAQ 학습 결과 분석
===================================
1. Koopman 고유값 분포 + A_k 시각화
2. 스킬별 trajectory (skill weight w_t over time)
3. Rollout 예측 품질 (predicted vs true x_t)

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
import matplotlib.gridspec as gridspec
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


def sample_episodes(x_seq, actions, terminals, assignments, n_ep=5, device='cuda'):
    """에피소드 경계 기준으로 n_ep개 샘플링."""
    ends   = list(np.where(terminals)[0])
    starts = [0] + [e + 1 for e in ends[:-1]]
    eps    = list(zip(starts, ends))

    # 길이 기준 상위 n_ep 선택
    eps_sorted = sorted(eps, key=lambda se: se[1]-se[0], reverse=True)[:n_ep]

    samples = []
    for s, e in eps_sorted:
        L = e - s + 1
        samples.append({
            'x':      torch.FloatTensor(x_seq[s:e+1]).unsqueeze(0).to(device),
            'a':      torch.FloatTensor(actions[s:e+1]).unsqueeze(0).to(device),
            'labels': assignments[s:e+1],
            'length': L,
        })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 1. Koopman 고유값 분포 + A_k 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_eigenvalues(model: KoopmanCVAE, out_path: str):
    """
    A_k (K개) 각각의 복소 고유값을 단위원 위에 표시.
    skill별 색상 구분.
    """
    K  = model.cfg.num_skills
    A_k = model.koopman.get_A_k().detach().cpu()   # (K, m, m)

    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 3.5))
    if K == 1: axes = [axes]

    all_moduli = []
    for k in range(K):
        ax    = axes[k]
        A     = A_k[k]
        eigvals = torch.linalg.eigvals(A)          # (m,) complex
        re    = eigvals.real.numpy()
        im    = eigvals.imag.numpy()
        mod   = np.sqrt(re**2 + im**2)
        all_moduli.append(mod)

        # Unit circle
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.4)
        ax.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)

        ax.scatter(re, im, color=PAL[k % len(PAL)], s=25, alpha=0.8, zorder=3)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
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

    # 콘솔 요약
    print("\n=== Eigenvalue moduli per skill ===")
    for k, mod in enumerate(all_moduli):
        print(f"  Skill {k}: mean={mod.mean():.4f}  max={mod.max():.4f}  "
              f"min={mod.min():.4f}  stable={( mod <= 1.0).all()}")


def plot_A_heatmap(model: KoopmanCVAE, out_path: str):
    """A_k 행렬 heatmap (K개). 구조적 패턴 확인."""
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
# 2. 스킬별 trajectory (skill weight w_t over time)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_skill_trajectories(model: KoopmanCVAE, samples: list, out_path: str):
    """
    각 에피소드에 대해:
      상단: EXTRACT ground-truth 스킬 레이블 (color bar)
      하단: 모델 예측 skill weight w_t = softmax(W_c h_t) (stacked area)
    """
    K  = model.cfg.num_skills
    n  = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 2.5 * n),
                             gridspec_kw={'width_ratios': [1, 1]})
    if n == 1: axes = axes.reshape(1, 2)

    for i, samp in enumerate(samples):
        x, a, gt_labels = samp['x'], samp['a'], samp['labels']
        L = samp['length']

        # Encode
        enc = model.encode_sequence(x, a)
        w_seq = enc['w_seq'][0].cpu().numpy()        # (T, K)  skill weights

        # ── GT labels (left) ──────────────────────────────────────────────
        ax_gt = axes[i, 0]
        for t in range(L):
            ax_gt.axvspan(t, t+1, color=PAL[gt_labels[t] % len(PAL)],
                          alpha=0.85, linewidth=0)
        ax_gt.set_xlim(0, L)
        ax_gt.set_yticks([])
        ax_gt.set_ylabel(f'Ep {i}', rotation=0, labelpad=35, fontsize=8)
        ax_gt.set_title('EXTRACT GT labels' if i == 0 else '', fontsize=9)
        ax_gt.set_xlabel('timestep', fontsize=8)

        # ── Predicted w_t (right) ─────────────────────────────────────────
        ax_w = axes[i, 1]
        ts   = np.arange(min(L, w_seq.shape[0]))
        w_   = w_seq[:len(ts)]
        bottom = np.zeros(len(ts))
        for k in range(K):
            ax_w.fill_between(ts, bottom, bottom + w_[:, k],
                              color=PAL[k % len(PAL)], alpha=0.75,
                              label=f'Skill {k}' if i == 0 else '_')
            bottom += w_[:, k]
        ax_w.set_xlim(0, len(ts))
        ax_w.set_ylim(0, 1)
        ax_w.set_yticks([0, 0.5, 1])
        ax_w.set_title('Predicted skill weights w_t' if i == 0 else '', fontsize=9)
        ax_w.set_xlabel('timestep', fontsize=8)
        ax_w.set_ylabel('weight', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=PAL[k % len(PAL)], label=f'Skill {k}') for k in range(K)]
    fig.legend(handles=handles, loc='upper right', ncol=min(K, 4),
               fontsize=8, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle('Skill Trajectories: GT vs Predicted Weights', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rollout 예측 품질
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_rollout_quality(model: KoopmanCVAE, samples: list, out_path: str,
                         cond_len: int = 16, horizon: int = 32):
    """
    각 에피소드에서:
      - 처음 cond_len 스텝으로 conditioning
      - 이후 horizon 스텝 rollout
      - predicted vs true: Δp_t (42-dim → PCA 1D), q_t (9-dim → mean)
    """
    from models.koopman_cvae import KoopmanCVAEConfig
    cfg    = model.cfg
    dp_sl  = slice(cfg.dim_delta_e, cfg.dim_delta_e + cfg.dim_delta_p)
    q_sl   = slice(cfg.dim_delta_e + cfg.dim_delta_p,
                   cfg.dim_delta_e + cfg.dim_delta_p + cfg.dim_q)

    n   = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1: axes = axes.reshape(1, 2)

    rmse_dp_list, rmse_q_list = [], []

    for i, samp in enumerate(samples):
        x, a = samp['x'], samp['a']
        L     = samp['length']
        if L < cond_len + horizon + 2:
            for ax in axes[i]: ax.set_visible(False)
            continue

        x_cond = x[:, :cond_len]
        a_cond = a[:, :cond_len]
        a_plan = a[:, cond_len:cond_len + horizon]
        x_true = x[0, cond_len:cond_len + horizon].cpu().numpy()  # (H, x_dim)

        # Rollout
        pred = model.rollout(x_cond, a_cond, a_plan)

        # Δp_t: mean across 42 dims
        dp_pred = pred['delta_p'][0].cpu().numpy()     # (H, 42)
        dp_true = x_true[:, dp_sl]                     # (H, 42)
        dp_pred_m = dp_pred.mean(axis=1)
        dp_true_m = dp_true.mean(axis=1)

        # q_t: mean across 9 dims
        q_pred  = pred['q'][0].cpu().numpy()           # (H, 9)
        q_true  = x_true[:, q_sl]                      # (H, 9)
        q_pred_m = q_pred.mean(axis=1)
        q_true_m = q_true.mean(axis=1)

        rmse_dp = np.sqrt(((dp_pred - dp_true)**2).mean())
        rmse_q  = np.sqrt(((q_pred  - q_true)**2).mean())
        rmse_dp_list.append(rmse_dp)
        rmse_q_list.append(rmse_q)

        ts = np.arange(horizon)

        # Δp
        ax = axes[i, 0]
        ax.plot(ts, dp_true_m, 'k-',  lw=1.5, label='true')
        ax.plot(ts, dp_pred_m, '--',  lw=1.5, color=PAL[i % len(PAL)], label='pred')
        ax.axvline(0, color='gray', lw=0.8, linestyle=':')
        ax.set_title(f'Ep {i}  Δp_t (obj state mean)  RMSE={rmse_dp:.4f}',
                     fontsize=8)
        ax.set_xlabel('rollout step', fontsize=8)
        if i == 0: ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

        # q_t
        ax = axes[i, 1]
        ax.plot(ts, q_true_m, 'k-',  lw=1.5, label='true')
        ax.plot(ts, q_pred_m, '--',  lw=1.5, color=PAL[i % len(PAL)], label='pred')
        ax.set_title(f'Ep {i}  q_t (qpos mean)  RMSE={rmse_q:.4f}',
                     fontsize=8)
        ax.set_xlabel('rollout step', fontsize=8)
        if i == 0: ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle(f'Rollout Quality  (cond={cond_len} steps → pred {horizon} steps)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n=== Rollout RMSE (horizon={horizon}) ===")
    print(f"  Δp_t: {np.mean(rmse_dp_list):.4f} ± {np.std(rmse_dp_list):.4f}")
    print(f"  q_t:  {np.mean(rmse_q_list):.4f} ± {np.std(rmse_q_list):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      default='checkpoints/kodaq/best.pt')
    p.add_argument('--x_cache',   default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--skill_h5',  default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--out_dir',   default='checkpoints/kodaq/analysis')
    p.add_argument('--n_ep',      type=int, default=5,  help='시각화할 에피소드 수')
    p.add_argument('--cond_len',  type=int, default=16, help='Rollout conditioning 길이')
    p.add_argument('--horizon',   type=int, default=32, help='Rollout 예측 horizon')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = args.device
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    model   = load_model(args.ckpt, device)
    x_seq, actions, terminals, assignments, K = load_data(args.x_cache, args.skill_h5)
    samples = sample_episodes(x_seq, actions, terminals, assignments,
                              n_ep=args.n_ep, device=device)
    print(f"Sampled {len(samples)} episodes\n")

    # 1. Eigenvalues
    print("=== 1. Koopman Eigenvalues ===")
    plot_eigenvalues(model, str(out / 'eigenvalues.png'))
    plot_A_heatmap(model,   str(out / 'A_heatmap.png'))

    # 2. Skill trajectories
    print("\n=== 2. Skill Trajectories ===")
    plot_skill_trajectories(model, samples, str(out / 'skill_trajectories.png'))

    # 3. Rollout
    print("\n=== 3. Rollout Quality ===")
    plot_rollout_quality(model, samples, str(out / 'rollout_quality.png'),
                         cond_len=args.cond_len, horizon=args.horizon)

    print(f"\nAll outputs → {out}/")