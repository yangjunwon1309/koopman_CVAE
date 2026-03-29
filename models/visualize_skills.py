"""
visualize_skills.py
====================
Skill pretraining 결과 시각화:
  1. t-SNE of skill latents (z_all), colored by skill label
  2. K evolution would be logged during training
  3. Skill label distribution (per-episode)
  4. Action reconstruction (sampled episodes per skill)

Usage:
    python visualize_skills.py \
        --npz checkpoints/skill_pretrain/labels.npz \
        --ckpt checkpoints/skill_pretrain/best.pt \
        --out checkpoints/skill_pretrain/tsne.png
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse


# ─── palette ──────────────────────────────────────────────────
PALETTE = [
    '#E53935', '#1E88E5', '#43A047', '#FB8C00',
    '#8E24AA', '#00ACC1', '#FFB300', '#6D4C41',
    '#546E7A', '#D81B60', '#00897B', '#F4511E',
]

def color_for(k: int) -> str:
    return PALETTE[k % len(PALETTE)]


def load_npz(path: str):
    data = np.load(path)
    labels_hard = data['labels_hard']   # (N, T) int32
    z_all       = data['z_all']          # (N, T, d_z)
    K           = int(data['K'][0])
    labels_soft = data.get('labels_soft', None)  # (N, T, K)
    return labels_hard, z_all, K, labels_soft


def run_tsne(Z: np.ndarray, perplexity: float = 30) -> np.ndarray:
    """Z: (M, d_z) → (M, 2)"""
    print(f"  Running t-SNE on {Z.shape[0]} points, d={Z.shape[1]} ...")
    # PCA pre-reduction if d > 50
    if Z.shape[1] > 50:
        Z = PCA(n_components=50).fit_transform(Z)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000, verbose=0)
    return tsne.fit_transform(Z)


def plot_tsne_by_skill(Z2d, labels, K, ax, title="t-SNE by skill"):
    """Z2d: (M,2), labels: (M,)"""
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1],
                   c=color_for(k), alpha=0.5, s=8,
                   label=f'Skill {k} (n={mask.sum()})')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, markerscale=2, loc='best')
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',  default='checkpoints/skill_pretrain/labels.npz')
    parser.add_argument('--ckpt', default='checkpoints/skill_pretrain/best.pt')
    parser.add_argument('--out',  default='checkpoints/skill_pretrain/tsne.png')
    parser.add_argument('--subsample', type=int, default=5000,
                        help='max points for t-SNE (speed)')
    parser.add_argument('--perplexity', type=float, default=30)
    args = parser.parse_args()

    npz_path = os.path.expanduser(args.npz)
    out_path = os.path.expanduser(args.out)

    if not os.path.exists(npz_path):
        print(f"NPZ not found: {npz_path}")
        return

    labels_hard, z_all, K, labels_soft = load_npz(npz_path)
    N, T, d_z = z_all.shape
    print(f"Loaded: N={N}  T={T}  d_z={d_z}  K={K}")
    print(f"  label unique: {np.unique(labels_hard).tolist()}")

    # ── 1. Flatten for t-SNE ─────────────────────────────────
    # Use last timestep of each segment
    Z_last   = z_all[:, -1, :]             # (N, d_z)
    L_last   = labels_hard[:, -1]          # (N,)

    # Also use every 10th timestep for temporal view
    stride   = max(1, T // 10)
    Z_temp   = z_all[:, ::stride, :].reshape(-1, d_z)
    L_temp   = labels_hard[:, ::stride].reshape(-1)

    # Subsample for speed
    def subsample(Z, L, maxn):
        if len(Z) <= maxn:
            return Z, L
        idx = np.random.choice(len(Z), maxn, replace=False)
        return Z[idx], L[idx]

    Z_last_s, L_last_s = subsample(Z_last, L_last, args.subsample)
    Z_temp_s, L_temp_s = subsample(Z_temp, L_temp, args.subsample)

    # ── 2. t-SNE ─────────────────────────────────────────────
    print("Computing t-SNE (last timestep)...")
    Z2d_last = run_tsne(Z_last_s, args.perplexity)

    print("Computing t-SNE (temporal)...")
    Z2d_temp = run_tsne(Z_temp_s, args.perplexity)

    # ── 3. PCA for fast view ──────────────────────────────────
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z_last_s)
    var_exp = pca.explained_variance_ratio_

    # ── 4. Label distribution ─────────────────────────────────
    label_counts = np.bincount(labels_hard.flatten(), minlength=K)
    label_frac   = label_counts / label_counts.sum()

    # ── 5. Per-episode skill trajectory (first 8 episodes) ───
    n_ep_show = min(8, N)
    ep_indices = np.linspace(0, N-1, n_ep_show, dtype=int)

    # ── Plotting ─────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = plt.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Row 0: t-SNE last timestep
    ax00 = fig.add_subplot(gs[0, 0])
    plot_tsne_by_skill(Z2d_last, L_last_s, K, ax00,
                       f't-SNE (last step, n={len(Z_last_s)})')

    # Row 0: t-SNE temporal
    ax01 = fig.add_subplot(gs[0, 1])
    plot_tsne_by_skill(Z2d_temp, L_temp_s, K, ax01,
                       f't-SNE (temporal, stride={stride})')

    # Row 0: PCA
    ax02 = fig.add_subplot(gs[0, 2])
    plot_tsne_by_skill(Z_pca, L_last_s, K, ax02,
                       f'PCA (var={var_exp[0]:.2f}+{var_exp[1]:.2f}={sum(var_exp):.2f})')

    # Row 1: label distribution pie
    ax10 = fig.add_subplot(gs[1, 0])
    wedge_labels = [f'Skill {k}\n{label_frac[k]*100:.1f}%'
                    for k in range(K) if label_frac[k] > 0]
    wedge_sizes  = [label_frac[k] for k in range(K) if label_frac[k] > 0]
    wedge_colors = [color_for(k) for k in range(K) if label_frac[k] > 0]
    ax10.pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
             autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    ax10.set_title('Skill Label Distribution', fontsize=10)

    # Row 1: per-timestep label fraction across all episodes
    ax11 = fig.add_subplot(gs[1, 1])
    frac_over_t = np.zeros((T, K))
    for k in range(K):
        frac_over_t[:, k] = (labels_hard == k).mean(axis=0)
    t_ax = np.arange(T)
    bottom = np.zeros(T)
    for k in range(K):
        ax11.fill_between(t_ax, bottom, bottom + frac_over_t[:, k],
                          color=color_for(k), alpha=0.8,
                          label=f'Skill {k}')
        bottom += frac_over_t[:, k]
    ax11.set_xlabel('Timestep t'); ax11.set_ylabel('Fraction')
    ax11.set_title('Skill Label Fraction over Time\n(stacked, all episodes)', fontsize=10)
    ax11.legend(fontsize=7, loc='upper right')

    # Row 1: z_all mean norm over time
    ax12 = fig.add_subplot(gs[1, 2])
    z_norm_t = np.linalg.norm(z_all, axis=-1).mean(axis=0)  # (T,)
    ax12.plot(t_ax, z_norm_t, lw=2, color='#2196F3')
    ax12.set_xlabel('Timestep t'); ax12.set_ylabel('||z||')
    ax12.set_title('Mean ||z_t|| over Time', fontsize=10)
    ax12.grid(alpha=0.3)

    # Row 2: per-episode skill trajectory heatmap
    ax20 = fig.add_subplot(gs[2, :])
    ep_labels = labels_hard[ep_indices, :]  # (n_ep_show, T)
    im = ax20.imshow(ep_labels, aspect='auto', interpolation='nearest',
                     cmap=plt.cm.get_cmap('tab10', K),
                     vmin=-0.5, vmax=K-0.5)
    plt.colorbar(im, ax=ax20, ticks=range(K),
                 label='Skill index')
    ax20.set_yticks(range(n_ep_show))
    ax20.set_yticklabels([f'Ep {ep_indices[i]}' for i in range(n_ep_show)],
                          fontsize=8)
    ax20.set_xlabel('Timestep t')
    ax20.set_title(f'Skill Label Trajectory per Episode (K={K})', fontsize=10)

    fig.suptitle(
        f'Skill Pretraining Visualization  |  K={K}  N={N}  T={T}  d_z={d_z}',
        fontsize=13, fontweight='bold',
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # ── Summary stats ─────────────────────────────────────────
    print(f"\nSkill statistics:")
    for k in range(K):
        n_k = label_counts[k]
        print(f"  Skill {k}: {n_k:6d} steps ({label_frac[k]*100:.1f}%)")


if __name__ == '__main__':
    main()