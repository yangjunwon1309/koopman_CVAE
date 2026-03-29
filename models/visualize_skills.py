"""
visualize_skills.py
====================
Skill pretraining 결과 시각화.

Usage:
    python visualize_skills.py \
        --npz checkpoints/skill_pretrain/labels.npz \
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

PALETTE = [
    '#E53935', '#1E88E5', '#43A047', '#FB8C00',
    '#8E24AA', '#00ACC1', '#FFB300', '#6D4C41',
    '#546E7A', '#D81B60', '#00897B', '#F4511E',
]
def color_for(k): return PALETTE[k % len(PALETTE)]


def run_tsne(Z, perplexity=30):
    print(f"  t-SNE: {Z.shape[0]} pts, d={Z.shape[1]}")
    if Z.shape[1] > 50:
        Z = PCA(n_components=50).fit_transform(Z)
    return TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000).fit_transform(Z)


def scatter_by_skill(Z2d, labels, K, ax, title):
    for k in range(K):
        m = labels == k
        if m.sum() == 0: continue
        ax.scatter(Z2d[m,0], Z2d[m,1], c=color_for(k), alpha=0.5,
                   s=10, label=f'Skill {k} (n={m.sum()})')
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7, markerscale=2); ax.axis('off')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',  default='checkpoints/skill_pretrain/labels.npz')
    p.add_argument('--out',  default='checkpoints/skill_pretrain/tsne.png')
    p.add_argument('--n',    type=int, default=5000, help='max pts for t-SNE')
    p.add_argument('--perp', type=float, default=30)
    args = p.parse_args()
    npz = os.path.expanduser(args.npz)
    if not os.path.exists(npz):
        print(f"Not found: {npz}"); return

    data = np.load(npz, allow_pickle=True)
    labels = data['labels_hard']   # (N,) or (N,T)
    Z_all  = data['z_all']         # (N, d_z) or (N, T, d_z)
    K      = int(data['K'][0])
    L      = int(data['skill_horizon'][0]) if 'skill_horizon' in data else 10

    # Flatten to (N, d_z) and (N,)
    if Z_all.ndim == 3:
        N, T, d_z = Z_all.shape
        Z = Z_all[:, -1, :]
        Y = labels[:, -1] if labels.ndim == 2 else labels
    else:
        Z, Y = Z_all, labels
        N, d_z = Z.shape

    print(f"N={N}  d_z={d_z}  K={K}  skill_horizon={L}")
    counts = {int(k): int((Y==k).sum()) for k in np.unique(Y)}
    print(f"counts: {counts}")

    # Subsample
    if N > args.n:
        idx = np.random.choice(N, args.n, replace=False)
        Zs, Ys = Z[idx], Y[idx]
    else:
        Zs, Ys = Z, Y

    # t-SNE
    Z2d  = run_tsne(Zs, args.perp)
    Zpca = PCA(n_components=2).fit_transform(Zs)
    var  = PCA(n_components=2).fit(Zs).explained_variance_ratio_

    # Plot
    fig = plt.figure(figsize=(16, 12))
    gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # t-SNE
    scatter_by_skill(Z2d, Ys, K, fig.add_subplot(gs[0,0]),
                     f't-SNE (n={len(Zs)}, perp={args.perp})')
    # PCA
    scatter_by_skill(Zpca, Ys, K, fig.add_subplot(gs[0,1]),
                     f'PCA (var={var[0]:.2f}+{var[1]:.2f}={sum(var):.2f})')

    # Count bar
    ax = fig.add_subplot(gs[0,2])
    for k in range(K):
        ax.bar(k, counts.get(k,0), color=color_for(k), alpha=0.8)
    ax.set_xticks(range(K)); ax.set_xlabel('Skill k')
    ax.set_title(f'Sample count per skill  (total={N})'); ax.grid(alpha=0.3)

    # ||z|| per skill
    ax = fig.add_subplot(gs[1,0])
    norms = np.linalg.norm(Z, axis=-1)
    for k in range(K):
        m = Y == k
        if m.sum() > 0:
            ax.hist(norms[m], bins=30, alpha=0.6,
                    color=color_for(k), label=f'Sk{k}', density=True)
    ax.set_xlabel('||z||'); ax.set_title('||z|| distribution by skill')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Skill labels over sequence index
    ax = fig.add_subplot(gs[1,1])
    show = min(1000, N)
    ax.scatter(range(show), Y[:show],
               c=[color_for(int(l)) for l in Y[:show]], s=8, alpha=0.7)
    ax.set_xlabel('Sequence index'); ax.set_ylabel('Skill')
    ax.set_yticks(range(K)); ax.grid(alpha=0.3)
    ax.set_title(f'Skill sequence (first {show})')

    # Pie
    ax = fig.add_subplot(gs[1,2])
    vals  = [counts.get(k,0) for k in range(K) if counts.get(k,0)>0]
    lbls  = [f'Sk{k}\n{counts.get(k,0)/N*100:.1f}%'
             for k in range(K) if counts.get(k,0)>0]
    cols  = [color_for(k) for k in range(K) if counts.get(k,0)>0]
    ax.pie(vals, labels=lbls, colors=cols, startangle=90,
           textprops={'fontsize':8})
    ax.set_title('Skill distribution')

    fig.suptitle(f'Skill Pretraining  K={K}  N={N}  d_z={d_z}  L={L}',
                 fontsize=12, fontweight='bold')
    out = os.path.expanduser(args.out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()