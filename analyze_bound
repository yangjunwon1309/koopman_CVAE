"""
analyze_u_bounds.py — Action Encoder Range Visualization
=========================================================
survey_action_encoder_range() 결과 분석:

1. Per-dim range bar chart  (u_min, u_max, width per dimension)
2. Distribution histogram   (u_min / u_max across dims)
3. Symmetry analysis        (|u_min| vs u_max scatter)
4. Constraint tightness     (현재 u_max=1.0 scalar clip 대비 per-dim bounds)

Usage:
    python analyze_u_bounds.py \
        --bounds checkpoints/kodaq/lqr/u_bounds.npz \
        --ckpt   checkpoints/kodaq_v3/final.pt \
        --out    checkpoints/kodaq/lqr/u_bounds_analysis.png
"""

import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def analyze_u_bounds(bounds_path: str, ckpt_path: str = None,
                     out_path: str = 'u_bounds_analysis.png',
                     scalar_u_max: float = 1.0):

    # ── Load bounds ───────────────────────────────────────────────────────────
    data  = np.load(bounds_path)
    u_min = data['u_min']   # (d_u,)
    u_max = data['u_max']   # (d_u,)
    d_u   = len(u_min)
    width = u_max - u_min   # per-dim range width
    dims  = np.arange(d_u)

    print(f"=== U-bounds Analysis  d_u={d_u} ===")
    print(f"  u_min : mean={u_min.mean():.4f}  std={u_min.std():.4f}"
          f"  min={u_min.min():.4f}  max={u_min.max():.4f}")
    print(f"  u_max : mean={u_max.mean():.4f}  std={u_max.std():.4f}"
          f"  min={u_max.min():.4f}  max={u_max.max():.4f}")
    print(f"  width : mean={width.mean():.4f}  std={width.std():.4f}"
          f"  min={width.min():.4f}  max={width.max():.4f}")

    # Symmetry: how asymmetric is ψ(a)?
    asym  = np.abs(np.abs(u_min) - u_max)
    print(f"  asymmetry |u_min|-|u_max|: mean={asym.mean():.4f}  max={asym.max():.4f}")

    # Tightness vs scalar clip
    n_tighter_min = (np.abs(u_min) < scalar_u_max).sum()
    n_tighter_max = (u_max < scalar_u_max).sum()
    print(f"  dims tighter than scalar clip ±{scalar_u_max}:")
    print(f"    u_min side: {n_tighter_min}/{d_u} dims")
    print(f"    u_max side: {n_tighter_max}/{d_u} dims")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    BLUE  = '#1E88E5'
    RED   = '#E53935'
    GREEN = '#43A047'
    GRAY  = '#9E9E9E'

    # ── 1. Per-dim range bar ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(dims, u_max,  color=BLUE,  alpha=0.7, label='u_max',  width=0.8)
    ax1.bar(dims, u_min,  color=RED,   alpha=0.7, label='u_min',  width=0.8)
    ax1.axhline(scalar_u_max,  color=BLUE,  ls='--', lw=1.2, alpha=0.5,
                label=f'scalar clip +{scalar_u_max}')
    ax1.axhline(-scalar_u_max, color=RED,   ls='--', lw=1.2, alpha=0.5,
                label=f'scalar clip -{scalar_u_max}')
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_xlabel('Latent dimension $u$ index', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title(f'Per-dim u bounds  (d_u={d_u})', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(-1, d_u)
    ax1.spines[['top','right']].set_visible(False)

    # ── 2. Range width per dim ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.barh(dims, width, color=GREEN, alpha=0.7)
    ax2.axvline(2 * scalar_u_max, color=GRAY, ls='--', lw=1.2,
                label=f'scalar width={2*scalar_u_max}')
    ax2.set_xlabel('Range width (u_max - u_min)', fontsize=9)
    ax2.set_ylabel('Latent dim', fontsize=9)
    ax2.set_title('Range width per dim', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.spines[['top','right']].set_visible(False)

    # ── 3. Distribution of u_min, u_max ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bins = 20
    ax3.hist(u_min, bins=bins, color=RED,  alpha=0.6, label='u_min', density=True)
    ax3.hist(u_max, bins=bins, color=BLUE, alpha=0.6, label='u_max', density=True)
    ax3.axvline(-scalar_u_max, color=RED,  ls='--', lw=1.2)
    ax3.axvline( scalar_u_max, color=BLUE, ls='--', lw=1.2)
    ax3.set_xlabel('Value', fontsize=9)
    ax3.set_ylabel('Density', fontsize=9)
    ax3.set_title('Distribution of u_min / u_max', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.spines[['top','right']].set_visible(False)

    # ── 4. Symmetry: |u_min| vs u_max scatter ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(np.abs(u_min), u_max, c=width, cmap='viridis',
                     s=30, alpha=0.8, zorder=3)
    # Perfect symmetry line
    lim = max(np.abs(u_min).max(), u_max.max()) * 1.05
    ax4.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='perfect symmetry')
    ax4.set_xlabel('|u_min|', fontsize=9)
    ax4.set_ylabel('u_max', fontsize=9)
    ax4.set_title('Symmetry: |u_min| vs u_max\n(diagonal = symmetric)',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    plt.colorbar(sc, ax=ax4, label='range width')
    ax4.spines[['top','right']].set_visible(False)

    # ── 5. Constraint tightness vs scalar clip ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    # How much tighter is per-dim vs scalar?
    slack_min = scalar_u_max - np.abs(u_min)   # positive = per-dim tighter
    slack_max = scalar_u_max - u_max
    ax5.plot(dims, slack_min, color=RED,  lw=1.2, alpha=0.8, label='slack on min side')
    ax5.plot(dims, slack_max, color=BLUE, lw=1.2, alpha=0.8, label='slack on max side')
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.fill_between(dims, 0, np.minimum(slack_min, slack_max),
                     alpha=0.15, color=GREEN, label='both sides tighter')
    ax5.set_xlabel('Latent dim', fontsize=9)
    ax5.set_ylabel(f'Slack vs ±{scalar_u_max}', fontsize=9)
    ax5.set_title('Per-dim tightness\n(+: tighter than scalar clip)',
                  fontsize=10, fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.spines[['top','right']].set_visible(False)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Action Encoder ψ(a) Range Analysis\n'
        f'u_min ∈ [{u_min.min():.3f}, {u_min.max():.3f}]  '
        f'u_max ∈ [{u_max.min():.3f}, {u_max.max():.3f}]  '
        f'mean_width={width.mean():.3f}',
        fontsize=12, fontweight='bold'
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")

    # ── Per-dim summary table (top 10 widest / narrowest) ────────────────────
    idx_wide   = np.argsort(width)[::-1][:10]
    idx_narrow = np.argsort(width)[:10]
    print("\nTop 10 widest dims:")
    for i in idx_wide:
        print(f"  dim {i:3d}: [{u_min[i]:+.4f}, {u_max[i]:+.4f}]  width={width[i]:.4f}")
    print("\nTop 10 narrowest dims:")
    for i in idx_narrow:
        print(f"  dim {i:3d}: [{u_min[i]:+.4f}, {u_max[i]:+.4f}]  width={width[i]:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bounds',       default='checkpoints/kodaq/lqr/u_bounds.npz')
    p.add_argument('--ckpt',         default=None)
    p.add_argument('--out',          default='checkpoints/kodaq/lqr/u_bounds_analysis.png')
    p.add_argument('--scalar_u_max', type=float, default=1.0)
    args = p.parse_args()

    analyze_u_bounds(
        bounds_path=args.bounds,
        ckpt_path=args.ckpt,
        out_path=args.out,
        scalar_u_max=args.scalar_u_max,
    )


if __name__ == '__main__':
    main()