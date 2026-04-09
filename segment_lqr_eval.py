"""
segment_lqr_eval.py — Segment-level LQR Evaluation
=====================================================

Pipeline:
  1. Load complete/mixed episodes (reward=max, task fully solved)
  2. Cut each episode into H-step segments
     G_k = x_{H*k}  (segment end = sub-goal)
  3. For each segment k:
       given: (0 : H*k-1)  real context
       LQR:   (H*(k-1) : H*k)  predict with z*=encode(G_k)
       real:  (H*(k-1) : H*k)  ground truth
  4. Metrics per segment: RMSE_Δq, RMSE_Δp, LQR cost
  5. Visualization: per-segment metrics + GIF of LQR decoded actions

Usage:
    python segment_lqr_eval.py \
        --ckpt  checkpoints/kodaq_v3/final.pt \
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \
        --quality complete \
        --H 16 \
        --n_ep 5 \
        --out_dir checkpoints/kodaq/segment_eval \
        --device cuda
"""

import os, sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models.koopman_cvae import KoopmanCVAE
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_planner import (
    KODAQLQRPlanner, LQRConfig,
    load_kitchen_episodes,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
)


# ──────────────────────────────────────────────────────────────────────────────
# Action decoder (batch, gradient-enabled)
# ──────────────────────────────────────────────────────────────────────────────

@torch.enable_grad()
def decode_action_batch(model, u_batch: torch.Tensor,
                        n_steps: int = 50, lr: float = 0.05) -> torch.Tensor:
    N, d_u = u_batch.shape
    da  = model.cfg.action_dim
    dev = u_batch.device
    a   = torch.zeros(N, da, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        F.mse_loss(model.action_encoder(a), u_batch.detach()).backward()
        opt.step()
        with torch.no_grad(): a.clamp_(-1., 1.)
    return a.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Segment-level LQR evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_segments(
    planner:     KODAQLQRPlanner,
    episode:     Dict,
    x_seq_full:  np.ndarray,
    H:           int   = 16,
    device:      str   = 'cuda',
) -> List[Dict]:
    """
    에피소드를 H-step segments로 분할해서 각 segment LQR 평가.

    Segment k (k=1,...,L//H):
      context: x[0 : H*k-1]  (given)
      goal:    x[H*k]        = G_k  (segment end)
      LQR:     predict x[H*(k-1) : H*k]  with horizon=H
      real:    x[H*(k-1) : H*k]

    Returns list of per-segment result dicts.
    """
    dev   = torch.device(device)
    model = planner.model
    s_t   = episode['start_t']
    L     = episode['length']
    x_ep  = x_seq_full[s_t : s_t + L]    # (L, 2108)
    acts  = episode['actions']            # (L, 9)

    n_seg = L // H
    if n_seg < 1:
        return []

    results = []
    for k in range(1, n_seg + 1):
        ctx_end  = H * k - 1          # context end (exclusive of goal)
        seg_start = H * (k - 1)       # segment start
        seg_end   = H * k             # segment end (goal timestep)

        if seg_end >= L:
            break

        # ── Context ─────────────────────────────────────────────────────────
        ctx_start = max(0, ctx_end - 64)   # at most 64 context steps
        x_cond = torch.FloatTensor(x_ep[ctx_start:ctx_end]).unsqueeze(0).to(dev)
        a_cond = torch.FloatTensor(acts[ctx_start:ctx_end]).unsqueeze(0).to(dev)

        if x_cond.shape[1] < 1:
            continue

        # ── Sub-goal: G_k = x[seg_end] ──────────────────────────────────────
        x_goal = torch.FloatTensor(x_ep[seg_end]).unsqueeze(0).to(dev)

        # ── LQR plan (horizon = H) ───────────────────────────────────────────
        with torch.no_grad():
            plan = planner.plan(
                x_cond, a_cond, x_goal,
                horizon=H,
                compute_uncertainty=False,
            )

        # ── Ground truth segment ─────────────────────────────────────────────
        true_x = x_ep[seg_start:seg_end]   # (H, 2108)

        # ── Metrics ──────────────────────────────────────────────────────────
        pred_x  = plan['x_traj'].cpu().numpy()        # (H+1, 2108)
        H_comp  = min(len(pred_x), len(true_x))

        rmse_dq = float(np.sqrt((
            (pred_x[:H_comp, X_DQ_START:X_DQ_END] -
             true_x[:H_comp, X_DQ_START:X_DQ_END])**2
        ).mean()))

        rmse_dp = float(np.sqrt((
            (pred_x[:H_comp, X_DP_START:X_DP_END] -
             true_x[:H_comp, X_DP_START:X_DP_END])**2
        ).mean()))

        # Decoded actions (H, 9)
        u_traj  = plan['u_traj'].to(dev)   # (H, d_u)
        a_lqr   = decode_action_batch(model, u_traj).cpu().numpy()

        results.append({
            'k':           k,
            'seg_start':   seg_start,
            'seg_end':     seg_end,
            'ctx_start':   ctx_start,
            'rmse_dq':     rmse_dq,
            'rmse_dp':     rmse_dp,
            'lqr_cost':    plan['total_cost'],
            'pred_x':      pred_x,       # (H+1, 2108)
            'true_x':      true_x,       # (H, 2108)
            'a_lqr':       a_lqr,        # (H, 9)
            'a_real':      acts[seg_start:seg_end],   # (H, 9)
            'u_traj':      u_traj.cpu().numpy(),
            'o_traj':      plan['o_traj'].cpu().numpy(),
        })

        print(f"  Seg {k:3d}  [{seg_start:4d}:{seg_end:4d}]"
              f"  RMSE_Δq={rmse_dq:.4f}  RMSE_Δp={rmse_dp:.4f}"
              f"  cost={plan['total_cost']:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualization: per-segment metrics
# ──────────────────────────────────────────────────────────────────────────────

def visualize_segment_metrics(
    all_results:  List[List[Dict]],   # [ep0_results, ep1_results, ...]
    ep_labels:    List[str],
    H:            int,
    out_path:     str,
):
    """
    Per-segment RMSE_Δq, RMSE_Δp, LQR cost across episodes.
    """
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for ep_i, (ep_res, label) in enumerate(zip(all_results, ep_labels)):
        if not ep_res: continue
        ks      = [r['k']        for r in ep_res]
        rmse_dq = [r['rmse_dq']  for r in ep_res]
        rmse_dp = [r['rmse_dp']  for r in ep_res]
        costs   = [r['lqr_cost'] for r in ep_res]
        col     = PAL[ep_i % len(PAL)]

        axes[0].plot(ks, rmse_dq, 'o-', color=col, lw=1.5,
                     ms=4, alpha=0.8, label=label)
        axes[1].plot(ks, rmse_dp, 'o-', color=col, lw=1.5,
                     ms=4, alpha=0.8, label=label)
        axes[2].plot(ks, costs,   'o-', color=col, lw=1.5,
                     ms=4, alpha=0.8, label=label)

    # Summary stats across all segments
    all_dq = [r['rmse_dq'] for res in all_results for r in res]
    all_dp = [r['rmse_dp'] for res in all_results for r in res]
    all_c  = [r['lqr_cost'] for res in all_results for r in res]

    for ax, vals, ylabel, title in [
        (axes[0], all_dq, 'RMSE Δq_t', f'RMSE Δq_t per segment  (H={H})'),
        (axes[1], all_dp, 'RMSE Δp_t', f'RMSE Δp_t per segment  (H={H})'),
        (axes[2], all_c,  'LQR cost',  f'LQR cost per segment   (H={H})'),
    ]:
        ax.axhline(np.mean(vals), color='k', ls='--', lw=1.2,
                   label=f'mean={np.mean(vals):.4f}')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, ncol=4)
        ax.spines[['top','right']].set_visible(False)

    axes[2].set_xlabel('Segment index k', fontsize=9)
    fig.suptitle(
        f'Segment LQR Evaluation  H={H}\n'
        f'mean RMSE_Δq={np.mean(all_dq):.4f}  '
        f'mean RMSE_Δp={np.mean(all_dp):.4f}  '
        f'mean cost={np.mean(all_c):.3f}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def visualize_episode_segments(
    ep_results: List[Dict],
    ep_idx:     int,
    H:          int,
    out_path:   str,
    n_show:     int = 6,
):
    """
    한 에피소드의 각 segment에서 pred vs true Δq, Δp 비교.
    최대 n_show segments.
    """
    segs = ep_results[:n_show]
    n    = len(segs)
    if n == 0: return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)
    PAL = ['#1E88E5', '#E53935']

    for i, seg in enumerate(segs):
        H_c    = min(len(seg['pred_x']), len(seg['true_x']))
        ts     = np.arange(H_c)

        # Δq
        ax = axes[i, 0]
        ax.plot(ts, seg['true_x'][:H_c, X_DQ_START:X_DQ_END].mean(1),
                color=PAL[0], lw=1.5, label='true')
        ax.plot(ts, seg['pred_x'][:H_c, X_DQ_START:X_DQ_END].mean(1),
                color=PAL[1], lw=1.5, ls='--', label='LQR pred')
        ax.set_title(f"Seg {seg['k']} [{seg['seg_start']}:{seg['seg_end']}]"
                     f"  Δq RMSE={seg['rmse_dq']:.4f}", fontsize=8)
        ax.set_xlabel('step'); ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

        # Δp
        ax = axes[i, 1]
        ax.plot(ts, seg['true_x'][:H_c, X_DP_START:X_DP_END].mean(1),
                color=PAL[0], lw=1.5, label='true')
        ax.plot(ts, seg['pred_x'][:H_c, X_DP_START:X_DP_END].mean(1),
                color=PAL[1], lw=1.5, ls='--', label='LQR pred')
        ax.set_title(f"Seg {seg['k']}  Δp RMSE={seg['rmse_dp']:.4f}", fontsize=8)
        ax.set_xlabel('step'); ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle(f'Ep {ep_idx}  Segment-level LQR pred vs true  (H={H})',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# GIF: LQR decoded action simulation
# ──────────────────────────────────────────────────────────────────────────────

def save_gif(frames: List[np.ndarray], path: str, fps: int = 12):
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)
        print(f"  GIF: {path}  ({len(frames)} frames @ {fps}fps)")
    except ImportError:
        # fallback: save frame strip
        n = min(len(frames), 16)
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        step = max(1, len(frames) // n)
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[i * step])
            ax.axis('off')
        plt.tight_layout()
        strip = path.replace('.gif', '_strip.png')
        plt.savefig(strip, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  Strip: {strip}")


def render_frame(sim) -> np.ndarray:
    f = sim.render(512, 512, camera_id=-1)
    return f[192:320, 192:320]   # 128x128 crop


def make_segment_gif(
    episode:     Dict,
    ep_results:  List[Dict],
    env_name:    str,
    out_dir:     str,
    fps:         int  = 12,
    seg_indices: Optional[List[int]] = None,   # None = all
):
    """
    각 segment의 LQR decoded action sequence를 시뮬레이션해서 GIF로 저장.

    각 segment k마다:
      - t=0 : H*(k-1) 까지 real action으로 replay (context 재현)
      - t=H*(k-1) : H*k 까지 LQR decoded action 실행
      → GIF 저장
    """
    import gym, d4rl
    acts_ep = episode['actions']
    ep_s    = episode['start_t']

    segs_to_render = ep_results if seg_indices is None \
                     else [ep_results[i] for i in seg_indices if i < len(ep_results)]

    env = gym.make(env_name)
    sim = env.unwrapped.sim

    for seg in segs_to_render:
        k         = seg['k']
        seg_start = seg['seg_start']
        a_lqr     = seg['a_lqr']   # (H, 9)

        print(f"  Rendering seg {k} [{seg_start}:{seg['seg_end']}] ...")

        env.reset()
        # replay context
        for t in range(seg_start):
            env.step(acts_ep[t])

        # LQR action rollout → collect frames
        frames = [render_frame(sim)]
        for a in a_lqr:
            env.step(a)
            frames.append(render_frame(sim))

        label   = f"ep{ep_s}_seg{k:03d}_lqr"
        gif_path = f"{out_dir}/{label}.gif"
        save_gif(frames, gif_path, fps=fps)

    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: List[List[Dict]], H: int):
    all_segs = [r for res in all_results for r in res]
    if not all_segs:
        print("No segments.")
        return

    dq = np.array([r['rmse_dq']  for r in all_segs])
    dp = np.array([r['rmse_dp']  for r in all_segs])
    c  = np.array([r['lqr_cost'] for r in all_segs])

    print(f"\n{'='*55}")
    print(f"Segment LQR Evaluation  H={H}  N_segs={len(all_segs)}")
    print(f"{'='*55}")
    print(f"  RMSE Δq: {dq.mean():.4f} ± {dq.std():.4f}"
          f"  [min={dq.min():.4f} max={dq.max():.4f}]")
    print(f"  RMSE Δp: {dp.mean():.4f} ± {dp.std():.4f}"
          f"  [min={dp.min():.4f} max={dp.max():.4f}]")
    print(f"  LQR cost:{c.mean():.3f} ± {c.std():.3f}")

    # Best / worst segments
    best  = all_segs[dq.argmin()]
    worst = all_segs[dq.argmax()]
    print(f"\n  Best  seg: k={best['k']}  RMSE_Δq={best['rmse_dq']:.4f}")
    print(f"  Worst seg: k={worst['k']}  RMSE_Δq={worst['rmse_dq']:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      default='checkpoints/kodaq_v3/final.pt')
    p.add_argument('--x_cache',   default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--u_bounds',  default=None)
    p.add_argument('--env',       default='kitchen-mixed-v0')
    p.add_argument('--quality',   default='complete',
                   choices=['mixed','partial','complete'],
                   help='complete: fully solved episodes (recommended)')
    p.add_argument('--H',         type=int,   default=16,
                   help='Segment length in timesteps')
    p.add_argument('--n_ep',      type=int,   default=5)
    p.add_argument('--Q_scale',   type=float, default=1.0)
    p.add_argument('--R_scale',   type=float, default=10.0)
    p.add_argument('--lambda_unc',type=float, default=0.1)
    p.add_argument('--fps',       type=int,   default=12)
    p.add_argument('--gif_ep',    type=int,   default=0,
                   help='Episode index to render GIFs for')
    p.add_argument('--gif_segs',  type=str,   default=None,
                   help='Comma-separated segment indices to render (None=all)')
    p.add_argument('--skip_gif',  action='store_true')
    p.add_argument('--out_dir',   default='checkpoints/kodaq/segment_eval')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt  = torch.load(args.ckpt, map_location=args.device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(args.device)
    print(f"Model: K={model.cfg.num_skills}  m={model.cfg.koopman_dim}")

    lqr_cfg = LQRConfig(Q_scale=args.Q_scale, R_scale=args.R_scale,
                        lambda_unc=args.lambda_unc)
    planner = KODAQLQRPlanner(model, lqr_cfg)
    if args.u_bounds and Path(args.u_bounds).exists():
        planner.load_u_bounds(args.u_bounds)

    # ── Data ──────────────────────────────────────────────────────────────────
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality,
                                             min_len=args.H * 2)
    # prefer episodes with max reward (fully solved)
    episodes = sorted(episodes, key=lambda e: -e['goal_info']['reward_total'])
    episodes = episodes[:args.n_ep]
    print(f"Using {len(episodes)} episodes  "
          f"(reward range: {episodes[-1]['goal_info']['reward_total']:.0f}"
          f"~{episodes[0]['goal_info']['reward_total']:.0f})")

    # ── Segment evaluation ────────────────────────────────────────────────────
    all_results = []
    ep_labels   = []

    for ep_i, ep in enumerate(episodes):
        print(f"\nEp {ep_i}  len={ep['length']}  tasks={ep['tasks']}")
        res = evaluate_segments(planner, ep, x_seq_full,
                                H=args.H, device=args.device)
        all_results.append(res)
        ep_labels.append(f"Ep{ep_i}(r={ep['goal_info']['reward_total']:.0f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(all_results, args.H)

    # ── Visualization ─────────────────────────────────────────────────────────
    # 1. Cross-episode metrics
    visualize_segment_metrics(
        all_results, ep_labels, args.H,
        out_path=f"{args.out_dir}/segment_metrics_H{args.H}.png",
    )

    # 2. Per-episode pred vs true
    for ep_i, (ep_res, ep) in enumerate(zip(all_results, episodes)):
        if not ep_res: continue
        visualize_episode_segments(
            ep_res, ep_i, args.H,
            out_path=f"{args.out_dir}/ep{ep_i}_segments_H{args.H}.png",
        )

    # ── GIF ───────────────────────────────────────────────────────────────────
    if not args.skip_gif and args.gif_ep < len(episodes):
        ep    = episodes[args.gif_ep]
        res   = all_results[args.gif_ep]
        segs  = ([int(s) for s in args.gif_segs.split(',')]
                 if args.gif_segs else None)
        print(f"\nRendering GIFs: ep={args.gif_ep}  segs={segs or 'all'}")
        make_segment_gif(
            episode=ep,
            ep_results=res,
            env_name=args.env,
            out_dir=args.out_dir,
            fps=args.fps,
            seg_indices=segs,
        )

    print(f"\nDone. → {args.out_dir}/")


if __name__ == '__main__':
    main()