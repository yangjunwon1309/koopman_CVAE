"""
validate_lqr.py — LQR Validation: Action Decoder Error + Sim Rollout + GIF
===========================================================================

1. Action Decoder Error Analysis
   ||ψ(ψ⁻¹(u_t)) - u_t||  over all LQR u_t vectors
   → determines if decoded actions faithfully represent LQR intent

2. Simulation Rollout + GIF
   For one episode, compare:
     A. Real actions (ground truth)
     B. LQR decoded actions (cond_len given, then LQR)
   Each subtask stage gets its own GIF:
     - A_real:  actual demonstration frames
     - B_lqr:   decoded action simulation frames

Usage:
    python validate_lqr.py \
        --ckpt  checkpoints/kodaq_v3/final.pt \
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \
        --u_bounds checkpoints/kodaq/lqr/u_bounds.npz \
        --env   kitchen-mixed-v0 \
        --ep_idx 0 \
        --cond_len 32 \
        --out_dir checkpoints/kodaq/validation \
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
from models.losses import symlog, symexp
from data.extract_skill_label import load_x_sequences, obs_to_x_goal
from lqr_planner import (
    KODAQLQRPlanner, LQRConfig,
    load_kitchen_episodes, detect_completed_tasks_by_reward,
    OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS,
    X_DQ_START, X_DQ_END,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Action Decoder Error Analysis
# ──────────────────────────────────────────────────────────────────────────────

@torch.enable_grad()
def decode_action_batch(model, u_batch: torch.Tensor,
                        n_steps: int = 50, lr: float = 0.05) -> torch.Tensor:
    """
    Batch ψ⁻¹(u): argmin_a ||ψ(a) - u||²  for each u in u_batch.
    u_batch: (N, d_u)
    Returns a_batch: (N, da)
    """
    N  = u_batch.shape[0]
    da = model.cfg.action_dim
    dev = u_batch.device

    a = torch.zeros(N, da, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        u_pred = model.action_encoder(a)
        loss   = F.mse_loss(u_pred, u_batch.detach())
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-1., 1.)

    return a.detach()


def analyze_decoder_error(
    planner: KODAQLQRPlanner,
    results: List[Dict],
    out_dir: str = 'checkpoints/kodaq/validation',
) -> Dict:
    """
    1. Collect all u_t* from LQR results
    2. Decode: a_t = ψ⁻¹(u_t*)
    3. Re-encode: u_recon = ψ(a_t)
    4. Measure ||u_recon - u_t*||

    Returns stats dict + saves plot.
    """
    model  = planner.model
    device = planner.device
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Collect all u_t from rollouts
    all_u = []
    for res in results:
        for stage in res.get('stages', []):
            plan = stage['plan']
            if plan.get('u_traj') is not None:
                all_u.append(plan['u_traj'].to(device))   # (H, d_u)

    if not all_u:
        print("No u_traj found in results.")
        return {}

    u_all = torch.cat(all_u, dim=0)   # (N_total, d_u)
    N     = len(u_all)
    print(f"\n=== Action Decoder Error Analysis ===")
    print(f"  Total u_t vectors: {N}")

    # Batch decode in chunks
    chunk = 256
    a_decoded_list = []
    for i in range(0, N, chunk):
        a_chunk = decode_action_batch(model, u_all[i:i+chunk])
        a_decoded_list.append(a_chunk)
    a_decoded = torch.cat(a_decoded_list, dim=0)   # (N, da)

    # Re-encode
    with torch.no_grad():
        u_recon = model.action_encoder(a_decoded)   # (N, d_u)

    # Error
    err     = (u_recon - u_all).abs()              # (N, d_u)
    err_np  = err.cpu().numpy()
    u_np    = u_all.cpu().numpy()
    a_np    = a_decoded.cpu().numpy()

    per_vec  = err_np.mean(axis=1)   # (N,)  mean over d_u
    per_dim  = err_np.mean(axis=0)   # (d_u,) mean over N

    rel_err  = (err_np / (np.abs(u_np) + 1e-6)).mean(axis=1)  # relative

    print(f"  ||u_recon - u*|| per vector: mean={per_vec.mean():.5f}  "
          f"std={per_vec.std():.5f}  max={per_vec.max():.5f}")
    print(f"  Relative error:              mean={rel_err.mean():.4f}  "
          f"max={rel_err.max():.4f}")
    print(f"  a_decoded range: [{a_np.min():.4f}, {a_np.max():.4f}]  "
          f"(should be within [-1,1])")
    print(f"  Dims at boundary (|a|>0.99): "
          f"{(np.abs(a_np) > 0.99).sum()} / {a_np.size}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    PAL = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    # 1. Per-vector error distribution
    ax = axes[0, 0]
    ax.hist(per_vec, bins=40, color=PAL[0], alpha=0.7, density=True)
    ax.axvline(per_vec.mean(), color='k', ls='--', lw=1.5,
               label=f'mean={per_vec.mean():.4f}')
    ax.set_xlabel('Mean |u_recon - u*| per vector', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Decoder error distribution\n(per u_t vector)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 2. Per-dim error
    ax = axes[0, 1]
    ax.bar(np.arange(len(per_dim)), per_dim, color=PAL[1], alpha=0.7, width=0.8)
    ax.axhline(per_dim.mean(), color='k', ls='--', lw=1.2,
               label=f'mean={per_dim.mean():.4f}')
    ax.set_xlabel('Latent dim', fontsize=9)
    ax.set_ylabel('Mean |error|', fontsize=9)
    ax.set_title('Per-dim decoder error', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 3. u* vs u_recon scatter (first 2 dims)
    ax = axes[1, 0]
    ax.scatter(u_np[:200, 0], u_recon.cpu().numpy()[:200, 0],
               s=10, alpha=0.5, color=PAL[2], label='dim 0')
    ax.scatter(u_np[:200, 1], u_recon.cpu().numpy()[:200, 1],
               s=10, alpha=0.5, color=PAL[3], label='dim 1')
    lim = max(np.abs(u_np[:, :2]).max(), np.abs(u_recon.cpu().numpy()[:, :2]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, alpha=0.5)
    ax.set_xlabel('u* (LQR)', fontsize=9)
    ax.set_ylabel('ψ(ψ⁻¹(u*))', fontsize=9)
    ax.set_title('u* vs reconstructed u\n(first 2 dims, 200 samples)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 4. Decoded action distribution
    ax = axes[1, 1]
    ax.hist(a_np.flatten(), bins=40, color=PAL[0], alpha=0.7, density=True)
    ax.axvline(-1, color='k', ls='--', lw=1.0, alpha=0.6)
    ax.axvline( 1, color='k', ls='--', lw=1.0, alpha=0.6, label='bounds ±1')
    ax.set_xlabel('Decoded action value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Decoded action distribution\n(all dims, all steps)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    fig.suptitle(
        f'Action Decoder Error  N={N}  '
        f'mean_err={per_vec.mean():.4f}  rel_err={rel_err.mean():.4f}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    out = f"{out_dir}/decoder_error.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    return {
        'per_vec_mean': float(per_vec.mean()),
        'per_vec_max':  float(per_vec.max()),
        'rel_err_mean': float(rel_err.mean()),
        'a_decoded':    a_np,
        'u_all':        u_np,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Simulation Rollout + GIF
# ──────────────────────────────────────────────────────────────────────────────

RENDER_W, RENDER_H = 512, 512
CROP_S, CROP_E     = 192, 320   # 128x128 center crop


def render_frame(sim) -> np.ndarray:
    frame = sim.render(RENDER_W, RENDER_H, camera_id=-1)
    return frame[CROP_S:CROP_E, CROP_S:CROP_E]   # (128, 128, 3)


def rollout_sim(env, actions: np.ndarray,
                render: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    actions: (T, 9)
    Returns: obs_seq (T+1, 60), frames list of (128,128,3)
    """
    obs_list    = [env.unwrapped._get_obs()]
    frames      = [render_frame(env.unwrapped.sim)] if render else []

    for a in actions:
        obs, _, done, _ = env.step(a)
        obs_list.append(obs)
        if render:
            frames.append(render_frame(env.unwrapped.sim))
        if done:
            break

    return np.array(obs_list), frames


def save_gif(frames: List[np.ndarray], path: str, fps: int = 12):
    """frames: list of (H,W,3) uint8. Saves as GIF."""
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(
            path, save_all=True, append_images=imgs[1:],
            duration=int(1000 / fps), loop=0
        )
        print(f"  GIF saved: {path}  ({len(frames)} frames @ {fps}fps)")
    except ImportError:
        print("  PIL not available, saving frames as PNG strip instead")
        fig, axes = plt.subplots(1, min(len(frames), 16),
                                 figsize=(2 * min(len(frames), 16), 2))
        step = max(1, len(frames) // 16)
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[min(i * step, len(frames)-1)])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(path.replace('.gif', '_strip.png'), dpi=80)
        plt.close()


def run_sim_validation(
    planner:   KODAQLQRPlanner,
    episode:   Dict,
    x_seq_full: np.ndarray,
    env_name:  str  = 'kitchen-mixed-v0',
    cond_len:  int  = 32,
    out_dir:   str  = 'checkpoints/kodaq/validation',
    fps:       int  = 12,
    device:    str  = 'cuda',
):
    """
    한 에피소드에 대해:
      A. Real actions → sim → GIF (per subtask stage)
      B. LQR decoded actions (cond_len given + LQR) → sim → GIF

    GIF 하나당 한 subtask stage 구간.
    """
    import d4rl, gym
    dev = torch.device(device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    obs_ep   = episode['obs']      # (L, 60)
    acts_ep  = episode['actions']  # (L, 9)
    gi       = episode['goal_info']
    tasks    = episode['tasks']
    s_t      = episode['start_t']
    L        = episode['length']
    x_ep     = x_seq_full[s_t:s_t + L]

    jump_t   = sorted(gi['completions'].values())
    stage_ends = jump_t + [L - 1]

    print(f"\n=== Sim Validation  ep_len={L}  stages={len(stage_ends)} ===")
    print(f"  env={env_name}  cond_len={cond_len}")

    # Load env
    env = gym.make(env_name)
    model = planner.model

    for stage_idx, stage_end_t in enumerate(stage_ends):
        stage_start = 0 if stage_idx == 0 else stage_ends[stage_idx - 1] + 1
        stage_len   = stage_end_t - stage_start + 1
        label       = tasks[stage_idx] if stage_idx < len(tasks) else f'stage{stage_idx}'

        print(f"\n  Stage {stage_idx} [{label}]  t={stage_start}→{stage_end_t}  len={stage_len}")

        # ── A. Real action rollout ──────────────────────────────────────────
        env.reset()
        # replay to stage_start
        for t in range(stage_start):
            env.step(acts_ep[t])
        real_acts_stage = acts_ep[stage_start:stage_end_t + 1]   # (stage_len, 9)
        obs_real, frames_real = rollout_sim(env, real_acts_stage, render=True)

        gif_real = f"{out_dir}/ep{episode['start_t']}_stage{stage_idx}_{label}_A_real.gif"
        save_gif(frames_real, gif_real, fps=fps)

        # ── B. LQR decoded action rollout ─────────────────────────────────
        # conditioning: cond_len steps before stage_start
        cond_s = max(0, stage_start - cond_len)
        cond_e = stage_start
        if cond_e <= cond_s:
            cond_s = 0; cond_e = min(cond_len, stage_end_t)

        x_cond   = torch.FloatTensor(x_ep[cond_s:cond_e]).unsqueeze(0).to(dev)
        a_cond   = torch.FloatTensor(acts_ep[cond_s:cond_e]).unsqueeze(0).to(dev)

        goal_obs  = obs_ep[stage_end_t]
        x_goal_np = obs_to_x_goal(goal_obs, obs_ep[0])
        x_goal_t  = torch.FloatTensor(x_goal_np).unsqueeze(0).to(dev)

        # LQR plan
        with torch.no_grad():
            plan = planner.plan(x_cond, a_cond, x_goal_t,
                                horizon=stage_len,
                                compute_uncertainty=False)

        u_traj = plan['u_traj'].to(dev)   # (H, d_u)
        H_plan = len(u_traj)

        # Batch decode actions
        a_lqr = decode_action_batch(model, u_traj,
                                    n_steps=planner.cfg.action_inv_steps).cpu().numpy()  # (H, 9)

        # Sim rollout with decoded actions
        env.reset()
        for t in range(stage_start):
            env.step(acts_ep[t])
        # conditioning steps: use real actions
        for t in range(cond_s, cond_e):
            env.step(acts_ep[t])

        obs_lqr, frames_lqr = rollout_sim(env, a_lqr, render=True)

        gif_lqr = f"{out_dir}/ep{episode['start_t']}_stage{stage_idx}_{label}_B_lqr.gif"
        save_gif(frames_lqr, gif_lqr, fps=fps)

        # ── Comparison plot ────────────────────────────────────────────────
        _plot_stage_comparison(
            obs_real=obs_real,
            obs_lqr=obs_lqr,
            real_acts=real_acts_stage,
            lqr_acts=a_lqr,
            label=label,
            stage_idx=stage_idx,
            out_path=f"{out_dir}/ep{episode['start_t']}_stage{stage_idx}_{label}_compare.png",
        )

    env.close()


def _plot_stage_comparison(
    obs_real: np.ndarray,   # (T+1, 60)
    obs_lqr:  np.ndarray,   # (T+1, 60)
    real_acts: np.ndarray,  # (T, 9)
    lqr_acts:  np.ndarray,  # (T, 9)
    label: str,
    stage_idx: int,
    out_path: str,
):
    """Stage별 obs 비교 + action 비교 플롯."""
    T = min(len(obs_real), len(obs_lqr)) - 1
    ts = np.arange(T + 1)
    PAL = ['#1E88E5', '#E53935']

    # object state (obs[18:60]) mean
    obj_real = obs_real[:T+1, 18:60].mean(axis=1)
    obj_lqr  = obs_lqr[:T+1,  18:60].mean(axis=1)

    # qpos mean
    q_real = obs_real[:T+1, 0:9].mean(axis=1)
    q_lqr  = obs_lqr[:T+1,  0:9].mean(axis=1)

    # action comparison
    T_act = min(len(real_acts), len(lqr_acts))
    act_real_mean = real_acts[:T_act].mean(axis=1)
    act_lqr_mean  = lqr_acts[:T_act].mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(ts, obj_real, color=PAL[0], lw=1.5, label='real')
    ax.plot(ts, obj_lqr,  color=PAL[1], lw=1.5, ls='--', label='LQR')
    ax.set_title(f'Object state mean\n[{label}]', fontsize=9, fontweight='bold')
    ax.set_xlabel('step'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    ax.plot(ts, q_real, color=PAL[0], lw=1.5, label='real')
    ax.plot(ts, q_lqr,  color=PAL[1], lw=1.5, ls='--', label='LQR')
    ax.set_title(f'qpos mean\n[{label}]', fontsize=9, fontweight='bold')
    ax.set_xlabel('step'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    ax = axes[2]
    ax.plot(np.arange(T_act), act_real_mean, color=PAL[0], lw=1.5, label='real action')
    ax.plot(np.arange(T_act), act_lqr_mean,  color=PAL[1], lw=1.5, ls='--', label='LQR action')
    ax.set_title(f'Action mean (9 joints)\n[{label}]', fontsize=9, fontweight='bold')
    ax.set_xlabel('step'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    fig.suptitle(f'Stage {stage_idx} [{label}]  A:Real vs B:LQR',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      default='checkpoints/kodaq_v3/final.pt')
    p.add_argument('--x_cache',   default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--u_bounds',  default=None)
    p.add_argument('--env',       default='kitchen-mixed-v0')
    p.add_argument('--quality',   default='mixed')
    p.add_argument('--ep_idx',    type=int, default=0,
                   help='Episode index for sim validation')
    p.add_argument('--n_ep_decoder', type=int, default=5,
                   help='Episodes for decoder error analysis')
    p.add_argument('--cond_len',  type=int,   default=32)
    p.add_argument('--Q_scale',   type=float, default=1.0)
    p.add_argument('--R_scale',   type=float, default=10.0)
    p.add_argument('--lambda_unc',type=float, default=0.1)
    p.add_argument('--fps',       type=int,   default=12)
    p.add_argument('--out_dir',   default='checkpoints/kodaq/validation')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--skip_decoder', action='store_true')
    p.add_argument('--skip_sim',     action='store_true')
    args = p.parse_args()

    # Load model
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

    # Load data
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality, min_len=40)

    # ── 1. Decoder error analysis ─────────────────────────────────────────────
    if not args.skip_decoder:
        from lqr_planner import run_lqr_on_episodes
        eps_for_decoder = [ep for ep in episodes[:args.n_ep_decoder] if ep['tasks']]
        results = run_lqr_on_episodes(
            planner, eps_for_decoder, x_seq_full,
            cond_len=args.cond_len, horizon=32,
            unc_real_len=8, device=args.device)
        analyze_decoder_error(planner, results, args.out_dir)

    # ── 2. Sim rollout + GIF ──────────────────────────────────────────────────
    if not args.skip_sim:
        valid_eps = [ep for ep in episodes if ep['tasks']]
        if args.ep_idx >= len(valid_eps):
            print(f"ep_idx={args.ep_idx} out of range ({len(valid_eps)} valid eps)")
        else:
            ep = valid_eps[args.ep_idx]
            print(f"\nSim validation: ep_idx={args.ep_idx}  "
                  f"tasks={ep['tasks']}  len={ep['length']}")
            run_sim_validation(
                planner=planner,
                episode=ep,
                x_seq_full=x_seq_full,
                env_name=args.env,
                cond_len=args.cond_len,
                out_dir=args.out_dir,
                fps=args.fps,
                device=args.device,
            )

    print(f"\nDone. → {args.out_dir}/")


if __name__ == '__main__':
    main()