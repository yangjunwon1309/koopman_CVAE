"""
analyze_v5.py — KODAQ v5 Skill-Segment Rollout Analysis
=========================================================

각 episode의 skill segment를 단위로 rollout 분석:
  - segment 시작까지 전체 history를 given으로 인코딩 → h, o 확보
  - 그 시점부터 horizon step 예측
  - 10+ episode에 대해 실행 → 통계적으로 의미있는 결과

출력:
  Fig A. Per-episode rollout quality (Δq RMSE per segment)
  Fig B. Per-skill-category 통계 (RMSE boxplot)
  Fig C. Ensemble reward vs GT reward (per segment)
  Fig D. Summary stats table (CSV + console)

Usage:
    python analyze_v5.py \\
        --ckpt   checkpoints/kodaq_v5_lqr/final_heads.pt \\
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \\
        --env    kitchen-mixed-v0 \\
        --n_ep   10 \\
        --horizon 8 \\
        --out_dir checkpoints/kodaq_v5_lqr/analysis \\
        --device cuda
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    load_kitchen_episodes,
    obs_to_x_goal,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
)

PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
       '#8E24AA','#00ACC1','#FFB300','#6D4C41','#546E7A','#D81B60']


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model_v5(ckpt_path: str, device: str) -> KoopmanCVAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt['cfg']
    v5_defaults = dict(
        num_bins=101, v_min=0.0, v_max=5.0, num_q=2, tau=0.005,
        gamma=0.99, entropy_coef=0.01, log_std_min=-5.0, log_std_max=2.0,
        lambda_reward=1.0, lambda_q=1.0, lambda_pi=0.1,
        reward_ensemble_n=5, td_horizon=4, mopo_beta=1.0,
        use_ensemble_reward=True, use_lqr_policy=False, lqr_horizon=4,
    )
    for k, v in v5_defaults.items():
        if not hasattr(cfg, k): setattr(cfg, k, v)

    model = KoopmanCVAE(cfg)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval().to(device)
    print(f"Loaded: {ckpt_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    return model


def crop_episodes_by_reward(
    episodes: List[Dict],
    reward_crop: Optional[float],
    min_len: int,
) -> List[Dict]:
    """Crop episodes at the first timestep whose Kitchen reward reaches target."""
    if reward_crop is None:
        return episodes

    cropped = []
    no_hit = 0
    for ep in episodes:
        rew_ep = ep['rewards'].astype(np.float32)
        hits = np.where(rew_ep >= float(reward_crop))[0]
        if len(hits) == 0:
            no_hit += 1
            continue

        crop_t = int(hits[0])
        L = crop_t + 1
        if L < min_len:
            continue

        obs_crop = ep['obs'][:L]
        act_crop = ep['actions'][:L]
        rew_crop = ep['rewards'][:L]
        completed = {
            k: int(v)
            for k, v in ep['goal_info']['completions'].items()
            if int(v) <= crop_t
        }
        subtask_goals = {
            task: {'obs': obs_crop[t], 'timestep': t, 'completed': True}
            for task, t in completed.items()
        }
        goal_info = dict(ep['goal_info'])
        goal_info.update({
            'final_goal': obs_crop[-1],
            'midpoint_goal': obs_crop[L // 2],
            'subtask_goals': subtask_goals,
            'completions': completed,
            'episode_len': L,
            'n_completed': len(completed),
            'reward_total': float(rew_crop[-1]),
        })

        ep_crop = dict(ep)
        ep_crop.update({
            'obs': obs_crop,
            'actions': act_crop,
            'rewards': rew_crop,
            'end_t': ep['start_t'] + crop_t,
            'length': L,
            'tasks': list(completed.keys()),
            'goal_info': goal_info,
        })
        cropped.append(ep_crop)

    if cropped:
        lengths = np.asarray([e['length'] for e in cropped], dtype=np.int64)
        print(
            f"Reward crop: target={reward_crop}  episodes={len(cropped)}/{len(episodes)}  "
            f"no_hit={no_hit}  len=[{lengths.min()},{lengths.max()}]  "
            f"mean={lengths.mean():.1f}"
        )
    else:
        print(f"Reward crop: target={reward_crop} left no episodes.")
    return cropped


# ──────────────────────────────────────────────────────────────────────────────
# Core: skill-segment rollout analysis
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_skill_segments(
    model:       KoopmanCVAE,
    episodes:    List[Dict],
    x_seq_full:  np.ndarray,
    horizon:     int  = 8,
    device:      str  = 'cuda',
    min_seg_len: int  = 8,
) -> List[Dict]:
    """
    각 episode의 각 skill segment에 대해 rollout 분석.

    Segment 정의:
      completions = sorted task completion timesteps
      seg 0: [0,       c0)  goal = obs[c0]
      seg 1: [c0,      c1)  goal = obs[c1]
      ...
      seg N: [c_{N-1}, L)   goal = obs[L-1]

    Given:
      t=0 ~ seg_start: 전체 history를 encode_sequence로 h 확보
      rollout: seg_start 이후 horizon step Koopman 예측

    Returns list of segment result dicts.
    """
    dev     = torch.device(device)
    results = []

    cfg    = model.cfg
    dq_sl  = slice(X_DQ_START, X_DQ_END)   # Δq: joint position delta
    dp_sl  = slice(X_DP_START, X_DP_END)   # Δp: object position delta

    for ep_idx, ep in enumerate(episodes):
        s, e    = ep['start_t'], ep['end_t']
        L       = ep['length']
        obs_ep  = ep['obs']
        acts_ep = ep['actions']
        rew_ep  = ep['rewards']
        gi      = ep['goal_info']
        tasks   = ep['tasks']

        if not tasks or L < min_seg_len + horizon + 1:
            continue

        x_ep = x_seq_full[s:e+1]   # (L, x_dim)

        # ── Full episode encode: h_seq, o_seq ─────────────────────────────
        x_t  = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)
        a_t  = torch.FloatTensor(acts_ep).unsqueeze(0).to(dev)
        enc  = model.encode_sequence(x_t, a_t)
        h_seq = enc['h_seq'][0]   # (L, d_h)
        o_seq = enc['o_seq'][0]   # (L, m)
        u_seq = model.action_encoder(a_t[0])  # (L, d_u)

        # ── Segment boundaries ────────────────────────────────────────────
        completions    = gi['completions']   # {task: t}
        boundaries     = sorted(completions.values())
        boundaries.append(L - 1)

        # task label per segment
        sorted_tasks = [t for t, _ in sorted(completions.items(),
                                              key=lambda x: x[1])]

        seg_start = 0
        for seg_idx, seg_end in enumerate(boundaries):
            seg_len = seg_end - seg_start
            if seg_len < min_seg_len or seg_start + horizon >= L:
                seg_start = seg_end
                continue

            task_label = (sorted_tasks[seg_idx]
                          if seg_idx < len(sorted_tasks) else 'final')

            # ── h_init: h just before seg_start ───────────────────────────
            if seg_start > 0:
                h_init = h_seq[seg_start - 1].unsqueeze(0).to(dev)  # (1, d_h)
            else:
                h_init = model.recurrent.init_hidden(1, dev)

            # ── Rollout window ─────────────────────────────────────────────
            t0 = seg_start
            t1 = min(t0 + horizon, L - 1)
            H  = t1 - t0

            x_cond = x_t[:, max(0, t0-1):t0]   # 1-step conditioning at seg_start
            # If seg_start=0, use first frame
            if x_cond.shape[1] == 0:
                x_cond = x_t[:, :1]

            a_cond = a_t[:, max(0, t0-1):t0]
            if a_cond.shape[1] == 0:
                a_cond = a_t[:, :1]

            a_plan = a_t[:, t0:t1]   # GT actions for rollout

            # Rollout with correct h_init (full history)
            pred = model.rollout(x_cond, a_cond, a_plan, h_init=h_init)

            # ── Ground truth ───────────────────────────────────────────────
            x_true = x_ep[t0:t1]   # (H, x_dim)
            H_c    = min(pred['q'].shape[1], len(x_true))

            dq_pred = pred['q'][0, :H_c].cpu().numpy()   # (H_c, 9)
            dq_true = x_true[:H_c, dq_sl]
            dp_pred = pred['delta_p'][0, :H_c].cpu().numpy()
            dp_true = x_true[:H_c, dp_sl]

            rmse_dq = float(np.sqrt(((dq_pred - dq_true)**2).mean()))
            rmse_dp = float(np.sqrt(((dp_pred - dp_true)**2).mean()))
            rmse_per_joint = np.sqrt(((dq_pred - dq_true)**2).mean(0))  # (9,)

            # ── Ensemble reward along GT trajectory ───────────────────────
            z_seg = o_seq[t0:t1]   # (H, m)
            u_seg = u_seq[t0:t1]   # (H, d_u)
            r_gt  = rew_ep[t0:t1]  # (H,)

            has_ens  = hasattr(model, 'reward_ensemble_head')
            r_mu_seg = r_std_seg = r_pen_seg = None
            if has_ens:
                probs = model.reward_ensemble_head.member_probs(
                    z_seg, u_seg)                  # (N, H)
                r_mu_seg  = probs.mean(0).cpu().numpy()   # (H,)
                r_std_seg = probs.std(0).cpu().numpy()    # (H,)
                r_pen_seg = (r_mu_seg
                             - cfg.mopo_beta * r_std_seg).clip(0, 1)

            results.append({
                'ep_idx':    ep_idx,
                'seg_idx':   seg_idx,
                'task':      task_label,
                'seg_start': seg_start,
                'seg_end':   seg_end,
                'H':         H_c,
                'rmse_dq':   rmse_dq,
                'rmse_dp':   rmse_dp,
                'rmse_per_joint': rmse_per_joint,
                'dq_pred':   dq_pred,
                'dq_true':   dq_true,
                'dp_pred':   dp_pred,
                'dp_true':   dp_true,
                'r_gt':      r_gt[:H_c],
                'r_mu':      r_mu_seg,
                'r_std':     r_std_seg,
                'r_pen':     r_pen_seg,
                'reward_total': float(gi['reward_total']),
            })
            seg_start = seg_end

    print(f"\nSegment analysis: {len(results)} segments from "
          f"{len(set(r['ep_idx'] for r in results))} episodes")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_statistics(results: List[Dict]) -> Dict:
    """Per-task and per-episode statistics."""
    # Per-task
    by_task = defaultdict(list)
    for r in results:
        by_task[r['task']].append(r['rmse_dq'])

    task_stats = {}
    for task, vals in by_task.items():
        v = np.array(vals)
        task_stats[task] = {
            'n':      len(v),
            'mean':   float(v.mean()),
            'std':    float(v.std()),
            'median': float(np.median(v)),
            'q25':    float(np.percentile(v, 25)),
            'q75':    float(np.percentile(v, 75)),
            'min':    float(v.min()),
            'max':    float(v.max()),
        }

    # Per-episode
    by_ep = defaultdict(list)
    for r in results:
        by_ep[r['ep_idx']].append(r['rmse_dq'])

    ep_stats = {}
    for ep_idx, vals in by_ep.items():
        v = np.array(vals)
        ep_stats[ep_idx] = {
            'n_segs': len(v),
            'mean':   float(v.mean()),
            'std':    float(v.std()),
            'reward': next(r['reward_total']
                          for r in results if r['ep_idx'] == ep_idx),
        }

    # Overall
    all_rmse = np.array([r['rmse_dq'] for r in results])
    overall  = {
        'n':      len(all_rmse),
        'mean':   float(all_rmse.mean()),
        'std':    float(all_rmse.std()),
        'median': float(np.median(all_rmse)),
    }

    # Per-joint RMSE averaged
    per_joint = np.stack([r['rmse_per_joint'] for r in results]).mean(0)  # (9,)

    return {
        'task_stats':  task_stats,
        'ep_stats':    ep_stats,
        'overall':     overall,
        'per_joint':   per_joint,
    }


def print_statistics(stats: Dict, out_dir: Path):
    """Console + CSV 출력."""
    print("\n" + "="*70)
    print("ROLLOUT QUALITY STATISTICS (Δq RMSE)")
    print("="*70)

    print("\n[Overall]")
    ov = stats['overall']
    print(f"  n={ov['n']}  mean={ov['mean']:.4f}  "
          f"std={ov['std']:.4f}  median={ov['median']:.4f}")

    print("\n[Per Task]")
    header = f"  {'Task':<20} {'N':>4} {'Mean':>8} {'Std':>8} "
    header += f"{'Median':>8} {'Q25':>8} {'Q75':>8}"
    print(header)
    print("  " + "-"*68)
    rows = []
    for task, s in sorted(stats['task_stats'].items(),
                           key=lambda x: x[1]['mean']):
        row = (f"  {task:<20} {s['n']:>4} {s['mean']:>8.4f} "
               f"{s['std']:>8.4f} {s['median']:>8.4f} "
               f"{s['q25']:>8.4f} {s['q75']:>8.4f}")
        print(row)
        rows.append([task, s['n'], s['mean'], s['std'],
                     s['median'], s['q25'], s['q75']])

    print("\n[Per Episode]")
    print(f"  {'Ep':>4} {'Segs':>5} {'Mean RMSE':>10} {'Std':>8} {'Reward':>8}")
    print("  " + "-"*40)
    for ep_idx, s in sorted(stats['ep_stats'].items()):
        print(f"  {ep_idx:>4} {s['n_segs']:>5} {s['mean']:>10.4f} "
              f"{s['std']:>8.4f} {s['reward']:>8.1f}")

    print("\n[Per Joint RMSE (avg over all segments)]")
    pj = stats['per_joint']
    for j, v in enumerate(pj):
        bar = '█' * int(v * 100)
        print(f"  Joint {j}: {v:.4f}  {bar}")

    # Save CSV
    import csv
    csv_path = out_dir / 'task_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'n', 'mean', 'std', 'median', 'q25', 'q75'])
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig A: Per-episode rollout quality
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_episode_rollout(results: List[Dict], out_dir: Path):
    """
    각 episode의 각 segment에 대해 GT vs pred Δq 비교.
    Episode별로 한 figure, segment별로 subplot.
    """
    by_ep = defaultdict(list)
    for r in results:
        by_ep[r['ep_idx']].append(r)

    cmap9 = plt.get_cmap('tab10')

    for ep_idx, segs in sorted(by_ep.items()):
        n   = len(segs)
        fig, axes = plt.subplots(n, 3, figsize=(18, 3.5*n), squeeze=False)

        for si, seg in enumerate(segs):
            H  = seg['H']
            ts = np.arange(H)

            # Δq GT vs pred
            ax = axes[si, 0]
            for d in range(9):
                ax.plot(ts, seg['dq_true'][:,d], '-',
                        color=cmap9(d), lw=1.2, alpha=0.8)
                ax.plot(ts, seg['dq_pred'][:,d], '--',
                        color=cmap9(d), lw=1.2, alpha=0.8)
            ax.set_title(f"[{seg['task']}] t={seg['seg_start']}→{seg['seg_end']}"
                         f"  RMSE={seg['rmse_dq']:.4f}", fontsize=9)
            ax.set_ylabel('Δq [rad]', fontsize=8)
            ax.set_xlabel('step', fontsize=8)
            ax.spines[['top','right']].set_visible(False)

            # RMSE per joint bar
            ax = axes[si, 1]
            colors = [cmap9(d) for d in range(9)]
            ax.bar(range(9), seg['rmse_per_joint'], color=colors, alpha=0.8)
            ax.set_title(f"RMSE per joint  dp={seg['rmse_dp']:.4f}", fontsize=9)
            ax.set_xlabel('joint'); ax.set_ylabel('RMSE')
            ax.spines[['top','right']].set_visible(False)

            # Reward head vs GT
            ax = axes[si, 2]
            r_gt = seg['r_gt']
            if r_gt is not None and r_gt.sum() > 0:
                ax.bar(ts[:len(r_gt)], r_gt, color='#43A047',
                       alpha=0.3, width=1.0, label='GT sparse')
                for rs in np.where(r_gt > 0)[0]:
                    ax.axvline(rs, color='#43A047', lw=1.5, ls='--', alpha=0.5)
            if seg['r_mu'] is not None:
                rm = seg['r_mu'][:H]
                rs = seg['r_std'][:H]
                ax.fill_between(ts[:len(rm)], rm-rs, rm+rs,
                                color='#1E88E5', alpha=0.2, label='±std')
                ax.plot(ts[:len(rm)], rm, color='#1E88E5',
                        lw=2.0, label='Ens. mean')
                if seg['r_pen'] is not None:
                    ax.plot(ts[:len(seg['r_pen'])], seg['r_pen'][:H],
                            color='#E53935', lw=1.5, ls='--', label='Penalized')
            ax.set_title('Ensemble reward vs GT', fontsize=9)
            ax.set_xlabel('step'); ax.set_ylabel('P(r=1)')
            ax.legend(fontsize=7, loc='upper right')
            ax.spines[['top','right']].set_visible(False)

        fig.suptitle(f'Ep {ep_idx}  (solid=GT, dashed=pred)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = str(out_dir / f'ep{ep_idx:03d}_segments.png')
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()

    print(f"Saved: {out_dir}/ep*_segments.png")


# ──────────────────────────────────────────────────────────────────────────────
# Fig B: Per-skill-category statistics (boxplot)
# ──────────────────────────────────────────────────────────────────────────────

def plot_skill_statistics(results: List[Dict], stats: Dict, out_dir: Path):
    """
    Skill category별 RMSE boxplot + mean/std bar.
    """
    by_task = defaultdict(list)
    for r in results:
        by_task[r['task']].append(r['rmse_dq'])

    tasks      = sorted(by_task.keys(), key=lambda t: np.mean(by_task[t]))
    task_vals  = [by_task[t] for t in tasks]
    n_tasks    = len(tasks)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_tasks*1.5), 5))

    # Boxplot
    ax = axes[0]
    bp = ax.boxplot(task_vals, patch_artist=True, notch=False,
                    vert=True, showfliers=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PAL[i % len(PAL)])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, n_tasks+1))
    ax.set_xticklabels([t.replace(' ', '\n') for t in tasks],
                       fontsize=8)
    ax.set_ylabel('Δq RMSE', fontsize=10)
    ax.set_title('RMSE by Skill Category (boxplot)', fontsize=11)
    ax.spines[['top','right']].set_visible(False)

    # Mean ± std bar
    ax = axes[1]
    means = [np.mean(by_task[t]) for t in tasks]
    stds  = [np.std(by_task[t])  for t in tasks]
    ns    = [len(by_task[t])      for t in tasks]
    x     = np.arange(n_tasks)
    bars  = ax.bar(x, means, yerr=stds, capsize=4,
                   color=[PAL[i % len(PAL)] for i in range(n_tasks)],
                   alpha=0.8, ecolor='black', error_kw={'lw': 1.5})
    for i, (xi, ni) in enumerate(zip(x, ns)):
        ax.text(xi, means[i] + stds[i] + 0.002, f'n={ni}',
                ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(' ', '\n') for t in tasks], fontsize=8)
    ax.set_ylabel('Mean RMSE ± Std', fontsize=10)
    ax.set_title('RMSE Mean ± Std by Skill', fontsize=11)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = str(out_dir / 'skill_category_stats.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig C: Reward prediction quality per skill
# ──────────────────────────────────────────────────────────────────────────────

def plot_reward_by_skill(results: List[Dict], out_dir: Path):
    """
    Skill별로 reward head 예측 품질.
    각 segment에서 GT reward가 발생하는 step에서의 r_mu 분포.
    """
    segs_with_reward = [r for r in results
                        if r['r_mu'] is not None and r['r_gt'].sum() > 0]
    segs_no_reward   = [r for r in results
                        if r['r_mu'] is not None and r['r_gt'].sum() == 0]

    if not segs_with_reward:
        print("No segments with reward for Fig C.")
        return

    # Collect: at reward timesteps, what does r_mu predict?
    by_task_reward = defaultdict(list)    # r_mu at reward step
    by_task_no_rew = defaultdict(list)    # r_mu at non-reward steps

    for seg in results:
        if seg['r_mu'] is None:
            continue
        r_gt  = seg['r_gt']
        r_mu  = seg['r_mu']
        H     = min(len(r_gt), len(r_mu))
        task  = seg['task']
        for t in range(H):
            if r_gt[t] > 0:
                by_task_reward[task].append(r_mu[t])
            else:
                by_task_no_rew[task].append(r_mu[t])

    all_tasks = sorted(set(list(by_task_reward.keys())
                           + list(by_task_no_rew.keys())))
    n = len(all_tasks)
    if n == 0:
        return

    fig, axes = plt.subplots(1, min(n, 5), figsize=(4*min(n,5), 4),
                             squeeze=False)
    axes = axes[0]

    for i, task in enumerate(all_tasks[:5]):
        ax   = axes[i]
        rew  = np.array(by_task_reward.get(task, []))
        nrew = np.array(by_task_no_rew.get(task, []))

        data   = []
        labels = []
        if len(rew) > 0:
            data.append(rew);  labels.append(f'r=1\n(n={len(rew)})')
        if len(nrew) > 0:
            # Sample to avoid huge violin
            idx = np.random.choice(len(nrew), min(200, len(nrew)), replace=False)
            data.append(nrew[idx]); labels.append(f'r=0\n(n={len(nrew)})')

        if len(data) >= 2:
            vp = ax.violinplot(data, showmedians=True, showextrema=True)
            for pc, col in zip(vp['bodies'],
                                ['#E53935', '#1E88E5'][:len(data)]):
                pc.set_facecolor(col); pc.set_alpha(0.6)
        elif len(data) == 1:
            ax.boxplot(data)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(task.replace(' ', '\n'), fontsize=9)
        ax.set_ylabel('P(r=1) prediction', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # Discrimination stat
        if len(rew) > 0 and len(nrew) > 0:
            sep = float(rew.mean()) - float(nrew.mean())
            ax.text(0.5, 0.95, f'Δmean={sep:.4f}',
                    transform=ax.transAxes, ha='center',
                    fontsize=8, color='#333333')

    fig.suptitle('Reward Head: P(r=1) at reward vs non-reward steps',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = str(out_dir / 'reward_discrimination.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig D: Summary heatmap (episode × task)
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary_heatmap(results: List[Dict], stats: Dict, out_dir: Path):
    """Episode × task RMSE heatmap."""
    ep_ids  = sorted(set(r['ep_idx']  for r in results))
    tasks   = sorted(set(r['task']    for r in results))

    # Build matrix
    mat     = np.full((len(ep_ids), len(tasks)), np.nan)
    ep_map  = {e: i for i, e in enumerate(ep_ids)}
    tk_map  = {t: j for j, t in enumerate(tasks)}

    for r in results:
        i = ep_map[r['ep_idx']]
        j = tk_map[r['task']]
        if np.isnan(mat[i, j]):
            mat[i, j] = r['rmse_dq']
        else:
            mat[i, j] = (mat[i, j] + r['rmse_dq']) / 2   # avg if multiple segs

    fig, ax = plt.subplots(figsize=(max(8, len(tasks)*1.2),
                                    max(5, len(ep_ids)*0.4)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=np.nanpercentile(mat, 95))

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.replace(' ', '\n') for t in tasks], fontsize=9)
    ax.set_yticks(range(len(ep_ids)))
    ax.set_yticklabels([f'Ep{e}' for e in ep_ids], fontsize=8)
    ax.set_title('RMSE Δq heatmap  (ep × skill)', fontsize=11)

    # Annotate
    for i in range(len(ep_ids)):
        for j in range(len(tasks)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if mat[i,j] > np.nanmedian(mat) else 'black')

    plt.colorbar(im, ax=ax, label='RMSE Δq')
    plt.tight_layout()
    path = str(out_dir / 'summary_heatmap.png')
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
    p.add_argument('--env',        default='kitchen-mixed-v0')
    p.add_argument('--quality',    default='mixed',
                   choices=['mixed', 'partial', 'complete'])
    p.add_argument('--n_ep',       type=int, default=10,
                   help='Number of episodes to analyze (≥10 for statistics)')
    p.add_argument('--horizon',    type=int, default=8,
                   help='Rollout horizon per segment (≤8 recommended)')
    p.add_argument('--min_seg_len',type=int, default=8,
                   help='Minimum segment length to analyze')
    p.add_argument('--reward_crop', type=float, default=2.0,
                   help='Crop each episode at first reward >= target; set <0 to disable')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v5_lqr/analysis')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.ckpt}")
    model = load_model_v5(args.ckpt, args.device)

    print(f"\nLoading dataset: {args.env}")
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(
        quality=args.quality, min_len=args.min_seg_len + args.horizon + 2)
    reward_crop = None if args.reward_crop < 0 else args.reward_crop
    episodes = crop_episodes_by_reward(
        episodes, reward_crop, min_len=args.min_seg_len + args.horizon + 2)

    # Select episodes with tasks, sorted by reward (descending for variety)
    eps_with_tasks = [e for e in episodes if e['tasks']]
    eps_with_tasks.sort(key=lambda e: e['goal_info']['reward_total'], reverse=True)
    selected = eps_with_tasks[:args.n_ep]
    if not selected:
        print("No episodes selected. Check reward_crop, quality, and min lengths.")
        exit(1)
    print(f"Selected {len(selected)} episodes  "
          f"(reward range: {selected[-1]['goal_info']['reward_total']:.0f}"
          f"~{selected[0]['goal_info']['reward_total']:.0f})")

    # ── Core analysis ─────────────────────────────────────────────────────────
    print(f"\nAnalyzing skill segments (horizon={args.horizon}) ...")
    results = analyze_skill_segments(
        model, selected, x_seq_full,
        horizon=args.horizon, device=args.device,
        min_seg_len=args.min_seg_len,
    )

    if not results:
        print("No segments to analyze. Check episodes and min_seg_len.")
        exit(1)

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = compute_statistics(results)
    print_statistics(stats, out)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n=== Fig A: Per-episode segment rollout ===")
    plot_per_episode_rollout(results, out)

    print("\n=== Fig B: Skill category statistics ===")
    plot_skill_statistics(results, stats, out)

    print("\n=== Fig C: Reward discrimination ===")
    plot_reward_by_skill(results, out)

    print("\n=== Fig D: Summary heatmap ===")
    plot_summary_heatmap(results, stats, out)

    print(f"\nAll outputs → {out}/")
