"""
eval_online_policy.py
=====================
KODAQ-Online policy (kodaq_online.py로 학습된) 평가 + GIF 생성.

기능:
  1. N개 episode 실행 (fixed seed + random seed 각각)
  2. 각 episode별 policy 동작 GIF 저장
  3. episode별 reward, task completion 분석 플롯
  4. info dict에서 실제 task completion key 탐색 및 출력

Usage:
    MUJOCO_GL=egl python eval_online_policy.py \\
        --world_ckpt  checkpoints/kodaq_v4/final.pt \\
        --online_ckpt checkpoints/kodaq_v4/online_v7/kodaq_online_final.pt \\
        --cat_ckpt    checkpoints/kodaq_v4/cat_reward/final.pt \\
        --env         kitchen-mixed-v0 \\
        --n_ep        10 \\
        --out_dir     checkpoints/kodaq_v4/eval_online \\
        --device      cuda:1
"""

import os, sys, math
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.koopman_cvae import KoopmanCVAE
from kodaq_online import (
    KODAQOnlineTrainer, KoopmanWorldModelWrapper, EnvContext, OnlineConfig,
    HighLevelPolicy, LowLevelPolicy, QNetwork,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_gif(frames: List[np.ndarray], path: str, fps: int = 10):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        print(f"    GIF saved: {path}  ({len(frames)} frames @ {fps}fps)")
    except ImportError:
        # PIL 없으면 strip PNG로 대체
        n = min(len(frames), 12)
        step = max(1, len(frames) // n)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[min(i * step, len(frames) - 1)])
            ax.axis('off')
        plt.tight_layout()
        strip = path.replace('.gif', '_strip.png')
        plt.savefig(strip, dpi=80); plt.close()
        print(f"    Strip saved: {strip}")


def render_frame(env, width=256, height=256) -> np.ndarray:
    """kitchen env에서 프레임 렌더링"""
    try:
        frame = env.render(mode='rgb_array', width=width, height=height)
        if frame is None:
            frame = env.unwrapped.sim.render(width, height, camera_name='main_cam')
    except Exception:
        try:
            frame = env.unwrapped.sim.render(width, height)
        except Exception:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
    return frame


def inspect_info(info: dict) -> Tuple[int, List[str]]:
    """
    D4RL kitchen info dict에서 task completion 정보를 탐색.
    여러 가지 가능한 키를 순서대로 시도.
    Returns: (n_tasks_completed, completed_task_names)
    """
    # D4RL kitchen에서 사용하는 실제 키들
    # 'tasks_to_complete': remaining tasks (list가 줄어들면 완료)
    # 'num_success': 완료된 수 (일부 버전)
    # 'goal': dict with 'completed' 등

    n_tasks = 0
    completed = []

    if 'num_success' in info:
        n_tasks = int(info['num_success'])
    elif 'completed_tasks' in info:
        completed = list(info['completed_tasks'])
        n_tasks = len(completed)
    elif 'tasks_to_complete' in info:
        # 처음 tasks 수에서 남은 수 빼기 (episode 전체 추적 필요)
        pass
    elif 'goal_achieved' in info:
        n_tasks = int(info['goal_achieved'])

    # 기타 numeric info keys 탐색 (디버깅용)
    return n_tasks, completed


def get_total_tasks_from_env(env) -> int:
    """episode의 목표 task 총 수 확인"""
    try:
        tasks = env.unwrapped.tasks_to_complete
        return len(tasks)
    except Exception:
        return 4  # kitchen-mixed default


# ─────────────────────────────────────────────────────────────────────────────
# Load KODAQ-Online policy
# ─────────────────────────────────────────────────────────────────────────────

def load_online_policy(
    world_ckpt:  str,
    online_ckpt: str,
    cat_ckpt:    Optional[str],
    device:      str,
):
    """
    KoopmanCVAE + KODAQOnlineTrainer 로드.
    online_ckpt에서 pi_hi, pi_lo, Q 복원.
    """
    dev = device
    print(f"\n[Load] World model: {world_ckpt}")
    ckpt  = torch.load(world_ckpt, map_location=dev)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(dev)

    z_dim    = model.cfg.koopman_dim
    n_skills = model.cfg.num_skills
    a_dim    = model.cfg.action_dim
    print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")

    # cat_head
    cat_head = None
    if cat_ckpt and Path(cat_ckpt).exists():
        from train_reward_head import load_cat_reward_model
        _, cat_head = load_cat_reward_model(cat_ckpt, dev)
        print(f"  CategoricalRewardHead loaded")

    cfg = OnlineConfig()
    wm  = KoopmanWorldModelWrapper(model, cfg.wm_lr, dev,
                                    cat_head=cat_head,
                                    reward_H=8, reward_gamma=0.9)

    trainer = KODAQOnlineTrainer(cfg, wm, z_dim, n_skills, a_dim, dev)

    print(f"[Load] Online ckpt: {online_ckpt}")
    ok_ckpt = torch.load(online_ckpt, map_location=dev)

    # world model이 online ckpt에 포함되어 있으면 덮어쓰기
    if 'world_model' in ok_ckpt:
        model.load_state_dict(ok_ckpt['world_model'])
        model.eval()

    trainer.pi_hi.load_state_dict(ok_ckpt['pi_hi'])
    trainer.pi_lo.load_state_dict(ok_ckpt['pi_lo'])
    trainer.Q1.load_state_dict(ok_ckpt['Q1'])
    trainer.Q2.load_state_dict(ok_ckpt['Q2'])
    trainer.pi_hi.eval(); trainer.pi_lo.eval()
    trainer.Q1.eval();    trainer.Q2.eval()
    print(f"  step={ok_ckpt.get('step', 'N/A')}")

    return model, trainer, wm


# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_episode(
    env,
    model:   KoopmanCVAE,
    trainer: KODAQOnlineTrainer,
    cfg:     OnlineConfig,
    device:  str,
    max_steps: int = 280,
    render:    bool = True,
) -> Dict:
    """
    한 episode 전체 rollout.
    Returns: {
        frames, rewards, n_tasks, skill_ids, info_history,
        total_reward, completed_tasks
    }
    """
    dev = torch.device(device)
    obs = env.reset()
    ctx = EnvContext(model, device, cfg.cond_len)
    ctx.reset(obs)

    frames      = [render_frame(env)] if render else []
    rewards     = []
    skill_ids   = []
    info_hist   = []
    total_r     = 0.0
    n_tasks_max = 0
    done        = False
    hi_timer    = 0
    sid         = 0
    n_tasks_init= get_total_tasks_from_env(env)

    for t in range(max_steps):
        if done: break

        # context 아직 없으면 random step
        if ctx.z_t is None:
            a = env.action_space.sample()
            obs, r, done, info = env.step(a)
            ctx.step(obs, a)
            if render: frames.append(render_frame(env))
            total_r += r; rewards.append(r)
            skill_ids.append(-1)
            n_t, _ = inspect_info(info)
            n_tasks_max = max(n_tasks_max, n_t)
            info_hist.append(info)
            continue

        # Hi-level: H_hi마다 skill 재선택
        if hi_timer == 0:
            sid, _ = trainer.pi_hi.hard_sample(ctx.z_t)
            hi_timer = cfg.H_hi

        sp = torch.zeros(1, trainer.n_skills, device=dev)
        sp[0, sid] = 1.0

        # Lo-level: action chunk
        with torch.no_grad():
            a_seq, _ = trainer.pi_lo.sample(ctx.z_t, sp)
        a_np = a_seq[0].cpu().numpy()   # (H_lo, 9)

        # Execute H_lo steps
        for k in range(cfg.H_lo):
            if done: break
            ak = a_np[k].clip(-1, 1)
            obs, r, done, info = env.step(ak)
            ctx.step(obs, ak)
            if render: frames.append(render_frame(env))
            total_r += r; rewards.append(r)
            skill_ids.append(sid)
            n_t, _ = inspect_info(info)
            n_tasks_max = max(n_tasks_max, n_t)
            info_hist.append(info)

        hi_timer = max(0, hi_timer - cfg.H_lo)

    # info key 탐색 (첫 번째 non-empty info에서)
    info_keys = set()
    for info in info_hist:
        info_keys.update(info.keys())

    return {
        'frames':      frames,
        'rewards':     rewards,
        'total_reward': total_r,
        'n_tasks':     n_tasks_max,
        'n_tasks_total': n_tasks_init,
        'skill_ids':   skill_ids,
        'info_keys':   info_keys,
        'info_hist':   info_hist[-1] if info_hist else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualize episode summary
# ─────────────────────────────────────────────────────────────────────────────

def visualize_episode(result: Dict, ep_idx: int, out_path: str):
    """episode별 reward/skill 분석 플롯"""
    rewards   = result['rewards']
    skill_ids = result['skill_ids']
    T = len(rewards)
    if T == 0: return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']

    # Reward per step
    ax = axes[0]
    ax.plot(rewards, color='#1E88E5', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    cum_r = np.cumsum(rewards)
    ax2 = ax.twinx()
    ax2.plot(cum_r, color='#E53935', lw=1.5, ls='--', alpha=0.7)
    ax2.set_ylabel('Cumulative Reward', color='#E53935', fontsize=8)
    ax.set_title(f'Ep {ep_idx}  |  total_r={result["total_reward"]:.3f}  '
                 f'tasks={result["n_tasks"]}/{result["n_tasks_total"]}',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('Step Reward'); ax.spines[['top','right']].set_visible(False)

    # Skill ID over time
    ax = axes[1]
    valid_mask = np.array(skill_ids) >= 0
    if valid_mask.any():
        ts = np.where(valid_mask)[0]
        sids = np.array(skill_ids)[valid_mask]
        for k in range(7):
            mask = sids == k
            if mask.any():
                ax.scatter(ts[mask], [k] * mask.sum(),
                           c=PAL[k % len(PAL)], s=4, alpha=0.7,
                           label=f'skill {k}')
    ax.set_ylabel('Skill ID'); ax.set_ylim(-0.5, 6.5)
    ax.set_yticks(range(7))
    ax.legend(loc='upper right', fontsize=6, ncol=7)
    ax.spines[['top','right']].set_visible(False)

    # Reward spikes (task completion events)
    ax = axes[2]
    r_arr = np.array(rewards)
    spike_idx = np.where(r_arr > 0.5)[0]
    ax.bar(range(T), r_arr, color='#43A047', alpha=0.5, width=1.0)
    if len(spike_idx) > 0:
        ax.scatter(spike_idx, r_arr[spike_idx], color='#E53935',
                   s=30, zorder=5, label=f'task complete ({len(spike_idx)})')
        ax.legend(fontsize=8)
    ax.set_xlabel('Step'); ax.set_ylabel('Reward')
    ax.set_title('Task Completion Events', fontsize=9)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def visualize_summary(results: List[Dict], out_path: str):
    """전체 episode 요약 플롯"""
    total_rs = [r['total_reward'] for r in results]
    n_tasks  = [r['n_tasks']      for r in results]
    ep_ids   = list(range(len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(ep_ids, total_rs, color='#1E88E5', alpha=0.8)
    ax.axhline(np.mean(total_rs), color='#E53935', ls='--', lw=1.5,
               label=f'mean={np.mean(total_rs):.3f}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    ax = axes[1]
    ax.bar(ep_ids, n_tasks, color='#43A047', alpha=0.8)
    ax.axhline(np.mean(n_tasks), color='#E53935', ls='--', lw=1.5,
               label=f'mean={np.mean(n_tasks):.2f}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Tasks Completed')
    ax.set_title('Tasks Completed per Episode', fontsize=10, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    fig.suptitle(
        f'KODAQ-Online Evaluation  ({len(results)} episodes)\n'
        f'mean_reward={np.mean(total_rs):.3f}  mean_tasks={np.mean(n_tasks):.2f}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_ckpt',  required=True)
    p.add_argument('--online_ckpt', required=True)
    p.add_argument('--cat_ckpt',    default=None)
    p.add_argument('--env',         default='kitchen-mixed-v0')
    p.add_argument('--n_ep',        type=int,  default=10)
    p.add_argument('--fixed_seed',  type=int,  default=42,
                   help='Fixed seed for first half of episodes (reproducibility)')
    p.add_argument('--fps',         type=int,  default=10)
    p.add_argument('--max_steps',   type=int,  default=280)
    p.add_argument('--no_gif',      action='store_true',
                   help='Skip GIF generation (faster)')
    p.add_argument('--out_dir',     default='checkpoints/kodaq_v4/eval_online')
    p.add_argument('--device',      default='cuda:1'
                   if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    import gym, d4rl
    device = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load policy ────────────────────────────────────────────────────────
    model, trainer, wm = load_online_policy(
        args.world_ckpt, args.online_ckpt, args.cat_ckpt, device)
    cfg = OnlineConfig()

    # ── Run episodes ───────────────────────────────────────────────────────
    results  = []
    all_info_keys = set()

    print(f"\n{'='*55}")
    print(f"Evaluating {args.n_ep} episodes  env={args.env}")
    print(f"{'='*55}\n")

    for ep_i in range(args.n_ep):
        env = gym.make(args.env)

        # 앞 절반: fixed seed (재현 가능), 뒤 절반: random
        if ep_i < args.n_ep // 2:
            env.seed(args.fixed_seed + ep_i)
            seed_label = f'seed{args.fixed_seed + ep_i}'
        else:
            seed_label = 'random'

        print(f"  Episode {ep_i:2d} [{seed_label}]  ", end='', flush=True)

        result = rollout_episode(
            env, model, trainer, cfg, device,
            max_steps=args.max_steps,
            render=not args.no_gif,
        )
        env.close()

        all_info_keys.update(result['info_keys'])
        results.append(result)

        print(f"reward={result['total_reward']:.3f}  "
              f"tasks={result['n_tasks']}/{result['n_tasks_total']}  "
              f"steps={len(result['rewards'])}")

        # GIF 저장
        if not args.no_gif and result['frames']:
            gif_path = f"{args.out_dir}/gif/ep{ep_i:02d}_{seed_label}_r{result['total_reward']:.2f}.gif"
            save_gif(result['frames'], gif_path, fps=args.fps)

        # Episode 분석 플롯
        plot_path = f"{args.out_dir}/plots/ep{ep_i:02d}_{seed_label}.png"
        visualize_episode(result, ep_i, plot_path)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results Summary ({args.n_ep} episodes):")
    total_rs = [r['total_reward'] for r in results]
    n_tasks  = [r['n_tasks']      for r in results]
    print(f"  mean_reward: {np.mean(total_rs):.4f} ± {np.std(total_rs):.4f}")
    print(f"  max_reward:  {np.max(total_rs):.4f}")
    print(f"  mean_tasks:  {np.mean(n_tasks):.4f}")
    print(f"  max_tasks:   {int(np.max(n_tasks))}")
    print(f"\n  info dict keys found: {sorted(all_info_keys)}")
    print(f"  → use one of these for task completion count")
    print(f"{'='*55}")

    visualize_summary(results, f"{args.out_dir}/summary.png")

    # ── info key 탐색 결과로 eval_tasks 수정 가이드 ────────────────────────
    print(f"\n[Fix eval_tasks] In kodaq_online.py, update:")
    print(f"  n_tasks = max(n_tasks, int(info.get('num_success', 0)))")
    print(f"  → change key to one of: {sorted(all_info_keys)}")
    task_keys = [k for k in all_info_keys
                 if any(w in k.lower()
                        for w in ['task','success','complete','solve','goal'])]
    if task_keys:
        print(f"  Likely key(s): {task_keys}")

    print(f"\nOutputs → {args.out_dir}/")


if __name__ == '__main__':
    main()