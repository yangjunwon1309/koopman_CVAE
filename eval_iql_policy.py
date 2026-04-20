"""
eval_iql_policy.py — IQL Policy Evaluation + GIF Generation
============================================================

평가 항목:
  1. Training curve 재분석 (Q/V/adv 진단 플롯)
  2. Policy rollout vs LQR vs Real 비교 (world model 기반)
  3. Sim rollout GIF:
       A. 학습에 사용한 에피소드 (train split)
       B. 학습에 사용하지 않은 에피소드 (held-out split)
     각각 real / LQR / policy 3-way GIF

Usage:
    MUJOCO_GL=egl python eval_iql_policy.py \
        --world_ckpt checkpoints/kodaq_v4/final.pt \
        --iql_ckpt   checkpoints/kodaq_v4/iql/iql_final.pt \
        --lqr_cache  checkpoints/kodaq_v4/iql/lqr_cache.npz \
        --x_cache    checkpoints/skill_pretrain/x_sequences.npz \
        --out_dir    checkpoints/kodaq_v4/eval \
        --device     cuda:1
"""

import os, sys, math
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.koopman_cvae import KoopmanCVAE
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    KODAQLQRPlanner, LQRConfig, load_kitchen_episodes,
    obs_to_x_goal, blend_koopman,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
)

# IQL 네트워크 정의를 iql_koopman에서 가져옴
from iql_koopman import QNetwork, VNetwork, GaussianPolicy, IQLConfig


PAL = ['#1E88E5', '#E53935', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']


# ─────────────────────────────────────────────────────────────────────────────
# IQL Policy Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_iql(iql_ckpt_path: str, z_dim: int, u_dim: int,
             device: str) -> Tuple[QNetwork, QNetwork, VNetwork, GaussianPolicy]:
    cfg = IQLConfig()
    Q1 = QNetwork(z_dim, u_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    Q2 = QNetwork(z_dim, u_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    V  = VNetwork(z_dim, cfg.hidden_dim, cfg.n_layers).to(device)
    pi = GaussianPolicy(z_dim, u_dim, cfg.hidden_dim, cfg.n_layers).to(device)

    ckpt = torch.load(iql_ckpt_path, map_location=device)
    Q1.load_state_dict(ckpt['Q1'])
    Q2.load_state_dict(ckpt['Q2'])
    V.load_state_dict(ckpt['V'])
    pi.load_state_dict(ckpt['pi'])
    Q1.eval(); Q2.eval(); V.eval(); pi.eval()
    print(f"Loaded IQL: {iql_ckpt_path}  step={ckpt.get('step','?')}")
    return Q1, Q2, V, pi


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curve 재분석
# ─────────────────────────────────────────────────────────────────────────────

def analyze_training_log(log_path: str, out_dir: str):
    """
    train.log 파싱 → 세부 진단 플롯.
    Q/V gap, advantage 분포, reward target 추이.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    steps, q_loss, v_loss, pi_loss = [], [], [], []
    q_mean, v_mean, adv_mean, r_target = [], [], [], []

    with open(log_path) as f:
        for line in f:
            if not line.startswith('Step'):
                continue
            try:
                parts = line.split('|')
                step = int(parts[0].split()[1])
                # Q=x.xxxx  V=x.xxxx  π=x.xxxx
                losses = parts[1].strip().split()
                ql  = float(losses[0].split('=')[1])
                vl  = float(losses[1].split('=')[1])
                pil = float(losses[2].split('=')[1])
                # q_μ=  v_μ=  adv_μ=
                stats = parts[2].strip().split()
                qm  = float(stats[0].split('=')[1])
                vm  = float(stats[1].split('=')[1])
                am  = float(stats[2].split('=')[1])
                # r_target=
                rt  = float(parts[3].strip().split('=')[1].split()[0])

                steps.append(step); q_loss.append(ql); v_loss.append(vl)
                pi_loss.append(pil); q_mean.append(qm); v_mean.append(vm)
                adv_mean.append(am); r_target.append(rt)
            except Exception:
                continue

    if not steps:
        print(f"  No parseable log lines in {log_path}")
        return

    steps = np.array(steps)

    def smooth(x, w=20):
        k = np.ones(w) / w
        return np.convolve(x, k, mode='valid')

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    datasets = [
        (q_loss,   'Q Loss',         '#E53935'),
        (v_loss,   'V Loss',         '#1E88E5'),
        (pi_loss,  'π Loss',         '#43A047'),
        (q_mean,   'Q mean',         '#FB8C00'),
        (v_mean,   'V mean',         '#8E24AA'),
        (adv_mean, 'Advantage mean', '#00ACC1'),
        (r_target, 'TD Target mean', '#FFB300'),
    ]

    for i, (vals, title, col) in enumerate(datasets):
        ax = axes[i]
        vals = np.array(vals)
        ax.plot(steps, vals, color=col, alpha=0.2, lw=0.8)
        if len(vals) >= 20:
            ax.plot(steps[19:], smooth(vals), color=col, lw=2.0)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('step', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    # Q-V gap (진단)
    ax = axes[7]
    qv_gap = np.array(q_mean) - np.array(v_mean)
    ax.plot(steps, qv_gap, color='#607D8B', alpha=0.3, lw=0.8)
    if len(qv_gap) >= 20:
        ax.plot(steps[19:], smooth(qv_gap), color='#607D8B', lw=2.0)
    ax.axhline(0, color='k', ls='--', lw=1.0, alpha=0.5)
    ax.set_title('Q - V gap\n(>0 → policy improves over BC)', fontsize=8, fontweight='bold')
    ax.set_xlabel('step', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('IQL + H-step TD — Detailed Training Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = f"{out_dir}/training_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # 진단 출력
    print(f"\n=== Training Diagnosis ===")
    print(f"  Final Q loss:    {q_loss[-1]:.4f}  (converged: {abs(q_loss[-1]-q_loss[-1]):.4f})")
    print(f"  Final V loss:    {v_loss[-1]:.5f}")
    print(f"  Final π loss:    {pi_loss[-1]:.4f}")
    print(f"  Q-V gap (final): {q_mean[-1]-v_mean[-1]:.4f}  "
          f"({'policy > BC' if q_mean[-1]>v_mean[-1] else 'BC-like'})")
    print(f"  Adv mean (final):{adv_mean[-1]:.4f}  "
          f"({'positive → AWR pushes policy' if adv_mean[-1]>0 else 'negative → AWR suppresses OOD'})")
    print(f"  TD target mean:  {r_target[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. World-Model Rollout 비교
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_world_model(
    model:   KoopmanCVAE,
    planner: KODAQLQRPlanner,
    pi:      GaussianPolicy,
    episode: Dict,
    x_seq_full: np.ndarray,
    device:  str,
    cond_len: int = 16,
    horizon:  int = 64,
) -> Dict:
    """
    동일한 초기 context에서 세 가지 action source로 world model rollout:
      A. Real action (데이터)
      B. LQR action (planner)
      C. Policy action (π)

    반환: 각 rollout의 x_hat 시퀀스, reward 예측, Δq RMSE
    """
    dev    = torch.device(device)
    L      = episode['length']
    obs_ep = episode['obs']
    acts_ep= episode['actions']
    s_t    = episode['start_t']
    x_ep   = x_seq_full[s_t:s_t + L]

    # Context encoding
    x_cond = torch.FloatTensor(x_ep[:cond_len]).unsqueeze(0).to(dev)
    a_cond = torch.FloatTensor(acts_ep[:cond_len]).unsqueeze(0).to(dev)
    enc    = model.encode_sequence(x_cond, a_cond)
    z0     = enc['o_seq'][0, -1:]   # (1, m)
    h0     = enc['h_seq'][0, -1:]   # (1, d_h)

    # Sub-goal: 첫 번째 task completion
    gi       = episode['goal_info']
    jump_ts  = sorted(gi['completions'].values())
    goal_t   = jump_ts[0] if jump_ts else L - 1
    goal_obs = obs_ep[goal_t]
    x_goal_t = torch.FloatTensor(
        obs_to_x_goal(goal_obs, obs_ep[0])
    ).unsqueeze(0).to(dev)

    H = min(horizon, goal_t - cond_len, L - cond_len)
    if H < 4:
        return {}

    results = {}

    # ── A. Real action rollout ────────────────────────────────────────────
    z_cur, h_cur = z0.clone(), h0.clone()
    x_hats_real, r_hats_real = [], []
    for t in range(H):
        real_t = cond_len + t
        if real_t >= L:
            break
        a_t = torch.FloatTensor(acts_ep[real_t]).unsqueeze(0).to(dev)
        u_t = model.action_encoder(a_t)
        w_t = model.skill_prior.soft_weights(h_cur)
        log_lam = model.koopman.get_log_lambdas()
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, model.koopman.theta_k, model.koopman.G_k,
            model.koopman.U, w_t)
        A_bar = A_bar[0]; B_bar = B_bar[0]
        z_next = (A_bar @ z_cur.T).T + (B_bar @ u_t.T).T
        recon  = model.decoder(z_next)
        x_hat  = torch.cat([symexp(recon['delta_e']), symexp(recon['delta_p']),
                             symexp(recon['q']),       symexp(recon['qdot'])],
                            dim=-1).cpu().numpy()[0]
        r_hat  = torch.sigmoid(recon['reward']).item() if 'reward' in recon else 0.0
        x_hats_real.append(x_hat); r_hats_real.append(r_hat)
        h_cur = model.recurrent(h_cur, z_cur, a_t)
        z_cur = z_next
    results['real'] = {
        'x_hats': np.array(x_hats_real),
        'r_hats': np.array(r_hats_real),
    }

    # ── B. LQR rollout ───────────────────────────────────────────────────
    try:
        plan = planner.plan(x_cond, a_cond, x_goal_t,
                            horizon=H, compute_uncertainty=False)
        o_traj = plan['o_traj']   # (H+1, m)
        recon_seq = model.decoder(o_traj[1:].to(dev))
        x_hats_lqr = torch.cat([
            symexp(recon_seq['delta_e']), symexp(recon_seq['delta_p']),
            symexp(recon_seq['q']),       symexp(recon_seq['qdot']),
        ], dim=-1).cpu().numpy()   # (H, 2108)
        r_hats_lqr = (torch.sigmoid(recon_seq['reward']).squeeze(-1).cpu().numpy()
                      if 'reward' in recon_seq else np.zeros(H))
        results['lqr'] = {
            'x_hats': x_hats_lqr,
            'r_hats': r_hats_lqr,
            'u_traj': plan['u_traj'].cpu().numpy(),
        }
    except Exception as e:
        print(f"    LQR rollout failed: {e}")

    # ── C. Policy rollout ────────────────────────────────────────────────
    z_cur, h_cur = z0.clone(), h0.clone()
    x_hats_pi, r_hats_pi, u_traj_pi = [], [], []
    for _ in range(H):
        u_t = pi.sample(z_cur)   # (1, d_u)
        w_t = model.skill_prior.soft_weights(h_cur)
        log_lam = model.koopman.get_log_lambdas()
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, model.koopman.theta_k, model.koopman.G_k,
            model.koopman.U, w_t)
        A_bar = A_bar[0]; B_bar = B_bar[0]
        z_next = (A_bar @ z_cur.T).T + (B_bar @ u_t.T).T
        recon  = model.decoder(z_next)
        x_hat  = torch.cat([symexp(recon['delta_e']), symexp(recon['delta_p']),
                             symexp(recon['q']),       symexp(recon['qdot'])],
                            dim=-1).cpu().numpy()[0]
        r_hat  = torch.sigmoid(recon['reward']).item() if 'reward' in recon else 0.0
        x_hats_pi.append(x_hat); r_hats_pi.append(r_hat)
        u_traj_pi.append(u_t.cpu().numpy()[0])
        a_dec  = planner._decode_action(u_t)
        h_cur  = model.recurrent(h_cur, z_cur, a_dec)
        z_cur  = z_next
    results['policy'] = {
        'x_hats': np.array(x_hats_pi),
        'r_hats': np.array(r_hats_pi),
        'u_traj': np.array(u_traj_pi),
    }

    # RMSE vs true
    true_x = x_ep[cond_len:cond_len + H]
    for key in results:
        xh = results[key]['x_hats']
        Hc = min(len(xh), len(true_x))
        dq_rmse = float(np.sqrt(((
            xh[:Hc, X_DQ_START:X_DQ_END] -
            true_x[:Hc, X_DQ_START:X_DQ_END])**2).mean()))
        dp_rmse = float(np.sqrt(((
            xh[:Hc, X_DP_START:X_DP_END] -
            true_x[:Hc, X_DP_START:X_DP_END])**2).mean()))
        results[key]['rmse_dq'] = dq_rmse
        results[key]['rmse_dp'] = dp_rmse

    results['true_x'] = true_x
    results['H']      = H
    results['goal_t'] = goal_t
    results['tasks']  = episode['tasks']
    return results


def visualize_rollout_comparison(
    all_results: List[Dict],
    ep_labels:   List[str],
    out_path:    str,
    split_name:  str = '',
):
    """
    세 rollout source (real / LQR / policy)의 Δq, Δp, reward 비교.
    """
    n_ep = len(all_results)
    if n_ep == 0:
        return

    fig, axes = plt.subplots(n_ep, 3, figsize=(18, 4 * n_ep), squeeze=False)
    colors = {'real': PAL[0], 'lqr': PAL[1], 'policy': PAL[2]}
    styles = {'real': '-', 'lqr': '--', 'policy': '-.'}

    for ep_i, (res, label) in enumerate(zip(all_results, ep_labels)):
        if not res:
            continue
        true_x = res['true_x']
        H      = res['H']
        ts     = np.arange(H)

        # Δq mean
        ax = axes[ep_i, 0]
        ax.plot(ts[:len(true_x)],
                true_x[:H, X_DQ_START:X_DQ_END].mean(1),
                color='k', lw=2.0, label='true', zorder=5)
        for src in ['real', 'lqr', 'policy']:
            if src not in res:
                continue
            xh = res[src]['x_hats']
            Hc = min(len(xh), H)
            ax.plot(ts[:Hc], xh[:Hc, X_DQ_START:X_DQ_END].mean(1),
                    color=colors[src], ls=styles[src], lw=1.5,
                    label=f"{src} (RMSE={res[src]['rmse_dq']:.4f})")
        ax.set_title(f"{label}\nΔq mean", fontsize=8, fontweight='bold')
        ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

        # Δp mean
        ax = axes[ep_i, 1]
        ax.plot(ts[:len(true_x)],
                true_x[:H, X_DP_START:X_DP_END].mean(1),
                color='k', lw=2.0, label='true', zorder=5)
        for src in ['real', 'lqr', 'policy']:
            if src not in res:
                continue
            xh = res[src]['x_hats']
            Hc = min(len(xh), H)
            ax.plot(ts[:Hc], xh[:Hc, X_DP_START:X_DP_END].mean(1),
                    color=colors[src], ls=styles[src], lw=1.5,
                    label=f"{src} (RMSE={res[src]['rmse_dp']:.4f})")
        ax.set_title("Δp mean", fontsize=8, fontweight='bold')
        ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

        # Reward predictions
        ax = axes[ep_i, 2]
        for src in ['real', 'lqr', 'policy']:
            if src not in res:
                continue
            rh = res[src]['r_hats']
            ax.plot(ts[:len(rh)], rh,
                    color=colors[src], ls=styles[src], lw=1.5,
                    label=f"{src} (sum={rh.sum():.3f})")
        ax.set_title("Reward head predictions", fontsize=8, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'World-Model Rollout Comparison  [{split_name}]\n'
        f'blue=real  red=LQR  green=policy  black=true',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_rollout_summary(all_results: List[Dict], split_name: str):
    src_rmse = {'real': [], 'lqr': [], 'policy': []}
    src_rew  = {'real': [], 'lqr': [], 'policy': []}
    for res in all_results:
        if not res:
            continue
        for src in ['real', 'lqr', 'policy']:
            if src in res:
                src_rmse[src].append(res[src]['rmse_dq'])
                src_rew[src].append(res[src]['r_hats'].sum())

    print(f"\n=== Rollout Summary [{split_name}] ===")
    for src in ['real', 'lqr', 'policy']:
        if src_rmse[src]:
            print(f"  {src:7s}  RMSE_Δq={np.mean(src_rmse[src]):.4f}±{np.std(src_rmse[src]):.4f}"
                  f"  reward_sum={np.mean(src_rew[src]):.4f}±{np.std(src_rew[src]):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GIF Generation
# ─────────────────────────────────────────────────────────────────────────────

def save_gif(frames: List[np.ndarray], path: str, fps: int = 12):
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)
        print(f"  GIF: {path}  ({len(frames)} frames @ {fps}fps)")
    except ImportError:
        n = min(len(frames), 16)
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        step = max(1, len(frames) // n)
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[min(i*step, len(frames)-1)])
            ax.axis('off')
        plt.tight_layout()
        strip = path.replace('.gif', '_strip.png')
        plt.savefig(strip, dpi=80); plt.close()
        print(f"  Strip: {strip}")


def render_frame(sim) -> np.ndarray:
    f = sim.render(512, 512, camera_id=-1)
    return f[192:320, 192:320]


@torch.enable_grad()
def decode_action_batch(model, u_batch: torch.Tensor,
                        n_steps: int = 50, lr: float = 0.05) -> np.ndarray:
    N, d_u = u_batch.shape
    da  = model.cfg.action_dim
    dev = u_batch.device
    a   = torch.zeros(N, da, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        F.mse_loss(model.action_encoder(a), u_batch.detach()).backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-1., 1.)
    return a.detach().cpu().numpy()


def make_three_way_gif(
    model:    KoopmanCVAE,
    planner:  KODAQLQRPlanner,
    pi:       GaussianPolicy,
    episode:  Dict,
    x_seq_full: np.ndarray,
    env_name: str,
    out_dir:  str,
    split:    str,
    cond_len: int = 16,
    horizon:  int = 64,
    fps:      int = 10,
    device:   str = 'cuda',
):
    """
    한 에피소드에 대해 real / LQR / policy 3-way GIF 생성.
    """
    import gym, d4rl
    dev    = torch.device(device)
    L      = episode['length']
    obs_ep = episode['obs']
    acts_ep= episode['actions']
    s_t    = episode['start_t']
    x_ep   = x_seq_full[s_t:s_t + L]
    tasks  = episode['tasks']
    gi     = episode['goal_info']

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # sub-goal
    jump_ts = sorted(gi['completions'].values())
    goal_t  = jump_ts[0] if jump_ts else min(cond_len + horizon, L - 1)
    H_gif   = min(horizon, goal_t - cond_len, L - cond_len)
    if H_gif < 4:
        print(f"  Too short for GIF: H={H_gif}")
        return

    print(f"  {split} ep_s={s_t}  tasks={tasks}  "
          f"goal_t={goal_t}  H_gif={H_gif}")

    env = gym.make(env_name)

    # ── A. Real ────────────────────────────────────────────────────────────
    env.reset()
    for t in range(cond_len):
        env.step(acts_ep[t])
    frames_real = [render_frame(env.unwrapped.sim)]
    for t in range(H_gif):
        env.step(acts_ep[cond_len + t])
        frames_real.append(render_frame(env.unwrapped.sim))
    save_gif(frames_real,
             f"{out_dir}/{split}_ep{s_t}_A_real.gif", fps=fps)

    # ── B. LQR ─────────────────────────────────────────────────────────────
    x_cond  = torch.FloatTensor(x_ep[:cond_len]).unsqueeze(0).to(dev)
    a_cond  = torch.FloatTensor(acts_ep[:cond_len]).unsqueeze(0).to(dev)
    x_goal_t= torch.FloatTensor(
        obs_to_x_goal(obs_ep[goal_t], obs_ep[0])
    ).unsqueeze(0).to(dev)

    try:
        with torch.no_grad():
            plan = planner.plan(x_cond, a_cond, x_goal_t,
                                horizon=H_gif, compute_uncertainty=False)
        u_lqr  = plan['u_traj'].to(dev)          # (H, d_u)
        a_lqr  = decode_action_batch(model, u_lqr)  # (H, 9)

        env.reset()
        for t in range(cond_len):
            env.step(acts_ep[t])
        frames_lqr = [render_frame(env.unwrapped.sim)]
        for a in a_lqr:
            env.step(a)
            frames_lqr.append(render_frame(env.unwrapped.sim))
        save_gif(frames_lqr,
                 f"{out_dir}/{split}_ep{s_t}_B_lqr.gif", fps=fps)
    except Exception as e:
        print(f"    LQR GIF failed: {e}")

    # ── C. Policy ──────────────────────────────────────────────────────────
    # Policy rollout: z → u → decode action → sim
    with torch.no_grad():
        enc   = model.encode_sequence(x_cond, a_cond)
        z_cur = enc['o_seq'][0, -1:]   # (1, m)
        h_cur = enc['h_seq'][0, -1:]   # (1, d_h)

    u_pi_list = []
    for _ in range(H_gif):
        with torch.no_grad():
            u_t = pi.sample(z_cur)   # (1, d_u)
            w_t = model.skill_prior.soft_weights(h_cur)
            log_lam = model.koopman.get_log_lambdas()
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, model.koopman.theta_k,
                model.koopman.G_k, model.koopman.U, w_t)
            A_bar = A_bar[0]; B_bar = B_bar[0]
            z_next = (A_bar @ z_cur.T).T + (B_bar @ u_t.T).T
            a_dec  = planner._decode_action(u_t)
            h_cur  = model.recurrent(h_cur, z_cur, a_dec)
            z_cur  = z_next
        u_pi_list.append(u_t.cpu())

    u_pi_batch = torch.cat(u_pi_list, dim=0).to(dev)   # (H, d_u)
    a_pi       = decode_action_batch(model, u_pi_batch)  # (H, 9)

    env.reset()
    for t in range(cond_len):
        env.step(acts_ep[t])
    frames_pi = [render_frame(env.unwrapped.sim)]
    for a in a_pi:
        env.step(a)
        frames_pi.append(render_frame(env.unwrapped.sim))
    save_gif(frames_pi,
             f"{out_dir}/{split}_ep{s_t}_C_policy.gif", fps=fps)

    env.close()
    print(f"  Done: {split} ep{s_t}  "
          f"real={len(frames_real)} lqr={len(u_lqr)} policy={len(frames_pi)} frames")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Q/V Value Landscape 분석
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_value_landscape(
    Q1: QNetwork, Q2: QNetwork, V: VNetwork, pi: GaussianPolicy,
    cache: Dict[str, np.ndarray],
    device: str,
    out_path: str,
    n_samples: int = 2000,
):
    """
    학습된 Q, V 함수의 분포 분석:
      - Q(z, u_real) vs Q(z, u_policy) vs Q(z, u_random) 비교
      - Advantage 분포
      - V vs Q 상관관계
    """
    dev = device
    idx = np.random.choice(len(cache['z_real']), min(n_samples, len(cache['z_real'])),
                           replace=False)
    z  = torch.FloatTensor(cache['z_real'][idx]).to(dev)
    u_r= torch.FloatTensor(cache['u_real'][idx]).to(dev)

    u_dim = u_r.shape[1]

    # Policy action
    u_pi  = pi.sample(z)

    # Random action (OOD baseline)
    u_rand= torch.randn_like(u_r) * u_r.std(0) + u_r.mean(0)

    q_real  = torch.min(Q1(z, u_r),    Q2(z, u_r)).cpu().numpy()
    q_pi    = torch.min(Q1(z, u_pi),   Q2(z, u_pi)).cpu().numpy()
    q_rand  = torch.min(Q1(z, u_rand), Q2(z, u_rand)).cpu().numpy()
    v_vals  = V(z).cpu().numpy()

    adv_real = q_real - v_vals
    adv_pi   = q_pi   - v_vals
    adv_rand = q_rand - v_vals

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Q 분포 비교
    ax = axes[0, 0]
    bins = np.linspace(min(q_real.min(), q_pi.min(), q_rand.min()),
                       max(q_real.max(), q_pi.max(), q_rand.max()), 50)
    ax.hist(q_real, bins=bins, alpha=0.5, color=PAL[0], label=f'real μ={q_real.mean():.3f}')
    ax.hist(q_pi,   bins=bins, alpha=0.5, color=PAL[2], label=f'policy μ={q_pi.mean():.3f}')
    ax.hist(q_rand, bins=bins, alpha=0.5, color='gray',  label=f'random μ={q_rand.mean():.3f}')
    ax.set_title('Q distribution\n(real vs policy vs random)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    # Advantage 분포
    ax = axes[0, 1]
    ax.hist(adv_real, bins=40, alpha=0.6, color=PAL[0], label=f'real μ={adv_real.mean():.4f}')
    ax.hist(adv_pi,   bins=40, alpha=0.6, color=PAL[2], label=f'policy μ={adv_pi.mean():.4f}')
    ax.axvline(0, color='k', ls='--', lw=1.5)
    ax.set_title('Advantage = Q - V\n(>0 → better than average)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    # Q vs V scatter
    ax = axes[0, 2]
    ax.scatter(v_vals[:200], q_real[:200], s=8, alpha=0.5, color=PAL[0], label='real')
    ax.scatter(v_vals[:200], q_pi[:200],   s=8, alpha=0.5, color=PAL[2], label='policy')
    lim = [min(v_vals.min(), q_real.min()), max(v_vals.max(), q_real.max())]
    ax.plot(lim, lim, 'k--', lw=1.0, alpha=0.5, label='Q=V')
    ax.set_xlabel('V(z)', fontsize=8); ax.set_ylabel('Q(z,u)', fontsize=8)
    ax.set_title('Q vs V\n(above diagonal → positive advantage)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    # AWR weight 분포
    from iql_koopman import IQLConfig
    beta = IQLConfig().beta
    w_real = np.exp(beta * adv_real).clip(0, 100)
    w_pi   = np.exp(beta * adv_pi).clip(0, 100)
    ax = axes[1, 0]
    ax.hist(w_real, bins=40, alpha=0.6, color=PAL[0],
            label=f'real  μ={w_real.mean():.3f}')
    ax.hist(w_pi,   bins=40, alpha=0.6, color=PAL[2],
            label=f'policy μ={w_pi.mean():.3f}')
    ax.set_title(f'AWR weights exp(β={beta}·A)\n(higher → policy update stronger)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    # Q_real - Q_random (OOD gap)
    ax = axes[1, 1]
    gap = q_real - q_rand
    ax.hist(gap, bins=40, color='#607D8B', alpha=0.7)
    ax.axvline(gap.mean(), color='k', ls='--', lw=1.5,
               label=f'mean={gap.mean():.4f}')
    ax.axvline(0, color='r', ls=':', lw=1.0, alpha=0.7)
    ax.set_title('Q(real) - Q(random)\n(>0 → OOD suppressed correctly)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    # Q_policy - Q_random (policy > random 여부)
    ax = axes[1, 2]
    gap2 = q_pi - q_rand
    ax.hist(gap2, bins=40, color=PAL[2], alpha=0.7)
    ax.axvline(gap2.mean(), color='k', ls='--', lw=1.5,
               label=f'mean={gap2.mean():.4f}')
    ax.axvline(0, color='r', ls=':', lw=1.0, alpha=0.7)
    ax.set_title('Q(policy) - Q(random)\n(>0 → policy better than random)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.spines[['top','right']].set_visible(False)

    fig.suptitle('IQL Value Landscape Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n=== Value Landscape ===")
    print(f"  Q(real):   μ={q_real.mean():.4f}  σ={q_real.std():.4f}")
    print(f"  Q(policy): μ={q_pi.mean():.4f}  σ={q_pi.std():.4f}")
    print(f"  Q(random): μ={q_rand.mean():.4f}  σ={q_rand.std():.4f}")
    print(f"  Q(real)-Q(rand) mean: {(q_real-q_rand).mean():.4f}  "
          f"({'✓ OOD suppressed' if (q_real-q_rand).mean()>0 else '✗ OOD not suppressed'})")
    print(f"  Q(pi)-Q(rand)  mean: {(q_pi-q_rand).mean():.4f}  "
          f"({'✓ policy > random' if (q_pi-q_rand).mean()>0 else '✗ policy ≤ random'})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_ckpt', default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--iql_ckpt',   default='checkpoints/kodaq_v4/iql/iql_final.pt')
    p.add_argument('--lqr_cache',  default='checkpoints/kodaq_v4/iql/lqr_cache.npz')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--log_path',   default='checkpoints/kodaq_v4/iql/train.log')
    p.add_argument('--env',        default='kitchen-mixed-v0')
    p.add_argument('--quality',    default='mixed')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v4/eval')
    p.add_argument('--n_train_ep', type=int, default=5,
                   help='학습에 사용한 에피소드 수 (앞에서)')
    p.add_argument('--n_heldout_ep', type=int, default=5,
                   help='학습에 사용 안한 에피소드 (뒤에서)')
    p.add_argument('--cond_len',   type=int, default=16)
    p.add_argument('--horizon',    type=int, default=48)
    p.add_argument('--skip_gif',   action='store_true')
    p.add_argument('--n_gif_ep',   type=int, default=2)
    p.add_argument('--fps',        type=int, default=10)
    p.add_argument('--Q_scale',    type=float, default=1.0)
    p.add_argument('--R_scale',    type=float, default=10.0)
    p.add_argument('--device',     default='cuda:1' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"Device: {device}")

    # ── World model ─────────────────────────────────────────────────────────
    print(f"\nLoading world model: {args.world_ckpt}")
    ckpt  = torch.load(args.world_ckpt, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    z_dim = model.cfg.koopman_dim
    u_dim = model.cfg.action_latent

    lqr_cfg = LQRConfig(Q_scale=args.Q_scale, R_scale=args.R_scale)
    planner = KODAQLQRPlanner(model, lqr_cfg)

    # ── IQL policy ─────────────────────────────────────────────────────────
    Q1, Q2, V, pi = load_iql(args.iql_ckpt, z_dim, u_dim, device)

    # ── Data ───────────────────────────────────────────────────────────────
    print(f"\nLoading data...")
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality, min_len=32)
    valid_eps        = [ep for ep in episodes if ep['tasks']]
    print(f"  Total episodes with tasks: {len(valid_eps)}")

    # Train / held-out split
    # lqr_cache 생성 시 앞 n_ep_lqr 에피소드 사용 → 뒤쪽이 held-out
    cache = dict(np.load(args.lqr_cache))
    n_lqr_used = 500   # iql_koopman.py --n_ep_lqr 기본값

    train_eps   = valid_eps[:min(args.n_train_ep, n_lqr_used)]
    heldout_eps = valid_eps[n_lqr_used:n_lqr_used + args.n_heldout_ep]
    if not heldout_eps:
        heldout_eps = valid_eps[-args.n_heldout_ep:]
    print(f"  Train sample: {len(train_eps)}, Held-out: {len(heldout_eps)}")

    # ── 1. Training log 분석 ─────────────────────────────────────────────
    if Path(args.log_path).exists():
        print(f"\n[1] Analyzing training log...")
        analyze_training_log(args.log_path, args.out_dir)

    # ── 2. Value landscape 분석 ──────────────────────────────────────────
    print(f"\n[2] Analyzing value landscape...")
    analyze_value_landscape(
        Q1, Q2, V, pi, cache, device,
        out_path=f"{args.out_dir}/value_landscape.png",
    )

    # ── 3. World-model rollout 비교 ──────────────────────────────────────
    for split, eps in [('train', train_eps), ('heldout', heldout_eps)]:
        print(f"\n[3] World-model rollout comparison [{split}]...")
        all_res, labels = [], []
        for ep_i, ep in enumerate(eps):
            print(f"  Ep {ep_i}  tasks={ep['tasks']}  len={ep['length']}")
            res = rollout_world_model(
                model, planner, pi, ep, x_seq_full,
                device=device,
                cond_len=args.cond_len,
                horizon=args.horizon,
            )
            all_res.append(res)
            labels.append(f"Ep{ep_i}({ep['tasks'][0] if ep['tasks'] else '?'})")

        print_rollout_summary(all_res, split)
        visualize_rollout_comparison(
            all_res, labels,
            out_path=f"{args.out_dir}/rollout_{split}.png",
            split_name=split,
        )

    # ── 4. GIF 생성 ──────────────────────────────────────────────────────
    if not args.skip_gif:
        for split, eps in [('train', train_eps[:args.n_gif_ep]),
                            ('heldout', heldout_eps[:args.n_gif_ep])]:
            print(f"\n[4] Generating GIFs [{split}]...")
            for ep in eps:
                make_three_way_gif(
                    model=model, planner=planner, pi=pi,
                    episode=ep, x_seq_full=x_seq_full,
                    env_name=args.env,
                    out_dir=f"{args.out_dir}/gif",
                    split=split,
                    cond_len=args.cond_len,
                    horizon=args.horizon,
                    fps=args.fps, device=device,
                )

    print(f"\n{'='*55}")
    print(f"Evaluation complete. → {args.out_dir}/")
    print(f"  training_analysis.png   — log 파싱 진단")
    print(f"  value_landscape.png     — Q/V/adv 분포")
    print(f"  rollout_train.png       — 학습 에피소드 rollout 비교")
    print(f"  rollout_heldout.png     — held-out 에피소드 rollout 비교")
    print(f"  gif/                    — real/LQR/policy 3-way GIF")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()