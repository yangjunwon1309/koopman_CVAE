"""
lqr_planner.py — KODAQ LQR Trajectory Planner (Real Data)
===========================================================

D4RL Kitchen task notation (D4RL kitchen_envs.py):
  OBS_ELEMENT_INDICES: 60-dim obs에서 각 task object 인덱스 (obj offset=9)
  OBS_ELEMENT_GOALS:   각 task의 goal joint value
  BONUS_THRESH=0.3:    completion threshold

  dataset['infos/tasks_to_complete']: (N, 4) string array
    → 각 timestep에 남은 task 목록 → task completion timestep 식별

Goal 식별:
  1. final_goal:    에피소드 마지막 obs (전체 완료 상태)
  2. subtask_goals: task별 completion timestep obs (중간 goal)

LQR:
  o_{t+1} = Ā(w_t)·o_t + B̄(w_t)·u_t
  u_t*    = -Σ_k w_k·L_k·(o_t - o*)
  L_k     = DARE(A_k, B_k, Q, R)

Uncertainty:
  τ_d = real(0:d) ⊕ LQR(d:d+H)
  U   = mean_t Var_d[o_t^(τ_d)]
"""

import os, sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from scipy.linalg import solve_discrete_are
from typing import Dict, List, Optional, Tuple

from models.koopman_cvae import KoopmanCVAE
from models.losses import symlog, symexp, blend_koopman
from data.extract_skill_label import load_x_sequences


# ─────────────────────────────────────────────────────────────────────────────
# D4RL Kitchen constants (from kitchen_envs.py)
# obs[0:9]  = robot qpos
# obs[9:60] = obj_qp (51-dim)
# OBS_ELEMENT_INDICES below are in raw 60-dim obs space (obj offset +9 applied)
# ─────────────────────────────────────────────────────────────────────────────

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]) + 9,
    'top burner':    np.array([15, 16]) + 9,
    'light switch':  np.array([17, 18]) + 9,
    'slide cabinet': np.array([19])     + 9,
    'hinge cabinet': np.array([20, 21]) + 9,
    'microwave':     np.array([22])     + 9,
    'kettle':        np.array([23, 24, 25, 26, 27, 28, 29]) + 9,
}

OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner':    np.array([-0.92, -0.01]),
    'light switch':  np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0.,   1.45]),
    'microwave':     np.array([-0.75]),
    'kettle':        np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
}

BONUS_THRESH = 0.3
ALL_TASKS    = list(OBS_ELEMENT_INDICES.keys())

# x_t = [Δe(2048) | Δp(42) | Δq(9) | q̇(9)]
X_DE_START = 0;    X_DE_END = 2048
X_DP_START = 2048; X_DP_END = 2090
X_DQ_START = 2090; X_DQ_END = 2099
X_QD_START = 2099; X_QD_END = 2108


# ─────────────────────────────────────────────────────────────────────────────
# Task completion detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_task_completions(obs_ep: np.ndarray, tasks: List[str]) -> Dict[str, int]:
    """
    각 task의 첫 completion timestep 반환.
    D4RL 기준: ||obs[element_idx] - goal|| < BONUS_THRESH
    Returns {task: timestep}  (-1 = not completed)
    """
    out = {}
    for task in tasks:
        idx  = OBS_ELEMENT_INDICES.get(task)
        goal = OBS_ELEMENT_GOALS.get(task)
        if idx is None:
            out[task] = -1
            continue
        found = -1
        for t in range(len(obs_ep)):
            if np.linalg.norm(obs_ep[t, idx] - goal) < BONUS_THRESH:
                found = t
                break
        out[task] = found
    return out


def identify_episode_goals(obs_ep: np.ndarray, tasks: List[str]) -> Dict:
    """
    에피소드에서 goal 식별:
      final_goal:    마지막 obs
      subtask_goals: task별 completion obs (미완료시 OBS_ELEMENT_GOALS로 대체)
      midpoint_goal: L//2 obs
    """
    L           = len(obs_ep)
    completions = detect_task_completions(obs_ep, tasks)
    subtask_goals = {}
    for task in tasks:
        t = completions[task]
        if t >= 0:
            subtask_goals[task] = {'obs': obs_ep[t], 'timestep': t, 'completed': True}
        else:
            # goal 미달성 → OBS_ELEMENT_GOALS로 이상적 goal 구성
            g_obs            = obs_ep[0].copy()
            g_obs[OBS_ELEMENT_INDICES[task]] = OBS_ELEMENT_GOALS[task]
            subtask_goals[task] = {'obs': g_obs, 'timestep': -1, 'completed': False}

    return {
        'final_goal':    obs_ep[-1],
        'midpoint_goal': obs_ep[L // 2],
        'subtask_goals': subtask_goals,
        'completions':   completions,
        'episode_len':   L,
    }


# ─────────────────────────────────────────────────────────────────────────────
# obs → x_goal (2108-dim)
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_x_goal(goal_obs: np.ndarray, ref_obs: np.ndarray) -> np.ndarray:
    """
    x_goal = [Δe=0 | Δp=goal_obj-ref_obj | Δq=goal_q-ref_q | q̇=0]
    """
    delta_e = np.zeros(2048, dtype=np.float32)
    delta_p = (goal_obs[18:60] - ref_obs[18:60]).astype(np.float32)
    delta_q = (goal_obs[0:9]   - ref_obs[0:9]).astype(np.float32)
    qdot    = np.zeros(9, dtype=np.float32)
    x_goal  = np.concatenate([delta_e, delta_p, delta_q, qdot])
    assert x_goal.shape[0] == 2108
    return x_goal


# ─────────────────────────────────────────────────────────────────────────────
# LQR Config + DARE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LQRConfig:
    Q_scale:       float = 1.0
    R_scale:       float = 0.1
    eps_A:         float = 0.01
    T_replan:      int   = 8
    dare_max_iter: int   = 300
    dare_tol:      float = 1e-8
    lambda_unc:    float = 0.1
    u_max:         float = 1.0
    action_inv_steps: int = 30


def solve_dare_safe(A, B, Q, R, max_iter=300, tol=1e-8):
    """
    DARE → (P, L, M)

    L = (R + BᵀPB)⁻¹ BᵀPA   feedback gain
    M = (R + BᵀPB)⁻¹ BᵀP    feedforward gain

    Optimal control (z* 평형점 가정 없음):
        u_t* = M·z* - L·z_t
    """
    try:
        P = solve_discrete_are(A, B, Q, R)
    except Exception:
        P = Q.copy()
        for _ in range(max_iter):
            BtP   = B.T @ P
            P_new = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A) + Q
            if np.max(np.abs(P_new - P)) < tol:
                P = P_new; break
            P = P_new
    BtP   = B.T @ P
    S_inv = np.linalg.inv(R + BtP @ B)   # (d_u, d_u)
    L     = S_inv @ BtP @ A               # (d_u, m) feedback
    M     = S_inv @ BtP                   # (d_u, m) feedforward
    return P, L, M


# ─────────────────────────────────────────────────────────────────────────────
# LQR Planner
# ─────────────────────────────────────────────────────────────────────────────

class KODAQLQRPlanner:
    def __init__(self, model: KoopmanCVAE, cfg: LQRConfig):
        self.model  = model
        self.cfg    = cfg
        self.m_cfg  = model.cfg
        self.device = next(model.parameters()).device
        m, d_u = self.m_cfg.koopman_dim, self.m_cfg.action_latent
        self.Q  = np.eye(m)   * cfg.Q_scale
        self.R  = np.eye(d_u) * cfg.R_scale
        self._cache: Dict = {}

    def _get_gain(self, A: np.ndarray, B: np.ndarray):
        """Cache DARE by (A, B) → (P, L, M)."""
        key = (A.tobytes()[:64], B.tobytes()[:32])
        if key not in self._cache:
            self._cache[key] = solve_dare_safe(
                A, B, self.Q, self.R, self.cfg.dare_max_iter, self.cfg.dare_tol)
        return self._cache[key]

    def _blended(self, w: torch.Tensor):
        """
        Ā(w), B̄(w) 계산 후 단일 DARE(Ā, B̄) → L, M.

        Per-skill L_k 블렌딩 방식 대신:
          1. log-eigenvalue space 보간 → Ā(w), B̄(w)
          2. DARE(Ā, B̄, Q, R) → P, L, M
        → 블렌딩된 dynamics와 gain이 정확히 일치.
        """
        koop    = self.model.koopman
        log_lam = koop.get_log_lambdas()
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, koop.theta_k, koop.G_k, koop.U, w)
        A_bar = A_bar[0]   # (m, m)
        B_bar = B_bar[0]   # (m, d_u)

        A_np = A_bar.detach().cpu().numpy()
        B_np = B_bar.detach().cpu().numpy()
        _, L, M = self._get_gain(A_np, B_np)
        return A_bar, B_bar, L, M

    def _step(self, o, u, A, B):
        return (A @ o.T).T + (B @ u.T).T

    def _lqr_u(self, o, o_star, L, M):
        """
        u_t* = M·z* - L·z_t   (평형점 가정 없는 정확한 형태)

        L = (R+BᵀPB)⁻¹ BᵀPA   feedback
        M = (R+BᵀPB)⁻¹ BᵀP    feedforward
        """
        z_t   = o[0].cpu().numpy()          # (m,)
        z_star = o_star[0].cpu().numpy()    # (m,)
        u_np  = M @ z_star - L @ z_t
        return torch.FloatTensor(
            np.clip(u_np, -self.cfg.u_max, self.cfg.u_max)
        ).unsqueeze(0).to(self.device)

    def _decode_action(self, u: torch.Tensor) -> torch.Tensor:
        da = self.m_cfg.action_dim
        a  = torch.zeros(1, da, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(self.cfg.action_inv_steps):
            opt.zero_grad()
            F.mse_loss(self.model.action_encoder(a), u.detach()).backward()
            opt.step()
            with torch.no_grad(): a.clamp_(-1., 1.)
        return a.detach()

    @torch.no_grad()
    def encode_goal(self, x_goal: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
        if x_goal.dim() == 1: x_goal = x_goal.unsqueeze(0)
        mu, _ = self.model.posterior(x_goal, h_ref)
        return mu

    @torch.no_grad()
    def _lqr_rollout(self, o0, h0, x_goal_t, horizon) -> Dict:
        """
        x_goal_t: (1, x_dim) tensor — goal in observation space.
        z* = μ_φ(x_goal, h_t) 를 T_replan마다 재인코딩 (Option C).
        u_t* = M(w_t)·z*(h_t) - L(w_t)·z_t  (Option A+C).
        """
        model = self.model
        o_list, u_list, a_list, w_list, costs = [o0], [], [], [], []
        o_cur, h_cur = o0, h0

        w_cur               = model.skill_prior.soft_weights(h_cur)
        w_list.append(w_cur)
        A_cur, B_cur, L_cur, M_cur = self._blended(w_cur)
        A_prev_np           = A_cur.detach().cpu().numpy()
        o_star              = self.encode_goal(x_goal_t, h_cur)   # (1, m)

        for t in range(horizon):
            u_t    = self._lqr_u(o_cur, o_star, L_cur, M_cur)
            with torch.enable_grad():
                a_t    = self._decode_action(u_t)
            o_next = self._step(o_cur, u_t, A_cur, B_cur)
            h_next = model.recurrent(h_cur, o_cur, a_t)
            w_next = model.skill_prior.soft_weights(h_next)

            A_next, B_next, L_next, M_next = self._blended(w_next)
            A_next_np = A_next.detach().cpu().numpy()

            # Hybrid replan: Ā 변화 크거나 T_replan 주기
            if (np.linalg.norm(A_next_np - A_prev_np, 'fro') > self.cfg.eps_A
                    or (t + 1) % self.cfg.T_replan == 0):
                A_cur, B_cur, L_cur, M_cur = A_next, B_next, L_next, M_next
                A_prev_np = A_next_np
                # Option C: h_t가 업데이트됐으므로 z* 재인코딩
                o_star = self.encode_goal(x_goal_t, h_next)

            # LQR cost: (z_t - z*)ᵀQ(z_t - z*) + u_tᵀRu_t
            e    = (o_cur - o_star)[0].cpu().numpy()
            u_np = u_t[0].cpu().numpy()
            costs.append(float(e @ self.Q @ e + u_np @ self.R @ u_np))

            o_list.append(o_next); u_list.append(u_t)
            a_list.append(a_t);    w_list.append(w_next)
            o_cur, h_cur, w_cur = o_next, h_next, w_next

        return {
            'o_traj':     torch.cat(o_list, 0),
            'u_traj':     torch.cat(u_list, 0),
            'a_traj':     torch.cat(a_list, 0),
            'w_traj':     torch.cat(w_list, 0),
            'costs':      np.array(costs),
            'total_cost': float(np.sum(costs)),
            'o_star':     o_star,   # 마지막 재인코딩된 z*
        }

    def _decode_x_traj(self, o_traj: torch.Tensor) -> torch.Tensor:
        recon = self.model.decoder(o_traj)
        return torch.cat([symexp(recon['delta_e']), symexp(recon['delta_p']),
                          symexp(recon['q']),       symexp(recon['qdot'])], dim=-1)

    @torch.no_grad()
    def uncertainty_rollout(self, o0, h0, x_goal_t, real_actions, lqr_horizon=32) -> Dict:
        """τ_d = real(0:d) ⊕ LQR(d:d+H). σ²_t = Var_d[o_t^(τ_d)]"""
        model  = self.model
        K_real = real_actions.shape[0]
        base   = self._lqr_rollout(o0, h0, x_goal_t, lqr_horizon)
        trajs  = [base['o_traj']]
        o_cur, h_cur = o0.clone(), h0.clone()

        for d in range(1, K_real + 1):
            a_d    = real_actions[d-1].unsqueeze(0)
            u_d    = model.action_encoder(a_d)
            w_d    = model.skill_prior.soft_weights(h_cur)
            A_d, B_d, _, _ = self._blended(w_d)
            o_next = self._step(o_cur, u_d, A_d, B_d)
            h_next = model.recurrent(h_cur, o_cur, a_d)
            o_cur, h_cur = o_next, h_next

            lqr_d  = self._lqr_rollout(o_cur, h_cur, x_goal_t, lqr_horizon)
            trajs.append(torch.cat([trajs[0][:d], lqr_d['o_traj']], 0))

        T_com   = lqr_horizon + 1
        stacked = torch.stack([t[:T_com] for t in trajs if t.shape[0] >= T_com])
        if stacked.shape[0] > 1:
            sigma2_t    = stacked.var(dim=0).mean(dim=1)
            uncertainty = sigma2_t.mean().item()
        else:
            sigma2_t    = torch.zeros(T_com, device=self.device)
            uncertainty = 0.0

        return {
            'uncertainty':    uncertainty,
            'sigma2_t':       sigma2_t.cpu().numpy(),
            'all_o_trajs':    [t.cpu() for t in trajs],
            'penalized_cost': base['total_cost'] + self.cfg.lambda_unc * uncertainty,
        }

    @torch.no_grad()
    def plan(self, x_cond, a_cond, x_goal, horizon=32,
             real_actions=None, compute_uncertainty=True) -> Dict:
        dev = self.device
        enc      = self.model.encode_sequence(x_cond, a_cond)
        o0       = enc['o_seq'][0, -1:].to(dev)
        h0       = enc['h_seq'][0, -1:].to(dev)
        x_goal_t = x_goal.to(dev)
        if x_goal_t.dim() == 1: x_goal_t = x_goal_t.unsqueeze(0)

        lqr    = self._lqr_rollout(o0, h0, x_goal_t, horizon)
        x_traj = self._decode_x_traj(lqr['o_traj'].to(dev))
        result = {**lqr, 'o0': o0, 'h0': h0, 'x_traj': x_traj}

        if compute_uncertainty and real_actions is not None:
            unc = self.uncertainty_rollout(
                o0, h0, x_goal_t, real_actions.to(dev), horizon)
            result.update(unc)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Real dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_kitchen_episodes(
    quality: str = 'mixed',
    min_len: int = 64,
) -> Tuple[List[Dict], np.ndarray]:
    """
    D4RL Kitchen 로드 + 에피소드 분리 + task completion 탐지.
    infos/tasks_to_complete 키가 있으면 tasks_to_complete로 task 식별,
    없으면 OBS_ELEMENT_GOALS 기반 직접 탐지 사용.
    """
    import d4rl, gym
    name_map = {'mixed': 'kitchen-mixed-v0',
                'partial': 'kitchen-partial-v0',
                'complete': 'kitchen-complete-v0'}
    d4rl_name = name_map[quality]
    print(f"Loading {d4rl_name} ...")
    env     = gym.make(d4rl_name)
    dataset = env.get_dataset()

    obs       = dataset['observations']
    actions   = dataset['actions']
    terminals = dataset['terminals'].astype(bool)
    N         = len(obs)

    # infos 키 탐지
    task_info = None
    for key in ['infos/tasks_to_complete', 'infos/task_completions']:
        if key in dataset:
            task_info = dataset[key]
            print(f"  Found '{key}': shape={np.array(task_info).shape}")
            break
    if task_info is None:
        print("  No task_info key → using OBS_ELEMENT_GOALS-based detection")

    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    episodes  = []

    for ep_s, ep_e in zip(ep_starts, ep_ends):
        L = ep_e - ep_s + 1
        if L < min_len: continue

        obs_ep  = obs[ep_s:ep_e+1]
        acts_ep = actions[ep_s:ep_e+1]

        # task 목록 결정
        if task_info is not None:
            try:
                row = task_info[ep_s]
                tasks = [t for t in row if t and str(t) not in ('', 'None', 'b\'\'')]
            except Exception:
                tasks = ALL_TASKS[:4]
        else:
            tasks = ALL_TASKS

        gi = identify_episode_goals(obs_ep, tasks)
        episodes.append({
            'obs': obs_ep, 'actions': acts_ep,
            'start_t': ep_s, 'end_t': ep_e, 'length': L,
            'tasks': tasks, 'goal_info': gi,
        })

    print(f"Episodes: {len(episodes)} (min_len={min_len})")
    return episodes, obs


def run_lqr_on_episodes(
    planner:      KODAQLQRPlanner,
    episodes:     List[Dict],
    x_seq_full:   np.ndarray,      # (N, 2108) cached x_t
    cond_len:     int = 16,
    horizon:      int = 32,
    unc_real_len: int = 8,
    device:       str = 'cuda',
) -> List[Dict]:
    dev     = torch.device(device)
    results = []

    for ep_idx, ep in enumerate(episodes):
        L, obs_ep, acts_ep = ep['length'], ep['obs'], ep['actions']
        gi, tasks           = ep['goal_info'], ep['tasks']
        s, e                = ep['start_t'], ep['end_t']
        x_ep                = x_seq_full[s:e+1]   # (L, 2108)

        if L < cond_len + horizon + 4:
            continue

        print(f"\nEp {ep_idx}  len={L}  tasks={tasks}")
        print(f"  Completions: {gi['completions']}")

        # Goal 목록: final + 첫 완료 subtask
        goals = [('final', obs_to_x_goal(gi['final_goal'], obs_ep[0]), gi['final_goal'])]
        completed = [(t, ti['timestep']) for t, ti in gi['subtask_goals'].items()
                     if ti['completed'] and ti['timestep'] > cond_len]
        if completed:
            ft, f_t = min(completed, key=lambda x: x[1])
            goals.append((f'sub_{ft}@{f_t}',
                          obs_to_x_goal(gi['subtask_goals'][ft]['obs'], obs_ep[0]),
                          gi['subtask_goals'][ft]['obs']))

        for goal_name, x_goal_np, _ in goals:
            x_cond   = torch.FloatTensor(x_ep[:cond_len]).unsqueeze(0).to(dev)
            a_cond   = torch.FloatTensor(acts_ep[:cond_len]).unsqueeze(0).to(dev)
            x_goal_t = torch.FloatTensor(x_goal_np).unsqueeze(0).to(dev)
            real_a   = torch.FloatTensor(
                acts_ep[cond_len:min(cond_len + unc_real_len, L)]).to(dev)

            plan = planner.plan(x_cond, a_cond, x_goal_t, horizon=horizon,
                                real_actions=real_a, compute_uncertainty=True)

            # RMSE Δq_t
            true_x  = x_ep[cond_len:min(cond_len+horizon, L)]
            H_c     = min(plan['x_traj'].shape[0], len(true_x))
            pred_dq = plan['x_traj'][:H_c, X_DQ_START:X_DQ_END].cpu().numpy()
            true_dq = true_x[:H_c, X_DQ_START:X_DQ_END]
            rmse    = float(np.sqrt(((pred_dq - true_dq)**2).mean()))

            print(f"  [{goal_name}] cost={plan['total_cost']:.4f}  "
                  f"unc={plan['uncertainty']:.6f}  "
                  f"pen={plan['penalized_cost']:.4f}  "
                  f"RMSE_Δq={rmse:.4f}")

            results.append({'ep_idx': ep_idx, 'goal_name': goal_name,
                            'plan': plan, 'true_x': true_x,
                            'rmse_dq': rmse, 'tasks': tasks,
                            'completions': gi['completions']})
    return results


def visualize_lqr_results(results: List[Dict], out_dir: str = 'checkpoints/kodaq/lqr'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i, res in enumerate(results):
        plan    = res['plan']
        true_x  = res['true_x']
        x_traj  = plan['x_traj'].cpu().numpy()
        sig2    = plan.get('sigma2_t', None)
        H       = x_traj.shape[0] - 1

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(plan['costs'], color=PAL[i % len(PAL)], lw=1.5)
        axes[0].set_title(f"LQR Cost  total={plan['total_cost']:.3f}", fontsize=9)
        axes[0].set_xlabel('step')
        axes[0].spines[['top','right']].set_visible(False)

        dq_pred = x_traj[:, X_DQ_START:X_DQ_END].mean(1)
        H_c     = min(H+1, len(true_x))
        dq_true = true_x[:H_c, X_DQ_START:X_DQ_END].mean(1)
        axes[1].plot(dq_pred, '--', color=PAL[i%len(PAL)], lw=1.5, label='pred')
        axes[1].plot(np.arange(H_c), dq_true, 'k-', lw=1.5, label='true')
        axes[1].set_title(f"Δq_t mean  RMSE={res['rmse_dq']:.4f}", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].spines[['top','right']].set_visible(False)

        if sig2 is not None:
            axes[2].fill_between(np.arange(len(sig2)), 0, sig2,
                                 alpha=0.4, color=PAL[i%len(PAL)])
            axes[2].plot(sig2, color=PAL[i%len(PAL)], lw=1.5)
            axes[2].set_title(f"σ²_t  U={plan['uncertainty']:.5f}", fontsize=9)
        axes[2].spines[['top','right']].set_visible(False)

        gname = res['goal_name'].replace('/','_').replace(' ','_').replace('@','_')
        fig.suptitle(f"Ep{res['ep_idx']}  {res['goal_name']}\n"
                     f"Tasks:{res['tasks']}  Pen:{plan['penalized_cost']:.4f}",
                     fontsize=8, fontweight='bold')
        plt.tight_layout()
        out = f"{out_dir}/ep{res['ep_idx']}_{gname}.png"
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='checkpoints/kodaq_v3/final.pt')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--quality',    default='mixed',
                   choices=['mixed','partial','complete'])
    p.add_argument('--n_ep',       type=int,   default=5)
    p.add_argument('--cond_len',   type=int,   default=16)
    p.add_argument('--horizon',    type=int,   default=32)
    p.add_argument('--unc_len',    type=int,   default=8)
    p.add_argument('--Q_scale',    type=float, default=1.0)
    p.add_argument('--R_scale',    type=float, default=0.1)
    p.add_argument('--lambda_unc', type=float, default=0.1)
    p.add_argument('--out_dir',    default='checkpoints/kodaq/lqr')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ckpt  = torch.load(args.ckpt, map_location=args.device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(args.device)
    print(f"Model: K={model.cfg.num_skills}  m={model.cfg.koopman_dim}  "
          f"phase={ckpt.get('phase',3)}")

    planner = KODAQLQRPlanner(model, LQRConfig(
        Q_scale=args.Q_scale, R_scale=args.R_scale, lambda_unc=args.lambda_unc))

    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    print(f"x_seq: {x_seq_full.shape}")

    episodes, _ = load_kitchen_episodes(quality=args.quality,
                                        min_len=args.cond_len + args.horizon + 4)
    episodes = episodes[:args.n_ep]

    results = run_lqr_on_episodes(
        planner, episodes, x_seq_full,
        cond_len=args.cond_len, horizon=args.horizon,
        unc_real_len=args.unc_len, device=args.device)

    if results:
        costs = [r['plan']['total_cost']     for r in results]
        uncs  = [r['plan']['uncertainty']    for r in results]
        pens  = [r['plan']['penalized_cost'] for r in results]
        rmses = [r['rmse_dq']               for r in results]
        print(f"\n=== Summary ({len(results)} plans) ===")
        print(f"  cost:      {np.mean(costs):.4f} ± {np.std(costs):.4f}")
        print(f"  unc:       {np.mean(uncs):.6f} ± {np.std(uncs):.6f}")
        print(f"  penalized: {np.mean(pens):.4f} ± {np.std(pens):.4f}")
        print(f"  RMSE_Δq:   {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    visualize_lqr_results(results, args.out_dir)
    print(f"\nDone. → {args.out_dir}/")


if __name__ == '__main__':
    main()