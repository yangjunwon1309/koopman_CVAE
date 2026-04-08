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
    use_u_bounds:  bool  = True   # clip u_t to surveyed ψ(a) range



# ─────────────────────────────────────────────────────────────────────────────
# Action encoder range survey: u = ψ(a),  a ∈ [-1,1]^da
# ─────────────────────────────────────────────────────────────────────────────

def survey_action_encoder_range(
    model,
    n_random:    int   = 20000,
    n_pertub:    int   = 5000,
    eps:         float = 1e-3,
    device:      str   = 'cuda',
    save_path:   str   = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ψ(a) 범위 조사: a ∈ [-1-eps, 1+eps]^da + perturbation

    Returns:
        u_min: (d_u,)  각 latent 차원 하한
        u_max: (d_u,)  각 latent 차원 상한
    """
    model.eval()
    da = model.cfg.action_dim
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    samples = []

    # 1. Random uniform in [-1, 1]^da
    a_rand = torch.FloatTensor(n_random, da).uniform_(-1., 1.)
    samples.append(a_rand)

    # 2. Extreme corners (2^min(da,8) subset)
    n_corners = min(da, 8)
    for i in range(2**n_corners):
        corner = np.array([((-1)**((i >> j) & 1)) for j in range(n_corners)]
                          + [0.0] * (da - n_corners), dtype=np.float32)
        samples.append(torch.FloatTensor(corner).unsqueeze(0))

    # 3. Boundary perturbation: a ∈ {-1, 1}^da + small noise
    a_boundary = torch.FloatTensor(n_pertub, da).uniform_(-1., 1.)
    a_boundary = a_boundary.sign() + torch.randn(n_pertub, da) * eps
    a_boundary = a_boundary.clamp(-1. - eps, 1. + eps)
    samples.append(a_boundary)

    # 4. Per-axis extremes (each dim at ±1, rest at 0)
    for d in range(da):
        for v in [-1., 1.]:
            a_ax = torch.zeros(1, da)
            a_ax[0, d] = v
            samples.append(a_ax)

    a_all = torch.cat(samples, dim=0).to(dev)

    # Batch forward
    batch_size = 2048
    u_list = []
    with torch.no_grad():
        for i in range(0, len(a_all), batch_size):
            u_batch = model.action_encoder(a_all[i:i+batch_size])
            u_list.append(u_batch.cpu())
    u_all = torch.cat(u_list, dim=0).numpy()   # (N, d_u)

    u_min = u_all.min(axis=0)   # (d_u,)
    u_max = u_all.max(axis=0)   # (d_u,)

    print(f"\nAction encoder range survey: {len(a_all)} samples  d_u={u_all.shape[1]}")
    print(f"  u_min: mean={u_min.mean():.4f}  std={u_min.std():.4f}"
          f"  min={u_min.min():.4f}  max={u_min.max():.4f}")
    print(f"  u_max: mean={u_max.mean():.4f}  std={u_max.std():.4f}"
          f"  min={u_max.min():.4f}  max={u_max.max():.4f}")
    print(f"  range width: mean={( u_max - u_min).mean():.4f}")

    if save_path:
        np.savez(save_path, u_min=u_min, u_max=u_max)
        print(f"  Saved: {save_path}")

    return u_min, u_max


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
        # Per-dim u bounds from survey_action_encoder_range()
        # None until survey() is called
        self.u_min: Optional[np.ndarray] = None   # (d_u,)
        self.u_max: Optional[np.ndarray] = None   # (d_u,)
        # Clip statistics buffers (reset per rollout)
        self._clip_raw:  List = []
        self._clip_post: List = []

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

    def survey(
        self,
        n_random:  int   = 20000,
        n_pertub:  int   = 5000,
        eps:       float = 1e-3,
        save_path: str   = None,
    ):
        """
        ψ(a) 범위 조사 후 u_min, u_max를 내부에 저장.
        _lqr_u에서 per-dim clip에 사용됨.
        """
        self.u_min, self.u_max = survey_action_encoder_range(
            self.model, n_random=n_random, n_pertub=n_pertub,
            eps=eps, device=str(self.device), save_path=save_path,
        )
        return self.u_min, self.u_max

    def load_u_bounds(self, path: str):
        """저장된 u_bounds npz 파일 로드."""
        data = np.load(path)
        self.u_min = data['u_min']
        self.u_max = data['u_max']
        print(f"Loaded u_bounds: {path}  "
              f"u_min=[{self.u_min.min():.4f}, {self.u_min.max():.4f}]  "
              f"u_max=[{self.u_max.min():.4f}, {self.u_max.max():.4f}]")

    def _lqr_u(self, o, o_star, L, M):
        """
        u_t* = M·z* - L·z_t   (평형점 가정 없는 정확한 형태)

        L = (R+BᵀPB)⁻¹ BᵀPA   feedback
        M = (R+BᵀPB)⁻¹ BᵀP    feedforward
        """
        z_t   = o[0].cpu().numpy()          # (m,)
        z_star = o_star[0].cpu().numpy()    # (m,)
        u_raw = M @ z_star - L @ z_t
        # Per-dim clip using surveyed ψ(a) range
        if self.cfg.use_u_bounds and self.u_min is not None:
            u_np = np.clip(u_raw, self.u_min, self.u_max)
        else:
            u_np = np.clip(u_raw, -self.cfg.u_max, self.cfg.u_max)
        # Record clip statistics
        self._clip_raw.append(u_raw)
        self._clip_post.append(u_np)
        return torch.FloatTensor(u_np).unsqueeze(0).to(self.device)

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
        self._clip_raw  = []   # reset per rollout
        self._clip_post = []
        o_list, u_list, a_list, w_list, costs = [o0], [], [], [], []
        o_cur, h_cur = o0, h0

        w_cur               = model.skill_prior.soft_weights(h_cur)
        w_list.append(w_cur)
        A_cur, B_cur, L_cur, M_cur = self._blended(w_cur)
        A_prev_np           = A_cur.detach().cpu().numpy()
        o_star              = self.encode_goal(x_goal_t, h_cur)   # (1, m)

        for t in range(horizon):
            u_t    = self._lqr_u(o_cur, o_star, L_cur, M_cur)
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

        # Clip statistics
        clip_raw  = np.stack(self._clip_raw,  axis=0) if self._clip_raw  else None  # (H, d_u)
        clip_post = np.stack(self._clip_post, axis=0) if self._clip_post else None  # (H, d_u)
        clip_delta = np.abs(clip_raw - clip_post) if (clip_raw is not None) else None
        return {
            'o_traj':      torch.cat(o_list, 0),
            'u_traj':      torch.cat(u_list, 0),
            'a_traj':      torch.cat(a_list, 0),
            'w_traj':      torch.cat(w_list, 0),
            'costs':       np.array(costs),
            'total_cost':  float(np.sum(costs)),
            'o_star':      o_star,
            'clip_raw':    clip_raw,    # (H, d_u) unconstrained u
            'clip_post':   clip_post,   # (H, d_u) clipped u
            'clip_delta':  clip_delta,  # (H, d_u) |raw - post|
            'clip_active_rate': float(clip_delta.mean()) if clip_delta is not None else 0.0,
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

def detect_completed_tasks_by_reward(
    obs_ep:   np.ndarray,   # (L, 60)
    rew_ep:   np.ndarray,   # (L,)
) -> Dict[str, int]:
    """
    reward jump 시점 + OBS_ELEMENT_GOALS 거리로 실제 완료 task 식별.

    D4RL kitchen reward:
      - task 하나 완료 시 +1 스파스 보상
      - reward[t] > reward[t-1] → 해당 시점에 task 완료

    각 jump 시점에서 OBS_ELEMENT_GOALS와 가장 가까운 task를 할당.
    Returns: {task_name: completion_timestep}  (완료된 task만 포함)
    """
    # reward jump 시점 찾기
    jump_steps = []
    for t in range(1, len(rew_ep)):
        if rew_ep[t] > rew_ep[t - 1]:
            jump_steps.append(t)

    if not jump_steps:
        return {}

    completed = {}
    used_tasks = set()

    for t in jump_steps:
        # 이 시점에서 각 task의 OBS_ELEMENT_GOALS 거리 계산
        best_task, best_dist = None, float('inf')
        for task in ALL_TASKS:
            if task in used_tasks:
                continue
            idx  = OBS_ELEMENT_INDICES[task]
            goal = OBS_ELEMENT_GOALS[task]
            dist = np.linalg.norm(obs_ep[t, idx] - goal)
            if dist < best_dist:
                best_dist = dist
                best_task = task

        if best_task is not None and best_dist < BONUS_THRESH * 2:
            completed[best_task] = t
            used_tasks.add(best_task)

    return completed


def load_kitchen_episodes(
    quality: str = 'mixed',
    min_len: int = 64,
) -> Tuple[List[Dict], np.ndarray]:
    """
    D4RL Kitchen 로드 + 에피소드 분리.

    Task 식별 전략 (우선순위):
      1. reward jump + OBS_ELEMENT_GOALS 거리  ← mixed/partial에 유효
      2. OBS_ELEMENT_GOALS 직접 threshold 탐지 ← fallback

    각 에피소드에 실제 완료된 task만 포함, 완료 timestep 기록.
    """
    import d4rl, gym
    name_map = {'mixed':    'kitchen-mixed-v0',
                'partial':  'kitchen-partial-v0',
                'complete': 'kitchen-complete-v0'}
    d4rl_name = name_map[quality]
    print(f"Loading {d4rl_name} ...")
    env     = gym.make(d4rl_name)
    dataset = env.get_dataset()

    obs       = dataset['observations']
    actions   = dataset['actions']
    rewards   = dataset.get('rewards', np.zeros(len(obs)))
    terminals = dataset['terminals'].astype(bool)

    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    episodes  = []
    n_with_tasks = 0

    for ep_s, ep_e in zip(ep_starts, ep_ends):
        L = ep_e - ep_s + 1
        if L < min_len:
            continue

        obs_ep  = obs[ep_s:ep_e+1]
        acts_ep = actions[ep_s:ep_e+1]
        rew_ep  = rewards[ep_s:ep_e+1]

        # 1. reward jump 기반 task 식별
        completed = detect_completed_tasks_by_reward(obs_ep, rew_ep)

        # 2. reward 없는 경우 OBS_ELEMENT_GOALS threshold fallback
        if not completed:
            completed = detect_task_completions(obs_ep, ALL_TASKS)
            completed = {k: v for k, v in completed.items() if v >= 0}

        actual_tasks = list(completed.keys())
        if actual_tasks:
            n_with_tasks += 1

        # Goal 구성
        subtask_goals = {}
        for task, t in completed.items():
            subtask_goals[task] = {
                'obs':       obs_ep[t],
                'timestep':  t,
                'completed': True,
            }

        goal_info = {
            'final_goal':    obs_ep[-1],
            'midpoint_goal': obs_ep[L // 2],
            'subtask_goals': subtask_goals,
            'completions':   completed,
            'episode_len':   L,
            'n_completed':   len(completed),
            'reward_total':  float(rew_ep.sum()),
        }

        episodes.append({
            'obs':      obs_ep,
            'actions':  acts_ep,
            'rewards':  rew_ep,
            'start_t':  ep_s,
            'end_t':    ep_e,
            'length':   L,
            'tasks':    actual_tasks,   # 실제 완료된 task만
            'goal_info': goal_info,
        })

    print(f"Episodes: {len(episodes)}  (min_len={min_len}  "
          f"with_tasks={n_with_tasks}/{len(episodes)})")
    return episodes, obs


def run_lqr_on_episodes(
    planner:      KODAQLQRPlanner,
    episodes:     List[Dict],
    x_seq_full:   np.ndarray,
    cond_len:     int = 16,
    horizon:      int = 32,
    unc_real_len: int = 8,
    device:       str = 'cuda',
) -> List[Dict]:
    """
    각 에피소드를 reward jump 기준으로 단계별 LQR로 실행.

    단계 구조:
      stage 0: t=0        → t=jump_0   (sub-goal: obs[jump_0])
      stage 1: t=jump_0   → t=jump_1   (sub-goal: obs[jump_1])
      ...
      stage N: t=jump_{N-1} → ep_end   (sub-goal: obs[ep_end])

    각 단계:
      - conditioning: 해당 단계 시작 시점 앞 cond_len 스텝
      - goal: 해당 단계의 reward jump 시점 obs
      - horizon: 해당 단계 길이 (min(jump_t - start_t, horizon)로 clamp)
      - uncertainty: 해당 구간의 실제 action으로 계산
    """
    dev     = torch.device(device)
    results = []

    for ep_idx, ep in enumerate(episodes):
        L, obs_ep, acts_ep = ep['length'], ep['obs'], ep['actions']
        gi    = ep['goal_info']
        tasks = ep['tasks']
        s, e  = ep['start_t'], ep['end_t']
        x_ep  = x_seq_full[s:e+1]   # (L, 2108)

        if L < cond_len + 4:
            continue

        if not tasks:
            print(f"Ep {ep_idx}  len={L}  → skip (no completed tasks)")
            continue

        # ── Sub-goal 시퀀스 구성 ────────────────────────────────────────────
        # reward jump 시점을 순서대로 정렬 → 각 단계의 goal timestep
        jump_timesteps = sorted(gi['completions'].values())   # [t1, t2, ...]
        # 마지막 단계는 에피소드 끝
        stage_ends = jump_timesteps + [L - 1]

        print(f"\nEp {ep_idx}  len={L}  tasks={tasks}  "
              f"reward={gi['reward_total']:.0f}  stages={len(stage_ends)}")
        print(f"  Sub-goal timesteps: {stage_ends}")

        ep_result = {
            'ep_idx':       ep_idx,
            'tasks':        tasks,
            'reward_total': gi['reward_total'],
            'stages':       [],              # per-stage plan results
            'full_o_traj':  [],              # concatenated Koopman trajectory
            'full_a_traj':  [],              # concatenated decoded actions
            'full_x_traj':  [],              # concatenated decoded x_t
            'full_costs':   [],
            'full_sigma2':  [],
            'full_true_x':  [],
        }

        stage_start = 0   # 현재 단계 시작 (에피소드 내 상대 인덱스)

        for stage_idx, stage_end_t in enumerate(stage_ends):
            # ── Conditioning 구간 ──────────────────────────────────────────
            # cond 시작: stage_start에서 cond_len만큼 앞으로, 음수 방지
            cond_start = max(0, stage_start - cond_len)
            cond_end   = stage_start   # 마지막 conditioning step (exclusive)

            # cond가 너무 짧으면 시작점부터
            actual_cond = cond_end - cond_start
            if actual_cond < 1:
                cond_start = 0
                cond_end   = min(cond_len, stage_end_t)

            # ── Stage horizon (동적) ───────────────────────────────────────
            # stage 실제 길이 전체를 horizon으로 사용
            # min_horizon=16으로 하한, horizon 파라미터는 더 이상 상한 아님
            stage_len     = stage_end_t - max(stage_start, cond_end)
            min_horizon   = 16
            stage_horizon = max(min_horizon, stage_len)

            if stage_horizon < 2:
                stage_start = stage_end_t + 1
                continue

            # ── Goal obs: 해당 단계 완료 시점의 obs ──────────────────────
            goal_obs   = obs_ep[stage_end_t]
            ref_obs    = obs_ep[0]                 # episode-first 기준
            x_goal_np  = obs_to_x_goal(goal_obs, ref_obs)
            goal_label = (tasks[stage_idx] if stage_idx < len(tasks)
                          else f'stage{stage_idx}')

            # ── Tensors ────────────────────────────────────────────────────
            cond_slice = slice(cond_start, cond_end) if cond_end > cond_start \
                         else slice(0, min(cond_len, stage_end_t))
            x_cond = torch.FloatTensor(
                x_ep[cond_slice]).unsqueeze(0).to(dev)
            a_cond = torch.FloatTensor(
                acts_ep[cond_slice]).unsqueeze(0).to(dev)
            x_goal_t = torch.FloatTensor(x_goal_np).unsqueeze(0).to(dev)

            # uncertainty용 real actions: stage 구간의 실제 action
            real_start = max(stage_start, cond_end)
            real_end   = min(real_start + unc_real_len, stage_end_t + 1)
            real_a     = torch.FloatTensor(
                acts_ep[real_start:real_end]).to(dev)

            # ── LQR plan ──────────────────────────────────────────────────
            plan = planner.plan(
                x_cond, a_cond, x_goal_t,
                horizon=stage_horizon,
                real_actions=real_a if len(real_a) > 0 else None,
                compute_uncertainty=len(real_a) > 0,
            )

            # ── True trajectory for this stage ────────────────────────────
            true_start = max(stage_start, cond_end)
            true_end   = min(true_start + stage_horizon, L)
            true_x     = x_ep[true_start:true_end]
            H_c        = min(plan['x_traj'].shape[0], len(true_x))
            pred_dq    = plan['x_traj'][:H_c, X_DQ_START:X_DQ_END].cpu().numpy()
            true_dq    = true_x[:H_c, X_DQ_START:X_DQ_END]
            rmse       = float(np.sqrt(((pred_dq - true_dq)**2).mean()))

            unc = plan.get('uncertainty', 0.0)
            pen = plan.get('penalized_cost', plan['total_cost'])
            print(f"  Stage {stage_idx} [{goal_label}@t={stage_end_t}]  "
                  f"horizon={stage_horizon}  "
                  f"cost={plan['total_cost']:.3f}  "
                  f"unc={unc:.5f}  pen={pen:.3f}  RMSE_Δq={rmse:.4f}")

            stage_res = {
                'stage_idx':   stage_idx,
                'goal_label':  goal_label,
                'goal_t':      stage_end_t,
                'horizon':     stage_horizon,
                'plan':        plan,
                'true_x':      true_x,
                'rmse_dq':     rmse,
            }
            ep_result['stages'].append(stage_res)

            # Full trajectory 누적
            ep_result['full_o_traj'].append(plan['o_traj'].cpu())
            ep_result['full_a_traj'].append(plan['a_traj'].cpu())
            ep_result['full_x_traj'].append(plan['x_traj'].cpu())
            ep_result['full_costs'].extend(plan['costs'].tolist())
            ep_result['full_true_x'].append(torch.FloatTensor(true_x))
            if plan.get('sigma2_t') is not None:
                ep_result['full_sigma2'].extend(plan['sigma2_t'].tolist())

            stage_start = stage_end_t + 1

        # Concatenate full trajectory
        if ep_result['full_o_traj']:
            ep_result['full_o_traj'] = torch.cat(ep_result['full_o_traj'], 0)
            ep_result['full_a_traj'] = torch.cat(ep_result['full_a_traj'], 0)
            ep_result['full_x_traj'] = torch.cat(ep_result['full_x_traj'], 0)
            ep_result['full_true_x'] = torch.cat(ep_result['full_true_x'], 0)
            ep_result['full_costs']  = np.array(ep_result['full_costs'])
            ep_result['full_sigma2'] = np.array(ep_result['full_sigma2']) \
                                       if ep_result['full_sigma2'] else None

            total_cost = float(ep_result['full_costs'].sum())
            all_rmse   = [st['rmse_dq'] for st in ep_result['stages']]
            print(f"  → Episode total_cost={total_cost:.3f}  "
                  f"mean_RMSE={np.mean(all_rmse):.4f}")

        results.append(ep_result)

    return results


def visualize_lqr_results(results, out_dir='checkpoints/kodaq/lqr'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for res in results:
        ep_idx = res['ep_idx']
        stages = res['stages']
        if not stages:
            continue
        n = len(stages)
        fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * max(n, 1)), squeeze=False)

        for si, stage in enumerate(stages):
            plan   = stage['plan']
            true_x = stage['true_x']
            x_traj = plan['x_traj'].cpu().numpy()
            sig2   = plan.get('sigma2_t', None)
            H      = x_traj.shape[0] - 1
            col    = PAL[si % len(PAL)]

            ax = axes[si, 0]
            ax.plot(plan['costs'], color=col, lw=1.5)
            ax.set_title(
                f"Stage {si} [{stage['goal_label']}@t={stage['goal_t']}]"
                f"\ncost={plan['total_cost']:.3f}", fontsize=8)
            ax.set_xlabel('step')
            ax.spines[['top','right']].set_visible(False)

            ax = axes[si, 1]
            dq_p = x_traj[:, X_DQ_START:X_DQ_END].mean(1)
            Hc   = min(H + 1, len(true_x))
            dq_t = true_x[:Hc, X_DQ_START:X_DQ_END].mean(1)
            ax.plot(dq_p, '--', color=col, lw=1.5, label='pred')
            ax.plot(np.arange(Hc), dq_t, 'k-', lw=1.5, label='true')
            ax.set_title(f"Dq_t mean  RMSE={stage['rmse_dq']:.4f}", fontsize=8)
            ax.legend(fontsize=7)
            ax.spines[['top','right']].set_visible(False)

            ax = axes[si, 2]
            if sig2 is not None:
                ax.fill_between(np.arange(len(sig2)), 0, sig2, alpha=0.4, color=col)
                ax.plot(sig2, color=col, lw=1.5)
                ax.set_title(f"sigma2_t  U={plan.get('uncertainty', 0.0):.5f}", fontsize=8)
            else:
                ax.set_title("Uncertainty N/A", fontsize=8)
            ax.spines[['top','right']].set_visible(False)

        tc = float(res['full_costs'].sum()) if isinstance(res['full_costs'], np.ndarray)              else sum(st['plan']['total_cost'] for st in stages)
        mr = np.mean([st['rmse_dq'] for st in stages])
        fig.suptitle(
            f"Ep {ep_idx}  Tasks:{res['tasks']}  reward={res['reward_total']:.0f}\n"
            f"TotalCost={tc:.3f}  MeanRMSE={mr:.4f}",
            fontsize=9, fontweight='bold')
        plt.tight_layout()
        out = f"{out_dir}/ep{ep_idx}_sequential.png"
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
    p.add_argument('--quality',    default='mixed', choices=['mixed','partial','complete'])
    p.add_argument('--n_ep',       type=int,   default=5)
    p.add_argument('--cond_len',   type=int,   default=16)
    p.add_argument('--horizon',    type=int,   default=32)
    p.add_argument('--unc_len',    type=int,   default=8)
    p.add_argument('--Q_scale',    type=float, default=1.0)
    p.add_argument('--R_scale',    type=float, default=10.0)
    p.add_argument('--lambda_unc', type=float, default=0.1)
    p.add_argument('--out_dir',    default='checkpoints/kodaq/lqr')
    p.add_argument('--survey',     action='store_true',
                   help='Run action encoder range survey before planning')
    p.add_argument('--u_bounds',   default=None,
                   help='Path to saved u_bounds.npz (skip survey if provided)')
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

    # ── U-space bounds ──────────────────────────────────────────────────────
    if args.u_bounds and Path(args.u_bounds).exists():
        planner.load_u_bounds(args.u_bounds)
    elif args.survey:
        save_path = str(Path(args.out_dir) / 'u_bounds.npz')
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        planner.survey(save_path=save_path)
    else:
        print('No u_bounds: using symmetric clip u_max=cfg.u_max  '
              '(run with --survey to enable per-dim bounds)')

    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    print(f"x_seq: {x_seq_full.shape}")

    episodes, _ = load_kitchen_episodes(quality=args.quality,
                                        min_len=args.cond_len + 8)
    episodes = episodes[:args.n_ep]

    results = run_lqr_on_episodes(
        planner, episodes, x_seq_full,
        cond_len=args.cond_len, horizon=args.horizon,
        unc_real_len=args.unc_len, device=args.device)

    all_costs = [st['plan']['total_cost'] for r in results for st in r['stages']]
    all_rmse  = [st['rmse_dq']           for r in results for st in r['stages']]
    if all_costs:
        print(f"\n=== Summary ({len(results)} eps, {len(all_costs)} stages) ===")
        print(f"  stage cost: {np.mean(all_costs):.4f} +/- {np.std(all_costs):.4f}")
        print(f"  RMSE_Dq:    {np.mean(all_rmse):.4f} +/- {np.std(all_rmse):.4f}")

    visualize_lqr_results(results, args.out_dir)
    print(f"\nDone. -> {args.out_dir}/")
    from analyze_bound import analyze_clip_effect
    analyze_clip_effect(results, args.out_dir + '/clip_analysis.png')


if __name__ == '__main__':
    main()
