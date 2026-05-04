"""
lqr_koopman.py — KODAQ LQR Planner (refactored v2)
====================================================

v2 변경사항:
  1. KODAQLQRPlanner.build_goal_latent_map()
     - 각 episode에 대해 skill 구간별 goal latent z* 매핑 생성
     - 구간 끝점 obs → z* 사전 계산 → 배치 학습에 재사용
  2. KODAQLQRPlanner.lqr_rollout_batch()
     - (B, T, m) 배치 latent에서 H-step LQR rollout → Q target용
     - decode_action 없이 u^LQR만 계산 (속도 최적화)
  3. build_episode_goal_z_dataset()
     - 전체 dataset의 goal_z_seq 사전 계산 → npz 저장
  4. 기존 plan() / run_lqr_on_episodes() / visualize 그대로 유지

Offline Q target 계산 흐름:
  episode t: skill_segment → goal_t = completion_obs[t]
  z*_t = encode_goal(goal_obs, h_t)
  u_k^LQR = M·z*_t - L·z_k  (k=0..H-1)
  Q_target = Σ γ^k · R̂_pen(z_k, u_k^LQR) + γ^H · Q̄(z_H, u_H^LQR)
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


def torch_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        return torch.linalg.solve(A, B)
    return torch.solve(B, A).solution


# ─────────────────────────────────────────────────────────────────────────────
# D4RL Kitchen constants
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

X_DE_START = 0;    X_DE_END = 2048
X_DP_START = 2048; X_DP_END = 2090
X_DQ_START = 2090; X_DQ_END = 2099
X_QD_START = 2099; X_QD_END = 2108


# ─────────────────────────────────────────────────────────────────────────────
# Task completion detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_task_completions(obs_ep: np.ndarray,
                             tasks: List[str]) -> Dict[str, int]:
    """각 task의 첫 completion timestep 반환 (-1 = not completed)."""
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


def detect_completed_tasks_by_reward(obs_ep: np.ndarray,
                                      rew_ep: np.ndarray) -> Dict[str, int]:
    """reward jump 시점 + OBS_ELEMENT_GOALS 거리로 실제 완료 task 식별."""
    jump_steps = [t for t in range(1, len(rew_ep)) if rew_ep[t] > rew_ep[t-1]]
    if not jump_steps:
        return {}
    completed, used_tasks = {}, set()
    for t in jump_steps:
        best_task, best_dist = None, float('inf')
        for task in ALL_TASKS:
            if task in used_tasks:
                continue
            idx  = OBS_ELEMENT_INDICES[task]
            goal = OBS_ELEMENT_GOALS[task]
            dist = np.linalg.norm(obs_ep[t, idx] - goal)
            if dist < best_dist:
                best_dist, best_task = dist, task
        if best_task is not None and best_dist < BONUS_THRESH * 2:
            completed[best_task] = t
            used_tasks.add(best_task)
    return completed


def identify_episode_goals(obs_ep: np.ndarray, tasks: List[str]) -> Dict:
    """goal 식별: final_goal / subtask_goals / midpoint_goal."""
    L           = len(obs_ep)
    completions = detect_task_completions(obs_ep, tasks)
    subtask_goals = {}
    for task in tasks:
        t = completions[task]
        if t >= 0:
            subtask_goals[task] = {'obs': obs_ep[t], 'timestep': t, 'completed': True}
        else:
            g_obs = obs_ep[0].copy()
            g_obs[OBS_ELEMENT_INDICES[task]] = OBS_ELEMENT_GOALS[task]
            subtask_goals[task] = {'obs': g_obs, 'timestep': -1, 'completed': False}
    return {
        'final_goal':    obs_ep[-1],
        'midpoint_goal': obs_ep[L // 2],
        'subtask_goals': subtask_goals,
        'completions':   completions,
        'episode_len':   L,
    }


def obs_to_x_goal(goal_obs: np.ndarray, ref_obs: np.ndarray) -> np.ndarray:
    """x_goal = [Δe=0 | Δp=goal_obj-ref_obj | Δq=goal_q-ref_q | q̇=0]"""
    delta_e = np.zeros(2048, dtype=np.float32)
    delta_p = (goal_obs[18:60] - ref_obs[18:60]).astype(np.float32)
    delta_q = (goal_obs[0:9]   - ref_obs[0:9]).astype(np.float32)
    qdot    = np.zeros(9, dtype=np.float32)
    return np.concatenate([delta_e, delta_p, delta_q, qdot])


# ─────────────────────────────────────────────────────────────────────────────
# LQR Config + DARE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LQRConfig:
    Q_scale:          float = 1.0
    R_scale:          float = 0.1
    eps_A:            float = 0.01
    T_replan:         int   = 8
    dare_max_iter:    int   = 300
    dare_tol:         float = 1e-8
    lambda_unc:       float = 0.1
    u_max:            float = 1.0
    action_inv_steps: int   = 30
    use_u_bounds:     bool  = True


def solve_dare_safe(A, B, Q, R, max_iter=300, tol=1e-8):
    """
    DARE → (P, L, M)
    u_t* = M·z* - L·z_t   (equilibrium-free)
    """
    try:
        P = solve_discrete_are(A, B, Q, R)
    except Exception:
        P = Q.copy()
        for _ in range(max_iter):
            BtP   = B.T @ P
            P_new = (A.T @ P @ A
                     - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
                     + Q)
            if np.max(np.abs(P_new - P)) < tol:
                P = P_new; break
            P = P_new
    BtP   = B.T @ P
    S_inv = np.linalg.inv(R + BtP @ B)
    L     = S_inv @ BtP @ A     # (d_u, m) feedback
    M     = S_inv @ BtP         # (d_u, m) feedforward
    return P, L, M


# ─────────────────────────────────────────────────────────────────────────────
# Action encoder range survey
# ─────────────────────────────────────────────────────────────────────────────

def survey_action_encoder_range(
    model,
    n_random:  int   = 20000,
    n_pertub:  int   = 5000,
    eps:       float = 1e-3,
    device:    str   = 'cuda',
    save_path: str   = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ψ(a) 범위 조사 → (u_min, u_max) per dim."""
    model.eval()
    da  = model.cfg.action_dim
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    samples = [torch.FloatTensor(n_random, da).uniform_(-1., 1.)]

    n_corners = min(da, 8)
    for i in range(2**n_corners):
        corner = np.array([((-1)**((i >> j) & 1)) for j in range(n_corners)]
                          + [0.0] * (da - n_corners), dtype=np.float32)
        samples.append(torch.FloatTensor(corner).unsqueeze(0))

    a_boundary = torch.FloatTensor(n_pertub, da).uniform_(-1., 1.)
    a_boundary = a_boundary.sign() + torch.randn(n_pertub, da) * eps
    a_boundary = a_boundary.clamp(-1. - eps, 1. + eps)
    samples.append(a_boundary)

    for d in range(da):
        for v in [-1., 1.]:
            a_ax = torch.zeros(1, da); a_ax[0, d] = v
            samples.append(a_ax)

    a_all  = torch.cat(samples, dim=0).to(dev)
    u_list = []
    with torch.no_grad():
        for i in range(0, len(a_all), 2048):
            u_list.append(model.action_encoder(a_all[i:i+2048]).cpu())
    u_all = torch.cat(u_list).numpy()
    u_min, u_max = u_all.min(0), u_all.max(0)

    print(f"Survey: {len(a_all)} samples  "
          f"u_range_mean={( u_max - u_min).mean():.4f}")
    if save_path:
        np.savez(save_path, u_min=u_min, u_max=u_max)
        print(f"  Saved: {save_path}")
    return u_min, u_max


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
        self._dare_cache: Dict = {}
        self.u_min: Optional[np.ndarray] = None
        self.u_max: Optional[np.ndarray] = None
        # Pre-computed gain tensors for batch LQR (set by _precompute_gains)
        self._L_tensor: Optional[torch.Tensor] = None  # (K, d_u, m)
        self._M_tensor: Optional[torch.Tensor] = None  # (K, d_u, m)

    # ── DARE & gain utilities ─────────────────────────────────────────────────

    def _get_gain(self, A: np.ndarray, B: np.ndarray):
        """Cache DARE by (A, B) hash → (P, L, M)."""
        key = (A.tobytes()[:64], B.tobytes()[:32])
        if key not in self._dare_cache:
            self._dare_cache[key] = solve_dare_safe(
                A, B, self.Q, self.R,
                self.cfg.dare_max_iter, self.cfg.dare_tol)
        return self._dare_cache[key]

    def _blended(self, w: torch.Tensor):
        """Blended Ā(w), B̄(w) + DARE → (A_bar, B_bar, L, M)."""
        koop    = self.model.koopman
        log_lam = koop.get_log_lambdas()
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, koop.theta_k, koop.G_k, koop.U, w)
        A_bar = A_bar[0]; B_bar = B_bar[0]
        _, L, M = self._get_gain(
            A_bar.detach().cpu().numpy(),
            B_bar.detach().cpu().numpy())
        return A_bar, B_bar, L, M

    def precompute_gains(self, H: int = 4):
        """
        Finite-Horizon Riccati Backward Recursion for blended dynamics.

        각 pure skill k에 대해 H-step finite horizon LQR gain 계산.
        blended dynamics Ā(w), B̄(w) 기반이 맞으나, w는 배치마다 달라서
        per-skill gain을 precompute하고 skill-weighted sum으로 근사합니다.

        단, lqr_rollout_batch에서는 blended Ā,B̄를 직접 계산 후
        단일 Riccati recursion을 적용합니다 (더 정확한 방법).

        Riccati backward recursion (제공된 수식):
          P_H = Q_f  (= Q_scale * I)
          K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1}
          P_k = Q + A^T P_{k+1} A - A^T P_{k+1} B (B^T P_{k+1} B + R)^{-1} B^T P_{k+1} A

        최적 제어: u_k* = -K_k (A z_k - z*)
        """
        koop = self.model.koopman
        K    = self.m_cfg.num_skills
        d_u  = self.m_cfg.action_latent
        m    = self.m_cfg.koopman_dim
        dev  = self.device
        self._lqr_H = H   # store for rollout

        A_k_t = koop.get_A_k()   # (K, m, m)  tensor
        B_k_t = koop.get_B_k()   # (K, m, d_u) tensor

        # per-skill: list of K × [K_0,...,K_{H-1}]  each (d_u, m)
        K_gains_list = []  # (K, H, d_u, m)

        for k in range(K):
            A_np = A_k_t[k].detach().cpu().numpy().astype(np.float64)
            B_np = B_k_t[k].detach().cpu().numpy().astype(np.float64)
            K_seq = self._finite_horizon_riccati(A_np, B_np, H)
            K_gains_list.append(K_seq)   # (H, d_u, m)

        # Stack: (K, H, d_u, m)
        self._K_gains = torch.FloatTensor(
            np.stack(K_gains_list, axis=0)).to(dev)
        print(f"Finite-horizon Riccati: K={K}  H={H}  d_u={d_u}  m={m}")

        # Also keep infinite-horizon DARE for single-step plan()
        L_list, M_list = [], []
        for k in range(K):
            A_np = A_k_t[k].detach().cpu().numpy()
            B_np = B_k_t[k].detach().cpu().numpy()
            _, L, M = solve_dare_safe(A_np, B_np, self.Q, self.R)
            L_list.append(torch.FloatTensor(L))
            M_list.append(torch.FloatTensor(M))
        self._L_tensor = torch.stack(L_list).to(dev)  # (K, d_u, m)
        self._M_tensor = torch.stack(M_list).to(dev)  # (K, d_u, m)

    def _finite_horizon_riccati(
        self,
        A: np.ndarray,   # (m, m)
        B: np.ndarray,   # (m, d_u)
        H: int,
    ) -> np.ndarray:
        """
        Backward Riccati recursion → K_0,...,K_{H-1}  (H, d_u, m)

        P_H = Q_f
        for k = H-1 down to 0:
            K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1}
            P_k = Q + A^T P_{k+1} A
                    - A^T P_{k+1} B (B^T P_{k+1} B + R)^{-1} B^T P_{k+1} A

        u_k* = -K_k (A z_k - z*)
        """
        m   = A.shape[0]
        d_u = B.shape[1]
        Q   = self.Q.astype(np.float64)
        R   = self.R.astype(np.float64)
        Qf  = Q * 10.0   # Q_f: terminal cost weight (10× state cost)

        P = Qf.copy()
        K_seq = []
        for _ in range(H):   # H steps: k=H-1 down to 0
            BtP  = B.T @ P                           # (d_u, m)
            BtPB = BtP @ B                           # (d_u, d_u)
            S    = BtPB + R                          # (d_u, d_u)
            K    = np.linalg.solve(S, BtP)           # (d_u, m)
            P    = Q + A.T @ P @ A - A.T @ P @ B @ K  # (m, m)
            K_seq.append(K)
        K_seq.reverse()   # K_seq[0] = K_0 (first step)
        return np.array(K_seq, dtype=np.float32)     # (H, d_u, m)

    def _lqr_u_batch(
        self,
        z:      torch.Tensor,   # (B, m)
        z_star: torch.Tensor,   # (B, m)
        w:      torch.Tensor,   # (B, K)  skill weights
    ) -> torch.Tensor:
        """
        Batched LQR action: u = Σ_k w_k · (M_k·z* - L_k·z)
        → (B, d_u)

        Skill-weighted average of per-skill LQR gains.
        Faster than re-computing DARE per step.
        """
        if self._L_tensor is None:
            self.precompute_gains()

        # (B, K, d_u, m) × (B, m, 1) → (B, K, d_u)
        z_e     = z.unsqueeze(-1)       # (B, m, 1)
        zs_e    = z_star.unsqueeze(-1)  # (B, m, 1)

        # u_k = M_k·z* - L_k·z  for each k
        # L: (K, d_u, m),  z: (B, m, 1)
        Lz  = torch.einsum('kdm,bm->bkd', self._L_tensor, z)      # (B, K, d_u)
        Mzs = torch.einsum('kdm,bm->bkd', self._M_tensor, z_star) # (B, K, d_u)
        u_k = Mzs - Lz                                              # (B, K, d_u)

        # Skill-weighted sum: Σ_k w_k · u_k
        u = (w.unsqueeze(-1) * u_k).sum(1)   # (B, d_u)

        # Clip to surveyed action encoder range
        if self.cfg.use_u_bounds and self.u_min is not None:
            u_min_t = torch.FloatTensor(self.u_min).to(z.device)
            u_max_t = torch.FloatTensor(self.u_max).to(z.device)
            u = u.clamp(u_min_t, u_max_t)
        else:
            u = u.clamp(-self.cfg.u_max, self.cfg.u_max)
        return u

    # ── Goal latent map ───────────────────────────────────────────────────────

    @torch.no_grad()
    def build_goal_latent_map(
        self,
        episodes:      List[Dict],
        x_seq_full:    np.ndarray,
        save_path:     Optional[str] = None,
        batch_size:    int = 16,
    ) -> Dict[int, np.ndarray]:
        """
        각 episode × timestep t에 대해 goal_z[t] 사전 계산.

        Skill 구간 설정:
          completions = sorted(task_completion_timesteps)
          구간: [0, c0), [c0, c1), ..., [c_{N-1}, L)
          구간 i의 goal = obs[c_i] (구간 끝점)
          마지막 구간 goal = obs[L-1]

        Returns:
          goal_z_map: {ep_start_t: np.ndarray (L, m)}
          각 timestep t에서의 goal latent z* ∈ ℝ^m
        """
        if self._L_tensor is None:
            self.precompute_gains()

        dev       = self.device
        model     = self.model
        goal_z_map: Dict[int, np.ndarray] = {}
        m         = self.m_cfg.koopman_dim

        print(f"\nBuilding goal latent map: {len(episodes)} episodes ...")

        for ep_idx, ep in enumerate(episodes):
            s, e    = ep['start_t'], ep['end_t']
            L       = ep['length']
            obs_ep  = ep['obs']        # (L, 60)
            acts_ep = ep['actions']    # (L, 9)
            gi      = ep['goal_info']
            ref_obs = obs_ep[0]
            x_ep    = x_seq_full[s:e+1]  # (L, 2108)

            # ── Encode full episode to get h_seq ──────────────────────────
            x_t  = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)    # (1, L, x_dim)
            a_t  = torch.FloatTensor(acts_ep).unsqueeze(0).to(dev) # (1, L, 9)
            enc  = model.encode_sequence(x_t, a_t)
            h_seq = enc['h_seq'][0]   # (L, d_h)
            o_seq = enc['o_seq'][0]   # (L, m)

            # ── Skill segments: completion timesteps as boundaries ─────────
            completions = gi['completions']   # {task: t}
            # sorted completion timesteps → segment boundaries
            boundaries  = sorted(completions.values())  # [c0, c1, ...]
            # add episode end as final boundary
            boundaries.append(L - 1)

            # goal_obs per segment:
            # segment 0: [0, boundaries[0])  → goal = obs[boundaries[0]]
            # segment i: [boundaries[i-1], boundaries[i]) → goal = obs[boundaries[i]]
            # (마지막 segment의 goal = obs[L-1])
            goal_z_seq = np.zeros((L, m), dtype=np.float32)

            seg_start = 0
            for seg_idx, seg_end in enumerate(boundaries):
                goal_obs = obs_ep[min(seg_end, L-1)]
                x_goal_np = obs_to_x_goal(goal_obs, ref_obs)  # (2108,)
                x_goal_t  = torch.FloatTensor(x_goal_np).to(dev)  # (2108,)

                # For each timestep t in [seg_start, seg_end):
                # z* = μ_φ(x_goal, h_{t})
                # Batch encode for this segment
                seg_len = min(seg_end, L) - seg_start
                if seg_len <= 0:
                    seg_start = seg_end
                    continue

                # Encode goal for each t in segment using h_t
                # h_seq[t]: hidden state after processing x_t
                # → posterior(x_goal, h_t) gives goal latent conditioned on current context
                h_seg = h_seq[seg_start:seg_start + seg_len]  # (seg_len, d_h)
                x_goal_rep = x_goal_t.unsqueeze(0).expand(
                    seg_len, -1)  # (seg_len, x_dim)

                # Batch encode
                for i in range(0, seg_len, batch_size):
                    i_end = min(i + batch_size, seg_len)
                    mu, _ = model.posterior(
                        x_goal_rep[i:i_end], h_seg[i:i_end])
                    goal_z_seq[seg_start + i: seg_start + i_end] = \
                        mu.cpu().numpy()

                seg_start = seg_end

            goal_z_map[s] = goal_z_seq  # keyed by global episode start index

            if (ep_idx + 1) % 20 == 0:
                print(f"  {ep_idx+1}/{len(episodes)} episodes done")

        print(f"Goal latent map: {len(goal_z_map)} episodes  "
              f"total_steps={sum(v.shape[0] for v in goal_z_map.values())}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Save as dict: key=ep_start_t (as string), value=array
            np.savez(save_path,
                     ep_starts=np.array(list(goal_z_map.keys())),
                     **{str(k): v for k, v in goal_z_map.items()})
            print(f"  Saved: {save_path}")

        return goal_z_map

    @staticmethod
    def load_goal_latent_map(path: str) -> Dict[int, np.ndarray]:
        """저장된 goal_latent_map.npz 로드."""
        data      = np.load(path, allow_pickle=True)
        ep_starts = data['ep_starts']
        return {int(k): data[str(k)] for k in ep_starts}

    # ── Batch LQR rollout for Q target computation ────────────────────────────

    @torch.no_grad()
    def lqr_rollout_batch(
        self,
        o_seq:      torch.Tensor,   # (B, T, m)
        h_seq:      torch.Tensor,   # (B, T, d_h)
        goal_z_seq: torch.Tensor,   # (B, T, m)  goal z* per timestep
        H:          int = 4,
        gamma:      float = 0.99,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finite-Horizon LQR rollout using blended dynamics Ā(w), B̄(w).

        For each starting timestep t:
          1. Compute blended Ā(w_t), B̄(w_t) using skill weights at t
          2. Run backward Riccati on THIS blended system → K_0,...,K_{H-1}
          3. Roll forward:
               u_k = -K_k (Ā·ẑ_k - z*)   [optimal control toward z*]
               ẑ_{k+1} = Ā·ẑ_k + B̄·u_k

        This uses a SINGLE (Ā,B̄) pair per trajectory (w fixed at t=0),
        which is exact for the blended dynamics rather than interpolating
        per-skill gains (L(Ā) ≠ Σ w_k L(A_k) in general).

        Returns:
          u_lqr_seq:  (B, T-1, H, d_u)
          z_roll_seq: (B, T-1, H+1, m)
        """
        if self._K_gains is None:
            self.precompute_gains(H)

        model   = self.model
        koop    = model.koopman
        log_lam = koop.get_log_lambdas()
        B_b, T, m_dim = o_seq.shape
        T1 = T - 1
        dev = o_seq.device

        # ── Skill weights at t (fixed for the entire H-step rollout) ──────────
        h_t = h_seq[:, :T1]                              # (B, T1, d_h)
        w_t = model.skill_prior.soft_weights(
            h_t.reshape(B_b * T1, -1)
        ).reshape(B_b, T1, -1)                           # (B, T1, K)

        # ── Blended Ā(w_t), B̄(w_t) ────────────────────────────────────────────
        w_flat = w_t.reshape(B_b * T1, -1)               # (B*T1, K)
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, koop.theta_k, koop.G_k, koop.U, w_flat)
        # A_bar: (B*T1, m, m),  B_bar: (B*T1, m, d_u)
        A_bar = A_bar.reshape(B_b, T1, m_dim, m_dim)     # (B, T1, m, m)
        B_bar = B_bar.reshape(B_b, T1, m_dim, -1)        # (B, T1, m, d_u)

        # ── Finite-horizon Riccati on blended dynamics ────────────────────────
        # Compute K_0,...,K_{H-1} for each (A_bar[b,t], B_bar[b,t])
        # This is the key fix: one Riccati per blended system, not per skill.
        #
        # For efficiency: compute batched Riccati in torch
        Q_mat = torch.eye(m_dim,  device=dev) * self.cfg.Q_scale
        R_mat = torch.eye(B_bar.shape[-1], device=dev) * self.cfg.R_scale
        Qf    = Q_mat * 10.0   # terminal cost

        # Flatten batch for Riccati: (B*T1, m, m)
        A_flat = A_bar.reshape(B_b * T1, m_dim, m_dim)   # (N, m, m)
        B_flat = B_bar.reshape(B_b * T1, m_dim, -1)      # (N, m, d_u)
        N      = B_b * T1
        d_u    = B_flat.shape[-1]

        # Backward Riccati
        P = Qf.unsqueeze(0).expand(N, -1, -1).clone()    # (N, m, m)
        K_list = []
        for _ in range(H):
            BtP  = B_flat.transpose(-2, -1) @ P          # (N, d_u, m)
            BtPB = BtP @ B_flat                           # (N, d_u, d_u)
            S    = BtPB + R_mat.unsqueeze(0)              # (N, d_u, d_u)
            K    = torch_solve(S, BtP)                   # (N, d_u, m)
            AtP  = A_flat.transpose(-2, -1) @ P          # (N, m, m)
            P    = (Q_mat.unsqueeze(0)
                   + AtP @ A_flat
                   - AtP @ B_flat @ K)                   # (N, m, m)
            K_list.append(K)
        K_list.reverse()   # K_list[0] = K_0

        # K_gains: (H, N, d_u, m) → (H, B, T1, d_u, m)
        K_gains = torch.stack(K_list, dim=0).reshape(
            H, B_b, T1, d_u, m_dim)

        # ── Forward rollout ───────────────────────────────────────────────────
        z_star = goal_z_seq[:, :T1]                      # (B, T1, m)
        z_cur  = o_seq[:, :T1].clone()                   # (B, T1, m)
        u_lqr_list  = []
        z_roll_list = [z_cur]

        for k in range(H):
            Kk = K_gains[k]   # (B, T1, d_u, m)

            # Koopman predict from current z
            Az = (A_bar @ z_cur.unsqueeze(-1)).squeeze(-1)  # (B, T1, m)

            # u_k* = -K_k (Ā z_k - z*)
            error = Az - z_star                            # (B, T1, m)
            u_k   = -(Kk @ error.unsqueeze(-1)).squeeze(-1)  # (B, T1, d_u)

            # Clip to surveyed action encoder range
            if self.cfg.use_u_bounds and self.u_min is not None:
                u_min_t = torch.FloatTensor(self.u_min).to(dev)
                u_max_t = torch.FloatTensor(self.u_max).to(dev)
                u_k = u_k.clamp(u_min_t, u_max_t)

            u_lqr_list.append(u_k)

            # ẑ_{k+1} = Ā·ẑ_k + B̄·u_k
            z_cur = Az + (B_bar @ u_k.unsqueeze(-1)).squeeze(-1)
            z_roll_list.append(z_cur)

        u_lqr_seq  = torch.stack(u_lqr_list,  dim=2)   # (B, T1, H, d_u)
        z_roll_seq = torch.stack(z_roll_list, dim=2)   # (B, T1, H+1, m)
        return u_lqr_seq, z_roll_seq

    # ── Survey and bounds ─────────────────────────────────────────────────────

    def survey(self, n_random=20000, n_pertub=5000, eps=1e-3, save_path=None):
        self.u_min, self.u_max = survey_action_encoder_range(
            self.model, n_random=n_random, n_pertub=n_pertub,
            eps=eps, device=str(self.device), save_path=save_path)
        return self.u_min, self.u_max

    def load_u_bounds(self, path: str):
        data = np.load(path)
        self.u_min = data['u_min']
        self.u_max = data['u_max']
        print(f"Loaded u_bounds: {path}  "
              f"range=[{self.u_min.min():.4f}, {self.u_max.max():.4f}]")

    # ── Single-sample LQR utilities (online / analysis용) ─────────────────────

    def _lqr_u_single(self, o, o_star, L, M):
        """단일 샘플 LQR action (original plan() 호환)."""
        z_t    = o[0].cpu().numpy()
        z_star = o_star[0].cpu().numpy()
        u_raw  = M @ z_star - L @ z_t
        if self.cfg.use_u_bounds and self.u_min is not None:
            u_np = np.clip(u_raw, self.u_min, self.u_max)
        else:
            u_np = np.clip(u_raw, -self.cfg.u_max, self.cfg.u_max)
        return torch.FloatTensor(u_np).unsqueeze(0).to(self.device)

    def _step(self, o, u, A, B):
        return (A @ o.T).T + (B @ u.T).T

    @torch.enable_grad()
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
    def encode_goal(self, x_goal, h_ref):
        if x_goal.dim() == 1: x_goal = x_goal.unsqueeze(0)
        mu, _ = self.model.posterior(x_goal, h_ref)
        return mu

    @torch.no_grad()
    def _lqr_rollout_single(self, o0, h0, x_goal_t, horizon) -> Dict:
        """단일 샘플 전체 LQR rollout (plan() 호환)."""
        model = self.model
        o_list, u_list, a_list, w_list, costs = [o0], [], [], [], []
        o_cur, h_cur = o0, h0

        w_cur                       = model.skill_prior.soft_weights(h_cur)
        A_cur, B_cur, L_cur, M_cur  = self._blended(w_cur)
        A_prev_np                   = A_cur.detach().cpu().numpy()
        o_star                      = self.encode_goal(x_goal_t, h_cur)

        for t in range(horizon):
            u_t    = self._lqr_u_single(o_cur, o_star, L_cur, M_cur)
            a_t    = self._decode_action(u_t)
            o_next = self._step(o_cur, u_t, A_cur, B_cur)
            h_next = model.recurrent(h_cur, o_cur, a_t)
            w_next = model.skill_prior.soft_weights(h_next)

            A_next, B_next, L_next, M_next = self._blended(w_next)
            A_next_np = A_next.detach().cpu().numpy()

            if (np.linalg.norm(A_next_np - A_prev_np, 'fro') > self.cfg.eps_A
                    or (t + 1) % self.cfg.T_replan == 0):
                A_cur, B_cur, L_cur, M_cur = A_next, B_next, L_next, M_next
                A_prev_np = A_next_np
                o_star = self.encode_goal(x_goal_t, h_next)

            e    = (o_cur - o_star)[0].cpu().numpy()
            u_np = u_t[0].cpu().numpy()
            costs.append(float(e @ self.Q @ e + u_np @ self.R @ u_np))

            o_list.append(o_next); u_list.append(u_t)
            a_list.append(a_t);    w_list.append(w_next)
            o_cur, h_cur = o_next, h_next

        return {
            'o_traj':     torch.cat(o_list, 0),
            'u_traj':     torch.cat(u_list, 0),
            'a_traj':     torch.cat(a_list, 0),
            'w_traj':     torch.cat(w_list, 0),
            'costs':      np.array(costs),
            'total_cost': float(np.sum(costs)),
            'o_star':     o_star,
        }

    def _decode_x_traj(self, o_traj: torch.Tensor) -> torch.Tensor:
        recon = self.model.decoder(o_traj)
        return torch.cat([symexp(recon['delta_e']), symexp(recon['delta_p']),
                          symexp(recon['q']),       symexp(recon['qdot'])], dim=-1)

    @torch.no_grad()
    def plan(self, x_cond, a_cond, x_goal, horizon=32,
             real_actions=None, compute_uncertainty=False) -> Dict:
        """Single-episode plan (analysis / original interface 호환)."""
        dev      = self.device
        enc      = self.model.encode_sequence(x_cond, a_cond)
        o0       = enc['o_seq'][0, -1:].to(dev)
        h0       = enc['h_seq'][0, -1:].to(dev)
        x_goal_t = x_goal.to(dev)
        if x_goal_t.dim() == 1: x_goal_t = x_goal_t.unsqueeze(0)

        lqr    = self._lqr_rollout_single(o0, h0, x_goal_t, horizon)
        x_traj = self._decode_x_traj(lqr['o_traj'].to(dev))
        return {**lqr, 'o0': o0, 'h0': h0, 'x_traj': x_traj}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_kitchen_episodes(
    quality: str = 'mixed',
    min_len: int = 64,
) -> Tuple[List[Dict], np.ndarray]:
    """D4RL Kitchen 로드 + 에피소드 분리."""
    import d4rl, gym
    name_map = {'mixed':    'kitchen-mixed-v0',
                'partial':  'kitchen-partial-v0',
                'complete': 'kitchen-complete-v0'}
    env     = gym.make(name_map[quality])
    dataset = env.get_dataset()

    obs       = dataset['observations']
    actions   = dataset['actions']
    rewards   = dataset.get('rewards', np.zeros(len(obs)))
    terminals = dataset['terminals'].astype(bool)

    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    episodes  = []

    for ep_s, ep_e in zip(ep_starts, ep_ends):
        L = ep_e - ep_s + 1
        if L < min_len:
            continue
        obs_ep  = obs[ep_s:ep_e+1]
        acts_ep = actions[ep_s:ep_e+1]
        rew_ep  = rewards[ep_s:ep_e+1]

        completed = detect_completed_tasks_by_reward(obs_ep, rew_ep)
        if not completed:
            completed = detect_task_completions(obs_ep, ALL_TASKS)
            completed = {k: v for k, v in completed.items() if v >= 0}

        subtask_goals = {
            task: {'obs': obs_ep[t], 'timestep': t, 'completed': True}
            for task, t in completed.items()
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
            'tasks':    list(completed.keys()),
            'goal_info': goal_info,
        })

    n_tasks = sum(1 for e in episodes if e['tasks'])
    print(f"Episodes: {len(episodes)}  with_tasks={n_tasks}/{len(episodes)}")
    return episodes, obs


# ─────────────────────────────────────────────────────────────────────────────
# Build goal_z dataset (offline preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def build_episode_goal_z_dataset(
    model:       KoopmanCVAE,
    planner:     KODAQLQRPlanner,
    episodes:    List[Dict],
    x_seq_full:  np.ndarray,
    save_path:   str,
    device:      str = 'cuda',
    batch_size:  int = 16,
):
    """
    전체 dataset에 대해 goal_z_seq 사전 계산 → npz 저장.

    저장 형식:
      goal_latent_map.npz:
        ep_starts: (N_ep,) int array
        {ep_start}: (L, m) float32 array  per episode
    """
    model.eval()
    goal_z_map = planner.build_goal_latent_map(
        episodes, x_seq_full,
        save_path=save_path,
        batch_size=batch_size,
    )
    return goal_z_map


# ─────────────────────────────────────────────────────────────────────────────
# Existing episode analysis (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_lqr_on_episodes(
    planner:      KODAQLQRPlanner,
    episodes:     List[Dict],
    x_seq_full:   np.ndarray,
    cond_len:     int = 16,
    horizon:      int = 32,
    unc_real_len: int = 8,
    device:       str = 'cuda',
) -> List[Dict]:
    """각 에피소드를 reward jump 기준으로 단계별 LQR 실행."""
    dev     = torch.device(device)
    results = []

    for ep_idx, ep in enumerate(episodes):
        L, obs_ep, acts_ep = ep['length'], ep['obs'], ep['actions']
        gi, tasks = ep['goal_info'], ep['tasks']
        s, e  = ep['start_t'], ep['end_t']
        x_ep  = x_seq_full[s:e+1]
        if L < cond_len + 4 or not tasks:
            continue

        jump_timesteps = sorted(gi['completions'].values())
        stage_ends     = jump_timesteps + [L - 1]

        print(f"\nEp {ep_idx}  len={L}  tasks={tasks}  "
              f"reward={gi['reward_total']:.0f}  stages={len(stage_ends)}")

        ep_result = {
            'ep_idx': ep_idx, 'tasks': tasks,
            'reward_total': gi['reward_total'],
            'stages': [], 'full_o_traj': [], 'full_a_traj': [],
            'full_x_traj': [], 'full_costs': [], 'full_true_x': [],
        }
        stage_start = 0

        for stage_idx, stage_end_t in enumerate(stage_ends):
            cond_start    = max(0, stage_start - cond_len)
            cond_end      = stage_start if stage_start > 0 else min(cond_len, stage_end_t)
            stage_len     = stage_end_t - max(stage_start, cond_end)
            stage_horizon = max(16, stage_len)
            if stage_horizon < 2:
                stage_start = stage_end_t + 1; continue

            goal_obs   = obs_ep[min(stage_end_t, L-1)]
            x_goal_np  = obs_to_x_goal(goal_obs, obs_ep[0])
            goal_label = (tasks[stage_idx] if stage_idx < len(tasks)
                          else f'stage{stage_idx}')

            cond_sl = (slice(cond_start, cond_end) if cond_end > cond_start
                       else slice(0, min(cond_len, stage_end_t)))
            x_cond   = torch.FloatTensor(x_ep[cond_sl]).unsqueeze(0).to(dev)
            a_cond   = torch.FloatTensor(acts_ep[cond_sl]).unsqueeze(0).to(dev)
            x_goal_t = torch.FloatTensor(x_goal_np).unsqueeze(0).to(dev)

            plan = planner.plan(x_cond, a_cond, x_goal_t, horizon=stage_horizon)

            true_start = max(stage_start, cond_end)
            true_end   = min(true_start + stage_horizon, L)
            true_x     = x_ep[true_start:true_end]
            H_c        = min(plan['x_traj'].shape[0], len(true_x))
            pred_dq    = plan['x_traj'][:H_c, X_DQ_START:X_DQ_END].cpu().numpy()
            rmse       = float(np.sqrt(((pred_dq - true_x[:H_c, X_DQ_START:X_DQ_END])**2).mean()))

            print(f"  Stage {stage_idx} [{goal_label}@t={stage_end_t}]  "
                  f"cost={plan['total_cost']:.3f}  RMSE_Δq={rmse:.4f}")
            ep_result['stages'].append({
                'stage_idx': stage_idx, 'goal_label': goal_label,
                'goal_t': stage_end_t, 'plan': plan,
                'true_x': true_x, 'rmse_dq': rmse,
            })
            ep_result['full_o_traj'].append(plan['o_traj'].cpu())
            ep_result['full_a_traj'].append(plan['a_traj'].cpu())
            ep_result['full_x_traj'].append(plan['x_traj'].cpu())
            ep_result['full_costs'].extend(plan['costs'].tolist())
            ep_result['full_true_x'].append(torch.FloatTensor(true_x))
            stage_start = stage_end_t + 1

        if ep_result['full_o_traj']:
            for k in ['full_o_traj', 'full_a_traj', 'full_x_traj', 'full_true_x']:
                ep_result[k] = torch.cat(ep_result[k], 0)
            ep_result['full_costs'] = np.array(ep_result['full_costs'])

        results.append(ep_result)
    return results


def visualize_lqr_results(results, out_dir='checkpoints/kodaq/lqr'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for res in results:
        stages = res['stages']
        if not stages: continue
        n   = len(stages)
        fig, axes = plt.subplots(n, 2, figsize=(14, 3.5*max(n,1)), squeeze=False)

        for si, stage in enumerate(stages):
            plan   = stage['plan']
            true_x = stage['true_x']
            x_traj = plan['x_traj'].cpu().numpy()
            col    = PAL[si % len(PAL)]

            ax = axes[si, 0]
            ax.plot(plan['costs'], color=col, lw=1.5)
            ax.set_title(f"Stage {si} [{stage['goal_label']}@t={stage['goal_t']}]"
                         f"\ncost={plan['total_cost']:.3f}", fontsize=8)
            ax.spines[['top','right']].set_visible(False)

            ax = axes[si, 1]
            H_c  = min(x_traj.shape[0], len(true_x))
            dq_p = x_traj[:H_c, X_DQ_START:X_DQ_END].mean(1)
            dq_t = true_x[:H_c, X_DQ_START:X_DQ_END].mean(1)
            ax.plot(dq_p, '--', color=col, lw=1.5, label='pred')
            ax.plot(np.arange(H_c), dq_t, 'k-', lw=1.5, label='true')
            ax.set_title(f"Dq_t mean  RMSE={stage['rmse_dq']:.4f}", fontsize=8)
            ax.legend(fontsize=7)
            ax.spines[['top','right']].set_visible(False)

        tc = float(res['full_costs'].sum()) if isinstance(res.get('full_costs'), np.ndarray) else 0.0
        mr = np.mean([st['rmse_dq'] for st in stages])
        fig.suptitle(f"Ep {res['ep_idx']}  Tasks:{res['tasks']}\n"
                     f"TotalCost={tc:.3f}  MeanRMSE={mr:.4f}",
                     fontsize=9, fontweight='bold')
        plt.tight_layout()
        path = f"{out_dir}/ep{res['ep_idx']}_lqr.png"
        plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
        print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',         default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',      default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--quality',      default='mixed')
    p.add_argument('--n_ep',         type=int,   default=5)
    p.add_argument('--cond_len',     type=int,   default=16)
    p.add_argument('--horizon',      type=int,   default=32)
    p.add_argument('--Q_scale',      type=float, default=1.0)
    p.add_argument('--R_scale',      type=float, default=10.0)
    p.add_argument('--out_dir',      default='checkpoints/kodaq/lqr')
    p.add_argument('--survey',       action='store_true')
    p.add_argument('--u_bounds',     default=None)
    p.add_argument('--build_goal_z', action='store_true',
                   help='Pre-compute goal_z_seq for all episodes')
    p.add_argument('--goal_z_path',  default='checkpoints/kodaq/goal_latent_map.npz')
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ckpt  = torch.load(args.ckpt, map_location=args.device)
    model = KoopmanCVAE(ckpt['cfg'])

    # v5 compat
    for k, v in dict(num_bins=16, v_min=0., v_max=5., num_q=2, tau=0.005,
                     gamma=0.99, entropy_coef=0.01, log_std_min=-5., log_std_max=2.,
                     lambda_reward=1., lambda_q=1., lambda_pi=0.1,
                     reward_ensemble_n=5, td_horizon=4, mopo_beta=1.,
                     use_ensemble_reward=False).items():
        if not hasattr(model.cfg, k): setattr(model.cfg, k, v)

    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval().to(args.device)

    planner = KODAQLQRPlanner(model, LQRConfig(
        Q_scale=args.Q_scale, R_scale=args.R_scale))

    if args.u_bounds and Path(args.u_bounds).exists():
        planner.load_u_bounds(args.u_bounds)
    elif args.survey:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        planner.survey(save_path=f"{args.out_dir}/u_bounds.npz")

    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality,
                                              min_len=args.cond_len + 8)

    if args.build_goal_z:
        print("\n=== Building goal latent map ===")
        build_episode_goal_z_dataset(
            model, planner, episodes, x_seq_full,
            save_path=args.goal_z_path, device=args.device)
        print("Done.")
        return

    episodes = episodes[:args.n_ep]
    results  = run_lqr_on_episodes(
        planner, episodes, x_seq_full,
        cond_len=args.cond_len, horizon=args.horizon,
        device=args.device)

    visualize_lqr_results(results, args.out_dir)
    print(f"\nDone → {args.out_dir}/")


if __name__ == '__main__':
    main()
