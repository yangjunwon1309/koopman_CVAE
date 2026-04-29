"""
iql_koopman.py  — KODAQ Offline IQL v2
========================================

변경사항:
  1. Q(z_t, a_t, skill_probs): skill prior w_k(h_t) 추가 입력
  2. compute_r_blend(): episode reward diff + event/acc reward blend
  3. GaussianPolicy: H_lo=16 action chunk 출력 (robot action space)
  4. AWR → GAE (Generalized Advantage Estimation) policy update
  5. evaluate_policy() 삭제 (eval_policy.py 사용)
  6. OOP 리팩토링: KODAQOfflineIQL 메인 클래스

Usage:
    python iql_koopman.py \\
        --ckpt   checkpoints/kodaq_v4/final.pt \\
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \\
        --out_dir checkpoints/kodaq_v4/iql_v3 \\
        --device cuda:1
"""

import os, sys, time, math
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import argparse
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from models.koopman_cvae import KoopmanCVAE
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    KODAQLQRPlanner, LQRConfig,
    load_kitchen_episodes, obs_to_x_goal,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IQLConfig:
    # IQL
    tau:          float = 0.8    # expectile for V
    gamma:        float = 0.7    # discount (낮게 → bootstrap 안정)
    gae_lambda:   float = 0.95   # GAE lambda

    # Policy chunk
    H_lo:         int   = 4      # action chunk length (0.32s)

    # H-step TD
    H:            int   = 8      # LQR rollout horizon

    # Networks
    hidden_dim:   int   = 256
    n_layers:     int   = 2

    # Optimization
    lr:           float = 3e-4
    batch_size:   int   = 256
    n_steps:      int   = 500_000
    target_ema:   float = 0.005
    grad_clip:    float = 1.0

    # Data
    real_ratio:   float = 0.5

    # Reward
    w_env:        float = 0.4
    w_event:      float = 0.2
    w_acc:        float = 0.4

    # Logging
    log_every:    int   = 1_000
    save_every:   int   = 50_000

    # LQR cache
    n_ep_lqr:     int   = 500
    lqr_quality:  str   = 'mixed'
    cond_len:     int   = 16


# ─────────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_r_blend(r_env: float, r_hat_event: float = 0.0,
                    r_hat_acc: float = 0.0,
                    w_env: float = 0.4, w_event: float = 0.2,
                    w_acc: float = 0.4) -> float:
    """
    3-way reward blend.
    r_env:        sparse env reward (diff된 0/1 값)
    r_hat_event:  BCE event reward head output ∈ (0,1)
    r_hat_acc:    categorical accumulated reward head ∈ [0,4], /4 정규화
    """
    MAX_ACC = 4.0
    r = (w_env   * float(r_env) +
         w_event * float(r_hat_event) +
         w_acc   * (float(r_hat_acc) / MAX_ACC))
    return float(np.clip(r, 0.0, 1.0))


def episode_reward_to_diff(rew_ep: np.ndarray) -> np.ndarray:
    """
    누적 reward array → diff (subtask 완료 순간만 1, 나머지 0).
    rew_ep가 0~4 누적합이면 차분, 이미 sparse면 그대로.
    """
    if rew_ep.max() <= 1.0:
        return rew_ep.astype(np.float32)
    # 누적합 → diff
    diff = np.zeros_like(rew_ep, dtype=np.float32)
    diff[0] = rew_ep[0]
    diff[1:] = np.diff(rew_ep)
    return diff.clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int,
             output_scale: float = 1.0) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ELU()]
        d = hidden
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=output_scale)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """
    Q(z_t, a_t, skill_probs) → scalar

    skill_probs: w_k(h_t) from koopman skill prior (K-dim softmax)
    이를 통해 현재 state의 skill context를 Q에 반영.
    """
    def __init__(self, z_dim: int, action_dim: int, n_skills: int,
                 hidden: int, n_layers: int):
        super().__init__()
        in_dim = z_dim + action_dim + n_skills
        self.net = make_mlp(in_dim, 1, hidden, n_layers, output_scale=0.01)

    def forward(self, z: torch.Tensor, a: torch.Tensor,
                skill_probs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a, skill_probs], dim=-1)).squeeze(-1)


class VNetwork(nn.Module):
    """V(z_t, skill_probs) → scalar"""
    def __init__(self, z_dim: int, n_skills: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim + n_skills, 1, hidden, n_layers,
                            output_scale=0.01)

    def forward(self, z: torch.Tensor,
                skill_probs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, skill_probs], dim=-1)).squeeze(-1)


class ChunkPolicy(nn.Module):
    """
    π(a_{0:H_lo} | z_t) → action chunk (H_lo, action_dim)

    IQL과 동일한 구조 (skill conditioning 없음) → online π_lo와 weight 이식 가능.
    H_lo=16: 1.28s action chunk
    """
    def __init__(self, z_dim: int, action_dim: int, H_lo: int,
                 hidden: int, n_layers: int,
                 log_std_min: float = -4.0, log_std_max: float = 1.0):
        super().__init__()
        self.action_dim  = action_dim
        self.H_lo        = H_lo
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net   = make_mlp(z_dim, hidden, hidden, n_layers - 1)
        self.mu    = nn.Linear(hidden, H_lo * action_dim)
        self.log_s = nn.Linear(hidden, H_lo * action_dim)
        nn.init.uniform_(self.mu.weight, -0.01, 0.01)
        nn.init.zeros_(self.mu.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(z)
        mu   = self.mu(feat).view(-1, self.H_lo, self.action_dim)
        ls   = self.log_s(feat).view(-1, self.H_lo, self.action_dim)
        return mu, ls.clamp(self.log_std_min, self.log_std_max)

    def log_prob(self, z: torch.Tensor,
                 a_chunk: torch.Tensor) -> torch.Tensor:
        """a_chunk: (B, H_lo, action_dim) tanh-squashed"""
        mu, ls = self(z)
        u    = torch.atanh(a_chunk.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mu, ls.exp())
        lp   = dist.log_prob(u) - torch.log(1 - a_chunk.pow(2) + 1e-6)
        return lp.sum(dim=(-2, -1))  # (B,)

    def sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, ls = self(z)
        u = torch.distributions.Normal(mu, ls.exp()).rsample()
        a = torch.tanh(u)
        lp = (torch.distributions.Normal(mu, ls.exp()).log_prob(u)
              - torch.log(1 - a.pow(2) + 1e-6))
        return a, lp.sum(dim=(-2, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    real:  (z_t, a_chunk, skill_probs, z_next, r_blend)
           a_chunk: (H_lo, 9) consecutive actions from offline data
           skill_probs: w_k(h_t) from skill prior
    lqr:   (z0, skill_probs0, z_hat_seq, r_hat_seq, r_real_seq)
    """
    def __init__(self, device: str):
        self.device = device
        self._real: Dict[str, np.ndarray] = {}
        self._lqr:  Dict[str, np.ndarray] = {}
        self._rn = self._ln = 0

    def _init_real(self, z_dim, action_dim, H_lo, n_skills, n):
        self._real = {
            'z':      np.zeros((n, z_dim),          dtype=np.float32),
            'a':      np.zeros((n, H_lo, action_dim),dtype=np.float32),
            'sp':     np.zeros((n, n_skills),        dtype=np.float32),
            'z_next': np.zeros((n, z_dim),           dtype=np.float32),
            'r':      np.zeros(n,                    dtype=np.float32),
        }

    def _init_lqr(self, z_dim, n_skills, H, m):
        self._lqr = {
            'z':      np.zeros((m, z_dim),    dtype=np.float32),
            'sp':     np.zeros((m, n_skills),  dtype=np.float32),
            'z_hat':  np.zeros((m, H, z_dim), dtype=np.float32),
            'r_hat':  np.zeros((m, H),        dtype=np.float32),
            'r_real': np.zeros((m, H),        dtype=np.float32),
        }

    def add_real_batch(self, z, a, sp, z_next, r):
        n = len(z)
        if not self._real: self._init_real(z.shape[1], a.shape[2], a.shape[1],
                                            sp.shape[1], n)
        for k, v in zip(['z','a','sp','z_next','r'], [z, a, sp, z_next, r]):
            self._real[k][:n] = v
        self._rn = n

    def add_lqr_batch(self, z, sp, z_hat, r_hat, r_real):
        n = len(z)
        if not self._lqr: self._init_lqr(z.shape[1], sp.shape[1],
                                          z_hat.shape[1], n)
        for k, v in zip(['z','sp','z_hat','r_hat','r_real'],
                        [z, sp, z_hat, r_hat, r_real]):
            self._lqr[k][:n] = v
        self._ln = n

    def _to_tensor(self, d, idx):
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in d.items()}

    def sample_real(self, B):
        return self._to_tensor(self._real,
                               np.random.randint(0, self._rn, B))

    def sample_lqr(self, B):
        return self._to_tensor(self._lqr,
                               np.random.randint(0, self._ln, B))

    @property
    def real_size(self): return self._rn

    @property
    def lqr_size(self): return self._ln


# ─────────────────────────────────────────────────────────────────────────────
# LQR Cache Builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_lqr_cache(model: KoopmanCVAE, planner: KODAQLQRPlanner,
                    episodes: List[Dict], x_seq_full: np.ndarray,
                    cfg: IQLConfig, device: str,
                    save_path: Optional[str] = None,
                    cat_head=None) -> Dict[str, np.ndarray]:
    """
    Sub-goal 기반 LQR rollout 캐시 생성.

    real data:
      - z_t, a_chunk(H_lo, 9), skill_probs, z_next, r_blend
      - r_blend = compute_r_blend(r_env_diff, r_event, r_acc)

    lqr data (TD target용):
      - z0, skill_probs0, z_hat_seq(H, m), r_hat_seq(H), r_real_seq(H)
    """
    dev = torch.device(device)
    model.eval()
    H, H_lo = cfg.H, cfg.H_lo
    K = model.cfg.num_skills

    # real data lists
    rz, ra, rsp, rzn, rr = [], [], [], [], []
    # lqr data lists
    lz, lsp, lzh, lrh, lrr = [], [], [], [], []

    print(f"\n=== Building LQR Cache: {len(episodes)} episodes  "
          f"H={H}  H_lo={H_lo} ===")
    total_stages = 0

    for ep_idx, ep in enumerate(episodes):
        L       = ep['length']
        obs_ep  = ep['obs']
        acts_ep = ep['actions']   # (L, 9)
        rew_ep  = ep['rewards']   # (L,) raw
        gi      = ep['goal_info']
        s_t     = ep['start_t']
        x_ep    = x_seq_full[s_t:s_t + L]

        if not ep['tasks']: continue

        # episode reward → diff (subtask completion only)
        # env reward: raw 그대로 사용 (no diff)

        # encode full episode
        x_t = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)
        a_t = torch.FloatTensor(acts_ep).unsqueeze(0).to(dev)
        enc = model.encode_sequence(x_t, a_t)
        z_ep  = enc['o_seq'][0].cpu().numpy()   # (L, m)
        h_ep  = enc['h_seq'][0]                  # (L, d_h) tensor

        # skill probs per step
        sp_ep = model.skill_prior.soft_weights(
            h_ep.to(dev)).cpu().numpy()           # (L, K)

        # event reward (BCE head)
        z_dev = torch.FloatTensor(z_ep).to(dev)
        r_event_ep = np.zeros(L, dtype=np.float32)
        if model.cfg.use_reward_head:
            if hasattr(model.decoder, 'head_reward'):
                r_event_ep = torch.sigmoid(
                    model.decoder.head_reward(z_dev)
                ).squeeze(-1).cpu().numpy()
            elif hasattr(model, 'reward_head'):
                r_event_ep = torch.sigmoid(
                    model.reward_head(z_dev)
                ).squeeze(-1).cpu().numpy()

        # accumulated reward (cat_head)
        r_acc_ep = np.zeros(L, dtype=np.float32)
        if cat_head is not None:
            r_acc_ep = cat_head.expected_reward(z_dev).cpu().numpy()

        # ── real data: H_lo-step chunks ──────────────────────────────────
        acts_clip = acts_ep.clip(-1, 1).astype(np.float32)
        for t in range(L - H_lo - 1):
            a_chunk = acts_clip[t:t + H_lo]          # (H_lo, 9)
            r_env_t = float(rew_ep[t])
            r_blend = compute_r_blend(
                r_env_t, r_event_ep[t], r_acc_ep[t],
                cfg.w_env, cfg.w_event, cfg.w_acc)
            rz.append(z_ep[t])
            ra.append(a_chunk)
            rsp.append(sp_ep[t])
            rzn.append(z_ep[t + H_lo])
            rr.append(r_blend)

        # ── lqr rollout: sub-goal stage별 ────────────────────────────────
        jump_ts   = sorted(gi['completions'].values())
        stage_ends= jump_ts + [L - 1]
        stage_start = 0

        for stage_idx, stage_end_t in enumerate(stage_ends):
            if stage_end_t - stage_start < H:
                stage_start = stage_end_t + 1; continue

            cond_s = max(0, stage_start - cfg.cond_len)
            cond_e = stage_start if stage_start > 0 else min(cfg.cond_len, stage_end_t)
            x_cond = torch.FloatTensor(x_ep[cond_s:cond_e]).unsqueeze(0).to(dev)
            a_cond = torch.FloatTensor(acts_ep[cond_s:cond_e]).unsqueeze(0).to(dev)
            x_goal_t = torch.FloatTensor(
                obs_to_x_goal(obs_ep[stage_end_t], obs_ep[0])
            ).unsqueeze(0).to(dev)

            try:
                plan  = planner.plan(x_cond, a_cond, x_goal_t,
                                     horizon=H, compute_uncertainty=False)
            except Exception:
                stage_start = stage_end_t + 1; continue

            o_traj = plan['o_traj'].cpu()   # (H+1, m)
            z0_np  = o_traj[0].numpy()
            z_hat  = o_traj[1:].to(dev)     # (H, m)

            # event reward along LQR rollout
            r_hat_seq = np.zeros(H, dtype=np.float32)
            if model.cfg.use_reward_head:
                if hasattr(model.decoder, 'head_reward'):
                    r_hat_seq = torch.sigmoid(
                        model.decoder.head_reward(z_hat)
                    ).squeeze(-1).cpu().numpy()

            # real env reward along this stage (broadcast)
            r_real_stage = float(rew_ep[stage_start:stage_end_t].sum())
            r_real_seq   = np.full(H, r_real_stage / max(H, 1), dtype=np.float32)

            # acc reward proxy: use stage start z's cat_head value
            r_acc_stage = float(r_acc_ep[min(stage_start, L-1)]) if cat_head is not None else 0.0
            # blend for each step (acc reward included)
            r_blend_seq = np.array([
                compute_r_blend(r_real_seq[k], r_hat_seq[k], r_acc_stage,
                                cfg.w_env, cfg.w_event, cfg.w_acc)
                for k in range(H)], dtype=np.float32)

            # skill probs at z0
            enc0 = model.encode_sequence(x_cond, a_cond)
            h0   = enc0['h_seq'][0, -1:]
            sp0  = model.skill_prior.soft_weights(h0.to(dev)).cpu().numpy()[0]

            lz.append(z0_np)
            lsp.append(sp0)
            lzh.append(z_hat.cpu().numpy())
            lrh.append(r_blend_seq)
            lrr.append(r_real_seq)
            total_stages += 1
            stage_start = stage_end_t + 1

        if (ep_idx + 1) % 50 == 0:
            print(f"  Ep {ep_idx+1}/{len(episodes)}  "
                  f"stages={total_stages}  real={len(rz)}")

    print(f"\nCache built: {total_stages} LQR stages, {len(rz)} real transitions")

    cache = {
        'z_real':    np.array(rz,  dtype=np.float32),
        'a_real':    np.array(ra,  dtype=np.float32),  # (N, H_lo, 9)
        'sp_real':   np.array(rsp, dtype=np.float32),  # (N, K)
        'z_next_real':np.array(rzn, dtype=np.float32),
        'r_real':    np.array(rr,  dtype=np.float32),
        'z0':        np.array(lz,  dtype=np.float32),
        'sp0':       np.array(lsp, dtype=np.float32),
        'z_hat_seq': np.array(lzh, dtype=np.float32),  # (N, H, m)
        'r_hat_seq': np.array(lrh, dtype=np.float32),  # (N, H) blended
        'r_real_seq':np.array(lrr, dtype=np.float32),
    }
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **cache)
        print(f"Saved → {save_path}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# KODAQ Offline IQL
# ─────────────────────────────────────────────────────────────────────────────

class KODAQOfflineIQL:
    """
    KODAQ Offline IQL v2

    Q(z, a_chunk, skill_probs), V(z, skill_probs), π(a_chunk | z)
    GAE policy update
    """
    def __init__(self, cfg: IQLConfig, model: KoopmanCVAE,
                 z_dim: int, action_dim: int, n_skills: int,
                 device: str):
        self.cfg      = cfg
        self.model    = model
        self.device   = device
        self.z_dim    = z_dim
        self.a_dim    = action_dim
        self.n_skills = n_skills
        self.step     = 0

        h, nl = cfg.hidden_dim, cfg.n_layers

        # Networks
        self.Q1   = QNetwork(z_dim, action_dim, n_skills, h, nl).to(device)
        self.Q2   = QNetwork(z_dim, action_dim, n_skills, h, nl).to(device)
        self.Q1_t = QNetwork(z_dim, action_dim, n_skills, h, nl).to(device)
        self.Q2_t = QNetwork(z_dim, action_dim, n_skills, h, nl).to(device)
        self.Q1_t.load_state_dict(self.Q1.state_dict())
        self.Q2_t.load_state_dict(self.Q2.state_dict())

        self.V  = VNetwork(z_dim, n_skills, h, nl).to(device)
        self.pi = ChunkPolicy(z_dim, action_dim, cfg.H_lo, h, nl).to(device)

        lr = cfg.lr
        self.opt_q  = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=lr)
        self.opt_v  = torch.optim.Adam(self.V.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)

        self.buf = ReplayBuffer(device)
        self.cat_head = None  # optional CategoricalRewardHead

    # ── TD Target ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _td_target(self, lqr_b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        H-step discounted return + bootstrap V.
        y_t = Σ γ^k r_blend_k + γ^H V(z_H, sp_H)
        """
        z_hat = lqr_b['z_hat']    # (B, H, m)
        r_hat = lqr_b['r_hat']    # (B, H) blended
        z0    = lqr_b['z']
        sp0   = lqr_b['sp']
        B, H, m = z_hat.shape
        gm    = self.cfg.gamma

        gm_pw = torch.tensor([gm**k for k in range(H)],
                             dtype=torch.float32, device=self.device)
        r_sum = (r_hat * gm_pw.unsqueeze(0)).sum(dim=1)   # (B,)

        # skill probs at z_H (use sp0 as proxy — h_t not stored in lqr cache)
        z_H   = z_hat[:, -1]
        v_H   = self.V(z_H, sp0)
        r_max = sum(gm**k for k in range(H))
        y_t   = (r_sum + gm**H * v_H).clamp(0.0, r_max)
        return y_t

    # ── GAE ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _gae(self, real_b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generalized Advantage Estimation for policy update.

        각 transition에서 H_lo step만큼의 advantage를 계산:
          δ_t = r_t + γ V(z_{t+H_lo}) - V(z_t)
          GAE = Σ (γλ)^k δ_{t+k}  (단일 step이라 δ_t 그대로 사용)

        multi-step GAE를 위해 z_hat을 따라가는 형태:
          Q_target에서 V를 빼서 advantage 계산
        """
        z  = real_b['z'];  sp = real_b['sp']
        zn = real_b['z_next']; r = real_b['r']

        gm, lam = self.cfg.gamma, self.cfg.gae_lambda

        q_min  = torch.min(self.Q1_t(z, real_b['a'][:,0,:], sp),
                           self.Q2_t(z, real_b['a'][:,0,:], sp))
        v_cur  = self.V(z, sp)
        v_next = self.V(zn, sp)  # sp_next 없으므로 sp로 근사

        delta  = r + gm * v_next - v_cur
        # single-step GAE (multi-step은 sequential data 필요)
        adv    = delta
        return adv

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, real_b: Dict[str, torch.Tensor],
               lqr_b:  Dict[str, torch.Tensor]) -> Dict[str, float]:

        z  = real_b['z'];   sp = real_b['sp']
        a  = real_b['a']    # (B, H_lo, 9)
        zn = real_b['z_next']; r = real_b['r']

        # ── Q loss ───────────────────────────────────────────────────────
        y_t = self._td_target(lqr_b)          # (B,)
        # Q uses first action of chunk as representative
        a0  = a[:, 0, :]                       # (B, 9)
        q1  = self.Q1(z, a0, sp)
        q2  = self.Q2(z, a0, sp)
        loss_q = F.mse_loss(q1, y_t) + F.mse_loss(q2, y_t)
        self.opt_q.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.cfg.grad_clip)
        self.opt_q.step()

        # ── V loss (expectile) ────────────────────────────────────────────
        with torch.no_grad():
            q_min = torch.min(self.Q1_t(z, a0, sp),
                              self.Q2_t(z, a0, sp))
        v   = self.V(z, sp)
        adv = q_min - v
        tau = self.cfg.tau
        w   = torch.where(adv >= 0,
                          torch.full_like(adv, tau),
                          torch.full_like(adv, 1 - tau))
        loss_v = (w * adv.pow(2)).mean()
        self.opt_v.zero_grad(); loss_v.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), self.cfg.grad_clip)
        self.opt_v.step()

        # ── Policy loss (GAE-weighted log prob) ───────────────────────────
        with torch.no_grad():
            gae = self._gae(real_b)            # (B,)
            # normalize advantage
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            # positive-only weighting (IQL style)
            w_pi = gae.clamp(min=0.0)

        log_prob = self.pi.log_prob(z, a)      # (B,)
        loss_pi  = -(w_pi * log_prob).mean()
        self.opt_pi.zero_grad(); loss_pi.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.grad_clip)
        self.opt_pi.step()

        # ── Soft update ───────────────────────────────────────────────────
        ema = self.cfg.target_ema
        for p, pt in zip(self.Q1.parameters(), self.Q1_t.parameters()):
            pt.data.mul_(1 - ema).add_(p.data, alpha=ema)
        for p, pt in zip(self.Q2.parameters(), self.Q2_t.parameters()):
            pt.data.mul_(1 - ema).add_(p.data, alpha=ema)

        self.step += 1
        return {
            'loss_q':  loss_q.item(),
            'loss_v':  loss_v.item(),
            'loss_pi': loss_pi.item(),
            'q_mean':  q_min.mean().item(),
            'v_mean':  v.mean().item(),
            'adv_mean':adv.mean().item(),
            'r_target':y_t.mean().item(),
            'gae_mean':gae.mean().item(),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': self.step,
            'Q1':   self.Q1.state_dict(),  'Q2':  self.Q2.state_dict(),
            'Q1_t': self.Q1_t.state_dict(),'Q2_t':self.Q2_t.state_dict(),
            'V':    self.V.state_dict(),
            'pi':   self.pi.state_dict(),
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.Q1.load_state_dict(ck['Q1']);  self.Q2.load_state_dict(ck['Q2'])
        self.Q1_t.load_state_dict(ck.get('Q1_t', ck['Q1']))
        self.Q2_t.load_state_dict(ck.get('Q2_t', ck['Q2']))
        self.V.load_state_dict(ck['V'])
        self.pi.load_state_dict(ck['pi'])
        self.step = ck.get('step', 0)
        print(f"  Loaded: {path}  step={self.step}")
        return self.step


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_training(log: Dict[str, List], out_path: str):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys   = ['loss_q','loss_v','loss_pi','q_mean','v_mean','adv_mean','r_target']
        titles = ['Q Loss','V Loss','π Loss','Q mean','V mean','Adv mean','TD Target']
        PAL    = ['#E53935','#1E88E5','#43A047','#FB8C00',
                  '#8E24AA','#00ACC1','#FFB300']
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, (k, t) in enumerate(zip(keys, titles)):
            if not log.get(k): continue
            vals = np.array(log[k]); ax = axes[i]
            ax.plot(vals, color=PAL[i], alpha=0.25, lw=0.8)
            w = max(1, min(50, len(vals)//5))
            if len(vals) >= w:
                ax.plot(np.convolve(vals, np.ones(w)/w, 'valid'),
                        color=PAL[i], lw=1.8)
            ax.set_title(t, fontsize=9, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('KODAQ Offline IQL v2', fontsize=12, fontweight='bold')
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  [Vis] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',     default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--lqr_cache',   default=None)
    p.add_argument('--iql_ckpt',    default=None)
    p.add_argument('--cat_ckpt',    default=None)
    p.add_argument('--out_dir',     default='checkpoints/kodaq_v4/iql_v3')
    p.add_argument('--quality',     default='mixed')
    p.add_argument('--n_ep_lqr',    type=int,   default=500)
    p.add_argument('--H',           type=int,   default=8)
    p.add_argument('--H_lo',        type=int,   default=4)
    p.add_argument('--tau',         type=float, default=0.8)
    p.add_argument('--gamma',       type=float, default=0.7)
    p.add_argument('--gae_lambda',  type=float, default=0.95)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--batch_size',  type=int,   default=256)
    p.add_argument('--n_steps',     type=int,   default=500_000)
    p.add_argument('--real_ratio',  type=float, default=0.5)
    p.add_argument('--w_env',       type=float, default=0.4)
    p.add_argument('--w_event',     type=float, default=0.2)
    p.add_argument('--w_acc',       type=float, default=0.4)
    p.add_argument('--Q_scale',     type=float, default=1.0)
    p.add_argument('--R_scale',     type=float, default=10.0)
    p.add_argument('--device',      default='cuda:1' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run',   default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"Device: {device}")

    # wandb
    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run or 'iql_v3',
                   config=vars(args))

    # World model
    print(f"\nLoading: {args.ckpt}")
    ck    = torch.load(args.ckpt, map_location=device)
    model = KoopmanCVAE(ck['cfg'])
    model.load_state_dict(ck['model_state'])
    model.eval().to(device)
    z_dim    = model.cfg.koopman_dim
    a_dim    = model.cfg.action_dim
    n_skills = model.cfg.num_skills
    print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")

    # cat_head
    cat_head = None
    cat_path = args.cat_ckpt or str(
        Path(args.ckpt).parent / 'cat_reward' / 'final.pt')
    if Path(cat_path).exists():
        from train_reward_head import load_cat_reward_model
        _, cat_head = load_cat_reward_model(cat_path, device)
        print(f"  CategoricalRewardHead loaded")

    # Config
    cfg = IQLConfig(
        H=args.H, H_lo=args.H_lo, tau=args.tau, gamma=args.gamma,
        gae_lambda=args.gae_lambda, lr=args.lr,
        batch_size=args.batch_size, n_steps=args.n_steps,
        real_ratio=args.real_ratio,
        w_env=args.w_env, w_event=args.w_event, w_acc=args.w_acc,
    )

    planner = KODAQLQRPlanner(model,
                               LQRConfig(Q_scale=args.Q_scale,
                                         R_scale=args.R_scale))

    # Data
    print(f"\nLoading x_sequences: {args.x_cache}")
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality, min_len=32)
    eps_lqr          = [e for e in episodes if e['tasks']][:args.n_ep_lqr]
    print(f"  Episodes with tasks: {len(eps_lqr)}")

    # LQR cache
    cache_path = args.lqr_cache or f"{args.out_dir}/lqr_cache.npz"
    if args.lqr_cache and Path(args.lqr_cache).exists():
        print(f"\nLoading LQR cache: {args.lqr_cache}")
        cache = dict(np.load(args.lqr_cache))
        # check if new format (has skill probs)
        if 'sp_real' not in cache:
            print("  Old cache format detected — rebuilding with skill probs...")
            cache = build_lqr_cache(model, planner, eps_lqr, x_seq_full,
                                    cfg, device, cache_path, cat_head)
    else:
        cache = build_lqr_cache(model, planner, eps_lqr, x_seq_full,
                                cfg, device, cache_path, cat_head)

    # Trainer
    trainer = KODAQOfflineIQL(cfg, model, z_dim, a_dim, n_skills, device)
    trainer.cat_head = cat_head
    if args.iql_ckpt and Path(args.iql_ckpt).exists():
        trainer.load(args.iql_ckpt)

    # Buffer
    buf = trainer.buf
    buf.add_real_batch(
        z    = cache['z_real'],
        a    = cache['a_real'],
        sp   = cache['sp_real'],
        z_next=cache['z_next_real'],
        r    = cache['r_real'],
    )
    buf.add_lqr_batch(
        z    = cache['z0'],
        sp   = cache['sp0'],
        z_hat= cache['z_hat_seq'],
        r_hat= cache['r_hat_seq'],
        r_real=cache['r_real_seq'],
    )
    print(f"\nBuffer — real: {buf.real_size}  lqr: {buf.lqr_size}")
    print(f"  r_real mean: {cache['r_real'].mean():.4f}  "
          f"max: {cache['r_real'].max():.4f}")

    # Train
    print(f"\n{'='*60}")
    print(f"KODAQ Offline IQL v2  steps={cfg.n_steps}")
    print(f"  H={cfg.H}  H_lo={cfg.H_lo}  τ={cfg.tau}  γ={cfg.gamma}")
    print(f"  r_blend: w_env={cfg.w_env} w_event={cfg.w_event} w_acc={cfg.w_acc}")
    print(f"{'='*60}\n")

    log_keys = ['loss_q','loss_v','loss_pi','q_mean','v_mean',
                'adv_mean','r_target','gae_mean']
    log    = {k: [] for k in log_keys}
    recent = {k: deque(maxlen=cfg.log_every) for k in log_keys}
    B_real = max(1, int(cfg.batch_size * cfg.real_ratio))
    B_lqr  = cfg.batch_size - B_real
    t0 = time.time()

    start_step = trainer.step
    for step in range(start_step, cfg.n_steps):
        real_b = buf.sample_real(B_real)
        lqr_b  = buf.sample_lqr(B_lqr)
        info   = trainer.update(real_b, lqr_b)
        for k, v in info.items():
            if k in recent: recent[k].append(v)

        if (step + 1) % cfg.log_every == 0:
            ms  = {k: np.mean(list(recent[k])) if recent[k] else 0.0
                   for k in log_keys}
            for k in log_keys: log[k].append(ms[k])
            sps = cfg.log_every / (time.time() - t0 + 1e-6); t0 = time.time()
            print(f"Step {step+1:7d} | "
                  f"Q={ms['loss_q']:.4f} V={ms['loss_v']:.4f} "
                  f"π={ms['loss_pi']:.4f} | "
                  f"q={ms['q_mean']:.3f} v={ms['v_mean']:.3f} "
                  f"adv={ms['adv_mean']:.3f} | "
                  f"r_target={ms['r_target']:.3f} | "
                  f"{sps:.0f} sps")
            if use_wandb:
                wandb.log({f'train/{k}': v for k, v in ms.items()},
                          step=step+1)

        if (step + 1) % cfg.save_every == 0:
            trainer.save(f"{args.out_dir}/iql_step{step+1}.pt")
            visualize_training(log, f"{args.out_dir}/training_curves.png")

    trainer.save(f"{args.out_dir}/iql_final.pt")
    visualize_training(log, f"{args.out_dir}/training_curves_final.png")
    print(f"\nDone. {cfg.n_steps} steps → {args.out_dir}/")
    if use_wandb: wandb.finish()


if __name__ == '__main__':
    main()