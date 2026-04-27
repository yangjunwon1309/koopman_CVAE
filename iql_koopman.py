"""
iql_koopman.py — IQL + H-step TD in Koopman Latent Space
=========================================================

학습 구조:
  - Q network: real (z_t, u_real_t) 위에서만 학습
  - TD target: Koopman world model H-step rollout으로 Σγ^k r̂ + γ^H V(z_hat_H)
  - V network: expectile regression τ=0.8
  - Policy: AWR (Advantage Weighted Regression), exp(β*A) weighted BC
  - OOD 방지: explicit penalization 없이 expectile V로 자연스럽게

Usage:
    # Step 1: LQR rollout 캐시 생성 (없으면 자동 생성)
    python iql_koopman.py \
        --ckpt   checkpoints/kodaq_v4/final.pt \
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \
        --out_dir checkpoints/kodaq_v4/iql \
        --device cuda:1

    # Step 2: 이미 캐시 있으면 바로 학습
    python iql_koopman.py \
        --ckpt       checkpoints/kodaq_v4/final.pt \
        --x_cache    checkpoints/skill_pretrain/x_sequences.npz \
        --lqr_cache  checkpoints/kodaq_v4/iql/lqr_cache.npz \
        --out_dir    checkpoints/kodaq_v4/iql \
        --device cuda:1
"""

import os, sys, time, math
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
    tau:            float = 0.8      # expectile for V (0.5=mean, 0.9=high percentile)
    beta:           float = 3.0      # AWR temperature
    gamma:          float = 0.99     # discount

    # H-step TD
    H:              int   = 8        # rollout horizon for TD target

    # networks
    hidden_dim:     int   = 256
    n_layers:       int   = 2

    # optimization
    lr_q:           float = 3e-4
    lr_v:           float = 3e-4
    lr_pi:          float = 3e-4
    batch_size:     int   = 256
    n_steps:        int   = 500_000
    target_ema:     float = 0.005    # soft update coefficient

    # data
    real_ratio:     float = 0.5      # fraction of real data in each batch
    # remaining (1 - real_ratio) comes from LQR cache for TD target only

    # reward normalization
    reward_scale:   float = 10.0     # multiply sparse 0/1 reward
    reward_min:     float = 0.0
    reward_max:     float = 1.0

    # logging
    log_every:      int   = 1_000
    save_every:     int   = 50_000

    # LQR cache generation
    n_ep_lqr:       int   = 500      # episodes for LQR cache
    lqr_quality:    str   = 'mixed'

    # policy evaluation
    eval_every:     int   = 50_000
    n_eval_ep:      int   = 5


# ─────────────────────────────────────────────────────────────────────────────
# Networks (모두 Koopman latent space에서 작동)
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int,
             output_activation=None) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q(z_t, u_t) → scalar"""
    def __init__(self, z_dim: int, u_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim + u_dim, 1, hidden, n_layers)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, u], dim=-1)).squeeze(-1)  # (B,)


class VNetwork(nn.Module):
    """V(z_t) → scalar"""
    def __init__(self, z_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim, 1, hidden, n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


class GaussianPolicy(nn.Module):
    """π(u | z_t) — Gaussian policy in latent action space"""
    def __init__(self, z_dim: int, u_dim: int, hidden: int, n_layers: int,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.u_dim = u_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net   = make_mlp(z_dim, hidden, hidden, n_layers - 1)
        self.mu    = nn.Linear(hidden, u_dim)
        self.log_s = nn.Linear(hidden, u_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(z)
        mu   = self.mu(feat)
        log_std = self.log_s(feat).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def log_prob(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        mu, log_std = self(z)
        dist = torch.distributions.Normal(mu, log_std.exp())
        return dist.log_prob(u).sum(dim=-1)  # (B,)

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        mu, log_std = self(z)
        return torch.distributions.Normal(mu, log_std.exp()).rsample()


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    두 종류의 데이터를 별도로 저장:
      real:  (z_t, u_real_t, z_{t+1}, r_t)           — Q/V/π 학습용
      lqr:   (z_t, z_hat_{1..H}, r_hat_{1..H})        — TD target 계산용
    """
    def __init__(self, device: str):
        self.device = device
        self.real: Dict[str, np.ndarray] = {}
        self.lqr:  Dict[str, np.ndarray] = {}
        self._real_n = 0
        self._lqr_n  = 0

    def add_real(self, z_t, u_t, z_next, r_t):
        """실제 데이터 추가"""
        if not self.real:
            self.real = {
                'z':      np.empty((len(z_t), z_t.shape[1]), dtype=np.float32),
                'u':      np.empty((len(u_t), u_t.shape[1]), dtype=np.float32),
                'z_next': np.empty((len(z_next), z_next.shape[1]), dtype=np.float32),
                'r':      np.empty(len(r_t), dtype=np.float32),
            }
        n = len(z_t)
        self.real['z'][:n]      = z_t
        self.real['u'][:n]      = u_t
        self.real['z_next'][:n] = z_next
        self.real['r'][:n]      = r_t
        self._real_n = n

    def add_lqr(self, z_t, z_hat_seq, r_hat_seq):
        """
        z_t:      (N, m)
        z_hat_seq: (N, H, m)   — Koopman rollout states
        r_hat_seq: (N, H)      — reward head predictions
        """
        if not self.lqr:
            N, H, m = z_hat_seq.shape
            self.lqr = {
                'z':         np.empty((N, m),    dtype=np.float32),
                'z_hat_seq': np.empty((N, H, m), dtype=np.float32),
                'r_hat_seq': np.empty((N, H),    dtype=np.float32),
            }
        n = len(z_t)
        self.lqr['z'][:n]         = z_t
        self.lqr['z_hat_seq'][:n] = z_hat_seq
        self.lqr['r_hat_seq'][:n] = r_hat_seq
        self._lqr_n = n

    def sample_real(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._real_n, batch_size)
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in self.real.items()}

    def sample_lqr(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._lqr_n, batch_size)
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in self.lqr.items()}

    @property
    def real_size(self): return self._real_n

    @property
    def lqr_size(self): return self._lqr_n


# ─────────────────────────────────────────────────────────────────────────────
# Reward Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class RewardNormalizer:
    """
    H-step discounted reward를 러닝 통계로 정규화.
    BCE reward (0/1 sparse) → Σγ^k r̂ 범위가 불안정하므로 필수.
    """
    def __init__(self, clip: float = 10.0, momentum: float = 0.001):
        self.mean   = 0.0
        self.var    = 1.0
        self.count  = 0
        self.clip   = clip
        self.momentum = momentum

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var  = x.var() + 1e-8
        self.mean  = (1 - self.momentum) * self.mean + self.momentum * batch_mean
        self.var   = (1 - self.momentum) * self.var  + self.momentum * batch_var

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.var) + 1e-8
        x_n = (x - self.mean) / std
        return x_n.clamp(-self.clip, self.clip)


# ─────────────────────────────────────────────────────────────────────────────
# LQR Cache Builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_lqr_cache(
    model:      KoopmanCVAE,
    planner:    KODAQLQRPlanner,
    episodes:   List[Dict],
    x_seq_full: np.ndarray,
    H:          int   = 8,
    device:     str   = 'cuda',
    save_path:  str   = None,
) -> Dict[str, np.ndarray]:
    """
    Sub-goal 기반 LQR rollout 캐시 생성.

    각 stage에서:
      - context: [stage_start - cond_len : stage_start]
      - goal:    obs[stage_end]
      - LQR rollout H 스텝
      - 저장: (z_0, z_hat_{1..H}, r_hat_{1..H})

    z는 이미 encode_sequence()로 인코딩된 상태.
    r_hat은 model.decoder의 reward head 예측값.
    """
    dev = torch.device(device)
    model.eval()

    all_z0        = []   # (N, m)
    all_z_hat_seq = []   # (N, H, m)
    all_r_hat_seq = []   # (N, H)

    # real data용
    all_z_real      = []  # (N, m)
    all_u_real      = []  # (N, d_u)
    all_z_next_real = []  # (N, m)
    all_r_real      = []  # (N,)

    cond_len = 16
    total_stages = 0

    print(f"\n=== Building LQR Cache: {len(episodes)} episodes, H={H} ===")

    for ep_idx, ep in enumerate(episodes):
        L        = ep['length']
        obs_ep   = ep['obs']       # (L, 60)
        acts_ep  = ep['actions']   # (L, 9)
        rew_ep   = ep['rewards']   # (L,)
        gi       = ep['goal_info']
        s_t      = ep['start_t']
        x_ep     = x_seq_full[s_t:s_t + L]  # (L, 2108)

        if not ep['tasks']:
            continue

        # ── Real data: 에피소드 전체를 consecutive pair로 저장 ──────────────
        # encode 전체 시퀀스 한 번에
        x_ep_t  = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)     # (1, L, 2108)
        a_ep_t  = torch.FloatTensor(acts_ep).unsqueeze(0).to(dev)   # (1, L, 9)
        enc     = model.encode_sequence(x_ep_t, a_ep_t)
        z_ep    = enc['o_seq'][0].cpu().numpy()    # (L, m)
        h_ep    = enc['h_seq'][0]                  # (L, d_h)

        # real (z_t, a_t, z_{t+1}, r_t) — robot action 직접 저장
        acts_clipped = acts_ep.clip(-1.0, 1.0).astype(np.float32)

        for t in range(L - 1):
            all_z_real.append(z_ep[t])
            all_u_real.append(acts_clipped[t])   # robot action (9-dim)
            all_z_next_real.append(z_ep[t + 1])
            all_r_real.append(float(rew_ep[t]))

        # ── LQR rollout: sub-goal stage별 ──────────────────────────────────
        jump_ts    = sorted(gi['completions'].values())
        stage_ends = jump_ts + [L - 1]

        stage_start = 0
        for stage_idx, stage_end_t in enumerate(stage_ends):
            stage_len = stage_end_t - stage_start
            if stage_len < H:
                stage_start = stage_end_t + 1
                continue

            # conditioning
            cond_s = max(0, stage_start - cond_len)
            cond_e = stage_start if stage_start > 0 else min(cond_len, stage_end_t)

            x_cond = torch.FloatTensor(x_ep[cond_s:cond_e]).unsqueeze(0).to(dev)
            a_cond = torch.FloatTensor(acts_ep[cond_s:cond_e]).unsqueeze(0).to(dev)

            goal_obs  = obs_ep[stage_end_t]
            ref_obs   = obs_ep[0]
            x_goal_np = obs_to_x_goal(goal_obs, ref_obs)
            x_goal_t  = torch.FloatTensor(x_goal_np).unsqueeze(0).to(dev)

            # LQR plan
            try:
                plan = planner.plan(
                    x_cond, a_cond, x_goal_t,
                    horizon=H,
                    compute_uncertainty=False,
                )
            except Exception as e:
                print(f"  Ep {ep_idx} stage {stage_idx} plan failed: {e}")
                stage_start = stage_end_t + 1
                continue

            # z_hat rollout: (H+1, m) → 0번이 z_0, 1..H가 z_hat_{1..H}
            o_traj = plan['o_traj'].cpu()   # (H+1, m)
            z0     = o_traj[0].numpy()      # (m,)
            z_hat  = o_traj[1:].numpy()     # (H, m)

            # reward head로 r_hat 계산
            with torch.no_grad():
                recon = model.decoder(o_traj[1:].to(dev))   # (H,) heads
                # reward head: BCE logit → sigmoid → probability
                if 'reward' in recon:
                    r_hat = torch.sigmoid(recon['reward']).squeeze(-1).cpu().numpy()
                else:
                    # reward head 없으면 delta_q 변화량으로 proxy
                    dq = symexp(recon['q'])  # (H, 9)
                    r_hat = dq.abs().mean(-1).cpu().numpy()

            all_z0.append(z0)
            all_z_hat_seq.append(z_hat)          # (H, m)
            all_r_hat_seq.append(r_hat)          # (H,)
            total_stages += 1

            stage_start = stage_end_t + 1

        if (ep_idx + 1) % 50 == 0:
            print(f"  Ep {ep_idx+1}/{len(episodes)}  stages so far: {total_stages}")

    print(f"\nCache built: {total_stages} LQR stages, "
          f"{len(all_z_real)} real transitions")

    cache = {
        # LQR cache
        'z0':         np.array(all_z0,        dtype=np.float32),   # (N_lqr, m)
        'z_hat_seq':  np.array(all_z_hat_seq, dtype=np.float32),   # (N_lqr, H, m)
        'r_hat_seq':  np.array(all_r_hat_seq, dtype=np.float32),   # (N_lqr, H)
        # Real transitions
        'z_real':     np.array(all_z_real,      dtype=np.float32), # (N_real, m)
        'u_real':     np.array(all_u_real,      dtype=np.float32), # (N_real, d_u)
        'z_next_real': np.array(all_z_next_real, dtype=np.float32),# (N_real, m)
        'r_real':     np.array(all_r_real,      dtype=np.float32), # (N_real,)
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **cache)
        print(f"Saved cache → {save_path}")

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# IQL Trainer
# ─────────────────────────────────────────────────────────────────────────────

class IQLTrainer:
    def __init__(
        self,
        cfg:        IQLConfig,
        z_dim:      int,
        action_dim: int,       # robot action dim (9)
        device:     str,
    ):
        self.cfg    = cfg
        self.device = device

        # Networks — robot action space
        self.Q1 = QNetwork(z_dim, action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.Q2 = QNetwork(z_dim, action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.Q1_target = QNetwork(z_dim, action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.Q2_target = QNetwork(z_dim, action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.V  = VNetwork(z_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.pi = GaussianPolicy(z_dim, action_dim, cfg.hidden_dim, cfg.n_layers).to(device)

        # Copy weights to targets
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        # Optimizers
        self.opt_q  = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=cfg.lr_q)
        self.opt_v  = torch.optim.Adam(self.V.parameters(),  lr=cfg.lr_v)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        # Reward normalizer
        self.r_norm = RewardNormalizer()

        # CategoricalRewardHead (optional, 3-way reward)
        self.cat_head = None

        # Logging
        self.log_history = {
            'loss_q': [], 'loss_v': [], 'loss_pi': [],
            'q_mean': [], 'v_mean': [], 'adv_mean': [],
            'r_target_mean': [],
        }

    @torch.no_grad()
    def _compute_h_step_target(
        self,
        z_hat_seq:  torch.Tensor,  # (B, H, m)  Koopman rollout states
        r_hat_seq:  torch.Tensor,  # (B, H)     BCE event reward
        r_real_seq: torch.Tensor,  # (B, H)     offline real env reward
        cat_head=None,             # CategoricalRewardHead (optional)
    ) -> torch.Tensor:
        """
        H-step discounted return + bootstrapped V.

        r_blend_t = 0.5*r_env + 0.2*(r_acc/4) + 0.3*r_event
        y_t = sum_{k=0}^{H-1} gamma^k * normalize(r_blend_{t+k}) + gamma^H * V(z_H)

        IQL/Online 통일 reward 구조.
        """
        H   = self.cfg.H
        gm  = self.cfg.gamma
        dev = self.device

        gm_powers = torch.tensor(
            [gm**k for k in range(H)], dtype=torch.float32, device=dev)

        # r_acc: categorical head expected reward (없으면 event reward로 대체)
        if cat_head is not None:
            B, Hs, m = z_hat_seq.shape
            z_flat  = z_hat_seq.reshape(B * Hs, m)
            r_acc   = cat_head.expected_reward(z_flat).reshape(B, Hs)  # (B, H) in [0,4]
        else:
            r_acc   = r_hat_seq * 4.0   # BCE proxy

        # 3-way blend (per step)
        r_blend = (0.5 * r_real_seq +
                   0.2 * (r_acc / 4.0) +
                   0.3 * r_hat_seq)           # (B, H)
        r_blend = r_blend.clamp(0.0, 1.0)

        # normalize per batch
        r_normalized = self.r_norm.normalize(r_blend)

        r_sum = (r_normalized * gm_powers.unsqueeze(0)).sum(dim=1)  # (B,)

        z_H = z_hat_seq[:, -1]
        v_H = self.V(z_H)
        y_t = r_sum + (gm**H) * v_H
        return y_t.detach()

    def update(
        self,
        real_batch: Dict[str, torch.Tensor],
        lqr_batch:  Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single update step.

        real_batch: z, u, z_next, r  — Q/V/π 학습용
        lqr_batch:  z0, z_hat_seq, r_hat_seq — TD target 계산용
        """
        z_r    = real_batch['z']        # (B, m)
        u_r    = real_batch['u']        # (B, d_u)
        z_r_nxt= real_batch['z_next']   # (B, m)
        r_r    = real_batch['r']        # (B,)

        z_l    = lqr_batch['z']         # (B, m)
        z_hat  = lqr_batch['z_hat_seq'] # (B, H, m)
        r_hat  = lqr_batch['r_hat_seq'] # (B, H)

        # ── H-step TD target (3-way reward) ──────────────────────────────
        # r_real_seq: r_r를 H step에 broadcast (offline 데이터의 단일 step reward)
        # 실제로는 LQR rollout의 각 step reward가 이상적이나
        # offline cache에 없으므로 현재 step reward를 H step에 동일하게 사용
        r_real_seq = r_r.unsqueeze(1).expand(-1, self.cfg.H)  # (B, H)

        # cat_head가 있으면 accumulated reward 사용
        cat_head = getattr(self, 'cat_head', None)
        y_lqr  = self._compute_h_step_target(
            z_hat, r_hat, r_real_seq, cat_head=cat_head)   # (B,)

        # y_t = y_lqr (3-way reward로 통일, 1-step 혼합 제거)
        y_t = y_lqr

        # ── Q loss: real (z, u) 위에서만 ────────────────────────────────
        q1 = self.Q1(z_r, u_r)   # (B,)
        q2 = self.Q2(z_r, u_r)   # (B,)

        loss_q = F.mse_loss(q1, y_t) + F.mse_loss(q2, y_t)

        self.opt_q.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), 1.0)
        self.opt_q.step()

        # ── V loss: expectile regression ─────────────────────────────────
        with torch.no_grad():
            q_min = torch.min(
                self.Q1_target(z_r, u_r),
                self.Q2_target(z_r, u_r),
            )  # (B,)

        v     = self.V(z_r)            # (B,)
        adv   = q_min - v              # (B,)
        τ     = self.cfg.tau

        # Asymmetric L2: τ for positive adv, (1-τ) for negative
        weight   = torch.where(adv >= 0,
                               torch.full_like(adv, τ),
                               torch.full_like(adv, 1 - τ))
        loss_v   = (weight * adv**2).mean()

        self.opt_v.zero_grad()
        loss_v.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), 1.0)
        self.opt_v.step()

        # ── Policy loss: AWR ──────────────────────────────────────────────
        with torch.no_grad():
            q_min_pi = torch.min(
                self.Q1(z_r, u_r),
                self.Q2(z_r, u_r),
            )
            v_pi   = self.V(z_r)
            adv_pi = q_min_pi - v_pi                         # (B,)
            # exp(β * A), clamped for stability
            w_pi   = torch.exp(self.cfg.beta * adv_pi).clamp(max=100.0)

        log_prob = self.pi.log_prob(z_r, u_r)               # (B,)
        loss_pi  = -(w_pi * log_prob).mean()

        self.opt_pi.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt_pi.step()

        # ── Soft update Q target ──────────────────────────────────────────
        ema = self.cfg.target_ema
        for p, pt in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            pt.data.mul_(1 - ema).add_(p.data, alpha=ema)
        for p, pt in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            pt.data.mul_(1 - ema).add_(p.data, alpha=ema)

        return {
            'loss_q':        loss_q.item(),
            'loss_v':        loss_v.item(),
            'loss_pi':       loss_pi.item(),
            'q_mean':        q_min.mean().item(),
            'v_mean':        v.mean().item(),
            'adv_mean':      adv.mean().item(),
            'r_target_mean': y_t.mean().item(),
        }

    def save(self, path: str, step: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step':       step,
            'Q1':         self.Q1.state_dict(),
            'Q2':         self.Q2.state_dict(),
            'Q1_target':  self.Q1_target.state_dict(),
            'Q2_target':  self.Q2_target.state_dict(),
            'V':          self.V.state_dict(),
            'pi':         self.pi.state_dict(),
            'r_norm_mean': self.r_norm.mean,
            'r_norm_var':  self.r_norm.var,
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.Q1.load_state_dict(ckpt['Q1'])
        self.Q2.load_state_dict(ckpt['Q2'])
        self.Q1_target.load_state_dict(ckpt['Q1_target'])
        self.Q2_target.load_state_dict(ckpt['Q2_target'])
        self.V.load_state_dict(ckpt['V'])
        self.pi.load_state_dict(ckpt['pi'])
        self.r_norm.mean = ckpt.get('r_norm_mean', 0.0)
        self.r_norm.var  = ckpt.get('r_norm_var', 1.0)
        print(f"Loaded IQL checkpoint: {path}  step={ckpt.get('step', 0)}")
        return ckpt.get('step', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_training(log: Dict[str, List], out_path: str):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = ['loss_q', 'loss_v', 'loss_pi', 'q_mean', 'v_mean',
            'adv_mean', 'r_target_mean']
    titles = ['Q Loss', 'V Loss', 'π Loss', 'Q mean',
              'V mean', 'Advantage mean', 'TD Target mean']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    PAL = ['#E53935', '#1E88E5', '#43A047', '#FB8C00',
           '#8E24AA', '#00ACC1', '#FFB300']

    for i, (key, title) in enumerate(zip(keys, titles)):
        if not log[key]: continue
        vals = np.array(log[key])
        # smoothing
        w = min(50, len(vals) // 10 + 1)
        smooth = np.convolve(vals, np.ones(w)/w, mode='valid')
        ax = axes[i]
        ax.plot(vals,   color=PAL[i], alpha=0.25, lw=0.8)
        ax.plot(smooth, color=PAL[i], lw=1.8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('IQL + H-step TD Training Curves', fontsize=12, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_policy(
    trainer:    IQLTrainer,
    model:      KoopmanCVAE,
    planner:    KODAQLQRPlanner,
    episodes:   List[Dict],
    x_seq_full: np.ndarray,
    device:     str,
    n_ep:       int = 5,
    cond_len:   int = 16,
) -> Dict[str, float]:
    """
    학습된 policy π(u|z)로 롤아웃 → Koopman 공간에서 성능 평가.
    (실제 시뮬레이터 없이 world model 기반 평가)
    """
    dev = torch.device(device)
    model.eval()
    trainer.pi.eval()

    all_rewards = []
    all_rmse_dq = []

    for ep in episodes[:n_ep]:
        L       = ep['length']
        obs_ep  = ep['obs']
        acts_ep = ep['actions']
        rew_ep  = ep['rewards']
        s_t     = ep['start_t']
        x_ep    = x_seq_full[s_t:s_t + L]

        # context encoding
        x_cond = torch.FloatTensor(x_ep[:cond_len]).unsqueeze(0).to(dev)
        a_cond = torch.FloatTensor(acts_ep[:cond_len]).unsqueeze(0).to(dev)
        enc    = model.encode_sequence(x_cond, a_cond)
        z_cur  = enc['o_seq'][0, -1:]   # (1, m)
        h_cur  = enc['h_seq'][0, -1:]   # (1, d_h)

        rollout_rewards = []
        rollout_dq      = []
        horizon         = min(64, L - cond_len)

        for t in range(horizon):
            # Policy action
            u_t   = trainer.pi.sample(z_cur)               # (1, d_u)
            # Koopman step
            w_t   = model.skill_prior.soft_weights(h_cur)
            from lqr_planner import blend_koopman
            log_lam = model.koopman.get_log_lambdas()
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, model.koopman.theta_k, model.koopman.G_k,
                model.koopman.U, w_t,
            )
            A_bar = A_bar[0]; B_bar = B_bar[0]
            z_next = (A_bar @ z_cur.T).T + (B_bar @ u_t.T).T  # (1, m)

            # Reward prediction
            recon = model.decoder(z_next)
            if 'reward' in recon:
                r_hat = torch.sigmoid(recon['reward']).item()
            else:
                r_hat = 0.0

            rollout_rewards.append(r_hat)

            # Decoded x_t for RMSE
            x_hat = torch.cat([
                symexp(recon['delta_e']),
                symexp(recon['delta_p']),
                symexp(recon['q']),
                symexp(recon['qdot']),
            ], dim=-1).cpu().numpy()  # (1, 2108)

            true_idx = cond_len + t
            if true_idx < L:
                dq_err = ((x_hat[0, X_DQ_START:X_DQ_END] -
                           x_ep[true_idx, X_DQ_START:X_DQ_END])**2).mean()
                rollout_dq.append(float(dq_err**0.5))

            # GRU update
            a_decoded = planner._decode_action(u_t)
            h_cur = model.recurrent(h_cur, z_cur, a_decoded)
            z_cur = z_next

        all_rewards.append(sum(rollout_rewards))
        if rollout_dq:
            all_rmse_dq.append(np.mean(rollout_dq))

    trainer.pi.train()
    return {
        'mean_reward':  np.mean(all_rewards),
        'std_reward':   np.std(all_rewards),
        'mean_rmse_dq': np.mean(all_rmse_dq) if all_rmse_dq else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--lqr_cache',  default=None,
                   help='Path to precomputed lqr_cache.npz (skip generation if exists)')
    p.add_argument('--iql_ckpt',   default=None,
                   help='Resume from existing IQL checkpoint')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v4/iql')
    p.add_argument('--quality',    default='mixed')
    p.add_argument('--n_ep_lqr',   type=int,   default=500)
    p.add_argument('--H',          type=int,   default=8)
    p.add_argument('--tau',        type=float, default=0.8)
    p.add_argument('--beta',       type=float, default=3.0)
    p.add_argument('--gamma',      type=float, default=0.99)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--n_steps',    type=int,   default=500_000)
    p.add_argument('--real_ratio', type=float, default=0.5)
    p.add_argument('--Q_scale',    type=float, default=1.0)
    p.add_argument('--R_scale',    type=float, default=10.0)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"Device: {device}")

    # ── World model 로드 ────────────────────────────────────────────────────
    print(f"\nLoading world model: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    m_cfg = model.cfg
    z_dim = m_cfg.koopman_dim      # Koopman latent dim (m)
    u_dim = m_cfg.action_latent    # action latent dim (d_u)
    print(f"  K={m_cfg.num_skills}  m={z_dim}  d_u={u_dim}  "
          f"reward_head={m_cfg.use_reward_head}")

    lqr_cfg = LQRConfig(Q_scale=args.Q_scale, R_scale=args.R_scale)
    planner = KODAQLQRPlanner(model, lqr_cfg)

    # ── 데이터 로드 ─────────────────────────────────────────────────────────
    print(f"\nLoading x_sequences: {args.x_cache}")
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    print(f"  x_seq: {x_seq_full.shape}")

    episodes, _ = load_kitchen_episodes(quality=args.quality, min_len=32)
    episodes_for_lqr = [ep for ep in episodes if ep['tasks']]
    episodes_for_lqr = episodes_for_lqr[:args.n_ep_lqr]
    print(f"  Episodes with tasks: {len(episodes_for_lqr)}")

    # ── LQR 캐시 로드 또는 생성 ─────────────────────────────────────────────
    cache_path = args.lqr_cache or f"{args.out_dir}/lqr_cache.npz"

    if args.lqr_cache and Path(args.lqr_cache).exists():
        print(f"\nLoading LQR cache: {args.lqr_cache}")
        cache = dict(np.load(args.lqr_cache))
        print(f"  LQR transitions: {len(cache['z0'])}")
        print(f"  Real transitions: {len(cache['z_real'])}")
    else:
        print(f"\nBuilding LQR cache (H={args.H}) ...")
        cache = build_lqr_cache(
            model=model, planner=planner,
            episodes=episodes_for_lqr,
            x_seq_full=x_seq_full,
            H=args.H, device=device,
            save_path=cache_path,
        )

    # ── Replay Buffer 구성 ──────────────────────────────────────────────────
    buf = ReplayBuffer(device)

    buf.add_real(
        z_t    = cache['z_real'],
        u_t    = cache['u_real'],
        z_next = cache['z_next_real'],
        r_t    = cache['r_real'],
    )
    buf.add_lqr(
        z_t       = cache['z0'],
        z_hat_seq = cache['z_hat_seq'],
        r_hat_seq = cache['r_hat_seq'],
    )
    print(f"\nBuffer — real: {buf.real_size}  lqr: {buf.lqr_size}")

    # Reward normalizer 초기화 (전체 H-step target 분포로)
    γ_powers  = np.array([args.gamma**k for k in range(args.H)])
    r_hat_all = cache['r_hat_seq']                   # (N, H)
    r_sum_all = (r_hat_all * γ_powers).sum(axis=1)   # (N,)
    iql_cfg   = IQLConfig(
        tau=args.tau, beta=args.beta, gamma=args.gamma,
        H=args.H, lr_q=args.lr, lr_v=args.lr, lr_pi=args.lr,
        batch_size=args.batch_size, n_steps=args.n_steps,
        real_ratio=args.real_ratio,
    )
    trainer = IQLTrainer(iql_cfg, z_dim, a_dim, device)
    trainer.r_norm.update(r_sum_all)

    # CategoricalRewardHead (3-way reward)
    cat_ckpt = str(Path(args.out_dir).parent / 'cat_reward' / 'final.pt')
    if Path(cat_ckpt).exists():
        from train_reward_head import load_cat_reward_model
        _, trainer.cat_head = load_cat_reward_model(cat_ckpt, device)
        print(f'  CategoricalRewardHead loaded: 3-way reward enabled')
    else:
        print(f'  No cat_head ({cat_ckpt}), using BCE only')
    print(f"Reward normalizer: mean={trainer.r_norm.mean:.4f}  "
          f"std={math.sqrt(trainer.r_norm.var):.4f}")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if args.iql_ckpt and Path(args.iql_ckpt).exists():
        start_step = trainer.load(args.iql_ckpt)

    # ── 학습 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"IQL + H-step TD  |  steps={args.n_steps}  H={args.H}")
    print(f"  τ={args.tau}  β={args.beta}  γ={args.gamma}")
    print(f"  batch={args.batch_size}  real_ratio={args.real_ratio}")
    print(f"{'='*60}\n")

    log: Dict[str, List] = {k: [] for k in [
        'loss_q', 'loss_v', 'loss_pi', 'q_mean', 'v_mean',
        'adv_mean', 'r_target_mean',
    ]}
    recent = {k: deque(maxlen=iql_cfg.log_every) for k in log}
    t0 = time.time()

    B_real = max(1, int(args.batch_size * args.real_ratio))
    B_lqr  = args.batch_size - B_real

    for step in range(start_step, args.n_steps):

        # Batch sampling
        real_batch = buf.sample_real(B_real)
        lqr_batch  = buf.sample_lqr(B_lqr)

        # Reward normalizer 업데이트 (r_hat 분포 추적)
        r_hat_batch = lqr_batch['r_hat_seq'].cpu().numpy()
        γ_pow       = np.array([args.gamma**k for k in range(args.H)])
        r_sum_batch = (r_hat_batch * γ_pow).sum(axis=1)
        trainer.r_norm.update(r_sum_batch)

        # Update
        info = trainer.update(real_batch, lqr_batch)

        for k, v in info.items():
            recent[k].append(v)

        # Logging
        if (step + 1) % iql_cfg.log_every == 0:
            means = {k: np.mean(list(recent[k])) for k in log}
            for k in log:
                log[k].append(means[k])

            elapsed = time.time() - t0
            steps_per_sec = iql_cfg.log_every / (elapsed + 1e-6)
            t0 = time.time()

            print(
                f"Step {step+1:7d} | "
                f"Q={means['loss_q']:.4f}  "
                f"V={means['loss_v']:.4f}  "
                f"π={means['loss_pi']:.4f}  |  "
                f"q_μ={means['q_mean']:.3f}  "
                f"v_μ={means['v_mean']:.3f}  "
                f"adv_μ={means['adv_mean']:.3f}  |  "
                f"r_target={means['r_target_mean']:.3f}  |  "
                f"{steps_per_sec:.0f} steps/s"
            )

        # Checkpoint
        if (step + 1) % iql_cfg.save_every == 0:
            trainer.save(f"{args.out_dir}/iql_step{step+1}.pt", step + 1)
            visualize_training(log, f"{args.out_dir}/training_curves.png")

        # Policy evaluation
        if (step + 1) % iql_cfg.eval_every == 0:
            eval_eps = [ep for ep in episodes[:20] if ep['tasks']]
            eval_res = evaluate_policy(
                trainer, model, planner,
                eval_eps, x_seq_full,
                device=device, n_ep=iql_cfg.n_eval_ep,
            )
            print(
                f"\n  [EVAL step {step+1}]  "
                f"mean_reward={eval_res['mean_reward']:.4f}  "
                f"±{eval_res['std_reward']:.4f}  "
                f"RMSE_Δq={eval_res['mean_rmse_dq']:.4f}\n"
            )

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    trainer.save(f"{args.out_dir}/iql_final.pt", args.n_steps)
    visualize_training(log, f"{args.out_dir}/training_curves_final.png")

    # Summary
    if log['loss_q']:
        print(f"\n{'='*60}")
        print(f"Training complete.  {args.n_steps} steps")
        print(f"  Final Q loss:  {log['loss_q'][-1]:.4f}")
        print(f"  Final V loss:  {log['loss_v'][-1]:.4f}")
        print(f"  Final π loss:  {log['loss_pi'][-1]:.4f}")
        print(f"  Outputs → {args.out_dir}/")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()