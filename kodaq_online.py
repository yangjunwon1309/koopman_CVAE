"""
kodaq_online.py — KODAQ-Online: Hierarchical Online RL with Koopman World Model
================================================================================

구조:
  High-level policy:  π_hi(skill_id | z_t, z_goal)
    - K=7 discrete skills, H_hi steps마다 재선택
    - Prior: KL(π_hi || p_skill(c|h_t))  ← frozen KODAQ skill_prior
    - SAC update

  Low-level policy:   π_lo(a_{t:t+H_lo} | z_t, skill_embed, z_goal)
    - a ∈ R^9 직접 출력 (action sequence, H_lo=4~8)
    - Gaussian(μ_a, σ_a), tanh squash
    - SAC update

  World model (부분 active):
    Frozen:  Koopman operator (A_k, B_k), decoder
    Active:  posterior(encoder μ_φ), recurrent(GRU), reward_head, skill_prior

  Reward blend:
    r_blend = α * r_env + (1-α) * r̂_world
    α schedule: 0 → 1 over warmup_steps

  Goal conditioning:
    z_goal = encode_goal(x_goal, h_t)
    x_goal = obs_to_x_goal(target_obs, ref_obs)

EXTRACT와의 비교:
  EXTRACT:       VLM prior + online SAC + real env only
  KODAQ-Online:  Koopman prior + online SAC + world model reward blend

Usage:
    MUJOCO_GL=egl python kodaq_online.py \
        --world_ckpt checkpoints/kodaq_v4/final.pt \
        --x_cache    checkpoints/skill_pretrain/x_sequences.npz \
        --out_dir    checkpoints/kodaq_v4/online \
        --env        kitchen-mixed-v0 \
        --device     cuda:1
"""

import os, sys, time, math, copy
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
from dataclasses import dataclass, field
from collections import deque

from models.koopman_cvae import KoopmanCVAE
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_planner import (
    KODAQLQRPlanner, LQRConfig,
    load_kitchen_episodes, obs_to_x_goal,
    blend_koopman,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
    OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OnlineConfig:
    # Hierarchy
    H_hi:        int   = 8      # high-level skill 유지 스텝 수
    H_lo:        int   = 4      # low-level action sequence 길이 (4~8)
    # ↑ H_lo만큼 action을 출력하고 env에서 연속 실행
    # LQR prior 누적 오차가 커지지 않도록 4~8로 제한

    # SAC
    gamma:       float = 0.99
    tau_ema:     float = 0.005  # soft update
    alpha_hi:    float = 0.1    # high-level entropy coeff
    alpha_lo:    float = 0.1    # low-level entropy coeff
    kl_weight:   float = 1.0    # skill prior KL weight

    # Networks
    hidden_dim:  int   = 256
    n_layers:    int   = 2
    skill_embed_dim: int = 32   # skill_id embedding dim

    # Optimization
    lr:          float = 3e-4
    batch_size:  int   = 256
    grad_clip:   float = 1.0
    n_updates_per_step: int = 1  # env step 당 update 횟수

    # World model fine-tune
    wm_lr:       float = 1e-4
    wm_update_freq: int = 10     # N env steps마다 world model 업데이트

    # Reward blend
    r_alpha_warmup: int = 10_000  # r_blend = α*r_env + (1-α)*r_hat
    # warmup 동안 α: 0→1 (초반엔 world model reward 비중 높음)

    # Training
    n_env_steps:     int = 300_000
    warmup_random:   int = 1_000   # 초반 random action
    buffer_size:     int = 200_000
    log_every:       int = 1_000
    save_every:      int = 50_000
    eval_every:      int = 25_000
    n_eval_ep:       int = 5

    # Goal
    cond_len:    int = 16       # goal 인코딩용 context 길이


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim, out_dim, hidden, n_layers, output_act=None):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if output_act: layers.append(output_act)
    return nn.Sequential(*layers)


class SkillEmbedding(nn.Module):
    """discrete skill_id → continuous embedding"""
    def __init__(self, n_skills: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_skills, embed_dim)

    def forward(self, skill_id: torch.Tensor) -> torch.Tensor:
        return self.emb(skill_id)   # (B, embed_dim)


class HighLevelPolicy(nn.Module):
    """
    π_hi(skill_id | z_t, z_goal) → Categorical over K skills

    입력: z_t (m), z_goal (m) concat → (2m,)
    출력: logits over K skills
    """
    def __init__(self, z_dim: int, n_skills: int, hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim * 2, n_skills, hidden, n_layers)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, K)"""
        return self.net(torch.cat([z, z_goal], dim=-1))

    def sample(self, z: torch.Tensor, z_goal: torch.Tensor,
               greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (skill_id (B,), log_prob (B,))"""
        logits = self(z, z_goal)
        dist   = torch.distributions.Categorical(logits=logits)
        if greedy:
            skill_id = logits.argmax(dim=-1)
        else:
            skill_id = dist.sample()
        return skill_id, dist.log_prob(skill_id)

    def log_prob(self, z: torch.Tensor, z_goal: torch.Tensor,
                 skill_id: torch.Tensor) -> torch.Tensor:
        logits = self(z, z_goal)
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(skill_id)

    def entropy(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        logits = self(z, z_goal)
        return torch.distributions.Categorical(logits=logits).entropy()


class LowLevelPolicy(nn.Module):
    """
    π_lo(a | z_t, skill_embed, z_goal) → a ∈ R^{action_dim}

    - action을 직접 출력 (latent u 거치지 않음)
    - H_lo step 길이의 action sequence μ, log_σ를 한 번에 출력
    - tanh squash로 [-1, 1]에 클리핑

    입력: z_t(m) + skill_embed(e) + z_goal(m) → (2m+e,)
    출력: μ_a (H_lo * da), log_σ_a (H_lo * da)
    """
    def __init__(self, z_dim: int, skill_embed_dim: int, action_dim: int,
                 H_lo: int, hidden: int, n_layers: int,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.action_dim  = action_dim
        self.H_lo        = H_lo
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        in_dim = z_dim * 2 + skill_embed_dim
        self.net    = make_mlp(in_dim, hidden, hidden, n_layers - 1)
        self.mu     = nn.Linear(hidden, H_lo * action_dim)
        self.log_s  = nn.Linear(hidden, H_lo * action_dim)

    def _feat(self, z: torch.Tensor, skill_embed: torch.Tensor,
              z_goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, skill_embed, z_goal], dim=-1))

    def forward(self, z: torch.Tensor, skill_embed: torch.Tensor,
                z_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std) each (B, H_lo, da)"""
        feat    = self._feat(z, skill_embed, z_goal)
        mu      = self.mu(feat).view(-1, self.H_lo, self.action_dim)
        log_std = self.log_s(feat).view(-1, self.H_lo, self.action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, z: torch.Tensor, skill_embed: torch.Tensor,
               z_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          a_seq:    (B, H_lo, da)  tanh-squashed actions ∈ [-1,1]
          log_prob: (B,)  sum log_prob over H_lo steps and da dims
        """
        mu, log_std = self(z, skill_embed, z_goal)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u    = dist.rsample()                      # (B, H_lo, da)
        a    = torch.tanh(u)

        # log_prob with tanh correction
        lp   = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        lp   = lp.sum(dim=(-2, -1))               # (B,) sum over H_lo, da
        return a, lp

    def log_prob(self, z: torch.Tensor, skill_embed: torch.Tensor,
                 z_goal: torch.Tensor,
                 a_seq: torch.Tensor) -> torch.Tensor:
        """a_seq: (B, H_lo, da) tanh-squashed"""
        mu, log_std = self(z, skill_embed, z_goal)
        std  = log_std.exp()
        u    = torch.atanh(a_seq.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mu, std)
        lp   = dist.log_prob(u) - torch.log(1 - a_seq.pow(2) + 1e-6)
        return lp.sum(dim=(-2, -1))               # (B,)

    def entropy(self, z: torch.Tensor, skill_embed: torch.Tensor,
                z_goal: torch.Tensor) -> torch.Tensor:
        _, log_std = self(z, skill_embed, z_goal)
        # Gaussian entropy: 0.5 * log(2πe * σ²) per dim
        return (0.5 * (1 + 2 * log_std + math.log(2 * math.pi))
                ).sum(dim=(-2, -1))               # (B,)


class QNetworkHi(nn.Module):
    """Q_hi(z_t, z_goal, skill_id) → scalar"""
    def __init__(self, z_dim: int, skill_embed_dim: int,
                 hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim * 2 + skill_embed_dim, 1, hidden, n_layers)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor,
                skill_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, z_goal, skill_embed], dim=-1)).squeeze(-1)


class QNetworkLo(nn.Module):
    """Q_lo(z_t, skill_embed, z_goal, a_seq_flat) → scalar"""
    def __init__(self, z_dim: int, skill_embed_dim: int,
                 action_dim: int, H_lo: int,
                 hidden: int, n_layers: int):
        super().__init__()
        in_dim = z_dim * 2 + skill_embed_dim + action_dim * H_lo
        self.net = make_mlp(in_dim, 1, hidden, n_layers)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor,
                skill_embed: torch.Tensor,
                a_seq: torch.Tensor) -> torch.Tensor:
        """a_seq: (B, H_lo, da) or (B, H_lo*da)"""
        a_flat = a_seq.reshape(a_seq.shape[0], -1)
        return self.net(torch.cat([z, z_goal, skill_embed, a_flat],
                                  dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalReplayBuffer:
    """
    Hi-level:  (z_t, z_goal, skill_id, R_hi, z_t_next, done_hi)
               R_hi = Σ γ^k r_{t+k}  (H_hi step return)
    Lo-level:  (z_t, z_goal, skill_embed, a_seq, r_lo, z_next, done_lo)
               r_lo = world model reward (H_lo step)
    """
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device   = device
        self.hi: Dict[str, np.ndarray] = {}
        self.lo: Dict[str, np.ndarray] = {}
        self._hi_ptr = 0; self._hi_n = 0
        self._lo_ptr = 0; self._lo_n = 0

    def _init_hi(self, z_dim, n_skills, h_dim):
        C = self.capacity
        self.hi = {
            'z':        np.zeros((C, z_dim),  dtype=np.float32),
            'z_goal':   np.zeros((C, z_dim),  dtype=np.float32),
            'z_next':   np.zeros((C, z_dim),  dtype=np.float32),
            'h_t':      np.zeros((C, h_dim),  dtype=np.float32),  # skill prior용
            'skill_id': np.zeros(C,            dtype=np.int64),
            'R_hi':     np.zeros(C,            dtype=np.float32),
            'done':     np.zeros(C,            dtype=np.float32),
        }

    def _init_lo(self, z_dim, skill_embed_dim, action_dim, H_lo):
        C = self.capacity
        self.lo = {
            'z':          np.zeros((C, z_dim),                dtype=np.float32),
            'z_goal':     np.zeros((C, z_dim),                dtype=np.float32),
            'z_next':     np.zeros((C, z_dim),                dtype=np.float32),
            'skill_embed':np.zeros((C, skill_embed_dim),      dtype=np.float32),
            'a_seq':      np.zeros((C, H_lo, action_dim),     dtype=np.float32),
            'r_lo':       np.zeros(C,                         dtype=np.float32),
            'done':       np.zeros(C,                         dtype=np.float32),
        }

    def add_hi(self, z, z_goal, h_t, skill_id, R_hi, z_next, done):
        if not self.hi:
            self._init_hi(z.shape[-1], int(skill_id) + 1, h_t.shape[-1])
        p = self._hi_ptr
        self.hi['z'][p]        = z
        self.hi['z_goal'][p]   = z_goal
        self.hi['z_next'][p]   = z_next
        self.hi['h_t'][p]      = h_t
        self.hi['skill_id'][p] = skill_id
        self.hi['R_hi'][p]     = R_hi
        self.hi['done'][p]     = float(done)
        self._hi_ptr = (p + 1) % self.capacity
        self._hi_n   = min(self._hi_n + 1, self.capacity)

    def add_lo(self, z, z_goal, skill_embed, a_seq, r_lo, z_next, done):
        if not self.lo:
            self._init_lo(z.shape[-1], skill_embed.shape[-1],
                          a_seq.shape[-1], a_seq.shape[-2])
        p = self._lo_ptr
        self.lo['z'][p]           = z
        self.lo['z_goal'][p]      = z_goal
        self.lo['z_next'][p]      = z_next
        self.lo['skill_embed'][p] = skill_embed
        self.lo['a_seq'][p]       = a_seq
        self.lo['r_lo'][p]        = r_lo
        self.lo['done'][p]        = float(done)
        self._lo_ptr = (p + 1) % self.capacity
        self._lo_n   = min(self._lo_n + 1, self.capacity)

    def sample_hi(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._hi_n, batch_size)
        return {k: torch.tensor(v[idx]).to(self.device)
                for k, v in self.hi.items()}

    def sample_lo(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._lo_n, batch_size)
        return {k: torch.tensor(v[idx]).to(self.device)
                for k, v in self.lo.items()}

    @property
    def hi_size(self): return self._hi_n

    @property
    def lo_size(self): return self._lo_n


# ─────────────────────────────────────────────────────────────────────────────
# World Model Wrapper (active/frozen 분리)
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanWorldModelWrapper:
    """
    KoopmanCVAE를 active/frozen 두 부분으로 분리하여 관리.

    Frozen: koopman operator (A_k, B_k, eigenvalues), decoder
    Active: posterior(encoder μ_φ), recurrent(GRU), reward_head, skill_prior
    """
    def __init__(self, model: KoopmanCVAE, wm_lr: float, device: str):
        self.model  = model
        self.device = device

        # Freeze Koopman operator + decoder reconstruction heads
        # (reward_head는 active로 유지 → online env reward로 fine-tune)
        for p in model.koopman.parameters():
            p.requires_grad_(False)

        # decoder 전체를 먼저 freeze
        for p in model.decoder.parameters():
            p.requires_grad_(False)

        # reward_head 브랜치만 다시 active로 전환
        # KoopmanCVAE에서 reward head는 두 가지 위치 중 하나:
        #   A. model.decoder.head_reward  (MultiHeadDecoder 내부)
        #   B. model.reward_head          (별도 모듈)
        reward_head_params = []
        if model.cfg.use_reward_head:
            if hasattr(model.decoder, 'head_reward'):
                # Case A: decoder 내부에 있음 → 해당 파라미터만 unfreeze
                for p in model.decoder.head_reward.parameters():
                    p.requires_grad_(True)
                reward_head_params = list(model.decoder.head_reward.parameters())
            elif hasattr(model, 'reward_head'):
                # Case B: 별도 모듈
                for p in model.reward_head.parameters():
                    p.requires_grad_(True)
                reward_head_params = list(model.reward_head.parameters())

        # Active params
        active_params = (
            list(model.posterior.parameters()) +
            list(model.recurrent.parameters()) +
            list(model.skill_prior.parameters()) +
            reward_head_params
        )

        self.opt = torch.optim.Adam(active_params, lr=wm_lr)

    @torch.no_grad()
    def encode(self, x_seq: torch.Tensor,
               a_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x_seq: (1, T, 2108), a_seq: (1, T, 9)"""
        return self.model.encode_sequence(x_seq, a_seq)

    @torch.no_grad()
    def encode_goal(self, x_goal: torch.Tensor,
                    h_ref: torch.Tensor) -> torch.Tensor:
        """x_goal: (1, 2108) → z_goal: (1, m)"""
        if x_goal.dim() == 1:
            x_goal = x_goal.unsqueeze(0)
        mu, _ = self.model.posterior(x_goal, h_ref)
        return mu

    @torch.no_grad()
    def koopman_step(self, z: torch.Tensor, a: torch.Tensor,
                     h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        단일 스텝 Koopman 전파.

        Args:
          z: (1, m)  현재 Koopman state
          a: (1, 9)  robot action
          h: (1, d_h) GRU hidden

        Returns:
          z_next: (1, m)
          h_next: (1, d_h)
          r_hat:  float  reward prediction
        """
        model = self.model
        u = model.action_encoder(a)                       # (1, d_u)
        w = model.skill_prior.soft_weights(h)             # (1, K)

        log_lam = model.koopman.get_log_lambdas()
        A_bar, B_bar, _, _ = blend_koopman(
            log_lam, model.koopman.theta_k,
            model.koopman.G_k, model.koopman.U, w
        )
        A_bar = A_bar[0]; B_bar = B_bar[0]               # (m,m), (m,d_u)

        z_next = (A_bar @ z.T).T + (B_bar @ u.T).T       # (1, m)
        h_next = model.recurrent(h, z, a)                 # (1, d_h)

        # reward head 접근 (decoder 내부/외부 모두 처리)
        r_hat = 0.0
        if model.cfg.use_reward_head:
            if hasattr(model.decoder, 'head_reward'):
                r_logit = model.decoder.head_reward(z_next)
            elif hasattr(model, 'reward_head'):
                r_logit = model.reward_head(z_next)
            else:
                r_logit = None
            if r_logit is not None:
                r_hat = torch.sigmoid(r_logit).mean().item()

        return z_next, h_next, r_hat

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, h0: torch.Tensor,
                a_seq: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        H_lo step rollout with given action sequence.

        a_seq: (H_lo, 9)
        Returns: z_final (1, m), r_sum (float, discounted)
        """
        z, h = z0, h0
        r_sum, gamma = 0.0, 0.99
        for k, a in enumerate(a_seq):
            z, h, r_hat = self.koopman_step(z, a.unsqueeze(0), h)
            r_sum += (gamma ** k) * r_hat
        return z, r_sum

    def update(self, obs_batch: torch.Tensor, act_batch: torch.Tensor,
               obs_next_batch: torch.Tensor,
               r_env_batch: torch.Tensor) -> float:
        """
        Active 파라미터 (posterior, reward_head) fine-tune.

        gradient 경로:
          obs_batch → posterior(x_t, h_dummy) → z_t → reward_head → BCE loss
          encode_sequence()는 내부적으로 @no_grad일 수 있으므로
          posterior를 직접 호출해서 gradient를 살림.

        reward_head BCE:  reward_head(z_t) ≈ r_env (실제 env reward)
        """
        if not self.model.cfg.use_reward_head:
            return 0.0

        model = self.model
        model.train()

        B   = obs_batch.shape[0]
        dev = self.device

        # posterior 직접 호출 (gradient 필요)
        # h_dummy: context 없는 단일 스텝이므로 zero hidden
        h_dummy = torch.zeros(B, model.cfg.gru_hidden, device=dev)
        z_t, _  = model.posterior(obs_batch, h_dummy)   # (B, m)

        # reward head 직접 호출 (decoder 내부/외부 모두 처리)
        # decoder는 frozen이지만 head_reward만 active
        if hasattr(model.decoder, 'head_reward'):
            # Case A: MultiHeadDecoder.head_reward (active)
            r_logit = model.decoder.head_reward(z_t).squeeze(-1)   # (B,)
        elif hasattr(model, 'reward_head'):
            # Case B: 별도 모듈 (active)
            r_logit = model.reward_head(z_t).squeeze(-1)            # (B,)
        else:
            model.eval()
            return 0.0

        r_target = r_env_batch.clamp(0.0, 1.0).float()   # (B,)
        loss = F.binary_cross_entropy_with_logits(r_logit, r_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        self.opt.step()

        model.eval()
        return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# SAC Updater (Hi / Lo 공용)
# ─────────────────────────────────────────────────────────────────────────────

class SACUpdater:
    """범용 SAC update 유틸. Hi-level과 Lo-level 모두 사용."""

    @staticmethod
    def soft_update(net: nn.Module, net_target: nn.Module, tau: float):
        for p, pt in zip(net.parameters(), net_target.parameters()):
            pt.data.mul_(1 - tau).add_(p.data, alpha=tau)

    @staticmethod
    def update_critic(Q1, Q2, Q1_t, Q2_t, opt_q,
                      z, z_goal, key, R,
                      V_next, gamma, done, grad_clip) -> float:
        """
        key: skill_embed + a_flat for Lo, skill_embed for Hi
        V_next: target value at next state
        """
        y = R + gamma * (1 - done) * V_next.detach()
        q1 = Q1(z, z_goal, key) if hasattr(Q1, 'forward') else Q1(z, z_goal, key)
        q2 = Q2(z, z_goal, key)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_q.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            list(Q1.parameters()) + list(Q2.parameters()), grad_clip)
        opt_q.step()
        return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class KODAQOnlineTrainer:
    def __init__(self, cfg: OnlineConfig, world_model: KoopmanWorldModelWrapper,
                 z_dim: int, n_skills: int, action_dim: int, device: str):
        self.cfg    = cfg
        self.wm     = world_model
        self.device = device
        self.z_dim  = z_dim
        self.n_skills = n_skills
        self.action_dim = action_dim

        H_lo  = cfg.H_lo
        h_dim = cfg.hidden_dim
        n_lay = cfg.n_layers
        e_dim = cfg.skill_embed_dim

        # Skill embedding (shared between hi/lo)
        self.skill_emb    = SkillEmbedding(n_skills, e_dim).to(device)
        self.skill_emb_t  = copy.deepcopy(self.skill_emb)

        # High-level
        self.pi_hi    = HighLevelPolicy(z_dim, n_skills, h_dim, n_lay).to(device)
        self.Q_hi_1   = QNetworkHi(z_dim, e_dim, h_dim, n_lay).to(device)
        self.Q_hi_2   = QNetworkHi(z_dim, e_dim, h_dim, n_lay).to(device)
        self.Q_hi_1_t = copy.deepcopy(self.Q_hi_1)
        self.Q_hi_2_t = copy.deepcopy(self.Q_hi_2)

        # Low-level
        self.pi_lo    = LowLevelPolicy(z_dim, e_dim, action_dim, H_lo,
                                       h_dim, n_lay).to(device)
        self.Q_lo_1   = QNetworkLo(z_dim, e_dim, action_dim, H_lo,
                                   h_dim, n_lay).to(device)
        self.Q_lo_2   = QNetworkLo(z_dim, e_dim, action_dim, H_lo,
                                   h_dim, n_lay).to(device)
        self.Q_lo_1_t = copy.deepcopy(self.Q_lo_1)
        self.Q_lo_2_t = copy.deepcopy(self.Q_lo_2)

        lr = cfg.lr
        self.opt_hi = torch.optim.Adam(
            list(self.pi_hi.parameters()) + list(self.skill_emb.parameters()), lr=lr)
        self.opt_q_hi = torch.optim.Adam(
            list(self.Q_hi_1.parameters()) + list(self.Q_hi_2.parameters()), lr=lr)
        self.opt_lo = torch.optim.Adam(self.pi_lo.parameters(), lr=lr)
        self.opt_q_lo = torch.optim.Adam(
            list(self.Q_lo_1.parameters()) + list(self.Q_lo_2.parameters()), lr=lr)

        # Automatic entropy tuning
        self.log_alpha_hi = torch.zeros(1, requires_grad=True, device=device)
        self.log_alpha_lo = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha_hi = torch.optim.Adam([self.log_alpha_hi], lr=lr)
        self.opt_alpha_lo = torch.optim.Adam([self.log_alpha_lo], lr=lr)
        # Target entropy: -log(1/K) for hi, -H_lo*da for lo
        self.target_ent_hi = -math.log(1.0 / n_skills) * cfg.alpha_hi
        self.target_ent_lo = -H_lo * action_dim * cfg.alpha_lo

        self.buf = HierarchicalReplayBuffer(cfg.buffer_size, device)
        self.step = 0

    @property
    def alpha_hi(self): return self.log_alpha_hi.exp().item()

    @property
    def alpha_lo(self): return self.log_alpha_lo.exp().item()

    # ── Hi-level update ──────────────────────────────────────────────────────

    def update_hi(self) -> Dict[str, float]:
        if self.buf.hi_size < self.cfg.batch_size:
            return {}
        b    = self.buf.sample_hi(self.cfg.batch_size)
        z    = b['z'];      z_g  = b['z_goal']
        z_nx = b['z_next']; done = b['done']
        sid  = b['skill_id']
        R_hi = b['R_hi']

        se   = self.skill_emb(sid)          # (B, e)
        se_t = self.skill_emb_t(sid)

        # ── Critic ────────────────────────────────────────────────────────
        with torch.no_grad():
            # V_next = E_{d'~π_hi}[Q_target - α*H]
            logits_nx  = self.pi_hi(z_nx, z_g)
            dist_nx    = torch.distributions.Categorical(logits=logits_nx)
            sid_nx     = dist_nx.sample()
            lp_nx      = dist_nx.log_prob(sid_nx)
            se_nx      = self.skill_emb_t(sid_nx)
            q1_nx = self.Q_hi_1_t(z_nx, z_g, se_nx)
            q2_nx = self.Q_hi_2_t(z_nx, z_g, se_nx)
            V_next = torch.min(q1_nx, q2_nx) - self.alpha_hi * lp_nx

        y = R_hi + self.cfg.gamma * (1 - done) * V_next
        q1 = self.Q_hi_1(z, z_g, se)
        q2 = self.Q_hi_2(z, z_g, se)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q_hi.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q_hi_1.parameters()) + list(self.Q_hi_2.parameters()),
            self.cfg.grad_clip)
        self.opt_q_hi.step()

        # ── Actor (prior-regularized) ──────────────────────────────────────
        logits  = self.pi_hi(z, z_g)
        dist    = torch.distributions.Categorical(logits=logits)
        sid_new = dist.sample()
        lp_new  = dist.log_prob(sid_new)

        se_new  = self.skill_emb(sid_new)
        q_new   = torch.min(self.Q_hi_1(z, z_g, se_new),
                            self.Q_hi_2(z, z_g, se_new))

        # Skill prior KL: KL(π_hi || p_skill(c|h_t))
        # h_t는 buffer에 저장된 KODAQ GRU hidden state
        # frozen KODAQ skill_prior.soft_weights(h_t) → 현재 context에서
        # 어떤 skill이 자연스러운지 Koopman world model이 알고 있음
        with torch.no_grad():
            h_stored = b['h_t']   # (B, d_h) — buffer에 저장된 h_t
            # soft_weights: (B, d_h) → (B, K) 확률 분포
            p_prior_weights = self.wm.model.skill_prior.soft_weights(
                h_stored)                          # (B, K)
            # 확률값 그대로 log 취하기 (이미 softmax 통과)
            log_p_prior = torch.log(p_prior_weights + 1e-8)   # (B, K)

        log_pi = torch.log_softmax(logits, dim=-1)             # (B, K)
        # KL(π || p_prior) = Σ_k π_k * (log π_k - log p_k)
        kl     = (torch.exp(log_pi) * (log_pi - log_p_prior)).sum(-1)  # (B,)

        loss_hi  = (self.alpha_hi * lp_new
                    + self.cfg.kl_weight * kl
                    - q_new).mean()
        self.opt_hi.zero_grad(); loss_hi.backward()
        nn.utils.clip_grad_norm_(
            list(self.pi_hi.parameters()) + list(self.skill_emb.parameters()),
            self.cfg.grad_clip)
        self.opt_hi.step()

        # ── Alpha ─────────────────────────────────────────────────────────
        ent_hi    = dist.entropy().mean()
        loss_a_hi = -(self.log_alpha_hi * (ent_hi - self.target_ent_hi).detach())
        self.opt_alpha_hi.zero_grad(); loss_a_hi.backward()
        self.opt_alpha_hi.step()

        # Soft update
        SACUpdater.soft_update(self.Q_hi_1, self.Q_hi_1_t, self.cfg.tau_ema)
        SACUpdater.soft_update(self.Q_hi_2, self.Q_hi_2_t, self.cfg.tau_ema)
        SACUpdater.soft_update(self.skill_emb, self.skill_emb_t, self.cfg.tau_ema)

        return {
            'loss_q_hi': loss_q.item(),
            'loss_pi_hi': loss_hi.item(),
            'kl_hi': kl.mean().item(),
            'alpha_hi': self.alpha_hi,
            'ent_hi': ent_hi.item(),
        }

    # ── Lo-level update ──────────────────────────────────────────────────────

    def update_lo(self) -> Dict[str, float]:
        if self.buf.lo_size < self.cfg.batch_size:
            return {}
        b    = self.buf.sample_lo(self.cfg.batch_size)
        z    = b['z'];     z_g   = b['z_goal']
        z_nx = b['z_next']; done = b['done']
        se   = b['skill_embed']
        a    = b['a_seq']          # (B, H_lo, da)
        r    = b['r_lo']

        # ── Critic ────────────────────────────────────────────────────────
        with torch.no_grad():
            a_nx, lp_nx = self.pi_lo.sample(z_nx, se, z_g)
            q1_nx = self.Q_lo_1_t(z_nx, z_g, se, a_nx)
            q2_nx = self.Q_lo_2_t(z_nx, z_g, se, a_nx)
            V_next = torch.min(q1_nx, q2_nx) - self.alpha_lo * lp_nx

        y    = r + self.cfg.gamma * (1 - done) * V_next
        q1   = self.Q_lo_1(z, z_g, se, a)
        q2   = self.Q_lo_2(z, z_g, se, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q_lo.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q_lo_1.parameters()) + list(self.Q_lo_2.parameters()),
            self.cfg.grad_clip)
        self.opt_q_lo.step()

        # ── Actor ─────────────────────────────────────────────────────────
        a_new, lp_new = self.pi_lo.sample(z, se, z_g)
        q_new = torch.min(self.Q_lo_1(z, z_g, se, a_new),
                          self.Q_lo_2(z, z_g, se, a_new))
        loss_lo = (self.alpha_lo * lp_new - q_new).mean()
        self.opt_lo.zero_grad(); loss_lo.backward()
        nn.utils.clip_grad_norm_(self.pi_lo.parameters(), self.cfg.grad_clip)
        self.opt_lo.step()

        # ── Alpha ─────────────────────────────────────────────────────────
        ent_lo    = self.pi_lo.entropy(z, se, z_g).mean()
        loss_a_lo = -(self.log_alpha_lo * (ent_lo - self.target_ent_lo).detach())
        self.opt_alpha_lo.zero_grad(); loss_a_lo.backward()
        self.opt_alpha_lo.step()

        # Soft update
        SACUpdater.soft_update(self.Q_lo_1, self.Q_lo_1_t, self.cfg.tau_ema)
        SACUpdater.soft_update(self.Q_lo_2, self.Q_lo_2_t, self.cfg.tau_ema)

        return {
            'loss_q_lo':  loss_q.item(),
            'loss_pi_lo': loss_lo.item(),
            'alpha_lo':   self.alpha_lo,
            'ent_lo':     ent_lo.item(),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step':         self.step,
            'pi_hi':        self.pi_hi.state_dict(),
            'pi_lo':        self.pi_lo.state_dict(),
            'skill_emb':    self.skill_emb.state_dict(),
            'Q_hi_1':       self.Q_hi_1.state_dict(),
            'Q_hi_2':       self.Q_hi_2.state_dict(),
            'Q_lo_1':       self.Q_lo_1.state_dict(),
            'Q_lo_2':       self.Q_lo_2.state_dict(),
            'log_alpha_hi': self.log_alpha_hi,
            'log_alpha_lo': self.log_alpha_lo,
            'world_model':  self.wm.model.state_dict(),
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.pi_hi.load_state_dict(ckpt['pi_hi'])
        self.pi_lo.load_state_dict(ckpt['pi_lo'])
        self.skill_emb.load_state_dict(ckpt['skill_emb'])
        self.Q_hi_1.load_state_dict(ckpt['Q_hi_1'])
        self.Q_hi_2.load_state_dict(ckpt['Q_hi_2'])
        self.Q_lo_1.load_state_dict(ckpt['Q_lo_1'])
        self.Q_lo_2.load_state_dict(ckpt['Q_lo_2'])
        self.step = ckpt.get('step', 0)
        print(f"Loaded: {path}  step={self.step}")
        return self.step


# ─────────────────────────────────────────────────────────────────────────────
# Environment Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_goal_obs(env_name: str) -> np.ndarray:
    """
    Franka Kitchen 목표 obs 구성.
    모든 task가 완료된 상태 = obs_element_goals 값으로 채운 obs.
    """
    import gym, d4rl
    env  = gym.make(env_name)
    obs  = env.reset()
    goal = obs.copy()
    for task, idx in OBS_ELEMENT_INDICES.items():
        g_val = OBS_ELEMENT_GOALS[task]
        goal[idx] = g_val
    env.close()
    return goal


def build_context_from_env(env, acts_recent: List[np.ndarray],
                           x_seq_full: np.ndarray,
                           model: KoopmanCVAE,
                           device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    최근 obs/action 시퀀스로 z_t, h_t 계산.
    acts_recent: 최근 cond_len개 action
    """
    raise NotImplementedError("Use rollout_context instead")


class EnvContext:
    """
    Episode 동안의 z_t, h_t 상태를 tracking.
    """
    def __init__(self, model: KoopmanCVAE, x_cache: np.ndarray,
                 device: str, cond_len: int = 16):
        self.model    = model
        self.x_cache  = x_cache   # 이 env에서는 직접 x_t 계산 필요
        self.device   = device
        self.cond_len = cond_len

        self.obs_buf: List[np.ndarray]  = []
        self.act_buf: List[np.ndarray]  = []
        self.z_t:   Optional[torch.Tensor] = None
        self.h_t:   Optional[torch.Tensor] = None

    def reset(self, init_obs: np.ndarray):
        self.obs_buf = [init_obs]
        self.act_buf = []
        self.z_t = None
        self.h_t = None
        self._ref_obs = init_obs.copy()

    def _obs_to_x(self, obs: np.ndarray) -> np.ndarray:
        """
        D4RL obs(60-dim) → x_t(2108-dim).
        Δe=0 (R3M 없이), Δp, Δq, qdot만 사용.
        episode-first diff.
        """
        x = np.zeros(2108, dtype=np.float32)
        # Δp: obj_state diff (obs[18:60] - ref[18:60])
        x[X_DP_START:X_DP_END] = (obs[18:60] - self._ref_obs[18:60]).astype(np.float32)
        # Δq: joint pos diff
        x[X_DQ_START:X_DQ_END] = (obs[0:9]   - self._ref_obs[0:9]).astype(np.float32)
        # qdot
        x[X_QD_START:X_QD_END] = obs[9:18].astype(np.float32)
        return x

    def step(self, obs: np.ndarray, action: np.ndarray):
        """obs, action을 받아서 z_t, h_t 업데이트."""
        self.obs_buf.append(obs)
        self.act_buf.append(action)

        # x_t 계산
        T = min(len(self.act_buf), self.cond_len)
        obs_win = self.obs_buf[-T-1:-1]   # action 이전 obs
        act_win = self.act_buf[-T:]

        if len(act_win) < 1:
            return

        dev = torch.device(self.device)
        x_win = np.array([self._obs_to_x(o) for o in obs_win])  # (T, 2108)
        a_win = np.array(act_win)                                 # (T, 9)

        x_t = torch.FloatTensor(x_win).unsqueeze(0).to(dev)
        a_t = torch.FloatTensor(a_win).unsqueeze(0).to(dev)

        with torch.no_grad():
            enc    = self.model.encode_sequence(x_t, a_t)
            self.z_t = enc['o_seq'][0, -1:]   # (1, m)
            self.h_t = enc['h_seq'][0, -1:]   # (1, d_h)

    def get_z_goal(self, goal_obs: np.ndarray) -> torch.Tensor:
        """목표 obs → z_goal"""
        if self.h_t is None:
            return torch.zeros(1, self.model.cfg.koopman_dim,
                               device=self.device)
        x_goal = self._obs_to_x(goal_obs)
        x_goal_t = torch.FloatTensor(x_goal).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.posterior(x_goal_t, self.h_t)
        return mu

    # x_t slice constants 추가
    @staticmethod
    def _get_x_slices():
        return X_DP_START, X_DP_END, X_DQ_START, X_DQ_END

# x_t QD slice (eval_iql_policy와 동일)
X_QD_START = 2099; X_QD_END = 2108


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def compute_r_blend(r_env: float, r_hat: float, step: int,
                    warmup: int) -> float:
    """
    α = min(1.0, step / warmup)
    r_blend = α * r_env + (1-α) * r_hat
    초반: world model reward 비중 높음 (sparse r_env 보완)
    후반: real env reward 비중 높음
    """
    alpha = min(1.0, step / max(warmup, 1))
    return alpha * r_env + (1 - alpha) * r_hat


def visualize_training(log: Dict, out_path: str):
    keys = ['loss_q_hi', 'loss_pi_hi', 'kl_hi',
            'loss_q_lo', 'loss_pi_lo',
            'ep_reward', 'ep_tasks', 'alpha_hi', 'alpha_lo']
    titles = ['Q Hi Loss', 'π Hi Loss', 'KL Hi',
              'Q Lo Loss', 'π Lo Loss',
              'Episode Reward', 'Tasks Completed',
              'α Hi', 'α Lo']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    PAL  = ['#E53935','#1E88E5','#43A047','#FB8C00',
            '#8E24AA','#00ACC1','#FFB300','#607D8B','#795548']

    for i, (key, title) in enumerate(zip(keys, titles)):
        if key not in log or not log[key]:
            continue
        vals = np.array(log[key])
        ax   = axes[i]
        ax.plot(vals, color=PAL[i], alpha=0.25, lw=0.8)
        w = max(1, min(50, len(vals) // 5))
        if len(vals) >= w:
            smooth = np.convolve(vals, np.ones(w)/w, mode='valid')
            ax.plot(smooth, color=PAL[i], lw=1.8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('KODAQ-Online Training Curves', fontsize=12, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved curves: {out_path}")


def evaluate(trainer: KODAQOnlineTrainer,
             wm: KoopmanWorldModelWrapper,
             env_name: str,
             goal_obs: np.ndarray,
             n_ep: int,
             cfg: OnlineConfig,
             device: str) -> Dict:
    import gym, d4rl
    dev  = torch.device(device)
    model= wm.model
    model.eval()

    results = []
    for ep_i in range(n_ep):
        import gym
        env = gym.make(env_name)
        obs = env.reset()
        ctx = EnvContext(model, None, device, cfg.cond_len)
        ctx.reset(obs)

        total_r, n_tasks, done_ep = 0.0, 0, False
        skill_id = 0
        hi_timer = 0

        for t in range(280):   # kitchen episode 최대 길이
            if ctx.z_t is None:
                # 초반: random action으로 context 채우기
                a = env.action_space.sample()
                obs_next, r, done_ep, info = env.step(a)
                ctx.step(obs_next, a)
                total_r += r
                if done_ep: break
                continue

            z_t   = ctx.z_t
            z_g   = ctx.get_z_goal(goal_obs)

            # Hi-level: H_hi마다 skill 재선택
            if hi_timer == 0:
                with torch.no_grad():
                    sid_t, _ = trainer.pi_hi.sample(z_t, z_g, greedy=True)
                skill_id  = sid_t.item()
                hi_timer  = cfg.H_hi

            se = trainer.skill_emb(
                torch.tensor([skill_id], device=dev))  # (1, e)

            # Lo-level: H_lo step action sequence 생성
            with torch.no_grad():
                a_seq, _ = trainer.pi_lo.sample(z_t, se, z_g)
            a_np = a_seq[0].cpu().numpy()   # (H_lo, 9)

            # H_lo step 실행
            for k in range(cfg.H_lo):
                if done_ep or t + k >= 280:
                    break
                a_k = a_np[k].clip(-1, 1)
                obs_next, r, done_ep, info = env.step(a_k)
                ctx.step(obs_next, a_k)
                total_r += r
                n_tasks = max(n_tasks, int(info.get('num_success', 0)))
                if done_ep:
                    break

            hi_timer = max(0, hi_timer - cfg.H_lo)
            if done_ep:
                break

        results.append({'reward': total_r, 'n_tasks': n_tasks})
        env.close()

    mean_r = np.mean([r['reward']  for r in results])
    mean_t = np.mean([r['n_tasks'] for r in results])
    print(f"  [EVAL] mean_reward={mean_r:.3f}  mean_tasks={mean_t:.2f}")
    return {'mean_reward': mean_r, 'mean_tasks': mean_t}


def train(cfg: OnlineConfig, trainer: KODAQOnlineTrainer,
          wm: KoopmanWorldModelWrapper,
          env_name: str, goal_obs: np.ndarray,
          out_dir: str, device: str):
    import gym, d4rl
    dev   = torch.device(device)
    model = wm.model
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log: Dict[str, List] = {k: [] for k in [
        'loss_q_hi', 'loss_pi_hi', 'kl_hi',
        'loss_q_lo', 'loss_pi_lo',
        'ep_reward', 'ep_tasks',
        'alpha_hi', 'alpha_lo',
        'wm_loss',
    ]}
    recent = {k: deque(maxlen=cfg.log_every) for k in log}

    env   = gym.make(env_name)
    obs   = env.reset()
    ctx   = EnvContext(model, None, device, cfg.cond_len)
    ctx.reset(obs)

    ep_r, ep_tasks, ep_steps = 0.0, 0, 0
    skill_id, hi_timer = 0, 0
    global_step = trainer.step
    t0 = time.time()

    # World model update용 미니 버퍼
    wm_obs_buf, wm_act_buf, wm_obs_next_buf, wm_r_env_buf = [], [], [], []

    print(f"\n{'='*60}")
    print(f"KODAQ-Online Training  |  steps={cfg.n_env_steps}")
    print(f"  H_hi={cfg.H_hi}  H_lo={cfg.H_lo}  γ={cfg.gamma}")
    print(f"  env={env_name}")
    print(f"{'='*60}\n")

    while global_step < cfg.n_env_steps:

        # ── Action selection ───────────────────────────────────────────────
        if global_step < cfg.warmup_random or ctx.z_t is None:
            a_seq_np = np.array([env.action_space.sample()
                                 for _ in range(cfg.H_lo)])
        else:
            z_t = ctx.z_t
            z_g = ctx.get_z_goal(goal_obs)

            if hi_timer == 0:
                with torch.no_grad():
                    sid_t, _ = trainer.pi_hi.sample(z_t, z_g)
                skill_id = sid_t.item()
                hi_timer = cfg.H_hi

            se = trainer.skill_emb(torch.tensor([skill_id], device=dev))
            with torch.no_grad():
                a_seq, _ = trainer.pi_lo.sample(z_t, se, z_g)
            a_seq_np = a_seq[0].cpu().numpy()   # (H_lo, 9)

        # ── Execute H_lo steps in env ──────────────────────────────────────
        r_env_total = 0.0
        r_hat_total = 0.0
        z_before = ctx.z_t.clone() if ctx.z_t is not None else None
        h_before = ctx.h_t.clone() if ctx.h_t is not None else None
        obs_before = obs.copy()

        for k in range(cfg.H_lo):
            a_k = a_seq_np[k].clip(-1, 1)
            obs_next, r_env, done, info = env.step(a_k)

            r_env_total += r_env
            ep_r += r_env
            ep_tasks = max(ep_tasks, int(info.get('num_success', 0)))
            ep_steps += 1
            global_step += 1

            # World model update 버퍼 (실제 env reward 포함)
            x_cur  = ctx._obs_to_x(obs)
            x_next = ctx._obs_to_x(obs_next)
            wm_obs_buf.append(x_cur)
            wm_act_buf.append(a_k)
            wm_obs_next_buf.append(x_next)
            wm_r_env_buf.append(float(r_env))
            if len(wm_obs_buf) > 512:
                wm_obs_buf.pop(0); wm_act_buf.pop(0)
                wm_obs_next_buf.pop(0); wm_r_env_buf.pop(0)

            # World model 1-step reward
            if ctx.z_t is not None:
                with torch.no_grad():
                    a_t = torch.FloatTensor(a_k).unsqueeze(0).to(dev)
                    _, _, r_hat = wm.koopman_step(ctx.z_t, a_t, ctx.h_t)
                    r_hat_total += r_hat

            ctx.step(obs_next, a_k)
            obs = obs_next

            if done:
                break

        # Reward blend
        r_blend = compute_r_blend(r_env_total, r_hat_total,
                                  global_step, cfg.r_alpha_warmup)

        # ── Add to buffer ──────────────────────────────────────────────────
        if z_before is not None and h_before is not None and ctx.z_t is not None:
            z_g  = ctx.get_z_goal(goal_obs)
            se_np = trainer.skill_emb(
                torch.tensor([skill_id], device=dev)
            ).detach().cpu().numpy()[0]

            # Hi-level (H_hi마다)
            if hi_timer == cfg.H_hi:
                trainer.buf.add_hi(
                    z=z_before.cpu().numpy()[0],
                    z_goal=z_g.cpu().numpy()[0],
                    h_t=h_before.cpu().numpy()[0],   # skill prior 계산용
                    skill_id=skill_id,
                    R_hi=r_blend,
                    z_next=ctx.z_t.cpu().numpy()[0],
                    done=float(done),
                )

            # Lo-level
            a_pad = np.zeros((cfg.H_lo, trainer.action_dim), dtype=np.float32)
            actual = min(len(a_seq_np), cfg.H_lo)
            a_pad[:actual] = a_seq_np[:actual]

            trainer.buf.add_lo(
                z=z_before.cpu().numpy()[0],
                z_goal=z_g.cpu().numpy()[0],
                skill_embed=se_np,
                a_seq=a_pad,
                r_lo=r_blend,
                z_next=ctx.z_t.cpu().numpy()[0],
                done=float(done),
            )

        hi_timer = max(0, hi_timer - cfg.H_lo)

        # ── Policy update ──────────────────────────────────────────────────
        for _ in range(cfg.n_updates_per_step):
            info_hi = trainer.update_hi()
            info_lo = trainer.update_lo()
            for k, v in {**info_hi, **info_lo}.items():
                recent[k].append(v)

        # ── World model update ─────────────────────────────────────────────
        if global_step % cfg.wm_update_freq == 0 and len(wm_obs_buf) >= 32:
            idx  = np.random.choice(len(wm_obs_buf),
                                    min(32, len(wm_obs_buf)), replace=False)
            x_b  = torch.FloatTensor(
                np.array([wm_obs_buf[i]      for i in idx])).to(dev)
            a_b  = torch.FloatTensor(
                np.array([wm_act_buf[i]      for i in idx])).to(dev)
            xn_b = torch.FloatTensor(
                np.array([wm_obs_next_buf[i] for i in idx])).to(dev)
            # 실제 env reward를 reward_head fine-tune target으로 전달
            r_b  = torch.FloatTensor(
                np.array([wm_r_env_buf[i]    for i in idx])).to(dev)
            wm_loss = wm.update(x_b, a_b, xn_b, r_b)
            recent['wm_loss'].append(wm_loss)

        # ── Episode end ────────────────────────────────────────────────────
        if done:
            recent['ep_reward'].append(ep_r)
            recent['ep_tasks'].append(ep_tasks)
            obs     = env.reset()
            ctx.reset(obs)
            ep_r    = 0.0; ep_tasks = 0; ep_steps = 0
            skill_id = 0; hi_timer = 0

        # ── Logging ───────────────────────────────────────────────────────
        if global_step % cfg.log_every == 0:
            means = {k: np.mean(list(recent[k]))
                     if recent[k] else 0.0 for k in log}
            for k in log:
                if recent[k]:
                    log[k].append(means[k])

            elapsed = time.time() - t0
            sps     = cfg.log_every / (elapsed + 1e-6)
            t0      = time.time()

            print(
                f"Step {global_step:7d} | "
                f"Q_hi={means['loss_q_hi']:.3f}  "
                f"π_hi={means['loss_pi_hi']:.3f}  "
                f"KL_hi={means['kl_hi']:.3f}  |  "
                f"Q_lo={means['loss_q_lo']:.3f}  "
                f"π_lo={means['loss_pi_lo']:.3f}  |  "
                f"ep_r={means['ep_reward']:.3f}  "
                f"tasks={means['ep_tasks']:.2f}  |  "
                f"{sps:.0f} sps"
            )

        if global_step % cfg.save_every == 0:
            trainer.step = global_step
            trainer.save(f"{out_dir}/kodaq_online_step{global_step}.pt")
            visualize_training(log, f"{out_dir}/training_curves.png")

        if global_step % cfg.eval_every == 0:
            eval_res = evaluate(trainer, wm, env_name, goal_obs,
                                cfg.n_eval_ep, cfg, device)
            log['ep_reward'].append(eval_res['mean_reward'])
            log['ep_tasks'].append(eval_res['mean_tasks'])

    # Final save
    trainer.step = global_step
    trainer.save(f"{out_dir}/kodaq_online_final.pt")
    visualize_training(log, f"{out_dir}/training_curves_final.png")
    env.close()

    print(f"\n{'='*55}")
    print(f"Training complete. {global_step} steps")
    print(f"  Outputs → {out_dir}/")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_ckpt', default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--resume',     default=None,
                   help='Resume from kodaq_online checkpoint')
    p.add_argument('--env',        default='kitchen-mixed-v0')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v4/online')
    # Hierarchy
    p.add_argument('--H_hi',       type=int,   default=8)
    p.add_argument('--H_lo',       type=int,   default=4,
                   help='Low-level action sequence length (4~8)')
    # SAC
    p.add_argument('--gamma',      type=float, default=0.99)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--kl_weight',  type=float, default=1.0)
    # Training
    p.add_argument('--n_steps',    type=int,   default=300_000)
    p.add_argument('--warmup',     type=int,   default=1_000)
    p.add_argument('--r_warmup',   type=int,   default=10_000,
                   help='Steps to blend world model reward → real reward')
    p.add_argument('--wm_lr',      type=float, default=1e-4)
    # Eval
    p.add_argument('--eval_every', type=int,   default=25_000)
    p.add_argument('--n_eval_ep',  type=int,   default=5)
    p.add_argument('--device',     default='cuda:1'
                   if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── World model ─────────────────────────────────────────────────────────
    print(f"\nLoading world model: {args.world_ckpt}")
    ckpt  = torch.load(args.world_ckpt, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    m_cfg   = model.cfg
    z_dim   = m_cfg.koopman_dim
    n_skills= m_cfg.num_skills
    a_dim   = m_cfg.action_dim
    print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")

    cfg = OnlineConfig(
        H_hi=args.H_hi, H_lo=args.H_lo,
        gamma=args.gamma, lr=args.lr,
        batch_size=args.batch_size,
        kl_weight=args.kl_weight,
        n_env_steps=args.n_steps,
        warmup_random=args.warmup,
        r_alpha_warmup=args.r_warmup,
        wm_lr=args.wm_lr,
        eval_every=args.eval_every,
        n_eval_ep=args.n_eval_ep,
    )

    wm      = KoopmanWorldModelWrapper(model, cfg.wm_lr, device)
    trainer = KODAQOnlineTrainer(cfg, wm, z_dim, n_skills, a_dim, device)

    if args.resume and Path(args.resume).exists():
        trainer.load(args.resume)

    # Goal obs
    goal_obs = get_goal_obs(args.env)
    print(f"Goal obs shape: {goal_obs.shape}")

    train(cfg, trainer, wm, args.env, goal_obs, args.out_dir, device)


if __name__ == '__main__':
    main()