"""
kodaq_online.py — KODAQ-Online v3
===================================
IQL offline pretrained weights → Online hierarchical RL fine-tune

Architecture:
  π_hi(skill_probs | z_t)      Gumbel-Softmax K-dim skill distribution
  π_lo(a_{0:H_lo}   | z_t)     H_lo-step action chunk (IQL weight init)
  Q(z_t, u_t, skill_probs)     action encoder latent + skill bin probs
  V = min(Q1_t, Q2_t)          target network value

Training:
  Q  : TD(γ^H_lo) target, EMA soft update
  Hi : KL(sp || p_skill(h)) - Q
  Lo : AWR exp(β·Adv) + KL(π_lo || π_iql_frozen)
  Reward: compute_r_blend(r_env, r_acc, r_event)
            w_env=0.4, w_acc=0.4, w_event=0.2

Usage:
    MUJOCO_GL=egl python kodaq_online.py \\
        --world_ckpt checkpoints/kodaq_v4/final.pt \\
        --iql_ckpt   checkpoints/kodaq_v4/iql_v3/iql_final.pt \\
        --cat_ckpt   checkpoints/kodaq_v4/cat_reward/final.pt \\
        --x_cache    checkpoints/skill_pretrain/x_sequences.npz \\
        --env        kitchen-mixed-v0 \\
        --n_steps 1000000 --device cuda:1 \\
        --out_dir checkpoints/kodaq_v4/online_v9
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
from dataclasses import dataclass
from collections import deque

from models.koopman_cvae import KoopmanCVAE
from models.losses import q_categorical_loss, two_hot_decode, policy_prior_loss, symexp
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    blend_koopman,
    KODAQLQRPlanner, LQRConfig,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
    OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS,
    load_kitchen_episodes,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

X_QD_START, X_QD_END = 2099, 2108


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OnlineConfig:
    # Hierarchy
    H_hi:           int   = 8
    H_lo:           int   = 4

    # SAC / AWR
    gamma:          float = 0.99
    tau_ema:        float = 0.005
    alpha:          float = 0.0
    kl_weight:      float = 1.0
    kl_lqr_weight:  float = 0.1
    awr_beta:       float = 3.0
    gumbel_tau:     float = 1.0
    gumbel_tau_min: float = 0.3

    # Network
    hidden_dim:     int   = 256
    n_layers:       int   = 2

    # Optimization
    lr:             float = 3e-4
    batch_size:     int   = 256
    grad_clip:      float = 1.0
    n_updates_per_step: int = 1
    positive_fraction: float = 0.5
    offline_fraction:  float = 0.5
    reward_positive_fraction: float = 0.5
    reward_offline_fraction:  float = 0.5
    q_positive_fraction:      float = 0.05
    q_offline_fraction:       float = 0.8
    pi_positive_fraction:     float = 0.02
    pi_offline_fraction:      float = 0.8
    reward_elite_fraction:    float = 0.0
    q_elite_fraction:         float = 0.0
    pi_elite_fraction:        float = 0.0
    elite_buffer_size:        int   = 50_000
    elite_reward_threshold:   float = 1.0
    offline_elite_threshold:  float = 2.0
    offline_elite_verify_env: bool  = False
    lambda_elite_bc_pi:       float = 0.0
    lambda_elite_skill_bc_pi: float = 0.0
    lambda_elite_arg_bc_pi:   float = 0.0
    q_n_step:                 int   = 1
    q_n_step_gamma:           float = 0.99
    q_reward_source:          str   = 'env'
    q_bootstrap:              float = 0.0
    q_success_only:           bool  = False
    q_success_value:          float = 1.0
    q_success_threshold:      float = 0.5
    q_success_include_source: bool  = False
    lambda_q_negative:        float = 0.0
    lambda_q_success_rank:    float = 0.0
    q_success_margin:         float = 0.5
    lambda_q_wm_penalty:      float = 0.0
    q_wm_penalty_percentile:  float = 95.0
    q_wm_penalty_clip:        float = 1.0
    q_wm_penalty_min_elite:   int   = 128
    q_wm_penalty_q_weight:    float = 1.0
    q_wm_penalty_obj_weight:  float = 1.0
    pi_update_start:          int   = 10_000
    actor_mode:               str   = 'policy_prior'  # policy_prior | skill_decoder
    pi_q_weight:              float = 1.0
    pi_q_guide_loss_threshold: float = -1.0
    pi_q_guide_min_step:      int   = 0
    pi_q_guide_ema_beta:      float = 0.99
    action_noise_std:         float = 0.0
    skilldec_exec_horizon:    int   = 1
    lambda_bc_pi:             float = 0.0
    bc_target_clip:           float = 1.0
    lambda_q_cql_policy:      float = 0.0
    lambda_q_rank_data:       float = 0.0
    lambda_q_rank_bc:         float = 0.0
    lambda_q_rank_lqr:        float = 0.0
    q_rank_margin:            float = 0.05
    lambda_skilldec_anchor:   float = 0.0
    skilldec_anchor_min:      float = 0.0
    skilldec_anchor_decay_steps: int = 200_000
    skill_arg_alpha_d:        float = 0.1
    skill_arg_alpha_z:        float = 0.01
    skill_d_no_h:             bool  = False
    skill_arg_mean_after_pi_start: bool = True
    lambda_lqr_pi:     float = 0.0
    lqr_horizon:       int   = 4
    lqr_aux_k:         int   = 4
    lqr_aux_weight:    float = 0.25

    # World model
    wm_lr:          float = 1e-4
    wm_update_freq: int   = 10

    # Reward blend (iql와 동일)
    w_env:          float = 0.4
    w_acc:          float = 0.4
    w_event:        float = 0.2

    # Training
    n_env_steps:    int   = 1_000_000
    buffer_size:    int   = 200_000
    log_every:      int   = 1_000
    save_every:     int   = 50_000
    eval_every:     int   = 10_000
    n_eval_ep:      int   = 10
    cond_len:       int   = 16


# ─────────────────────────────────────────────────────────────────────────────
# Reward blend (iql와 동일한 가중치)
# ─────────────────────────────────────────────────────────────────────────────

def compute_r_blend(r_env: float, r_hat_acc: float = 0.0,
                    r_hat_event: float = 0.0,
                    w_env: float = 0.4, w_acc: float = 0.4,
                    w_event: float = 0.2) -> float:
    """
    3-way reward blend (iql_koopman과 동일한 가중치 구조).
    r_env:       실제 환경 보상
    r_hat_acc:   categorical accumulated head E[R|z] ∈ [0,4], /4 정규화
    r_hat_event: BCE event head P(task|z) ∈ (0,1)
    """
    MAX_ACC = 4.0
    r = (w_env   * float(r_env) +
         w_acc   * (float(r_hat_acc) / MAX_ACC) +
         w_event * float(r_hat_event))
    return float(np.clip(r, 0.0, 1.0))


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


class HighLevelPolicy(nn.Module):
    """π_hi(skill_probs | z_t): Categorical over K skills via Gumbel-Softmax"""
    def __init__(self, z_dim: int, n_skills: int, hidden: int, n_layers: int):
        super().__init__()
        self.n_skills = n_skills
        self.net = make_mlp(z_dim, n_skills, hidden, n_layers)

    def logits(self, z): return self.net(z)
    def probs(self, z):  return torch.softmax(self.logits(z), dim=-1)

    def gumbel_sample(self, z: torch.Tensor,
                      tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sp (B,K) soft, log_prob (B,))"""
        logits = self.logits(z)
        sp     = F.gumbel_softmax(logits, tau=tau, hard=False)
        lp     = (sp * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return sp, lp

    def kl_prior(self, sp_new: torch.Tensor,
                 p_prior: torch.Tensor) -> torch.Tensor:
        """KL(sp_new || p_prior): bin 확률 벡터 간 KL divergence"""
        return (sp_new * (torch.log(sp_new + 1e-8) -
                          torch.log(p_prior + 1e-8))).sum(dim=-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.logits(z)).entropy()


class LowLevelPolicy(nn.Module):
    """π_lo(a_{0:H_lo} | z_t): H_lo-step action chunk, IQL ChunkPolicy와 동일 구조"""
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

    def sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, ls = self(z)
        u = torch.distributions.Normal(mu, ls.exp()).rsample()
        a = torch.tanh(u)
        lp = (torch.distributions.Normal(mu, ls.exp()).log_prob(u)
              - torch.log(1 - a.pow(2) + 1e-6))
        return a, lp.sum(dim=(-2, -1))

    def log_prob(self, z: torch.Tensor,
                 a_chunk: torch.Tensor) -> torch.Tensor:
        """a_chunk: (B, H_lo, action_dim) tanh-squashed"""
        mu, ls = self(z)
        u   = torch.atanh(a_chunk.clamp(-1 + 1e-6, 1 - 1e-6))
        dist= torch.distributions.Normal(mu, ls.exp())
        lp  = dist.log_prob(u) - torch.log(1 - a_chunk.pow(2) + 1e-6)
        return lp.sum(dim=(-2, -1))

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        _, ls = self(z)
        return (0.5 * (1 + 2 * ls + math.log(2 * math.pi))).sum(dim=(-2, -1))


class QNetwork(nn.Module):
    """Q(z_t, u_t, skill_probs) → scalar. IQL QNetwork과 동일 구조/weight 호환"""
    def __init__(self, z_dim: int, u_dim: int, n_skills: int,
                 hidden: int, n_layers: int):
        super().__init__()
        self.net = make_mlp(z_dim + u_dim + n_skills, 1,
                            hidden, n_layers, output_scale=0.01)

    def forward(self, z: torch.Tensor, u: torch.Tensor,
                sp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, u, sp], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device   = device
        self._d: Dict[str, np.ndarray] = {}
        self._ptr = 0
        self._n   = 0

    def _init(self, z_dim, n_skills, action_dim, H_lo, h_dim):
        C = self.capacity
        self._d = {
            'z':     np.zeros((C, z_dim),           dtype=np.float32),
            'z_next':np.zeros((C, z_dim),           dtype=np.float32),
            'h_t':   np.zeros((C, h_dim),           dtype=np.float32),
            'sp':    np.zeros((C, n_skills),         dtype=np.float32),
            'a_seq': np.zeros((C, H_lo, action_dim), dtype=np.float32),
            'r':     np.zeros(C,                    dtype=np.float32),
            'done':  np.zeros(C,                    dtype=np.float32),
        }

    def add(self, z, z_next, h_t, sp, a_seq, r, done):
        if not self._d:
            self._init(z.shape[-1], sp.shape[-1], a_seq.shape[-1],
                       a_seq.shape[-2], h_t.shape[-1])
        p = self._ptr
        self._d['z'][p]     = z;     self._d['z_next'][p] = z_next
        self._d['h_t'][p]   = h_t;   self._d['sp'][p]     = sp
        self._d['a_seq'][p] = a_seq; self._d['r'][p]      = r
        self._d['done'][p]  = float(done)
        self._ptr = (p + 1) % self.capacity
        self._n   = min(self._n + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._n, batch_size)
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in self._d.items()}

    @property
    def size(self): return self._n


# ─────────────────────────────────────────────────────────────────────────────
# World Model Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanWorldModelWrapper:
    """Koopman CVAE wrapper: reward heads + world model update."""

    def __init__(self, model: KoopmanCVAE, wm_lr: float, device: str,
                 cat_head=None, reward_H: int = 8, reward_gamma: float = 0.9):
        self.model        = model
        self.device       = device
        self.cat_head     = cat_head
        self.reward_H     = reward_H
        self.reward_gamma = reward_gamma

        # Freeze Koopman operator and decoder reconstruction
        for p in model.koopman.parameters():     p.requires_grad_(False)
        for p in model.decoder.parameters():     p.requires_grad_(False)

        # Active: posterior, recurrent, skill_prior, reward_head
        reward_params = []
        if getattr(model.cfg, 'use_reward_head', True):
            head = getattr(model.decoder, 'head_reward',
                           getattr(model, 'reward_head', None))
            if head is not None:
                for p in head.parameters(): p.requires_grad_(True)
                reward_params = list(head.parameters())

        active = (list(model.posterior.parameters()) +
                  list(model.recurrent.parameters()) +
                  list(model.skill_prior.parameters()) +
                  reward_params)
        self.opt = torch.optim.Adam(active, lr=wm_lr)

    def _bce_head(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        head = getattr(self.model.decoder, 'head_reward',
                       getattr(self.model, 'reward_head', None))
        return head(z) if head is not None else None

    def _r_hat_event(self, z: torch.Tensor) -> float:
        """P(task completion | z_t) via BCE head."""
        if not getattr(self.model.cfg, 'use_reward_head', True): return 0.0
        logit = self._bce_head(z)
        return torch.sigmoid(logit).mean().item() if logit is not None else 0.0

    def _r_hat_accumulated(self, z0: torch.Tensor, h0: torch.Tensor,
                           a_seq: torch.Tensor) -> float:
        """Σ_{k=0}^{H-1} γ^k * E[R | z_{k+1}] via Koopman rollout."""
        m   = self.model
        H   = min(self.reward_H, len(a_seq))
        gm  = self.reward_gamma
        z, h = z0, h0
        r_acc= 0.0
        wll  = m.koopman.get_log_lambdas()
        for k in range(H):
            ak = a_seq[k].unsqueeze(0) if a_seq.dim() == 2 else a_seq[k:k+1]
            u  = m.action_encoder(ak)
            w  = m.skill_prior.soft_weights(h)
            A, B, _, _ = blend_koopman(wll, m.koopman.theta_k,
                                       m.koopman.G_k, m.koopman.U, w)
            z_nx = (A[0] @ z.T).T + (B[0] @ u.T).T
            h_nx = m.recurrent(h, z, ak)
            if self.cat_head is not None:
                r_acc += (gm**k) * self.cat_head.expected_reward(z_nx).mean().item()
            else:
                r_acc += (gm**k) * self._r_hat_event(z_nx)
            z, h = z_nx, h_nx
        return r_acc

    def update(self, x_b: torch.Tensor, r_env_b: torch.Tensor) -> float:
        """Fine-tune reward head with BCE loss."""
        if not self.model.cfg.use_reward_head: return 0.0
        m = self.model; m.train()
        h0 = torch.zeros(x_b.shape[0], m.cfg.gru_hidden, device=self.device)
        z, _ = m.posterior(x_b, h0)
        logit = self._bce_head(z)
        if logit is None: m.eval(); return 0.0
        loss = F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), r_env_b.clamp(0, 1).float())
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 1.0)
        self.opt.step(); m.eval()
        return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# EnvContext: z_t, h_t 추적
# ─────────────────────────────────────────────────────────────────────────────

class EnvContext:
    def __init__(self, model: KoopmanCVAE, device: str, cond_len: int = 16):
        self.model    = model
        self.device   = device
        self.cond_len = cond_len
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[np.ndarray] = []
        self.z_t: Optional[torch.Tensor] = None
        self.h_t: Optional[torch.Tensor] = None
        self._ref: Optional[np.ndarray]  = None

    def reset(self, obs: np.ndarray):
        self.obs_buf = [obs]; self.act_buf = []
        self.z_t = None; self.h_t = None
        self._ref = obs.copy()

    def _obs_to_x(self, obs: np.ndarray) -> np.ndarray:
        x = np.zeros(2108, dtype=np.float32)
        x[X_DP_START:X_DP_END] = (obs[18:60] - self._ref[18:60])
        x[X_DQ_START:X_DQ_END] = (obs[0:9]   - self._ref[0:9])
        x[X_QD_START:X_QD_END] = obs[9:18]
        return x

    def step(self, obs: np.ndarray, action: np.ndarray):
        self.obs_buf.append(obs); self.act_buf.append(action)
        T = min(len(self.act_buf), self.cond_len)
        if T < 1: return
        dev = torch.device(self.device)
        xw  = np.array([self._obs_to_x(o) for o in self.obs_buf[-T-1:-1]])
        aw  = np.array(self.act_buf[-T:])
        with torch.no_grad():
            enc = self.model.encode_sequence(
                torch.FloatTensor(xw).unsqueeze(0).to(dev),
                torch.FloatTensor(aw).unsqueeze(0).to(dev))
            self.z_t = enc['o_seq'][0, -1:]
            self.h_t = enc['h_seq'][0, -1:]

    @torch.no_grad()
    def current_latent(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.h_t is None or not self.obs_buf:
            return None
        dev = torch.device(self.device)
        h = self.h_t
        x_now = torch.FloatTensor(
            self._obs_to_x(self.obs_buf[-1])).unsqueeze(0).to(dev)
        z_now, _, _ = self.model.posterior.sample(x_now, h)
        return z_now, h


# ─────────────────────────────────────────────────────────────────────────────
# KODAQ Online Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PriorReplayBuffer:
    def __init__(self, capacity: int, device: str, h_dim: int = 0,
                 action_dim: int = 0, chunk_horizon: int = 1,
                 arg_dim: int = 0, skill_dim: int = 0):
        self.capacity = capacity
        self.device = device
        self.h_dim = h_dim
        self.action_dim = int(action_dim)
        self.chunk_horizon = max(1, int(chunk_horizon))
        self.arg_dim = int(arg_dim)
        self.skill_dim = int(skill_dim)
        self._d: Dict[str, np.ndarray] = {}
        self._ptr = 0
        self._n = 0

    def _init(self, z_dim: int, u_dim: int):
        C = self.capacity
        h_dim = int(self.h_dim)
        a_dim = int(self.action_dim or u_dim)
        H = int(self.chunk_horizon)
        self._d = {
            'z': np.zeros((C, z_dim), dtype=np.float32),
            'h': np.zeros((C, h_dim), dtype=np.float32),
            'u': np.zeros((C, u_dim), dtype=np.float32),
            'a': np.zeros((C, a_dim), dtype=np.float32),
            'a_chunk': np.zeros((C, H, a_dim), dtype=np.float32),
            'chunk_mask': np.zeros((C, H), dtype=np.float32),
            'r': np.zeros(C, dtype=np.float32),
            'r_return': np.zeros(C, dtype=np.float32),
            'z_next': np.zeros((C, z_dim), dtype=np.float32),
            'h_next': np.zeros((C, h_dim), dtype=np.float32),
            'z_boot': np.zeros((C, z_dim), dtype=np.float32),
            'h_boot': np.zeros((C, h_dim), dtype=np.float32),
            'discount': np.zeros(C, dtype=np.float32),
            'done': np.zeros(C, dtype=np.float32),
            'u_lqr': np.zeros((C, u_dim), dtype=np.float32),
            'u_lqr_aux': np.zeros((C, u_dim), dtype=np.float32),
            'has_lqr': np.zeros(C, dtype=np.float32),
            'skill_id': np.zeros(C, dtype=np.float32),
            'skill_prob': np.zeros((C, int(self.skill_dim)), dtype=np.float32),
            'skill_arg': np.zeros((C, int(self.arg_dim)), dtype=np.float32),
            'has_skill_arg': np.zeros(C, dtype=np.float32),
            'wm_err': np.zeros(C, dtype=np.float32),
            'source': np.zeros(C, dtype=np.float32),
        }

    def add(self, z, u, r, z_next, done,
            h=None, h_next=None,
            a=None, a_chunk=None, chunk_mask=None,
            r_return=None, z_boot=None, h_boot=None, discount=None,
            u_lqr=None, u_lqr_aux=None, has_lqr: float = 0.0,
            skill_id=None, skill_prob=None,
            skill_arg=None, has_skill_arg: float = 0.0,
            wm_err: float = 0.0,
            source: float = 0.0):
        if not self._d:
            self._init(z.shape[-1], u.shape[-1])
        p = self._ptr
        self._d['z'][p] = z
        if h is not None and self.h_dim > 0:
            self._d['h'][p] = h
        self._d['u'][p] = u
        if a is not None:
            self._d['a'][p] = a
        self._d['a_chunk'][p].fill(0.0)
        self._d['chunk_mask'][p].fill(0.0)
        if a_chunk is not None:
            arr = np.asarray(a_chunk, dtype=np.float32)
            n = min(arr.shape[0], self._d['a_chunk'].shape[1])
            self._d['a_chunk'][p, :n] = arr[:n]
            if chunk_mask is None:
                self._d['chunk_mask'][p, :n] = 1.0
            else:
                mask = np.asarray(chunk_mask, dtype=np.float32)
                self._d['chunk_mask'][p, :n] = mask[:n]
        self._d['r'][p] = r
        self._d['r_return'][p] = float(r if r_return is None else r_return)
        self._d['z_next'][p] = z_next
        if h_next is not None and self.h_dim > 0:
            self._d['h_next'][p] = h_next
        self._d['z_boot'][p] = z_next if z_boot is None else z_boot
        if self.h_dim > 0:
            if h_boot is not None:
                self._d['h_boot'][p] = h_boot
            elif h_next is not None:
                self._d['h_boot'][p] = h_next
        self._d['discount'][p] = float(0.0 if done else (
            1.0 if discount is None else discount))
        self._d['done'][p] = float(done)
        if u_lqr is not None:
            self._d['u_lqr'][p] = u_lqr
        if u_lqr_aux is not None:
            self._d['u_lqr_aux'][p] = u_lqr_aux
        self._d['has_lqr'][p] = float(has_lqr)
        if self.skill_dim > 0:
            self._d['skill_prob'][p].fill(0.0)
        if skill_id is not None:
            self._d['skill_id'][p] = float(skill_id)
            if self.skill_dim > 0:
                sid = int(skill_id)
                if 0 <= sid < self.skill_dim:
                    self._d['skill_prob'][p, sid] = 1.0
        if skill_prob is not None and self.skill_dim > 0:
            prob = np.asarray(skill_prob, dtype=np.float32)
            n = min(prob.shape[-1], self.skill_dim)
            self._d['skill_prob'][p, :n] = prob[:n]
        if skill_arg is not None and self.arg_dim > 0:
            self._d['skill_arg'][p] = np.asarray(skill_arg, dtype=np.float32)
            has_skill_arg = 1.0
        self._d['has_skill_arg'][p] = float(has_skill_arg)
        self._d['wm_err'][p] = float(wm_err)
        self._d['source'][p] = float(source)
        self._ptr = (p + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def _choice(self, candidates: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or len(candidates) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.random.choice(candidates, n, replace=len(candidates) < n)

    def sample(self, batch_size: int,
               positive_fraction: float = 0.0,
               offline_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        valid = np.arange(self._n)
        idx_parts = []

        n_pos = int(round(batch_size * max(0.0, min(1.0, positive_fraction))))
        pos = valid[self._d['r'][:self._n] > 0.0]
        idx_parts.append(self._choice(pos, n_pos))

        n_off = int(round(batch_size * max(0.0, min(1.0, offline_fraction))))
        off = valid[self._d['source'][:self._n] > 0.5]
        idx_parts.append(self._choice(off, n_off))

        n_used = sum(len(x) for x in idx_parts)
        n_rest = max(0, batch_size - n_used)
        idx_parts.append(self._choice(valid, n_rest))
        idx = np.concatenate(idx_parts) if idx_parts else self._choice(valid, batch_size)
        if len(idx) > batch_size:
            idx = np.random.choice(idx, batch_size, replace=False)
        np.random.shuffle(idx)
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in self._d.items()}

    @property
    def size(self): return self._n


class PolicyPriorOnlineTrainer:
    def __init__(self, cfg: OnlineConfig, model: KoopmanCVAE, device: str,
                 action_inv_steps: int = 30, action_inv_lr: float = 0.05,
                 lqr_planner: Optional[KODAQLQRPlanner] = None):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.step = 0
        self.q_loss_ema: Optional[float] = None
        self.q_guide_active = cfg.pi_q_guide_loss_threshold <= 0.0
        self.elite_return_ema: float = 0.0
        self.elite_return_last: float = 0.0
        self.elite_episode_count: int = 0
        self.wm_penalty_threshold: float = 0.0
        self.action_inv_steps = action_inv_steps
        self.action_inv_lr = action_inv_lr
        self.lqr_planner = lqr_planner
        self.actor_mode = cfg.actor_mode.lower()
        if self.actor_mode not in ('policy_prior', 'skill_decoder', 'skill_arg'):
            raise ValueError(
                f"Unknown actor_mode={cfg.actor_mode!r}; "
                "expected policy_prior, skill_decoder, or skill_arg")
        skill_h = int(getattr(model.cfg, 'skill_decoder_horizon',
                              cfg.skilldec_exec_horizon))
        arg_dim = int(getattr(model.cfg, 'skill_arg_dim', 0))
        self.buf = PriorReplayBuffer(
            cfg.buffer_size, device, h_dim=model.cfg.gru_hidden,
            action_dim=model.cfg.action_dim, chunk_horizon=skill_h,
            arg_dim=arg_dim, skill_dim=model.cfg.num_skills)
        self.elite_buf = PriorReplayBuffer(
            cfg.elite_buffer_size, device, h_dim=model.cfg.gru_hidden,
            action_dim=model.cfg.action_dim, chunk_horizon=skill_h,
            arg_dim=arg_dim, skill_dim=model.cfg.num_skills
        ) if cfg.elite_buffer_size > 0 else None
        self.skill_decoder_anchor = copy.deepcopy(model.skill_action_decoder)
        self.skill_decoder_anchor.to(device).eval()
        for p in self.skill_decoder_anchor.parameters():
            p.requires_grad_(False)
        self.skill_arg_policy_anchor = copy.deepcopy(model.skill_argument_policy)
        self.skill_arg_policy_anchor.to(device).eval()
        for p in self.skill_arg_policy_anchor.parameters():
            p.requires_grad_(False)

        for p in model.parameters():
            p.requires_grad_(False)
        if self.actor_mode == 'skill_decoder':
            actor_mod = model.skill_action_decoder
        elif self.actor_mode == 'skill_arg':
            actor_mod = nn.ModuleList([
                model.skill_discrete_policy,
                model.skill_argument_policy,
            ])
        else:
            actor_mod = model.policy_prior
        q_mod = model.skill_arg_q_head if self.actor_mode == 'skill_arg' else model.q_head
        for mod in [model.reward_ensemble_head, q_mod, actor_mod]:
            for p in mod.parameters():
                p.requires_grad_(True)
            mod.train()
        model.q_head_target.eval()
        model._detach_q_head.eval()
        if hasattr(model, 'skill_arg_q_head_target'):
            model.skill_arg_q_head_target.eval()
            model._detach_skill_arg_q_head.eval()

        self.opt_reward = torch.optim.Adam(model.reward_ensemble_head.parameters(),
                                           lr=cfg.wm_lr)
        self.opt_q = torch.optim.Adam(q_mod.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(actor_mod.parameters(), lr=cfg.lr)
        self._last_skillarg_meta: Dict[str, np.ndarray] = {}

    def decode_action(self, u: torch.Tensor) -> np.ndarray:
        if u.dim() == 1:
            u = u.unsqueeze(0)
        u_target = u.detach().to(self.device)
        da = self.model.cfg.action_dim
        a = torch.zeros(u_target.shape[0], da, device=self.device,
                        requires_grad=True)
        opt = torch.optim.Adam([a], lr=self.action_inv_lr)
        with torch.enable_grad():
            for _ in range(self.action_inv_steps):
                opt.zero_grad()
                loss = F.mse_loss(self.model.action_encoder(a), u_target)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    a.clamp_(-1.0, 1.0)
        return a.detach()[0].cpu().numpy()

    def _skill_prior_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.skill_d_no_h:
            return h.new_zeros((*h.shape[:-1], self.model.cfg.num_skills))
        return self.model.skill_prior(h).detach()

    def _skill_policy_logits(self, z: torch.Tensor,
                             h: torch.Tensor) -> torch.Tensor:
        prior_logits = self._skill_prior_logits(h)
        h_policy = torch.zeros_like(h) if self.cfg.skill_d_no_h else h
        return self.model.skill_discrete_policy(z, h_policy, prior_logits)

    def _one_hot_skill(self, skill_id: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            skill_id.long(), num_classes=self.model.cfg.num_skills
        ).float()

    def _skill_arg_normal_kl(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_std: torch.Tensor,
    ) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        prior_var = torch.exp(2.0 * prior_log_std)
        kl = (
            prior_log_std - log_std
            + (var + (mean - prior_mean).pow(2)) / (2.0 * prior_var)
            - 0.5
        )
        return kl.sum(-1)

    def _sample_skillarg_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        deterministic: bool = False,
        mean_arg: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_d = self._skill_policy_logits(z, h)
        probs_d = torch.softmax(logits_d, dim=-1)
        if deterministic:
            skill_id = probs_d.argmax(dim=-1)
        else:
            skill_id = torch.multinomial(probs_d, num_samples=1).squeeze(-1)
        skill_oh = self._one_hot_skill(skill_id)
        if deterministic or mean_arg:
            arg = self.model.skill_argument_policy.mean_arg(z, h, skill_oh)
            log_arg = z.new_zeros(z.shape[:-1])
        else:
            arg, log_arg, _, _ = self.model.skill_argument_policy(z, h, skill_oh)
        a_seq = self.model.skill_argument_decoder(z, h, skill_oh, arg)
        return a_seq, skill_id, arg, log_arg

    def last_skillarg_meta(self) -> Dict[str, np.ndarray]:
        return dict(self._last_skillarg_meta)

    @torch.no_grad()
    def act_sequence(
        self,
        z: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.actor_mode == 'skill_decoder':
            if h is None:
                raise ValueError("h is required when actor_mode=skill_decoder")
            w = self.model.skill_prior.soft_weights(h)
            a_seq = self.model.skill_action_decoder(z, h, w)[0].clamp(-1.0, 1.0)
            n = max(1, int(horizon or self.cfg.skilldec_exec_horizon))
            a_seq = a_seq[:min(n, a_seq.shape[0])]
            if self.cfg.action_noise_std > 0.0:
                a_seq = (
                    a_seq + self.cfg.action_noise_std * torch.randn_like(a_seq)
                ).clamp(-1.0, 1.0)
            u_seq = self.model.action_encoder(a_seq)
            return a_seq.cpu().numpy(), u_seq.detach().cpu().numpy()
        if self.actor_mode == 'skill_arg':
            if h is None:
                raise ValueError("h is required when actor_mode=skill_arg")
            use_mean_arg = (
                self.cfg.skill_arg_mean_after_pi_start
                and self.step >= self.cfg.pi_update_start
            )
            a_seq_t, skill_id, arg, _ = self._sample_skillarg_action(
                z, h, deterministic=False, mean_arg=use_mean_arg)
            self._last_skillarg_meta = {
                'skill_id': skill_id.detach().cpu().numpy().copy(),
                'skill_arg': arg.detach().cpu().numpy().copy(),
            }
            a_seq = a_seq_t[0].clamp(-1.0, 1.0)
            n = max(1, int(horizon or self.cfg.skilldec_exec_horizon))
            a_seq = a_seq[:min(n, a_seq.shape[0])]
            if self.cfg.action_noise_std > 0.0:
                a_seq = (
                    a_seq + self.cfg.action_noise_std * torch.randn_like(a_seq)
                ).clamp(-1.0, 1.0)
            u_seq = self.model.action_encoder(a_seq)
            return a_seq.cpu().numpy(), u_seq.detach().cpu().numpy()
        u, _, _, _ = self.model.policy_prior(z)
        a = self.decode_action(u)
        return a[None], u[0:1].detach().cpu().numpy()

    @torch.no_grad()
    def act(self, z: torch.Tensor,
            h: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        a_seq, u_seq = self.act_sequence(z, h, horizon=1)
        return a_seq[0], u_seq[0]

    def _skilldec_action_seq(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        anchor: bool = False,
    ) -> torch.Tensor:
        w = self.model.skill_prior.soft_weights(h)
        dec = self.skill_decoder_anchor if anchor else self.model.skill_action_decoder
        return dec(z, h, w)

    def _skillarg_action_seq(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        anchor: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        w = self.model.skill_prior.soft_weights(h)
        policy = self.skill_arg_policy_anchor if anchor else self.model.skill_argument_policy
        if anchor or deterministic:
            arg = policy.mean_arg(z, h, w)
            log_pi = None
        else:
            arg, log_pi, _, _ = policy(z, h, w)
        a_seq = self.model.skill_argument_decoder(z, h, w, arg)
        return a_seq, arg, log_pi

    def _actor_u_for_q(
        self,
        z: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if self.actor_mode == 'skill_decoder':
            if h is None:
                raise ValueError("h is required when actor_mode=skill_decoder")
            a_seq = self._skilldec_action_seq(z, h, anchor=False)
            a0 = a_seq[:, 0].clamp(-1.0, 1.0)
            return self.model.action_encoder(a0)
        if self.actor_mode == 'skill_arg':
            if h is None:
                raise ValueError("h is required when actor_mode=skill_arg")
            a_seq, _, _ = self._skillarg_action_seq(
                z, h, anchor=False, deterministic=deterministic)
            a0 = a_seq[:, 0].clamp(-1.0, 1.0)
            return self.model.action_encoder(a0)
        if deterministic:
            return self._policy_mean_action(z)
        u, _, _, _ = self.model.policy_prior(z)
        return u

    def _anchor_weight(self) -> float:
        lam0 = float(self.cfg.lambda_skilldec_anchor)
        lam_min = float(self.cfg.skilldec_anchor_min)
        decay = max(1, int(self.cfg.skilldec_anchor_decay_steps))
        frac = min(1.0, max(0.0, self.step / decay))
        return lam_min + (lam0 - lam_min) * (1.0 - frac)

    @staticmethod
    def _concat_batches(batches: List[Dict[str, torch.Tensor]]
                        ) -> Dict[str, torch.Tensor]:
        if len(batches) == 1:
            return batches[0]
        keys = batches[0].keys()
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

    def _sample_with_elite(
        self,
        batch_size: int,
        positive_fraction: float,
        offline_fraction: float,
        elite_fraction: float,
    ) -> Dict[str, torch.Tensor]:
        if (self.elite_buf is None or self.elite_buf.size <= 0
                or elite_fraction <= 0.0):
            return self.buf.sample(
                batch_size,
                positive_fraction=positive_fraction,
                offline_fraction=offline_fraction,
            )

        n_elite = int(round(
            batch_size * max(0.0, min(1.0, elite_fraction))))
        n_elite = min(batch_size, n_elite)
        n_main = batch_size - n_elite
        batches = []
        if n_main > 0:
            batches.append(self.buf.sample(
                n_main,
                positive_fraction=positive_fraction,
                offline_fraction=offline_fraction,
            ))
        if n_elite > 0:
            batches.append(self.elite_buf.sample(n_elite))
        return self._concat_batches(batches)

    def add_elite_episode(
        self,
        transitions: List[Dict[str, np.ndarray]],
        ep_return: float,
    ) -> int:
        if (self.elite_buf is None or not transitions
                or ep_return < self.cfg.elite_reward_threshold):
            return 0
        self.elite_return_last = float(ep_return)
        self.elite_episode_count += 1
        if self.elite_episode_count == 1:
            self.elite_return_ema = float(ep_return)
        else:
            self.elite_return_ema = (
                0.95 * self.elite_return_ema + 0.05 * float(ep_return)
            )
        H = int(self.elite_buf.chunk_horizon)
        n_step = max(1, int(self.cfg.q_n_step))
        gam = float(self.cfg.q_n_step_gamma)
        actions = [
            None if tr.get('a', None) is None
            else np.asarray(tr['a'], dtype=np.float32)
            for tr in transitions
        ]
        for i, tr in enumerate(transitions):
            elite_tr = dict(tr)
            elite_tr['source'] = 2.0
            if tr.get('a_chunk', None) is not None:
                elite_tr['a_chunk'] = np.asarray(
                    tr['a_chunk'], dtype=np.float32)
                if tr.get('chunk_mask', None) is not None:
                    elite_tr['chunk_mask'] = np.asarray(
                        tr['chunk_mask'], dtype=np.float32)
            elif actions and actions[i] is not None:
                a_chunk = np.zeros((H, self.elite_buf.action_dim), dtype=np.float32)
                chunk_mask = np.zeros(H, dtype=np.float32)
                n_chunk = min(H, len(actions) - i)
                valid = [
                    a for a in actions[i:i + n_chunk]
                    if a is not None
                ]
                if valid:
                    a_chunk[:len(valid)] = np.stack(valid, axis=0)
                    chunk_mask[:len(valid)] = 1.0
                elite_tr['a_chunk'] = a_chunk
                elite_tr['chunk_mask'] = chunk_mask

            if 'r_return' in tr and 'z_boot' in tr:
                elite_tr['r_return'] = float(tr.get('r_return', tr.get('r', 0.0)))
                elite_tr['z_boot'] = tr.get('z_boot', tr['z_next'])
                elite_tr['h_boot'] = tr.get('h_boot', tr.get('h_next', None))
                elite_tr['discount'] = float(tr.get('discount', 0.0))
            else:
                ret = 0.0
                boot = tr
                done_n = False
                n_used = 0
                for j in range(n_step):
                    k = i + j
                    if k >= len(transitions):
                        break
                    boot = transitions[k]
                    ret += (gam ** j) * float(boot.get('r', 0.0))
                    n_used = j + 1
                    if float(boot.get('done', 0.0)) > 0.5:
                        done_n = True
                        break
                elite_tr['r_return'] = float(ret)
                elite_tr['z_boot'] = boot.get('z_next', tr['z_next'])
                elite_tr['h_boot'] = boot.get('h_next', tr.get('h_next', None))
                elite_tr['discount'] = float(0.0 if done_n else gam ** max(1, n_used))
            self.elite_buf.add(**elite_tr)
        return len(transitions)

    @property
    def elite_size(self) -> int:
        return 0 if self.elite_buf is None else self.elite_buf.size

    def _sample_update_batches(self) -> Tuple[Dict[str, torch.Tensor],
                                              Dict[str, torch.Tensor],
                                              Dict[str, torch.Tensor]]:
        bs = self.cfg.batch_size
        b_r = self._sample_with_elite(
            bs,
            positive_fraction=self.cfg.reward_positive_fraction,
            offline_fraction=self.cfg.reward_offline_fraction,
            elite_fraction=self.cfg.reward_elite_fraction,
        )
        b_q = self._sample_with_elite(
            bs,
            positive_fraction=self.cfg.q_positive_fraction,
            offline_fraction=self.cfg.q_offline_fraction,
            elite_fraction=self.cfg.q_elite_fraction,
        )
        b_pi = self._sample_with_elite(
            bs,
            positive_fraction=self.cfg.pi_positive_fraction,
            offline_fraction=self.cfg.pi_offline_fraction,
            elite_fraction=self.cfg.pi_elite_fraction,
        )
        return b_r, b_q, b_pi

    def _plain_r_hat(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.model.reward_ensemble_head.member_probs(z, u).mean(0).clamp(0.0, 1.0)

    def _q_reward(self, z: torch.Tensor, u: torch.Tensor,
                  r_env: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.cfg.q_reward_source.lower()
        r_hat = self._plain_r_hat(z, u)
        if source == 'env':
            return r_env.clamp(0.0, 1.0), r_hat
        if source == 'rhat':
            return r_hat, r_hat
        if source == 'penalized':
            return self.model.reward_ensemble_head.penalized_reward(z, u), r_hat
        raise ValueError(
            f"Unknown q_reward_source={self.cfg.q_reward_source!r}; "
            "expected env, rhat, or penalized"
        )

    @torch.no_grad()
    def _wm_prediction_error_tensor(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        u: torch.Tensor,
        x_next: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        w = torch.softmax(m.skill_prior(h), dim=-1)
        z_pred, _, _ = m.koopman(z, u, w)
        recon = m.decoder(z_pred)
        q_pred = symexp(recon['q'])
        p_pred = symexp(recon['delta_p'])
        q_true = x_next[..., X_DQ_START:X_DQ_END]
        p_true = x_next[..., X_DP_START:X_DP_END]
        q_mse = (q_pred - q_true).pow(2).mean(-1)
        p_mse = (p_pred - p_true).pow(2).mean(-1)
        err = (
            float(self.cfg.q_wm_penalty_q_weight) * q_mse
            + float(self.cfg.q_wm_penalty_obj_weight) * p_mse
        ).clamp_min(0.0).sqrt()
        return err

    @torch.no_grad()
    def wm_prediction_error_np(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        u_np: np.ndarray,
        x_next_np: np.ndarray,
    ) -> float:
        dev = torch.device(self.device)
        u = torch.as_tensor(u_np, dtype=torch.float32, device=dev).view(1, -1)
        x_next = torch.as_tensor(
            x_next_np, dtype=torch.float32, device=dev).view(1, -1)
        return float(self._wm_prediction_error_tensor(z, h, u, x_next)[0].item())

    def _elite_wm_penalty_threshold(self) -> Optional[float]:
        if self.cfg.lambda_q_wm_penalty <= 0.0 or self.elite_buf is None:
            return None
        n = int(self.elite_buf.size)
        if n < int(self.cfg.q_wm_penalty_min_elite):
            return None
        err = self.elite_buf._d.get('wm_err', None)
        if err is None:
            return None
        vals = err[:n]
        vals = vals[np.isfinite(vals)]
        if vals.size < int(self.cfg.q_wm_penalty_min_elite):
            return None
        pct = max(0.0, min(100.0, float(self.cfg.q_wm_penalty_percentile)))
        thr = float(np.percentile(vals, pct))
        self.wm_penalty_threshold = thr
        return thr

    def _q_wm_penalty(
        self,
        batch: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.lambda_q_wm_penalty <= 0.0:
            return like.new_zeros(like.shape)
        thr = self._elite_wm_penalty_threshold()
        if thr is None or 'wm_err' not in batch:
            return like.new_zeros(like.shape)
        wm_err = batch['wm_err'].to(like.device)
        excess = (wm_err - float(thr)).clamp_min(0.0)
        if self.cfg.q_wm_penalty_clip > 0.0:
            excess = excess.clamp_max(float(self.cfg.q_wm_penalty_clip))
        return float(self.cfg.lambda_q_wm_penalty) * excess

    def _skillarg_q_logits(
        self,
        q_head: nn.Module,
        z: torch.Tensor,
        h: torch.Tensor,
        arg: torch.Tensor,
        skill_id: torch.Tensor,
    ) -> torch.Tensor:
        return q_head.gather_skill_logits(z, h, arg, skill_id)

    def _skillarg_q_value(
        self,
        q_head: nn.Module,
        z: torch.Tensor,
        h: torch.Tensor,
        arg: torch.Tensor,
        skill_id: torch.Tensor,
        return_type: str = 'min',
    ) -> torch.Tensor:
        return q_head.expected_value(
            z, h, arg, skill_id=skill_id, return_type=return_type)

    def _update_q_guide_gate(self, loss_q: torch.Tensor) -> float:
        q_loss = float(loss_q.detach().item())
        beta = max(0.0, min(0.9999, float(self.cfg.pi_q_guide_ema_beta)))
        if self.q_loss_ema is None:
            self.q_loss_ema = q_loss
        else:
            self.q_loss_ema = beta * self.q_loss_ema + (1.0 - beta) * q_loss
        threshold = float(self.cfg.pi_q_guide_loss_threshold)
        if threshold <= 0.0:
            self.q_guide_active = True
        elif (self.step >= int(self.cfg.pi_q_guide_min_step)
              and self.q_loss_ema <= threshold):
            self.q_guide_active = True
        return self.q_loss_ema

    def _effective_pi_q_weight(self) -> float:
        return float(self.cfg.pi_q_weight) if self.q_guide_active else 0.0

    def _expand_skill_inputs(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        K = self.model.cfg.num_skills
        skill_ids = torch.arange(K, device=z.device).view(1, K).expand(B, K)
        skill_oh = F.one_hot(skill_ids, num_classes=K).float()
        z_rep = z[:, None].expand(B, K, z.shape[-1]).reshape(B * K, -1)
        h_rep = h[:, None].expand(B, K, h.shape[-1]).reshape(B * K, -1)
        skill_flat = skill_oh.reshape(B * K, K)
        skill_id_flat = skill_ids.reshape(B * K)
        return z_rep, h_rep, skill_flat, skill_id_flat

    def _skillarg_soft_value(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        q_head: nn.Module,
        detach_actor: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        m = self.model
        B = z.shape[0]
        K = m.cfg.num_skills
        prior_logits = self._skill_prior_logits(h)
        logits_d = self._skill_policy_logits(z, h)
        probs_d = torch.softmax(logits_d, dim=-1)
        log_pi_d = F.log_softmax(logits_d, dim=-1)
        log_p_d = F.log_softmax(prior_logits, dim=-1)
        kl_d = (probs_d * (log_pi_d - log_p_d)).sum(-1)

        z_rep, h_rep, skill_flat, skill_id_flat = self._expand_skill_inputs(z, h)
        if detach_actor:
            with torch.no_grad():
                arg, _, mean_arg, log_std = m.skill_argument_policy(
                    z_rep, h_rep, skill_flat)
                mean_raw, log_std_raw = m.skill_argument_policy.dist_params(
                    z_rep, h_rep, skill_flat)
                prior_mean, prior_log_std = self.skill_arg_policy_anchor.dist_params(
                    z_rep, h_rep, skill_flat)
        else:
            arg, _, mean_arg, log_std = m.skill_argument_policy(
                z_rep, h_rep, skill_flat)
            mean_raw, log_std_raw = m.skill_argument_policy.dist_params(
                z_rep, h_rep, skill_flat)
            with torch.no_grad():
                prior_mean, prior_log_std = self.skill_arg_policy_anchor.dist_params(
                    z_rep, h_rep, skill_flat)

        kl_z = self._skill_arg_normal_kl(
            mean_raw, log_std_raw, prior_mean, prior_log_std).reshape(B, K)
        q = self._skillarg_q_value(
            q_head, z_rep, h_rep, arg, skill_id_flat,
            return_type='min').reshape(B, K)
        v = (
            probs_d * (q - self.cfg.skill_arg_alpha_z * kl_z)
        ).sum(-1) - self.cfg.skill_arg_alpha_d * kl_d
        return v, {
            'q_by_skill': q,
            'kl_d': kl_d,
            'kl_z': kl_z,
            'probs_d': probs_d,
        }

    def _skillarg_actor_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        q_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        m = self.model
        B = z.shape[0]
        K = m.cfg.num_skills
        prior_logits = self._skill_prior_logits(h)
        logits_d = self._skill_policy_logits(z, h)
        probs_d = torch.softmax(logits_d, dim=-1)
        log_pi_d = F.log_softmax(logits_d, dim=-1)
        log_p_d = F.log_softmax(prior_logits, dim=-1)
        kl_d = (probs_d * (log_pi_d - log_p_d)).sum(-1)

        z_rep, h_rep, skill_flat, skill_id_flat = self._expand_skill_inputs(z, h)
        arg, _, _, _ = m.skill_argument_policy(z_rep, h_rep, skill_flat)
        mean_raw, log_std_raw = m.skill_argument_policy.dist_params(
            z_rep, h_rep, skill_flat)
        with torch.no_grad():
            prior_mean, prior_log_std = self.skill_arg_policy_anchor.dist_params(
                z_rep, h_rep, skill_flat)
        kl_z = self._skill_arg_normal_kl(
            mean_raw, log_std_raw, prior_mean, prior_log_std).reshape(B, K)
        q = self._skillarg_q_value(
            m._detach_skill_arg_q_head, z_rep, h_rep, arg, skill_id_flat,
            return_type='min').reshape(B, K)
        m.scale_tracker.update(q.detach().reshape(-1))
        q_w = self.cfg.pi_q_weight if q_weight is None else float(q_weight)
        objective = (
            probs_d
            * (q_w * q * m.scale_tracker.rho
               - self.cfg.skill_arg_alpha_z * kl_z)
        ).sum(-1) - self.cfg.skill_arg_alpha_d * kl_d
        loss = -objective.mean()

        loss_elite_bc = z.new_tensor(0.0)
        loss_elite_skill_bc = z.new_tensor(0.0)
        loss_elite_arg_bc = z.new_tensor(0.0)
        skill_id_bc = batch.get('skill_id', None)
        skill_prob_bc = batch.get('skill_prob', None)
        elite_mask = batch.get('source', z.new_zeros(z.shape[0])).to(z.device) > 1.5
        if skill_id_bc is not None and bool(elite_mask.any()):
            skill_id_bc = skill_id_bc.long()
            if skill_prob_bc is not None:
                skill_prob_bc = skill_prob_bc.to(z.device)
                has_soft_skill = skill_prob_bc.sum(-1) > 0.5
            else:
                has_soft_skill = torch.zeros_like(elite_mask)
            if self.cfg.lambda_elite_skill_bc_pi > 0.0:
                log_pi_bc = F.log_softmax(logits_d, dim=-1)
                soft_mask = elite_mask & has_soft_skill
                hard_mask = elite_mask & (~has_soft_skill)
                parts = []
                if bool(soft_mask.any()):
                    target = skill_prob_bc[soft_mask]
                    target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
                    parts.append(-(target * log_pi_bc[soft_mask]).sum(-1).mean())
                if bool(hard_mask.any()):
                    parts.append(F.cross_entropy(
                        logits_d[hard_mask], skill_id_bc[hard_mask]))
                loss_elite_skill_bc = (
                    torch.stack(parts).mean()
                    if parts else z.new_tensor(0.0)
                )
                loss = (
                    loss
                    + self.cfg.lambda_elite_skill_bc_pi * loss_elite_skill_bc
                )
            if (self.cfg.lambda_elite_arg_bc_pi > 0.0
                    or self.cfg.lambda_elite_bc_pi > 0.0):
                skill_oh = self._one_hot_skill(skill_id_bc.long())
                if skill_prob_bc is not None:
                    soft_mask = has_soft_skill.view(-1, 1)
                    skill_oh = torch.where(soft_mask, skill_prob_bc, skill_oh)
                arg_bc = m.skill_argument_policy.mean_arg(z, h, skill_oh)
            if self.cfg.lambda_elite_arg_bc_pi > 0.0:
                arg_target = batch.get('skill_arg', None)
                if arg_target is not None:
                    arg_mask = elite_mask
                    has_arg = batch.get('has_skill_arg', None)
                    if has_arg is not None:
                        arg_mask = arg_mask & (has_arg.to(z.device) > 0.5)
                    if bool(arg_mask.any()):
                        arg_target = arg_target.to(z.device).clamp(-1.0, 1.0)
                        loss_elite_arg_bc = (
                            arg_bc[arg_mask] - arg_target[arg_mask]
                        ).pow(2).mean()
                        loss = (
                            loss
                            + self.cfg.lambda_elite_arg_bc_pi * loss_elite_arg_bc
                        )
            if self.cfg.lambda_elite_bc_pi > 0.0:
                a_seq = m.skill_argument_decoder(z, h, skill_oh, arg_bc)
                loss_elite_bc = self._elite_chunk_bc_loss(a_seq, batch)
                loss = loss + self.cfg.lambda_elite_bc_pi * loss_elite_bc

        return loss, {
            'q_pi': (probs_d * q).sum(-1).detach(),
            'kl_d': kl_d.detach(),
            'kl_z': (probs_d * kl_z).sum(-1).detach(),
            'loss_elite_bc': loss_elite_bc.detach(),
            'loss_elite_skill_bc': loss_elite_skill_bc.detach(),
            'loss_elite_arg_bc': loss_elite_arg_bc.detach(),
        }

    def _combo_skillarg_q_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        z = batch['z']
        zero = z.new_tensor(0.0)
        if (self.cfg.lambda_q_cql_policy <= 0.0
                and self.cfg.lambda_q_rank_data <= 0.0):
            return {
                'loss_q_cql_policy': zero,
                'loss_q_rank_data': zero,
                'loss_q_rank_bc': zero,
                'loss_q_rank_lqr': zero,
                'q_policy': zero,
                'q_data': zero,
            }
        h = batch['h']
        skill_id = batch['skill_id'].long()
        arg_data = batch['skill_arg'].detach()
        q_data = self._skillarg_q_value(
            self.model.skill_arg_q_head, z, h, arg_data, skill_id,
            return_type='min')
        v_pi, _ = self._skillarg_soft_value(
            z, h, self.model.skill_arg_q_head, detach_actor=True)
        q_policy = v_pi
        loss_cql = (q_policy - q_data).mean()
        loss_rank_data = F.relu(
            self.cfg.q_rank_margin + q_policy - q_data).mean()
        return {
            'loss_q_cql_policy': loss_cql,
            'loss_q_rank_data': loss_rank_data,
            'loss_q_rank_bc': zero,
            'loss_q_rank_lqr': zero,
            'q_policy': q_policy.detach(),
            'q_data': q_data.detach(),
        }

    def _update_skill_arg_q_pi(
        self,
        b_q: Dict[str, torch.Tensor],
        b_pi: Dict[str, torch.Tensor],
        loss_r: torch.Tensor,
    ) -> Dict[str, float]:
        m = self.model
        z, h = b_q['z'], b_q['h']
        arg = b_q['skill_arg']
        skill_id = b_q['skill_id'].long()
        r, z_next, done = b_q['r'], b_q['z_next'], b_q['done']
        h_next_q = b_q.get('h_next', None)

        with torch.no_grad():
            r_q, r_hat = self._q_reward(z, b_q['u'], r)
            source_q = b_q.get('source', z.new_zeros(z.shape[0])).to(z.device)
            ret_q = b_q.get('r_return', r).to(z.device)
            success_mask = ret_q > self.cfg.q_success_threshold
            if self.cfg.q_success_include_source:
                success_mask = success_mask | (source_q > 1.5)
            if self.cfg.q_success_only:
                r_targ = torch.where(
                    success_mask,
                    z.new_full((z.shape[0],), float(self.cfg.q_success_value)),
                    z.new_zeros(z.shape[0]),
                )
                z_boot = b_q.get('z_boot', z_next)
                h_boot = b_q.get('h_boot', h_next_q)
                discount = z.new_zeros(z.shape[0])
            elif self.cfg.q_reward_source.lower() == 'env':
                r_targ = b_q.get('r_return', r).clamp(0.0, 5.0)
                z_boot = b_q.get('z_boot', z_next)
                h_boot = b_q.get('h_boot', h_next_q)
                discount = b_q.get(
                    'discount',
                    (1 - done) * float(self.cfg.q_n_step_gamma),
                )
            else:
                r_targ = r_q
                z_boot = z_next
                h_boot = h_next_q
                discount = (1 - done) * float(self.cfg.q_n_step_gamma)
            wm_penalty = self._q_wm_penalty(b_q, r_targ)
            r_targ_pen = (r_targ - wm_penalty).clamp(
                m.cfg.v_min, m.cfg.v_max)
            q_next, next_info = self._skillarg_soft_value(
                z_boot, h_boot, m.skill_arg_q_head_target, detach_actor=True)
            y = (
                r_targ_pen + self.cfg.q_bootstrap * discount * q_next
            ).clamp(m.cfg.v_min, m.cfg.v_max)

        q_logits_d = self._skillarg_q_logits(
            m.skill_arg_q_head, z, h, arg, skill_id)
        q_logits = q_logits_d.permute(1, 0, 2).unsqueeze(1)
        loss_q_td = q_categorical_loss(
            q_logits=q_logits,
            reward_seq=torch.zeros_like(y).unsqueeze(1),
            q_target_scalar=y.unsqueeze(1),
            bins=m.skill_arg_q_head.bins,
            gamma=1.0,
        )
        q_vals_d = two_hot_decode(q_logits_d, m.skill_arg_q_head.bins).min(0).values
        source_f = b_q.get('source', z.new_zeros(z.shape[0])).to(z.device)
        ret_f = b_q.get('r_return', z.new_zeros(z.shape[0])).to(z.device)
        success_mask_f = ret_f > self.cfg.q_success_threshold
        if self.cfg.q_success_include_source:
            success_mask_f = success_mask_f | (source_f > 1.5)
        neg_mask = ~success_mask_f
        loss_q_negative = z.new_tensor(0.0)
        if self.cfg.lambda_q_negative > 0.0 and bool(neg_mask.any()):
            neg_logits = q_logits_d[:, neg_mask]
            neg_logits = neg_logits.permute(1, 0, 2).unsqueeze(1)
            neg_y = z.new_zeros(int(neg_mask.sum()))
            loss_q_negative = q_categorical_loss(
                q_logits=neg_logits,
                reward_seq=torch.zeros_like(neg_y).unsqueeze(1),
                q_target_scalar=neg_y.unsqueeze(1),
                bins=m.skill_arg_q_head.bins,
                gamma=1.0,
            )
        loss_q_success_rank = z.new_tensor(0.0)
        if (self.cfg.lambda_q_success_rank > 0.0
                and bool(success_mask_f.any())
                and bool(neg_mask.any())):
            q_succ = q_vals_d[success_mask_f]
            q_neg = q_vals_d[neg_mask]
            loss_q_success_rank = F.relu(
                self.cfg.q_success_margin + q_neg - q_succ.mean()
            ).mean()
        combo = self._combo_skillarg_q_loss(b_q)
        loss_q = (
            loss_q_td
            + self.cfg.lambda_q_cql_policy * combo['loss_q_cql_policy']
            + self.cfg.lambda_q_rank_data * combo['loss_q_rank_data']
            + self.cfg.lambda_q_negative * loss_q_negative
            + self.cfg.lambda_q_success_rank * loss_q_success_rank
        )
        self.opt_q.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(m.skill_arg_q_head.parameters(),
                                 self.cfg.grad_clip)
        self.opt_q.step()
        m.soft_update_target_Q()
        q_loss_ema = self._update_q_guide_gate(loss_q)
        pi_q_weight_eff = self._effective_pi_q_weight()

        pi_active = float(self.step >= self.cfg.pi_update_start)
        if pi_active:
            loss_pi, pi_info = self._skillarg_actor_loss(
                b_pi['z'], b_pi['h'], b_pi,
                q_weight=pi_q_weight_eff)
            self.opt_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(
                self.opt_pi.param_groups[0]['params'], self.cfg.grad_clip)
            self.opt_pi.step()
        else:
            loss_pi = z.new_tensor(0.0)
            pi_info = {
                'q_pi': z.new_zeros(z.shape[0]),
                'kl_d': z.new_zeros(z.shape[0]),
                'kl_z': z.new_zeros(z.shape[0]),
                'loss_elite_bc': z.new_tensor(0.0),
                'loss_elite_skill_bc': z.new_tensor(0.0),
                'loss_elite_arg_bc': z.new_tensor(0.0),
            }

        q_data = combo['q_data']
        q_policy = combo['q_policy']
        return {
            'loss_reward': loss_r.item(),
            'loss_q': loss_q.item(),
            'loss_q_td': loss_q_td.item(),
            'loss_q_cql_policy': combo['loss_q_cql_policy'].item(),
            'loss_q_rank_data': combo['loss_q_rank_data'].item(),
            'loss_q_rank_bc': loss_q_negative.item(),
            'loss_q_rank_lqr': loss_q_success_rank.item(),
            'loss_pi': loss_pi.item(),
            'loss_lqr_pi': 0.0,
            'loss_bc_pi': 0.0,
            'loss_elite_bc_pi': pi_info['loss_elite_bc'].item(),
            'loss_elite_skill_bc_pi': pi_info['loss_elite_skill_bc'].item(),
            'loss_elite_arg_bc_pi': pi_info['loss_elite_arg_bc'].item(),
            'loss_skilldec_anchor': (
                self.cfg.skill_arg_alpha_d * pi_info['kl_d'].mean()
                + self.cfg.skill_arg_alpha_z * pi_info['kl_z'].mean()
            ).item(),
            'skilldec_anchor_w': 0.0,
            'q_mean': pi_info['q_pi'].detach().mean().item(),
            'q_policy_cql': q_policy.detach().mean().item(),
            'q_data': q_data.detach().mean().item(),
            'rho': m.scale_tracker.rho,
            'r_hat': r_hat.detach().mean().item(),
            'q_reward': r_targ.detach().mean().item(),
            'q_wm_penalty': wm_penalty.detach().mean().item(),
            'q_wm_error': b_q.get('wm_err', z.new_zeros(z.shape[0])).mean().item(),
            'q_wm_threshold': float(self.wm_penalty_threshold),
            'q_target': y.detach().mean().item(),
            'q_next': q_next.detach().mean().item(),
            'pi_active': pi_active,
            'pi_q_weight_eff': pi_q_weight_eff,
            'q_loss_ema': q_loss_ema,
            'q_guide_active': float(self.q_guide_active),
            'elite_size': float(self.elite_size),
        }

    def update(self) -> Dict[str, float]:
        if self.buf.size < self.cfg.batch_size:
            return {}
        b_r, b_q, b_pi = self._sample_update_batches()
        z_r, u_r, r_r = b_r['z'], b_r['u'], b_r['r']
        z, u = b_q['z'], b_q['u']
        h_q, h_next_q = b_q.get('h', None), b_q.get('h_next', None)
        r, z_next, done = b_q['r'], b_q['z_next'], b_q['done']
        m = self.model

        loss_r = m.reward_ensemble_head.ensemble_loss(
            z_r, u_r, r_r.clamp(0.0, 1.0))
        self.opt_reward.zero_grad()
        loss_r.backward()
        nn.utils.clip_grad_norm_(m.reward_ensemble_head.parameters(),
                                 self.cfg.grad_clip)
        self.opt_reward.step()

        if self.actor_mode == 'skill_arg':
            return self._update_skill_arg_q_pi(b_q, b_pi, loss_r)

        with torch.no_grad():
            r_q, r_hat = self._q_reward(z, u, r)
            if self.cfg.q_reward_source.lower() == 'env':
                r_targ = b_q.get('r_return', r).clamp(0.0, 5.0)
                z_boot = b_q.get('z_boot', z_next)
                h_boot = b_q.get('h_boot', h_next_q)
                discount = b_q.get(
                    'discount',
                    (1 - done) * float(self.cfg.q_n_step_gamma),
                )
            else:
                r_targ = r_q
                z_boot = z_next
                h_boot = h_next_q
                discount = (1 - done) * float(self.cfg.q_n_step_gamma)
            wm_penalty = self._q_wm_penalty(b_q, r_targ)
            r_targ_pen = (r_targ - wm_penalty).clamp(
                m.cfg.v_min, m.cfg.v_max)
            u_next = self._actor_u_for_q(z_boot, h_boot)
            q_next = m.q_head_target.expected_value(
                z_boot, u_next, return_type='min')
            y = (
                r_targ_pen
                + self.cfg.q_bootstrap * discount * q_next
            ).clamp(m.cfg.v_min, m.cfg.v_max)

        q_logits = m.q_head(z, u).permute(1, 0, 2).unsqueeze(1)
        loss_q_td = q_categorical_loss(
            q_logits=q_logits,
            reward_seq=torch.zeros_like(y).unsqueeze(1),
            q_target_scalar=y.unsqueeze(1),
            bins=m.q_head.bins,
            gamma=1.0,
        )
        combo = self._combo_q_loss(z, b_q)
        loss_q = (
            loss_q_td
            + self.cfg.lambda_q_cql_policy * combo['loss_q_cql_policy']
            + self.cfg.lambda_q_rank_data * combo['loss_q_rank_data']
            + self.cfg.lambda_q_rank_bc * combo['loss_q_rank_bc']
            + self.cfg.lambda_q_rank_lqr * combo['loss_q_rank_lqr']
        )
        self.opt_q.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(m.q_head.parameters(), self.cfg.grad_clip)
        self.opt_q.step()
        m.soft_update_target_Q()

        z_pi = b_pi['z']
        h_pi = b_pi.get('h', None)
        if self.actor_mode == 'skill_decoder':
            a_pi_seq = self._skilldec_action_seq(z_pi, h_pi, anchor=False)
            u_pi = m.action_encoder(a_pi_seq[:, 0].clamp(-1.0, 1.0))
            log_pi = None
            u_mean = u_pi
            with torch.no_grad():
                a_anchor_seq = self._skilldec_action_seq(z_pi, h_pi, anchor=True)
            loss_anchor = (a_pi_seq - a_anchor_seq).pow(2).mean()
            loss_elite_bc = self._elite_chunk_bc_loss(a_pi_seq, b_pi)
            anchor_w = self._anchor_weight()
        elif self.actor_mode == 'skill_arg':
            a_pi_seq, arg_pi, log_pi = self._skillarg_action_seq(
                z_pi, h_pi, anchor=False, deterministic=False)
            u_pi = m.action_encoder(a_pi_seq[:, 0].clamp(-1.0, 1.0))
            u_mean = u_pi
            with torch.no_grad():
                a_anchor_seq, arg_anchor, _ = self._skillarg_action_seq(
                    z_pi, h_pi, anchor=True, deterministic=True)
            loss_anchor = (
                (a_pi_seq - a_anchor_seq).pow(2).mean()
                + 0.1 * (arg_pi - arg_anchor).pow(2).mean()
            )
            loss_elite_bc = self._elite_chunk_bc_loss(a_pi_seq, b_pi)
            anchor_w = self._anchor_weight()
        else:
            u_pi, log_pi, mu_pi, _ = m.policy_prior(z_pi)
            u_mean = torch.tanh(mu_pi)
            loss_anchor = z_pi.new_tensor(0.0)
            loss_elite_bc = z_pi.new_tensor(0.0)
            anchor_w = 0.0
        q_logits_pi = m._detach_q_head(z_pi, u_pi)
        q_vals_pi = two_hot_decode(q_logits_pi, m.q_head.bins)
        n_pick = min(2, m.cfg.num_q)
        idx = torch.randperm(m.cfg.num_q, device=z_pi.device)[:n_pick]
        q_pi = q_vals_pi[idx].min(0).values
        m.scale_tracker.update(q_pi.detach())
        loss_lqr = self._lqr_reg_loss(u_mean, b_pi)
        loss_bc = self._bc_reg_loss(u_mean, b_pi)
        pi_active = float(self.step >= self.cfg.pi_update_start)
        if pi_active:
            if self.actor_mode in ('skill_decoder', 'skill_arg'):
                loss_pi = (
                    -self.cfg.pi_q_weight * q_pi * m.scale_tracker.rho
                ).mean() + anchor_w * loss_anchor
                loss_pi = (
                    loss_pi
                    + self.cfg.lambda_elite_bc_pi * loss_elite_bc
                )
                if log_pi is not None:
                    loss_pi = loss_pi + m.cfg.entropy_coef * log_pi.mean()
            else:
                log_pi_eff = log_pi
                if log_pi_eff.dim() > q_pi.dim():
                    log_pi_eff = log_pi_eff.sum(-1)
                loss_pi = (
                    (m.cfg.entropy_coef * log_pi_eff
                     - self.cfg.pi_q_weight * q_pi)
                    * m.scale_tracker.rho
                ).mean()
                loss_pi = (
                    loss_pi
                    + self.cfg.lambda_lqr_pi * loss_lqr
                    + self.cfg.lambda_bc_pi * loss_bc
                )
            self.opt_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(
                self.opt_pi.param_groups[0]['params'], self.cfg.grad_clip)
            self.opt_pi.step()
        else:
            loss_pi = z_pi.new_tensor(0.0)

        return {
            'loss_reward': loss_r.item(),
            'loss_q': loss_q.item(),
            'loss_q_td': loss_q_td.item(),
            'loss_q_cql_policy': combo['loss_q_cql_policy'].item(),
            'loss_q_rank_data': combo['loss_q_rank_data'].item(),
            'loss_q_rank_bc': combo['loss_q_rank_bc'].item(),
            'loss_q_rank_lqr': combo['loss_q_rank_lqr'].item(),
            'loss_pi': loss_pi.item(),
            'loss_lqr_pi': loss_lqr.item(),
            'loss_bc_pi': loss_bc.item(),
            'loss_elite_bc_pi': loss_elite_bc.item(),
            'loss_elite_skill_bc_pi': 0.0,
            'loss_elite_arg_bc_pi': 0.0,
            'loss_skilldec_anchor': loss_anchor.item(),
            'skilldec_anchor_w': anchor_w,
            'q_mean': q_pi.detach().mean().item(),
            'q_policy_cql': combo['q_policy'].detach().mean().item(),
            'q_data': combo['q_data'].detach().mean().item(),
            'rho': m.scale_tracker.rho,
            'r_hat': r_hat.detach().mean().item(),
            'q_reward': r_targ.detach().mean().item(),
            'q_wm_penalty': wm_penalty.detach().mean().item(),
            'q_wm_error': b_q.get('wm_err', z.new_zeros(z.shape[0])).mean().item(),
            'q_wm_threshold': float(self.wm_penalty_threshold),
            'q_target': y.detach().mean().item(),
            'q_next': q_next.detach().mean().item(),
            'pi_active': pi_active,
            'elite_size': float(self.elite_size),
        }

    def _q_min_value(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q_logits = self.model.q_head(z, u)
        q_vals = two_hot_decode(q_logits, self.model.q_head.bins)
        return q_vals.min(0).values

    def _policy_mean_action(self, z: torch.Tensor) -> torch.Tensor:
        out = self.model.policy_prior.net(z)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)

    def _combo_q_loss(self, z: torch.Tensor,
                      batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zero = z.new_tensor(0.0)
        if (self.cfg.lambda_q_cql_policy <= 0.0
                and self.cfg.lambda_q_rank_data <= 0.0
                and self.cfg.lambda_q_rank_bc <= 0.0
                and self.cfg.lambda_q_rank_lqr <= 0.0):
            return {
                'loss_q_cql_policy': zero,
                'loss_q_rank_data': zero,
                'loss_q_rank_bc': zero,
                'loss_q_rank_lqr': zero,
                'q_policy': zero,
                'q_data': zero,
            }

        u_pi = self._actor_u_for_q(z, batch.get('h', None), deterministic=True).detach()
        q_policy = self._q_min_value(z, u_pi)
        q_data = self._q_min_value(z, batch['u'].detach())
        loss_cql = (q_policy - q_data).mean()

        loss_rank_data = zero
        if self.cfg.lambda_q_rank_data > 0.0:
            loss_rank_data = F.relu(
                self.cfg.q_rank_margin + q_policy - q_data).mean()

        loss_rank_bc = zero
        if self.cfg.lambda_q_rank_bc > 0.0 and 'source' in batch and 'u' in batch:
            mask_bc = batch['source'] > 0.5
            if mask_bc.numel() > 0 and bool(mask_bc.any()):
                rank_bc = F.relu(
                    self.cfg.q_rank_margin + q_policy - q_data)
                loss_rank_bc = rank_bc[mask_bc].mean()

        loss_rank_lqr = zero
        if (self.cfg.lambda_q_rank_lqr > 0.0
                and 'has_lqr' in batch and 'u_lqr' in batch):
            mask_lqr = batch['has_lqr'] > 0.5
            if mask_lqr.numel() > 0 and bool(mask_lqr.any()):
                u_lqr = batch['u_lqr'].detach()
                q_lqr = self._q_min_value(z, u_lqr)
                rank_lqr = F.relu(
                    self.cfg.q_rank_margin + q_policy - q_lqr)
                loss_rank_lqr = rank_lqr[mask_lqr].mean()

        return {
            'loss_q_cql_policy': loss_cql,
            'loss_q_rank_data': loss_rank_data,
            'loss_q_rank_bc': loss_rank_bc,
            'loss_q_rank_lqr': loss_rank_lqr,
            'q_policy': q_policy.detach(),
            'q_data': q_data.detach(),
        }

    def _bc_reg_loss(self, u_mean: torch.Tensor,
                     batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.lambda_bc_pi <= 0.0 or 'source' not in batch or 'u' not in batch:
            return u_mean.new_tensor(0.0)
        mask = batch['source'] > 0.5
        if mask.numel() == 0 or not bool(mask.any()):
            return u_mean.new_tensor(0.0)
        target = batch['u'].to(u_mean.device)
        if self.cfg.bc_target_clip > 0.0:
            target = target.clamp(-self.cfg.bc_target_clip,
                                  self.cfg.bc_target_clip)
        loss = (u_mean - target).pow(2).mean(-1)
        return loss[mask].mean()

    def _elite_chunk_bc_loss(
        self,
        a_seq: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if (self.cfg.lambda_elite_bc_pi <= 0.0
                or 'a_chunk' not in batch
                or 'chunk_mask' not in batch
                or 'source' not in batch):
            return a_seq.new_tensor(0.0)
        target = batch['a_chunk'].to(a_seq.device).clamp(-1.0, 1.0)
        mask = batch['chunk_mask'].to(a_seq.device)
        mask = mask * (batch['source'].to(a_seq.device) > 1.5).float().unsqueeze(-1)
        H = min(a_seq.shape[1], target.shape[1])
        if H <= 0:
            return a_seq.new_tensor(0.0)
        err = (a_seq[:, :H] - target[:, :H]).pow(2).mean(-1)
        mask = mask[:, :H]
        denom = mask.sum().clamp_min(1.0)
        return (err * mask).sum() / denom

    def _lqr_reg_loss(self, u_mean: torch.Tensor,
                      batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.lambda_lqr_pi <= 0.0 or 'has_lqr' not in batch:
            return u_mean.new_tensor(0.0)
        mask = batch['has_lqr'] > 0.5
        if mask.numel() == 0 or not bool(mask.any()):
            return u_mean.new_tensor(0.0)
        main = (u_mean - batch['u_lqr'].to(u_mean.device)).pow(2).mean(-1)
        aux = (u_mean - batch['u_lqr_aux'].to(u_mean.device)).pow(2).mean(-1)
        return (
            main[mask].mean()
            + self.cfg.lqr_aux_weight * aux[mask].mean()
        )

    @torch.no_grad()
    def lqr_target_for_goal(self, z: torch.Tensor, h: torch.Tensor,
                            goal_z: torch.Tensor):
        if self.lqr_planner is None or self.cfg.lambda_lqr_pi <= 0.0:
            return None
        z = z.detach().to(self.device)
        h = h.detach().to(self.device)
        goal_z = goal_z.detach().to(self.device)
        if z.dim() == 1: z = z.unsqueeze(0)
        if h.dim() == 1: h = h.unsqueeze(0)
        if goal_z.dim() == 1: goal_z = goal_z.unsqueeze(0)

        o_seq = torch.stack([z[0], goal_z[0]], dim=0).unsqueeze(0)
        h_seq = torch.stack([h[0], h[0]], dim=0).unsqueeze(0)
        g_seq = torch.stack([goal_z[0], goal_z[0]], dim=0).unsqueeze(0)
        targets = self.lqr_planner.sample_lqr_latent_action_targets(
            o_seq=o_seq, h_seq=h_seq, goal_z_seq=g_seq,
            H=self.cfg.lqr_horizon, aux_k=0)
        u_lqr = targets['main'][0, 0].detach().cpu().numpy()
        return u_lqr, u_lqr, 1.0

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': self.step,
            'cfg': self.model.cfg,
            'model_state': self.model.state_dict(),
            'opt_reward': self.opt_reward.state_dict(),
            'opt_q': self.opt_q.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'scale_tracker': self.model.scale_tracker.state_dict(),
            'q_loss_ema': self.q_loss_ema,
            'q_guide_active': self.q_guide_active,
            'elite_return_ema': self.elite_return_ema,
            'elite_return_last': self.elite_return_last,
            'elite_episode_count': self.elite_episode_count,
            'wm_penalty_threshold': self.wm_penalty_threshold,
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model_state'], strict=False)
        if 'opt_reward' in ck:
            self.opt_reward.load_state_dict(ck['opt_reward'])
            self.opt_q.load_state_dict(ck['opt_q'])
            self.opt_pi.load_state_dict(ck['opt_pi'])
        if 'scale_tracker' in ck:
            self.model.scale_tracker.load_state_dict(ck['scale_tracker'])
        self.q_loss_ema = ck.get('q_loss_ema', self.q_loss_ema)
        self.q_guide_active = bool(
            ck.get('q_guide_active', self.q_guide_active))
        self.elite_return_ema = float(
            ck.get('elite_return_ema', self.elite_return_ema))
        self.elite_return_last = float(
            ck.get('elite_return_last', self.elite_return_last))
        self.elite_episode_count = int(
            ck.get('elite_episode_count', self.elite_episode_count))
        self.wm_penalty_threshold = float(
            ck.get('wm_penalty_threshold', self.wm_penalty_threshold))
        self.step = int(ck.get('step', 0))
        return self.step

@torch.no_grad()
def verify_offline_elite_returns(
    env_name: str,
    episodes: List[Dict],
    candidate_returns: Dict[int, float],
    threshold: float,
) -> Dict[int, float]:
    """Replay offline actions in the real env and keep only verified elites."""
    import gym
    import d4rl  # noqa: F401

    env = gym.make(env_name)
    verified: Dict[int, float] = {}
    candidates = set(candidate_returns.keys())
    n_candidates = len(candidates)
    for ep in sorted(episodes, key=lambda e: int(e['start_t'])):
        start_t = int(ep['start_t'])
        if start_t not in candidates:
            continue
        env.reset()
        total_r = 0.0
        done = False
        acts = ep['actions'].astype(np.float32)
        for a in acts:
            _, r, done, _ = env.step(a.clip(-1.0, 1.0))
            total_r += float(r)
            if done:
                break
        if total_r >= threshold:
            verified[start_t] = total_r
    try:
        env.close()
    except Exception:
        pass
    print(
        f"[Prior Offline Elite Verify] env={env_name} "
        f"candidates={n_candidates} verified={len(verified)} "
        f"thr={threshold:.2f}"
    )
    return verified


@torch.no_grad()
def prefill_prior_buffer_from_offline(trainer: PolicyPriorOnlineTrainer,
                                      x_cache_path: str,
                                      quality: str = 'mixed',
                                      max_transitions: int = 50_000,
                                      device: str = 'cuda',
                                      env_name: str = 'kitchen-mixed-v0') -> int:
    """Offline data prefill for --prior_online with LQR regularization targets."""
    dev = torch.device(device)
    model = trainer.model
    model.eval()
    x_seq_full, _, _ = load_x_sequences(x_cache_path)
    episodes, _ = load_kitchen_episodes(quality=quality, min_len=trainer.cfg.cond_len + 2)
    verified_elite_returns: Dict[int, float] = {}
    if trainer.cfg.offline_elite_verify_env:
        candidate_returns: Dict[int, float] = {}
        for ep in episodes:
            rews = ep['rewards'].astype(np.float32)
            rew_max = float(rews.max()) if len(rews) else 0.0
            if len(rews) > 1 and np.all(np.diff(rews) >= -1e-6) and rew_max > 1.0:
                r_step = np.diff(rews, prepend=rews[0])
            else:
                r_step = rews
            r_step = np.clip(r_step, 0.0, 1.0).astype(np.float32)
            ep_return = float((r_step > 0.0).sum())
            if ep_return >= trainer.cfg.offline_elite_threshold:
                candidate_returns[int(ep['start_t'])] = ep_return
        verified_elite_returns = verify_offline_elite_returns(
            env_name,
            episodes,
            candidate_returns,
            trainer.cfg.offline_elite_threshold,
        )
    np.random.shuffle(episodes)

    n_added = 0
    n_elite_added = 0
    n_elite_eps = 0
    print(f"\n[Prior Offline Pre-fill] {x_cache_path}")
    for ep in episodes:
        if n_added >= max_transitions:
            break
        L = ep['length']
        if L < 2:
            continue
        s_t = ep['start_t']
        x_ep = x_seq_full[s_t:s_t + L]
        acts = ep['actions'].astype(np.float32)
        rews = ep['rewards'].astype(np.float32)

        x_t = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)
        a_t = torch.FloatTensor(acts).unsqueeze(0).to(dev)
        enc = model.encode_sequence(x_t, a_t)
        z_ep = enc['o_seq'][0]
        h_ep = enc['h_seq'][0]
        h_pre_ep = enc.get('h_pre_seq', enc['h_seq'])[0]
        u_ep = model.action_encoder(torch.FloatTensor(acts).to(dev))
        if L > 1:
            x_next_all = torch.FloatTensor(x_ep[1:L]).to(dev)
            wm_err_step = trainer._wm_prediction_error_tensor(
                z_ep[:L - 1], h_pre_ep[:L - 1], u_ep[:L - 1], x_next_all,
            ).detach().cpu().numpy().astype(np.float32)
        else:
            wm_err_step = np.zeros(0, dtype=np.float32)

        rew_max = float(rews.max()) if len(rews) else 0.0
        if len(rews) > 1 and np.all(np.diff(rews) >= -1e-6) and rew_max > 1.0:
            r_step = np.diff(rews, prepend=rews[0])
        else:
            r_step = rews
        r_step = np.clip(r_step, 0.0, 1.0).astype(np.float32)
        ep_return = float((r_step > 0.0).sum())

        u_lqr_np = np.zeros((L - 1, model.cfg.action_latent), dtype=np.float32)
        u_aux_np = np.zeros_like(u_lqr_np)
        has_lqr_np = np.zeros(L - 1, dtype=np.float32)
        if trainer.lqr_planner is not None and trainer.cfg.lambda_lqr_pi > 0.0:
            goal_z = trainer.lqr_planner.build_reward_goal_z_seq(
                ep['obs'], h_ep, ep['goal_info'].get('completions', {}),
                batch_size=64)
            targets = trainer.lqr_planner.sample_lqr_latent_action_targets(
                o_seq=z_ep.unsqueeze(0),
                h_seq=h_ep.unsqueeze(0),
                goal_z_seq=goal_z.unsqueeze(0),
                H=trainer.cfg.lqr_horizon,
                aux_k=trainer.cfg.lqr_aux_k,
            )
            u_lqr_np = targets['main'][0].cpu().numpy().astype(np.float32)
            u_aux_np = targets['aux'][0].cpu().numpy().astype(np.float32)
            has_lqr_np = (
                targets['main_mask'][0] | targets['aux_mask'][0]
            ).cpu().numpy().astype(np.float32)

        H_chunk = int(trainer.buf.chunk_horizon)
        n_step = max(1, int(trainer.cfg.q_n_step))
        gam = float(trainer.cfg.q_n_step_gamma)
        episode_transitions: List[Dict[str, np.ndarray]] = []
        for t in range(L - 1):
            a_chunk = np.zeros((H_chunk, model.cfg.action_dim), dtype=np.float32)
            chunk_mask = np.zeros(H_chunk, dtype=np.float32)
            n_chunk = min(H_chunk, max(0, len(acts) - t))
            if n_chunk > 0:
                a_chunk[:n_chunk] = acts[t:t + n_chunk]
                chunk_mask[:n_chunk] = 1.0
            skill_logits_t = model.skill_prior(h_pre_ep[t:t + 1])
            skill_prob_t = torch.softmax(skill_logits_t, dim=-1)
            skill_id_t = int(skill_prob_t.argmax(dim=-1).item())
            a_chunk_t = torch.FloatTensor(a_chunk).unsqueeze(0).to(dev)
            _, arg_mu_t, _ = model.skill_argument_encoder(
                z_ep[t:t + 1], h_pre_ep[t:t + 1], skill_prob_t, a_chunk_t)
            skill_arg_t = torch.tanh(arg_mu_t)[0].detach().cpu().numpy()
            skill_prob_np = skill_prob_t[0].detach().cpu().numpy().astype(np.float32)

            n_avail = min(n_step, (L - 1) - t)
            r_ret = 0.0
            for j in range(n_avail):
                r_ret += (gam ** j) * float(r_step[t + j] > 0.0)
            boot_idx = min(t + n_avail, L - 1)
            done_n = boot_idx >= (L - 1)
            wm_end = min(t + max(1, n_avail), len(wm_err_step))
            wm_err = (
                float(np.max(wm_err_step[t:wm_end]))
                if wm_end > t else 0.0
            )
            trans = {
                'z': z_ep[t].cpu().numpy(),
                'h': h_pre_ep[t].cpu().numpy(),
                'u': u_ep[t].cpu().numpy(),
                'a': acts[t],
                'a_chunk': a_chunk,
                'chunk_mask': chunk_mask,
                'r': float(r_step[t] > 0.0),
                'r_return': float(r_ret),
                'z_next': z_ep[t + 1].cpu().numpy(),
                'h_next': h_pre_ep[t + 1].cpu().numpy(),
                'z_boot': z_ep[boot_idx].cpu().numpy(),
                'h_boot': h_pre_ep[boot_idx].cpu().numpy(),
                'discount': float(0.0 if done_n else gam ** max(1, n_avail)),
                'done': float(t + 1 >= L - 1),
                'u_lqr': u_lqr_np[t],
                'u_lqr_aux': u_aux_np[t],
                'has_lqr': has_lqr_np[t],
                'skill_id': skill_id_t,
                'skill_prob': skill_prob_np,
                'skill_arg': skill_arg_t,
                'has_skill_arg': 1.0,
                'wm_err': wm_err,
                'source': 1.0,
            }
            trainer.buf.add(**trans)
            episode_transitions.append(trans)
            n_added += 1
            if n_added >= max_transitions:
                break
        elite_return = ep_return
        if trainer.cfg.offline_elite_verify_env:
            elite_return = verified_elite_returns.get(int(ep['start_t']), -1.0)
        if (episode_transitions
                and elite_return >= trainer.cfg.offline_elite_threshold):
            n_elite_added += trainer.add_elite_episode(
                episode_transitions,
                max(elite_return, trainer.cfg.elite_reward_threshold),
            )
            n_elite_eps += 1

    print(
        f"[Prior Offline Pre-fill] added {n_added}  buf={trainer.buf.size} "
        f"offline_elite={n_elite_added} eps={n_elite_eps} "
        f"thr={trainer.cfg.offline_elite_threshold:.2f} "
        f"verify={int(trainer.cfg.offline_elite_verify_env)} "
        f"elite_buf={trainer.elite_size}"
    )
    return n_added


class KODAQOnlineTrainer:
    """
    Hierarchical online RL trainer.
    π_hi: Gumbel-Softmax skill selection
    π_lo: action chunk, IQL pretrained + AWR fine-tune
    Q:    z + u(action encoder) + skill_probs
    """

    def __init__(self, cfg: OnlineConfig, wm: KoopmanWorldModelWrapper,
                 z_dim: int, n_skills: int, action_dim: int, device: str):
        self.cfg        = cfg
        self.wm         = wm
        self.device     = device
        self.z_dim      = z_dim
        self.n_skills   = n_skills
        self.action_dim = action_dim
        self.gumbel_tau = cfg.gumbel_tau
        self.step       = 0

        h, nl   = cfg.hidden_dim, cfg.n_layers
        u_dim   = wm.model.cfg.action_latent
        self.u_dim = u_dim

        self.pi_hi = HighLevelPolicy(z_dim, n_skills, h, nl).to(device)
        self.pi_lo = LowLevelPolicy(z_dim, action_dim, cfg.H_lo, h, nl).to(device)
        self.Q1    = QNetwork(z_dim, u_dim, n_skills, h, nl).to(device)
        self.Q2    = QNetwork(z_dim, u_dim, n_skills, h, nl).to(device)
        self.Q1_t  = copy.deepcopy(self.Q1)
        self.Q2_t  = copy.deepcopy(self.Q2)

        lr = cfg.lr
        self.opt_hi = torch.optim.Adam(self.pi_hi.parameters(), lr=lr)
        self.opt_lo = torch.optim.Adam(self.pi_lo.parameters(), lr=lr)
        self.opt_q  = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=lr)

        # IQL prior policy (stop_gradient) for lo KL regularization
        self.iql_pi: Optional[nn.Module] = None

        self.buf = ReplayBuffer(cfg.buffer_size, device)

    @property
    def alpha(self): return self.cfg.alpha

    def _anneal_tau(self):
        frac = min(1.0, self.step / max(self.cfg.n_env_steps, 1))
        self.gumbel_tau = (self.cfg.gumbel_tau -
                           (self.cfg.gumbel_tau - self.cfg.gumbel_tau_min) * frac)

    @torch.no_grad()
    def _encode_u(self, a_seq: torch.Tensor) -> torch.Tensor:
        """a_seq: (B, H_lo, 9) → u: (B, d_u) via action_encoder on first step"""
        return self.wm.model.action_encoder(a_seq[:, 0, :])

    def update(self) -> Dict[str, float]:
        if self.buf.size < self.cfg.batch_size: return {}
        b    = self.buf.sample(self.cfg.batch_size)
        z    = b['z']; z_nx = b['z_next']; h_t = b['h_t']
        sp   = b['sp']; a = b['a_seq']; r = b['r']; done = b['done']
        u    = self._encode_u(a)

        # ── Q Critic ────────────────────────────────────────────────────────
        with torch.no_grad():
            sp_nx, _ = self.pi_hi.gumbel_sample(z_nx, self.gumbel_tau)
            a_nx, _  = self.pi_lo.sample(z_nx)
            u_nx     = self._encode_u(a_nx)
            V_next   = torch.min(self.Q1_t(z_nx, u_nx, sp_nx),
                                 self.Q2_t(z_nx, u_nx, sp_nx))
        gHlo = self.cfg.gamma ** self.cfg.H_lo
        y    = (r + gHlo * (1 - done) * V_next).clamp(0.0, 5.0)
        q1   = self.Q1(z, u, sp); q2 = self.Q2(z, u, sp)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.cfg.grad_clip)
        self.opt_q.step()

        # ── Hi Actor: KL(sp || p_skill) - Q ─────────────────────────────────
        with torch.no_grad():
            p_prior = self.wm.model.skill_prior.soft_weights(h_t)
        sp_new, _ = self.pi_hi.gumbel_sample(z, self.gumbel_tau)
        with torch.no_grad():
            a_det, _ = self.pi_lo.sample(z)
            u_det    = self._encode_u(a_det)
            q_hi     = torch.min(self.Q1(z, u_det, sp_new),
                                 self.Q2(z, u_det, sp_new))
        kl_hi   = self.pi_hi.kl_prior(sp_new, p_prior)
        loss_hi = (self.cfg.kl_weight * kl_hi - q_hi).mean()
        self.opt_hi.zero_grad(); loss_hi.backward()
        nn.utils.clip_grad_norm_(self.pi_hi.parameters(), self.cfg.grad_clip)
        self.opt_hi.step()

        # ── Lo Actor: AWR + KL(π_lo || π_iql) ───────────────────────────────
        sp_d = sp_new.detach()
        with torch.no_grad():
            a_lo, _ = self.pi_lo.sample(z)
            u_lo    = self._encode_u(a_lo)
            q_lo    = torch.min(self.Q1(z, u_lo, sp_d),
                                self.Q2(z, u_lo, sp_d))
            v_lo    = torch.min(self.Q1(z, u, sp_d),
                                self.Q2(z, u, sp_d))
            adv     = q_lo - v_lo
            w_awr   = torch.exp(self.cfg.awr_beta * adv).clamp(max=100.0)

        log_prob = self.pi_lo.log_prob(z, a)

        lp_iql = torch.zeros_like(log_prob)
        if self.iql_pi is not None:
            with torch.no_grad():
                a_iql, _ = self.iql_pi.sample(z)
                a_iql    = a_iql.clamp(-1 + 1e-6, 1 - 1e-6)
            lp_iql = self.pi_lo.log_prob(z, a_iql.detach())

        loss_lo = (-(w_awr * log_prob) -
                   self.cfg.kl_lqr_weight * lp_iql).mean()
        self.opt_lo.zero_grad(); loss_lo.backward()
        nn.utils.clip_grad_norm_(self.pi_lo.parameters(), self.cfg.grad_clip)
        self.opt_lo.step()

        # entropy (logging only)
        with torch.no_grad():
            ent = (self.pi_hi.entropy(z) + self.pi_lo.entropy(z)).mean()

        # ── Q Target EMA ────────────────────────────────────────────────────
        tau = self.cfg.tau_ema
        for p, pt in zip(self.Q1.parameters(), self.Q1_t.parameters()):
            pt.data.mul_(1 - tau).add_(p.data, alpha=tau)
        for p, pt in zip(self.Q2.parameters(), self.Q2_t.parameters()):
            pt.data.mul_(1 - tau).add_(p.data, alpha=tau)
        self._anneal_tau()

        return {
            'loss_q':  loss_q.item(), 'loss_hi': loss_hi.item(),
            'loss_lo': loss_lo.item(), 'kl_hi':  kl_hi.mean().item(),
            'ent':     ent.item(),    'q_mean':  q_hi.mean().item(),
            'gumbel_tau': self.gumbel_tau,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step':       self.step,
            'pi_hi':      self.pi_hi.state_dict(),
            'pi_lo':      self.pi_lo.state_dict(),
            'Q1':         self.Q1.state_dict(),   'Q2':  self.Q2.state_dict(),
            'Q1_t':       self.Q1_t.state_dict(), 'Q2_t':self.Q2_t.state_dict(),
            'opt_hi':     self.opt_hi.state_dict(),
            'opt_lo':     self.opt_lo.state_dict(),
            'opt_q':      self.opt_q.state_dict(),
            'world_model':self.wm.model.state_dict(),
            'wm_opt':     self.wm.opt.state_dict(),
            'gumbel_tau': self.gumbel_tau,
        }, path)
        print(f"  Saved: {path}")

    def load(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.pi_hi.load_state_dict(ck['pi_hi'])
        self.pi_lo.load_state_dict(ck['pi_lo'])
        self.Q1.load_state_dict(ck['Q1']);  self.Q2.load_state_dict(ck['Q2'])
        self.Q1_t.load_state_dict(ck.get('Q1_t', ck['Q1']))
        self.Q2_t.load_state_dict(ck.get('Q2_t', ck['Q2']))
        if 'opt_hi' in ck:
            self.opt_hi.load_state_dict(ck['opt_hi'])
            self.opt_lo.load_state_dict(ck['opt_lo'])
            self.opt_q.load_state_dict(ck['opt_q'])
        if 'wm_opt' in ck: self.wm.opt.load_state_dict(ck['wm_opt'])
        if 'world_model' in ck:
            self.wm.model.load_state_dict(ck['world_model'], strict=False)
            self.wm.model.eval()
        self.gumbel_tau = ck.get('gumbel_tau', self.cfg.gumbel_tau)
        self.step = ck.get('step', 0)
        print(f"  Loaded: {path}  step={self.step}")
        return self.step


# ─────────────────────────────────────────────────────────────────────────────
# IQL Weight Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_iql_for_online(iql_ckpt: str, trainer: KODAQOnlineTrainer,
                        device: str):
    """IQL checkpoint → π_lo + Q 초기화 + iql_pi frozen 보관"""
    from iql_koopman import ChunkPolicy, IQLConfig
    ck      = torch.load(iql_ckpt, map_location=device)
    z_dim   = trainer.z_dim
    a_dim   = trainer.action_dim
    H_lo    = trainer.cfg.H_lo
    cfg_iql = IQLConfig()

    pi_iql = ChunkPolicy(z_dim, a_dim, H_lo, cfg_iql.hidden_dim, cfg_iql.n_layers)
    pi_iql.load_state_dict(ck['pi'])
    pi_iql.eval().to(device)

    # π_lo backbone + mu/log_s 이식
    with torch.no_grad():
        for p1, p2 in zip(trainer.pi_lo.net.parameters(),
                          pi_iql.net.parameters()):
            p1.data.copy_(p2.data)
        for k in range(H_lo):
            trainer.pi_lo.mu.weight.data[k*a_dim:(k+1)*a_dim].copy_(pi_iql.mu.weight.data)
            trainer.pi_lo.mu.bias.data[k*a_dim:(k+1)*a_dim].copy_(pi_iql.mu.bias.data)
            trainer.pi_lo.log_s.weight.data[k*a_dim:(k+1)*a_dim].copy_(pi_iql.log_s.weight.data)
            trainer.pi_lo.log_s.bias.data[k*a_dim:(k+1)*a_dim].copy_(pi_iql.log_s.bias.data)

    # Q weights 복사
    for qname in ['Q1', 'Q2', 'Q1_t', 'Q2_t']:
        src_key = qname if qname in ck else qname.replace('_t', '')
        if src_key in ck:
            try:
                getattr(trainer, qname).load_state_dict(ck[src_key])
            except Exception as e:
                print(f"  Q weight copy {qname} failed: {e}")

    # iql_pi: stop_gradient 보관
    trainer.iql_pi = pi_iql
    for p in trainer.iql_pi.parameters():
        p.requires_grad_(False)

    print(f"  π_lo + Q + iql_pi loaded from: {iql_ckpt}  H_lo={H_lo}")
    return pi_iql


# ─────────────────────────────────────────────────────────────────────────────
# Offline Pre-fill
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prefill_buffer_from_offline(trainer: KODAQOnlineTrainer,
                                wm: KoopmanWorldModelWrapper,
                                x_cache_path: str,
                                quality: str = 'mixed',
                                max_transitions: int = 50_000,
                                device: str = 'cuda') -> int:
    """offline mixed dataset → replay buffer pre-fill with compute_r_blend"""
    dev   = torch.device(device)
    model = wm.model; model.eval()
    cfg   = trainer.cfg
    H_lo  = cfg.H_lo

    print(f"\n[Offline Pre-fill] {x_cache_path}")
    x_seq_full, _, _ = load_x_sequences(x_cache_path)
    episodes, _      = load_kitchen_episodes(quality=quality, min_len=H_lo + 16)
    np.random.shuffle(episodes)

    n_added = 0
    for ep in episodes:
        if n_added >= max_transitions: break
        L    = ep['length']; acts = ep['actions']; rews = ep['rewards']
        s_t  = ep['start_t']
        x_ep = x_seq_full[s_t:s_t + L]

        x_t = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)
        a_t = torch.FloatTensor(acts).unsqueeze(0).to(dev)
        enc = model.encode_sequence(x_t, a_t)
        z_ep = enc['o_seq'][0].cpu().numpy()   # (L, m)
        h_ep = enc['h_seq'][0]                  # (L, d_h)

        for t in range(L - H_lo - 1):
            z_dev = torch.FloatTensor(z_ep[t]).unsqueeze(0).to(dev)
            sp    = model.skill_prior.soft_weights(
                h_ep[t:t+1].to(dev)).cpu().numpy()[0]

            a_chunk = acts[t:t+H_lo].clip(-1, 1).astype(np.float32)

            # compute_r_blend with event + acc rewards
            r_event = wm._r_hat_event(z_dev)
            r_acc   = (wm.cat_head.expected_reward(z_dev).item()
                       if wm.cat_head is not None else 0.0)
            r_blend = compute_r_blend(float(rews[t]), r_acc, r_event,
                                      cfg.w_env, cfg.w_acc, cfg.w_event)

            trainer.buf.add(
                z      = z_ep[t],
                z_next = z_ep[t + H_lo],
                h_t    = h_ep[t].cpu().numpy(),
                sp     = sp,
                a_seq  = a_chunk,
                r      = r_blend,
                done   = 0.0,
            )
            n_added += 1
            if n_added >= max_transitions: break

    print(f"[Offline Pre-fill] added {n_added}  buf={trainer.buf.size}")
    return n_added


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(trainer: KODAQOnlineTrainer, wm: KoopmanWorldModelWrapper,
             env_name: str, n_ep: int, cfg: OnlineConfig, device: str) -> Dict:
    import gym, d4rl
    dev   = torch.device(device)
    model = wm.model; model.eval()
    results = []

    for _ in range(n_ep):
        env = gym.make(env_name); obs = env.reset()
        ctx = EnvContext(model, device, cfg.cond_len); ctx.reset(obs)
        total_r = 0.0; n_tasks = 0; done = False; hi_timer = 0; sp = None

        for t in range(400):  # Kitchen 4 subtasks × ~100 steps
            if done: break
            if ctx.z_t is None:
                obs, r, done, info = env.step(env.action_space.sample())
                ctx.step(obs, env.action_space.sample()); total_r += r; continue

            if hi_timer == 0:
                sp, _ = trainer.pi_hi.gumbel_sample(ctx.z_t, tau=0.1)
                hi_timer = cfg.H_hi

            with torch.no_grad():
                a_seq, _ = trainer.pi_lo.sample(ctx.z_t)

            for k in range(cfg.H_lo):
                if done: break
                obs, r, done, info = env.step(a_seq[0, k].cpu().numpy().clip(-1, 1))
                ctx.step(obs, a_seq[0, k].cpu().numpy()); total_r += r
                ep_comp = info.get('episode_task_completions',
                                   info.get('completed_tasks', []))
                n_tasks = max(n_tasks, len(ep_comp) if isinstance(ep_comp, list)
                              else int(ep_comp))
            hi_timer = max(0, hi_timer - cfg.H_lo)

        results.append({'reward': total_r, 'n_tasks': n_tasks}); env.close()

    mr = np.mean([x['reward']  for x in results])
    mt = np.mean([x['n_tasks'] for x in results])
    print(f"  [EVAL] mean_reward={mr:.3f}  mean_tasks={mt:.2f}")
    return {'eval_reward': mr, 'eval_tasks': mt}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def visualize_training(log: Dict, out_path: str):
    keys   = ['loss_q','loss_hi','loss_lo','kl_hi','ent','q_mean','ep_reward','ep_tasks']
    titles = ['Q Loss','π Hi Loss','π Lo Loss','KL Hi','Entropy','Q mean','Ep Reward','Tasks']
    PAL    = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300','#607D8B']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.flatten()
    for i, (k, t) in enumerate(zip(keys, titles)):
        if k not in log or not log[k]: continue
        vals = np.array(log[k]); ax = axes[i]
        ax.plot(vals, color=PAL[i], alpha=0.25, lw=0.8)
        w = max(1, min(50, len(vals)//5))
        if len(vals) >= w:
            ax.plot(np.convolve(vals, np.ones(w)/w, 'valid'), color=PAL[i], lw=1.8)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('KODAQ-Online v3', fontsize=12, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()


def train(cfg: OnlineConfig, trainer: KODAQOnlineTrainer,
          wm: KoopmanWorldModelWrapper, env_name: str,
          out_dir: str, device: str, use_wandb: bool = False):
    import gym, d4rl
    dev   = torch.device(device)
    model = wm.model
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log_keys = ['loss_q','loss_hi','loss_lo','kl_hi','ent',
                'q_mean','wm_loss','ep_reward','ep_tasks']
    log    = {k: [] for k in log_keys}
    recent = {k: deque(maxlen=cfg.log_every) for k in log_keys}

    env = gym.make(env_name); obs = env.reset()
    ctx = EnvContext(model, device, cfg.cond_len); ctx.reset(obs)

    ep_r = 0.0; ep_tasks = 0; hi_timer = 0; sp = None
    global_step = trainer.step; t0 = time.time()
    wm_obs_buf: List[np.ndarray] = []; wm_r_buf: List[float] = []

    print(f"\n{'='*60}")
    print(f"KODAQ-Online v3  steps={cfg.n_env_steps}  env={env_name}")
    print(f"  H_hi={cfg.H_hi}  H_lo={cfg.H_lo}  buf={trainer.buf.size}")
    print(f"  r_blend: w_env={cfg.w_env} w_acc={cfg.w_acc} w_event={cfg.w_event}")
    print(f"{'='*60}\n")

    while global_step < cfg.n_env_steps:

        # context warmup: random until z_t available
        if ctx.z_t is None:
            obs, r, done, info = env.step(env.action_space.sample())
            ctx.step(obs, env.action_space.sample())
            global_step += 1; ep_r += r
            if done:
                recent['ep_reward'].append(ep_r); recent['ep_tasks'].append(ep_tasks)
                obs = env.reset(); ctx.reset(obs); ep_r = 0.0; ep_tasks = 0
            continue

        z_t = ctx.z_t; h_t = ctx.h_t

        # Hi-level skill selection
        if hi_timer == 0:
            with torch.no_grad():
                sp, _ = trainer.pi_hi.gumbel_sample(z_t, tau=0.3)
            hi_timer = cfg.H_hi

        # Lo-level action chunk
        with torch.no_grad():
            a_seq, _ = trainer.pi_lo.sample(z_t)
        a_np = a_seq[0].cpu().numpy()

        z_b = z_t.clone(); h_b = h_t.clone()
        r_env_tot = 0.0; r_hat_acc = 0.0; r_hat_event = 0.0; done = False
        a_pad = np.zeros((cfg.H_lo, trainer.action_dim), dtype=np.float32)

        for k in range(cfg.H_lo):
            ak = a_np[k].clip(-1, 1)
            obs_nx, r_env, done, info = env.step(ak)
            r_env_tot += r_env; ep_r += r_env
            ep_c = info.get('episode_task_completions',
                            info.get('completed_tasks', []))
            ep_tasks = max(ep_tasks, len(ep_c) if isinstance(ep_c, list) else int(ep_c))
            global_step += 1; a_pad[k] = ak

            # r_hat_event per step
            if ctx.z_t is not None:
                with torch.no_grad():
                    r_hat_event += wm._r_hat_event(ctx.z_t)

            xnp = ctx._obs_to_x(obs_nx)
            wm_obs_buf.append(xnp); wm_r_buf.append(float(r_env))
            if len(wm_obs_buf) > 512: wm_obs_buf.pop(0); wm_r_buf.pop(0)

            ctx.step(obs_nx, ak); obs = obs_nx
            if done: break

        # r_hat: H_lo 구간 누적 → 평균
        r_hat_event /= max(cfg.H_lo, 1)
        with torch.no_grad():
            a_t_full  = torch.FloatTensor(a_pad).to(dev)
            r_hat_acc = wm._r_hat_accumulated(z_b, h_b, a_t_full)

        r_blend = compute_r_blend(r_env_tot, r_hat_acc, r_hat_event,
                                  cfg.w_env, cfg.w_acc, cfg.w_event)

        if ctx.z_t is not None:
            sp_np = sp.cpu().numpy()[0] if sp is not None else \
                    np.ones(trainer.n_skills, dtype=np.float32) / trainer.n_skills
            trainer.buf.add(
                z      = z_b.cpu().numpy()[0],
                z_next = ctx.z_t.cpu().numpy()[0],
                h_t    = h_b.cpu().numpy()[0],
                sp     = sp_np,
                a_seq  = a_pad,
                r      = r_blend,
                done   = float(done),
            )

        hi_timer = max(0, hi_timer - cfg.H_lo)

        # Update
        for _ in range(cfg.n_updates_per_step):
            info_d = trainer.update()
            for k, v in info_d.items():
                if k in recent: recent[k].append(v)

        # WM update
        if global_step % cfg.wm_update_freq == 0 and len(wm_obs_buf) >= 32:
            idx = np.random.choice(len(wm_obs_buf), 32, replace=False)
            xb  = torch.FloatTensor(np.array([wm_obs_buf[i] for i in idx])).to(dev)
            rb  = torch.FloatTensor(np.array([wm_r_buf[i]   for i in idx])).to(dev)
            recent['wm_loss'].append(wm.update(xb, rb))

        if done:
            recent['ep_reward'].append(ep_r); recent['ep_tasks'].append(ep_tasks)
            obs = env.reset(); ctx.reset(obs); ep_r = 0.0; ep_tasks = 0; hi_timer = 0; sp = None

        # Logging
        if global_step % cfg.log_every == 0:
            ms  = {k: np.mean(list(recent[k])) if recent[k] else 0.0
                   for k in log_keys}
            for k in log_keys: log[k].append(ms[k])
            sps = cfg.log_every / (time.time() - t0 + 1e-6); t0 = time.time()
            trainer.step = global_step
            print(f"Step {global_step:7d} | "
                  f"Q={ms['loss_q']:.3f} Hi={ms['loss_hi']:.3f} "
                  f"Lo={ms['loss_lo']:.3f} KL={ms['kl_hi']:.3f} | "
                  f"ent={ms['ent']:.2f} q={ms['q_mean']:.3f} | "
                  f"ep_r={ms['ep_reward']:.3f} tasks={ms['ep_tasks']:.2f} | "
                  f"{sps:.0f}sps")
            if use_wandb:
                wandb.log({f"train/{k}": v for k, v in ms.items()},
                          step=global_step)

        if global_step % cfg.save_every == 0:
            trainer.save(f"{out_dir}/kodaq_online_step{global_step}.pt")
            visualize_training(log, f"{out_dir}/training_curves.png")

        if global_step % cfg.eval_every == 0:
            er = evaluate(trainer, wm, env_name, cfg.n_eval_ep, cfg, device)
            for k, v in er.items(): log.setdefault(k, []).append(v)
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in er.items()},
                          step=global_step)

    trainer.save(f"{out_dir}/kodaq_online_final.pt")
    visualize_training(log, f"{out_dir}/training_curves_final.png")
    env.close()
    print(f"\nDone. {global_step} steps → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_policy_prior_online(cfg: OnlineConfig,
                              trainer: PolicyPriorOnlineTrainer,
                              env_name: str,
                              out_dir: str,
                              device: str,
                              use_wandb: bool = False):
    import gym, d4rl
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = gym.make(env_name)
    obs = env.reset()
    ctx = EnvContext(trainer.model, device, cfg.cond_len)
    ctx.reset(obs)
    recent = {k: deque(maxlen=cfg.log_every) for k in
              ['loss_reward', 'loss_q', 'loss_q_td', 'loss_q_cql_policy',
               'loss_q_rank_data', 'loss_q_rank_bc', 'loss_q_rank_lqr',
               'loss_pi', 'loss_lqr_pi', 'loss_bc_pi',
               'loss_elite_bc_pi', 'loss_elite_skill_bc_pi',
               'loss_elite_arg_bc_pi',
                'loss_skilldec_anchor', 'skilldec_anchor_w',
                 'q_mean', 'q_policy_cql', 'q_data', 'rho', 'r_hat',
                 'q_reward', 'q_wm_penalty', 'q_wm_error',
                 'q_wm_threshold', 'q_target', 'q_next', 'pi_active',
                 'pi_q_weight_eff', 'q_loss_ema', 'q_guide_active',
                 'ep_reward', 'ep_tasks', 'elite_size',
                 'elite_ep_reward', 'elite_added']}
    global_step = trainer.step
    ep_r = 0.0
    ep_tasks = 0
    episode_transitions: List[Dict[str, np.ndarray]] = []
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"PolicyPrior Online  steps={cfg.n_env_steps}  env={env_name}")
    print(f"  actor={cfg.actor_mode}  exec_h={cfg.skilldec_exec_horizon} "
          f"noise={cfg.action_noise_std:.3f}")
    print(f"  batch={cfg.batch_size}  buf={cfg.buffer_size}  cond={cfg.cond_len}")
    print(f"  reward batch pos/off={cfg.reward_positive_fraction:.2f}/"
          f"{cfg.reward_offline_fraction:.2f} "
          f"elite={cfg.reward_elite_fraction:.2f}")
    print(f"  q batch pos/off={cfg.q_positive_fraction:.2f}/"
          f"{cfg.q_offline_fraction:.2f}  reward={cfg.q_reward_source} "
          f"boot={cfg.q_bootstrap:.2f} n={cfg.q_n_step} "
          f"g={cfg.q_n_step_gamma:.2f} elite={cfg.q_elite_fraction:.2f}")
    print(f"  q combo cql={cfg.lambda_q_cql_policy:.3f} "
          f"rank_data={cfg.lambda_q_rank_data:.3f} "
          f"rank_bc={cfg.lambda_q_rank_bc:.3f} "
          f"rank_lqr={cfg.lambda_q_rank_lqr:.3f} "
          f"margin={cfg.q_rank_margin:.3f}")
    print(f"  q success_only={int(cfg.q_success_only)} "
          f"success_value={cfg.q_success_value:.2f} "
          f"neg={cfg.lambda_q_negative:.2f} "
          f"succ_rank={cfg.lambda_q_success_rank:.2f} "
          f"succ_margin={cfg.q_success_margin:.2f} "
          f"succ_thr={cfg.q_success_threshold:.2f} "
          f"succ_src={int(cfg.q_success_include_source)}")
    print(f"  q wm_penalty lambda={cfg.lambda_q_wm_penalty:.3f} "
          f"pct={cfg.q_wm_penalty_percentile:.1f} "
          f"clip={cfg.q_wm_penalty_clip:.3f} "
          f"min_elite={cfg.q_wm_penalty_min_elite}")
    print(f"  pi batch pos/off={cfg.pi_positive_fraction:.2f}/"
          f"{cfg.pi_offline_fraction:.2f}  pi_start={cfg.pi_update_start} "
          f"elite={cfg.pi_elite_fraction:.2f} "
          f"q_w={cfg.pi_q_weight:.3f} "
          f"q_gate={cfg.pi_q_guide_loss_threshold:.3f}/"
          f"{cfg.pi_q_guide_min_step} bc={cfg.lambda_bc_pi:.3f} "
          f"ebc={cfg.lambda_elite_bc_pi:.3f} "
          f"eskill={cfg.lambda_elite_skill_bc_pi:.3f} "
          f"earg={cfg.lambda_elite_arg_bc_pi:.3f} "
          f"anchor={cfg.lambda_skilldec_anchor:.3f}->{cfg.skilldec_anchor_min:.3f}")
    if cfg.pi_q_guide_loss_threshold > 0.0:
        print(
            "  phase P0=critic-only before pi_start | "
            "P1=elite BC only(QW=0) until "
            f"step>={cfg.pi_q_guide_min_step} and "
            f"qema<={cfg.pi_q_guide_loss_threshold:.3f} | "
            f"P2=elite BC + Q guide(QW={cfg.pi_q_weight:.3f})"
        )
    else:
        print(
            "  phase P0=critic-only before pi_start | "
            f"P2=actor uses Q guide after pi_start(QW={cfg.pi_q_weight:.3f})"
        )
    if cfg.actor_mode == 'skill_arg':
        print(f"  skill-arg SAC alpha_d={cfg.skill_arg_alpha_d:.3f} "
              f"alpha_z={cfg.skill_arg_alpha_z:.3f} "
              f"arg_use_h={int(getattr(trainer.model.cfg, 'skill_arg_use_h', True))} "
              f"skill_d_no_h={int(cfg.skill_d_no_h)} "
              f"mean_after_pi={int(cfg.skill_arg_mean_after_pi_start)} "
              "critic=Q(z,h,arg,d)")
    print(f"  elite buffer size={cfg.elite_buffer_size} "
          f"threshold={cfg.elite_reward_threshold:.2f} "
          f"offline_threshold={cfg.offline_elite_threshold:.2f} "
          f"offline_verify={int(cfg.offline_elite_verify_env)}")
    print(f"{'='*60}\n")

    while global_step < cfg.n_env_steps:
        cur = ctx.current_latent()
        if cur is None:
            a = env.action_space.sample()
            obs_nx, r, done, info = env.step(a)
            ctx.step(obs_nx, a)
            obs = obs_nx
            ep_r += r
            global_step += 1
            if done:
                recent['ep_reward'].append(ep_r)
                recent['ep_tasks'].append(ep_tasks)
                added = trainer.add_elite_episode(episode_transitions, ep_r)
                if added > 0:
                    recent['elite_ep_reward'].append(ep_r)
                    recent['elite_added'].append(float(added))
                obs = env.reset()
                ctx.reset(obs)
                ep_r = 0.0
                ep_tasks = 0
                episode_transitions = []
            continue

        z_plan, h_plan = cur
        exec_h = (
            cfg.skilldec_exec_horizon
            if cfg.actor_mode in ('skill_decoder', 'skill_arg')
            else 1
        )
        a_seq_np, u_seq_np = trainer.act_sequence(
            z_plan.clone(), h_plan.clone(), horizon=exec_h)
        skillarg_meta = trainer.last_skillarg_meta()

        if cfg.actor_mode == 'skill_arg':
            z_start = z_plan.cpu().numpy()[0].copy()
            h_start = h_plan.cpu().numpy()[0].copy()
            exec_actions: List[np.ndarray] = []
            exec_us: List[np.ndarray] = []
            exec_rewards: List[float] = []
            exec_wm_errors: List[float] = []
            z_final = None
            h_final = None
            done_chunk = False

            for a_np, u_np in zip(a_seq_np, u_seq_np):
                cur_step = ctx.current_latent()
                if cur_step is None or global_step >= cfg.n_env_steps:
                    break
                z_step, h_step = cur_step
                obs_nx, r_env, done, info = env.step(a_np.clip(-1, 1))
                x_next_np = ctx._obs_to_x(obs_nx)
                exec_wm_errors.append(
                    trainer.wm_prediction_error_np(
                        z_step, h_step, np.asarray(u_np, dtype=np.float32),
                        x_next_np,
                    )
                )
                ctx.step(obs_nx, a_np)

                ep_r += r_env
                ep_c = info.get('episode_task_completions',
                                info.get('completed_tasks', []))
                ep_tasks = max(ep_tasks, len(ep_c) if isinstance(ep_c, list)
                               else int(ep_c) if ep_c is not None else 0)

                exec_actions.append(np.asarray(a_np, dtype=np.float32).copy())
                exec_us.append(np.asarray(u_np, dtype=np.float32).copy())
                exec_rewards.append(float(r_env > 0.0))
                nxt = ctx.current_latent()
                if nxt is not None:
                    z_next_t, h_next_t = nxt
                    z_final = z_next_t.cpu().numpy()[0].copy()
                    h_final = h_next_t.cpu().numpy()[0].copy()
                obs = obs_nx
                global_step += 1
                trainer.step = global_step
                done_chunk = bool(done)
                if done:
                    break

            if exec_actions and z_final is not None:
                H_exec = len(exec_actions)
                r_ret = sum(
                    (cfg.q_n_step_gamma ** j) * exec_rewards[j]
                    for j in range(H_exec)
                )
                chunk_mask = np.ones(H_exec, dtype=np.float32)
                trans = {
                    'z': z_start,
                    'h': h_start,
                    'u': exec_us[0],
                    'a': exec_actions[0],
                    'a_chunk': np.stack(exec_actions, axis=0),
                    'chunk_mask': chunk_mask,
                    'r': exec_rewards[0],
                    'r_return': float(r_ret),
                    'z_next': z_final,
                    'h_next': h_final,
                    'z_boot': z_final,
                    'h_boot': h_final,
                    'discount': float(
                        0.0 if done_chunk else cfg.q_n_step_gamma ** H_exec),
                    'done': float(done_chunk),
                    'skill_id': (
                        None if 'skill_id' not in skillarg_meta
                        else int(np.asarray(skillarg_meta['skill_id']).reshape(-1)[0])
                    ),
                    'skill_arg': (
                        None if 'skill_arg' not in skillarg_meta
                        else np.asarray(skillarg_meta['skill_arg'],
                                        dtype=np.float32).reshape(-1).copy()
                    ),
                    'has_skill_arg': 1.0 if 'skill_arg' in skillarg_meta else 0.0,
                    'wm_err': (
                        float(np.max(exec_wm_errors))
                        if exec_wm_errors else 0.0
                    ),
                    'source': 0.0,
                }
                trainer.buf.add(**trans)
                episode_transitions.append(trans)

                for _ in range(cfg.n_updates_per_step * H_exec):
                    info_d = trainer.update()
                    for k, v in info_d.items():
                        if k in recent:
                            recent[k].append(v)

            if done_chunk:
                recent['ep_reward'].append(ep_r)
                recent['ep_tasks'].append(ep_tasks)
                added = trainer.add_elite_episode(episode_transitions, ep_r)
                if added > 0:
                    recent['elite_ep_reward'].append(ep_r)
                    recent['elite_added'].append(float(added))
                obs = env.reset()
                ctx.reset(obs)
                ep_r = 0.0
                ep_tasks = 0
                episode_transitions = []

            if global_step % cfg.log_every < max(1, len(exec_actions)):
                trainer.step = global_step
                ms = {k: np.mean(list(v)) if v else 0.0
                      for k, v in recent.items()}
                elite_epr = (ms['elite_ep_reward']
                             if ms['elite_ep_reward'] > 0.0
                             else trainer.elite_return_ema)
                sps = cfg.log_every / (time.time() - t0 + 1e-6)
                t0 = time.time()
                print(f"Step {global_step:7d} | "
                      f"R={ms['loss_reward']:.3f} Q={ms['loss_q']:.3f} "
                      f"TD={ms['loss_q_td']:.3f} CQ={ms['loss_q_cql_policy']:.3f} "
                      f"RD={ms['loss_q_rank_data']:.3f} "
                      f"Pi={ms['loss_pi']:.3f} Anc={ms['loss_skilldec_anchor']:.3f} "
                      f"EBC={ms['loss_elite_bc_pi']:.3f} "
                      f"ESK={ms['loss_elite_skill_bc_pi']:.3f} "
                      f"ARG={ms['loss_elite_arg_bc_pi']:.3f} "
                      f"QW={ms['pi_q_weight_eff']:.3f} "
                      f"Aw={ms['skilldec_anchor_w']:.2f} "
                      f"q={ms['q_mean']:.3f} "
                      f"qd={ms['q_data']:.3f} rho={ms['rho']:.3f} rhat={ms['r_hat']:.3f} "
                      f"rQ={ms['q_reward']:.3f} WMp={ms['q_wm_penalty']:.3f} "
                      f"WMe={ms['q_wm_error']:.3f} WMt={ms['q_wm_threshold']:.3f} "
                      f"y={ms['q_target']:.3f} "
                      f"pi={ms['pi_active']:.0f} qg={ms['q_guide_active']:.0f} "
                      f"qema={ms['q_loss_ema']:.3f} | "
                      f"ep_r={ms['ep_reward']:.2f} tasks={ms['ep_tasks']:.2f} | "
                      f"elite={trainer.elite_size} epr={elite_epr:.2f} | "
                      f"{sps:.0f}sps")
                if use_wandb:
                    wandb.log({f"prior_online/{k}": v for k, v in ms.items()},
                              step=global_step)

            if global_step % cfg.save_every < max(1, len(exec_actions)):
                trainer.step = global_step
                trainer.save(f"{out_dir}/policy_prior_online_step{global_step}.pt")

            continue

        for a_np, u_np in zip(a_seq_np, u_seq_np):
            cur_step = ctx.current_latent()
            if cur_step is None or global_step >= cfg.n_env_steps:
                break
            z_t, h_t = cur_step
            z_t = z_t.clone()
            h_t = h_t.clone()

            obs_nx, r_env, done, info = env.step(a_np.clip(-1, 1))
            x_next_np = ctx._obs_to_x(obs_nx)
            wm_err = trainer.wm_prediction_error_np(
                z_t, h_t, np.asarray(u_np, dtype=np.float32), x_next_np)
            ctx.step(obs_nx, a_np)

            ep_r += r_env
            ep_c = info.get('episode_task_completions',
                            info.get('completed_tasks', []))
            ep_tasks = max(ep_tasks, len(ep_c) if isinstance(ep_c, list)
                           else int(ep_c) if ep_c is not None else 0)

            nxt = ctx.current_latent()
            if nxt is not None:
                z_next_t, h_next_t = nxt
                u_lqr = None
                u_lqr_aux = None
                has_lqr = 0.0
                if r_env > 0.0:
                    lqr_t = trainer.lqr_target_for_goal(z_t, h_t, z_next_t)
                    if lqr_t is not None:
                        u_lqr, u_lqr_aux, has_lqr = lqr_t
                trans = {
                    'z': z_t.cpu().numpy()[0].copy(),
                    'h': h_t.cpu().numpy()[0].copy(),
                    'u': np.asarray(u_np, dtype=np.float32).copy(),
                    'a': np.asarray(a_np, dtype=np.float32).copy(),
                    'r': float(r_env > 0.0),
                    'r_return': float(r_env > 0.0),
                    'z_next': z_next_t.cpu().numpy()[0].copy(),
                    'h_next': h_next_t.cpu().numpy()[0].copy(),
                    'z_boot': z_next_t.cpu().numpy()[0].copy(),
                    'h_boot': h_next_t.cpu().numpy()[0].copy(),
                    'discount': float(0.0 if done else cfg.q_n_step_gamma),
                    'done': float(done),
                    'u_lqr': None if u_lqr is None else np.asarray(u_lqr, dtype=np.float32).copy(),
                    'u_lqr_aux': None if u_lqr_aux is None else np.asarray(u_lqr_aux, dtype=np.float32).copy(),
                    'has_lqr': has_lqr,
                    'skill_id': (
                        None if 'skill_id' not in skillarg_meta
                        else int(np.asarray(skillarg_meta['skill_id']).reshape(-1)[0])
                    ),
                    'skill_arg': (
                        None if 'skill_arg' not in skillarg_meta
                        else np.asarray(skillarg_meta['skill_arg'], dtype=np.float32).reshape(-1).copy()
                    ),
                    'has_skill_arg': 1.0 if 'skill_arg' in skillarg_meta else 0.0,
                    'wm_err': wm_err,
                    'source': 0.0,
                }
                trainer.buf.add(
                    **trans,
                )
                episode_transitions.append(trans)

            trainer.step = global_step
            for _ in range(cfg.n_updates_per_step):
                info_d = trainer.update()
                for k, v in info_d.items():
                    if k in recent:
                        recent[k].append(v)

            obs = obs_nx
            global_step += 1

            if done:
                recent['ep_reward'].append(ep_r)
                recent['ep_tasks'].append(ep_tasks)
                added = trainer.add_elite_episode(episode_transitions, ep_r)
                if added > 0:
                    recent['elite_ep_reward'].append(ep_r)
                    recent['elite_added'].append(float(added))
                obs = env.reset()
                ctx.reset(obs)
                ep_r = 0.0
                ep_tasks = 0
                episode_transitions = []

            if global_step % cfg.log_every == 0:
                trainer.step = global_step
                ms = {k: np.mean(list(v)) if v else 0.0
                      for k, v in recent.items()}
                elite_epr = (ms['elite_ep_reward']
                             if ms['elite_ep_reward'] > 0.0
                             else trainer.elite_return_ema)
                sps = cfg.log_every / (time.time() - t0 + 1e-6)
                t0 = time.time()
                print(f"Step {global_step:7d} | "
                      f"R={ms['loss_reward']:.3f} Q={ms['loss_q']:.3f} "
                      f"TD={ms['loss_q_td']:.3f} CQ={ms['loss_q_cql_policy']:.3f} "
                      f"RD={ms['loss_q_rank_data']:.3f} "
                      f"Pi={ms['loss_pi']:.3f} Anc={ms['loss_skilldec_anchor']:.3f} "
                      f"EBC={ms['loss_elite_bc_pi']:.3f} "
                      f"ESK={ms['loss_elite_skill_bc_pi']:.3f} "
                      f"ARG={ms['loss_elite_arg_bc_pi']:.3f} "
                      f"QW={ms['pi_q_weight_eff']:.3f} "
                      f"Aw={ms['skilldec_anchor_w']:.2f} "
                      f"q={ms['q_mean']:.3f} "
                      f"qd={ms['q_data']:.3f} rho={ms['rho']:.3f} rhat={ms['r_hat']:.3f} "
                      f"rQ={ms['q_reward']:.3f} WMp={ms['q_wm_penalty']:.3f} "
                      f"WMe={ms['q_wm_error']:.3f} WMt={ms['q_wm_threshold']:.3f} "
                      f"y={ms['q_target']:.3f} "
                      f"pi={ms['pi_active']:.0f} qg={ms['q_guide_active']:.0f} "
                      f"qema={ms['q_loss_ema']:.3f} | "
                      f"ep_r={ms['ep_reward']:.2f} tasks={ms['ep_tasks']:.2f} | "
                      f"elite={trainer.elite_size} epr={elite_epr:.2f} | "
                      f"{sps:.0f}sps")
                if use_wandb:
                    wandb.log({f"prior_online/{k}": v for k, v in ms.items()},
                              step=global_step)

            if global_step % cfg.save_every == 0:
                trainer.step = global_step
                trainer.save(f"{out_dir}/policy_prior_online_step{global_step}.pt")

            if done:
                break

    trainer.step = global_step
    trainer.save(f"{out_dir}/policy_prior_online_final.pt")
    env.close()
    print(f"\nDone. {global_step} steps -> {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_ckpt',    default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--iql_ckpt',      default=None)
    p.add_argument('--cat_ckpt',      default=None)
    p.add_argument('--x_cache',       default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--resume',        default=None)
    p.add_argument('--env',           default='kitchen-mixed-v0')
    p.add_argument('--out_dir',       default='checkpoints/kodaq_v4/online_v9')
    p.add_argument('--H_hi',          type=int,   default=8)
    p.add_argument('--H_lo',          type=int,   default=4)
    p.add_argument('--gamma',         type=float, default=0.99)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--batch_size',    type=int,   default=256)
    p.add_argument('--n_updates_per_step', type=int, default=1)
    p.add_argument('--kl_weight',     type=float, default=1.0)
    p.add_argument('--kl_lqr_weight', type=float, default=0.1)
    p.add_argument('--awr_beta',      type=float, default=3.0)
    p.add_argument('--n_steps',       type=int,   default=1_000_000)
    p.add_argument('--wm_lr',         type=float, default=1e-4)
    p.add_argument('--w_env',         type=float, default=0.4)
    p.add_argument('--w_acc',         type=float, default=0.4)
    p.add_argument('--w_event',       type=float, default=0.2)
    p.add_argument('--buffer_size',   type=int,   default=200_000)
    p.add_argument('--prefill_size',  type=int,   default=50_000)
    p.add_argument('--no_prefill',    action='store_true')
    p.add_argument('--eval_every',    type=int,   default=10_000)
    p.add_argument('--n_eval_ep',     type=int,   default=10)
    p.add_argument('--log_every',     type=int,   default=1_000)
    p.add_argument('--save_every',    type=int,   default=50_000)
    p.add_argument('--device',        default='cuda:1'
                   if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run',     default=None)
    p.add_argument('--prior_online',  action='store_true',
                   help='Fine-tune a KoopmanCVAE actor online. Use --actor_mode to choose policy_prior or skill_decoder.')
    p.add_argument('--action_inv_steps', type=int, default=30)
    p.add_argument('--action_inv_lr',    type=float, default=0.05)
    p.add_argument('--positive_fraction', type=float, default=0.5)
    p.add_argument('--offline_fraction',  type=float, default=0.5)
    p.add_argument('--reward_positive_fraction', type=float, default=None)
    p.add_argument('--reward_offline_fraction',  type=float, default=None)
    p.add_argument('--q_positive_fraction',      type=float, default=0.05)
    p.add_argument('--q_offline_fraction',       type=float, default=0.8)
    p.add_argument('--pi_positive_fraction',     type=float, default=0.02)
    p.add_argument('--pi_offline_fraction',      type=float, default=0.8)
    p.add_argument('--reward_elite_fraction',    type=float, default=0.0)
    p.add_argument('--q_elite_fraction',         type=float, default=0.0)
    p.add_argument('--pi_elite_fraction',        type=float, default=0.0)
    p.add_argument('--elite_buffer_size',        type=int,   default=50_000)
    p.add_argument('--elite_reward_threshold',   type=float, default=1.0)
    p.add_argument('--offline_elite_threshold',  type=float, default=2.0)
    p.add_argument('--offline_elite_verify_env', action='store_true',
                   help='Replay offline actions in the real env and only add verified successful episodes to elite buffer.')
    p.add_argument('--lambda_elite_bc_pi',       type=float, default=0.0)
    p.add_argument('--lambda_elite_skill_bc_pi', type=float, default=0.0)
    p.add_argument('--lambda_elite_arg_bc_pi',   type=float, default=0.0)
    p.add_argument('--q_n_step',                 type=int,   default=1)
    p.add_argument('--q_n_step_gamma',           type=float, default=0.99)
    p.add_argument('--q_reward_source', choices=['env', 'rhat', 'penalized'],
                   default='env')
    p.add_argument('--q_bootstrap', type=float, default=0.0)
    p.add_argument('--q_success_only', action='store_true',
                   help='For skill_arg Q, give positive targets only to elite/success chunks and zero to others.')
    p.add_argument('--q_success_value', type=float, default=1.0)
    p.add_argument('--q_success_threshold', type=float, default=0.5,
                   help='For skill_arg success-only Q, mark a chunk positive only when r_return exceeds this threshold.')
    p.add_argument('--q_success_include_source', action='store_true',
                   help='Also mark source>1.5 elite chunks positive for Q. Off by default; BC still uses source.')
    p.add_argument('--lambda_q_negative', type=float, default=0.0)
    p.add_argument('--lambda_q_success_rank', type=float, default=0.0)
    p.add_argument('--q_success_margin', type=float, default=0.5)
    p.add_argument('--lambda_q_wm_penalty', type=float, default=0.0)
    p.add_argument('--q_wm_penalty_percentile', type=float, default=95.0)
    p.add_argument('--q_wm_penalty_clip', type=float, default=1.0)
    p.add_argument('--q_wm_penalty_min_elite', type=int, default=128)
    p.add_argument('--q_wm_penalty_q_weight', type=float, default=1.0)
    p.add_argument('--q_wm_penalty_obj_weight', type=float, default=1.0)
    p.add_argument('--pi_update_start', type=int, default=10_000)
    p.add_argument('--actor_mode', choices=['policy_prior', 'skill_decoder', 'skill_arg'],
                   default='policy_prior')
    p.add_argument('--pi_q_weight', type=float, default=1.0)
    p.add_argument('--pi_q_guide_loss_threshold', type=float, default=-1.0,
                   help='If >0, keep skill_arg actor Q guide off until Q loss EMA is below this threshold.')
    p.add_argument('--pi_q_guide_min_step', type=int, default=0,
                   help='Minimum env step before enabling Q guide from the Q-loss gate.')
    p.add_argument('--pi_q_guide_ema_beta', type=float, default=0.99)
    p.add_argument('--action_noise_std', type=float, default=0.0)
    p.add_argument('--skilldec_exec_horizon', type=int, default=1,
                   help='Number of decoded skill actions to execute open-loop before replanning.')
    p.add_argument('--lambda_bc_pi', type=float, default=0.0)
    p.add_argument('--bc_target_clip', type=float, default=1.0)
    p.add_argument('--lambda_q_cql_policy', type=float, default=0.0)
    p.add_argument('--lambda_q_rank_data',   type=float, default=0.0)
    p.add_argument('--lambda_q_rank_bc',     type=float, default=0.0)
    p.add_argument('--lambda_q_rank_lqr',    type=float, default=0.0)
    p.add_argument('--q_rank_margin',        type=float, default=0.05)
    p.add_argument('--lambda_skilldec_anchor', type=float, default=0.0)
    p.add_argument('--skilldec_anchor_min', type=float, default=0.0)
    p.add_argument('--skilldec_anchor_decay_steps', type=int, default=200_000)
    p.add_argument('--skill_arg_alpha_d', type=float, default=0.1)
    p.add_argument('--skill_arg_alpha_z', type=float, default=0.01)
    p.add_argument('--skill_d_no_h', action='store_true',
                   help='For skill_arg actor, condition pi_d on z only by zeroing h and using a uniform skill prior.')
    p.add_argument('--skill_arg_sample_after_pi_start', action='store_true',
                   help='Keep sampling c after pi_start instead of using mean_arg for rollout.')
    p.add_argument('--lambda_lqr_pi',     type=float, default=0.0)
    p.add_argument('--lqr_horizon',       type=int,   default=4)
    p.add_argument('--lqr_aux_k',         type=int,   default=4)
    p.add_argument('--lqr_aux_weight',    type=float, default=0.25)
    p.add_argument('--lqr_Q_scale',       type=float, default=1.0)
    p.add_argument('--lqr_R_scale',       type=float, default=10.0)
    p.add_argument('--u_bounds_path',     default=None)
    args = p.parse_args()

    if not sys.stdout.isatty():
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except AttributeError:
            pass

    device = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run or 'kodaq_online_v3',
                   config=vars(args))
        print(f"[wandb] project={args.wandb_project} run={wandb.run.name} "
              f"url={wandb.run.url}", flush=True)
    else:
        print(f"[wandb] disabled available={WANDB_AVAILABLE} "
              f"project={args.wandb_project}", flush=True)

    # World model
    print(f"\nLoading: {args.world_ckpt}")
    ck    = torch.load(args.world_ckpt, map_location=device)
    model = KoopmanCVAE(ck['cfg'])
    model.load_state_dict(ck['model_state'], strict=False)
    model.eval().to(device)
    z_dim    = model.cfg.koopman_dim
    n_skills = model.cfg.num_skills
    a_dim    = model.cfg.action_dim
    print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")

    # cat_head
    cat_head = None
    cat_path = args.cat_ckpt or str(
        Path(args.world_ckpt).parent / 'cat_reward' / 'final.pt')
    if Path(cat_path).exists():
        from train_reward_head import load_cat_reward_model
        _, cat_head = load_cat_reward_model(cat_path, device)
        print(f"  CategoricalRewardHead loaded")

    cfg = OnlineConfig(
        H_hi=args.H_hi, H_lo=args.H_lo, gamma=args.gamma,
        lr=args.lr, batch_size=args.batch_size,
        kl_weight=args.kl_weight, kl_lqr_weight=args.kl_lqr_weight,
        awr_beta=args.awr_beta, n_env_steps=args.n_steps,
        buffer_size=args.buffer_size,
        wm_lr=args.wm_lr, w_env=args.w_env, w_acc=args.w_acc,
        w_event=args.w_event, eval_every=args.eval_every,
        n_eval_ep=args.n_eval_ep, log_every=args.log_every,
        save_every=args.save_every,
        n_updates_per_step=args.n_updates_per_step,
        positive_fraction=args.positive_fraction,
        offline_fraction=args.offline_fraction,
        reward_positive_fraction=(
            args.reward_positive_fraction
            if args.reward_positive_fraction is not None
            else args.positive_fraction
        ),
        reward_offline_fraction=(
            args.reward_offline_fraction
            if args.reward_offline_fraction is not None
            else args.offline_fraction
        ),
        q_positive_fraction=args.q_positive_fraction,
        q_offline_fraction=args.q_offline_fraction,
        pi_positive_fraction=args.pi_positive_fraction,
        pi_offline_fraction=args.pi_offline_fraction,
        reward_elite_fraction=args.reward_elite_fraction,
        q_elite_fraction=args.q_elite_fraction,
        pi_elite_fraction=args.pi_elite_fraction,
        elite_buffer_size=args.elite_buffer_size,
        elite_reward_threshold=args.elite_reward_threshold,
        offline_elite_threshold=args.offline_elite_threshold,
        offline_elite_verify_env=args.offline_elite_verify_env,
        lambda_elite_bc_pi=args.lambda_elite_bc_pi,
        lambda_elite_skill_bc_pi=args.lambda_elite_skill_bc_pi,
        lambda_elite_arg_bc_pi=args.lambda_elite_arg_bc_pi,
        q_n_step=args.q_n_step,
        q_n_step_gamma=args.q_n_step_gamma,
        q_reward_source=args.q_reward_source,
        q_bootstrap=args.q_bootstrap,
        q_success_only=args.q_success_only,
        q_success_value=args.q_success_value,
        q_success_threshold=args.q_success_threshold,
        q_success_include_source=args.q_success_include_source,
        lambda_q_negative=args.lambda_q_negative,
        lambda_q_success_rank=args.lambda_q_success_rank,
        q_success_margin=args.q_success_margin,
        lambda_q_wm_penalty=args.lambda_q_wm_penalty,
        q_wm_penalty_percentile=args.q_wm_penalty_percentile,
        q_wm_penalty_clip=args.q_wm_penalty_clip,
        q_wm_penalty_min_elite=args.q_wm_penalty_min_elite,
        q_wm_penalty_q_weight=args.q_wm_penalty_q_weight,
        q_wm_penalty_obj_weight=args.q_wm_penalty_obj_weight,
        pi_update_start=args.pi_update_start,
        actor_mode=args.actor_mode,
        pi_q_weight=args.pi_q_weight,
        pi_q_guide_loss_threshold=args.pi_q_guide_loss_threshold,
        pi_q_guide_min_step=args.pi_q_guide_min_step,
        pi_q_guide_ema_beta=args.pi_q_guide_ema_beta,
        action_noise_std=args.action_noise_std,
        skilldec_exec_horizon=args.skilldec_exec_horizon,
        lambda_bc_pi=args.lambda_bc_pi,
        bc_target_clip=args.bc_target_clip,
        lambda_q_cql_policy=args.lambda_q_cql_policy,
        lambda_q_rank_data=args.lambda_q_rank_data,
        lambda_q_rank_bc=args.lambda_q_rank_bc,
        lambda_q_rank_lqr=args.lambda_q_rank_lqr,
        q_rank_margin=args.q_rank_margin,
        lambda_skilldec_anchor=args.lambda_skilldec_anchor,
        skilldec_anchor_min=args.skilldec_anchor_min,
        skilldec_anchor_decay_steps=args.skilldec_anchor_decay_steps,
        skill_arg_alpha_d=args.skill_arg_alpha_d,
        skill_arg_alpha_z=args.skill_arg_alpha_z,
        skill_d_no_h=args.skill_d_no_h,
        skill_arg_mean_after_pi_start=not args.skill_arg_sample_after_pi_start,
        lambda_lqr_pi=args.lambda_lqr_pi,
        lqr_horizon=args.lqr_horizon,
        lqr_aux_k=args.lqr_aux_k,
        lqr_aux_weight=args.lqr_aux_weight,
    )

    if args.prior_online:
        lqr_planner = None
        if cfg.lambda_lqr_pi > 0.0:
            lqr_planner = KODAQLQRPlanner(
                model,
                LQRConfig(Q_scale=args.lqr_Q_scale, R_scale=args.lqr_R_scale),
            )
            if args.u_bounds_path and Path(args.u_bounds_path).exists():
                lqr_planner.load_u_bounds(args.u_bounds_path)
            lqr_planner.precompute_gains(H=cfg.lqr_horizon)
            print(f"  LQR policy regularizer enabled: "
                  f"lambda={cfg.lambda_lqr_pi} aux_k={cfg.lqr_aux_k}")
        trainer = PolicyPriorOnlineTrainer(
            cfg, model, device,
            action_inv_steps=args.action_inv_steps,
            action_inv_lr=args.action_inv_lr,
            lqr_planner=lqr_planner,
        )
        did_resume = args.resume and Path(args.resume).exists()
        if did_resume:
            trainer.load(args.resume)
            print("  Resume loaded model/optim states. Replay buffer is not "
                  "stored in checkpoints; use offline prefill unless this is "
                  "intentionally online-only.")
        if not args.no_prefill:
            quality = 'mixed'
            if 'partial' in args.env:
                quality = 'partial'
            elif 'complete' in args.env:
                quality = 'complete'
            prefill_prior_buffer_from_offline(
                trainer, args.x_cache, quality=quality,
                max_transitions=args.prefill_size, device=device,
                env_name=args.env)
        train_policy_prior_online(
            cfg, trainer, args.env, args.out_dir, device, use_wandb)
        if use_wandb:
            wandb.finish()
        return

    wm      = KoopmanWorldModelWrapper(model, cfg.wm_lr, device,
                                        cat_head=cat_head)
    trainer = KODAQOnlineTrainer(cfg, wm, z_dim, n_skills, a_dim, device)

    # Load IQL weights
    iql_path = args.iql_ckpt or str(
        Path(args.world_ckpt).parent / 'iql_v3' / 'iql_final.pt')
    if Path(iql_path).exists():
        load_iql_for_online(iql_path, trainer, device)
    else:
        print(f"  [Warning] IQL ckpt not found: {iql_path}")

    # Resume
    if args.resume and Path(args.resume).exists():
        trainer.load(args.resume)
    elif not args.no_prefill:
        prefill_buffer_from_offline(
            trainer, wm, args.x_cache,
            max_transitions=args.prefill_size, device=device)

    train(cfg, trainer, wm, args.env, args.out_dir, device, use_wandb)
    if use_wandb: wandb.finish()


if __name__ == '__main__':
    main()
