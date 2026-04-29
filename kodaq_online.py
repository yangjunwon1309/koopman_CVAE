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
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    blend_koopman,
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
        if model.cfg.use_reward_head:
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
        if not self.model.cfg.use_reward_head: return 0.0
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


# ─────────────────────────────────────────────────────────────────────────────
# KODAQ Online Trainer
# ─────────────────────────────────────────────────────────────────────────────

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
            self.wm.model.load_state_dict(ck['world_model'])
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
    p.add_argument('--kl_weight',     type=float, default=1.0)
    p.add_argument('--kl_lqr_weight', type=float, default=0.1)
    p.add_argument('--awr_beta',      type=float, default=3.0)
    p.add_argument('--n_steps',       type=int,   default=1_000_000)
    p.add_argument('--wm_lr',         type=float, default=1e-4)
    p.add_argument('--w_env',         type=float, default=0.4)
    p.add_argument('--w_acc',         type=float, default=0.4)
    p.add_argument('--w_event',       type=float, default=0.2)
    p.add_argument('--prefill_size',  type=int,   default=50_000)
    p.add_argument('--no_prefill',    action='store_true')
    p.add_argument('--eval_every',    type=int,   default=10_000)
    p.add_argument('--n_eval_ep',     type=int,   default=10)
    p.add_argument('--device',        default='cuda:1'
                   if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run',     default=None)
    args = p.parse_args()

    device = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run or 'kodaq_online_v3',
                   config=vars(args))

    # World model
    print(f"\nLoading: {args.world_ckpt}")
    ck    = torch.load(args.world_ckpt, map_location=device)
    model = KoopmanCVAE(ck['cfg'])
    model.load_state_dict(ck['model_state'])
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
        wm_lr=args.wm_lr, w_env=args.w_env, w_acc=args.w_acc,
        w_event=args.w_event, eval_every=args.eval_every,
        n_eval_ep=args.n_eval_ep,
    )

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