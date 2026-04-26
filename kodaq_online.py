"""
kodaq_online.py — KODAQ-Online v2
==================================

주요 변경:
  1. LQR warmup: 초반 random 대신 LQR action으로 buffer 채움
  2. Hi-level policy: Gumbel-Softmax로 gradient 통과 + KL prior regularization
  3. 단일 Q(z, skill_probs, a_seq): hi/lo Q 통합
  4. wandb 실시간 모니터링
  5. z_goal 제거 (state-only, EXTRACT와 동일)
  6. α clamp [-5, 1] (폭발 방지)
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
from models.losses import symexp
from data.extract_skill_label import load_x_sequences
from lqr_koopman import (
    KODAQLQRPlanner, LQRConfig,
    load_kitchen_episodes, obs_to_x_goal,
    blend_koopman,
    X_DQ_START, X_DQ_END, X_DP_START, X_DP_END,
    OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. pip install wandb")

X_QD_START = 2099; X_QD_END = 2108

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OnlineConfig:
    H_hi:           int   = 8
    H_lo:           int   = 4
    gamma:          float = 0.99
    tau_ema:        float = 0.005
    alpha:          float = 0.0    # entropy penalization 없음 — 순수 Q maximization
    kl_weight:      float = 1.0
    gumbel_tau:     float = 1.0
    gumbel_tau_min: float = 0.3
    hidden_dim:     int   = 256
    n_layers:       int   = 2
    lr:             float = 3e-4
    batch_size:     int   = 256
    grad_clip:      float = 1.0
    n_updates_per_step: int = 1
    wm_lr:          float = 1e-4
    wm_update_freq: int   = 10
    r_alpha_warmup: int   = 30_000
    warmup_lqr:     int   = 3_000
    lqr_H_warmup:   int   = 8
    n_env_steps:    int   = 300_000
    buffer_size:    int   = 200_000
    log_every:      int   = 1_000
    save_every:     int   = 50_000
    eval_every:     int   = 25_000
    n_eval_ep:      int   = 5
    cond_len:       int   = 16


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(in_dim, out_dim, hidden, n_layers, output_act=None):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if output_act:
        layers.append(output_act)
    return nn.Sequential(*layers)


class HighLevelPolicy(nn.Module):
    """
    π_hi(skill | z_t) — Categorical over K skills
    Gumbel-Softmax로 gradient 통과 (soft skill_probs → Q 계산)
    KL(π_hi || p_skill(c|h)) regularization
    """
    def __init__(self, z_dim: int, n_skills: int, hidden: int, n_layers: int):
        super().__init__()
        self.n_skills = n_skills
        self.net = make_mlp(z_dim, n_skills, hidden, n_layers)

    def logits(self, z): return self.net(z)
    def probs(self, z):  return torch.softmax(self.logits(z), dim=-1)

    def gumbel_sample(self, z, tau=1.0):
        """Returns (skill_probs (B,K) soft, log_prob (B,))"""
        logits = self.logits(z)
        sp     = F.gumbel_softmax(logits, tau=tau, hard=False)
        log_pi = torch.log_softmax(logits, dim=-1)
        lp     = (sp * log_pi).sum(dim=-1)
        return sp, lp

    def hard_sample(self, z):
        """실제 env 실행용 hard sample"""
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=self.logits(z))
            sid  = dist.sample()
        return sid.item(), dist.log_prob(sid).item()

    def kl_prior(self, z, p_prior):
        """KL(π_hi || p_skill(c|h)), p_prior: (B,K)"""
        log_pi    = torch.log_softmax(self.logits(z), dim=-1)
        log_prior = torch.log(p_prior + 1e-8)
        return (torch.exp(log_pi) * (log_pi - log_prior)).sum(dim=-1)

    def entropy(self, z):
        return torch.distributions.Categorical(logits=self.logits(z)).entropy()


class LowLevelPolicy(nn.Module):
    """
    π_lo(a | z_t, skill_probs) → a ∈ R^{H_lo * action_dim}
    직접 action 출력, z_goal 없음 (state-only)
    mu 작게 초기화 → LQR pretrain에 유리
    """
    def __init__(self, z_dim, n_skills, action_dim, H_lo, hidden, n_layers,
                 log_std_min=-4.0, log_std_max=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.H_lo = H_lo
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_dim = z_dim + n_skills
        self.net   = make_mlp(in_dim, hidden, hidden, n_layers - 1)
        self.mu    = nn.Linear(hidden, H_lo * action_dim)
        self.log_s = nn.Linear(hidden, H_lo * action_dim)
        nn.init.uniform_(self.mu.weight, -0.01, 0.01)
        nn.init.zeros_(self.mu.bias)

    def forward(self, z, sp):
        feat = self.net(torch.cat([z, sp], dim=-1))
        mu   = self.mu(feat).view(-1, self.H_lo, self.action_dim)
        ls   = self.log_s(feat).view(-1, self.H_lo, self.action_dim)
        return mu, ls.clamp(self.log_std_min, self.log_std_max)

    def sample(self, z, sp):
        mu, ls = self(z, sp)
        dist   = torch.distributions.Normal(mu, ls.exp())
        u      = dist.rsample()
        a      = torch.tanh(u)
        lp     = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6))
        return a, lp.sum(dim=(-2, -1))

    def log_prob(self, z, sp, a_seq):
        mu, ls = self(z, sp)
        u   = torch.atanh(a_seq.clamp(-1+1e-6, 1-1e-6))
        dist= torch.distributions.Normal(mu, ls.exp())
        lp  = dist.log_prob(u) - torch.log(1 - a_seq.pow(2) + 1e-6)
        return lp.sum(dim=(-2, -1))

    def entropy(self, z, sp):
        _, ls = self(z, sp)
        return (0.5*(1 + 2*ls + math.log(2*math.pi))).sum(dim=(-2,-1))

    def pretrain_lqr(self, z_b, sp_b, a_lqr_b, n_steps=500, lr=1e-3):
        """
        LQR action sequence를 mu 초기값으로 supervised pretrain.
        - n_steps=500: 충분히 수렴
        - log_std는 높게 유지 (초반 탐색 허용)
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_val = 0.0
        for i in range(n_steps):
            mu, log_std = self(z_b, sp_b)
            # mu → LQR 방향으로
            loss_mu = F.mse_loss(mu, a_lqr_b)
            # log_std → 초반엔 높게 유지 (-1.0 target: σ≈0.37로 탐색 가능)
            loss_std = F.mse_loss(log_std,
                                  torch.full_like(log_std, -1.0))
            loss = loss_mu + 0.1 * loss_std
            opt.zero_grad(); loss.backward(); opt.step()
            loss_val = loss_mu.item()
            if (i+1) % 100 == 0:
                print(f"    pretrain [{i+1}/{n_steps}] mu_loss={loss_mu.item():.4f}")
        return loss_val


class QNetwork(nn.Module):
    """
    단일 Q(z_t, skill_probs, a_seq_flat) → scalar
    factorized π = π_hi * π_lo 에 대한 통합 Q
    """
    def __init__(self, z_dim, n_skills, action_dim, H_lo, hidden, n_layers):
        super().__init__()
        self.net = make_mlp(z_dim + n_skills + action_dim * H_lo, 1,
                            hidden, n_layers)

    def forward(self, z, sp, a_seq):
        a_flat = a_seq.reshape(a_seq.shape[0], -1)
        return self.net(torch.cat([z, sp, a_flat], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device   = device
        self._d: Dict[str, np.ndarray] = {}
        self._ptr = 0; self._n = 0

    def _init(self, z_dim, n_skills, action_dim, H_lo, h_dim):
        C = self.capacity
        self._d = {
            'z':     np.zeros((C, z_dim),          dtype=np.float32),
            'z_next':np.zeros((C, z_dim),          dtype=np.float32),
            'h_t':   np.zeros((C, h_dim),          dtype=np.float32),
            'sp':    np.zeros((C, n_skills),        dtype=np.float32),
            'a_seq': np.zeros((C, H_lo, action_dim),dtype=np.float32),
            'r':     np.zeros(C,                   dtype=np.float32),
            'done':  np.zeros(C,                   dtype=np.float32),
        }

    def add(self, z, z_next, h_t, sp, a_seq, r, done):
        if not self._d:
            self._init(z.shape[-1], sp.shape[-1],
                       a_seq.shape[-1], a_seq.shape[-2], h_t.shape[-1])
        p = self._ptr
        self._d['z'][p]=z; self._d['z_next'][p]=z_next
        self._d['h_t'][p]=h_t; self._d['sp'][p]=sp
        self._d['a_seq'][p]=a_seq; self._d['r'][p]=r
        self._d['done'][p]=float(done)
        self._ptr=(p+1)%self.capacity
        self._n=min(self._n+1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._n, batch_size)
        return {k: torch.FloatTensor(v[idx]).to(self.device)
                for k, v in self._d.items()}

    @property
    def size(self): return self._n


# ─────────────────────────────────────────────────────────────────────────────
# World Model Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanWorldModelWrapper:
    def __init__(self, model, wm_lr, device):
        self.model = model; self.device = device
        for p in model.koopman.parameters(): p.requires_grad_(False)
        for p in model.decoder.parameters(): p.requires_grad_(False)
        reward_params = []
        if model.cfg.use_reward_head:
            if hasattr(model.decoder, 'head_reward'):
                for p in model.decoder.head_reward.parameters():
                    p.requires_grad_(True)
                reward_params = list(model.decoder.head_reward.parameters())
            elif hasattr(model, 'reward_head'):
                for p in model.reward_head.parameters():
                    p.requires_grad_(True)
                reward_params = list(model.reward_head.parameters())
        active = (list(model.posterior.parameters()) +
                  list(model.recurrent.parameters()) +
                  list(model.skill_prior.parameters()) +
                  reward_params)
        self.opt = torch.optim.Adam(active, lr=wm_lr)

    @torch.no_grad()
    def encode_ctx(self, x_seq, a_seq):
        enc = self.model.encode_sequence(x_seq, a_seq)
        return enc['o_seq'][0,-1:], enc['h_seq'][0,-1:]

    @torch.no_grad()
    def koopman_step(self, z, a, h):
        m = self.model
        u = m.action_encoder(a)
        w = m.skill_prior.soft_weights(h)
        ll= m.koopman.get_log_lambdas()
        A, B, _, _ = blend_koopman(ll, m.koopman.theta_k,
                                   m.koopman.G_k, m.koopman.U, w)
        A=A[0]; B=B[0]
        z_nx = (A@z.T).T + (B@u.T).T
        h_nx = m.recurrent(h, z, a)
        return z_nx, h_nx, self._r_hat(z_nx)

    def _r_hat(self, z):
        m = self.model
        if not m.cfg.use_reward_head: return 0.0
        if hasattr(m.decoder, 'head_reward'):
            r = m.decoder.head_reward(z)
        elif hasattr(m, 'reward_head'):
            r = m.reward_head(z)
        else: return 0.0
        return torch.sigmoid(r).mean().item()

    def update(self, x_b, r_env_b):
        if not self.model.cfg.use_reward_head: return 0.0
        m = self.model; m.train()
        B = x_b.shape[0]; dev = self.device
        h0 = torch.zeros(B, m.cfg.gru_hidden, device=dev)
        z, _ = m.posterior(x_b, h0)
        if hasattr(m.decoder, 'head_reward'):
            rl = m.decoder.head_reward(z).squeeze(-1)
        elif hasattr(m, 'reward_head'):
            rl = m.reward_head(z).squeeze(-1)
        else: m.eval(); return 0.0
        loss = F.binary_cross_entropy_with_logits(
            rl, r_env_b.clamp(0,1).float())
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 1.0)
        self.opt.step(); m.eval()
        return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# EnvContext
# ─────────────────────────────────────────────────────────────────────────────

class EnvContext:
    def __init__(self, model, device, cond_len=16):
        self.model=model; self.device=device; self.cond_len=cond_len
        self.obs_buf=[]; self.act_buf=[]
        self.z_t=None; self.h_t=None; self._ref=None

    def reset(self, obs):
        self.obs_buf=[obs]; self.act_buf=[]
        self.z_t=None; self.h_t=None; self._ref=obs.copy()

    def _obs_to_x(self, obs):
        x = np.zeros(2108, dtype=np.float32)
        x[X_DP_START:X_DP_END] = (obs[18:60]-self._ref[18:60]).astype(np.float32)
        x[X_DQ_START:X_DQ_END] = (obs[0:9]  -self._ref[0:9]).astype(np.float32)
        x[X_QD_START:X_QD_END] = obs[9:18].astype(np.float32)
        return x

    def step(self, obs, action):
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
            self.z_t = enc['o_seq'][0,-1:]
            self.h_t = enc['h_seq'][0,-1:]


# ─────────────────────────────────────────────────────────────────────────────
# LQR Warmup
# ─────────────────────────────────────────────────────────────────────────────

class LQRWarmup:
    def __init__(self, model, planner, cfg, device):
        self.model=model; self.planner=planner; self.cfg=cfg; self.device=device
        self.pz=[]; self.psp=[]; self.pa=[]

    @torch.no_grad()
    def step_episode(self, env, ctx, buf, n_skills, H_lo, obs):
        dev = torch.device(self.device)
        m   = self.model
        if ctx.z_t is None or ctx.h_t is None:
            a   = env.action_space.sample()
            obs, r, done, _ = env.step(a)
            ctx.step(obs, a)
            return obs, r, done

        z_t=ctx.z_t; h_t=ctx.h_t
        w_t = m.skill_prior.soft_weights(h_t)
        sp  = w_t.cpu().numpy()[0]

        cond_len = min(len(ctx.act_buf), self.cfg.cond_len)
        a_seq_np = np.zeros((H_lo, m.cfg.action_dim), dtype=np.float32)
        if cond_len >= 1:
            xw = np.array([ctx._obs_to_x(o) for o in ctx.obs_buf[-cond_len-1:-1]])
            aw = np.array(ctx.act_buf[-cond_len:])
            xc = torch.FloatTensor(xw).unsqueeze(0).to(dev)
            ac = torch.FloatTensor(aw).unsqueeze(0).to(dev)
            # goal: Koopman world model로 H_lo step 자연 dynamics 예측
            # self-goal(제자리) 대신 skill prior 방향으로 이동하는 goal 사용
            try:
                with torch.no_grad():
                    enc_tmp = m.encode_sequence(xc, ac)
                    z_tmp = enc_tmp['o_seq'][0, -1:]
                    h_tmp = enc_tmp['h_seq'][0, -1:]
                    w_tmp = m.skill_prior.soft_weights(h_tmp)
                    ll = m.koopman.get_log_lambdas()
                    A_bar, B_bar, _, _ = blend_koopman(
                        ll, m.koopman.theta_k, m.koopman.G_k,
                        m.koopman.U, w_tmp)
                    A_b = A_bar[0]; B_b = B_bar[0]
                    u_zero = torch.zeros(1, m.cfg.action_latent, device=dev)
                    z_pred = z_tmp
                    for _ in range(H_lo):
                        z_pred = (A_b @ z_pred.T).T + (B_b @ u_zero.T).T
                    recon_g = m.decoder(z_pred)
                    xg = torch.cat([
                        torch.zeros(1, 2048, device=dev),
                        symexp(recon_g['delta_p']),
                        symexp(recon_g['q']),
                        symexp(recon_g['qdot']),
                    ], dim=-1)  # (1, 2108)
                plan = self.planner.plan(xc, ac, xg,
                                         horizon=H_lo, compute_uncertainty=False)
                u_lqr = plan['u_traj'].to(dev)
                a_seq_np = self._decode(u_lqr)
            except Exception:
                a_seq_np = np.stack([env.action_space.sample()
                                     for _ in range(H_lo)])

        z_b = z_t.clone(); h_b = h_t.clone()
        r_tot=0.0; done=False
        for k in range(H_lo):
            obs_nx, r, done, _ = env.step(a_seq_np[k].clip(-1,1))
            r_tot += r; ctx.step(obs_nx, a_seq_np[k]); obs=obs_nx
            if done: break

        if ctx.z_t is not None:
            buf.add(z_b.cpu().numpy()[0], ctx.z_t.cpu().numpy()[0],
                    h_b.cpu().numpy()[0], sp, a_seq_np,
                    np.clip(r_tot,-1,1), float(done))
            self.pz.append(z_b.cpu().numpy()[0])
            self.psp.append(sp)
            self.pa.append(a_seq_np.copy())
        return obs, r_tot, done

    def _decode(self, u_lqr):
        H = u_lqr.shape[0]
        a = torch.zeros(H, self.model.cfg.action_dim,
                        device=u_lqr.device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(40):
            opt.zero_grad()
            F.mse_loss(self.model.action_encoder(a),
                       u_lqr.detach()).backward()
            opt.step()
            with torch.no_grad(): a.clamp_(-1,1)
        return a.detach().cpu().numpy()

    def get_pretrain_data(self):
        if not self.pz: return None
        dev = torch.device(self.device)
        return (torch.FloatTensor(np.array(self.pz)).to(dev),
                torch.FloatTensor(np.array(self.psp)).to(dev),
                torch.FloatTensor(np.array(self.pa)).to(dev))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class KODAQOnlineTrainer:
    def __init__(self, cfg, wm, z_dim, n_skills, action_dim, device):
        self.cfg=cfg; self.wm=wm; self.device=device
        self.z_dim=z_dim; self.n_skills=n_skills; self.action_dim=action_dim
        self.gumbel_tau=cfg.gumbel_tau; self.step=0

        h=cfg.hidden_dim; nl=cfg.n_layers
        self.pi_hi = HighLevelPolicy(z_dim, n_skills, h, nl).to(device)
        self.pi_lo = LowLevelPolicy(z_dim, n_skills, action_dim,
                                    cfg.H_lo, h, nl).to(device)
        self.Q1    = QNetwork(z_dim, n_skills, action_dim, cfg.H_lo, h, nl).to(device)
        self.Q2    = QNetwork(z_dim, n_skills, action_dim, cfg.H_lo, h, nl).to(device)
        self.Q1_t  = copy.deepcopy(self.Q1)
        self.Q2_t  = copy.deepcopy(self.Q2)

        lr = cfg.lr
        self.opt_hi = torch.optim.Adam(self.pi_hi.parameters(), lr=lr)
        self.opt_lo = torch.optim.Adam(self.pi_lo.parameters(), lr=lr)
        self.opt_q  = torch.optim.Adam(
            list(self.Q1.parameters())+list(self.Q2.parameters()), lr=lr)

        # α 고정값: entropy regularization penalization
        # auto-tuning 없음 — target entropy 개념 없음
        self.buf = ReplayBuffer(cfg.buffer_size, device)

    @property
    def alpha(self):
        return self.cfg.alpha

    def _anneal_tau(self):
        frac = min(1.0, self.step / max(self.cfg.n_env_steps, 1))
        self.gumbel_tau = (self.cfg.gumbel_tau -
            (self.cfg.gumbel_tau - self.cfg.gumbel_tau_min) * frac)

    def update(self):
        if self.buf.size < self.cfg.batch_size: return {}
        b    = self.buf.sample(self.cfg.batch_size)
        z    = b['z']; z_nx = b['z_next']; h_t = b['h_t']
        sp   = b['sp']; a = b['a_seq']; r = b['r']; done = b['done']

        # Critic
        with torch.no_grad():
            sp_nx, lp_hi_nx = self.pi_hi.gumbel_sample(z_nx, self.gumbel_tau)
            a_nx,  lp_lo_nx = self.pi_lo.sample(z_nx, sp_nx)
            lp_nx  = lp_hi_nx + lp_lo_nx
            V_next = (torch.min(self.Q1_t(z_nx,sp_nx,a_nx),
                                self.Q2_t(z_nx,sp_nx,a_nx))
                      - self.alpha * lp_nx)
        r_norm = (r - r.mean()) / r.std().clamp(min=1e-6)
        y = (r_norm + self.cfg.gamma*(1-done)*V_next).clamp(-50,50)
        q1 = self.Q1(z,sp,a); q2 = self.Q2(z,sp,a)
        loss_q = F.mse_loss(q1,y) + F.mse_loss(q2,y)
        self.opt_q.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters())+list(self.Q2.parameters()),
            self.cfg.grad_clip)
        self.opt_q.step()

        # Hi actor: KL(π_hi || p_skill) - Q  (α=0, entropy항 없음)
        # Q는 detach → critic gradient와 actor gradient 분리
        with torch.no_grad():
            p_prior = self.wm.model.skill_prior.soft_weights(h_t)
        sp_new, _ = self.pi_hi.gumbel_sample(z, self.gumbel_tau)
        # lo sample detach → hi actor는 skill selection에만 집중
        with torch.no_grad():
            a_hi, _ = self.pi_lo.sample(z, sp_new.detach())
        q_hi = torch.min(self.Q1(z, sp_new, a_hi.detach()),
                         self.Q2(z, sp_new, a_hi.detach()))
        kl_hi2 = self.pi_hi.kl_prior(z, p_prior)
        loss_hi = (self.cfg.kl_weight * kl_hi2 - q_hi).mean()
        self.opt_hi.zero_grad(); loss_hi.backward()
        nn.utils.clip_grad_norm_(self.pi_hi.parameters(), self.cfg.grad_clip)
        self.opt_hi.step()

        # Lo actor: -Q only  (α=0, entropy항 없음)
        # Q는 gradient 통과 (lo policy만 업데이트)
        sp_d = sp_new.detach()
        a_lo, _ = self.pi_lo.sample(z, sp_d)
        q_lo = torch.min(self.Q1(z, sp_d, a_lo),
                         self.Q2(z, sp_d, a_lo))
        loss_lo = -q_lo.mean()
        self.opt_lo.zero_grad(); loss_lo.backward()
        nn.utils.clip_grad_norm_(self.pi_lo.parameters(), self.cfg.grad_clip)
        self.opt_lo.step()

        # entropy logging only (gradient 없음)
        with torch.no_grad():
            ent = (self.pi_hi.entropy(z) + self.pi_lo.entropy(z, sp_d)).mean()
        q_hi2 = q_hi

        # Soft update
        for p,pt in zip(self.Q1.parameters(), self.Q1_t.parameters()):
            pt.data.mul_(1-self.cfg.tau_ema).add_(p.data, alpha=self.cfg.tau_ema)
        for p,pt in zip(self.Q2.parameters(), self.Q2_t.parameters()):
            pt.data.mul_(1-self.cfg.tau_ema).add_(p.data, alpha=self.cfg.tau_ema)
        self._anneal_tau()

        return {'loss_q': loss_q.item(), 'loss_hi': loss_hi.item(),
                'loss_lo': loss_lo.item(), 'kl_hi': kl_hi2.mean().item(),
                'alpha': self.cfg.alpha, 'ent': ent.item(),
                'q_mean': q_hi2.mean().item(), 'gumbel_tau': self.gumbel_tau}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            # network weights
            'step':        self.step,
            'pi_hi':       self.pi_hi.state_dict(),
            'pi_lo':       self.pi_lo.state_dict(),
            'Q1':          self.Q1.state_dict(),
            'Q2':          self.Q2.state_dict(),
            'Q1_t':        self.Q1_t.state_dict(),
            'Q2_t':        self.Q2_t.state_dict(),
            'world_model': self.wm.model.state_dict(),
            # optimizer states (Adam momentum/variance 유지)
            'opt_hi':      self.opt_hi.state_dict(),
            'opt_lo':      self.opt_lo.state_dict(),
            'opt_q':       self.opt_q.state_dict(),
            'wm_opt':      self.wm.opt.state_dict(),
            # misc
            'gumbel_tau':  self.gumbel_tau,
        }, path)
        print(f"  Saved: {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.pi_hi.load_state_dict(ck['pi_hi'])
        self.pi_lo.load_state_dict(ck['pi_lo'])
        self.Q1.load_state_dict(ck['Q1'])
        self.Q2.load_state_dict(ck['Q2'])
        self.Q1_t.load_state_dict(ck.get('Q1_t', ck['Q1']))
        self.Q2_t.load_state_dict(ck.get('Q2_t', ck['Q2']))
        if 'world_model' in ck:
            self.wm.model.load_state_dict(ck['world_model'])
        # optimizer states 복원 (없으면 무시 — 구버전 호환)
        if 'opt_hi' in ck:
            self.opt_hi.load_state_dict(ck['opt_hi'])
            self.opt_lo.load_state_dict(ck['opt_lo'])
            self.opt_q.load_state_dict(ck['opt_q'])
        if 'wm_opt' in ck:
            self.wm.opt.load_state_dict(ck['wm_opt'])
        self.gumbel_tau = ck.get('gumbel_tau', self.cfg.gumbel_tau)
        self.step = ck.get('step', 0)
        print(f"Loaded: {path}  step={self.step}  gumbel_tau={self.gumbel_tau:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Reward blend
# ─────────────────────────────────────────────────────────────────────────────

def compute_r_blend(r_env, r_hat, step, warmup):
    # 초반: world model reward 비중 높음 (sparse r_env 보완)
    # 후반: r_env 비중 높아짐 (world model fine-tune 후 r_hat도 calibrated)
    alpha = min(1.0, step / max(warmup, 1))
    return float(np.clip(alpha * r_env + (1.0 - alpha) * r_hat, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(trainer, wm, env_name, n_ep, cfg, device):
    import gym, d4rl
    dev = torch.device(device); model = wm.model; model.eval()
    results = []
    for _ in range(n_ep):
        env = gym.make(env_name); obs = env.reset()
        ctx = EnvContext(model, device, cfg.cond_len); ctx.reset(obs)
        total_r=0.0; n_tasks=0; done=False; hi_timer=0; sid=0
        for t in range(280):
            if ctx.z_t is None:
                obs,r,done,info = env.step(env.action_space.sample())
                ctx.step(obs, env.action_space.sample()); total_r+=r
                if done: break; continue
            if hi_timer==0:
                sid,_ = trainer.pi_hi.hard_sample(ctx.z_t); hi_timer=cfg.H_hi
            sp = torch.zeros(1, trainer.n_skills, device=dev); sp[0,sid]=1.0
            with torch.no_grad():
                a_seq,_ = trainer.pi_lo.sample(ctx.z_t, sp)
            for k in range(cfg.H_lo):
                if done: break
                obs,r,done,info = env.step(a_seq[0,k].cpu().numpy().clip(-1,1))
                ctx.step(obs, a_seq[0,k].cpu().numpy()); total_r+=r
                n_tasks=max(n_tasks, int(info.get('num_success',0)))
            hi_timer=max(0, hi_timer-cfg.H_lo)
        results.append({'reward':total_r,'n_tasks':n_tasks}); env.close()
    mr=np.mean([x['reward'] for x in results])
    mt=np.mean([x['n_tasks'] for x in results])
    print(f"  [EVAL] mean_reward={mr:.3f}  mean_tasks={mt:.2f}")
    return {'eval_reward': mr, 'eval_tasks': mt}


# ─────────────────────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────────────────────

def visualize_training(log, out_path):
    keys   = ['loss_q','loss_hi','loss_lo','kl_hi','alpha','ent','q_mean','ep_reward','ep_tasks']
    titles = ['Q Loss','π Hi Loss','π Lo Loss','KL Hi','α','Entropy','Q mean','Episode Reward','Tasks']
    PAL    = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300','#607D8B','#795548']
    fig, axes = plt.subplots(3, 3, figsize=(18, 12)); axes=axes.flatten()
    for i,(k,t) in enumerate(zip(keys,titles)):
        if k not in log or not log[k]: continue
        vals=np.array(log[k]); ax=axes[i]
        ax.plot(vals, color=PAL[i], alpha=0.25, lw=0.8)
        w=max(1,min(50,len(vals)//5))
        if len(vals)>=w:
            ax.plot(np.convolve(vals,np.ones(w)/w,'valid'), color=PAL[i], lw=1.8)
        ax.set_title(t, fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('KODAQ-Online v2', fontsize=12, fontweight='bold')
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, trainer, wm, planner, env_name, out_dir, device, use_wandb=False):
    import gym, d4rl
    dev=torch.device(device); model=wm.model
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log_keys=['loss_q','loss_hi','loss_lo','kl_hi','alpha','ent',
              'q_mean','wm_loss','ep_reward','ep_tasks']
    log    = {k:[] for k in log_keys}
    recent = {k: deque(maxlen=cfg.log_every) for k in log_keys}

    env=gym.make(env_name); obs=env.reset()
    ctx=EnvContext(model,device,cfg.cond_len); ctx.reset(obs)
    lqr_wu=LQRWarmup(model,planner,cfg,device)

    ep_r=0.0; ep_tasks=0; sid=0; hi_timer=0
    global_step=trainer.step; lqr_pretrained=False; t0=time.time()
    wm_obs_buf=[]; wm_r_buf=[]

    print(f"\n{'='*60}")
    print(f"KODAQ-Online v2  steps={cfg.n_env_steps}  H_hi={cfg.H_hi}  H_lo={cfg.H_lo}")
    print(f"  warmup_lqr={cfg.warmup_lqr}  env={env_name}")
    print(f"{'='*60}\n")

    while global_step < cfg.n_env_steps:

        # ── LQR Warmup ─────────────────────────────────────────────────────
        if global_step < cfg.warmup_lqr:
            obs, r_ep, done = lqr_wu.step_episode(
                env, ctx, trainer.buf, trainer.n_skills, cfg.H_lo, obs)
            ep_r += r_ep; global_step += cfg.H_lo
            x_np = ctx._obs_to_x(obs)
            wm_obs_buf.append(x_np); wm_r_buf.append(float(r_ep))
            if len(wm_obs_buf)>512: wm_obs_buf.pop(0); wm_r_buf.pop(0)
            if done:
                recent['ep_reward'].append(ep_r); recent['ep_tasks'].append(ep_tasks)
                obs=env.reset(); ctx.reset(obs); ep_r=0.0; ep_tasks=0
            # π_lo pretrain after warmup
            if global_step>=cfg.warmup_lqr and not lqr_pretrained:
                data=lqr_wu.get_pretrain_data()
                if data:
                    z_pt,sp_pt,a_pt=data
                    l=trainer.pi_lo.pretrain_lqr(z_pt,sp_pt,a_pt)
                    print(f"\n[LQR Pretrain] π_lo loss={l:.4f}  buf={trainer.buf.size}")
                    if use_wandb: wandb.log({'pretrain/lqr_loss':l}, step=global_step)
                lqr_pretrained=True
            continue

        # ── Normal rollout ─────────────────────────────────────────────────
        if ctx.z_t is None:
            obs,r,done,_ = env.step(env.action_space.sample())
            ctx.step(obs, env.action_space.sample())
            global_step+=1; ep_r+=r
            if done:
                recent['ep_reward'].append(ep_r); recent['ep_tasks'].append(ep_tasks)
                obs=env.reset(); ctx.reset(obs); ep_r=0.0; ep_tasks=0
            continue

        z_t=ctx.z_t; h_t=ctx.h_t
        if hi_timer==0:
            sid,_=trainer.pi_hi.hard_sample(z_t); hi_timer=cfg.H_hi
        sp=torch.zeros(1,trainer.n_skills,device=dev); sp[0,sid]=1.0
        with torch.no_grad(): a_seq,_=trainer.pi_lo.sample(z_t,sp)
        a_np=a_seq[0].cpu().numpy()

        z_b=z_t.clone(); h_b=h_t.clone()
        r_env_tot=0.0; r_hat_tot=0.0; done=False
        a_pad=np.zeros((cfg.H_lo,trainer.action_dim),dtype=np.float32)
        for k in range(cfg.H_lo):
            ak=a_np[k].clip(-1,1)
            obs_nx,r_env,done,info=env.step(ak)
            r_env_tot+=r_env; ep_r+=r_env
            ep_tasks=max(ep_tasks,int(info.get('num_success',0)))
            global_step+=1; a_pad[k]=ak
            xnp=ctx._obs_to_x(obs_nx)
            wm_obs_buf.append(xnp); wm_r_buf.append(float(r_env))
            if len(wm_obs_buf)>512: wm_obs_buf.pop(0); wm_r_buf.pop(0)
            if ctx.z_t is not None:
                with torch.no_grad():
                    at=torch.FloatTensor(ak).unsqueeze(0).to(dev)
                    _,_,rh=wm.koopman_step(ctx.z_t,at,ctx.h_t)
                    r_hat_tot+=rh
            ctx.step(obs_nx,ak); obs=obs_nx
            if done: break

        r_blend=compute_r_blend(r_env_tot,r_hat_tot,global_step,cfg.r_alpha_warmup)
        sp_np=sp.cpu().numpy()[0]
        if ctx.z_t is not None:
            trainer.buf.add(z_b.cpu().numpy()[0],ctx.z_t.cpu().numpy()[0],
                            h_b.cpu().numpy()[0],sp_np,a_pad,r_blend,float(done))
        hi_timer=max(0,hi_timer-cfg.H_lo)

        # Updates
        for _ in range(cfg.n_updates_per_step):
            info_d=trainer.update()
            for k,v in info_d.items():
                if k in recent: recent[k].append(v)

        if global_step%cfg.wm_update_freq==0 and len(wm_obs_buf)>=32:
            idx=np.random.choice(len(wm_obs_buf),min(32,len(wm_obs_buf)),replace=False)
            xb=torch.FloatTensor(np.array([wm_obs_buf[i] for i in idx])).to(dev)
            rb=torch.FloatTensor(np.array([wm_r_buf[i]   for i in idx])).to(dev)
            recent['wm_loss'].append(wm.update(xb,rb))

        if done:
            recent['ep_reward'].append(ep_r); recent['ep_tasks'].append(ep_tasks)
            obs=env.reset(); ctx.reset(obs); ep_r=0.0; ep_tasks=0; hi_timer=0

        # Logging
        if global_step%cfg.log_every==0:
            ms={k: np.mean(list(recent[k])) if recent[k] else 0.0 for k in recent}
            for k in log_keys:
                if k in ms: log[k].append(ms[k])
            sps=cfg.log_every/(time.time()-t0+1e-6); t0=time.time()
            trainer.step=global_step
            print(f"Step {global_step:7d} | "
                  f"Q={ms['loss_q']:.3f} Hi={ms['loss_hi']:.3f} "
                  f"Lo={ms['loss_lo']:.3f} KL={ms['kl_hi']:.3f} | "
                  f"α={ms['alpha']:.4f} ent={ms['ent']:.2f} | "
                  f"ep_r={ms['ep_reward']:.3f} tasks={ms['ep_tasks']:.2f} | "
                  f"{sps:.0f}sps")
            if use_wandb:
                wandb.log({f"train/{k}":v for k,v in ms.items()}, step=global_step)

        if global_step%cfg.save_every==0:
            trainer.save(f"{out_dir}/kodaq_online_step{global_step}.pt")
            visualize_training(log, f"{out_dir}/training_curves.png")

        if global_step%cfg.eval_every==0:
            er=evaluate(trainer,wm,env_name,cfg.n_eval_ep,cfg,device)
            for k,v in er.items(): log.setdefault(k,[]).append(v)
            if use_wandb: wandb.log({f"eval/{k}":v for k,v in er.items()}, step=global_step)

    trainer.save(f"{out_dir}/kodaq_online_final.pt")
    visualize_training(log, f"{out_dir}/training_curves_final.png")
    env.close(); print(f"\nDone. {global_step} steps → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_goal_obs(env_name):
    goal=np.zeros(60,dtype=np.float32)
    for task,idx in OBS_ELEMENT_INDICES.items():
        goal[idx]=OBS_ELEMENT_GOALS[task]
    return goal


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--world_ckpt', default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--resume',     default=None)
    p.add_argument('--env',        default='kitchen-mixed-v0')
    p.add_argument('--out_dir',    default='checkpoints/kodaq_v4/online')
    p.add_argument('--H_hi',       type=int,   default=8)
    p.add_argument('--H_lo',       type=int,   default=4)
    p.add_argument('--gamma',      type=float, default=0.99)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--kl_weight',  type=float, default=1.0)
    p.add_argument('--n_steps',    type=int,   default=300_000)
    p.add_argument('--warmup_lqr', type=int,   default=3_000)
    p.add_argument('--r_warmup',   type=int,   default=30_000)
    p.add_argument('--wm_lr',      type=float, default=1e-4)
    p.add_argument('--device',     default='cuda:1' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--wandb_run',  default=None)
    p.add_argument('--Q_scale',    type=float, default=1.0)
    p.add_argument('--R_scale',    type=float, default=10.0)
    args=p.parse_args()

    device=args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    use_wandb=(args.wandb_project is not None and WANDB_AVAILABLE)
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run or 'kodaq_online_v2',
                   config=vars(args))

    ckpt=torch.load(args.world_ckpt, map_location=device)
    model=KoopmanCVAE(ckpt['cfg']); model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    z_dim=model.cfg.koopman_dim; n_skills=model.cfg.num_skills; a_dim=model.cfg.action_dim
    print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")

    cfg=OnlineConfig(H_hi=args.H_hi, H_lo=args.H_lo, gamma=args.gamma,
                     lr=args.lr, batch_size=args.batch_size, kl_weight=args.kl_weight,
                     n_env_steps=args.n_steps, warmup_lqr=args.warmup_lqr,
                     r_alpha_warmup=args.r_warmup, wm_lr=args.wm_lr)
    wm      = KoopmanWorldModelWrapper(model, cfg.wm_lr, device)
    trainer = KODAQOnlineTrainer(cfg, wm, z_dim, n_skills, a_dim, device)
    planner = KODAQLQRPlanner(model, LQRConfig(Q_scale=args.Q_scale, R_scale=args.R_scale))

    if args.resume and Path(args.resume).exists():
        trainer.load(args.resume)

    train(cfg, trainer, wm, planner, args.env, args.out_dir, device, use_wandb)
    if use_wandb: wandb.finish()


if __name__=='__main__':
    main()