"""
skill_pretrain.py — TCN-DPM Skill Pretraining for KODAQ
=========================================================

HELIOS 논문의 구조를 TCN encoder로 대체하여 구현.
GRU encoder를 제외하면 논문의 variational posterior + MemoVB Birth/Merge 그대로.

Generative model (DPM):
    beta_k   ~ Beta(1, alpha)                   [GEM: stick-breaking]
    pi_k     = beta_k * prod_{j<k}(1 - beta_j)  [mixing weight]
    mu_k, Lambda_k ~ NIW(mu0, kappa0, nu0, Psi0) [component params]
    c_n      ~ Cat(pi)                           [cluster assignment]
    z_n      ~ N(mu_{c_n}, Lambda_{c_n}^{-1})   [observation]
    z_n       = TCNEncoder(s_{1:t}, a_{1:t})     [from data]

Variational posterior q (mean field):
    q(c_n)    = Cat(r_hat_n)                     [responsibilities, (N,K)]
    q(beta_k) = Beta(a_k1, a_k0)                [stick-breaking posterior]
    q(theta_k) = NIW(mu_hat_k, kappa_hat_k, nu_hat_k, Psi_hat_k)

CAVI update order:
    1. E-step: r_hat_nk  (local,  per datapoint)
    2. M-step: a_k1, a_k0, mu_hat_k, kappa_hat_k, nu_hat_k, Psi_hat_k (global)
    3. Birth:  ELBO 비교로 새 component 추가 여부 결정
    4. Merge:  ELBO 비교로 유사 component 합병 여부 결정

Alternating training:
    Step A (encoder frozen): collect z = TCN_mu(s, a) → fit DPM
    Step B (DPM frozen):     update encoder/decoder/prior with
        L = zeta1*L_rec + zeta2*L_dpm + zeta3*L_prior

Output:
    skill_labels[b, t] = argmax_k r_hat_{bt, k}
    → Koopman CVAE의 skill-conditioned (A_k, B_k) 학습에 사용
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.special import digamma, gammaln

# ── 기존 코드 재사용 ──────────────────────────────────────────
from models.koopman_cvae import CausalConv1d, symlog


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class SkillPretrainConfig:
    # Kitchen environment
    state_dim:  int   = 60
    action_dim: int   = 9

    # ── Encoder type ──────────────────────────────────────────
    # HELIOS 원본: 'gru', action-only
    # 대안:       'tcn', state_action (더 빠름)
    encoder_type:  str  = 'gru'       # 'gru' | 'tcn'
    encoder_input: str  = 'action'    # 'action' (HELIOS) | 'state_action'

    # GRU parameters (HELIOS 원본 구조)
    gru_hidden: int   = 128    # SPiRL GRU hidden=128
    gru_layers: int   = 2      # bidirectional=False (causal)

    # TCN parameters (encoder_type='tcn'일 때)
    tcn_hidden:  int  = 128
    tcn_layers:  int  = 5      # RF=63 steps
    tcn_kernel:  int  = 3
    dropout:     float= 0.1

    # Skill latent dimension
    skill_dim:   int  = 32

    # Skill decoder horizon (SPiRL: n_rollout_steps=10)
    skill_horizon: int = 10

    # DPM hyperparameters
    alpha:   float = 1.0       # GEM concentration
    K_init:  int   = 1         # 초기 component 수
    K_max:   int   = 20        # truncation upper bound

    # NIW prior  (mu0=0, Psi0=psi_scale*I)
    kappa0:    float = 1.0
    nu0_delta: float = 2.0     # nu0 = skill_dim + nu0_delta  (> d-1 필요)
    psi_scale: float = 0.1   # small initial covariance → more sensitive birth

    # Birth/Merge
    birth_thresh:  float = 0.3    # max_r < thresh → poorly explained (K>1 only)
    birth_min_pts: int   = 10
    birth_K_fresh: int   = 4      # Hughes&Sudderth: K_fresh sub-clusters per birth
    birth_start_epoch: int = 3    # epoch < this: birth disabled (encoder not ready)
    merge_cos:     float = 0.90   # cosine similarity threshold for merge

    # Loss weights (HELIOS Eq.7)
    # zeta1 dominant initially so encoder learns from reconstruction first
    zeta1: float = 1.0    # reconstruction
    zeta2: float = 0.1    # DPM alignment (reduced: early DPM is noisy)
    zeta3: float = 0.01   # prior alignment

    # Training
    epochs:    int   = 100
    batch_size:int   = 64
    lr:        float = 3e-4
    device:    str   = 'cuda'
    # Anti-collapse: z spread regularization
    zeta_spread: float = 0.5    # batch variance lower bound weight
    zeta_vae:    float = 0.1    # KL(q(z|a) || N(0,I)) weight
    min_z_std:   float = 0.3    # target minimum std per z dimension

    # Post-birth warm-up steps
    birth_warmup_steps: int = 5

    save_dir:  str   = 'checkpoints/skill_pretrain'


# ─────────────────────────────────────────────────────────────
# GRU Skill Encoder  (HELIOS 원본 구조)
# ─────────────────────────────────────────────────────────────

class GRUSkillEncoder(nn.Module):
    """
    HELIOS 논문 원본: q(z_skill | a_{1:L})
    
    핵심: action-only input.
    Skill은 state와 무관한 action pattern으로 정의됨.
    (state 넣으면 DPM이 skill cluster 대신 state cluster를 학습)

    GRU hidden state가 action sequence의 temporal pattern을 누적.
    마지막 hidden state가 전체 skill을 요약.
    """

    def __init__(self, cfg: SkillPretrainConfig):
        super().__init__()
        in_dim = cfg.action_dim if cfg.encoder_input == 'action'                  else cfg.state_dim + cfg.action_dim
        d_z    = cfg.skill_dim

        self.encoder_input = cfg.encoder_input
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.fc_mu     = nn.Linear(cfg.gru_hidden, d_z)
        self.fc_logvar = nn.Linear(cfg.gru_hidden, d_z)

    def _get_input(self, states, actions):
        if self.encoder_input == 'action':
            return symlog(actions)                     # (B, T, da)
        else:
            return torch.cat([symlog(states),
                               symlog(actions)], dim=-1)

    def forward(self, states, actions):
        """
        Returns z, mu, logvar: (B, T, d_z)

        각 timestep t의 출력 = a_{1:t}까지 본 GRU hidden state 기반.
        DPM fitting에는 마지막 timestep (전체 sequence summary)을 사용.
        """
        x, _ = self.gru(self._get_input(states, actions))  # (B,T,h)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-4.0, 2.0)
        std    = (0.5 * logvar).exp()
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @torch.no_grad()
    def encode_mu(self, states, actions):
        """
        DPM fitting용: 마지막 timestep mu 반환 (noise 없음).
        seq_len=10이면 10-step 전체를 본 summary.
        Returns (B, d_z) — timestep 차원 없음.
        """
        x, _ = self.gru(self._get_input(states, actions))   # (B,T,h)
        mu    = self.fc_mu(x[:, -1, :])                      # (B, d_z)
        return mu


# ─────────────────────────────────────────────────────────────
# TCN Skill Encoder  (빠른 대안)
# ─────────────────────────────────────────────────────────────

class TCNSkillEncoder(nn.Module):
    """
    Causal TCN: 병렬화 가능, GRU보다 3-5x 빠름.
    encoder_input='action'이면 HELIOS와 동일한 semantics.
    """

    def __init__(self, cfg: SkillPretrainConfig):
        super().__init__()
        in_dim = cfg.action_dim if cfg.encoder_input == 'action'                  else cfg.state_dim + cfg.action_dim
        d_c    = cfg.tcn_hidden
        d_z    = cfg.skill_dim

        self.encoder_input = cfg.encoder_input
        self.input_proj = nn.Conv1d(in_dim, d_c, kernel_size=1)
        self.layers = nn.ModuleList([
            CausalConv1d(d_c, d_c,
                         kernel_size=cfg.tcn_kernel,
                         dilation=2 ** i,
                         dropout=cfg.dropout)
            for i in range(cfg.tcn_layers)
        ])
        self.fc_mu     = nn.Linear(d_c, d_z)
        self.fc_logvar = nn.Linear(d_c, d_z)

    def _get_input(self, states, actions):
        if self.encoder_input == 'action':
            x = symlog(actions).transpose(1, 2)                    # (B,da,T)
        else:
            x = torch.cat([symlog(states), symlog(actions)], -1
                          ).transpose(1, 2)                        # (B,ds+da,T)
        return x

    def forward(self, states, actions):
        x = F.silu(self.input_proj(self._get_input(states, actions)))
        for layer in self.layers:
            x = x + layer(x)
        c = x.transpose(1, 2)                                      # (B,T,d_c)
        mu     = self.fc_mu(c)
        logvar = self.fc_logvar(c).clamp(-4.0, 2.0)
        std    = (0.5 * logvar).exp()
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @torch.no_grad()
    def encode_mu(self, states, actions):
        """Returns (B, d_z) — last timestep summary."""
        _, mu, _ = self.forward(states, actions)
        return mu[:, -1, :]


# ── Factory: cfg.encoder_type에 따라 선택 ────────────────────

def build_encoder(cfg: SkillPretrainConfig) -> nn.Module:
    if cfg.encoder_type == 'gru':
        return GRUSkillEncoder(cfg)
    elif cfg.encoder_type == 'tcn':
        return TCNSkillEncoder(cfg)
    else:
        raise ValueError(f"encoder_type must be 'gru' or 'tcn', got {cfg.encoder_type!r}")


# ── SkillEncoder = alias for backward compat ─────────────────
SkillEncoder = GRUSkillEncoder


# ─────────────────────────────────────────────────────────────
# Skill Decoder  p(a_{t:t+L} | z, s_t)
# ─────────────────────────────────────────────────────────────

class SkillDecoder(nn.Module):
    """
    skill latent z + 초기 state s_t → action sequence 재구성.
    reconstruction loss L_rec 계산에 사용.
    """

    def __init__(self, cfg: SkillPretrainConfig):
        super().__init__()
        d_z = cfg.skill_dim
        d_s = cfg.state_dim
        d_a = cfg.action_dim
        d_h = cfg.tcn_hidden
        L   = cfg.skill_horizon

        self.L          = L
        self.input_proj = nn.Linear(d_z + d_s, d_h)
        # time index를 추가해서 각 step을 구분
        self.mlp = nn.Sequential(
            nn.Linear(d_h + 1, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h),     nn.SiLU(),
            nn.Linear(d_h, d_a),
        )

    def forward(
        self,
        z:  torch.Tensor,   # (B, d_z)  skill window의 마지막 timestep
        s0: torch.Tensor,   # (B, d_s)  skill window의 초기 state
    ) -> torch.Tensor:
        """Returns a_hat: (B, L, da)"""
        B   = z.shape[0]
        h   = F.silu(self.input_proj(
                torch.cat([z, symlog(s0)], dim=-1)))   # (B, d_h)
        h   = h.unsqueeze(1).expand(-1, self.L, -1)    # (B, L, d_h)
        t   = torch.linspace(0, 1, self.L,
                             device=z.device
                             ).unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        return self.mlp(torch.cat([h, t], dim=-1))      # (B, L, da)


# ─────────────────────────────────────────────────────────────
# Skill Prior  p_θ(z_skill | s_t)
# ─────────────────────────────────────────────────────────────

class SkillPrior(nn.Module):
    """
    State-conditioned skill prior.
    Phase II downstream RL에서 SAC entropy term 대체로 사용.
    """

    def __init__(self, cfg: SkillPretrainConfig):
        super().__init__()
        d_h = cfg.tcn_hidden
        d_z = cfg.skill_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h),           nn.SiLU(),
        )
        self.fc_mu     = nn.Linear(d_h, d_z)
        self.fc_logvar = nn.Linear(d_h, d_z)

    def forward(self, s: torch.Tensor):
        """s: (B, ds) → mu, logvar: (B, d_z)"""
        h = self.net(symlog(s))
        return (self.fc_mu(h),
                self.fc_logvar(h).clamp(-4.0, 2.0))


# ─────────────────────────────────────────────────────────────
# DPM: Dirichlet Process Mixture with MemoVB + Birth/Merge
# ─────────────────────────────────────────────────────────────

class DPM:
    """
    논문의 variational posterior 구조와 MemoVB 알고리즘 그대로 구현.
    CAVI로 각 variational factor를 순차 업데이트.

    Variational factors (mean field 가정):
        q(c_n)    = Cat(r_hat_n)                  [local,  (N,K)]
        q(beta_k) = Beta(a_k1[k], a_k0[k])       [global, (K,)]
        q(theta_k) = NIW(mu_hat, kappa_hat,
                         nu_hat, Psi_hat)          [global, (K,...)]

    모든 연산은 numpy (CPU) — epoch 사이 offline fitting.
    """

    def __init__(self, cfg: SkillPretrainConfig):
        self.cfg   = cfg
        self.d     = cfg.skill_dim
        self.alpha = cfg.alpha

        # NIW prior hyperparameters
        self.mu0    = np.zeros(self.d)
        self.kappa0 = cfg.kappa0
        self.nu0    = self.d + cfg.nu0_delta   # > d-1 보장
        self.Psi0   = cfg.psi_scale * np.eye(self.d)

        # 초기 variational parameters
        self.K = cfg.K_init
        self._init_variational(cfg.K_init)

    # ── 초기화 ────────────────────────────────────────────────

    def _init_variational(self, K: int):
        d = self.d
        self.K         = K
        # NIW posterior params
        self.mu_hat    = np.random.randn(K, d) * 0.1
        self.kappa_hat = np.full(K, self.kappa0 + 1.0)
        self.nu_hat    = np.full(K, self.nu0 + 1.0)
        self.Psi_hat   = np.stack([self.Psi0.copy() for _ in range(K)])
        # Beta params for q(beta_k)
        self.a_k1 = np.ones(K)
        self.a_k0 = np.full(K, self.alpha)
        # Expected count (M-step에서 업데이트)
        self.N_hat = np.zeros(K)

    # ── 기댓값 계산 ───────────────────────────────────────────

    def _E_log_pi(self) -> np.ndarray:
        """
        E_q[log pi_k] for k=0..K-1
        Stick-breaking: log pi_k = log beta_k + sum_{j<k} log(1-beta_j)
        E[log beta_k]    = psi(a_k1) - psi(a_k1 + a_k0)
        E[log(1-beta_k)] = psi(a_k0) - psi(a_k1 + a_k0)
        """
        psi_1  = digamma(self.a_k1)
        psi_0  = digamma(self.a_k0)
        psi_10 = digamma(self.a_k1 + self.a_k0)

        E_log_beta    = psi_1  - psi_10   # (K,)
        E_log_1mbeta  = psi_0  - psi_10   # (K,)

        out = np.zeros(self.K)
        cumsum = 0.0
        for k in range(self.K):
            out[k] = E_log_beta[k] + cumsum
            cumsum += E_log_1mbeta[k]
        return out                         # (K,)

    def _E_log_det(self) -> np.ndarray:
        """
        E_q[log|Lambda_k|] for each component k.
        NIW: E[log|Lambda|] = sum_{i=1}^d psi((nu+1-i)/2) + d*log2 + log|Psi^{-1}|
        """
        d   = self.d
        out = np.zeros(self.K)
        for k in range(self.K):
            psi_sum = sum(
                digamma((self.nu_hat[k] + 1 - i) / 2.0)
                for i in range(1, d + 1)
            )
            # E[log|Lambda|] = psi_sum + d*log2 - log|Psi_hat|
            sign, ldet = np.linalg.slogdet(self.Psi_hat[k])
            out[k] = psi_sum + d * math.log(2) - ldet
        return out                         # (K,)

    def _E_mahal(self, X: np.ndarray) -> np.ndarray:
        """
        E_q[(x-mu_k)^T Lambda_k (x-mu_k)] for all (n, k).
        = nu_hat_k * (x-mu_hat_k)^T Psi_hat_k^{-1} (x-mu_hat_k) + d/kappa_hat_k
        Returns (N, K).
        """
        N  = X.shape[0]
        out = np.zeros((N, self.K))
        for k in range(self.K):
            Psi_inv = np.linalg.inv(self.Psi_hat[k])      # (d,d)
            diff    = X - self.mu_hat[k]                   # (N,d)
            maha    = np.einsum('nd,de,ne->n', diff, Psi_inv, diff)
            out[:, k] = self.nu_hat[k] * maha + self.d / self.kappa_hat[k]
        return out                         # (N, K)

    # ── E-step: responsibilities ──────────────────────────────

    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        q(c_n = k) = r_hat_nk
        log r_hat_nk ∝ E[log pi_k]
                      + (1/2) E[log|Lambda_k|]
                      - (d/2) log(2pi)
                      - (1/2) E_mahl_nk
        Returns r_hat: (N, K)
        """
        E_lpi  = self._E_log_pi()        # (K,)
        E_ldet = self._E_log_det()       # (K,)
        E_mah  = self._E_mahal(X)       # (N,K)

        log_r  = (E_lpi[None, :]
                  + 0.5 * E_ldet[None, :]
                  - 0.5 * (self.d * math.log(2 * math.pi) + E_mah))

        # numerical stability: subtract row max
        log_r -= log_r.max(axis=1, keepdims=True)
        r      = np.exp(log_r)
        r     /= r.sum(axis=1, keepdims=True) + 1e-10
        return r                         # (N, K)

    # ── M-step: global parameter update ──────────────────────

    def m_step(self, X: np.ndarray, r: np.ndarray):
        """
        NIW conjugate posterior update + Beta stick-breaking update.

        NIW posterior (weighted by r_hat_nk):
            N_k     = sum_n r_hat_nk
            x_bar_k = (sum_n r_hat_nk * x_n) / N_k
            kappa_n = kappa0 + N_k
            nu_n    = nu0 + N_k
            mu_n    = (kappa0*mu0 + N_k*x_bar_k) / kappa_n
            S_k     = sum_n r_hat_nk (x_n-x_bar_k)(x_n-x_bar_k)^T
            Psi_n   = Psi0 + S_k + kappa0*N_k/kappa_n * (x_bar-mu0)(x_bar-mu0)^T

        Beta:
            a_k1 = 1 + N_k
            a_k0 = alpha + sum_{j>k} N_j
        """
        for k in range(self.K):
            N_k = float(r[:, k].sum())
            self.N_hat[k] = N_k
            if N_k < 1e-6:
                continue

            # Weighted statistics
            x_bar = (r[:, k] @ X) / N_k               # (d,)
            diff  = X - x_bar                           # (N,d)
            S_k   = (r[:, k, None] * diff).T @ diff    # (d,d)

            # NIW posterior update
            kappa_n = self.kappa0 + N_k
            nu_n    = self.nu0    + N_k
            mu_n    = (self.kappa0 * self.mu0 + N_k * x_bar) / kappa_n
            outer   = np.outer(x_bar - self.mu0, x_bar - self.mu0)
            Psi_n   = (self.Psi0 + S_k
                       + self.kappa0 * N_k / kappa_n * outer)

            self.mu_hat[k]    = mu_n
            self.kappa_hat[k] = kappa_n
            self.nu_hat[k]    = nu_n
            self.Psi_hat[k]   = Psi_n

        # Beta (stick-breaking) update — depends on cumulative N_hat
        for k in range(self.K):
            self.a_k1[k] = 1.0 + self.N_hat[k]
            self.a_k0[k] = self.alpha + self.N_hat[k+1:].sum()

    # ── ELBO ─────────────────────────────────────────────────

    def elbo(self, X: np.ndarray, r: np.ndarray) -> float:
        """
        ELBO = E_q[log p(x,c,theta,beta)] - E_q[log q(c,theta,beta)]

        주요 항:
          + sum_n sum_k r_nk * (E[log pi_k] + (1/2)E[log|Lambda_k|]
                                - (d/2)log(2pi) - (1/2)E_mahal_nk)
          - sum_n sum_k r_nk * log(r_nk + eps)   [assignment entropy]
          - KL(q(beta) || p(beta))                [stick-breaking KL]
        """
        E_lpi  = self._E_log_pi()
        E_ldet = self._E_log_det()
        E_mah  = self._E_mahal(X)

        # Expected complete-data log likelihood + assignment entropy
        ll = float((r * (
            E_lpi[None, :]
            + 0.5 * E_ldet[None, :]
            - 0.5 * (self.d * math.log(2 * math.pi) + E_mah)
        )).sum())
        ll -= float((r * np.log(r + 1e-10)).sum())

        # KL(q(beta_k) || Beta(1, alpha)) for each k
        for k in range(self.K):
            a1, a0 = self.a_k1[k], self.a_k0[k]
            # KL of Beta(a1,a0) vs Beta(1,alpha)
            kl = (gammaln(a1 + a0) - gammaln(a1) - gammaln(a0)
                  - gammaln(1 + self.alpha)
                  + (a1 - 1) * (digamma(a1) - digamma(a1 + a0))
                  + (a0 - self.alpha) * (digamma(a0) - digamma(a1 + a0)))
            ll -= kl

        return float(ll)

    # ── Birth heuristic ───────────────────────────────────────

    def _try_birth(self, X: np.ndarray, r: np.ndarray) -> bool:
        """
        Poorly explained points로 새 component 제안.
        ELBO가 개선되면 accept.

        K=1 특수 처리:
            K=1이면 softmax 결과 r[:,0]=1.0 (항상).
            max-r threshold로는 bad points를 찾을 수 없다.
            Mahalanobis 거리가 상위 25%인 점들을 bad points로 사용.
        """
        if self.K >= self.cfg.K_max:
            return False

        if self.K == 1:
            # K=1: Mahalanobis 거리 기반 bad points
            try:
                Psi_inv = np.linalg.inv(self.Psi_hat[0])
            except np.linalg.LinAlgError:
                Psi_inv = np.linalg.pinv(self.Psi_hat[0])
            diff     = X - self.mu_hat[0]                  # (N, d)
            maha     = np.einsum("nd,de,ne->n", diff, Psi_inv, diff)
            thresh   = np.percentile(maha, 75)             # 상위 25%
            bad_mask = maha > thresh
        else:
            bad_mask = r.max(axis=1) < self.cfg.birth_thresh

        n_bad = int(bad_mask.sum())
        if n_bad < self.cfg.birth_min_pts:
            return False

        X_bad    = X[bad_mask]
        elbo_old = self.elbo(X, r)

        # Hughes & Sudderth (2013) MemoVB:
        # bad points에 K_fresh개 cluster를 k-means로 초기화
        # (단일 component가 아닌 여러 sub-cluster 제안)
        K_fresh = min(self.cfg.birth_K_fresh,
                      self.cfg.K_max - self.K,
                      max(1, n_bad // self.cfg.birth_min_pts))
        if K_fresh < 1:
            return False

        # K-means로 K_fresh 개 centroid 초기화
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K_fresh, n_init=3,
                             random_state=42, max_iter=50)
        km.fit(X_bad)
        new_centers = km.cluster_centers_               # (K_fresh, d)

        K_old = self.K
        # 새 component들 추가
        for center in new_centers:
            mask_c = np.linalg.norm(X_bad - center, axis=1) <                      np.linalg.norm(X_bad - center, axis=1).mean() * 1.5
            n_c = max(int(mask_c.sum()), 1)
            if n_c > self.d:
                # Psi = scatter matrix (sum of squared deviations)
                # NOT multiplied by n_c to keep variance tight
                Psi_c = np.cov(X_bad[mask_c].T) + 1e-4 * np.eye(self.d)
                # Psi_hat = Psi0 + scatter → tight initial covariance
                Psi_c = self.Psi0 + Psi_c
            else:
                Psi_c = self.Psi0.copy()

            self.K += 1
            self.mu_hat    = np.vstack([self.mu_hat,    center[None]])
            self.kappa_hat = np.append(self.kappa_hat,  self.kappa0 + n_c)
            self.nu_hat    = np.append(self.nu_hat,     self.nu0 + n_c)
            self.Psi_hat   = np.concatenate([self.Psi_hat, Psi_c[None]], axis=0)
            self.a_k1      = np.append(self.a_k1, 1.0 + n_c)
            self.a_k0      = np.append(self.a_k0, self.alpha)
            self.N_hat     = np.append(self.N_hat, float(n_c))

        r_new    = self.e_step(X)
        elbo_new = self.elbo(X, r_new)

        # 새 component들 중 하나라도 birth_min_pts 이상 할당된 것이 있어야 accept
        new_N_hat = [r_new[:, k].sum() for k in range(K_old, self.K)]
        has_viable = any(n > self.cfg.birth_min_pts for n in new_N_hat)

        if elbo_new > elbo_old and has_viable:
            return True   # accept birth
        else:
            # revert all new components
            self.K = K_old
            self.mu_hat    = self.mu_hat[:K_old]
            self.kappa_hat = self.kappa_hat[:K_old]
            self.nu_hat    = self.nu_hat[:K_old]
            self.Psi_hat   = self.Psi_hat[:K_old]
            self.a_k1      = self.a_k1[:K_old]
            self.a_k0      = self.a_k0[:K_old]
            self.N_hat     = self.N_hat[:K_old]
            return False

    def _try_merge(self, X: np.ndarray, r: np.ndarray) -> bool:
        """
        가장 유사한 component 쌍을 찾아 병합 시도.
        ELBO가 개선되면 accept.
        """
        if self.K < 2:
            return False

        # cosine similarity로 가장 유사한 쌍 탐색
        nrm     = np.linalg.norm(self.mu_hat, axis=1, keepdims=True) + 1e-8
        cos_sim = (self.mu_hat / nrm) @ (self.mu_hat / nrm).T
        np.fill_diagonal(cos_sim, -1.0)
        i, j    = np.unravel_index(cos_sim.argmax(), cos_sim.shape)

        if cos_sim[i, j] < self.cfg.merge_cos:
            return False

        elbo_old = self.elbo(X, r)

        # Merged component
        N_i, N_j = self.N_hat[i], self.N_hat[j]
        N_m      = N_i + N_j + 1e-8
        mu_m     = (N_i * self.mu_hat[i] + N_j * self.mu_hat[j]) / N_m
        Psi_m    = self.Psi_hat[i] + self.Psi_hat[j]
        nu_m     = self.nu_hat[i]  + self.nu_hat[j]
        kappa_m  = self.kappa_hat[i] + self.kappa_hat[j]

        # 백업
        saved = {a: getattr(self, a).copy()
                 for a in ('mu_hat','kappa_hat','nu_hat','Psi_hat',
                           'a_k1','a_k0','N_hat')}
        K_old = self.K

        # i, j 제거 후 merged 추가
        keep = [k for k in range(self.K) if k not in (i, j)]
        self.K = len(keep) + 1
        self.mu_hat    = np.vstack([self.mu_hat[keep],    mu_m[None]])
        self.kappa_hat = np.append(self.kappa_hat[keep],  kappa_m)
        self.nu_hat    = np.append(self.nu_hat[keep],     nu_m)
        self.Psi_hat   = np.concatenate(
            [self.Psi_hat[keep], Psi_m[None]], axis=0)
        self.a_k1      = np.append(self.a_k1[keep], 1.0 + N_m)
        self.a_k0      = np.append(self.a_k0[keep], self.alpha)
        self.N_hat     = np.append(self.N_hat[keep], N_m)

        r_new    = self.e_step(X)
        elbo_new = self.elbo(X, r_new)

        if elbo_new > elbo_old:
            return True
        else:
            self.K = K_old
            for a, v in saved.items():
                setattr(self, a, v)
            return False

    # ── 메인 fit ─────────────────────────────────────────────

    def fit_batch(
        self,
        X: np.ndarray,           # (N, d_z)  encoder mu (no noise)
        n_cavi: int = 5,
        epoch: int = 0,          # current epoch (for birth_start_epoch gating)
    ) -> np.ndarray:
        """
        1 epoch: CAVI n_cavi회 → Birth 시도 → Merge 시도
        Returns r: (N, K_final)

        bnpy (Hughes & Sudderth) 기준:
          - birth_start_epoch 이전에는 birth 비활성화
            (encoder가 안정화되기 전에 birth하면 noise cluster 생성)
          - merge는 birth보다 늦게 시작
        """
        for _ in range(n_cavi):
            r = self.e_step(X)
            self.m_step(X, r)

        r = self.e_step(X)
        birth_enabled = epoch >= self.cfg.birth_start_epoch
        if birth_enabled and self._try_birth(X, r):
            r = self.e_step(X)
        if birth_enabled and self._try_merge(X, r):
            r = self.e_step(X)
        return r

    # ── Inference helpers ─────────────────────────────────────

    def soft_assign(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d_z) → r: (N, K)  soft skill assignment."""
        return self.e_step(X)

    def hard_assign(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d_z) → labels: (N,)  argmax skill index."""
        return self.e_step(X).argmax(axis=1)

    @property
    def n_active(self) -> int:
        """
        N_hat > 0 이상인 component 수.
        birth 직후에는 새 component들이 N_hat이 작으므로
        낮은 임계값으로 진짜 active 여부를 판단.
        """
        return int((self.N_hat > 0.5).sum())


# ─────────────────────────────────────────────────────────────
# Skill Pretrainer: alternating training loop
# ─────────────────────────────────────────────────────────────

class SkillPretrainer:
    """
    HELIOS 교대 학습:
        Step A: encoder frozen → DPM fit (CAVI + Birth/Merge)
        Step B: DPM frozen    → encoder/decoder/prior 업데이트

    사용 예:
        trainer = SkillPretrainer(cfg)
        trainer.train(dataloader)
        labels, z_all = trainer.assign_skill_labels(dataloader)
        # labels: (N_segments, T)  int32 skill index
    """

    def __init__(self, cfg: SkillPretrainConfig):
        self.cfg    = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else 'cpu')

        self.encoder = build_encoder(cfg).to(self.device)
        self.decoder = SkillDecoder(cfg).to(self.device)
        self.prior   = SkillPrior(cfg).to(self.device)
        self.dpm     = DPM(cfg)

        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.prior.parameters()),
            lr=cfg.lr, weight_decay=1e-4,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs)

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Step A: encoder → DPM ────────────────────────────────

    @torch.no_grad()
    def _collect_z(self, loader) -> np.ndarray:
        """
        전체 데이터셋에서 encoder summary mu 수집.
        encode_mu()는 이미 (B, d_z)를 반환 (마지막 timestep).
        seq_len=skill_horizon=10이면 10-step action sequence summary.
        """
        self.encoder.eval()
        zs = []
        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)
            mu = self.encoder.encode_mu(states, actions)  # (B, d_z)
            zs.append(mu.cpu().numpy())
        return np.concatenate(zs, axis=0)                 # (N_total, d_z)

    def _step_A(self, loader, epoch: int = 0) -> Tuple[int, float]:
        """Step A: DPM fitting."""
        Z = self._collect_z(loader)
        r = self.dpm.fit_batch(Z, n_cavi=5, epoch=epoch)
        return self.dpm.K, self.dpm.elbo(Z, r)

    # ── Step B: DPM → networks ───────────────────────────────

    def _step_B(self, loader) -> Dict[str, float]:
        """
        Step B: L_rec + L_dpm + L_prior 로 네트워크 업데이트.

        L_rec:   action 재구성 (primary teacher)
        L_dpm:   sum_k pi_ik * KL(q(z|s,a) || N(mu_k, Sigma_k))
                 → encoder z를 DPM component에 정렬
        L_prior: KL(q(z|s,a) || p(z|s))
                 → state-conditioned prior 학습
        """
        self.encoder.train()
        self.decoder.train()
        self.prior.train()

        # DPM component params를 tensor로 변환 (frozen)
        K    = self.dpm.K
        d    = self.cfg.skill_dim
        dpm_mu  = [torch.FloatTensor(self.dpm.mu_hat[k]).to(self.device)
                   for k in range(K)]
        # Empirical variance from Psi (scatter matrix / N_k)
        # 이것이 NIW posterior에서 실제 data scatter를 반영하므로 더 안정적
        # var_k = diag(Psi_hat_k) / max(N_hat_k, 1)
        # clamp to [0.1, 5.0]: 너무 작으면 KL 폭발, 너무 크면 gradient 없음
        dpm_var = []
        for k in range(K):
            N_k   = max(float(self.dpm.N_hat[k]), 1.0)
            var_k = np.diag(self.dpm.Psi_hat[k]) / N_k
            var_k = np.clip(var_k, 0.1, 5.0)
            dpm_var.append(torch.FloatTensor(var_k).to(self.device))

        totals = dict(loss=0., l_rec=0., l_dpm=0., l_prior=0., l_spread=0., l_vae=0.)
        n_b    = 0

        for actions, states in loader:
            actions = actions.to(self.device)   # (B, T, da)
            states  = states.to(self.device)    # (B, T, ds)
            B, T, _ = states.shape
            L       = self.cfg.skill_horizon

            # Encode
            z, mu_q, lv_q = self.encoder(states, actions)
            # z: (B, T, d_z)  — 각 timestep의 skill latent

            # seq_len=skill_horizon=10 → T=10
            # GRU/TCN의 마지막 timestep = 전체 skill sequence summary
            z_last = z[:, -1, :]              # (B, d_z)  skill summary
            mu_q_l = mu_q[:, -1, :]          # (B, d_z)
            lv_q_l = lv_q[:, -1, :]          # (B, d_z)
            s0     = states[:, 0, :]          # (B, ds)  skill 시작 state

            # ── L_rec ────────────────────────────────────────
            # seq_len=10 이므로 전체 sequence를 reconstruct
            a_hat = self.decoder(z_last, s0)           # (B, L, da)
            L_eff = min(a_hat.shape[1], T)
            L_rec = F.mse_loss(a_hat[:, :L_eff, :],
                               actions[:, :L_eff, :])

            # ── L_dpm ────────────────────────────────────────
            # soft assignment (no grad, numpy)
            with torch.no_grad():
                z_np  = z_last.detach().cpu().numpy()
                pi_ik = self.dpm.soft_assign(z_np)    # (B, K)
                pi_t  = torch.FloatTensor(pi_ik).to(self.device)

            # Temperature softening on pi_ik:
            # pi_ik가 0에 가까운 component도 소량 gradient를 흘려주어
            # encoder가 birth로 생긴 새 component 방향을 학습할 수 있게 함.
            # min_pi = 1/K 의 10%: 각 component에 최소 gradient 보장
            min_pi = 0.1 / max(K, 1)
            pi_soft = pi_t.clamp(min=min_pi)
            pi_soft = pi_soft / pi_soft.sum(dim=-1, keepdim=True)  # renormalize

            L_dpm = torch.zeros(1, device=self.device)
            for k in range(K):
                kl_k_per_dim = 0.5 * (
                    dpm_var[k].log() - lv_q_l
                    + lv_q_l.exp() / dpm_var[k].clamp(min=1e-6)
                    + (mu_q_l - dpm_mu[k]) ** 2 / dpm_var[k].clamp(min=1e-6)
                    - 1.0
                )  # (B, d_z)
                kl_k = kl_k_per_dim.mean(dim=-1).clamp(min=0.0, max=100.0)
                L_dpm = L_dpm + (pi_soft[:, k] * kl_k).mean()

            # ── L_prior (reverse KL: q || p) ─────────────────
            mu_p, lv_p = self.prior(s0)               # (B, d_z)
            var_p = lv_p.exp().clamp(min=1e-6)
            L_prior = 0.5 * (
                lv_p - lv_q_l
                + lv_q_l.exp() / var_p
                + (mu_q_l - mu_p) ** 2 / var_p
                - 1.0
            ).sum(dim=-1).mean()

            # L_spread: batch std 하한 강제 (z collapse 방지)
            L_spread = F.relu(self.cfg.min_z_std
                               - z_last.std(dim=0)).mean()
            # L_vae: KL(q || N(0,I))
            L_vae = 0.5 * (lv_q_l.exp() + mu_q_l**2
                            - 1.0 - lv_q_l).mean()
            loss = (self.cfg.zeta1 * L_rec
                    + self.cfg.zeta2 * L_dpm
                    + self.cfg.zeta3 * L_prior
                    + self.cfg.zeta_spread * L_spread
                    + self.cfg.zeta_vae    * L_vae)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.prior.parameters()), 1.0)
            self.opt.step()

            totals['loss']    += loss.item()
            totals['l_rec']   += L_rec.item()
            totals['l_dpm']   += L_dpm.item()
            totals['l_prior'] += L_prior.item()
            totals['l_spread']+= L_spread.item()
            totals['l_vae']   += L_vae.item()
            n_b += 1

        self.sched.step()
        return {k: v / max(n_b, 1) for k, v in totals.items()}

    # ── Training loop ─────────────────────────────────────────

    def train(self, loader, val_loader=None):
        enc_str = (f"GRU(h={self.cfg.gru_hidden},L={self.cfg.gru_layers})"
                   if self.cfg.encoder_type == 'gru'
                   else f"TCN(h={self.cfg.tcn_hidden},L={self.cfg.tcn_layers})")
        print(f"[SkillPretrain] epochs={self.cfg.epochs}  "
              f"enc={enc_str}  input={self.cfg.encoder_input}  "
              f"d_z={self.cfg.skill_dim}  K_max={self.cfg.K_max}  "
              f"device={self.device}")

        best_loss = float('inf')

        K_prev = self.dpm.K
        for ep in range(1, self.cfg.epochs + 1):
            # Step A: DPM fitting
            K, elbo = self._step_A(loader, epoch=ep)

            birth_happened = K > K_prev
            K_prev = K

            # Step B: network update
            m = self._step_B(loader)

            if birth_happened:
                print(f"  [birth K={K}] warm-up "
                      f"{self.cfg.birth_warmup_steps}x step_B ...")
                all_m = [m]
                for _ in range(self.cfg.birth_warmup_steps - 1):
                    all_m.append(self._step_B(loader))
                m = {k: sum(mi[k] for mi in all_m)/len(all_m) for k in m}
                # warm-up 후 birth 없이 DPM 재피팅 (spread된 z에 맞춤)
                K, elbo = self._step_A(loader, epoch=ep + 10000)
                print(f"  [post-warmup] K={K}(act={self.dpm.n_active})")

            print(
                f"Ep {ep:4d} | K={K}(act={self.dpm.n_active}) | "
                f"rec={m['l_rec']:.4f} dpm={m['l_dpm']:.4f} "
                f"spr={m['l_spread']:.4f} vae={m['l_vae']:.4f} | "
                f"ELBO={elbo:.1f}",
                flush=True,
            )

            if m['loss'] < best_loss:
                best_loss = m['loss']
                self.save('best.pt')

        self.save('final.pt')
        print(f"Done. best_loss={best_loss:.4f}  final_K={self.dpm.K}")

    # ── Inference: skill label 할당 ──────────────────────────

    @torch.no_grad()
    def assign_skill_labels(
        self,
        loader,
        hard: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터셋의 모든 (s_t, a_t)에 skill label 부여.
        Koopman CVAE 학습 시 (A_k, B_k)의 supervision으로 사용.

        Returns:
            labels: (N, T)        int32  if hard=True
                    (N, T, K)     float32 if hard=False (soft)
            z_all:  (N, T, d_z)   skill latents (분석용)
        """
        self.encoder.eval()
        all_labels, all_z = [], []

        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)
            B, T, _ = states.shape

            # encode_mu: (B, d_z) — one skill label per sequence
            mu = self.encoder.encode_mu(states, actions)  # (B, d_z)
            z_np = mu.cpu().numpy()                        # (B, d_z)

            r = self.dpm.soft_assign(z_np)                # (B, K)

            if hard:
                lbl = r.argmax(axis=1).astype(np.int32)   # (B,)
            else:
                lbl = r.astype(np.float32)                 # (B, K)

            all_labels.append(lbl)
            all_z.append(z_np)

        labels = np.concatenate(all_labels, axis=0)       # (N,) or (N,K)
        z_all  = np.concatenate(all_z, axis=0)            # (N, d_z)

        print(f"Assigned labels: shape={labels.shape}  K={self.dpm.K}")
        return labels, z_all

    # ── Checkpoint ────────────────────────────────────────────

    def save(self, name: str):
        path = Path(self.cfg.save_dir) / name
        torch.save({
            'encoder':    self.encoder.state_dict(),
            'decoder':    self.decoder.state_dict(),
            'prior':      self.prior.state_dict(),
            'dpm_state': {
                'K':         self.dpm.K,
                'mu_hat':    self.dpm.mu_hat,
                'kappa_hat': self.dpm.kappa_hat,
                'nu_hat':    self.dpm.nu_hat,
                'Psi_hat':   self.dpm.Psi_hat,
                'a_k1':      self.dpm.a_k1,
                'a_k0':      self.dpm.a_k0,
                'N_hat':     self.dpm.N_hat,
            },
            'cfg': self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.prior.load_state_dict(ckpt['prior'])
        ds = ckpt['dpm_state']
        self.dpm.K         = ds['K']
        self.dpm.mu_hat    = ds['mu_hat']
        self.dpm.kappa_hat = ds['kappa_hat']
        self.dpm.nu_hat    = ds['nu_hat']
        self.dpm.Psi_hat   = ds['Psi_hat']
        self.dpm.a_k1      = ds['a_k1']
        self.dpm.a_k0      = ds['a_k0']
        self.dpm.N_hat     = ds['N_hat']
        print(f"Loaded: K={self.dpm.K}")