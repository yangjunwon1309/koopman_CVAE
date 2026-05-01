"""
action_mppi.py — KODAQ v5 MPPI Action Sampler
===============================================

TD-MPC2 스타일 MPPI를 KODAQ v5 구조에 적용.

핵심 구조:
  - latent action space (d_u=64)에서 MPPI 수행
  - estimate_value: H-step Koopman rollout + ensemble reward + terminal Q
  - num_pi_trajs개: policy prior π(z)로 warm-start (good region 초기화)
  - 나머지: mean ± std Gaussian 샘플
  - elite top-k → softmax(value / temperature) weight → CEM-style mean/std update
  - receding horizon: mean[0] → action 실행, mean[1:]은 다음 step warm-start
  - LQR 초기화 없이 mean=0+eps에서 시작 (online 환경)

Note:
  TD-MPC2는 raw action space에서 MPPI → action 직접 출력.
  KODAQ은 latent action space (u = ψ(a))에서 MPPI → action decoder로 복원.
  u* → a* 복원: gradient-based inversion (action_encoder inverse)

Usage:
    from action_mppi import KODAQMPPIPlanner, MPPIConfig

    planner = KODAQMPPIPlanner(model, MPPIConfig())
    planner.reset()

    # Online loop
    h = model.recurrent.init_hidden(1, device)
    for t in range(episode_len):
        obs_x = get_obs()   # (1, x_dim)
        a, info = planner.act(obs_x, h)
        obs_next, reward, done, _ = env.step(a)
        # Update h
        with torch.no_grad():
            o, _, _ = model.posterior.sample(obs_x, h)
            h = model.recurrent(h, o, torch.FloatTensor(a).unsqueeze(0))
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from models.koopman_cvae import KoopmanCVAE
from models.losses import blend_koopman, two_hot_decode


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MPPIConfig:
    # ── Sample sizes (TD-MPC2 defaults) ──────────────────────────────────────
    num_samples:    int   = 512    # total trajectory samples
    num_pi_trajs:   int   = 24     # policy prior warm-start samples
    num_elites:     int   = 64     # top-k for weight computation
    horizon:        int   = 5      # planning horizon H

    # ── MPPI hyperparameters ──────────────────────────────────────────────────
    temperature:    float = 0.5    # softmax temperature (higher=explore, lower=exploit)
    init_std:       float = 2.0    # initial std of action distribution
    min_std:        float = 0.05   # minimum std (prevents collapse)
    max_std:        float = 2.0    # maximum std

    # ── Discount ──────────────────────────────────────────────────────────────
    gamma:          float = 0.99

    # ── Action decoder ────────────────────────────────────────────────────────
    action_inv_steps: int  = 30    # gradient steps for u → a inversion
    action_inv_lr:    float = 0.05

    # ── Iterations per step ───────────────────────────────────────────────────
    iterations:     int   = 6      # MPPI CEM iterations
    # TD-MPC2: iterations += 2 * int(action_dim >= 20)
    # latent dim d_u=64 이므로 +2 적용
    iterations_large_action: int = 2

    # ── Momentum ──────────────────────────────────────────────────────────────
    momentum:       float = 0.1    # prev_mean에 대한 warm-start 비율 (0=no momentum)


# ──────────────────────────────────────────────────────────────────────────────
# MPPI Planner
# ──────────────────────────────────────────────────────────────────────────────

class KODAQMPPIPlanner:
    """
    TD-MPC2 스타일 MPPI (latent action space).

    TD-MPC2 원본 대비 차이:
      - action space:  raw a ∈ [-1,1]^9  →  latent u ∈ ℝ^{d_u}
      - value estimate: Koopman rollout (linear dynamics, no learned MLP dynamics)
      - reward:  ensemble reward head  R̂_pen(z, u) = mean_i[σ(r̂_i)] - β·std
      - terminal value: Q(z_H, u_H) decoded from Two-Hot categorical
      - pi_trajs: policy_prior π(z) → u sample (tanh-squashed Gaussian)
      - initialization: mean=0 (no LQR warm-start)
    """

    def __init__(self, model: KoopmanCVAE, cfg: MPPIConfig):
        self.model  = model
        self.cfg    = cfg
        self.m_cfg  = model.cfg
        self.device = next(model.parameters()).device

        self.d_u  = model.cfg.action_latent   # latent action dim
        self.d_a  = model.cfg.action_dim       # raw action dim (9)

        # Receding horizon: previous mean for warm-start
        self.prev_mean: Optional[torch.Tensor] = None  # (H, d_u)

        # Total iterations (large latent dim adjustment)
        self._n_iter = cfg.iterations + cfg.iterations_large_action

        print(f"[KODAQMPPIPlanner] d_u={self.d_u}  d_a={self.d_a}  "
              f"H={cfg.horizon}  N={cfg.num_samples}  "
              f"n_pi={cfg.num_pi_trajs}  elites={cfg.num_elites}  "
              f"iters={self._n_iter}")

    def reset(self):
        """에피소드 시작 시 prev_mean 초기화."""
        self.prev_mean = None

    # ── Value estimation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def estimate_value(
        self,
        z0:       torch.Tensor,   # (1, m)   current latent
        u_plans:  torch.Tensor,   # (H, N, d_u)  candidate action sequences
        w0:       torch.Tensor,   # (1, K)   current skill weights
    ) -> torch.Tensor:
        """
        H-step Koopman rollout + ensemble reward + terminal Q.

        G[n] = Σ_{k=0}^{H-1} γ^k · R̂_pen(z_k^n, u_k^n)
             + γ^H · Q(z_H^n, u_H^n)

        z_{k+1} = Ā(w_k)·z_k + B̄(w_k)·u_k  (Koopman linear)

        Returns: (N,) value for each candidate trajectory.
        """
        cfg     = self.cfg
        model   = self.model
        koop    = model.koopman
        N       = u_plans.shape[1]
        H       = cfg.horizon
        dev     = self.device
        gamma   = cfg.gamma

        log_lam = koop.get_log_lambdas()

        # Expand z0 to (N, m)
        z = z0.expand(N, -1).clone()    # (N, m)
        w = w0.expand(N, -1).clone()    # (N, K)

        G        = torch.zeros(N, device=dev)
        discount = 1.0

        for k in range(H):
            u_k = u_plans[k]   # (N, d_u)

            # Ensemble reward
            R_pen = model.reward_ensemble_head.penalized_reward(z, u_k)  # (N,)
            G     = G + discount * R_pen
            discount *= gamma

            # Koopman step: z_{k+1} = Ā(w)·z + B̄(w)·u
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, koop.theta_k, koop.G_k, koop.U, w)
            # A_bar: (N, m, m),  B_bar: (N, m, d_u)
            z = ((A_bar @ z.unsqueeze(-1)).squeeze(-1)
                 + (B_bar @ u_k.unsqueeze(-1)).squeeze(-1))  # (N, m)

        # Terminal Q: Q(z_H, u_H^last) — pessimistic (min of 2 random)
        u_H = u_plans[-1]   # (N, d_u)  last step action
        Q_terminal = model.q_head.expected_value(
            z, u_H, return_type='min')  # (N,)
        G = G + discount * Q_terminal

        return G   # (N,)

    # ── Policy prior trajectory sampling ─────────────────────────────────────

    @torch.no_grad()
    def _sample_pi_trajs(
        self,
        z0: torch.Tensor,   # (1, m)
        w0: torch.Tensor,   # (1, K)
        H:  int,
    ) -> torch.Tensor:
        """
        policy prior π(z)로 H-step rollout → (H, num_pi_trajs, d_u)

        TD-MPC2와 동일:
          at each step, sample u ~ π(z_t), then step z with Koopman.
          This gives num_pi_trajs 'good' candidate trajectories
          that seed the MPPI distribution.
        """
        n       = self.cfg.num_pi_trajs
        model   = self.model
        koop    = model.koopman
        log_lam = koop.get_log_lambdas()
        dev     = self.device

        z = z0.expand(n, -1).clone()    # (n, m)
        w = w0.expand(n, -1).clone()    # (n, K)

        u_seq = []
        for k in range(H):
            u, _, _, _ = model.policy_prior(z)   # (n, d_u)
            u_seq.append(u)

            # Koopman step
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, koop.theta_k, koop.G_k, koop.U, w)
            z = ((A_bar @ z.unsqueeze(-1)).squeeze(-1)
                 + (B_bar @ u.unsqueeze(-1)).squeeze(-1))

        return torch.stack(u_seq, dim=0)   # (H, n, d_u)

    # ── MPPI core ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,   # (1, m)  current latent
        w0: torch.Tensor,   # (1, K)  current skill weights
    ) -> torch.Tensor:
        """
        MPPI planning → optimal mean (H, d_u).

        Returns: mean[0] as the action to execute this step.
        Full mean is stored as prev_mean for next step.
        """
        cfg  = self.cfg
        H    = cfg.horizon
        N    = cfg.num_samples
        dev  = self.device
        d_u  = self.d_u

        # ── Init mean ─────────────────────────────────────────────────────────
        if self.prev_mean is not None:
            # Receding horizon warm-start: shift by 1, append zero
            mean = torch.cat([
                self.prev_mean[1:],
                torch.zeros(1, d_u, device=dev)
            ], dim=0)   # (H, d_u)
            if cfg.momentum > 0:
                mean = (1 - cfg.momentum) * mean
        else:
            mean = torch.zeros(H, d_u, device=dev)   # cold start = 0

        std = cfg.init_std * torch.ones(H, d_u, device=dev)

        # ── Policy prior trajectories (warm-start) ────────────────────────────
        pi_trajs = self._sample_pi_trajs(z0, w0, H)   # (H, n_pi, d_u)

        # ── MPPI iterations ───────────────────────────────────────────────────
        for it in range(self._n_iter):

            # Sample Gaussian perturbations around current mean
            n_gauss = N - cfg.num_pi_trajs
            eps     = torch.randn(H, n_gauss, d_u, device=dev)
            gauss_trajs = mean.unsqueeze(1) + std.unsqueeze(1) * eps
            # (H, n_gauss, d_u) — no clamp: latent u is unbounded
            # (clipping is done in reward_ensemble_head / Q head which are trained
            #  on encoder-mapped actions; soft signal discourages OOD values)

            # Concatenate: pi_trajs first, then gaussian
            u_plans = torch.cat([pi_trajs, gauss_trajs], dim=1)  # (H, N, d_u)

            # Estimate value for all N trajectories
            values = self.estimate_value(z0, u_plans, w0)   # (N,)
            values = values.nan_to_num_(0.0)

            # ── Elite selection ────────────────────────────────────────────
            elite_idx    = torch.topk(values, cfg.num_elites, dim=0).indices
            elite_values = values[elite_idx]               # (num_elites,)
            elite_trajs  = u_plans[:, elite_idx, :]        # (H, num_elites, d_u)

            # ── Softmax weights (TD-MPC2) ──────────────────────────────────
            score = torch.softmax(
                elite_values / cfg.temperature, dim=0
            ).unsqueeze(0).unsqueeze(-1)   # (1, num_elites, 1)

            # ── Weighted mean & std update ─────────────────────────────────
            mean = (score * elite_trajs).sum(dim=1)   # (H, d_u)

            # Weighted std
            diff    = elite_trajs - mean.unsqueeze(1)   # (H, num_elites, d_u)
            var     = (score * diff.pow(2)).sum(dim=1)  # (H, d_u)
            std     = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        # Store for next step (receding horizon)
        self.prev_mean = mean.clone()

        return mean   # (H, d_u)

    # ── Action decoding: u → a ────────────────────────────────────────────────

    def decode_action(self, u: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based inversion: argmin_a ||ψ(a) - u||²
        a ∈ [-1, 1]^{d_a}

        u: (1, d_u) or (d_u,)
        Returns: a: (d_a,) numpy array
        """
        if u.dim() == 1:
            u = u.unsqueeze(0)
        u_target = u.detach().to(self.device)

        a = torch.zeros(1, self.d_a, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=self.cfg.action_inv_lr)

        for _ in range(self.cfg.action_inv_steps):
            opt.zero_grad()
            u_hat = self.model.action_encoder(a)
            loss  = F.mse_loss(u_hat, u_target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-1.0, 1.0)

        return a.detach().squeeze(0).cpu().numpy()   # (d_a,)

    # ── Main act() interface ──────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_step(
        self,
        obs_x: torch.Tensor,   # (1, x_dim)
        h:     torch.Tensor,   # (1, d_h)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs → (z, w, h_new)
        z: posterior latent
        w: skill weights
        h_new: updated hidden state (BEFORE action, for MPPI)
        """
        o, _, _ = self.model.posterior.sample(obs_x, h)
        w       = self.model.skill_prior.soft_weights(h)
        return o, w, h

    def act(
        self,
        obs_x: torch.Tensor,   # (1, x_dim)
        h:     torch.Tensor,   # (1, d_h)
    ) -> Tuple[np.ndarray, Dict]:
        """
        One-step MPPI:
          1. Encode obs → z, w
          2. MPPI plan → optimal u_seq (H, d_u)
          3. Decode u[0] → raw action a ∈ [-1,1]^{d_a}

        Returns:
          a:    (d_a,) numpy array,  clipped to [-1, 1]
          info: dict with diagnostics
        """
        obs_x = obs_x.to(self.device)
        h     = h.to(self.device)

        # Encode
        z, w, _ = self._encode_step(obs_x, h)

        # MPPI plan
        mean = self.plan(z, w)   # (H, d_u)

        # Decode first action
        u0 = mean[0]             # (d_u,)
        a  = self.decode_action(u0)   # (d_a,)

        # Diagnostics
        with torch.no_grad():
            q_val  = self.model.q_head.expected_value(
                z, u0.unsqueeze(0), return_type='min').item()
            r_pred = self.model.reward_ensemble_head.penalized_reward(
                z, u0.unsqueeze(0)).item()

        info = {
            'u0':    u0.cpu().numpy(),
            'q_val': q_val,
            'r_pred': r_pred,
            'mean':  mean.cpu().numpy(),   # (H, d_u) full horizon
        }
        return a, info

    # ── Batch act for eval (no gradient) ─────────────────────────────────────

    def act_greedy(
        self,
        obs_x: torch.Tensor,
        h:     torch.Tensor,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Greedy action without MPPI: just π(z) mean.
        Faster for evaluation.
        """
        obs_x = obs_x.to(self.device)
        h     = h.to(self.device)
        z, w, _ = self._encode_step(obs_x, h)
        with torch.no_grad():
            _, _, mean_u, _ = self.model.policy_prior(z)   # deterministic mean
        a = self.decode_action(mean_u[0])
        return a, {'u0': mean_u[0].cpu().numpy()}


# ──────────────────────────────────────────────────────────────────────────────
# Online evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_mppi_episode(
    model:    KoopmanCVAE,
    planner:  KODAQMPPIPlanner,
    env,
    x_encoder,       # function: obs (60,) → x_seq (2108,)
    device:   str = 'cuda',
    max_steps: int = 280,
    use_mppi: bool = True,
    render:   bool = False,
) -> Dict:
    """
    단일 에피소드 실행.

    x_encoder: D4RL obs (60-dim) → KODAQ x (2108-dim)
    실제로는 R3M feature + delta_p + q + qdot 계산 필요.
    여기서는 간단히 obs slice로 x를 구성 (x_dim = 2108은 사전 처리 필요).

    Returns: episode result dict.
    """
    dev    = torch.device(device)
    model.eval()
    planner.reset()

    obs      = env.reset()
    h        = model.recurrent.init_hidden(1, dev)
    total_r  = 0.0
    step     = 0
    done     = False
    rewards  = []
    q_vals   = []
    r_preds  = []

    while not done and step < max_steps:
        # obs → x (2108-dim)
        x = torch.FloatTensor(x_encoder(obs)).unsqueeze(0).to(dev)  # (1, x_dim)

        # Act
        if use_mppi:
            a_np, info = planner.act(x, h)
        else:
            a_np, info = planner.act_greedy(x, h)

        # Environment step
        obs_next, reward, done, _ = env.step(a_np)
        if render:
            env.render()

        # Update h with posterior + recurrent
        with torch.no_grad():
            o, _, _ = model.posterior.sample(x, h)
            a_t = torch.FloatTensor(a_np).unsqueeze(0).to(dev)
            h   = model.recurrent(h, o, a_t)

        total_r  += reward
        rewards.append(reward)
        if 'q_val'  in info: q_vals.append(info['q_val'])
        if 'r_pred' in info: r_preds.append(info['r_pred'])

        obs  = obs_next
        step += 1

    return {
        'total_reward': total_r,
        'steps':        step,
        'rewards':      np.array(rewards),
        'q_vals':       np.array(q_vals) if q_vals else None,
        'r_preds':      np.array(r_preds) if r_preds else None,
        'n_tasks':      int(total_r),   # Kitchen: +1 per subtask
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI eval
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--env',          default='kitchen-mixed-v0')
    p.add_argument('--n_ep',         type=int,   default=10)
    p.add_argument('--horizon',      type=int,   default=5)
    p.add_argument('--num_samples',  type=int,   default=512)
    p.add_argument('--num_pi_trajs', type=int,   default=24)
    p.add_argument('--num_elites',   type=int,   default=64)
    p.add_argument('--temperature',  type=float, default=0.5)
    p.add_argument('--iterations',   type=int,   default=6)
    p.add_argument('--init_std',     type=float, default=2.0)
    p.add_argument('--greedy',       action='store_true',
                   help='Use greedy policy prior instead of MPPI')
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',      default='checkpoints/kodaq_v5_lqr/mppi_eval')
    args = p.parse_args()

    import os, sys
    sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
    from models.koopman_cvae import KoopmanCVAE

    # Load model
    ckpt  = torch.load(args.ckpt, map_location=args.device)
    cfg_m = ckpt['cfg']
    v5_defaults = dict(
        num_bins=101, v_min=0.0, v_max=5.0, num_q=2, tau=0.005,
        gamma=0.99, entropy_coef=0.01, log_std_min=-5.0, log_std_max=2.0,
        lambda_reward=1.0, lambda_q=1.0, lambda_pi=0.1,
        reward_ensemble_n=5, td_horizon=4, mopo_beta=1.0,
        use_ensemble_reward=True, use_lqr_policy=False, lqr_horizon=4,
    )
    for k, v in v5_defaults.items():
        if not hasattr(cfg_m, k): setattr(cfg_m, k, v)

    model = KoopmanCVAE(cfg_m)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval().to(args.device)

    # MPPI planner
    mppi_cfg = MPPIConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_pi_trajs=args.num_pi_trajs,
        num_elites=args.num_elites,
        temperature=args.temperature,
        iterations=args.iterations,
        init_std=args.init_std,
    )
    planner = KODAQMPPIPlanner(model, mppi_cfg)

    # Env
    import d4rl, gym
    env = gym.make(args.env)

    # x_encoder: for eval we use a simple obs→x placeholder
    # In production, this should use R3M + proper delta computation
    def x_encoder_placeholder(obs):
        """
        Placeholder: obs (60,) → x (2108,)
        delta_e=0 (R3M feature unavailable in simple eval)
        delta_p = obs[18:60] - obs_ref[18:60]  (relative to episode start)
        q       = obs[0:9]
        qdot    = obs[9:18]
        """
        # NOTE: proper implementation requires R3M features
        # This placeholder zeros out delta_e
        delta_e = np.zeros(2048, dtype=np.float32)
        delta_p = obs[18:60].astype(np.float32)
        q       = obs[0:9].astype(np.float32)
        qdot    = obs[9:18].astype(np.float32)
        return np.concatenate([delta_e, delta_p, q, qdot])

    # Run episodes
    from pathlib import Path
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    results  = []
    n_tasks_list = []
    print(f"\n{'='*55}")
    print(f"KODAQ v5 MPPI Eval  n_ep={args.n_ep}  "
          f"{'greedy' if args.greedy else 'MPPI'}")
    print(f"  H={args.horizon}  N={args.num_samples}  "
          f"elites={args.num_elites}  T={args.temperature}")
    print(f"{'='*55}")

    for ep in range(args.n_ep):
        res = run_mppi_episode(
            model, planner, env,
            x_encoder=x_encoder_placeholder,
            device=args.device,
            use_mppi=not args.greedy,
        )
        results.append(res)
        n_tasks_list.append(res['n_tasks'])
        print(f"Ep {ep+1:3d}/{args.n_ep}  "
              f"reward={res['total_reward']:.1f}  "
              f"tasks={res['n_tasks']}  "
              f"steps={res['steps']}", flush=True)

    # Summary
    rewards = [r['total_reward'] for r in results]
    tasks   = [r['n_tasks']      for r in results]
    print(f"\n{'='*55}")
    print(f"Summary ({args.n_ep} episodes)")
    print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}"
          f"  (max={max(rewards):.1f}  min={min(rewards):.1f})")
    print(f"  Tasks:  {np.mean(tasks):.2f} ± {np.std(tasks):.2f}"
          f"  (max={max(tasks)})")
    print(f"{'='*55}")

    # Save
    np.save(f"{args.out_dir}/mppi_rewards.npy", np.array(rewards))
    np.save(f"{args.out_dir}/mppi_tasks.npy",   np.array(tasks))
    print(f"Saved → {args.out_dir}/")


if __name__ == '__main__':
    main()