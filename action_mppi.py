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

    # ── Goal-conditioned sampling ─────────────────────────────────────────────
    # num_goal_samples개의 z_g를 π_goal에서 샘플,
    # 각 z_g당 trajs_per_goal개의 LQR rollout 생성
    # 전체 goal trajs = num_goal_samples * trajs_per_goal
    # 나머지 = num_pi_trajs (policy prior) + gaussian noise
    goal_ratio:        float = 0.0   # 0=disabled, >0 = fraction of N for goal trajs
    # num_goal_samples * trajs_per_goal ≈ goal_ratio * num_samples
    num_goal_samples:  int   = 10    # μ_goal 샘플 수
    # trajs_per_goal = (goal_ratio * num_samples) // num_goal_samples

    # ── MPPI hyperparameters ──────────────────────────────────────────────────
    temperature:    float = 0.5    # softmax temperature
    init_std:       float = 2.0    # initial std
    min_std:        float = 0.05
    max_std:        float = 2.0

    # ── Discount ──────────────────────────────────────────────────────────────
    gamma:          float = 0.99

    # ── Action decoder ────────────────────────────────────────────────────────
    action_inv_steps: int   = 30
    action_inv_lr:    float = 0.05

    # ── Iterations ────────────────────────────────────────────────────────────
    iterations:     int   = 6
    iterations_large_action: int = 2

    # ── Momentum ──────────────────────────────────────────────────────────────
    momentum:       float = 0.1


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

    # ── Goal-conditioned trajectory sampling ─────────────────────────────────

    @torch.no_grad()
    def _sample_goal_trajs(
        self,
        z0: torch.Tensor,   # (1, m)
        h0: torch.Tensor,   # (1, d_h)
        w0: torch.Tensor,   # (1, K)
        H:  int,
        n_goals:      int,  # num_goal_samples
        trajs_per_goal: int,
    ) -> torch.Tensor:
        """
        π_goal로 n_goals개 z_g 샘플 → 각 z_g에 대해 trajs_per_goal개 LQR rollout.
        Total: n_goals * trajs_per_goal trajectories → (H, n_goals*trajs_per_goal, d_u)

        u_k = M·z_g - L·z_k  (M, L: skill-weighted, fixed)
        z_{k+1} = Ā·z_k + B̄·u_k
        """
        planner = self.model._lqr_planner
        if planner is None or planner._L_tensor is None:
            return None

        model   = self.model
        koop    = model.koopman
        log_lam = koop.get_log_lambdas()
        dev     = self.device
        N_total = n_goals * trajs_per_goal

        # ── Sample n_goals goal latents ───────────────────────────────────
        z_rep  = z0.expand(n_goals, -1)   # (n_goals, m)
        h_rep  = h0.expand(n_goals, -1)   # (n_goals, d_h)
        z_g_samples, _, _ = model.goal_proposal(z_rep, h_rep)  # (n_goals, m)

        # ── Expand to N_total: each z_g repeated trajs_per_goal times ─────
        z_g_exp = z_g_samples.unsqueeze(1).expand(
            n_goals, trajs_per_goal, -1
        ).reshape(N_total, -1)   # (N_total, m)

        # Skill weights (same for all, since all start at z0)
        w_exp = w0.expand(N_total, -1)   # (N_total, K)

        # Gain matrices (skill-weighted, fixed)
        L_w = torch.einsum('bk,kdm->bkdm',
                            w_exp,
                            planner._L_tensor.to(dev).detach()).sum(1)  # (N_total, d_u, m)
        M_w = torch.einsum('bk,kdm->bkdm',
                            w_exp,
                            planner._M_tensor.to(dev).detach()).sum(1)  # (N_total, d_u, m)

        # ── H-step LQR rollout ────────────────────────────────────────────
        z_cur  = z0.expand(N_total, -1).clone()   # (N_total, m)
        u_seq  = []

        for k in range(H):
            # u_k = M·z_g - L·z_k  (z_g fixed per trajectory)
            u_k = ((M_w @ z_g_exp.unsqueeze(-1)).squeeze(-1)
                   - (L_w @ z_cur.unsqueeze(-1)).squeeze(-1))  # (N_total, d_u)

            # clip to surveyed bounds
            if planner.cfg.use_u_bounds and planner.u_min is not None:
                u_min_t = torch.FloatTensor(planner.u_min).to(dev)
                u_max_t = torch.FloatTensor(planner.u_max).to(dev)
                u_k = u_k.clamp(u_min_t, u_max_t)

            u_seq.append(u_k)

            # Koopman step
            A_bar, B_bar, _, _ = blend_koopman(
                log_lam, koop.theta_k, koop.G_k, koop.U, w_exp)
            z_cur = ((A_bar @ z_cur.unsqueeze(-1)).squeeze(-1)
                     + (B_bar @ u_k.unsqueeze(-1)).squeeze(-1))

        return torch.stack(u_seq, dim=0)   # (H, N_total, d_u)

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
        h0: Optional[torch.Tensor] = None,  # (1, d_h) for goal proposal
    ) -> torch.Tensor:
        """
        MPPI planning → optimal mean (H, d_u).

        Sample composition (N=512, goal_ratio=0.95, num_goal_samples=10 예시):
          goal trajs:   10 × 48 = 480  (π_goal → LQR rollout)
          pi trajs:     24              (policy prior rollout)
          gaussian:      8              (mean ± std)
          total:        512
        """
        cfg  = self.cfg
        H    = cfg.horizon
        N    = cfg.num_samples
        dev  = self.device
        d_u  = self.d_u

        # ── Init mean ─────────────────────────────────────────────────────────
        if self.prev_mean is not None:
            mean = torch.cat([
                self.prev_mean[1:],
                torch.zeros(1, d_u, device=dev)
            ], dim=0)
            if cfg.momentum > 0:
                mean = (1 - cfg.momentum) * mean
        else:
            mean = torch.zeros(H, d_u, device=dev)

        std = cfg.init_std * torch.ones(H, d_u, device=dev)

        # ── Compute sample budget ─────────────────────────────────────────────
        has_goal = (cfg.goal_ratio > 0
                    and h0 is not None
                    and hasattr(self.model, 'goal_proposal')
                    and self.model._lqr_planner is not None)

        if has_goal:
            n_goal_total   = int(cfg.goal_ratio * N)
            n_goals        = cfg.num_goal_samples
            trajs_per_goal = max(1, n_goal_total // n_goals)
            n_goal_total   = n_goals * trajs_per_goal   # actual
            n_pi           = cfg.num_pi_trajs
            n_gauss        = max(0, N - n_goal_total - n_pi)
        else:
            n_goal_total   = 0
            n_pi           = cfg.num_pi_trajs
            n_gauss        = N - n_pi

        # ── Policy prior trajectories ─────────────────────────────────────────
        pi_trajs = self._sample_pi_trajs(z0, w0, H)   # (H, n_pi, d_u)

        # ── MPPI iterations ───────────────────────────────────────────────────
        for it in range(self._n_iter):

            traj_list = [pi_trajs]   # always include pi_trajs

            # Goal-conditioned trajs
            if has_goal:
                goal_trajs = self._sample_goal_trajs(
                    z0, h0, w0, H,
                    n_goals=n_goals,
                    trajs_per_goal=trajs_per_goal,
                )
                if goal_trajs is not None:
                    traj_list.append(goal_trajs)

            # Gaussian noise trajs
            if n_gauss > 0:
                eps         = torch.randn(H, n_gauss, d_u, device=dev)
                gauss_trajs = mean.unsqueeze(1) + std.unsqueeze(1) * eps
                traj_list.append(gauss_trajs)

            u_plans = torch.cat(traj_list, dim=1)   # (H, N_actual, d_u)

            # Value estimation
            values = self.estimate_value(z0, u_plans, w0).nan_to_num_(0.0)

            # Elite selection
            n_elite      = min(cfg.num_elites, u_plans.shape[1])
            elite_idx    = torch.topk(values, n_elite, dim=0).indices
            elite_values = values[elite_idx]
            elite_trajs  = u_plans[:, elite_idx, :]

            # Softmax weights
            score = torch.softmax(
                elite_values / cfg.temperature, dim=0
            ).unsqueeze(0).unsqueeze(-1)

            # Mean & std update
            mean = (score * elite_trajs).sum(dim=1)
            diff = elite_trajs - mean.unsqueeze(1)
            var  = (score * diff.pow(2)).sum(dim=1)
            std  = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        self.prev_mean = mean.clone()
        return mean

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

        # MPPI plan (pass h for goal proposal)
        mean = self.plan(z, w, h0=h)   # (H, d_u)

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
# GIF / render utilities  (제공된 코드 통합)
# ──────────────────────────────────────────────────────────────────────────────

def save_gif(frames: list, path: str, fps: int = 10):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        print(f"    GIF: {path}  ({len(frames)} frames)")
    except ImportError:
        # Fallback: contact sheet PNG
        n    = min(len(frames), 12)
        step = max(1, len(frames) // n)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[min(i * step, len(frames) - 1)])
            ax.axis('off')
        plt.tight_layout()
        strip = path.replace('.gif', '_strip.png')
        plt.savefig(strip, dpi=80)
        plt.close()
        print(f"    Strip: {strip}")


def render_frame(
    env,
    w:           int   = 512,
    h:           int   = 512,
    crop_center: bool  = True,
    crop_ratio:  float = 0.55,
    upscale:     int   = 2,
) -> np.ndarray:
    """환경 프레임 렌더링 + 중앙 crop + 업스케일."""
    try:
        f = env.render(mode='rgb_array', width=w, height=h)
        if f is None:
            f = env.unwrapped.sim.render(w, h, camera_name='main_cam')
    except Exception:
        try:
            f = env.unwrapped.sim.render(w, h)
        except Exception:
            f = np.zeros((h, w, 3), dtype=np.uint8)

    if not crop_center or f is None:
        return f

    H_f, W_f = f.shape[:2]
    ch = int(H_f * crop_ratio)
    cw = int(W_f * crop_ratio)
    y0 = (H_f - ch) // 2
    x0 = (W_f - cw) // 2
    cropped = f[y0:y0 + ch, x0:x0 + cw]

    if upscale > 1:
        try:
            from PIL import Image as _PIL
            img = _PIL.fromarray(cropped.astype(np.uint8))
            img = img.resize((cw * upscale, ch * upscale), _PIL.LANCZOS)
            cropped = np.array(img)
        except ImportError:
            cropped = np.repeat(np.repeat(cropped, upscale, 0), upscale, 1)
    return cropped


def inspect_info(info: dict):
    """D4RL kitchen-mixed task completion 정보 추출."""
    if 'score' in info:
        return int(round(float(info['score']) * 4)), []
    if 'num_success'      in info: return int(info['num_success']), []
    if 'completed_tasks'  in info: return len(info['completed_tasks']), list(info['completed_tasks'])
    if 'goal_achieved'    in info: return int(info['goal_achieved']), []
    return 0, []


# ──────────────────────────────────────────────────────────────────────────────
# Online evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_mppi_episode(
    model:       KoopmanCVAE,
    planner:     KODAQMPPIPlanner,
    env,
    x_encoder,
    device:      str  = 'cuda',
    max_steps:   int  = 280,
    use_mppi:    bool = True,
    record_gif:  bool = False,
    gif_fps:     int  = 15,
    gif_path:    Optional[str] = None,
    render_w:    int  = 512,
    render_h:    int  = 512,
) -> Dict:
    """
    단일 에피소드 실행 + 선택적 GIF 저장.

    x_encoder : obs (60,) → x (2108,)
    record_gif: True이면 매 step 렌더링하고 gif_path에 저장
    """
    dev = torch.device(device)
    model.eval()
    planner.reset()

    obs     = env.reset()
    obs_ref = obs.copy()   # episode-start reference for delta_p
    h       = model.recurrent.init_hidden(1, dev)

    total_r      = 0.0
    step         = 0
    done         = False
    rewards      = []
    q_vals       = []
    r_preds      = []
    frames       = []
    task_events  = []    # (step, n_tasks, completed_list)
    n_tasks_done = 0

    while not done and step < max_steps:
        # ── Render frame ──────────────────────────────────────────────────
        if record_gif:
            frame = render_frame(env, w=render_w, h=render_h)
            if frame is not None:
                frames.append(frame)

        # ── obs → x ───────────────────────────────────────────────────────
        x = torch.FloatTensor(x_encoder(obs, obs_ref)).unsqueeze(0).to(dev)

        # ── Act ───────────────────────────────────────────────────────────
        if use_mppi:
            a_np, info = planner.act(x, h)
        else:
            a_np, info = planner.act_greedy(x, h)

        # ── Env step ──────────────────────────────────────────────────────
        obs_next, reward, done, env_info = env.step(a_np)

        # Task completion detection
        n_now, completed = inspect_info(env_info)
        if n_now > n_tasks_done:
            task_events.append((step, n_now, completed))
            n_tasks_done = n_now

        # ── Update h ──────────────────────────────────────────────────────
        with torch.no_grad():
            o, _, _ = model.posterior.sample(x, h)
            a_t     = torch.FloatTensor(a_np).unsqueeze(0).to(dev)
            h       = model.recurrent(h, o, a_t)

        total_r += reward
        rewards.append(reward)
        if 'q_val'  in info: q_vals.append(info['q_val'])
        if 'r_pred' in info: r_preds.append(info['r_pred'])

        obs  = obs_next
        step += 1

    # ── Last frame ────────────────────────────────────────────────────────
    if record_gif:
        frame = render_frame(env, w=render_w, h=render_h)
        if frame is not None:
            frames.append(frame)

    # ── Save GIF ──────────────────────────────────────────────────────────
    if record_gif and frames and gif_path:
        save_gif(frames, gif_path, fps=gif_fps)

    return {
        'total_reward': total_r,
        'steps':        step,
        'rewards':      np.array(rewards),
        'q_vals':       np.array(q_vals)   if q_vals   else None,
        'r_preds':      np.array(r_preds)  if r_preds  else None,
        'n_tasks':      n_tasks_done,
        'task_events':  task_events,
        'frames':       frames if record_gif else [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_episode_diagnostics(results: list, out_dir: Path):
    """
    Q value / reward prediction / cumulative reward 시계열 플롯.
    에피소드별 subplot.
    """
    n   = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n), squeeze=False)

    for i, res in enumerate(results):
        ts = np.arange(res['steps'])

        # Cumulative reward
        ax = axes[i, 0]
        cum_r = np.cumsum(res['rewards'])
        ax.plot(ts, cum_r, color='#43A047', lw=2.0)
        for ev_t, n_task, _ in res.get('task_events', []):
            ax.axvline(ev_t, color='#E53935', lw=1.5, ls='--', alpha=0.8)
            ax.text(ev_t, cum_r[min(ev_t, len(cum_r)-1)],
                    f' task{n_task}', fontsize=7, color='#E53935')
        ax.set_title(f"Ep {i+1}  reward={res['total_reward']:.1f}"
                     f"  tasks={res['n_tasks']}", fontsize=9)
        ax.set_xlabel('step'); ax.set_ylabel('cumulative reward')
        ax.spines[['top','right']].set_visible(False)

        # Q value
        ax = axes[i, 1]
        if res['q_vals'] is not None and len(res['q_vals']) > 0:
            ax.plot(ts[:len(res['q_vals'])], res['q_vals'],
                    color='#1E88E5', lw=1.5, alpha=0.8)
            ax.set_ylabel('Q(z, u)', fontsize=8)
            ax.set_title(f"Q mean={res['q_vals'].mean():.3f}", fontsize=9)
        else:
            ax.set_title("Q (no data)", fontsize=9)
        ax.set_xlabel('step')
        ax.spines[['top','right']].set_visible(False)

        # Reward prediction
        ax = axes[i, 2]
        if res['r_preds'] is not None and len(res['r_preds']) > 0:
            ax.plot(ts[:len(res['r_preds'])], res['r_preds'],
                    color='#FB8C00', lw=1.5, alpha=0.8, label='R̂_pen')
            ax.bar(ts[:len(res['rewards'])],
                   res['rewards'], color='#43A047', alpha=0.3,
                   width=1.0, label='GT r')
            ax.set_ylabel('reward', fontsize=8)
            ax.set_title(f"Reward pred mean={res['r_preds'].mean():.4f}",
                         fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.set_title("Reward pred (no data)", fontsize=9)
        ax.set_xlabel('step')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('KODAQ v5 MPPI Episode Diagnostics',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = str(out_dir / 'mppi_diagnostics.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI eval
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',          required=True)
    p.add_argument('--env',           default='kitchen-mixed-v0')
    p.add_argument('--n_ep',          type=int,   default=10)
    p.add_argument('--horizon',       type=int,   default=5)
    p.add_argument('--num_samples',   type=int,   default=512)
    p.add_argument('--num_pi_trajs',  type=int,   default=24)
    p.add_argument('--num_elites',    type=int,   default=64)
    p.add_argument('--temperature',   type=float, default=0.5)
    p.add_argument('--iterations',    type=int,   default=6)
    p.add_argument('--init_std',      type=float, default=2.0)
    p.add_argument('--greedy',        action='store_true',
                   help='Use greedy policy prior instead of MPPI')
    p.add_argument('--goal_ratio',    type=float, default=0.0,
                   help='Fraction of N samples from π_goal LQR (0=disabled, 0.95=recommended)')
    p.add_argument('--num_goal_samples', type=int, default=10,
                   help='Number of z_g samples from π_goal per MPPI step')
    p.add_argument('--gif',           action='store_true',
                   help='Record GIF for each episode')
    p.add_argument('--gif_fps',       type=int,   default=15)
    p.add_argument('--gif_every',     type=int,   default=1,
                   help='Record GIF every N episodes (1=all, 2=every other, ...)')
    p.add_argument('--render_w',      type=int,   default=512)
    p.add_argument('--render_h',      type=int,   default=512)
    p.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',       default='checkpoints/kodaq_v5_lqr/mppi_eval')
    args = p.parse_args()

    import os, sys
    sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
    os.environ.setdefault('MUJOCO_GL', 'egl')

    from models.koopman_cvae import KoopmanCVAE

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── MPPI planner ──────────────────────────────────────────────────────────
    mppi_cfg = MPPIConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_pi_trajs=args.num_pi_trajs,
        num_elites=args.num_elites,
        temperature=args.temperature,
        iterations=args.iterations,
        init_std=args.init_std,
        goal_ratio=args.goal_ratio,
        num_goal_samples=args.num_goal_samples,
    )
    planner = KODAQMPPIPlanner(model, mppi_cfg)

    # ── Env ───────────────────────────────────────────────────────────────────
    import d4rl, gym
    env = gym.make(args.env)

    # ── x_encoder ─────────────────────────────────────────────────────────────
    # obs (60,) + obs_ref (60,) → x (2108,)
    # delta_e = 0 (R3M unavailable; zeros)
    # delta_p = obs[18:60] - obs_ref[18:60]
    # q       = obs[0:9]
    # qdot    = obs[9:18]
    def x_encoder(obs: np.ndarray, obs_ref: np.ndarray) -> np.ndarray:
        delta_e = np.zeros(2048, dtype=np.float32)
        delta_p = (obs[18:60] - obs_ref[18:60]).astype(np.float32)
        q       = obs[0:9].astype(np.float32)
        qdot    = obs[9:18].astype(np.float32)
        return np.concatenate([delta_e, delta_p, q, qdot])

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / 'gifs'
    if args.gif:
        gif_dir.mkdir(exist_ok=True)

    # ── Run episodes ──────────────────────────────────────────────────────────
    results = []
    print(f"\n{'='*57}")
    print(f"KODAQ v5 MPPI Eval  n_ep={args.n_ep}  "
          f"{'greedy' if args.greedy else 'MPPI'}")
    print(f"  H={args.horizon}  N={args.num_samples}  "
          f"elites={args.num_elites}  T={args.temperature}")
    print(f"  GIF={'on (every '+str(args.gif_every)+' ep)' if args.gif else 'off'}")
    print(f"{'='*57}")

    for ep in range(args.n_ep):
        record = args.gif and (ep % args.gif_every == 0)
        gif_path = str(gif_dir / f'ep{ep+1:03d}.gif') if record else None

        res = run_mppi_episode(
            model, planner, env,
            x_encoder=x_encoder,
            device=args.device,
            use_mppi=not args.greedy,
            record_gif=record,
            gif_fps=args.gif_fps,
            gif_path=gif_path,
            render_w=args.render_w,
            render_h=args.render_h,
        )
        results.append(res)

        task_str = ''
        if res['task_events']:
            task_str = '  events=' + str([(t, n) for t, n, _ in res['task_events']])
        print(f"Ep {ep+1:3d}/{args.n_ep}  "
              f"reward={res['total_reward']:.1f}  "
              f"tasks={res['n_tasks']}  "
              f"steps={res['steps']}"
              f"{task_str}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    rewards  = [r['total_reward'] for r in results]
    tasks    = [r['n_tasks']      for r in results]
    n_solved = sum(1 for t in tasks if t >= 1)

    print(f"\n{'='*57}")
    print(f"Summary ({args.n_ep} episodes)")
    print(f"  Reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}"
          f"  max={max(rewards):.1f}  min={min(rewards):.1f}")
    print(f"  Tasks  : {np.mean(tasks):.2f} ± {np.std(tasks):.2f}"
          f"  max={max(tasks)}  (≥1 task: {n_solved}/{args.n_ep})")
    print(f"{'='*57}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(str(out_dir / 'mppi_rewards.npy'), np.array(rewards))
    np.save(str(out_dir / 'mppi_tasks.npy'),   np.array(tasks))

    plot_episode_diagnostics(results, out_dir)
    print(f"\nAll outputs → {out_dir}/")


if __name__ == '__main__':
    main()