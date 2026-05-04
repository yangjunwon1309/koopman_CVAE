"""
train.py — KODAQ v5 Training Script
=====================================

v5 changes over v4:
  - After each optimizer step: model.soft_update_target_Q()  (EMA target Q)
  - LOG_KEYS extended: loss_reward, loss_q, loss_pi, rho, q_scale
  - rewards = step reward {0,1} (not accumulated)
  - Policy prior training starts at phase 2 (controlled inside model)
  - wandb logging (--wandb_project / --wandb_run, iql_koopman.py 패턴)

Phase schedule:
  Phase 1: WM rec + L_R + L_Q   (world model + Q/reward heads, no pi)
  Phase 2: + L_dyn + L_skill + L_pi
  Phase 3: + L_reg

nohup example:
  mkdir -p logs checkpoints/kodaq_v5
  nohup python train_kodaq.py \\
      --env kitchen_mixed \\
      --wandb_project kodaq \\
      --wandb_run v5_kitchen_run1 \\
      --save_dir checkpoints/kodaq_v5 \\
      --epochs 400 \\
      --device cuda \\
  > logs/train_v5.log 2>&1 &
  echo $! > logs/train_v5.pid
  tail -f logs/train_v5.log
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.env_configs import build_config
from data.dataset_utils import (
    load_kodaq_dataset,
    make_synthetic_dataset,
)

# wandb optional
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Goal-Z Dataset Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class GoalZDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps KODAQWindowDataset to inject goal_z_seq into each sample.

    KODAQWindowDataset returns dict with 'x_seq', 'actions', etc.
    and also stores the global timestep range of each window
    via dataset.windows[i] = (global_start, global_end).

    goal_latent_map.npz stores:
      ep_starts: (N_ep,) int
      str(ep_start): (L, m) float32  — goal_z per global timestep

    We build a flat array goal_z_global[t] = z* at global timestep t,
    then slice [global_start:global_end] per window.
    """

    def __init__(self, base_ds, goal_z_path: str, koopman_dim: int):
        self.base_ds     = base_ds
        self.koopman_dim = koopman_dim

        # Load goal_z_map and build global flat array
        data      = np.load(goal_z_path, allow_pickle=True)
        ep_starts = data['ep_starts'].astype(int)

        # Find total global timesteps needed
        max_t = 0
        for k in ep_starts:
            arr = data[str(k)]
            max_t = max(max_t, int(k) + arr.shape[0])

        # Build flat array: goal_z_global[t] = z* at global t
        self.goal_z_global = np.zeros((max_t, koopman_dim), dtype=np.float32)
        for k in ep_starts:
            arr = data[str(k)]   # (L, m)
            t0  = int(k)
            self.goal_z_global[t0:t0 + arr.shape[0]] = arr

        print(f"  GoalZDatasetWrapper: global_steps={max_t}  "
              f"ep_starts={len(ep_starts)}  m={koopman_dim}")

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]

        # Get global window range from base dataset
        # KODAQWindowDataset stores windows as (global_start, global_end)
        if hasattr(self.base_ds, 'windows'):
            t0, t1 = self.base_ds.windows[idx]
        elif hasattr(self.base_ds, 'dataset') and hasattr(self.base_ds.dataset, 'windows'):
            # Subset wrapper (from random_split)
            real_idx = self.base_ds.indices[idx]
            t0, t1   = self.base_ds.dataset.windows[real_idx]
        else:
            # Fallback: no window info → return zeros (Mode D disabled)
            seq_len = sample['x_seq'].shape[0] if isinstance(sample, dict) else sample[0].shape[0]
            sample['goal_z_seq'] = torch.zeros(seq_len, self.koopman_dim)
            return sample

        # Slice goal_z for this window
        t1_clip = min(t1, len(self.goal_z_global))
        goal_z  = self.goal_z_global[t0:t1_clip]

        # Pad if needed (window might slightly exceed pre-computed map)
        seq_len = sample['x_seq'].shape[0] if isinstance(sample, dict) else sample[0].shape[0]
        if len(goal_z) < seq_len:
            pad = np.zeros((seq_len - len(goal_z), self.koopman_dim), dtype=np.float32)
            goal_z = np.concatenate([goal_z, pad], axis=0)
        else:
            goal_z = goal_z[:seq_len]

        if isinstance(sample, dict):
            sample['goal_z_seq'] = torch.FloatTensor(goal_z)
        else:
            # tuple/list: append goal_z as extra element
            sample = list(sample) + [torch.FloatTensor(goal_z)]

        return sample




# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_d4rl_name(env_key: str) -> str:
    _MAP = {
        'kitchen_complete': 'kitchen-complete-v0',
        'kitchen_partial':  'kitchen-partial-v0',
        'kitchen_mixed':    'kitchen-mixed-v0',
        'adroit_pen':       'pen-human-v1',
        'adroit_hammer':    'hammer-human-v1',
        'adroit_door':      'door-human-v1',
        'adroit_relocate':  'relocate-human-v1',
    }
    return _MAP.get(env_key, env_key)


def load_dataset(args, cfg: KoopmanCVAEConfig):
    if args.env == 'synthetic':
        print(f"Synthetic dataset: n={args.n_synthetic}  seq_len={args.seq_len}")
        return make_synthetic_dataset(
            n_samples=args.n_synthetic,
            seq_len=args.seq_len,
            K=cfg.num_skills,
        )
    try:
        ds = load_kodaq_dataset(
            env_name=_resolve_d4rl_name(args.env),
            seq_len=args.seq_len,
            stride=args.stride,
            use_r3m=not args.no_r3m,
            K=cfg.num_skills,
            out_dir=args.skill_dir,
            pca_dim=args.pca_dim,
            device=args.device,
            mode='window',
            reward_crop=None if args.reward_crop < 0 else args.reward_crop,
        )
        # goal_z_seq is computed on-the-fly from skill_labels inside model.forward()
        # No external npz needed: skill_labels already in dataset (4-tuple)
        return ds
    except Exception as e:
        print(f"Dataset load failed ({e}). Falling back to synthetic.")
        return make_synthetic_dataset(K=cfg.num_skills)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    LOG_KEYS = [
        # v4 world model losses
        'loss', 'loss_wm', 'loss_rec', 'loss_dyn', 'loss_skill',
        'loss_reg', 'loss_stab',
        'loss_rec_delta_e', 'loss_rec_delta_p', 'loss_rec_q', 'loss_rec_qdot',
        # v5 new head losses
        'loss_reward', 'loss_q', 'loss_pi', 'loss_goal',
        # v5 diagnostics
        'rho', 'q_scale',
    ]

    def __init__(self, model: KoopmanCVAE, cfg: KoopmanCVAEConfig, args):
        self.model  = model
        self.cfg    = cfg
        self.args   = args
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.save_dir     = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.phase2_epoch       = args.phase2_epoch
        self.phase3_epoch       = args.phase3_epoch
        self.freeze_world_model = getattr(args, 'freeze_world_model', False)
        # Two-stage resume: 'wm' = WM fine-tune only, 'heads' = Q/R/pi only
        self.resume_stage       = getattr(args, 'resume_stage', None)

        self.use_wandb = (
            _WANDB_AVAILABLE
            and args.wandb_project is not None
        )

    # ── Phase control ────────────────────────────────────────────────────────

    def _maybe_update_phase(self, epoch: int):
        if epoch == self.phase2_epoch:
            self.model.set_phase(2)
            if self.use_wandb:
                wandb.log({'phase': 2}, step=epoch)
        elif epoch == self.phase3_epoch:
            self.model.set_phase(3)
            if self.use_wandb:
                wandb.log({'phase': 3}, step=epoch)

    # ── Batch forward ────────────────────────────────────────────────────────

    def _forward_batch(self, batch) -> Dict:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                x_seq, actions, skill_labels, rewards = batch
            else:
                x_seq, actions, skill_labels = batch
                rewards = None
            mask = None
        else:
            x_seq        = batch['x_seq']
            actions      = batch['actions']
            skill_labels = batch['skill_labels']
            mask         = batch.get('mask', None)
            rewards      = batch.get('rewards', None)

        x_seq        = x_seq.to(self.device)
        actions      = actions.to(self.device)
        skill_labels = skill_labels.to(self.device)
        if mask    is not None: mask    = mask.to(self.device)
        if rewards is not None: rewards = rewards.to(self.device)

        # goal_z_seq is computed on-the-fly inside model.forward()
        # from skill_labels + x_batch → no need to pass from dataset
        return self.model(x_seq, actions, skill_labels, mask, rewards)

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _accumulate(self, totals: Dict, out: Dict):
        for k in self.LOG_KEYS:
            if k in out:
                v = out[k]
                totals[k] = totals.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )

    def _frozen_params(self):
        """Yield parameters of world model modules (frozen in resume mode)."""
        yield from self._wm_params()

    def _wm_modules(self):
        return [
            self.model.action_encoder,
            self.model.posterior,
            self.model.recurrent,
            self.model.skill_prior,
            self.model.koopman,
            self.model.decoder,
        ]

    def _wm_params(self):
        """Yield world-model parameters."""
        for mod in self._wm_modules():
            yield from mod.parameters()

    def _head_modules(self):
        return [
            self.model.reward_head,
            self.model.reward_ensemble_head,
            self.model.q_head,
            self.model.policy_prior,
            self.model.goal_proposal,
        ]

    def _head_params(self):
        """Yield reward/Q/policy head parameters."""
        for mod in self._head_modules():
            yield from mod.parameters()

    def _set_wm_requires_grad(self, flag: bool):
        for p in self._wm_params():
            p.requires_grad_(flag)

    def _set_head_requires_grad(self, flag: bool):
        for p in self._head_params():
            p.requires_grad_(flag)

    def train_epoch(self, loader) -> Dict:
        self.model.train()
        self.model.q_head_target.eval()   # target Q stays in eval always

        totals = {}
        for batch in loader:
            out  = self._forward_batch(batch)
            loss = out['loss']

            self.optimizer.zero_grad()
            loss.backward()

            # When world model is frozen, zero out gradients of frozen params
            # so grad_norm clipping is not skewed by zero grads.
            if self.freeze_world_model:
                for p in self._frozen_params():
                    if p.grad is not None:
                        p.grad.zero_()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # EMA soft update of target Q after every gradient step
            self.model.soft_update_target_Q()

            self._accumulate(totals, out)

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader) -> Dict:
        self.model.eval()
        totals = {}
        for batch in loader:
            out = self._forward_batch(batch)
            self._accumulate(totals, out)
        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    # ── Main loop ────────────────────────────────────────────────────────────

    def _run_loop(self, train_loader, val_loader,
                  n_epochs: int, epoch_offset: int,
                  stage_tag: str, t0: float) -> float:
        """
        Inner training loop for one stage.
        epoch_offset: global epoch counter offset (for wandb step continuity).
        Returns best val loss.
        """
        best_val = float('inf')
        for local_ep in range(1, n_epochs + 1):
            global_ep = epoch_offset + local_ep
            self._maybe_update_phase(global_ep)
            # Notify model of current epoch for warmup gating
            self.model.set_current_epoch(global_ep)
            t_ep = time.time()

            metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            if self.use_wandb:
                wandb.log({f'{stage_tag}/{k}': v for k, v in metrics.items()},
                          step=global_ep)

            val_metrics = {}
            if val_loader and local_ep % self.args.eval_freq == 0:
                val_metrics = self.eval_epoch(val_loader)
                if self.use_wandb:
                    wandb.log({f'{stage_tag}_val/{k}': v
                               for k, v in val_metrics.items()}, step=global_ep)
                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(f'best_{stage_tag}.pt')

            phase   = self.model.cfg.phase
            ep_sec  = time.time() - t_ep
            tot_min = (time.time() - t0) / 60.0

            # Goal proposal phase tag
            warmup_ep = getattr(self.model.cfg, 'warmup_goal_epochs', 0)
            use_goal  = getattr(self.model.cfg, 'use_goal_proposal', False)
            if use_goal:
                gp_tag = 'B' if global_ep >= warmup_ep else 'A'
            else:
                gp_tag = '-'

            line = (f"[{stage_tag}|Ph{phase}|G{gp_tag}] Ep {local_ep:4d}/{n_epochs}"
                    f"  {ep_sec:.1f}s  ({tot_min:.0f}m)")

            for k in ['loss', 'loss_wm', 'loss_rec', 'loss_dyn',
                      'loss_skill', 'loss_reg']:
                if metrics.get(k, 0.0) != 0.0:
                    line += f"  {k.replace('loss_','')[:4]}={metrics[k]:.4f}"
            for k in ['loss_reward', 'loss_q', 'loss_pi']:
                if k in metrics and metrics.get(k, 0.0) != 0.0:
                    line += f"  {k.replace('loss_','')[:4]}={metrics[k]:.4f}"
            # loss_goal: always show when use_goal_proposal (even if 0 in Phase A)
            if use_goal and 'loss_goal' in metrics:
                line += f"  goal={metrics['loss_goal']:.4f}"
            if 'rho' in metrics:
                line += f"  rho={metrics['rho']:.3f}"
            if 'q_scale' in metrics:
                line += f"  Qs={metrics['q_scale']:.2f}"
            if val_metrics:
                line += f"  | val={val_metrics.get('loss', 0):.4f}"
            print(line, flush=True)

            # Phase transition notice
            if use_goal and global_ep == warmup_ep:
                print(f"  >>> [Goal Proposal] Phase A→B: π_goal Q-maximize ACTIVATED "
                      f"(ep {global_ep})", flush=True)

            if local_ep % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{stage_tag}_{local_ep:04d}.pt')

        self.save_checkpoint(f'final_{stage_tag}.pt')
        return best_val

    def _rebuild_optimizer(self, param_iter, lr: float):
        """Replace optimizer + scheduler with new ones for a fresh stage."""
        params = list(param_iter)
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=self.args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.resume_epochs_heads
        )

    def train(self, train_loader, val_loader=None):
        if self.use_wandb:
            try:
                wandb.init(project=self.args.wandb_project,
                           name=self.args.wandb_run or None,
                           config=vars(self.args))
                print(f"[wandb] project={self.args.wandb_project}  "
                      f"run={wandb.run.name}  url={wandb.run.url}",
                      flush=True)
            except Exception as e:
                print(f"[wandb] WARNING: init failed — {e}", flush=True)
                print(f"[wandb] Hint: run 'wandb whoami' to verify entity. "
                      f"Disabling wandb.", flush=True)
                self.use_wandb = False
        t0 = time.time()

        # ── Two-stage resume ──────────────────────────────────────────────────
        if self.resume_stage is not None:
            self._train_two_stage(train_loader, val_loader, t0)
        else:
            # ── Normal (non-resume) single loop ──────────────────────────────
            self.args.epochs = getattr(self.args, 'epochs', 400)
            best = self._run_loop(train_loader, val_loader,
                                  n_epochs=self.args.epochs,
                                  epoch_offset=0,
                                  stage_tag='train', t0=t0)
            print(f"\nDone. best_val={best:.4f}  "
                  f"time={(time.time()-t0)/60:.1f}m", flush=True)

        if self.use_wandb:
            wandb.finish()

    def _train_two_stage(self, train_loader, val_loader, t0: float):
        """
        Stage 1 — WM fine-tune (resume_epochs_wm epochs):
          - ALL WM params unfrozen
          - Head params frozen  (lambda_reward/q/pi forced to 0 in model cfg)
          - lr = resume_lr_wm

        Stage 2 — Head training (resume_epochs_heads epochs):
          - WM params frozen
          - Only reward_ensemble_head / q_head / policy_prior trainable
          - lambda_reward/q/pi restored from args
          - lr = resume_lr_heads  (typically smaller)
          - optimizer rebuilt fresh (no momentum from stage 1)
        """
        n_wm    = self.args.resume_epochs_wm
        n_heads = self.args.resume_epochs_heads
        lr_wm   = self.args.resume_lr_wm
        lr_heads = self.args.resume_lr_heads

        # ── Stage 1: WM fine-tune ─────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print(f"[Stage 1] WM fine-tune  {n_wm} epochs  lr={lr_wm}", flush=True)
        print(f"  WM unfrozen,  heads frozen,  lambda_R/Q/pi = 0", flush=True)
        print(f"{'='*60}", flush=True)

        # Unfreeze WM, freeze heads
        self._set_wm_requires_grad(True)
        self._set_head_requires_grad(False)

        # Zero out head loss weights so WM loss is sole signal
        saved_lR  = self.model.cfg.lambda_reward
        saved_lQ  = self.model.cfg.lambda_q
        saved_lpi = self.model.cfg.lambda_pi
        self.model.cfg.lambda_reward = 0.0
        self.model.cfg.lambda_q      = 0.0
        self.model.cfg.lambda_pi     = 0.0
        # Phase 3 for full WM loss (L_rec + L_dyn + L_skill + L_reg)
        self.model.cfg.phase = 3
        self.phase2_epoch = 0
        self.phase3_epoch = 0

        # Rebuild optimizer for WM params only
        self._rebuild_optimizer(self._wm_params(), lr=lr_wm)
        # Override scheduler T_max for stage 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_wm
        )
        self.freeze_world_model = False  # grad zeroing off for stage 1

        self._run_loop(train_loader, val_loader,
                       n_epochs=n_wm, epoch_offset=0,
                       stage_tag='wm', t0=t0)

        # ── Stage 2: Head training ────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print(f"[Stage 2] Head training  {n_heads} epochs  lr={lr_heads}", flush=True)
        print(f"  WM frozen,  heads trainable,  lambda_R={saved_lR} ", flush=True)
        print(f"  lambda_Q={saved_lQ}  lambda_pi={saved_lpi}", flush=True)
        print(f"{'='*60}", flush=True)

        # Freeze WM, unfreeze heads
        self._set_wm_requires_grad(False)
        self._set_head_requires_grad(True)

        # Restore head loss weights
        self.model.cfg.lambda_reward = saved_lR
        self.model.cfg.lambda_q      = saved_lQ
        self.model.cfg.lambda_pi     = saved_lpi
        # Phase stays 3 (all WM losses computed but lambda=0 for R/Q/pi in wm)
        # For head stage we still need phase>=2 for policy prior activation
        self.model.cfg.phase = 3

        # Fresh optimizer for head params only (no stale momentum from stage 1)
        self._rebuild_optimizer(self._head_params(), lr=lr_heads)
        self.freeze_world_model = True   # zero WM grads if any leak through

        self._run_loop(train_loader, val_loader,
                       n_epochs=n_heads, epoch_offset=n_wm,
                       stage_tag='heads', t0=t0)

        print(f"\nTwo-stage resume done.  "
              f"time={(time.time()-t0)/60:.1f}m", flush=True)

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, name: str):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg':             self.cfg,
            'args':            vars(self.args),
            'phase':           self.model.cfg.phase,
            'scale_tracker':   self.model.scale_tracker.state_dict(),
        }, self.save_dir / name)
        print(f"  -> saved {self.save_dir / name}", flush=True)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt  = torch.load(path, map_location=device)
        model = KoopmanCVAE(ckpt['cfg'])
        model.load_state_dict(ckpt['model_state'])
        model.cfg.phase = ckpt.get('phase', 1)
        if 'scale_tracker' in ckpt:
            model.scale_tracker.load_state_dict(ckpt['scale_tracker'])
        return model, ckpt['cfg']


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train KODAQ v5')

    # env / data
    p.add_argument('--env',          type=str,   default='kitchen_mixed')
    p.add_argument('--seq_len',      type=int,   default=64)
    p.add_argument('--stride',       type=int,   default=None)
    p.add_argument('--no_r3m',       action='store_true')
    p.add_argument('--pca_dim',      type=int,   default=64)
    p.add_argument('--skill_dir',    type=str,   default='checkpoints/skill_pretrain')
    p.add_argument('--n_synthetic',  type=int,   default=2000)
    p.add_argument('--reward_crop',  type=float, default=2.0,
                   help='Use each episode only until cumulative reward reaches this value. '
                        'Set negative to disable cropping.')

    # architecture
    p.add_argument('--koopman_dim',   type=int,   default=None)
    p.add_argument('--gru_hidden',    type=int,   default=None)
    p.add_argument('--action_latent', type=int,   default=None)
    p.add_argument('--num_skills',    type=int,   default=None)
    p.add_argument('--mlp_hidden',    type=int,   default=None)
    p.add_argument('--enc_layers',    type=int,   default=None)
    p.add_argument('--dec_layers',    type=int,   default=None)
    p.add_argument('--dropout',       type=float, default=None)

    # v4 loss
    p.add_argument('--lambda1',  type=float, default=None)
    p.add_argument('--lambda2',  type=float, default=None)
    p.add_argument('--lambda3',  type=float, default=None)
    p.add_argument('--lambda4',  type=float, default=None)
    p.add_argument('--no_multistep_dyn', action='store_true')
    p.add_argument('--dyn_horizon',  type=int,   default=8)
    p.add_argument('--dyn_alpha',    type=float, default=0.95)

    # v5 Q / Reward / Policy
    p.add_argument('--num_bins',      type=int,   default=16,
                   help='Two-Hot bins. Range [0, v_max]')
    p.add_argument('--v_max',         type=float, default=5.0,
                   help='Bin upper bound: r_step_max(1) + gamma*Q_max(4)=4.99')
    p.add_argument('--num_q',         type=int,   default=2)
    p.add_argument('--tau',           type=float, default=0.005)
    p.add_argument('--gamma',         type=float, default=0.99)
    p.add_argument('--entropy_coef',  type=float, default=0.01)
    p.add_argument('--log_std_min',   type=float, default=-5.0)
    p.add_argument('--log_std_max',   type=float, default=2.0)
    p.add_argument('--lambda_reward', type=float, default=1.0)
    p.add_argument('--lambda_q',      type=float, default=1.0)
    p.add_argument('--lambda_pi',     type=float, default=0.1)

    # phase
    p.add_argument('--phase2_epoch',  type=int,   default=60)
    p.add_argument('--phase3_epoch',  type=int,   default=160)

    # training
    p.add_argument('--epochs',        type=int,   default=400)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--eval_freq',     type=int,   default=10)
    p.add_argument('--save_freq',     type=int,   default=50)
    p.add_argument('--save_dir',      type=str,   default='checkpoints/kodaq_v5')
    p.add_argument('--device',        type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers',   type=int,   default=2)
    p.add_argument('--val_ratio',     type=float, default=0.1)

    # ── resume (two-stage fine-tuning) ──────────────────────────────────
    p.add_argument('--resume_ckpt',        type=str,   default=None,
                   help='Checkpoint to resume from (v4 or v5 final.pt).')
    p.add_argument('--freeze_world_model', action='store_true',
                   help='Freeze WM in stage 2 (auto-set when using two-stage).')

    # Two-stage resume epochs + lr
    p.add_argument('--resume_epochs_wm',   type=int,   default=100,
                   help='Stage 1: WM fine-tune epochs.')
    p.add_argument('--resume_epochs_heads', type=int,  default=100,
                   help='Stage 2: reward/Q/policy head training epochs.')
    p.add_argument('--resume_lr_wm',       type=float, default=3e-5,
                   help='Stage 1 learning rate (WM fine-tune).')
    p.add_argument('--resume_lr_heads',    type=float, default=1e-4,
                   help='Stage 2 learning rate (head training).')

    # Legacy single-stage compat (still works)
    p.add_argument('--resume_epochs',      type=int,   default=100)
    p.add_argument('--resume_lr',          type=float, default=1e-4)

    p.add_argument('--use_goal_proposal',    action='store_true',
                   help='Train pi_goal for goal-conditioned LQR.')
    p.add_argument('--goal_kl_weight',       type=float, default=0.1)
    p.add_argument('--lambda_goal',          type=float, default=0.1)
    p.add_argument('--warmup_goal_epochs',   type=int,   default=100,
                   help='Phase A epochs: z_g=z_g^seg fixed. '
                        'Phase B starts at this epoch: z_g~pi_goal active.')
    p.add_argument('--no_recon_delta_e',     action='store_true',
                   help='Disable R3M feature recon to free encoder capacity.')
    p.add_argument('--use_lqr_policy',      action='store_true',
                   help='Use LQR rollout for Q target (Mode D). '
                        'Requires --goal_z_path and --u_bounds_path.')
    p.add_argument('--lqr_horizon',          type=int,   default=4,
                   help='H-step LQR rollout for Q target.')
    p.add_argument('--goal_z_path',          type=str,   default=None,
                   help='Path to pre-computed goal_latent_map.npz.')
    p.add_argument('--u_bounds_path',        type=str,   default=None,
                   help='Path to u_bounds.npz from survey.')
    p.add_argument('--lqr_Q_scale',          type=float, default=1.0)
    p.add_argument('--lqr_R_scale',          type=float, default=10.0)
    p.add_argument('--td_horizon',           type=int,   default=4,
                   help='H-step rollout horizon for MOPO TD target (4 or 8).')
    p.add_argument('--mopo_beta',          type=float, default=0.0,
                   help='MOPO penalty: mean - beta*std.')
    p.add_argument('--reward_ensemble_n',  type=int,   default=5,
                   help='Number of reward ensemble members.')

    # wandb  (iql_koopman.py 패턴: project + run 두 개만)
    p.add_argument('--wandb_project', type=str, default=None,
                   help='wandb project name. None=disabled.')
    p.add_argument('--wandb_run',     type=str, default=None,
                   help='Run name shown in wandb UI.')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # nohup 환경에서 stdout이 tty가 아닐 때 line-buffering 강제
    # 이렇게 해야 tail -f logs/train.log로 실시간 확인 가능
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    cfg = build_config(args)

    # v5 config overrides
    cfg.num_bins      = args.num_bins
    cfg.v_max         = args.v_max
    cfg.num_q         = args.num_q
    cfg.tau           = args.tau
    cfg.gamma         = args.gamma
    cfg.entropy_coef  = args.entropy_coef
    cfg.log_std_min   = args.log_std_min
    cfg.log_std_max   = args.log_std_max
    cfg.lambda_reward = args.lambda_reward
    cfg.lambda_q      = args.lambda_q
    cfg.lambda_pi     = args.lambda_pi

    print("=" * 65, flush=True)
    print("KODAQ v5")
    print("=" * 65, flush=True)
    print(f"  env={args.env}  device={args.device}")
    print(f"  koopman_dim={cfg.koopman_dim}  gru_hidden={cfg.gru_hidden}"
          f"  num_skills={cfg.num_skills}")
    print(f"  num_bins={cfg.num_bins}  v_max={cfg.v_max}"
          f"  num_q={cfg.num_q}  tau={cfg.tau}")
    print(f"  gamma={cfg.gamma}  entropy_coef={cfg.entropy_coef}")
    print(f"  loss: lam_R={cfg.lambda_reward} lam_Q={cfg.lambda_q}"
          f" lam_pi={cfg.lambda_pi}")
    print(f"  loss: lam1={cfg.lambda1} lam2={cfg.lambda2}"
          f" lam3={cfg.lambda3} lam4={cfg.lambda4}")
    print(f"  phase: 1->{args.phase2_epoch}  2->{args.phase3_epoch}"
          f"  3->{args.epochs}")
    print(f"  wandb: {args.wandb_project or 'off'}")
    print("=" * 65, flush=True)

    # ── Resume: load WM from checkpoint, reset Q/reward_ens/policy ─────────────
    is_resume = (args.resume_ckpt is not None)
    if is_resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume_ckpt}", flush=True)
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')

        # ── Rebuild cfg from checkpoint ────────────────────────────────────
        # ckpt['cfg'] may be a v4 KoopmanCVAEConfig (missing v5 fields).
        # Strategy: import v5 KoopmanCVAEConfig, copy all fields that exist
        # in the checkpoint cfg, then fill in v5 defaults + arg overrides.
        ckpt_cfg_raw = ckpt['cfg']

        # Start from v5 default config
        from models.koopman_cvae import KoopmanCVAEConfig as V5Config
        resume_cfg = V5Config()

        # Copy all architecture fields that exist in ckpt cfg (v4 fields)
        v4_fields = [
            'dim_delta_e', 'dim_delta_p', 'dim_q', 'dim_qdot',
            'action_dim', 'state_dim',
            'koopman_dim', 'gru_hidden', 'action_latent', 'num_skills',
            'mlp_hidden', 'enc_layers', 'dec_layers', 'dropout',
            'lambda1', 'lambda2', 'lambda3', 'lambda4',
            'alpha_delta_e', 'alpha_delta_p', 'alpha_q', 'alpha_qdot',
            'dt_control', 'multistep_dyn', 'dyn_horizon', 'dyn_alpha',
        ]
        for f in v4_fields:
            if hasattr(ckpt_cfg_raw, f):
                setattr(resume_cfg, f, getattr(ckpt_cfg_raw, f))

        # Apply v5 + resume-specific overrides from args
        resume_cfg.use_ensemble_reward = True
        resume_cfg.td_horizon          = args.td_horizon
        resume_cfg.mopo_beta           = args.mopo_beta
        resume_cfg.reward_ensemble_n   = args.reward_ensemble_n
        resume_cfg.num_bins            = args.num_bins
        resume_cfg.v_min               = 0.0
        resume_cfg.v_max               = args.v_max
        resume_cfg.num_q               = args.num_q
        resume_cfg.tau                 = args.tau
        resume_cfg.gamma               = args.gamma
        resume_cfg.entropy_coef        = args.entropy_coef
        resume_cfg.log_std_min         = args.log_std_min
        resume_cfg.log_std_max         = args.log_std_max
        resume_cfg.use_lqr_policy      = args.use_lqr_policy
        resume_cfg.lqr_horizon         = args.lqr_horizon
        resume_cfg.use_goal_proposal   = args.use_goal_proposal
        resume_cfg.goal_kl_weight      = args.goal_kl_weight
        resume_cfg.lambda_goal         = args.lambda_goal
        resume_cfg.warmup_goal_epochs  = args.warmup_goal_epochs
        resume_cfg.recon_delta_e       = not args.no_recon_delta_e
        resume_cfg.lambda_reward       = args.lambda_reward
        resume_cfg.lambda_q            = args.lambda_q
        resume_cfg.lambda_pi           = args.lambda_pi
        resume_cfg.phase               = 3
        cfg = resume_cfg

        print(f"  cfg rebuilt: koopman_dim={cfg.koopman_dim}"
              f"  gru_hidden={cfg.gru_hidden}"
              f"  num_skills={cfg.num_skills}", flush=True)

        # ── Build v5 model with resumed cfg ───────────────────────────────
        model = KoopmanCVAE(cfg)

        # ── Load WM weights, skip v5-only heads ───────────────────────────
        # v4 state dict keys: action_encoder.*, posterior.*, recurrent.*,
        #                     skill_prior.*, koopman.*, decoder.*
        # v5 adds:            reward_head.*, reward_ensemble_head.*,
        #                     q_head.*, q_head_target.*, _detach_q_head.*,
        #                     policy_prior.*
        full_sd = ckpt['model_state']
        resume_heads = {
            'reward_head', 'reward_ensemble_head',
            'q_head', 'q_head_target', '_detach_q_head',
            'policy_prior',
        }
        # Also exclude v4-only subkeys inside decoder (e.g. decoder.head_reward)
        # which don't exist in v5 decoder.
        filtered_sd = {}
        for k, v in full_sd.items():
            top = k.split('.')[0]
            if top in resume_heads:
                continue
            # v4 decoder had head_reward; v5 decoder does not
            if k.startswith('decoder.head_reward'):
                continue
            filtered_sd[k] = v

        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        wm_loaded = [k for k in filtered_sd if k.split('.')[0] not in resume_heads]
        print(f"  WM weights loaded: {len(filtered_sd)} tensors", flush=True)
        print(f"  Missing (new v5 heads, expected): {len(missing)}", flush=True)
        if unexpected:
            print(f"  Unexpected (check): {unexpected[:5]}", flush=True)

        # ── Freeze world model parameters ─────────────────────────────────
        if args.freeze_world_model:
            wm_mods = [
                model.action_encoder, model.posterior, model.recurrent,
                model.skill_prior, model.koopman, model.decoder,
            ]
            n_frozen = 0
            for mod in wm_mods:
                for p in mod.parameters():
                    p.requires_grad_(False)
                    n_frozen += p.numel()
            print(f"  Frozen {n_frozen:,} WM params (encoder/GRU/Koopman/decoder).",
                  flush=True)

        # ── Two-stage resume schedule ─────────────────────────────────────
        args.resume_stage  = 'two_stage'
        args.phase2_epoch  = 0    # phase 3 immediately in both stages
        args.phase3_epoch  = 0
        args.epochs        = args.resume_epochs_wm + args.resume_epochs_heads
        args.lr            = args.resume_lr_wm
        print(f"  Two-stage resume:", flush=True)
        print(f"    Stage 1 (WM fine-tune):  {args.resume_epochs_wm} ep"
              f"  lr={args.resume_lr_wm}", flush=True)
        print(f"    Stage 2 (Head training): {args.resume_epochs_heads} ep"
              f"  lr={args.resume_lr_heads}", flush=True)
        print(f"    H={cfg.td_horizon}  beta={cfg.mopo_beta}"
              f"  N_ens={cfg.reward_ensemble_n}", flush=True)
        if args.use_goal_proposal:
            print(f"    π_goal: Phase A (ep 1~{args.warmup_goal_epochs}) = "
                  f"fixed z_g^seg  |  "
                  f"Phase B (ep {args.warmup_goal_epochs+1}~) = π_goal active",
                  flush=True)
    else:
        model = KoopmanCVAE(cfg)
        args.resume_stage = None

    # ── LQR planner setup (Mode D) ────────────────────────────────────────
    if getattr(args, 'use_lqr_policy', False):
        if args.goal_z_path is None:
            print("[WARNING] --use_lqr_policy requires --goal_z_path. "
                  "Falling back to policy prior.")
            if is_resume: resume_cfg.use_lqr_policy = False
            else: cfg.use_lqr_policy = False
        else:
            from lqr_koopman import KODAQLQRPlanner, LQRConfig,                 KODAQLQRPlanner as _Planner
            lqr_cfg = LQRConfig(
                Q_scale=args.lqr_Q_scale,
                R_scale=args.lqr_R_scale,
            )
            lqr_planner = KODAQLQRPlanner(model, lqr_cfg)
            if args.u_bounds_path and Path(args.u_bounds_path).exists():
                lqr_planner.load_u_bounds(args.u_bounds_path)
            lqr_planner.precompute_gains(H=args.lqr_horizon)
            model.set_lqr_planner(lqr_planner)
            print(f"[LQR] Planner ready.  "
                  f"goal_z_path={args.goal_z_path}  "
                  f"lqr_horizon={args.lqr_horizon}", flush=True)

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen_total = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable params: {n_total:,}  Frozen: {n_frozen_total:,}")
    head_list = [
        ('posterior',          model.posterior),
        ('recurrent',          model.recurrent),
        ('skill_prior',        model.skill_prior),
        ('koopman',            model.koopman),
        ('decoder',            model.decoder),
        ('action_enc',         model.action_encoder),
        ('reward_head',        model.reward_head),
        ('reward_ens_head',    model.reward_ensemble_head),
        ('q_head',             model.q_head),
        ('policy_prior',       model.policy_prior),
    ]
    for nm, mod in head_list:
        n_req = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        n_frz = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
        flag  = ' [FROZEN]' if n_frz > 0 and n_req == 0 else ''
        print(f"  {nm:<18}: {n_req:>8,} trainable{flag}")
    print(flush=True)

    dataset = load_dataset(args, cfg)
    print(f"Dataset: {len(dataset)} samples", flush=True)

    n_val   = max(1, int(args.val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train={n_train}  Val={n_val}  iter/ep={len(train_loader)}\n",
          flush=True)

    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)
