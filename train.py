"""
train.py — Training script for KODAC
(Koopman Diagonal Matrix Prior CVAE with Skill-Specific Modes)

Supports: D4RL Adroit (adroit_pen/hammer/door/relocate), synthetic fallback.
Keeps original Trainer / DataLoader structure; updates model, config, logging.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.env_configs import ENV_CONFIGS, build_config
from data.dataset_utils import (
    load_d4rl_trajectories,
    make_synthetic_dataset,
)


# ─────────────────────────────────────────────
#  Dataset loader (unified)
# ─────────────────────────────────────────────

ADROIT_ENVS = ['adroit_pen', 'adroit_hammer', 'adroit_door', 'adroit_relocate']


def load_dataset(args, cfg: KoopmanCVAEConfig):
    """
    Route to correct dataset loader based on --env and --d4rl_quality.

    D4RL quality mapping (dataset_utils.D4RL_ENV_MAP):
      'expert'  → pen-expert-v1   (~500K steps)
      'human'   → pen-human-v1    (~5K steps, motion capture)
      'cloned'  → pen-cloned-v1   (~500K steps, BC policy)

    Returns TensorDataset of (actions (B,T,da), states (B,T,ds))
    """
    if args.env in ADROIT_ENVS:
        print(f"Loading D4RL: env={args.env}, quality={args.d4rl_quality}, "
              f"seq_len={args.seq_len}")
        try:
            dataset = load_d4rl_trajectories(
                env_name        = args.env,
                seq_len         = args.seq_len,
                stride          = args.stride if hasattr(args, 'stride') else None,
                min_episode_len = args.min_episode_len,
                quality         = args.d4rl_quality,
            )
            return dataset
        except Exception as e:
            print(f"D4RL load failed: {e}")
            print("Falling back to synthetic dataset.")

    # Synthetic fallback
    print(f"Using synthetic dataset: action_dim={cfg.action_dim}, "
          f"state_dim={cfg.state_dim}, n_samples={args.n_synthetic}, "
          f"seq_len={args.seq_len}")
    return make_synthetic_dataset(
        action_dim = cfg.action_dim,
        state_dim  = cfg.state_dim,
        n_samples  = args.n_synthetic,
        seq_len    = args.seq_len,
    )


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model: KoopmanCVAE, cfg: KoopmanCVAEConfig, args):
        self.model  = model
        self.cfg    = cfg
        self.args   = args
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = args.lr,
            weight_decay = args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── single epoch ──────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        totals: Dict[str, float] = {}

        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)

            out  = self.model(actions, states)
            loss = out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k, v in out.items():
                if k.startswith('loss'):
                    totals[k] = totals.get(k, 0.0) + v.item()

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        totals: Dict[str, float] = {}

        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)
            out = self.model(actions, states)
            for k, v in out.items():
                if k.startswith('loss'):
                    totals[k] = totals.get(k, 0.0) + v.item()

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    # ── full training loop ────────────────────

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None):
        best_val = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            self.scheduler.step()

            # Build log string — show dominant losses prominently
            log_str = f"Epoch {epoch:4d}"
            # Dominant: pred first
            key_order = ['loss', 'loss_pred', 'loss_recon', 'loss_kl',
                         'loss_eig', 'loss_cst', 'loss_div', 'loss_ent',
                         'loss_decorr']
            for k in key_order:
                if k in train_metrics:
                    log_str += f"  {k}: {train_metrics[k]:.4f}"

            # Validation
            if val_loader is not None and epoch % self.args.eval_freq == 0:
                val_metrics = self.eval_epoch(val_loader)
                val_loss    = val_metrics['loss']
                log_str    += f"  | val_loss: {val_loss:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint('best.pt')

            print(log_str, flush=True)

            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"Training complete. Best val loss: {best_val:.4f}")

    # ── checkpoint ────────────────────────────

    def save_checkpoint(self, name: str):
        path = self.save_dir / name
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg':             self.cfg,
            'args':            vars(self.args),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt  = torch.load(path, map_location=device)
        cfg   = ckpt['cfg']
        model = KoopmanCVAE(cfg)
        model.load_state_dict(ckpt['model_state'])
        return model, cfg


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train KODAC: Koopman Diagonal Matrix Prior CVAE')

    # ── Environment ──────────────────────────
    p.add_argument('--env',         type=str,   default='synthetic',
                   help='Environment key (e.g. adroit_pen, dm_walker, synthetic)')
    p.add_argument('--action_dim',  type=int,   default=None)
    p.add_argument('--state_dim',   type=int,   default=None)
    p.add_argument('--patch_size',  type=int,   default=None)
    p.add_argument('--dt_control',  type=float, default=None)

    # ── Dataset ───────────────────────────────
    p.add_argument('--d4rl_quality',    type=str,  default='human',
                   choices=['expert', 'human', 'cloned'],
                   help='D4RL dataset quality (only for adroit envs)')
    p.add_argument('--seq_len',         type=int,  default=100,
                   help='Sequence length (timesteps) per sample')
    p.add_argument('--stride',          type=int,  default=None,
                   help='Sliding window stride (default: seq_len//2)')
    p.add_argument('--min_episode_len', type=int,  default=30,
                   help='Minimum episode length to keep')
    p.add_argument('--n_synthetic',     type=int,  default=1000,
                   help='Number of synthetic samples (if env=synthetic)')

    # ── Architecture ─────────────────────────
    p.add_argument('--embed_dim',       type=int,   default=None)
    p.add_argument('--state_embed_dim', type=int,   default=None)
    p.add_argument('--gru_hidden_dim',  type=int,   default=None)
    p.add_argument('--mlp_hidden_dim',  type=int,   default=None)
    p.add_argument('--koopman_dim',     type=int,   default=None,
                   help='m: number of Koopman eigenfunction pairs')
    p.add_argument('--num_skills',      type=int,   default=None,
                   help='S: number of discrete skills')
    p.add_argument('--lora_rank',       type=int,   default=None,
                   help='r: low-rank residual rank for β parameterization')
    p.add_argument('--dropout',         type=float, default=None)

    # ── Eigenvalue ────────────────────────────
    p.add_argument('--mu_fixed',   type=float, default=None,
                   help='Fixed real part of eigenvalue (stability decay)')
    p.add_argument('--omega_max',  type=float, default=None,
                   help='Max frequency for eigenvalue initialization grid')

    # ── Loss weights ─────────────────────────
    p.add_argument('--kl_prior',     type=str,   default=None,
                   choices=['koopman', 'standard'],
                   help='KL prior type')
    p.add_argument('--alpha_pred',   type=float, default=None,
                   help='Prediction loss weight (dominant, default 1.0)')
    p.add_argument('--alpha_recon',  type=float, default=None)
    p.add_argument('--beta_kl',      type=float, default=None)
    p.add_argument('--gamma_eig',    type=float, default=None)
    p.add_argument('--delta_cst',    type=float, default=None)
    p.add_argument('--delta_div',    type=float, default=None)
    p.add_argument('--delta_ent',    type=float, default=None)
    p.add_argument('--delta_decorr', type=float, default=None)
    p.add_argument('--pred_steps',   type=int,   default=None,
                   help='Multi-step prediction horizon H')

    # ── Training ─────────────────────────────
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--eval_freq',    type=int,   default=10)
    p.add_argument('--save_freq',    type=int,   default=50)
    p.add_argument('--save_dir',     type=str,   default='checkpoints')
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers',  type=int,   default=2)
    p.add_argument('--val_ratio',    type=float, default=0.1)

    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    cfg  = build_config(args)

    print(f"Config: {cfg}")
    print(f"Eigenfunction patch Δt: {cfg.patch_size * cfg.dt_control * 1000:.1f}ms")
    print(f"Skills: S={cfg.num_skills}, Koopman dim: m={cfg.koopman_dim}")
    print(f"β parameterization: diag + LR(rank={cfg.lora_rank}), "
          f"params/skill: {cfg.koopman_dim + 2*cfg.koopman_dim*cfg.lora_rank}")

    model    = KoopmanCVAE(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    # Parameter breakdown
    phi_params    = sum(p.numel() for p in model.phi_encoder.parameters())
    gru_params    = sum(p.numel() for p in model.skill_gru.parameters())
    skill_params  = sum(p.numel() for p in model.skill_params.parameters())
    eig_params    = sum(p.numel() for p in model.koopman.parameters())
    dec_params    = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Eigenfunction encoder (shared): {phi_params:,}")
    print(f"  Skill GRU posterior (shared):   {gru_params:,}")
    print(f"  Skill parameters V,β (S={cfg.num_skills}):   {skill_params:,}")
    print(f"  Eigenvalues ω (shared):         {eig_params:,}")
    print(f"  Decoder:                        {dec_params:,}")

    # ── Load dataset ──────────────────────────
    dataset = load_dataset(args, cfg)
    print(f"Dataset size: {len(dataset)} samples")

    n_val   = max(1, int(args.val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_set,   batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val} | "
          f"Iter/epoch: {len(train_loader)}")

    # ── Train ─────────────────────────────────
    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)