"""
train.py — Training script for KODAC
(Koopman Diagonal Matrix Prior CVAE with Skill-Specific Modes)

Supports: D4RL Adroit (adroit_pen/hammer/door/relocate), synthetic fallback.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.envs_config import ENV_CONFIGS, build_config
from data.dataset_utils import (
    load_d4rl_trajectories,
    make_synthetic_dataset,
)


# ─────────────────────────────────────────────
#  Dataset loader
# ─────────────────────────────────────────────

ADROIT_ENVS = ['adroit_pen', 'adroit_hammer', 'adroit_door', 'adroit_relocate']


def load_dataset(args, cfg: KoopmanCVAEConfig):
    if args.env in ADROIT_ENVS:
        print(f"Loading D4RL: env={args.env}, quality={args.d4rl_quality}, "
              f"seq_len={args.seq_len}")
        try:
            return load_d4rl_trajectories(
                env_name        = args.env,
                seq_len         = args.seq_len,
                stride          = args.stride,
                min_episode_len = args.min_episode_len,
                quality         = args.d4rl_quality,
            )
        except Exception as e:
            print(f"D4RL load failed: {e}")
            print("Falling back to synthetic dataset.")

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

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None):
        best_val = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            self.scheduler.step()

            # pred first (dominant), then others
            key_order = [
                'loss', 'loss_pred',
                'loss_recon_s', 'loss_recon_a',
                'loss_kl', 'loss_eig',
                'loss_cst', 'loss_div', 'loss_ent', 'loss_decorr',
            ]
            log_str = f"Epoch {epoch:4d}"
            for k in key_order:
                if k in train_metrics:
                    log_str += f"  {k}: {train_metrics[k]:.4f}"

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

    def save_checkpoint(self, name: str):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg':             self.cfg,
            'args':            vars(self.args),
        }, self.save_dir / name)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt  = torch.load(path, map_location=device)
        model = KoopmanCVAE(ckpt['cfg'])
        model.load_state_dict(ckpt['model_state'])
        return model, ckpt['cfg']


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train KODAC: Koopman Diagonal Matrix Prior CVAE')

    # ── Environment ──────────────────────────────────────────────────────
    p.add_argument('--env',        type=str,   default='synthetic',
                   help='Environment key. Options: adroit_pen, adroit_hammer, '
                        'adroit_door, adroit_relocate, dm_walker, dm_cheetah, synthetic')
    p.add_argument('--action_dim', type=int,   default=None,
                   help='Override action dim (default: from ENV_CONFIGS)')
    p.add_argument('--state_dim',  type=int,   default=None,
                   help='Override state dim (default: from ENV_CONFIGS)')
    p.add_argument('--patch_size', type=int,   default=None,
                   help='Steps per dt patch (default: 5). dt_patch = patch_size * dt_control')
    p.add_argument('--dt_control', type=float, default=None,
                   help='Control timestep in seconds (default: 0.02)')

    # ── Dataset ──────────────────────────────────────────────────────────
    p.add_argument('--d4rl_quality',    type=str,  default='human',
                   choices=['expert', 'human', 'cloned'],
                   help='D4RL dataset quality tier (adroit envs only). default: human')
    p.add_argument('--seq_len',         type=int,  default=100,
                   help='Sequence length (timesteps) per sample. default: 100')
    p.add_argument('--stride',          type=int,  default=None,
                   help='Sliding window stride. default: seq_len // 2')
    p.add_argument('--min_episode_len', type=int,  default=30,
                   help='Minimum episode length to keep. default: 30')
    p.add_argument('--n_synthetic',     type=int,  default=1000,
                   help='Number of synthetic samples (env=synthetic only). default: 1000')

    # ── Architecture ─────────────────────────────────────────────────────
    p.add_argument('--mlp_hidden_dim', type=int,   default=None,
                   help='Hidden dim for StreamEncoder, StateDecoder, ActionDecoder MLPs. '
                        'default: 256')
    p.add_argument('--gru_hidden_dim', type=int,   default=None,
                   help='Hidden dim for SkillPosteriorGRU. default: 256')
    p.add_argument('--embed_dim',      type=int,   default=None,
                   help='Input projection dim for GRU. default: 128')
    p.add_argument('--koopman_dim',    type=int,   default=None,
                   help='m: eigenfunction pairs per stream. '
                        'State latent = 2m, action latent = 2m. default: 64')
    p.add_argument('--num_skills',     type=int,   default=None,
                   help='S: number of discrete skills. default: 8')
    p.add_argument('--lora_rank',      type=int,   default=None,
                   help='r: LoRA rank for beta_eff = diag + U@VT. default: 4')
    p.add_argument('--dropout',        type=float, default=None,
                   help='Dropout rate for GRU and StreamEncoder. default: 0.1')

    # ── Eigenvalue ───────────────────────────────────────────────────────
    p.add_argument('--mu_fixed',  type=float, default=None,
                   help='Fixed real part of eigenvalue mu_k (decay). '
                        'Must be in (-1, 0) for stability. default: -0.2')
    p.add_argument('--omega_max', type=float, default=None,
                   help='Max frequency for omega_k init grid. default: pi')

    # ── Loss weights ─────────────────────────────────────────────────────
    p.add_argument('--kl_prior',      type=str,   default=None,
                   choices=['koopman', 'standard'],
                   help='KL prior: koopman=CN(A*z_{t-1}, Sigma), standard=CN(0,Sigma). '
                        'default: koopman')
    p.add_argument('--alpha_pred',    type=float, default=None,
                   help='Prediction loss weight (dominant). default: 1.0')
    p.add_argument('--alpha_recon_s', type=float, default=None,
                   help='State reconstruction loss weight (D_s). default: 0.5')
    p.add_argument('--alpha_recon_a', type=float, default=None,
                   help='Action reconstruction loss weight (D_a). default: 0.5')
    p.add_argument('--beta_kl',       type=float, default=None,
                   help='KL divergence weight (state+action streams). default: 0.05')
    p.add_argument('--gamma_eig',     type=float, default=None,
                   help='Eigenvalue frequency repulsion weight. default: 0.05')
    p.add_argument('--delta_cst',     type=float, default=None,
                   help='Contrastive loss weight (GRU vs z_a summary). default: 0.1')
    p.add_argument('--delta_div',     type=float, default=None,
                   help='Skill mode diversity loss weight. default: 0.1')
    p.add_argument('--delta_ent',     type=float, default=None,
                   help='Posterior entropy regularization weight. default: 0.05')
    p.add_argument('--delta_decorr',  type=float, default=None,
                   help='Eigenfunction decorrelation weight. default: 0.05')
    p.add_argument('--pred_steps',    type=int,   default=None,
                   help='Multi-step prediction horizon H. default: 5')

    # ── Training ─────────────────────────────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=200,
                   help='Total training epochs. default: 200')
    p.add_argument('--batch_size',   type=int,   default=64,
                   help='Batch size. default: 64')
    p.add_argument('--lr',           type=float, default=3e-4,
                   help='AdamW learning rate. default: 3e-4')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW weight decay. default: 1e-4')
    p.add_argument('--eval_freq',    type=int,   default=10,
                   help='Validation every N epochs. default: 10')
    p.add_argument('--save_freq',    type=int,   default=50,
                   help='Checkpoint every N epochs. default: 50')
    p.add_argument('--save_dir',     type=str,   default='checkpoints',
                   help='Checkpoint directory. default: checkpoints')
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Device. default: cuda if available else cpu')
    p.add_argument('--num_workers',  type=int,   default=2,
                   help='DataLoader workers. default: 2')
    p.add_argument('--val_ratio',    type=float, default=0.1,
                   help='Fraction of data held out for validation. default: 0.1')

    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    cfg  = build_config(args)

    print("=" * 60)
    print("KODAC Configuration")
    print("=" * 60)
    print(f"  env            : {args.env}")
    print(f"  state_dim      : {cfg.state_dim}")
    print(f"  action_dim     : {cfg.action_dim}")
    print(f"  dt_patch       : {cfg.patch_size * cfg.dt_control * 1000:.1f} ms")
    print(f"  koopman_dim (m): {cfg.koopman_dim}  → state latent 2m={2*cfg.koopman_dim}, action latent 2m={2*cfg.koopman_dim}")
    print(f"  num_skills (S) : {cfg.num_skills}")
    print(f"  lora_rank (r)  : {cfg.lora_rank}")
    print(f"  kl_prior       : {cfg.kl_prior}")
    print(f"  pred_steps (H) : {cfg.pred_steps}")
    print()
    print("Loss weights:")
    print(f"  alpha_pred    = {cfg.alpha_pred}  (dominant)")
    print(f"  alpha_recon_s = {cfg.alpha_recon_s}")
    print(f"  alpha_recon_a = {cfg.alpha_recon_a}")
    print(f"  beta_kl       = {cfg.beta_kl}")
    print(f"  gamma_eig     = {cfg.gamma_eig}")
    print(f"  delta_cst     = {cfg.delta_cst}")
    print(f"  delta_div     = {cfg.delta_div}")
    print(f"  delta_ent     = {cfg.delta_ent}")
    print(f"  delta_decorr  = {cfg.delta_decorr}")
    print("=" * 60)

    model = KoopmanCVAE(cfg)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_total:,}")
    print(f"  phi_s  (state encoder)  : {sum(p.numel() for p in model.phi_s.parameters()):,}")
    print(f"  phi_a  (action encoder) : {sum(p.numel() for p in model.phi_a.parameters()):,}")
    print(f"  var_s  (state var head) : {sum(p.numel() for p in model.var_s.parameters()):,}")
    print(f"  var_a  (action var head): {sum(p.numel() for p in model.var_a.parameters()):,}")
    print(f"  dec_s  (state decoder)  : {sum(p.numel() for p in model.dec_s.parameters()):,}")
    print(f"  dec_a  (action decoder) : {sum(p.numel() for p in model.dec_a.parameters()):,}")
    print(f"  skill_gru               : {sum(p.numel() for p in model.skill_gru.parameters()):,}")
    print(f"  skill_params (V, beta)  : {sum(p.numel() for p in model.skill_params.parameters()):,}")
    print(f"  koopman (omega, sigma0) : {sum(p.numel() for p in model.koopman.parameters()):,}")

    # Load dataset
    dataset = load_dataset(args, cfg)
    print(f"\nDataset size: {len(dataset)} samples")

    n_val   = max(1, int(args.val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val} | Iter/epoch: {len(train_loader)}")
    print()

    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)