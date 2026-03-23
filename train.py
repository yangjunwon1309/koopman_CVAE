"""
train.py — KODAC-S Training Script
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.env_configs import ENV_CONFIGS, build_config

ADROIT_ENVS  = ['adroit_pen', 'adroit_hammer', 'adroit_door', 'adroit_relocate']
KITCHEN_ENVS = ['kitchen_complete', 'kitchen_partial', 'kitchen_mixed']

def load_dataset(args, cfg):
    from data.dataset_utils import (
        load_d4rl_trajectories,
        load_kitchen_all_qualities,
        make_synthetic_dataset,
    )

    if args.env in KITCHEN_ENVS:
        print(f"Loading Kitchen: env={args.env}, seq_len={args.seq_len}")
        try:
            if getattr(args, 'kitchen_all_qualities', False):
                # Load and merge complete + partial + mixed
                return load_kitchen_all_qualities(
                    seq_len=args.seq_len,
                    stride=args.stride,
                    min_episode_len=args.min_episode_len,
                )
            else:
                return load_d4rl_trajectories(
                    env_name=args.env,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    min_episode_len=args.min_episode_len,
                )
        except Exception as e:
            print(f"Kitchen load failed ({e}), falling back to synthetic")

    elif args.env in ADROIT_ENVS:
        print(f"Loading Adroit: env={args.env}, quality={args.d4rl_quality}")
        try:
            return load_d4rl_trajectories(
                env_name=args.env, seq_len=args.seq_len,
                stride=args.stride, min_episode_len=args.min_episode_len,
                quality=args.d4rl_quality,
            )
        except Exception as e:
            print(f"Adroit load failed ({e}), falling back to synthetic")

    print(f"Using synthetic: action_dim={cfg.action_dim}, state_dim={cfg.state_dim}")
    return make_synthetic_dataset(cfg.action_dim, cfg.state_dim,
                                  args.n_synthetic, args.seq_len)


class Trainer:
    def __init__(self, model, cfg, args):
        self.model  = model
        self.cfg    = cfg
        self.args   = args
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs)

        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader) -> Dict:
        self.model.train()
        totals = {}
        for actions, states in loader:
            actions, states = actions.to(self.device), states.to(self.device)
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
    def eval_epoch(self, loader) -> Dict:
        self.model.eval()
        totals = {}
        for actions, states in loader:
            actions, states = actions.to(self.device), states.to(self.device)
            out = self.model(actions, states)
            for k, v in out.items():
                if k.startswith('loss'):
                    totals[k] = totals.get(k, 0.0) + v.item()
        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def train(self, train_loader, val_loader=None):
        best_val = float('inf')
        key_order = ['loss', 'loss_pred', 'loss_recon',
                     'loss_eig_stab', 'loss_eig_div', 'loss_decorr']

        for epoch in range(1, self.args.epochs + 1):
            metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            log = f"Epoch {epoch:4d}"
            for k in key_order:
                if k in metrics:
                    log += f"  {k}: {metrics[k]:.4f}"

            if val_loader and epoch % self.args.eval_freq == 0:
                val = self.eval_epoch(val_loader)
                val_loss = val['loss']
                log += f"  | val: {val_loss:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint('best.pt')

            print(log, flush=True)
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"Done. Best val: {best_val:.4f}")

    def save_checkpoint(self, name):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg': self.cfg, 'args': vars(self.args),
        }, self.save_dir / name)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        ckpt  = torch.load(path, map_location=device)
        model = KoopmanCVAE(ckpt['cfg'])
        model.load_state_dict(ckpt['model_state'])
        return model, ckpt['cfg']


def parse_args():
    p = argparse.ArgumentParser(description='Train KODAC-S')

    # Environment
    p.add_argument('--env',        type=str,   default='kitchen_partial',
                   help='Env key: kitchen_complete | kitchen_partial | kitchen_mixed '
                        '| adroit_pen | adroit_hammer | adroit_door | adroit_relocate '
                        '| synthetic')
    p.add_argument('--action_dim', type=int,   default=None)
    p.add_argument('--state_dim',  type=int,   default=None)
    p.add_argument('--patch_size', type=int,   default=None)
    p.add_argument('--dt_control', type=float, default=None)

    # Dataset
    p.add_argument('--d4rl_quality',    type=str,  default='human',
                   choices=['expert', 'human', 'cloned'],
                   help='Quality tier (Adroit only). default: human')
    p.add_argument('--kitchen_all_qualities', action='store_true',
                   help='Kitchen: merge complete + partial + mixed datasets')
    p.add_argument('--seq_len',         type=int,  default=200,
                   help='Sequence length per sample. '
                        'Kitchen: 200 (~16s at 12.5Hz). Adroit: 100. default: 200')
    p.add_argument('--stride',          type=int,  default=None,
                   help='Sliding window stride. default: seq_len // 2')
    p.add_argument('--min_episode_len', type=int,  default=100,
                   help='Minimum episode length to keep. '
                        'Kitchen episodes are ~280 steps. default: 100')
    p.add_argument('--n_synthetic',     type=int,  default=1000)

    # Architecture
    p.add_argument('--mlp_hidden_dim',   type=int,   default=None,
                   help='MLP hidden dim. default: 256')
    p.add_argument('--tcn_hidden_dim',   type=int,   default=None,
                   help='TCN channel width. default: 256')
    p.add_argument('--tcn_n_layers',     type=int,   default=None,
                   help='TCN depth. default: 4')
    p.add_argument('--tcn_kernel_size',  type=int,   default=None,
                   help='TCN causal kernel size. default: 3')
    p.add_argument('--koopman_dim',      type=int,   default=None,
                   help='m: z dimension. default: 64')
    p.add_argument('--num_heads',        type=int,   default=None,
                   help='Nh: number of observable heads. default: 8')
    p.add_argument('--lora_rank',        type=int,   default=None,
                   help='r: rank of B^(l). default: 8')
    p.add_argument('--b_max',            type=float, default=None,
                   help='B magnitude bound (tanh scale). default: 0.3')
    p.add_argument('--dropout',          type=float, default=None)

    # Eigenvalue
    p.add_argument('--eig_target_radius', type=float, default=None,
                   help='Target spectral radius of A. default: 0.99')
    p.add_argument('--eig_margin',        type=float, default=None,
                   help='Soft margin for stability loss. default: 0.01')
    p.add_argument('--eig_div_sigma',     type=float, default=None,
                   help='Eigenvalue diversity bandwidth. default: 0.1')

    # Loss weights
    p.add_argument('--alpha_pred',   type=float, default=None,
                   help='Prediction loss weight. default: 1.0')
    p.add_argument('--alpha_recon',  type=float, default=None,
                   help='Reconstruction loss weight. default: 0.5')
    p.add_argument('--gamma_eig',    type=float, default=None,
                   help='Eigenvalue stability loss weight. default: 0.1')
    p.add_argument('--gamma_div',    type=float, default=None,
                   help='Eigenvalue diversity loss weight. default: 0.05')
    p.add_argument('--delta_decorr', type=float, default=None,
                   help='Decorrelation loss weight. default: 0.1')
    p.add_argument('--pred_steps',   type=int,   default=None,
                   help='Prediction horizon H. default: 5')

    # Training
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


if __name__ == '__main__':
    args = parse_args()
    cfg  = build_config(args)

    print("=" * 60)
    print("KODAC-S Configuration")
    print("=" * 60)
    print(f"  env          : {args.env}")
    print(f"  state_dim    : {cfg.state_dim}  action_dim: {cfg.action_dim}")
    print(f"  koopman_dim  : {cfg.koopman_dim}  (z in R^m)")
    print(f"  A            : {cfg.koopman_dim}x{cfg.koopman_dim} full matrix")
    print(f"  B            : {cfg.action_dim} x ({cfg.koopman_dim}x{cfg.lora_rank}) low-rank")
    print(f"  num_heads    : {cfg.num_heads}  pred_steps: {cfg.pred_steps}")
    print(f"  TCN          : {cfg.tcn_n_layers} layers, d={cfg.tcn_hidden_dim}")
    print(f"  dt_patch     : {cfg.patch_size * cfg.dt_control * 1000:.0f}ms")
    print()
    print("Loss weights:")
    print(f"  alpha_pred={cfg.alpha_pred}  alpha_recon={cfg.alpha_recon}")
    print(f"  gamma_eig={cfg.gamma_eig}   gamma_div={cfg.gamma_div}")
    print(f"  delta_decorr={cfg.delta_decorr}")
    print("=" * 60)

    model = KoopmanCVAE(cfg)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {n_total:,}")
    print(f"  encoder : {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  decoder : {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"  koopman : {sum(p.numel() for p in model.koopman.parameters()):,}")
    print(f"    A     : {cfg.koopman_dim**2:,}")
    print(f"    B     : {cfg.action_dim * cfg.koopman_dim * cfg.lora_rank * 2:,}")
    print(f"  tcn     : {sum(p.numel() for p in model.tcn.parameters()):,}")

    dataset = load_dataset(args, cfg)
    print(f"\nDataset: {len(dataset)} samples")

    n_val   = max(1, int(args.val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val} | Iter/epoch: {len(train_loader)}\n")

    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)