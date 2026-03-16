"""
Training script for Diagonal Koopman Prior CVAE
Supports: DMControl, D4RL (Adroit), HumanoidBench, Isaac Gym (offline data)
"""
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from pathlib import Path

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.env_configs import ENV_CONFIGS, build_config
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset


# ─────────────────────────────────────────────
#  Environment Configs
# ─────────────────────────────────────────────

# ENV_CONFIGS = {
#     # DMControl
#     'dm_reacher':    dict(action_dim=2,  state_dim=11, dt_control=0.02,  patch_size=5),
#     'dm_walker':     dict(action_dim=6,  state_dim=24, dt_control=0.02,  patch_size=5),
#     'dm_cheetah':    dict(action_dim=6,  state_dim=17, dt_control=0.02,  patch_size=5),
#     'dm_cartpole':   dict(action_dim=1,  state_dim=5,  dt_control=0.02,  patch_size=5),
#     'dm_humanoid':   dict(action_dim=21, state_dim=67, dt_control=0.02,  patch_size=5),

#     # D4RL Adroit (offline)
#     'adroit_pen':      dict(action_dim=24, state_dim=45, dt_control=0.04, patch_size=3),
#     'adroit_hammer':   dict(action_dim=26, state_dim=46, dt_control=0.04, patch_size=3),
#     'adroit_door':     dict(action_dim=28, state_dim=39, dt_control=0.04, patch_size=3),
#     'adroit_relocate': dict(action_dim=30, state_dim=39, dt_control=0.04, patch_size=3),

#     # HumanoidBench
#     'humanoid_stand':  dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
#     'humanoid_walk':   dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
#     'humanoid_reach':  dict(action_dim=19, state_dim=132, dt_control=0.01, patch_size=10),

#     # Isaac Gym
#     'isaac_franka':    dict(action_dim=7,  state_dim=23,  dt_control=0.0167, patch_size=6),
#     'isaac_allegro':   dict(action_dim=16, state_dim=92,  dt_control=0.0167, patch_size=6),
#     'isaac_humanoid':  dict(action_dim=21, state_dim=108, dt_control=0.0167, patch_size=6),
# }


# def build_config(args) -> KoopmanCVAEConfig:
#     env_cfg = ENV_CONFIGS.get(args.env, {})
#     return KoopmanCVAEConfig(
#         action_dim      = env_cfg.get('action_dim', args.action_dim),
#         state_dim       = env_cfg.get('state_dim', args.state_dim),
#         patch_size      = env_cfg.get('patch_size', args.patch_size),
#         dt_control      = env_cfg.get('dt_control', args.dt_control),
#         embed_dim       = args.embed_dim,
#         state_embed_dim = args.state_embed_dim,
#         gru_hidden_dim  = args.gru_hidden_dim,
#         mlp_hidden_dim  = args.mlp_hidden_dim,
#         koopman_dim     = args.koopman_dim,
#         beta_kl         = args.beta_kl,
#         alpha_pred      = args.alpha_pred,
#         gamma_eig       = args.gamma_eig,
#         dropout         = args.dropout,
#     )


# ─────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────

def load_d4rl_dataset(env_name: str, seq_len: int = 100):
    """Load D4RL offline dataset (requires d4rl installed)"""
    try:
        import d4rl, gym
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        obs = torch.tensor(dataset['observations'], dtype=torch.float32)
        acts = torch.tensor(dataset['actions'], dtype=torch.float32)

        N = len(obs) - seq_len
        states_list, actions_list = [], []
        for i in range(0, N, seq_len // 2):
            states_list.append(obs[i:i+seq_len])
            actions_list.append(acts[i:i+seq_len])

        states = torch.stack(states_list)    # (B, T, ds)
        actions = torch.stack(actions_list)  # (B, T, da)
        return TensorDataset(actions, states)
    except ImportError:
        print("d4rl not installed. Using synthetic data.")
        return None


def make_synthetic_dataset(cfg: KoopmanCVAEConfig, n_samples=1000, seq_len=100):
    """Generate synthetic sinusoidal data for quick testing"""
    T = seq_len
    t = torch.linspace(0, 2 * np.pi, T)

    actions = torch.zeros(n_samples, T, cfg.action_dim)
    states  = torch.zeros(n_samples, T, cfg.state_dim)

    for i in range(n_samples):
        freqs = torch.rand(cfg.action_dim) * 3 + 0.5
        phases = torch.rand(cfg.action_dim) * 2 * np.pi
        amps = torch.rand(cfg.action_dim) * 0.5 + 0.5
        for d in range(cfg.action_dim):
            actions[i, :, d] = amps[d] * torch.sin(freqs[d] * t + phases[d])

        freqs_s = torch.rand(cfg.state_dim) * 2 + 0.3
        phases_s = torch.rand(cfg.state_dim) * 2 * np.pi
        for d in range(cfg.state_dim):
            states[i, :, d] = torch.sin(freqs_s[d] * t + phases_s[d])

    return TensorDataset(actions, states)


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model: KoopmanCVAE, cfg: KoopmanCVAEConfig, args):
        self.model = model
        self.cfg = cfg
        self.args = args
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        totals = {}
        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)

            out = self.model(actions, states)
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
        totals = {}
        for actions, states in loader:
            actions = actions.to(self.device)
            states  = states.to(self.device)
            out = self.model(actions, states)
            for k, v in out.items():
                if k.startswith('loss'):
                    totals[k] = totals.get(k, 0.0) + v.item()
        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def train(self, train_loader, val_loader=None):
        best_val = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            log_str = f"Epoch {epoch:4d}"
            for k, v in train_metrics.items():
                log_str += f"  {k}: {v:.4f}"

            if val_loader is not None and epoch % self.args.eval_freq == 0:
                val_metrics = self.eval_epoch(val_loader)
                val_loss = val_metrics['loss']
                log_str += f"  | val_loss: {val_loss:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint('best.pt')

            print(log_str)

            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"Training complete. Best val loss: {best_val:.4f}")

    def save_checkpoint(self, name: str):
        path = self.save_dir / name
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg': self.cfg,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt['cfg']
        model = KoopmanCVAE(cfg)
        model.load_state_dict(ckpt['model_state'])
        return model, cfg


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument('--env',            type=str,   default='synthetic')
    p.add_argument('--action_dim',     type=int,   default=6)
    p.add_argument('--state_dim',      type=int,   default=24)
    p.add_argument('--patch_size',     type=int,   default=5)
    p.add_argument('--dt_control',     type=float, default=0.02)
    p.add_argument('--seq_len',        type=int,   default=100)
    # Architecture
    p.add_argument('--embed_dim',      type=int,   default=128)
    p.add_argument('--state_embed_dim',type=int,   default=64)
    p.add_argument('--gru_hidden_dim', type=int,   default=256)
    p.add_argument('--mlp_hidden_dim', type=int,   default=256)
    p.add_argument('--koopman_dim',    type=int,   default=64)
    # Loss
    p.add_argument('--beta_kl',        type=float, default=1.0)
    p.add_argument('--alpha_pred',     type=float, default=1.0)
    p.add_argument('--gamma_eig',      type=float, default=0.1)
    p.add_argument('--delta_cst',      type=float, default=1.0)
    p.add_argument('--dropout',        type=float, default=0.1)
    # Training
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=64)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--eval_freq',      type=int,   default=10)
    p.add_argument('--save_freq',      type=int,   default=50)
    p.add_argument('--save_dir',       type=str,   default='checkpoints')
    p.add_argument('--device',         type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--d4rl_quality', type=str, default='expert',
               choices=['expert', 'medium', 'random', 'human'])
    return p.parse_args()


if __name__ == '__main__':
    from typing import Dict
    args = parse_args()
    cfg = build_config(args)

    print(f"Config: {cfg}")
    print(f"Delta_t patch: {cfg.patch_size * cfg.dt_control:.4f}s")

    model = KoopmanCVAE(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Load data
    if args.env != 'synthetic' and args.env in ['adroit_pen', 'adroit_hammer',
                                                  'adroit_door', 'adroit_relocate']:
        env_map = {
            'adroit_pen': 'pen-expert-v1',
            'adroit_hammer': 'hammer-expert-v1',
            'adroit_door': 'door-expert-v1',
            'adroit_relocate': 'relocate-expert-v1',
        }
        dataset = load_d4rl_dataset(env_map[args.env], args.seq_len)
        if dataset is None:
            dataset = make_synthetic_dataset(cfg, seq_len=args.seq_len)
    else:
        dataset = make_synthetic_dataset(cfg, seq_len=args.seq_len)

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)
