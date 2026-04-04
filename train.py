"""
train.py — KODAQ Full RSSM-Koopman Training Script
====================================================

Training protocol (§6):
  Phase 1 (warm-up)  : L_rec only         → encoder/decoder convergence
  Phase 2 (Koopman)  : + L_dyn + L_skill  → linear dynamics structure
  Phase 3 (full)     : + L_reg            → posterior-prior alignment

A_k, B_k initialized near identity in SkillKoopmanOperator.__init__.
μ_k optionally warm-started from EXTRACT centroids via model.init_skill_centroids().

Data:
  KODAQWindowDataset: sliding-window over x_t = [Δe_t, Δp_t, q_t, q̇_t]
  Each batch: (x_seq, actions, skill_labels)
  Skill labels: EXTRACT cluster assignments ĉ_t (int64)
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from envs.env_configs import ENV_CONFIGS, KITCHEN_ENVS, ADROIT_ENVS, build_config
from data.dataset_utils import (
    load_kodaq_dataset,
    make_synthetic_dataset,
    collate_fn_pad,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

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
        )
        return ds
    except Exception as e:
        print(f"Dataset load failed ({e}). Falling back to synthetic.")
        return make_synthetic_dataset(K=cfg.num_skills)


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


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    LOG_KEYS = [
        'loss', 'loss_rec', 'loss_dyn', 'loss_skill', 'loss_reg', 'loss_stab',
        'loss_rec_delta_e', 'loss_rec_delta_p', 'loss_rec_q', 'loss_rec_qdot',
    ]

    def __init__(self, model: KoopmanCVAE, cfg: KoopmanCVAEConfig, args):
        self.model   = model
        self.cfg     = cfg
        self.args    = args
        self.device  = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Phase boundaries (epoch numbers, 1-indexed)
        self.phase2_epoch = args.phase2_epoch   # switch to phase 2
        self.phase3_epoch = args.phase3_epoch   # switch to phase 3

    def _maybe_update_phase(self, epoch: int):
        """Phase scheduling per §6."""
        if epoch == self.phase2_epoch:
            self.model.set_phase(2)
        elif epoch == self.phase3_epoch:
            self.model.set_phase(3)

    def _forward_batch(self, batch) -> Dict:
        if isinstance(batch, (list, tuple)):
            x_seq, actions, skill_labels = batch
            mask = None
        else:
            x_seq        = batch['x_seq']
            actions      = batch['actions']
            skill_labels = batch['skill_labels']
            mask         = batch.get('mask', None)

        x_seq        = x_seq.to(self.device)
        actions      = actions.to(self.device)
        skill_labels = skill_labels.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        return self.model(x_seq, actions, skill_labels, mask)

    def train_epoch(self, loader) -> Dict:
        self.model.train()
        totals = {}
        for batch in loader:
            out  = self._forward_batch(batch)
            loss = out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in self.LOG_KEYS:
                if k in out:
                    totals[k] = totals.get(k, 0.0) + out[k].item()

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def eval_epoch(self, loader) -> Dict:
        self.model.eval()
        totals = {}
        for batch in loader:
            out = self._forward_batch(batch)
            for k in self.LOG_KEYS:
                if k in out:
                    totals[k] = totals.get(k, 0.0) + out[k].item()
        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def train(self, train_loader, val_loader=None):
        best_val = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            self._maybe_update_phase(epoch)
            metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            phase = self.model.cfg.phase
            log   = f"[Ph{phase}] Ep {epoch:4d}"
            for k in ['loss', 'loss_rec', 'loss_dyn', 'loss_skill', 'loss_reg']:
                if k in metrics:
                    log += f"  {k.replace('loss_','')}={metrics[k]:.4f}"

            if val_loader and epoch % self.args.eval_freq == 0:
                val     = self.eval_epoch(val_loader)
                val_lss = val['loss']
                log    += f"  | val={val_lss:.4f}"
                if val_lss < best_val:
                    best_val = val_lss
                    self.save_checkpoint('best.pt')

            print(log, flush=True)

            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"\nDone. Best val loss: {best_val:.4f}")

    def save_checkpoint(self, name: str):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg':             self.cfg,
            'args':            vars(self.args),
            'phase':           self.model.cfg.phase,
        }, self.save_dir / name)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt  = torch.load(path, map_location=device)
        model = KoopmanCVAE(ckpt['cfg'])
        model.load_state_dict(ckpt['model_state'])
        model.cfg.phase = ckpt.get('phase', 1)
        return model, ckpt['cfg']


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train KODAQ Full RSSM-Koopman')

    # Environment
    p.add_argument('--env', type=str, default='kitchen_mixed',
                   help='Env: kitchen_complete|kitchen_partial|kitchen_mixed'
                        '|adroit_pen|...|synthetic')

    # Data
    p.add_argument('--seq_len',    type=int,   default=64,
                   help='Sliding window length. Kitchen: 64 steps = ~5s')
    p.add_argument('--stride',     type=int,   default=None,
                   help='Sliding window stride (default: seq_len//2)')
    p.add_argument('--no_r3m',     action='store_true',
                   help='Skip R3M rendering (state-only, Δe_t=0)')
    p.add_argument('--pca_dim',    type=int,   default=64,
                   help='PCA dim before K-means (0=off)')
    p.add_argument('--skill_dir',  type=str,   default='checkpoints/skill_pretrain',
                   help='Directory for cluster_data.h5 and x_sequences.npz')
    p.add_argument('--n_synthetic', type=int,  default=2000)

    # Architecture overrides (None → use env_configs defaults)
    p.add_argument('--koopman_dim',   type=int,   default=None)
    p.add_argument('--gru_hidden',    type=int,   default=None)
    p.add_argument('--action_latent', type=int,   default=None)
    p.add_argument('--num_skills',    type=int,   default=None)
    p.add_argument('--mlp_hidden',    type=int,   default=None)
    p.add_argument('--enc_layers',    type=int,   default=None)
    p.add_argument('--dec_layers',    type=int,   default=None)
    p.add_argument('--dropout',       type=float, default=None)

    # Loss weights
    p.add_argument('--lambda1', type=float, default=None, help='L_dyn weight')
    p.add_argument('--lambda2', type=float, default=None, help='L_skill weight')
    p.add_argument('--lambda3', type=float, default=None, help='L_reg weight')
    p.add_argument('--lambda4', type=float, default=None, help='L_stab weight')

    # Phase scheduling
    p.add_argument('--phase2_epoch', type=int, default=30,
                   help='Epoch to switch to Phase 2 (+L_dyn, +L_skill)')
    p.add_argument('--phase3_epoch', type=int, default=80,
                   help='Epoch to switch to Phase 3 (+L_reg)')

    # Training
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--eval_freq',    type=int,   default=10)
    p.add_argument('--save_freq',    type=int,   default=50)
    p.add_argument('--save_dir',     type=str,   default='checkpoints/kodaq')
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers',  type=int,   default=2)
    p.add_argument('--val_ratio',    type=float, default=0.1)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    cfg  = build_config(args)

    print("=" * 65)
    print("KODAQ — Full RSSM-Koopman")
    print("=" * 65)
    print(f"  env          : {args.env}")
    print(f"  x_dim        : {cfg.x_dim}  (Δe={cfg.dim_delta_e}, Δp={cfg.dim_delta_p},"
          f" q={cfg.dim_q}, q̇={cfg.dim_qdot})")
    print(f"  action_dim   : {cfg.action_dim}  →  action_latent: {cfg.action_latent}")
    print(f"  koopman_dim  : {cfg.koopman_dim}  (d_o)")
    print(f"  gru_hidden   : {cfg.gru_hidden}   (d_h)")
    print(f"  num_skills   : {cfg.num_skills}   (K)")
    print()
    print("Loss weights:")
    print(f"  λ1={cfg.lambda1} (L_dyn)  λ2={cfg.lambda2} (L_skill)"
          f"  λ3={cfg.lambda3} (L_reg)  λ4={cfg.lambda4} (L_stab)")
    print(f"  α_Δe={cfg.alpha_delta_e}  α_Δp={cfg.alpha_delta_p}"
          f"  α_q={cfg.alpha_q}  α_q̇={cfg.alpha_qdot}")
    print()
    print("Phase schedule:")
    print(f"  Phase 1: ep 1–{args.phase2_epoch-1}   (L_rec only)")
    print(f"  Phase 2: ep {args.phase2_epoch}–{args.phase3_epoch-1}  (+L_dyn, +L_skill)")
    print(f"  Phase 3: ep {args.phase3_epoch}–{args.epochs}  (+L_reg)")
    print("=" * 65)

    model   = KoopmanCVAE(cfg)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {n_total:,}")
    print(f"  posterior    : {sum(p.numel() for p in model.posterior.parameters()):,}")
    print(f"  recurrent    : {sum(p.numel() for p in model.recurrent.parameters()):,}")
    print(f"  skill_prior  : {sum(p.numel() for p in model.skill_prior.parameters()):,}")
    print(f"  koopman      : {sum(p.numel() for p in model.koopman.parameters()):,}")
    print(f"    U          : {cfg.koopman_dim**2:,}")
    print(f"    r_k,θ_k    : {2 * cfg.num_skills * cfg.koopman_dim:,}")
    print(f"    G_k        : {cfg.num_skills * cfg.koopman_dim * cfg.action_latent:,}")
    print(f"  decoder      : {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"  action_enc   : {sum(p.numel() for p in model.action_encoder.parameters()):,}")

    # Dataset
    dataset = load_dataset(args, cfg)
    print(f"\nDataset: {len(dataset)} samples  seq_len={args.seq_len}")

    n_val   = max(1, int(args.val_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"Train: {n_train}  Val: {n_val}  Iter/epoch: {len(train_loader)}\n")

    trainer = Trainer(model, cfg, args)
    trainer.train(train_loader, val_loader)