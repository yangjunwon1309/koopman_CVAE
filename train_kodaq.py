"""
train.py — KODAQ v5 Training Script
=====================================

v5 changes over v4:
  - After each optimizer step: model.soft_update_target_Q()  (EMA target Q)
  - LOG_KEYS extended: loss_reward, loss_q, loss_pi, rho, q_scale
  - rewards batch field now required for v5 heads
  - Policy prior training starts at phase 2 (controlled inside model)

Phase schedule (same as v4):
  Phase 1: L_rec + L_reward + L_Q  (world model + new heads, no π)
  Phase 2: + L_dyn + L_skill + L_π  (Koopman structure + policy)
  Phase 3: + L_reg

Note: L_reward and L_Q are active from phase 1 because the reward / Q heads
don't need the Koopman dynamics to be well-trained first — they just need
the posterior encoder to produce meaningful latents.  L_π starts at phase 2
because meaningful Q values are needed first.
"""
"""
train.py — KODAQ v5 Training Script
=====================================

v5 changes over v4:
  - After each optimizer step: model.soft_update_target_Q()  (EMA target Q)
  - LOG_KEYS extended: loss_reward, loss_q, loss_pi, rho, q_scale
  - rewards = step reward {0,1} (not accumulated)
  - Policy prior training starts at phase 2 (controlled inside model)
  - wandb logging (--wandb_project / --wandb_entity / --disable_wandb)

Phase schedule:
  Phase 1: WM rec + L_R + L_Q   (world model + Q/reward heads, no pi)
  Phase 2: + L_dyn + L_skill + L_pi
  Phase 3: + L_reg

nohup example:
  mkdir -p logs checkpoints/kodaq_v5
  nohup python train.py \\
      --env kitchen_mixed \\
      --wandb_project kodaq \\
      --wandb_entity YOUR_ENTITY \\
      --wandb_run_name v5_kitchen_run1 \\
      --wandb_tags v5,kitchen_mixed \\
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
        return load_kodaq_dataset(
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
        'loss_reward', 'loss_q', 'loss_pi',
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

        self.phase2_epoch = args.phase2_epoch
        self.phase3_epoch = args.phase3_epoch

        self.use_wandb = (
            _WANDB_AVAILABLE
            and not args.disable_wandb
            and args.wandb_project is not None
        )

    # ── wandb helpers ────────────────────────────────────────────────────────

    def _wandb_init(self):
        if not self.use_wandb:
            return
        args = self.args
        cfg  = self.cfg
        wandb.init(
            project  = args.wandb_project,
            entity   = args.wandb_entity or None,
            name     = args.wandb_run_name or None,
            group    = args.wandb_group or None,
            tags     = [t.strip() for t in args.wandb_tags.split(',') if t.strip()],
            dir      = str(self.save_dir),
            resume   = 'allow',
            config   = {
                # arch
                'koopman_dim':   cfg.koopman_dim,
                'gru_hidden':    cfg.gru_hidden,
                'action_latent': cfg.action_latent,
                'num_skills':    cfg.num_skills,
                'mlp_hidden':    cfg.mlp_hidden,
                # v5
                'num_bins':      cfg.num_bins,
                'v_min':         cfg.v_min,
                'v_max':         cfg.v_max,
                'num_q':         cfg.num_q,
                'tau':           cfg.tau,
                'gamma':         cfg.gamma,
                'entropy_coef':  cfg.entropy_coef,
                'log_std_min':   cfg.log_std_min,
                'log_std_max':   cfg.log_std_max,
                # loss weights
                'lambda1':       cfg.lambda1,
                'lambda2':       cfg.lambda2,
                'lambda3':       cfg.lambda3,
                'lambda4':       cfg.lambda4,
                'lambda_reward': cfg.lambda_reward,
                'lambda_q':      cfg.lambda_q,
                'lambda_pi':     cfg.lambda_pi,
                # train
                'lr':            args.lr,
                'weight_decay':  args.weight_decay,
                'batch_size':    args.batch_size,
                'epochs':        args.epochs,
                'seq_len':       args.seq_len,
                'env':           args.env,
                'phase2_epoch':  args.phase2_epoch,
                'phase3_epoch':  args.phase3_epoch,
            },
        )
        print(f"[wandb] project={args.wandb_project}  run={wandb.run.name}",
              flush=True)
        print(f"[wandb] url: {wandb.run.url}", flush=True)

    def _wandb_log(self, metrics: Dict, step: int, prefix: str):
        if not self.use_wandb:
            return
        wandb.log(
            {f"{prefix}/{k}": v for k, v in metrics.items()},
            step=step,
        )

    def _wandb_finish(self):
        if self.use_wandb:
            wandb.finish()

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
            rewards      = batch.get('rewards', None)   # step reward {0,1}

        x_seq        = x_seq.to(self.device)
        actions      = actions.to(self.device)
        skill_labels = skill_labels.to(self.device)
        if mask    is not None: mask    = mask.to(self.device)
        if rewards is not None: rewards = rewards.to(self.device)

        return self.model(x_seq, actions, skill_labels, mask, rewards)

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _accumulate(self, totals: Dict, out: Dict):
        for k in self.LOG_KEYS:
            if k in out:
                v = out[k]
                totals[k] = totals.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )

    def train_epoch(self, loader) -> Dict:
        self.model.train()
        self.model.q_head_target.eval()   # target Q stays in eval always

        totals = {}
        for batch in loader:
            out  = self._forward_batch(batch)
            loss = out['loss']

            self.optimizer.zero_grad()
            loss.backward()
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

    def train(self, train_loader, val_loader=None):
        self._wandb_init()
        best_val = float('inf')
        t0       = time.time()

        for epoch in range(1, self.args.epochs + 1):
            self._maybe_update_phase(epoch)
            t_ep = time.time()

            # train
            metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            self._wandb_log(metrics, step=epoch, prefix='train')

            # eval
            val_metrics = {}
            if val_loader and epoch % self.args.eval_freq == 0:
                val_metrics = self.eval_epoch(val_loader)
                self._wandb_log(val_metrics, step=epoch, prefix='val')
                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint('best.pt')

            # console log
            phase   = self.model.cfg.phase
            ep_sec  = time.time() - t_ep
            tot_min = (time.time() - t0) / 60.0

            line = (f"[Ph{phase}] Ep {epoch:4d}/{self.args.epochs}"
                    f"  {ep_sec:.1f}s  ({tot_min:.0f}m)")

            for k in ['loss', 'loss_wm', 'loss_rec', 'loss_dyn',
                      'loss_skill', 'loss_reg']:
                if metrics.get(k, 0.0) != 0.0:
                    line += f"  {k.replace('loss_','')[:4]}={metrics[k]:.4f}"

            for k in ['loss_reward', 'loss_q', 'loss_pi']:
                if k in metrics:
                    line += f"  {k.replace('loss_','')[:3]}={metrics[k]:.4f}"

            if 'rho' in metrics:
                line += f"  rho={metrics['rho']:.3f}"
            if 'q_scale' in metrics:
                line += f"  Qs={metrics['q_scale']:.2f}"
            if val_metrics:
                line += f"  | val={val_metrics.get('loss', 0):.4f}"

            print(line, flush=True)

            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"\nDone. best_val={best_val:.4f}  "
              f"time={(time.time()-t0)/60:.1f}m", flush=True)
        self._wandb_finish()

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

    # wandb
    p.add_argument('--wandb_project',  type=str, default=None,
                   help='wandb project. If None, wandb is off.')
    p.add_argument('--wandb_entity',   type=str, default=None,
                   help='wandb entity (username or team)')
    p.add_argument('--wandb_run_name', type=str, default=None,
                   help='Run name shown in wandb UI')
    p.add_argument('--wandb_group',    type=str, default=None,
                   help='Group name for comparing multiple runs')
    p.add_argument('--wandb_tags',     type=str, default='',
                   help='Comma-separated tags e.g. "v5,kitchen,debug"')
    p.add_argument('--disable_wandb',  action='store_true')

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
    print(f"  wandb: {'off' if args.disable_wandb or not args.wandb_project else args.wandb_project}")
    print("=" * 65, flush=True)

    model   = KoopmanCVAE(cfg)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {n_total:,}")
    for nm, mod in [
        ('posterior',    model.posterior),
        ('recurrent',    model.recurrent),
        ('skill_prior',  model.skill_prior),
        ('koopman',      model.koopman),
        ('decoder',      model.decoder),
        ('action_enc',   model.action_encoder),
        ('reward_head',  model.reward_head),
        ('q_head',       model.q_head),
        ('policy_prior', model.policy_prior),
    ]:
        print(f"  {nm:<14}: {sum(p.numel() for p in mod.parameters()):,}")
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