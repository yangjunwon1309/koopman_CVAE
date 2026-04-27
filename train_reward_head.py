"""
train_reward_head.py
====================
KoopmanCVAE v4의 파라미터를 그대로 유지하면서
reward_head만 Categorical distribution으로 교체 후 fine-tune.

설계:
  기존 BCE head:  sigmoid(MLP(z_t)) → P(r_t > 0 | z_t) ∈ (0,1)
  신규 Cat head:  softmax(MLP(z_t)) → P(R_total = k | z_t), k ∈ {0,1,2,3,4}

학습:
  - target: window 내 cumulative reward의 bin label (0~4 중 하나)
  - loss:   cross entropy CE(logits, bin_label)
  - freeze: koopman, decoder(reconstruction heads), posterior, recurrent, skill_prior
  - active: CategoricalRewardHead only

Inference (kodaq_online.py):
  r̂ = Σ_k k * P(k | z_t)  (expected value over bins)
  → 0~4 사이 값, Q signal이 풍부해짐

Usage:
    python train_reward_head.py \\
        --ckpt   checkpoints/kodaq_v4/final.pt \\
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \\
        --out    checkpoints/kodaq_v4/cat_reward/final.pt \\
        --device cuda:1
"""

import os, sys, math
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig
from data.extract_skill_label import load_x_sequences
from lqr_koopman import load_kitchen_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Categorical Reward Head
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalRewardHead(nn.Module):
    """
    P(R_total = k | z_t)  k ∈ {0, 1, ..., n_bins-1}

    Kitchen 최대 subtask = 4 → n_bins = 5 (0,1,2,3,4)

    학습: CE loss with bin label
    추론: E[R|z_t] = Σ_k k * P(k|z_t)  ← linear expectation
    """
    def __init__(self, z_dim: int, hidden: int, n_bins: int = 5,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        # bin values: 0, 1, 2, 3, 4
        self.register_buffer('bin_values',
                             torch.arange(n_bins, dtype=torch.float32))

        layers = []
        d = z_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                       nn.ELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., z_dim) → logits (..., n_bins)"""
        return self.net(z)

    def probs(self, z: torch.Tensor) -> torch.Tensor:
        """→ (..., n_bins) softmax probabilities"""
        return torch.softmax(self.forward(z), dim=-1)

    def expected_reward(self, z: torch.Tensor) -> torch.Tensor:
        """
        E[R|z_t] = Σ_k k * P(k|z_t)
        → (...,) scalar expected cumulative reward
        """
        p = self.probs(z)   # (..., n_bins)
        return (p * self.bin_values).sum(dim=-1)

    def loss(self, z: torch.Tensor, bin_labels: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy loss.
        z:          (B, z_dim) or (B, T, z_dim)
        bin_labels: (B,) or (B, T)  int64 ∈ {0,...,n_bins-1}
        """
        logits = self.forward(z)
        if logits.dim() == 3:
            # (B, T, n_bins) → (B*T, n_bins)
            B, T, _ = logits.shape
            logits = logits.reshape(B * T, -1)
            bin_labels = bin_labels.reshape(B * T)
        return F.cross_entropy(logits, bin_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: cumulative reward bin labels
# ─────────────────────────────────────────────────────────────────────────────

class RewardBinDataset(Dataset):
    """
    KoopmanCVAE 인코딩 + cumulative reward bin label 데이터셋.

    각 window에서:
      - x_seq, actions → model.encode_sequence() → z_seq (T, m)
      - rewards → cumulative sum → bin label (T,)
        bin_label[t] = int(cumulative_reward_at_end_of_window.round())
        clamp to [0, n_bins-1]

    Offline 인코딩 캐시를 사용하여 속도 향상.
    """
    def __init__(
        self,
        model:        KoopmanCVAE,
        x_seq_full:   np.ndarray,    # (N, 2108)
        episodes:     List[Dict],
        seq_len:      int   = 64,
        stride:       int   = 32,
        n_bins:       int   = 5,
        device:       str   = 'cuda',
        max_episodes: int   = None,
    ):
        self.n_bins  = n_bins
        self.seq_len = seq_len
        self.device  = device

        # 인코딩 캐시: (z_seq, bin_labels) per window
        self.z_windows:   List[np.ndarray] = []  # (T, m)
        self.bin_windows: List[np.ndarray] = []  # (T,)

        dev = torch.device(device)
        model.eval()

        eps = episodes[:max_episodes] if max_episodes else episodes
        print(f"\n[Dataset] Encoding {len(eps)} episodes, seq_len={seq_len}...")

        for ep_i, ep in enumerate(eps):
            L      = ep['length']
            s_t    = ep['start_t']
            acts   = ep['actions']   # (L, 9)
            rews   = ep['rewards']   # (L,)

            if L < seq_len:
                continue

            x_ep = x_seq_full[s_t:s_t + L]  # (L, 2108)

            # encode full episode
            with torch.no_grad():
                x_t  = torch.FloatTensor(x_ep).unsqueeze(0).to(dev)
                a_t  = torch.FloatTensor(acts).unsqueeze(0).to(dev)
                enc  = model.encode_sequence(x_t, a_t)
                z_ep = enc['o_seq'][0].cpu().numpy()   # (L, m)

            # cumulative reward
            cum_r = np.cumsum(rews)   # (L,)

            # sliding windows
            for start in range(0, L - seq_len + 1, stride):
                end = start + seq_len
                z_w   = z_ep[start:end]    # (T, m)

                # bin label per timestep:
                # = cumulative reward accumulated from start to t
                # t=0: cum_r[start]-cum_r[start-1] etc
                # 더 직관적으로: 각 t에서의 누적 reward 절댓값
                r_seg = cum_r[start:end]
                # relative to window start
                r_rel = r_seg - (cum_r[start - 1] if start > 0 else 0)
                bin_labels = np.clip(np.round(r_rel).astype(np.int64),
                                     0, n_bins - 1)

                self.z_windows.append(z_w.astype(np.float32))
                self.bin_windows.append(bin_labels.astype(np.int64))

            if (ep_i + 1) % 100 == 0:
                print(f"  ep {ep_i+1}/{len(eps)}  windows={len(self.z_windows)}")

        print(f"[Dataset] Total windows: {len(self.z_windows)}")

        # Label distribution 출력
        all_labels = np.concatenate(self.bin_windows)
        for k in range(n_bins):
            count = (all_labels == k).sum()
            print(f"  bin {k}: {count:7d} ({100*count/len(all_labels):.1f}%)")

    def __len__(self): return len(self.z_windows)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.z_windows[idx]),    # (T, m)
            torch.LongTensor(self.bin_windows[idx]),   # (T,)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def replace_reward_head(model: KoopmanCVAE,
                        n_bins: int = 5) -> CategoricalRewardHead:
    """
    model.decoder.head_reward (BCE, 1-dim) 또는 model.reward_head를
    CategoricalRewardHead (n_bins-dim)으로 교체.

    기존 BCE head의 첫 번째 층 weights를 재사용해서
    빠른 수렴을 유도 (transfer).
    """
    z_dim  = model.cfg.koopman_dim
    hidden = model.cfg.mlp_hidden

    cat_head = CategoricalRewardHead(
        z_dim=z_dim, hidden=hidden, n_bins=n_bins,
        n_layers=2, dropout=0.1
    )

    # 기존 head weights transfer (가능한 경우)
    old_head = None
    if hasattr(model.decoder, 'head_reward'):
        old_head = model.decoder.head_reward
        print("  Replacing model.decoder.head_reward → CategoricalRewardHead")
    elif hasattr(model, 'reward_head'):
        old_head = model.reward_head
        print("  Replacing model.reward_head → CategoricalRewardHead")

    if old_head is not None:
        # 첫 번째 Linear layer weights transfer
        try:
            old_w = old_head[0].weight.data  # (hidden, z_dim)
            old_b = old_head[0].bias.data
            with torch.no_grad():
                cat_head.net[0].weight.data[:] = old_w
                cat_head.net[0].bias.data[:]   = old_b
            print("  Transferred first layer weights from old head")
        except Exception as e:
            print(f"  Weight transfer failed ({e}), using random init")

    return cat_head


def freeze_except_reward(model: KoopmanCVAE):
    """reward_head 제외 전체 freeze"""
    for p in model.parameters():
        p.requires_grad_(False)
    print("  All params frozen")


def train_reward_head(
    model:      KoopmanCVAE,
    cat_head:   CategoricalRewardHead,
    train_ds:   RewardBinDataset,
    val_ds:     RewardBinDataset,
    args,
):
    """Categorical reward head fine-tune loop"""
    device  = torch.device(args.device)
    cat_head = cat_head.to(device)

    opt = torch.optim.Adam(cat_head.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=2)

    log = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val = float('inf')

    print(f"\n{'='*55}")
    print(f"Training Categorical Reward Head")
    print(f"  z_dim={model.cfg.koopman_dim}  n_bins={cat_head.n_bins}")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}")
    print(f"{'='*55}\n")

    for epoch in range(1, args.epochs + 1):

        # ── Train ────────────────────────────────────────────────────────────
        cat_head.train()
        train_losses = []
        for z_batch, label_batch in train_loader:
            z_batch     = z_batch.to(device)      # (B, T, m)
            label_batch = label_batch.to(device)  # (B, T)

            loss = cat_head.loss(z_batch, label_batch)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(cat_head.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)

        # ── Val ──────────────────────────────────────────────────────────────
        cat_head.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for z_batch, label_batch in val_loader:
                z_batch     = z_batch.to(device)
                label_batch = label_batch.to(device)
                loss = cat_head.loss(z_batch, label_batch)
                val_losses.append(loss.item())

                # accuracy
                logits = cat_head(z_batch.reshape(-1, z_batch.shape[-1]))
                preds  = logits.argmax(dim=-1)
                lbls   = label_batch.reshape(-1)
                correct += (preds == lbls).sum().item()
                total   += lbls.numel()

        val_loss = np.mean(val_losses)
        val_acc  = correct / total

        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"acc={val_acc*100:.1f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            _save(model, cat_head, args.out.replace('.pt', '_best.pt'),
                  epoch, val_loss, val_acc)

    # Final save
    _save(model, cat_head, args.out, args.epochs, val_loss, val_acc)
    _visualize(log, args.out.replace('.pt', '_training.png'))
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Saved: {args.out}")


def _save(model, cat_head, path, epoch, val_loss, val_acc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state':    model.state_dict(),
        'cfg':            model.cfg,
        'cat_head_state': cat_head.state_dict(),
        'n_bins':         cat_head.n_bins,
        'epoch':          epoch,
        'val_loss':       val_loss,
        'val_acc':        val_acc,
    }, path)


def _visualize(log, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    PAL = ['#E53935', '#1E88E5', '#43A047']

    for ax, (key, title, col) in zip(axes, [
        ('train_loss', 'Train Loss (CE)', PAL[0]),
        ('val_loss',   'Val Loss (CE)',   PAL[1]),
        ('val_acc',    'Val Accuracy',    PAL[2]),
    ]):
        vals = np.array(log[key])
        ax.plot(vals, color=col, lw=1.5)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('epoch'); ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Categorical Reward Head Fine-tune', fontsize=12, fontweight='bold')
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# kodaq_online.py에서 로드하는 helper
# ─────────────────────────────────────────────────────────────────────────────

def load_cat_reward_model(ckpt_path: str,
                          device: str) -> Tuple[KoopmanCVAE,
                                                CategoricalRewardHead]:
    """
    kodaq_online.py에서 호출:
        model, cat_head = load_cat_reward_model(path, device)

    사용 예:
        z_t = ...  # (1, m)
        r_hat = cat_head.expected_reward(z_t).item()
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)

    n_bins   = ckpt.get('n_bins', 5)
    cat_head = CategoricalRewardHead(
        z_dim   = model.cfg.koopman_dim,
        hidden  = model.cfg.mlp_hidden,
        n_bins  = n_bins,
    )
    cat_head.load_state_dict(ckpt['cat_head_state'])
    cat_head.eval().to(device)

    print(f"Loaded cat_reward model: n_bins={n_bins}  "
          f"val_loss={ckpt.get('val_loss', 'N/A'):.4f}  "
          f"val_acc={ckpt.get('val_acc', 0)*100:.1f}%")
    return model, cat_head


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='checkpoints/kodaq_v4/final.pt')
    p.add_argument('--x_cache',    default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--out',        default='checkpoints/kodaq_v4/cat_reward/final.pt')
    p.add_argument('--quality',    default='mixed')
    p.add_argument('--n_bins',     type=int,   default=5,
                   help='Number of reward bins (kitchen max=4 → 5 bins: 0~4)')
    p.add_argument('--seq_len',    type=int,   default=64)
    p.add_argument('--stride',     type=int,   default=32)
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--val_split',  type=float, default=0.1)
    p.add_argument('--max_ep',     type=int,   default=None,
                   help='Max episodes to encode (None=all)')
    p.add_argument('--device',     default='cuda:1' if torch.cuda.is_available()
                                            else 'cpu')
    args = p.parse_args()

    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"\nLoading: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = KoopmanCVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    print(f"  K={model.cfg.num_skills}  m={model.cfg.koopman_dim}")

    # ── Freeze all except reward head ─────────────────────────────────────────
    freeze_except_reward(model)

    # ── Replace reward head ───────────────────────────────────────────────────
    cat_head = replace_reward_head(model, n_bins=args.n_bins)
    total_params = sum(p.numel() for p in cat_head.parameters())
    print(f"  CategoricalRewardHead params: {total_params:,}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading x_sequences: {args.x_cache}")
    x_seq_full, _, _ = load_x_sequences(args.x_cache)
    episodes, _      = load_kitchen_episodes(quality=args.quality, min_len=args.seq_len)
    print(f"  Episodes: {len(episodes)}")

    # Train/val split
    n_val   = max(1, int(len(episodes) * args.val_split))
    np.random.seed(42)
    perm    = np.random.permutation(len(episodes))
    val_eps = [episodes[i] for i in perm[:n_val]]
    trn_eps = [episodes[i] for i in perm[n_val:]]

    print(f"\nBuilding datasets...")
    train_ds = RewardBinDataset(
        model, x_seq_full, trn_eps,
        seq_len=args.seq_len, stride=args.stride,
        n_bins=args.n_bins, device=device,
        max_episodes=args.max_ep,
    )
    val_ds = RewardBinDataset(
        model, x_seq_full, val_eps,
        seq_len=args.seq_len, stride=args.stride,
        n_bins=args.n_bins, device=device,
        max_episodes=args.max_ep,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    train_reward_head(model, cat_head, train_ds, val_ds, args)

    print(f"\n{'='*55}")
    print(f"Fine-tune complete.")
    print(f"  Output: {args.out}")
    print(f"\nUsage in kodaq_online.py:")
    print(f"  from train_reward_head import load_cat_reward_model")
    print(f"  model, cat_head = load_cat_reward_model('{args.out}', device)")
    print(f"  r_hat = cat_head.expected_reward(z_t).item()")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()