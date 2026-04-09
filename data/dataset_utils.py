"""
dataset_utils.py
================
KODAQ full RSSM-Koopman용 Dataset 클래스.

핵심 변경:
  - 입력이 raw state (60-dim) 아닌 x_t = [Δe_t, Δp_t, q_t, q̇_t] ∈ ℝ^{2108}
  - skill_labels (B, T) int64 → EXTRACT cluster assignments ĉ_t
  - 데이터 소스:
      (1) x_sequences.npz 캐시 (run_extract_pipeline() 생성) — 권장
      (2) skill_segments (split_into_skill_segments() 결과)
      (3) synthetic fallback (quick test)

KODAQDataset:
  __getitem__ → (x_seq, actions, skill_labels) 각각 (T, x_dim), (T, 9), (T,)
  collate_fn  → padding + mask 지원 (가변 길이 segments)
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset
from typing import Dict, List, Optional, Tuple

from data.extract_skill_label import (
    X_DIM, DIM_DELTA_E, DIM_DELTA_P, DIM_Q, DIM_QDOT,
    run_extract_pipeline, load_x_sequences, load_cluster_data,
    ExtractClusterConfig,
    split_into_skill_segments,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window dataset over flat x_t, actions, skill_labels arrays
# ──────────────────────────────────────────────────────────────────────────────

class KODAQWindowDataset(Dataset):
    """
    Sliding-window segmentation of flat (N,) arrays.

    Episode boundaries are respected: windows do not cross terminal flags.

    __getitem__ returns:
        x_seq:        (T, 2108)  float32
        actions:      (T, 9)     float32
        skill_labels: (T,)       int64
        rewards:      (T,)       float32  — reward diff (0/1 sparse)
    """

    def __init__(
        self,
        x_seq:        np.ndarray,   # (N, 2108)
        actions:      np.ndarray,   # (N, 9)
        skill_labels: np.ndarray,   # (N,) int
        terminals:    np.ndarray,   # (N,) bool
        seq_len:      int = 64,
        stride:       int = None,
        rewards:      np.ndarray = None,  # (N,) float — reward diff
    ):
        self.x_seq        = x_seq.astype(np.float32)
        self.actions      = actions.astype(np.float32)
        self.skill_labels = skill_labels.astype(np.int64)
        # reward diff: diff of cumulative reward → sparse 0/1
        if rewards is not None:
            r = rewards.astype(np.float32)
            r_diff = np.diff(r, prepend=r[0])
            r_diff = np.clip(r_diff, 0, 1)   # 0 or 1
            self.rewards = r_diff
        else:
            self.rewards = np.zeros(len(x_seq), dtype=np.float32)
        self.seq_len      = seq_len
        if stride is None:
            stride = seq_len // 2
        self.stride       = stride

        self.windows = self._build_windows(terminals)
        print(f"KODAQWindowDataset: {len(self.windows)} windows  "
              f"seq_len={seq_len}  stride={stride}  "
              f"N={len(x_seq)}  x_dim={x_seq.shape[1]}  "
              f"reward_rate={self.rewards.mean():.4f}")

    def _build_windows(self, terminals: np.ndarray) -> List[int]:
        """
        Returns list of start indices for valid windows.
        A window [start, start+seq_len) is valid if it contains no terminal.
        (Terminal at the very end of the window is allowed.)
        """
        N    = len(self.x_seq)
        T    = self.seq_len
        ends = set(np.where(terminals)[0].tolist())
        windows = []

        for start in range(0, N - T + 1, self.stride):
            end = start + T - 1
            # Check if any terminal lies strictly inside [start, end-1]
            # (terminal at end is OK: episode boundary after this window)
            interior_terminal = any(t in ends for t in range(start, end))
            if not interior_terminal:
                windows.append(start)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        end   = start + self.seq_len
        return (
            torch.from_numpy(self.x_seq[start:end]),         # (T, 2108)
            torch.from_numpy(self.actions[start:end]),        # (T, 9)
            torch.from_numpy(self.skill_labels[start:end]),   # (T,)
            torch.from_numpy(self.rewards[start:end]),        # (T,) float32
        )


# ──────────────────────────────────────────────────────────────────────────────
# Skill-segment-based dataset (variable-length segments → collate with padding)
# ──────────────────────────────────────────────────────────────────────────────

class KODAQSegmentDataset(Dataset):
    """
    One sample = one EXTRACT skill segment (variable length).
    Use with collate_fn_pad for DataLoader.

    __getitem__ returns:
        x_seq:        (L, 2108)  float32
        actions:      (L, 9)     float32
        skill_labels: (L,)       int64
        length:       int
    """

    def __init__(self, segments: List[Dict]):
        self.segs = segments
        lengths   = [s['length'] for s in segments]
        skill_ids = [int(s['skills'][0]) for s in segments]
        print(f"KODAQSegmentDataset: {len(segments)} segments  "
              f"len=[{min(lengths)},{max(lengths)}]  "
              f"mean={np.mean(lengths):.1f}")

    def __len__(self) -> int:
        return len(self.segs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segs[idx]
        return {
            'x_seq':        torch.from_numpy(seg['x_seq'].astype(np.float32)),
            'actions':      torch.from_numpy(seg['actions'].astype(np.float32)),
            'skill_labels': torch.from_numpy(seg['skills'].astype(np.int64)),
            'length':       seg['length'],
        }


def collate_fn_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length segments to max length in batch.
    Returns mask (B, T) bool — True = valid timestep.
    """
    max_len = max(b['length'] for b in batch)
    B       = len(batch)
    x_dim   = batch[0]['x_seq'].shape[-1]
    a_dim   = batch[0]['actions'].shape[-1]

    x_pad   = torch.zeros(B, max_len, x_dim)
    a_pad   = torch.zeros(B, max_len, a_dim)
    lbl_pad = torch.zeros(B, max_len, dtype=torch.long)
    mask    = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b['length']
        x_pad[i,   :L] = b['x_seq']
        a_pad[i,   :L] = b['actions']
        lbl_pad[i, :L] = b['skill_labels']
        mask[i,    :L] = True

    return {
        'x_seq':        x_pad,
        'actions':      a_pad,
        'skill_labels': lbl_pad,
        'mask':         mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main loader: builds dataset from cache or runs full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def load_kodaq_dataset(
    env_name:   str  = 'kitchen-mixed-v0',
    seq_len:    int  = 64,
    stride:     int  = None,
    use_r3m:    bool = True,
    K:          int  = 8,
    out_dir:    str  = 'checkpoints/skill_pretrain',
    pca_dim:    int  = 64,
    device:     str  = 'cuda',
    mode:       str  = 'window',   # 'window' | 'segment'
) -> Dataset:
    """
    Full KODAQ dataset loader.

    1. Checks cache (x_sequences.npz + cluster_data.h5)
    2. If missing: runs EXTRACT pipeline (R3M render → K-means → segments)
    3. Returns KODAQWindowDataset or KODAQSegmentDataset

    mode='window'  : sliding window over flat arrays (recommended for training)
    mode='segment' : one sample per EXTRACT skill segment (variable length)
    """
    out_dir_p  = Path(out_dir)
    h5_path    = str(out_dir_p / 'cluster_data.h5')
    x_cache    = str(out_dir_p / 'x_sequences.npz')

    # ── Check / run pipeline ──────────────────────────────────────────────────
    if not Path(h5_path).exists() or not Path(x_cache).exists():
        print(f"Cache not found. Running EXTRACT pipeline...")
        cfg = ExtractClusterConfig(
            K=K, use_r3m=use_r3m, pca_dim=pca_dim,
            device=device, env_name=env_name,
        )
        run_extract_pipeline(cfg, h5_path, x_cache)

    # ── Load from cache ───────────────────────────────────────────────────────
    print(f"Loading from cache: {x_cache}")
    x_seq, actions, terminals = load_x_sequences(x_cache)

    print(f"Loading skill labels: {h5_path}")
    assignments, logprobs = load_cluster_data(h5_path)

    # ── Load rewards from D4RL (for reward head) ──────────────────────────────
    rewards = None
    try:
        import d4rl, gym
        ds      = gym.make(env_name).get_dataset()
        rewards = ds['rewards'].astype(np.float32)   # (N,) cumulative sparse
        print(f"  Rewards loaded: shape={rewards.shape}  "
              f"nonzero={( rewards > 0).sum()}")
    except Exception as e:
        print(f"  Rewards not available ({e}). Using zeros.")

    print(f"  x_seq={x_seq.shape}  actions={actions.shape}  "
          f"terminals={terminals.sum()}  K={assignments.max()+1}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    if mode == 'window':
        return KODAQWindowDataset(
            x_seq=x_seq,
            actions=actions,
            skill_labels=assignments,
            terminals=terminals,
            seq_len=seq_len,
            stride=stride,
            rewards=rewards,
        )
    elif mode == 'segment':
        # Need obs to build segments (for backward compat)
        # Load obs from D4RL
        try:
            import d4rl, gym
            obs = gym.make(env_name).get_dataset()['observations']
        except Exception:
            obs = np.zeros((len(x_seq), 60), dtype=np.float32)

        segs = split_into_skill_segments(
            obs=obs, actions=actions, x_seq=x_seq,
            terminals=terminals,
            cluster_assignments=assignments,
            cluster_logprobs=logprobs,
        )
        return KODAQSegmentDataset(segs)
    else:
        raise ValueError(f"mode must be 'window' or 'segment', got '{mode}'")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (quick testing without D4RL/R3M)
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_dataset(
    n_samples: int = 1000,
    seq_len:   int = 64,
    K:         int = 8,
) -> KODAQWindowDataset:
    """
    Gaussian noise dataset for unit testing / quick runs.
    skill_labels: random integers in [0, K).
    """
    N    = n_samples * seq_len
    x    = np.random.randn(N, X_DIM).astype(np.float32) * 0.3
    acts = np.random.randn(N, 9).astype(np.float32) * 0.1
    lbls = np.random.randint(0, K, size=N).astype(np.int64)

    # Fake terminals: every seq_len steps
    terms = np.zeros(N, dtype=bool)
    terms[seq_len - 1::seq_len] = True

    print(f"Synthetic dataset: N={N}  x_dim={X_DIM}  K={K}")
    return KODAQWindowDataset(x, acts, lbls, terms, seq_len=seq_len)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-quality Kitchen loader
# ──────────────────────────────────────────────────────────────────────────────

def load_kitchen_all_qualities(
    seq_len:  int  = 64,
    use_r3m:  bool = True,
    K:        int  = 8,
    out_dir:  str  = 'checkpoints/skill_pretrain',
    device:   str  = 'cuda',
) -> KODAQWindowDataset:
    """
    Merge complete + partial + mixed kitchen datasets into one window dataset.
    Each quality tier has its own x_cache / cluster_data.h5.
    """
    envs = ['kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0']
    all_x, all_a, all_l, all_t = [], [], [], []
    offset = 0

    for env_name in envs:
        env_tag = env_name.split('-')[1]   # 'complete' | 'partial' | 'mixed'
        env_dir = str(Path(out_dir) / env_tag)
        try:
            ds = load_kodaq_dataset(
                env_name=env_name, seq_len=seq_len,
                use_r3m=use_r3m, K=K,
                out_dir=env_dir, device=device, mode='window',
            )
            all_x.append(ds.x_seq)
            all_a.append(ds.actions)
            all_l.append(ds.skill_labels)
            # Rebuild terminals from windows
            N = len(ds.x_seq)
            t_arr = np.zeros(N, dtype=bool)
            # Use original terminals embedded in KODAQWindowDataset
            # (re-run to get terminals — load from env_dir/x_sequences.npz)
            x_cache = str(Path(env_dir) / 'x_sequences.npz')
            _, _, terms = load_x_sequences(x_cache)
            all_t.append(terms)
            print(f"  {env_tag}: {N} steps")
        except Exception as e:
            print(f"  {env_name} failed: {e}")

    if not all_x:
        raise RuntimeError("All Kitchen quality tiers failed.")

    x_cat    = np.concatenate(all_x)
    a_cat    = np.concatenate(all_a)
    l_cat    = np.concatenate(all_l)
    t_cat    = np.concatenate(all_t)

    return KODAQWindowDataset(x_cat, a_cat, l_cat, t_cat, seq_len=seq_len)