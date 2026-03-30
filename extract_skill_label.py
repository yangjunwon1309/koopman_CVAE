"""
extract_skill_label.py
=======================
EXTRACT (CoRL 2024) kitchen_data_loader.py 로직을 최대한 그대로 활용.

EXTRACT 원본과의 차이:
  - 이미지(rendered_frames) 대신 observations(60-dim state) 사용
  - VLM embedding 대신 MLP로 state embedding 추출
  - cluster_data를 EXTRACT와 동일한 h5py 형식으로 저장

EXTRACT 원본 pipeline:
  1. generate_kitchen_data.py → rendered_frames 생성
  2. VLM embedding → Δe_t → K-means(K=8) → median filter(window=7)
  3. cluster_data.h5: clusters(N,), logprobs(K,N)
  4. SkillClusterD4RLSequenceSplitDataset:
     seq_end_idxs  = terminals 위치
     skill_end_idxs = cluster_assignments 변화 위치
     두 경계 합쳐서 segment 분리

Usage:
    python extract_skill_label.py --out checkpoints/skill_pretrain/cluster_data.h5
    python extract_skill_label.py --out checkpoints/skill_pretrain/cluster_data.h5 --visualize
"""

import os, sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter


# ─────────────────────────────────────────────────────────────
# State Embedding  (VLM 대체)
# ─────────────────────────────────────────────────────────────

class StateEmbedder(nn.Module):
    """
    60-dim state → embed_dim. EXTRACT의 VLM embedding에 대응.
    Pre-trained weights 불필요: random init MLP도 충분히
    state space를 embed_dim으로 projection.
    """
    def __init__(self, state_dim: int = 60, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class ExtractClusterConfig:
    K:             int   = 8       # EXTRACT default
    median_window: int   = 7       # EXTRACT default
    embed_dim:     int   = 64
    state_dim:     int   = 60
    kmeans_n_init: int   = 20
    kmeans_seed:   int   = 42
    min_seg_len:   int   = 5
    device:        str   = 'cuda'


# ─────────────────────────────────────────────────────────────
# D4RL 로드
# ─────────────────────────────────────────────────────────────

def load_d4rl_flat(env_name: str = 'kitchen-mixed-v0'):
    """
    D4RL flat dataset 로드.
    EXTRACT가 사용하는 키: terminals, observations, actions
    """
    import d4rl, gym
    env     = gym.make(env_name)
    dataset = env.get_dataset()
    obs       = dataset['observations']   # (N, 60)
    actions   = dataset['actions']        # (N, 9)
    terminals = dataset['terminals'].astype(bool)  # (N,)
    print(f"D4RL '{env_name}': {len(obs)} steps  "
          f"episodes={terminals.sum()}")
    return obs, actions, terminals


# ─────────────────────────────────────────────────────────────
# Embedding + Difference
# ─────────────────────────────────────────────────────────────

def compute_state_embeddings(
    obs:       np.ndarray,
    embedder:  StateEmbedder,
    device:    str = 'cuda',
    batch:     int = 4096,
) -> np.ndarray:
    """per-timestep state → embedding. Returns (N, embed_dim)"""
    embedder.eval()
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    embedder.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            x = torch.FloatTensor(obs[i:i+batch]).to(dev)
            out.append(embedder(x).cpu().numpy())
    E = np.vstack(out)
    print(f"Embeddings: {E.shape}  "
          f"mean_norm={np.linalg.norm(E, axis=1).mean():.4f}")
    return E


def compute_embedding_differences(E: np.ndarray) -> np.ndarray:
    """
    EXTRACT: Δe_t = e_t - e_{t-1}
    t=0: Δe_0 = Δe_1  (논문 footnote: e_1 ← e_2)
    Returns (N, embed_dim)
    """
    diff = np.diff(E, axis=0)             # (N-1, d)
    diff = np.vstack([diff[0:1], diff])   # (N, d)
    return diff


# ─────────────────────────────────────────────────────────────
# K-means
# ─────────────────────────────────────────────────────────────

def run_kmeans(
    diff:    np.ndarray,
    K:       int,
    n_init:  int = 20,
    seed:    int = 42,
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    """
    EXTRACT: K-means on Δe_t.
    Returns (km, labels (N,), logprobs (K, N))
    logprobs = negative distance to each centroid (proxy)
    """
    print(f"K-means K={K} on {diff.shape} ...")
    km = KMeans(n_clusters=K, init='k-means++',
                n_init=n_init, random_state=seed, max_iter=500)
    km.fit(diff)
    labels   = km.labels_                              # (N,)
    dists    = km.transform(diff)                      # (N, K)
    logprobs = -dists.T                                # (K, N)
    counts   = np.bincount(labels, minlength=K)
    print(f"  Inertia={km.inertia_:.1f}  counts={counts.tolist()}")
    return km, labels, logprobs


# ─────────────────────────────────────────────────────────────
# Median filter (per-episode, EXTRACT 방식)
# ─────────────────────────────────────────────────────────────

def apply_median_filter_per_episode(
    labels:    np.ndarray,   # (N,) int
    terminals: np.ndarray,   # (N,) bool
    window:    int = 7,
) -> np.ndarray:
    """
    EXTRACT: median filter는 각 에피소드 내에서만 적용.
    에피소드 경계를 넘지 않음.
    """
    smoothed = labels.copy()
    ends  = list(np.where(terminals)[0])
    start = 0
    for end_idx in ends:
        seg = labels[start:end_idx + 1]
        if len(seg) >= window:
            seg = medfilt(seg.astype(float),
                          kernel_size=window).astype(int)
        smoothed[start:end_idx + 1] = seg
        start = end_idx + 1
    # 마지막 에피소드
    if start < len(labels):
        seg = labels[start:]
        if len(seg) >= window:
            seg = medfilt(seg.astype(float),
                          kernel_size=window).astype(int)
        smoothed[start:] = seg
    changed = (smoothed != labels).sum()
    print(f"Median filter: {changed}/{len(labels)} labels changed "
          f"({changed/len(labels)*100:.1f}%)")
    return smoothed


# ─────────────────────────────────────────────────────────────
# EXTRACT SkillClusterD4RLSequenceSplitDataset 로직 재현
# ─────────────────────────────────────────────────────────────

def split_into_skill_segments(
    obs:                 np.ndarray,   # (N, 60)
    actions:             np.ndarray,   # (N, 9)
    terminals:           np.ndarray,   # (N,) bool
    cluster_assignments: np.ndarray,   # (N,) int  smoothed
    cluster_logprobs:    np.ndarray,   # (K, N)
    min_seg_len:         int = 5,
) -> List[Dict]:
    """
    EXTRACT SkillClusterD4RLSequenceSplitDataset.__init__ 핵심 로직 재현.

    ─ EXTRACT 코드 (그대로) ─────────────────────────────────
    seq_end_idxs = np.where(self.dataset["terminals"])[0]

    skill_end_idxs = np.where(
        (cluster_assignments[1:] - cluster_assignments[:-1]) != 0
    )[0]

    seq_end_idxs = np.concatenate((seq_end_idxs, skill_end_idxs))
    seq_end_idxs = np.unique(seq_end_idxs)
    ──────────────────────────────────────────────────────────

    각 segment: 같은 skill label을 가진 연속 구간
    (에피소드 경계 또는 skill 변화 지점에서 잘림)
    """
    # ── EXTRACT 코드 그대로 ──────────────────────────────────
    seq_end_idxs = np.where(terminals)[0]

    skill_end_idxs = np.where(
        (cluster_assignments[1:] - cluster_assignments[:-1]) != 0
    )[0]

    seq_end_idxs = np.concatenate((seq_end_idxs, skill_end_idxs))
    seq_end_idxs = np.unique(seq_end_idxs)
    # ────────────────────────────────────────────────────────

    seqs    = []
    skipped = 0
    start   = 0
    lp_T    = cluster_logprobs.T   # (N, K)

    for end_idx in seq_end_idxs:
        length = end_idx + 1 - start
        if length < min_seg_len:
            # EXTRACT: "continue  # skip too short demos"
            # start를 업데이트하지 않음
            skipped += 1
            continue

        seg_actions = actions[start:end_idx + 1]
        skill_progress = (np.linspace(0, 1, len(seg_actions))
                          if len(seg_actions) > 1
                          else np.array([1.0]))

        seqs.append(dict(
            states         = obs[start:end_idx + 1],         # (L, 60)
            actions        = seg_actions,                    # (L, 9)
            skills         = cluster_assignments[start:end_idx + 1],  # (L,)
            skill_logprobs = lp_T[start:end_idx + 1],       # (L, K)
            skill_progress = skill_progress,                  # (L,)
            start_t        = start,
            end_t          = end_idx + 1,
            length         = length,
        ))
        start = end_idx + 1

    # 통계
    if seqs:
        lengths = [s['length'] for s in seqs]
        skill_ids = [int(s['skills'][0]) for s in seqs]
        print(f"Segments: {len(seqs)}  skipped_short={skipped}")
        print(f"  Length: min={min(lengths)} max={max(lengths)} "
              f"mean={np.mean(lengths):.1f}")
        print(f"  Skill distribution: "
              f"{dict(sorted(Counter(skill_ids).items()))}")
    return seqs


# ─────────────────────────────────────────────────────────────
# h5 I/O  (EXTRACT cluster_data.h5 형식)
# ─────────────────────────────────────────────────────────────

def save_cluster_data(
    path:                str,
    cluster_assignments: np.ndarray,   # (N,)
    cluster_logprobs:    np.ndarray,   # (K, N)
):
    """
    EXTRACT cluster_data.h5 형식으로 저장.
    EXTRACT 로드 코드:
        cluster_assignments = f["clusters"][()]
        cluster_logprobs    = f["logprobs"][()]
        cluster_logprobs    = cluster_logprobs.transpose(1, 0)  # → (K, N)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('clusters', data=cluster_assignments)
        f.create_dataset('logprobs', data=cluster_logprobs)
    print(f"Saved: {path}  "
          f"clusters={cluster_assignments.shape}  "
          f"logprobs={cluster_logprobs.shape}")


def load_cluster_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    EXTRACT 형식 h5 파일 로드.
    EXTRACT 코드와 동일하게 logprobs를 transpose.
    Returns: cluster_assignments (N,), cluster_logprobs (K, N)
    """
    with h5py.File(path, 'r') as f:
        assignments = f['clusters'][()]
        logprobs    = f['logprobs'][()]
        if len(logprobs.shape) < 2:
            print("Warning: logprobs wrong shape")
            K = int(assignments.max()) + 1
            logprobs = np.zeros((K, len(assignments)))
        else:
            # EXTRACT: logprobs = logprobs.transpose(1, 0)
            logprobs = logprobs.transpose(1, 0)   # (K, N)
    return assignments, logprobs


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────

def run_extract_pipeline(
    cfg:      ExtractClusterConfig,
    out_h5:   str,
    env_name: str = 'kitchen-mixed-v0',
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """EXTRACT 방식 전체 pipeline 실행."""

    obs, actions, terminals = load_d4rl_flat(env_name)

    print("Computing state embeddings ...")
    embedder = StateEmbedder(cfg.state_dim, cfg.embed_dim)
    E = compute_state_embeddings(obs, embedder, cfg.device)

    print("Computing Δe_t ...")
    diff = compute_embedding_differences(E)

    km, raw_labels, logprobs = run_kmeans(
        diff, cfg.K, cfg.kmeans_n_init, cfg.kmeans_seed)

    smoothed = apply_median_filter_per_episode(
        raw_labels, terminals, cfg.median_window)

    save_cluster_data(out_h5, smoothed, logprobs)

    segs = split_into_skill_segments(
        obs, actions, terminals,
        cluster_assignments=smoothed,
        cluster_logprobs=logprobs,
        min_seg_len=cfg.min_seg_len,
    )
    return smoothed, logprobs, segs


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def visualize(
    assignments: np.ndarray,
    terminals:   np.ndarray,
    K:           int,
    out_path:    str,
    n_ep:        int = 8,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
           '#8E24AA','#00ACC1','#FFB300','#6D4C41',
           '#546E7A','#D81B60']

    ends   = list(np.where(terminals)[0])
    starts = [0] + [e + 1 for e in ends[:-1]]
    eps    = list(zip(starts, ends))[:n_ep]

    fig, axes = plt.subplots(len(eps), 1,
                             figsize=(14, 1.5 * len(eps)))
    if len(eps) == 1:
        axes = [axes]

    for ax, (s, e) in zip(axes, eps):
        lbl = assignments[s:e+1]
        for t in range(len(lbl)):
            ax.axvspan(t, t+1, color=PAL[lbl[t] % len(PAL)],
                       alpha=0.85, linewidth=0)
        ax.set_xlim(0, len(lbl))
        ax.set_yticks([])
        ax.set_ylabel(f'Ep {eps.index((s,e))}',
                      rotation=0, labelpad=40, fontsize=8)

    handles = [Patch(color=PAL[k % len(PAL)], label=f'Skill {k}')
               for k in range(K)]
    fig.legend(handles=handles, loc='upper right',
               ncol=min(K, 4), fontsize=8)
    fig.suptitle(f'EXTRACT Skill Labels  K={K}  '
                 f'(state Δe + K-means + median)', fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out',      default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--K',        type=int, default=8)
    p.add_argument('--window',   type=int, default=7)
    p.add_argument('--embed',    type=int, default=64)
    p.add_argument('--env',      default='kitchen-mixed-v0')
    p.add_argument('--device',   default='cuda')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--viz',      default='checkpoints/skill_pretrain/cluster_viz.png')
    args = p.parse_args()

    if args.visualize:
        import d4rl, gym
        env  = gym.make(args.env)
        term = env.get_dataset()['terminals'].astype(bool)
        asgn, _ = load_cluster_data(args.out)
        visualize(asgn, term, asgn.max()+1, args.viz)
        return

    cfg = ExtractClusterConfig(
        K=args.K, median_window=args.window,
        embed_dim=args.embed, device=args.device)

    smoothed, logprobs, segs = run_extract_pipeline(cfg, args.out, args.env)

    try:
        import d4rl, gym
        term = gym.make(args.env).get_dataset()['terminals'].astype(bool)
        visualize(smoothed, term, cfg.K, args.viz)
    except Exception:
        pass

    print(f"\nDone. cluster_data.h5 → {args.out}")


if __name__ == '__main__':
    main()