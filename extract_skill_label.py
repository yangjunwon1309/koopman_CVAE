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
from pathlib import Path
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter


# ─────────────────────────────────────────────────────────────
# State Embedding  (VLM 대체)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class ExtractClusterConfig:
    K:             int   = 8       # EXTRACT default
    median_window: int   = 7       # EXTRACT default
    state_dim:          int   = 60
    use_object_only:    bool  = True   # True: object states(18:60) only
    # True가 권장: qvel(9:18)은 noise, object states만 skill 구분에 유효
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

# Kitchen D4RL observation layout (60-dim)
# [0:9]   robot qpos    — joint positions
# [9:18]  robot qvel    — joint velocities  (noisy, skip)
# [18:60] object states — microwave, kettle, burner, light, cabinet...
#                         각 subtask마다 독립적으로 변함 → skill 구분 핵심
KITCHEN_OBJ_SLICE = slice(18, 60)   # 42-dim object states only


def compute_state_diff(obs: np.ndarray,
                       use_object_only: bool = True) -> np.ndarray:
    """
    Δs_t = s_t - s_{t-1}.

    use_object_only=True: object states (dim 18~59) 만 사용.
    qvel(dim 9~17)은 noise가 많아 cluster 구분을 방해.
    object states는 subtask별로 독립적으로 변하므로 skill 경계가 명확.

    t=0: Δs_0 = Δs_1 (EXTRACT footnote 방식)
    Returns (N, 42) if use_object_only else (N, 60)
    """
    if use_object_only:
        obs = obs[:, KITCHEN_OBJ_SLICE]   # (N, 42)

    diff = np.diff(obs, axis=0)           # (N-1, d)
    diff = np.vstack([diff[0:1], diff])   # (N, d)
    print(f"State diff Δs_t: {diff.shape}  "
          f"mean_norm={np.linalg.norm(diff, axis=1).mean():.4f}  "
          f"std={diff.std(axis=0).mean():.4f}")
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

    print(f"Computing Δs_t "
          f"({'object states only [18:60]' if cfg.use_object_only else 'full 60-dim'}) ...")
    diff = compute_state_diff(obs, use_object_only=cfg.use_object_only)

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
    return smoothed, logprobs, segs, diff, km


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def visualize_episodes(
    assignments: np.ndarray,
    terminals:   np.ndarray,
    K:           int,
    out_path:    str,
    n_ep:        int = 8,
):
    """Per-episode skill label timeline (색상 bar)."""
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
    print(f"Saved episode viz: {out_path}")


def visualize_pca_clusters(
    diff:        np.ndarray,    # (N, 60) — state differences Δs_t
    assignments: np.ndarray,    # (N,) cluster labels
    km:          KMeans,        # fitted KMeans (for centroids)
    K:           int,
    out_path:    str,
    subsample:   int = 3000,    # 너무 많으면 느리므로 서브샘플
):
    """
    전체 데이터셋의 embedding difference를 PCA 2D로 투영하여
    cluster 분포 시각화. 이미지와 동일한 스타일:
    - scatter: 각 점을 cluster 색상으로
    - centroid: 굵은 마커 + cluster 이름
    - Voronoi 경계선: centroid 간 수직이등분선
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA
    from scipy.spatial import Voronoi

    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
           '#8E24AA','#00ACC1','#FFB300','#6D4C41',
           '#546E7A','#D81B60']

    # ── 전처리: percentile clip + StandardScaler ────────────
    # Δs_t는 대부분 0 근방 (steady state) + 소수 큰 값 (skill boundary)
    # → heavy-tail 분포: StandardScaler + 3σ clip 으로도 뭉침
    #
    # Fix: percentile 1~99 범위로 먼저 clip → StandardScaler
    #      PCA 후에도 시각화 범위를 percentile로 추가 제한
    from sklearn.preprocessing import StandardScaler

    # Step 1: per-dim percentile clip (1~99th)
    lo  = np.percentile(diff, 1,  axis=0)
    hi  = np.percentile(diff, 99, axis=0)
    diff_c = np.clip(diff, lo, hi)

    # Step 2: StandardScaler
    scaler = StandardScaler()
    diff_s = scaler.fit_transform(diff_c)           # (N, d)

    # ── PCA 2D 투영 ──────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    Z2  = pca.fit_transform(diff_s)                 # (N, 2)
    var = pca.explained_variance_ratio_

    # centroid도 동일한 전처리 + PCA
    cent_c = np.clip(km.cluster_centers_, lo, hi)
    C2     = pca.transform(scaler.transform(cent_c))

    # 서브샘플 (시각화 속도)
    if len(Z2) > subsample:
        idx = np.random.choice(len(Z2), subsample, replace=False)
        Z2s = Z2[idx]; Ls = assignments[idx]
    else:
        Z2s = Z2; Ls = assignments

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#f8f8f8')

    # ── Voronoi 경계선 ───────────────────────────────────────
    # centroid 간 수직이등분선을 그려 영역 구분
    try:
        if K >= 4:
            # scipy Voronoi는 K>=4 필요
            vor = Voronoi(C2)
            for ridge_pts, ridge_vert in zip(vor.ridge_points,
                                              vor.ridge_vertices):
                if -1 in ridge_vert:
                    # 무한 ridge: 두 centroid의 색 중간으로 연장선
                    i, j   = ridge_pts
                    mid    = (C2[i] + C2[j]) / 2
                    tang   = C2[j] - C2[i]
                    tang   = tang / (np.linalg.norm(tang) + 1e-8)
                    perp   = np.array([-tang[1], tang[0]])
                    # 화면 밖으로 충분히 연장
                    far    = mid + perp * 50
                    ax.plot([mid[0], far[0]], [mid[1], far[1]],
                            color='#cccccc', lw=0.8, zorder=1)
                else:
                    v0, v1 = vor.vertices[ridge_vert[0]], vor.vertices[ridge_vert[1]]
                    ax.plot([v0[0], v1[0]], [v0[1], v1[1]],
                            color='#cccccc', lw=0.8, zorder=1)
        else:
            # K<4: centroid 간 직선만
            for i in range(K):
                for j in range(i+1, K):
                    mid = (C2[i] + C2[j]) / 2
                    ax.plot([C2[i,0], C2[j,0]], [C2[i,1], C2[j,1]],
                            color='#dddddd', lw=0.6, zorder=1)
    except Exception:
        pass  # Voronoi 실패 시 무시

    # ── Scatter: 각 점 ───────────────────────────────────────
    for k in range(K):
        mask = Ls == k
        if mask.sum() == 0:
            continue
        c = PAL[k % len(PAL)]
        ax.scatter(Z2s[mask, 0], Z2s[mask, 1],
                   color=c, alpha=0.45, s=22,
                   edgecolors='white', linewidths=0.3,
                   zorder=2)

    # ── Centroids + 라벨 ─────────────────────────────────────
    for k in range(K):
        if (assignments == k).sum() == 0:
            continue
        c = PAL[k % len(PAL)]
        ax.scatter(C2[k, 0], C2[k, 1],
                   color=c, s=160, marker='o',
                   edgecolors='white', linewidths=1.5,
                   zorder=4)
        ax.text(C2[k, 0], C2[k, 1] + (Z2[:,1].max()-Z2[:,1].min())*0.04,
                f'Cluster {k}',
                color=c, fontsize=9, fontweight='bold',
                ha='center', va='bottom', zorder=5)

    # ── 축 + 타이틀 ──────────────────────────────────────────
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=10)
    ax.set_title(
        f'PCA Cluster Embeddings  K={K}  N={len(diff):,}  '
        f'(var {var[0]*100:.0f}+{var[1]*100:.0f}={sum(var)*100:.0f}%)',
        fontsize=11, fontweight='bold')

    # 서브샘플 크기 표시
    if len(Z2) > subsample:
        ax.text(0.01, 0.01,
                f'(subsample {subsample:,}/{len(Z2):,})',
                transform=ax.transAxes, fontsize=7,
                color='#888888')

    # 범례
    handles = [Patch(color=PAL[k % len(PAL)], label=f'Cluster {k}')
               for k in range(K) if (assignments == k).sum() > 0]
    ax.legend(handles=handles, fontsize=8, loc='lower right',
              framealpha=0.8, ncol=2)

    # ── 시각화 범위: PCA 후에도 percentile 5~95로 제한 ─────────
    # 극소수 outlier가 전체를 작게 보이게 만드는 것을 방지
    x_lo, x_hi = np.percentile(Z2[:,0], [2, 98])
    y_lo, y_hi = np.percentile(Z2[:,1], [2, 98])
    margin = 0.1
    xr = x_hi - x_lo; yr = y_hi - y_lo
    ax.set_xlim(x_lo - xr*margin, x_hi + xr*margin)
    ax.set_ylim(y_lo - yr*margin, y_hi + yr*margin)

    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA viz: {out_path}")


# backward-compat alias
def visualize(assignments, terminals, K, out_path, n_ep=8):
    visualize_episodes(assignments, terminals, K, out_path, n_ep)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out',      default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--K',        type=int, default=8)
    p.add_argument('--window',   type=int, default=7)
    p.add_argument('--env',      default='kitchen-mixed-v0')
    p.add_argument('--device',   default='cuda')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--viz',      default='checkpoints/skill_pretrain/cluster_viz.png')
    args = p.parse_args()

    if args.visualize:
        import d4rl, gym
        env     = gym.make(args.env)
        dataset = env.get_dataset()
        term    = dataset['terminals'].astype(bool)
        obs     = dataset['observations']
        asgn, _ = load_cluster_data(args.out)
        K_viz   = int(asgn.max()) + 1
        # Episode timeline
        visualize_episodes(asgn, term, K_viz, args.viz)
        # PCA: 재계산
        print("Computing Δs_t for PCA viz ...")
        diff_v = compute_state_diff(obs, use_object_only=True)
        # centroid: 각 cluster 평균으로 설정 (K-means 재학습 없음)
        from sklearn.cluster import KMeans
        km_v = KMeans.__new__(KMeans)
        km_v.cluster_centers_ = np.array(
            [diff_v[asgn==k].mean(axis=0) if (asgn==k).sum() > 0
             else np.zeros(diff_v.shape[1])
             for k in range(K_viz)])
        pca_path = args.viz.replace('.png', '_pca.png')
        visualize_pca_clusters(diff_v, asgn, km_v, K_viz, pca_path)
        return

    cfg = ExtractClusterConfig(
        K=args.K, median_window=args.window, device=args.device)

    smoothed, logprobs, segs, diff, km = run_extract_pipeline(
        cfg, args.out, args.env)

    try:
        import d4rl, gym
        term = gym.make(args.env).get_dataset()['terminals'].astype(bool)
        # 1. Per-episode skill timeline
        visualize_episodes(smoothed, term, cfg.K, args.viz)
        # 2. PCA cluster distribution
        pca_path = args.viz.replace('.png', '_pca.png')
        visualize_pca_clusters(diff, smoothed, km, cfg.K, pca_path)
    except Exception as ex:
        print(f"Viz failed: {ex}")

    print(f"\nDone. cluster_data.h5 → {args.out}")


if __name__ == '__main__':
    main()