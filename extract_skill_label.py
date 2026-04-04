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
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
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
    use_r3m:            bool  = False
    use_object_only:    bool  = True
    r3m_device:         str   = 'cuda'
    img_size:           int   = 128
    pca_dim:            int   = 64     # K-means 전 PCA 차원 축소 (0=off)
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

# ─────────────────────────────────────────────────────────────
# R3M Image Embedding  (EXTRACT 원본 방식)
# ─────────────────────────────────────────────────────────────

def load_r3m(device: str = 'cuda'):
    """
    R3M pretrained visual encoder 로드.
    pip install r3m 필요.
    Returns: (model, transform)  model: image → 2048-dim embedding
    """
    try:
        from r3m import load_r3m as _load
    except ImportError:
        raise ImportError(
            "R3M not installed.\n"
            "  pip install r3m\n"
            "  or: pip install git+https://github.com/facebookresearch/r3m.git"
        )
    model = _load("resnet50")               # ResNet50 backbone, 2048-dim
    model.eval()
    model.to(torch.device(device if torch.cuda.is_available() else 'cpu'))
    # R3M expects [0-255] uint8 images, (B, 3, H, W)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),                       # → [0,1]
    ])
    return model, transform


# 렌더링 설정 (512→crop 128 확인됨)
RENDER_SIZE = 512
CROP_START  = 192
CROP_END    = 320
CROP_SIZE   = CROP_END - CROP_START   # 128


def render_frame(sim) -> np.ndarray:
    """512x512 렌더링 → 중앙 128x128 crop. flip 불필요."""
    frame = sim.render(RENDER_SIZE, RENDER_SIZE, camera_id=-1)
    return frame[CROP_START:CROP_END, CROP_START:CROP_END, :]


def render_and_embed_r3m(
    env_name:    str = 'kitchen-mixed-v0',
    model=None,
    transform=None,
    device:      str = 'cuda',
    batch_size:  int = 256,
    cache_path:  str = None,
) -> tuple:
    """
    EXTRACT generate_kitchen_data.py 방식:
    에피소드별 reset() → action replay → render_frame() → R3M(2048-dim).

    렌더링: 512x512 → 중앙 128x128 crop → R3M resnet50 → 2048-dim
    cache_path: 첫 실행에서 저장, 이후 즉시 로드.
    Returns: obs(N,60), actions(N,9), terminals(N,), embeddings(N,2048)
    """
    import os
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Set MUJOCO_GL=egl")

    import d4rl, gym
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"Loading {env_name} ...")
    env       = gym.make(env_name)
    dataset   = env.get_dataset()
    obs       = dataset['observations'].copy()
    actions   = dataset['actions'].copy()
    terminals = dataset['terminals'].astype(bool)
    N         = len(obs)

    # ── 캐시 확인 ─────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached R3M embeddings: {cache_path}")
        embeddings = np.load(cache_path)['embeddings']
        assert len(embeddings) == N, f"Cache mismatch: {len(embeddings)} vs {N}"
        norm = np.linalg.norm(embeddings, axis=1).mean()
        print(f"  Loaded: {embeddings.shape}  mean_norm={norm:.2f}")
        if norm < 0.1:
            print("  WARNING: mean_norm~0 → invalid cache! Delete and re-run.")
        return obs, actions, terminals, embeddings

    # ── R3M 로드 ──────────────────────────────────────────────
    if model is None:
        print("Loading R3M (resnet50) ...")
        model, transform = load_r3m(device)

    sim = env.unwrapped.sim
    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    print(f"Episodes: {len(ep_starts)}  Steps: {N}")
    print(f"Render: {RENDER_SIZE}px → crop[{CROP_START}:{CROP_END}]={CROP_SIZE}px → R3M")

    embeddings   = np.zeros((N, 2048), dtype=np.float32)
    frames_batch = []
    idx_batch    = []

    def flush():
        if not frames_batch:
            return
        imgs = torch.stack(frames_batch).to(dev)
        with torch.no_grad():
            emb = model(imgs * 255.0)
        for i, e in zip(idx_batch, emb.cpu().numpy()):
            embeddings[i] = e
        frames_batch.clear()
        idx_batch.clear()

    # ── Action replay ─────────────────────────────────────────
    for ep_i, (ep_s, ep_e) in enumerate(zip(ep_starts, ep_ends)):
        env.reset()
        for t in range(ep_s, ep_e + 1):
            frame = render_frame(sim)
            frames_batch.append(transform(Image.fromarray(frame)))
            idx_batch.append(t)
            if len(frames_batch) >= batch_size:
                flush()
            if t < ep_e:
                env.step(actions[t])

        if (ep_i + 1) % 50 == 0:
            print(f"  ep {ep_i+1}/{len(ep_starts)}  "
                  f"t={ep_e+1}/{N} ({(ep_e+1)/N*100:.0f}%)", flush=True)

    flush()
    norm = np.linalg.norm(embeddings, axis=1).mean()
    print(f"R3M done: {embeddings.shape}  mean_norm={norm:.2f}")
    if norm < 0.1:
        print("WARNING: mean_norm~0 — rendering may have failed!")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"Saved: {cache_path}")

    return obs, actions, terminals, embeddings



def compute_r3m_diff(embeddings: np.ndarray,
                     terminals: np.ndarray = None) -> np.ndarray:
    """
    EXTRACT 논문 수식 (1): e_t = VLM(s_t) - VLM(s_1)
    각 timestep에서 해당 episode의 첫 frame embedding을 뺀다.

    → 초기 로봇/환경 위치의 영향 제거
    → 'episode 시작 이후 얼마나 변했는가'를 표현
    → 같은 행동이라면 초기 위치 무관하게 비슷한 embedding

    terminals: (N,) bool. None이면 전체를 단일 episode로 처리.
    Returns (N, 2048)
    """
    N = len(embeddings)
    out = np.zeros_like(embeddings)

    if terminals is None:
        # 단일 episode
        out = embeddings - embeddings[0:1]
    else:
        ep_ends   = list(np.where(terminals)[0])
        ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
        for ep_s, ep_e in zip(ep_starts, ep_ends):
            e1 = embeddings[ep_s]              # episode 첫 frame
            out[ep_s:ep_e+1] = embeddings[ep_s:ep_e+1] - e1

    print(f"R3M diff (e_t - e_1): {out.shape}  "
          f"mean_norm={np.linalg.norm(out, axis=1).mean():.4f}")
    return out


# Kitchen D4RL observation layout (60-dim)
# [0:9]   robot qpos    — joint positions
# [9:18]  robot qvel    — joint velocities  (noisy, skip)
# [18:60] object states — microwave, kettle, burner, light, cabinet...
#                         각 subtask마다 독립적으로 변함 → skill 구분 핵심
KITCHEN_OBJ_SLICE = slice(18, 60)   # 42-dim object states only


def compute_state_diff(obs: np.ndarray,
                       terminals: np.ndarray = None,
                       use_object_only: bool = True) -> np.ndarray:
    """
    EXTRACT 논문 방식 적용: s_t - s_1 (episode 첫 state 차분).

    연속 차분(s_t - s_{t-1}) 대신 episode 시작 기준 차분:
    → 초기 로봇/물체 위치의 영향 제거
    → 특정 subtask 수행 중 object state 변화량만 포착

    use_object_only=True: object states (dim 18~59, 42-dim) 만 사용.
    terminals: None이면 전체를 단일 episode로 처리.
    Returns (N, 42) if use_object_only else (N, 60)
    """
    if use_object_only:
        obs = obs[:, KITCHEN_OBJ_SLICE]   # (N, 42)

    N   = len(obs)
    out = np.zeros_like(obs, dtype=np.float64)

    if terminals is None:
        out = obs - obs[0:1]
    else:
        ep_ends   = list(np.where(terminals)[0])
        ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
        for ep_s, ep_e in zip(ep_starts, ep_ends):
            s1 = obs[ep_s]                 # episode 첫 state
            out[ep_s:ep_e+1] = obs[ep_s:ep_e+1] - s1

    print(f"State diff (s_t - s_1): {out.shape}  "
          f"mean_norm={np.linalg.norm(out, axis=1).mean():.4f}  "
          f"std={out.std(axis=0).mean():.4f}")
    return out.astype(np.float32)


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

    if cfg.use_r3m:
        print("Using R3M + object state diff (concat) ...")
        model, transform = load_r3m(cfg.r3m_device)
        cache = str(Path(out_h5).parent / 'r3m_embeddings.npz')
        obs, actions, terminals, embeddings = render_and_embed_r3m(
            env_name, model, transform,
            cfg.r3m_device,
            cache_path=cache)
        r3m_diff   = compute_r3m_diff(embeddings,
                         terminals=terminals)               # (N, 2048)
        state_diff = compute_state_diff(obs,
                         terminals=terminals,
                         use_object_only=True)              # (N, 42)

        # StandardScaler로 각각 정규화 후 concat
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        r3m_scaled   = StandardScaler().fit_transform(r3m_diff)
        state_scaled = StandardScaler().fit_transform(state_diff)
        concat = np.concatenate([r3m_scaled, state_scaled], axis=1)  # (N, 2090)
        print(f"Concat diff: r3m(2048) + state(42) = {concat.shape[1]}-dim")

        # PCA 차원 축소: 차원의 저주 방지
        if cfg.pca_dim > 0 and cfg.pca_dim < concat.shape[1]:
            print(f"PCA {concat.shape[1]}d → {cfg.pca_dim}d ...")
            pca   = PCA(n_components=cfg.pca_dim, random_state=42)
            diff  = pca.fit_transform(concat)
            cumvar = pca.explained_variance_ratio_.sum()
            print(f"  cumvar={cumvar*100:.1f}%  diff={diff.shape}")
        else:
            diff = concat
    else:
        print(f"Computing Δs_t "
              f"({'object states [18:60]' if cfg.use_object_only else 'full 60-dim'}) ...")
        diff = compute_state_diff(obs, terminals=terminals,
                              use_object_only=cfg.use_object_only)

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
    p.add_argument('--pca_dim',  type=int, default=64,
                   help='PCA dim before K-means (0=off)')
    p.add_argument('--r3m',      action='store_true',
                   help='Use R3M image embedding (EXTRACT original)')
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
        K=args.K, median_window=args.window,
        use_r3m=args.r3m, pca_dim=args.pca_dim,
        device=args.device)

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