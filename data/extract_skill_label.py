"""
extract_skill_label.py
=======================
EXTRACT (CoRL 2024) 방식 skill labeling + KODAQ x_t 입력 구성.

KODAQ §1.1 x_t 구성 (수정):
  x_t = [Δe_t (2048), Δp_t (42), Δq_t (9), q̇_t (9)] ∈ ℝ^{2108}
  Δe_t = R3M(s_t) - R3M(s_1)   episode-first R3M diff
  Δp_t = p^obj_t - p^obj_1     episode-first object state diff (obs[18:60])
  Δq_t = q_t - q_0             episode-first qpos diff (obs[0:9]) ← 변경
  q̇_t  = obs[:, 9:18]          robot joint velocity (그대로)

변경 이유:
  - q_t 절대값은 에피소드마다 초기 구성이 달라 Koopman 학습 어려움
  - Δq_t는 Δe_t, Δp_t와 동일한 episode-first 철학으로 일관성 확보
  - gripper dim [7:9]은 translation (m) 단위 → 나머지 rad와 혼재하지만
    Δ 처리로 스케일이 작아져 상대적으로 안정
  - q̇_t는 그대로 유지 (velocity 정보 보존, alpha_qdot으로 weight 조절)

EXTRACT pipeline:
  1. render_and_embed_r3m()  → R3M embeddings 캐시 (npz)
  2. compute_r3m_diff()      → Δe_t (N, 2048)
  3. compute_state_diff()    → Δp_t (N, 42)
  4. K-means(K=8) + median filter → cluster_data.h5
  5. split_into_skill_segments() → EXTRACT SkillClusterD4RLSequenceSplitDataset 재현

추가:
  build_x_sequence() → 위 컴포넌트를 조합해 x_t ∈ ℝ^{2108} 생성
  캐시: r3m_embeddings.npz (R3M), x_sequences.npz (전체 x_t 시퀀스)

Usage:
    python extract_skill_label.py --out checkpoints/skill_pretrain/cluster_data.h5
    python extract_skill_label.py --out checkpoints/skill_pretrain/cluster_data.h5 --r3m
    python extract_skill_label.py --visualize --out checkpoints/skill_pretrain/cluster_data.h5
"""

import os
import sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Kitchen observation layout (D4RL, 60-dim)
# ──────────────────────────────────────────────────────────────────────────────
KITCHEN_QPOS_SLICE = slice(0,  9)    # robot joint positions (q_t)
KITCHEN_QVEL_SLICE = slice(9,  18)   # robot joint velocities (q̇_t)
KITCHEN_OBJ_SLICE  = slice(18, 60)   # object states Δp_t (42-dim)

# x_t component dims  (must match KoopmanCVAEConfig)
DIM_DELTA_E = 2048
DIM_DELTA_P = 42
DIM_Q       = 9
DIM_QDOT    = 9
X_DIM       = DIM_DELTA_E + DIM_DELTA_P + DIM_Q + DIM_QDOT  # 2108

# Rendering
RENDER_SIZE = 512
CROP_START  = 192
CROP_END    = 320
CROP_SIZE   = CROP_END - CROP_START   # 128


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractClusterConfig:
    K:              int   = 8       # EXTRACT default
    median_window:  int   = 7       # EXTRACT default
    state_dim:      int   = 60
    use_r3m:        bool  = True    # False: state-diff only (faster debug)
    use_object_only: bool = True    # Δp_t: object states only
    r3m_device:     str   = 'cuda'
    img_size:       int   = 128
    pca_dim:        int   = 64     # K-means 전 PCA 차원 축소 (0=off)
    kmeans_n_init:  int   = 20
    kmeans_seed:    int   = 42
    min_seg_len:    int   = 5
    device:         str   = 'cuda'
    env_name:       str   = 'kitchen-mixed-v0'


# ──────────────────────────────────────────────────────────────────────────────
# D4RL flat dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_d4rl_flat(env_name: str = 'kitchen-mixed-v0') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    D4RL flat dataset 로드.
    Returns: obs (N,60), actions (N,9), terminals (N,) bool
    """
    import d4rl, gym
    env       = gym.make(env_name)
    dataset   = env.get_dataset()
    obs       = dataset['observations']            # (N, 60)
    actions   = dataset['actions']                 # (N, 9)
    terminals = dataset['terminals'].astype(bool)  # (N,)
    print(f"D4RL '{env_name}': {len(obs)} steps  episodes={terminals.sum()}")
    return obs, actions, terminals


# ──────────────────────────────────────────────────────────────────────────────
# R3M image embedding (EXTRACT 원본 방식)
# ──────────────────────────────────────────────────────────────────────────────

def load_r3m(device: str = 'cuda'):
    """R3M resnet50 로드. Returns (model, transform)."""
    try:
        from r3m import load_r3m as _load
    except ImportError:
        raise ImportError(
            "R3M not installed.\n"
            "  pip install git+https://github.com/facebookresearch/r3m.git"
        )
    model = _load("resnet50")
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    return model, transform


def render_frame(sim) -> np.ndarray:
    """512x512 렌더링 → 중앙 128x128 crop."""
    frame = sim.render(RENDER_SIZE, RENDER_SIZE, camera_id=-1)
    return frame[CROP_START:CROP_END, CROP_START:CROP_END, :]


def render_and_embed_r3m(
    env_name:   str = 'kitchen-mixed-v0',
    model       = None,
    transform   = None,
    device:     str = 'cuda',
    batch_size: int = 256,
    cache_path: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    EXTRACT generate_kitchen_data.py 방식:
      env.reset() → action replay → render_frame() → R3M → 2048-dim

    캐시: cache_path (.npz) 존재 시 즉시 로드.
    Returns: obs(N,60), actions(N,9), terminals(N,), embeddings(N,2048)
    """
    import d4rl, gym

    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Set MUJOCO_GL=egl")

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"Loading {env_name} ...")
    env       = gym.make(env_name)
    dataset   = env.get_dataset()
    obs       = dataset['observations'].copy()
    actions   = dataset['actions'].copy()
    terminals = dataset['terminals'].astype(bool)
    N         = len(obs)

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached R3M embeddings: {cache_path}")
        embeddings = np.load(cache_path)['embeddings']
        assert len(embeddings) == N, f"Cache size mismatch: {len(embeddings)} vs {N}"
        norm = np.linalg.norm(embeddings, axis=1).mean()
        print(f"  Loaded: {embeddings.shape}  mean_norm={norm:.2f}")
        if norm < 0.1:
            print("  WARNING: mean_norm~0 → invalid cache! Delete and re-run.")
        return obs, actions, terminals, embeddings

    # ── R3M 로드 ─────────────────────────────────────────────────────────────
    if model is None:
        print("Loading R3M (resnet50) ...")
        model, transform = load_r3m(device)

    sim       = env.unwrapped.sim
    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    print(f"Episodes: {len(ep_starts)}  Steps: {N}")

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

    # ── Action replay ─────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Episode-first difference (EXTRACT §1 수식)
# ──────────────────────────────────────────────────────────────────────────────

def compute_r3m_diff(
    embeddings: np.ndarray,
    terminals:  np.ndarray,
) -> np.ndarray:
    """
    Δe_t = R3M(s_t) - R3M(s_1)   (KODAQ §1.1)
    episode-first difference로 layout-specific bias 제거.
    Returns (N, 2048)
    """
    N   = len(embeddings)
    out = np.zeros_like(embeddings)
    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    for ep_s, ep_e in zip(ep_starts, ep_ends):
        e1 = embeddings[ep_s]
        out[ep_s:ep_e+1] = embeddings[ep_s:ep_e+1] - e1
    norm = np.linalg.norm(out, axis=1).mean()
    print(f"Δe_t (R3M diff): {out.shape}  mean_norm={norm:.4f}")
    return out.astype(np.float32)


def compute_state_diff(
    obs:             np.ndarray,
    terminals:       np.ndarray,
    use_object_only: bool = True,
) -> np.ndarray:
    """
    Δp_t = p^obj_t - p^obj_1   (KODAQ §1.1)
    object state (obs[18:60]) episode-first difference.
    Returns (N, 42) if use_object_only else (N, 60)
    """
    data = obs[:, KITCHEN_OBJ_SLICE] if use_object_only else obs
    out  = np.zeros_like(data, dtype=np.float32)
    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    for ep_s, ep_e in zip(ep_starts, ep_ends):
        s1 = data[ep_s]
        out[ep_s:ep_e+1] = data[ep_s:ep_e+1] - s1
    norm = np.linalg.norm(out, axis=1).mean()
    print(f"Δp_t (obj diff): {out.shape}  mean_norm={norm:.4f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# KODAQ x_t 구성  (§1.1 핵심)
# ──────────────────────────────────────────────────────────────────────────────

def compute_qpos_diff(
    obs:       np.ndarray,   # (N, 60)
    terminals: np.ndarray,   # (N,) bool
) -> np.ndarray:
    """
    Δq_t = q_t - q_0   (episode-first qpos diff)

    Δe_t, Δp_t와 동일한 episode-first 처리:
    - 에피소드마다 초기 관절 구성(q_0)이 다른 bias 제거
    - Koopman이 "초기 대비 변화량"을 선형 예측 → 절대값보다 훨씬 쉬움
    - 스케일: max joint range ~2π rad → Δ는 보통 ±0.5 rad 이내

    obs[0:7] : panda joint angle (rad)
    obs[7:9] : gripper finger translation (m)  ← 단위 다르나 Δ처리로 스케일 완화
    Returns (N, 9) float32
    """
    q   = obs[:, KITCHEN_QPOS_SLICE].astype(np.float32)   # (N, 9)
    out = np.zeros_like(q)
    ep_ends   = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    for ep_s, ep_e in zip(ep_starts, ep_ends):
        q0 = q[ep_s]
        out[ep_s:ep_e+1] = q[ep_s:ep_e+1] - q0
    norm = np.linalg.norm(out, axis=1).mean()
    print(f"Δq_t (qpos diff): {out.shape}  mean_norm={norm:.4f}")
    return out


def build_x_sequence(
    obs:        np.ndarray,    # (N, 60)
    terminals:  np.ndarray,    # (N,) bool
    embeddings: Optional[np.ndarray] = None,  # (N, 2048) or None
) -> np.ndarray:
    """
    x_t = [Δe_t (2048), Δp_t (42), Δq_t (9), q̇_t (9)] ∈ ℝ^{2108}

    Δq_t = q_t - q_0  (episode-first, 절대값 q_t 대신 사용)
    embeddings=None: Δe_t를 0으로 채움 (state-only 실험용)
    Returns: x_seq (N, 2108) float32
    """
    N = len(obs)

    # Δe_t
    if embeddings is not None:
        delta_e = compute_r3m_diff(embeddings, terminals)       # (N, 2048)
    else:
        print("embeddings=None → Δe_t set to zeros (state-only mode)")
        delta_e = np.zeros((N, DIM_DELTA_E), dtype=np.float32)

    # Δp_t: object state episode-first diff
    delta_p = compute_state_diff(obs, terminals, use_object_only=True)  # (N, 42)

    # Δq_t: qpos episode-first diff  ← 변경 (절대값 q_t → Δq_t)
    delta_q = compute_qpos_diff(obs, terminals)                 # (N, 9)

    # q̇_t: joint velocity (그대로)
    qdot_t  = obs[:, KITCHEN_QVEL_SLICE].astype(np.float32)    # (N, 9)

    # Concatenate → (N, 2108)
    x_seq = np.concatenate([delta_e, delta_p, delta_q, qdot_t], axis=1)
    assert x_seq.shape[1] == X_DIM, f"x_dim mismatch: {x_seq.shape[1]} vs {X_DIM}"
    print(f"x_t built: {x_seq.shape}  (Δe={DIM_DELTA_E}, Δp={DIM_DELTA_P}, "
          f"Δq={DIM_Q}, q̇={DIM_QDOT})")
    return x_seq


def cache_x_sequences(
    x_seq:     np.ndarray,    # (N, 2108)
    actions:   np.ndarray,    # (N, 9)
    terminals: np.ndarray,    # (N,)
    cache_path: str,
):
    """x_t, actions, terminals를 npz 캐시로 저장."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_seq=x_seq,
        actions=actions,
        terminals=terminals,
    )
    print(f"Cached x_t sequences: {cache_path}  "
          f"x_seq={x_seq.shape}  actions={actions.shape}")


def load_x_sequences(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns: x_seq (N,2108), actions (N,9), terminals (N,)"""
    data = np.load(cache_path)
    return data['x_seq'], data['actions'], data['terminals']


# ──────────────────────────────────────────────────────────────────────────────
# K-means skill labeling
# ──────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    diff:   np.ndarray,
    K:      int,
    n_init: int = 20,
    seed:   int = 42,
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    """
    EXTRACT: K-means on Δe_t (or state diff).
    Returns (km, labels (N,), logprobs (K, N))
    logprobs = -distance to each centroid (proxy for log-prob)
    """
    print(f"K-means K={K} on {diff.shape} ...")
    km = KMeans(n_clusters=K, init='k-means++',
                n_init=n_init, random_state=seed, max_iter=500)
    km.fit(diff)
    labels   = km.labels_
    dists    = km.transform(diff)         # (N, K)
    logprobs = -dists.T                   # (K, N)
    counts   = np.bincount(labels, minlength=K)
    print(f"  Inertia={km.inertia_:.1f}  counts={counts.tolist()}")
    return km, labels, logprobs


def apply_median_filter_per_episode(
    labels:    np.ndarray,
    terminals: np.ndarray,
    window:    int = 7,
) -> np.ndarray:
    """EXTRACT: per-episode median filter on skill labels."""
    smoothed = labels.copy()
    ends  = list(np.where(terminals)[0])
    start = 0
    for end_idx in ends:
        seg = labels[start:end_idx + 1]
        if len(seg) >= window:
            seg = medfilt(seg.astype(float), kernel_size=window).astype(int)
        smoothed[start:end_idx + 1] = seg
        start = end_idx + 1
    if start < len(labels):
        seg = labels[start:]
        if len(seg) >= window:
            seg = medfilt(seg.astype(float), kernel_size=window).astype(int)
        smoothed[start:] = seg
    changed = (smoothed != labels).sum()
    print(f"Median filter: {changed}/{len(labels)} changed ({changed/len(labels)*100:.1f}%)")
    return smoothed


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACT SkillClusterD4RLSequenceSplitDataset 재현
# ──────────────────────────────────────────────────────────────────────────────

def split_into_skill_segments(
    obs:                 np.ndarray,   # (N, 60)
    actions:             np.ndarray,   # (N, 9)
    x_seq:               np.ndarray,   # (N, 2108)  KODAQ x_t
    terminals:           np.ndarray,   # (N,) bool
    cluster_assignments: np.ndarray,   # (N,) int  smoothed
    cluster_logprobs:    np.ndarray,   # (K, N)
    min_seg_len:         int = 5,
) -> List[Dict]:
    """
    EXTRACT SkillClusterD4RLSequenceSplitDataset.__init__ 핵심 로직 재현.

    seq_end_idxs = terminals 위치 + skill_label 변화 위치 (합집합)

    각 segment는 skill_label이 일정한 연속 구간.
    KODAQ용: obs, actions, x_seq (2108-dim), skill labels 모두 포함.
    """
    # EXTRACT 코드 그대로
    seq_end_idxs   = np.where(terminals)[0]
    skill_end_idxs = np.where(
        (cluster_assignments[1:] - cluster_assignments[:-1]) != 0
    )[0]
    seq_end_idxs = np.unique(np.concatenate([seq_end_idxs, skill_end_idxs]))

    seqs    = []
    skipped = 0
    start   = 0
    lp_T    = cluster_logprobs.T   # (N, K)

    for end_idx in seq_end_idxs:
        length = end_idx + 1 - start
        if length < min_seg_len:
            skipped += 1
            continue

        skill_progress = (np.linspace(0, 1, length)
                          if length > 1 else np.array([1.0]))
        seqs.append(dict(
            obs            = obs[start:end_idx + 1],              # (L, 60)
            actions        = actions[start:end_idx + 1],          # (L, 9)
            x_seq          = x_seq[start:end_idx + 1],            # (L, 2108)
            skills         = cluster_assignments[start:end_idx + 1],  # (L,)
            skill_logprobs = lp_T[start:end_idx + 1],             # (L, K)
            skill_progress = skill_progress,
            start_t        = start,
            end_t          = end_idx + 1,
            length         = length,
        ))
        start = end_idx + 1

    if seqs:
        lengths   = [s['length'] for s in seqs]
        skill_ids = [int(s['skills'][0]) for s in seqs]
        print(f"Segments: {len(seqs)}  skipped_short={skipped}")
        print(f"  Length: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")
        print(f"  Skill dist: {dict(sorted(Counter(skill_ids).items()))}")
    return seqs


# ──────────────────────────────────────────────────────────────────────────────
# h5 I/O  (EXTRACT cluster_data.h5 형식)
# ──────────────────────────────────────────────────────────────────────────────

def save_cluster_data(
    path:                str,
    cluster_assignments: np.ndarray,   # (N,)
    cluster_logprobs:    np.ndarray,   # (K, N)
):
    """EXTRACT cluster_data.h5 형식으로 저장."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('clusters', data=cluster_assignments)
        f.create_dataset('logprobs', data=cluster_logprobs)
    print(f"Saved: {path}  clusters={cluster_assignments.shape}  logprobs={cluster_logprobs.shape}")


def load_cluster_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """EXTRACT h5 포맷 로드. Returns: assignments (N,), logprobs (K, N)"""
    with h5py.File(path, 'r') as f:
        assignments = f['clusters'][()]
        logprobs    = f['logprobs'][()]
        if logprobs.ndim >= 2:
            logprobs = logprobs.transpose(1, 0)   # (K, N) — EXTRACT 컨벤션
        else:
            K = int(assignments.max()) + 1
            logprobs = np.zeros((K, len(assignments)))
    return assignments, logprobs


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_extract_pipeline(
    cfg:     ExtractClusterConfig,
    out_h5:  str,
    x_cache: str = None,   # x_t 시퀀스 캐시 경로 (None → out_h5 옆에 저장)
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, KMeans]:
    """
    EXTRACT + KODAQ x_t 전체 파이프라인.

    1. D4RL 로드 (or 캐시)
    2. R3M 렌더링 + 임베딩 (or 캐시)
    3. x_t = [Δe_t, Δp_t, q_t, q̇_t] 구성
    4. K-means on Δe_t (or Δp_t)
    5. median filter → cluster_data.h5 저장
    6. skill segment 분리

    Returns: smoothed_labels, logprobs, segments, diff_for_kmeans, kmeans_model
    """
    out_dir   = Path(out_h5).parent
    r3m_cache = str(out_dir / 'r3m_embeddings.npz')
    if x_cache is None:
        x_cache = str(out_dir / 'x_sequences.npz')

    # ── x_t 캐시 확인 (전체 파이프라인 재사용) ───────────────────────────────
    if Path(x_cache).exists() and Path(out_h5).exists():
        print(f"Loading cached x_t: {x_cache}")
        x_seq, actions, terminals = load_x_sequences(x_cache)
        obs, _, _ = load_d4rl_flat(cfg.env_name)
        # R3M diff is already baked into x_seq[:, :2048]
        embeddings = None
    else:
        # ── R3M 렌더링 ────────────────────────────────────────────────────────
        if cfg.use_r3m:
            model, transform = load_r3m(cfg.r3m_device)
            obs, actions, terminals, embeddings = render_and_embed_r3m(
                cfg.env_name, model, transform, cfg.r3m_device,
                cache_path=r3m_cache,
            )
        else:
            obs, actions, terminals = load_d4rl_flat(cfg.env_name)
            embeddings = None

        # ── x_t 구성 ──────────────────────────────────────────────────────────
        x_seq = build_x_sequence(obs, terminals, embeddings)
        cache_x_sequences(x_seq, actions, terminals, x_cache)

    # ── K-means embedding 선택 ────────────────────────────────────────────────
    if cfg.use_r3m and embeddings is not None:
        # EXTRACT 원본: K-means on Δe_t (R3M diff)
        r3m_diff = compute_r3m_diff(embeddings, terminals)
        diff_km  = r3m_diff
    else:
        # Fallback: K-means on Δp_t (object state diff)
        diff_km = compute_state_diff(obs, terminals, use_object_only=True)

    # ── PCA 차원 축소 ─────────────────────────────────────────────────────────
    if cfg.pca_dim > 0 and cfg.pca_dim < diff_km.shape[1]:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        diff_km_scaled = StandardScaler().fit_transform(diff_km)
        pca     = PCA(n_components=cfg.pca_dim, random_state=42)
        diff_km = pca.fit_transform(diff_km_scaled)
        cumvar  = pca.explained_variance_ratio_.sum()
        print(f"PCA: {diff_km.shape[1]}-dim  cumvar={cumvar*100:.1f}%")

    # ── K-means + median filter ───────────────────────────────────────────────
    km, raw_labels, logprobs = run_kmeans(diff_km, cfg.K, cfg.kmeans_n_init, cfg.kmeans_seed)
    smoothed = apply_median_filter_per_episode(raw_labels, terminals, cfg.median_window)
    save_cluster_data(out_h5, smoothed, logprobs)

    # ── Skill segments (EXTRACT SkillClusterD4RLSequenceSplitDataset) ─────────
    segs = split_into_skill_segments(
        obs=obs, actions=actions, x_seq=x_seq,
        terminals=terminals,
        cluster_assignments=smoothed,
        cluster_logprobs=logprobs,
        min_seg_len=cfg.min_seg_len,
    )

    return smoothed, logprobs, segs, diff_km, km


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_episodes(
    assignments: np.ndarray,
    terminals:   np.ndarray,
    K:           int,
    out_path:    str,
    n_ep:        int = 8,
):
    """Per-episode skill label timeline."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
           '#8E24AA','#00ACC1','#FFB300','#6D4C41','#546E7A','#D81B60']

    ends   = list(np.where(terminals)[0])
    starts = [0] + [e + 1 for e in ends[:-1]]
    eps    = list(zip(starts, ends))[:n_ep]

    fig, axes = plt.subplots(len(eps), 1, figsize=(14, 1.5 * len(eps)))
    if len(eps) == 1: axes = [axes]

    for ax, (s, e) in zip(axes, eps):
        lbl = assignments[s:e+1]
        for t in range(len(lbl)):
            ax.axvspan(t, t+1, color=PAL[lbl[t] % len(PAL)], alpha=0.85, linewidth=0)
        ax.set_xlim(0, len(lbl))
        ax.set_yticks([])
        ax.set_ylabel(f'Ep {eps.index((s,e))}', rotation=0, labelpad=40, fontsize=8)

    handles = [Patch(color=PAL[k % len(PAL)], label=f'Skill {k}') for k in range(K)]
    fig.legend(handles=handles, loc='upper right', ncol=min(K, 4), fontsize=8)
    fig.suptitle(f'EXTRACT Skill Labels  K={K}', fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def visualize_pca_clusters(
    diff:        np.ndarray,
    assignments: np.ndarray,
    km:          KMeans,
    K:           int,
    out_path:    str,
    subsample:   int = 3000,
):
    """PCA 2D cluster scatter."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00',
           '#8E24AA','#00ACC1','#FFB300','#6D4C41','#546E7A','#D81B60']

    lo   = np.percentile(diff, 1,  axis=0)
    hi   = np.percentile(diff, 99, axis=0)
    diff_c = np.clip(diff, lo, hi)
    scaler = StandardScaler()
    diff_s = scaler.fit_transform(diff_c)

    pca  = PCA(n_components=2, random_state=42)
    Z2   = pca.fit_transform(diff_s)
    var  = pca.explained_variance_ratio_

    cent_c = np.clip(km.cluster_centers_, lo, hi)
    C2     = pca.transform(scaler.transform(cent_c))

    if len(Z2) > subsample:
        idx = np.random.choice(len(Z2), subsample, replace=False)
        Z2s, Ls = Z2[idx], assignments[idx]
    else:
        Z2s, Ls = Z2, assignments

    fig, ax = plt.subplots(figsize=(8, 7))
    for k in range(K):
        mask = Ls == k
        if mask.sum() == 0: continue
        ax.scatter(Z2s[mask,0], Z2s[mask,1], color=PAL[k%len(PAL)],
                   alpha=0.45, s=22, edgecolors='white', linewidths=0.3)

    for k in range(K):
        if (assignments==k).sum() == 0: continue
        ax.scatter(C2[k,0], C2[k,1], color=PAL[k%len(PAL)], s=160,
                   marker='o', edgecolors='white', linewidths=1.5, zorder=4)
        ax.text(C2[k,0], C2[k,1] + (Z2[:,1].max()-Z2[:,1].min())*0.04,
                f'Skill {k}', color=PAL[k%len(PAL)], fontsize=9, fontweight='bold',
                ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.set_title(f'Skill Clusters K={K}  N={len(diff):,}', fontweight='bold')
    handles = [Patch(color=PAL[k%len(PAL)], label=f'Skill {k}')
               for k in range(K) if (assignments==k).sum()>0]
    ax.legend(handles=handles, fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA viz: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out',       default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--K',         type=int, default=8)
    p.add_argument('--pca_dim',   type=int, default=64)
    p.add_argument('--r3m',       action='store_true')
    p.add_argument('--window',    type=int, default=7)
    p.add_argument('--env',       default='kitchen-mixed-v0')
    p.add_argument('--device',    default='cuda')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--viz',       default='checkpoints/skill_pretrain/cluster_viz.png')
    args = p.parse_args()

    if args.visualize:
        import d4rl, gym
        env     = gym.make(args.env)
        dataset = env.get_dataset()
        term    = dataset['terminals'].astype(bool)
        asgn, _ = load_cluster_data(args.out)
        K_viz   = int(asgn.max()) + 1
        visualize_episodes(asgn, term, K_viz, args.viz)
        print("Viz done.")
        return

    cfg = ExtractClusterConfig(
        K=args.K, median_window=args.window,
        use_r3m=args.r3m, pca_dim=args.pca_dim,
        device=args.device, env_name=args.env,
    )
    smoothed, logprobs, segs, diff, km = run_extract_pipeline(cfg, args.out)

    # Visualization
    try:
        import d4rl, gym
        term = gym.make(args.env).get_dataset()['terminals'].astype(bool)
        visualize_episodes(smoothed, term, cfg.K, args.viz)
        pca_path = args.viz.replace('.png', '_pca.png')
        visualize_pca_clusters(diff, smoothed, km, cfg.K, pca_path)
    except Exception as ex:
        print(f"Viz failed: {ex}")

    print(f"\nDone. → {args.out}")


if __name__ == '__main__':
    main()