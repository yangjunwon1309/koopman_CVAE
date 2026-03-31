import h5py
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import argparse

# ─────────────────────────────────────────────────────────────
# Config: HELIOS 최적화 설정
# ─────────────────────────────────────────────────────────────

@dataclass
class ExtractClusterConfig:
    K: int = 8                # 기본 클러스터 개수 (HELIOS는 이후 DPM이 이를 조정) 
    median_window: int = 7     # 노이즈 제거를 위한 필터 크기
    state_dim: int = 60
    action_dim: int = 9
    use_r3m: bool = False
    # HELIOS 핵심: Action 패턴을 클러스터링 피처에 포함
    use_action_delta: bool = True  
    use_object_only: bool = True   
    kmeans_n_init: int = 20
    kmeans_seed: int = 42
    min_seg_len: int = 5      # 너무 짧은 세그먼트 제외 [cite: 204]
    device: str = 'cuda'

# ─────────────────────────────────────────────────────────────
# 피처 추출: HELIOS용 Δs + Δa 결합
# ─────────────────────────────────────────────────────────────

def compute_helios_features(obs: np.ndarray, actions: np.ndarray, cfg: ExtractClusterConfig) -> np.ndarray:
    """
    HELIOS를 위한 클러스터링 피처 생성.
    환경의 변화(Δs)와 로봇의 움직임 패턴(Δa)을 결합하여 
    단순한 상태 변화가 아닌 '동작(Skill)' 위주로 클러스터링 유도. [cite: 140]
    """
    # 1. State Difference (Object states 위주)
    if cfg.use_object_only:
        obj_obs = obs[:, 18:60] # Kitchen의 오브젝트 상태 슬라이스
    else:
        obj_obs = obs
    
    ds = np.diff(obj_obs, axis=0)
    ds = np.vstack([ds[0:1], ds]) # t=0 처리
    
    # 2. Action Difference (동작의 가속도/변화율 패턴)
    if cfg.use_action_delta:
        da = np.diff(actions, axis=0)
        da = np.vstack([da[0:1], da])
        # State diff와 Action diff 결합 (HELIOS Skill Prior 유도용) [cite: 145]
        features = np.hstack([ds, da])
    else:
        features = ds

    # 표준화 (K-means 성능에 필수)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"HELIOS Features computed: {features_scaled.shape}")
    return features_scaled

# ─────────────────────────────────────────────────────────────
# K-means & Median Filter (에피소드별 독립 처리)
# ─────────────────────────────────────────────────────────────

def run_kmeans(diff: np.ndarray, K: int, n_init: int, seed: int):
    print(f"Running K-means (K={K}) on features...")
    km = KMeans(n_clusters=K, init='k-means++', n_init=n_init, random_state=seed)
    labels = km.fit_predict(diff)
    
    # Responsibilities 계산 (Logprobs로 저장)
    dists = km.transform(diff)
    # Numerical stability를 위해 exp(-dists)의 log 형태 취함
    logprobs = -dists # (N, K)
    return km, labels, logprobs.T # (K, N) 형식으로 반환

def apply_median_filter(labels: np.ndarray, terminals: np.ndarray, window: int) -> np.ndarray:
    smoothed = labels.copy()
    ends = np.where(terminals)[0]
    start = 0
    for end_idx in ends:
        seg = labels[start:end_idx + 1]
        if len(seg) >= window:
            seg = medfilt(seg.astype(float), kernel_size=window).astype(int)
        smoothed[start:end_idx + 1] = seg
        start = end_idx + 1
    return smoothed

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

#-------------visualization

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
# .h5 저장 로직 (SkillClusterD4RLSequenceSplitDataset 호환)
# ─────────────────────────────────────────────────────────────

def save_for_helios(path: str, assignments: np.ndarray, logprobs: np.ndarray):
    """
    HELIOS/EXTRACT 데이터로더가 요구하는 정확한 포맷으로 저장.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        # 데이터로더가 "clusters"와 "logprobs" 키를 찾음
        f.create_dataset('clusters', data=assignments.astype(np.int64))
        # 저장 시 (N, K)로 저장하고 로더에서 transpose(1,0) 하여 (K, N)으로 복구
        f.create_dataset('logprobs', data=logprobs.T) 
    print(f"Successfully saved cluster data to {path}")

# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def load_minari_flat(hdf5_path: str):
    """
    Minari의 계층형(에피소드별) 구조를 
    기존 D4RL의 Flat(1차원 연속) 구조로 변환하여 로드합니다.
    (Dict Observation Space 호환 패치 적용)
    """
    obs_list, act_list, term_list = [], [], []
    
    with h5py.File(hdf5_path, 'r') as f:
        # episode_0, episode_1 ... 순서대로 숫자 기준 정렬
        ep_keys = sorted([k for k in f.keys() if k.startswith('episode_')], 
                         key=lambda x: int(x.split('_')[1]))
        
        for ep_key in ep_keys:
            ep = f[ep_key]
            act_len = ep['actions'].shape[0]
            
            # --- 수정된 부분: observations가 Group인지 검사 ---
            obs_data = ep['observations']
            if isinstance(obs_data, h5py.Group):
                # Minari의 Dict Observation (Gymnasium-Robotics 규격)
                # 주로 'observation', 'achieved_goal', 'desired_goal'로 구성됨
                if 'observation' in obs_data:
                    # 기본 상태값(State) 배열 추출
                    obs_array = obs_data['observation'][()]
                else:
                    # 'observation' 키가 명시적으로 없다면 내부 배열을 모두 병합 (Flatten)
                    obs_array = np.hstack([obs_data[k][()] for k in obs_data.keys()])
            else:
                obs_array = obs_data[()]
            # ---------------------------------------------------
            
            obs_list.append(obs_array[:act_len])
            act_list.append(ep['actions'][()])
            term_list.append(ep['terminations'][()])
            
    # 리스트들을 하나의 큰 Numpy 배열로 병합 (N, dim)
    obs = np.vstack(obs_list)
    actions = np.vstack(act_list)
    terminals = np.concatenate(term_list).astype(bool)
    
    return obs, actions, terminals

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='kitchen-mixed-v0', 
                   help='MuJoCo 렌더링을 위해 인스턴스화할 환경 이름')
    p.add_argument('--out', default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--K', type=int, default=8)
    # Minari 데이터셋 경로 인자 추가
    p.add_argument('--data_path', type=str, 
                   default='/home/yangjunwon1309/.minari/datasets/D4RL/kitchen/mixed-v2/data/main_data.hdf5')
    args = p.parse_args()

    cfg = ExtractClusterConfig(K=args.K)

    # 1. Minari HDF5에서 데이터 직접 로드
    print(f"Loading Minari dataset from {args.data_path} ...")
    obs, actions, terminals = load_minari_flat(args.data_path)
    print(f"Dataset Loaded -> obs: {obs.shape}, actions: {actions.shape}, terminals: {terminals.sum()} ends")

    # 2. (옵션) R3M 이미지 임베딩을 원할 경우 env 인스턴스화 필요
    # 시뮬레이터(sim)에 obs를 덮어씌워 렌더링해야 하므로 env 객체는 여전히 필요합니다.
    # env = gym.make(args.env) 
    
    # 3. Feature Engineering (HELIOS 최적화: Δs + Δa)
    features = compute_helios_features(obs, actions, cfg)

    # 4. Clustering (K-means)
    km, labels, logprobs = run_kmeans(features, cfg.K, cfg.kmeans_n_init, cfg.kmeans_seed)

    # 5. Smoothing (Median Filter)
    smoothed_labels = apply_median_filter(labels, terminals, cfg.median_window)

    # 6. Save (HDF5)
    save_for_helios(args.out, smoothed_labels, logprobs)

    print("Pipeline complete.")

if __name__ == '__main__':
    main()