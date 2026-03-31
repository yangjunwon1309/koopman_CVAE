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