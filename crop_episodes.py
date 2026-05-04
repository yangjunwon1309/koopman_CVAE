"""
preprocess_crop_episodes.py
============================
각 episode에서 2번째 reward 발생 시점까지만 잘라내서 새 dataset으로 저장.

배경:
  - Kitchen-mixed 전체 episode는 4개 subtask를 순서대로 수행
  - 첫 번째 reward (t~50): 전자레인지 열기 → 거의 모든 episode에서 동일
  - 두 번째 reward (t~100~): episode마다 다른 subtask
  - 1번째 reward까지만 보면 너무 짧고 단조로움
  - 2번째 reward 발생 시점까지 crop하면:
      * 첫 번째 subtask (전자레인지) 달성 포함
      * 두 번째 subtask 달성 포함
      * 이후 다양한 모션은 제외 → 학습 안정성 증가

출력:
  checkpoints/skill_pretrain/x_sequences_crop2.npz
    x_seq      : (N_total, 2108) float32
    actions    : (N_total, 9)    float32
    rewards    : (N_total,)      float32
    terminals  : (N_total,)      bool
    skill_labels: (N_total,)     int32
    ep_starts  : (N_ep,)         int32   각 episode 시작 global index
    ep_ends    : (N_ep,)         int32   각 episode 끝 global index

Usage:
    python preprocess_crop_episodes.py \\
        --x_cache checkpoints/skill_pretrain/x_sequences.npz \\
        --out     checkpoints/skill_pretrain/x_sequences_crop2.npz \\
        --n_rewards 2 \\
        --min_len   32 \\
        --verbose
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
from pathlib import Path
from data.extract_skill_label import load_x_sequences, load_cluster_data


def find_reward_timesteps(rewards_ep: np.ndarray) -> list:
    """
    Episode reward array에서 reward 발생 timestep 목록 반환.
    reward가 누적합이면 diff, sparse면 그대로.
    """
    r = rewards_ep.astype(np.float32)
    # diff가 양수인 시점 = reward 발생
    diff = np.zeros_like(r)
    diff[0] = r[0]
    diff[1:] = np.diff(r)
    reward_ts = list(np.where(diff > 0.5)[0])
    return reward_ts


def crop_episodes(
    x_seq:       np.ndarray,   # (N, 2108)
    actions:     np.ndarray,   # (N, 9)
    rewards:     np.ndarray,   # (N,)
    terminals:   np.ndarray,   # (N,) bool
    skill_labels: np.ndarray,  # (N,)
    n_rewards:   int = 2,
    min_len:     int = 32,
    verbose:     bool = True,
) -> dict:
    """
    각 episode에서 n_rewards 번째 reward 발생 직후까지 crop.
    """
    # episode 경계 추출
    term_idx   = np.where(terminals)[0]
    ep_starts  = np.concatenate([[0], term_idx[:-1] + 1])
    ep_ends    = term_idx   # inclusive

    x_list, a_list, r_list, t_list, s_list = [], [], [], [], []
    ep_start_out, ep_end_out = [], []
    total_out = 0

    n_total    = len(ep_starts)
    n_kept     = 0
    n_skipped  = 0
    crop_lens  = []

    for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        L       = e - s + 1
        r_ep    = rewards[s:e+1]
        x_ep    = x_seq[s:e+1]
        a_ep    = actions[s:e+1]
        sk_ep   = skill_labels[s:e+1]

        reward_ts = find_reward_timesteps(r_ep)

        if len(reward_ts) < n_rewards:
            # n_rewards번째 reward가 없는 episode → 건너뜀
            n_skipped += 1
            if verbose and i < 10:
                print(f"  Skip ep {i}: reward_ts={reward_ts} < {n_rewards}")
            continue

        # n_rewards번째 reward 시점까지 crop (해당 시점 포함)
        crop_end = reward_ts[n_rewards - 1] + 1   # +1: inclusive → exclusive
        crop_end = min(crop_end, L)

        if crop_end < min_len:
            n_skipped += 1
            continue

        # terminal: 마지막 step만 True, 나머지 False
        t_ep = np.zeros(crop_end, dtype=bool)
        t_ep[-1] = True

        x_list.append(x_ep[:crop_end])
        a_list.append(a_ep[:crop_end])
        r_list.append(r_ep[:crop_end])
        t_list.append(t_ep)
        s_list.append(sk_ep[:crop_end])

        ep_start_out.append(total_out)
        total_out += crop_end
        ep_end_out.append(total_out - 1)

        crop_lens.append(crop_end)
        n_kept += 1

    if verbose:
        print(f"\nCrop summary:")
        print(f"  Total episodes:   {n_total}")
        print(f"  Kept (≥{n_rewards} rewards): {n_kept}")
        print(f"  Skipped:          {n_skipped}")
        print(f"  Crop len: mean={np.mean(crop_lens):.1f}  "
              f"min={np.min(crop_lens)}  max={np.max(crop_lens)}  "
              f"std={np.std(crop_lens):.1f}")
        print(f"  Total timesteps:  {total_out}")

    x_out  = np.concatenate(x_list,  axis=0).astype(np.float32)
    a_out  = np.concatenate(a_list,  axis=0).astype(np.float32)
    r_out  = np.concatenate(r_list,  axis=0).astype(np.float32)
    t_out  = np.concatenate(t_list,  axis=0).astype(bool)
    s_out  = np.concatenate(s_list,  axis=0).astype(np.int32)

    return {
        'x_seq':       x_out,
        'actions':     a_out,
        'rewards':     r_out,
        'terminals':   t_out,
        'skill_labels': s_out,
        'ep_starts':   np.array(ep_start_out, dtype=np.int32),
        'ep_ends':     np.array(ep_end_out,   dtype=np.int32),
        'n_episodes':  n_kept,
        'crop_lens':   np.array(crop_lens,    dtype=np.int32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--x_cache',   default='checkpoints/skill_pretrain/x_sequences.npz')
    p.add_argument('--skill_h5',  default='checkpoints/skill_pretrain/cluster_data.h5')
    p.add_argument('--out',       default='checkpoints/skill_pretrain/x_sequences_crop2.npz')
    p.add_argument('--n_rewards', type=int, default=2,
                   help='Crop to N-th reward occurrence (default=2)')
    p.add_argument('--min_len',   type=int, default=32,
                   help='Minimum episode length after crop')
    p.add_argument('--verbose',   action='store_true')
    args = p.parse_args()

    print(f"Loading x_sequences: {args.x_cache}")
    data = np.load(args.x_cache)
    x_seq     = data['x_seq']       if 'x_seq'     in data else data['x_sequences']
    actions   = data['actions']
    rewards   = data['rewards']
    terminals = data['terminals'].astype(bool)

    print(f"  x_seq={x_seq.shape}  actions={actions.shape}  "
          f"rewards={rewards.shape}  terminals={terminals.sum()} episodes")

    # reward rate
    r_diff = np.zeros_like(rewards)
    r_diff[1:] = np.diff(rewards)
    r_diff = r_diff.clip(0, 1)
    print(f"  reward events: {int((r_diff > 0.5).sum())}")

    # skill labels
    print(f"Loading skill labels: {args.skill_h5}")
    try:
        skill_labels, _ = load_cluster_data(args.skill_h5)
        print(f"  skill_labels={skill_labels.shape}")
    except Exception as e:
        print(f"  skill_labels load failed ({e}), using zeros")
        skill_labels = np.zeros(len(x_seq), dtype=np.int32)

    # Crop
    print(f"\nCropping to {args.n_rewards} rewards ...")
    result = crop_episodes(
        x_seq, actions, r_diff, terminals, skill_labels,
        n_rewards=args.n_rewards,
        min_len=args.min_len,
        verbose=True,
    )

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: v for k, v in result.items()})
    print(f"\nSaved → {args.out}")
    print(f"  x_seq:      {result['x_seq'].shape}")
    print(f"  actions:    {result['actions'].shape}")
    print(f"  n_episodes: {result['n_episodes']}")

    # Quick stats
    r = result['rewards']
    print(f"  reward sum:  {r.sum():.0f}  "
          f"nonzero: {int((r > 0).sum())}  "
          f"rate: {(r > 0).mean():.4f}")


if __name__ == '__main__':
    main()