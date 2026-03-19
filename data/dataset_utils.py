"""
Dataset utilities for Koopman CVAE.
Supports: D4RL (Adroit, MuJoCo), synthetic, and custom numpy arrays.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────
# Episode splitting
# ─────────────────────────────────────────────

def split_into_trajectories(
    dataset: Dict[str, np.ndarray],
    min_len: int = 50,
) -> List[Dict[str, np.ndarray]]:
    """
    Split flat D4RL dataset into individual episodes.
    Episode boundary: terminals=True OR timeouts=True.

    Args:
        dataset:  dict from d4rl.qlearning_dataset()
        min_len:  discard episodes shorter than this

    Returns:
        List of dicts with keys: observations, actions, rewards
    """
    obs      = dataset['observations']
    acts     = dataset['actions']
    rews     = dataset['rewards']
    terms    = dataset['terminals'].astype(bool)
    timeouts = dataset.get('timeouts', np.zeros_like(terms, dtype=bool))
    ends     = terms | timeouts

    trajectories = []
    start = 0
    for i in range(len(ends)):
        if ends[i] or i == len(ends) - 1:
            end    = i + 1
            length = end - start
            if length >= min_len:
                trajectories.append({
                    'observations': obs[start:end],
                    'actions':      acts[start:end],
                    'rewards':      rews[start:end],
                })
            start = end

    print(f"  Extracted {len(trajectories)} episodes "
          f"(min_len={min_len}, total steps={len(obs)})")
    return trajectories


# ─────────────────────────────────────────────
# Sliding window segmentation
# ─────────────────────────────────────────────

def segment_trajectories(
    trajectories: List[Dict[str, np.ndarray]],
    seq_len: int = 100,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut trajectories into fixed-length segments with sliding window.

    Returns:
        actions_arr: (B, seq_len, action_dim)
        states_arr:  (B, seq_len, obs_dim)
    """
    if stride is None:
        stride = seq_len // 2

    actions_list, states_list = [], []
    for traj in trajectories:
        T = len(traj['observations'])
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            states_list.append(traj['observations'][start:end])
            actions_list.append(traj['actions'][start:end])

    if len(actions_list) == 0:
        raise ValueError(
            f"No segments found. Try reducing seq_len (current={seq_len})."
        )

    actions_arr = np.stack(actions_list)
    states_arr  = np.stack(states_list)
    print(f"  Segments: {actions_arr.shape[0]} x {seq_len} steps "
          f"(action_dim={actions_arr.shape[-1]}, state_dim={states_arr.shape[-1]})")
    return actions_arr, states_arr


# ─────────────────────────────────────────────
# D4RL loader
# ─────────────────────────────────────────────

def load_d4rl_trajectories(
    env_name: str,
    seq_len: int = 100,
    stride: Optional[int] = None,
    min_episode_len: int = 50,
    quality: str = 'expert',
) -> TensorDataset:
    """
    Load D4RL offline dataset and return TensorDataset.

    Args:
        env_name: our shorthand ('adroit_pen') OR full D4RL name ('pen-human-v1')
        seq_len:  sequence length per sample
        stride:   sliding window stride (default: seq_len // 2)
        quality:  'expert' | 'human' | 'cloned'
                  only used when env_name is our shorthand

    Returns:
        TensorDataset of (actions (B,T,da), states (B,T,ds))

    Examples:
        load_d4rl_trajectories('adroit_pen', quality='human')
        load_d4rl_trajectories('pen-human-v1')
    """
    try:
        import d4rl
        import gym
    except ImportError:
        raise ImportError(
            "d4rl not installed. Install with:\n"
            "  pip install d4rl\n"
            "  (requires MuJoCo and mujoco-py)"
        )

    # ── Resolve shorthand → full D4RL gym name ──────────────
    from envs.envs_config import D4RL_ENV_MAP, get_d4rl_env_name

    if env_name in D4RL_ENV_MAP:
        env_name = get_d4rl_env_name(env_name, quality)

    print(f"Loading D4RL dataset: {env_name}")

    env     = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    print(f"  Raw dataset: {len(dataset['observations'])} steps, "
          f"obs_dim={dataset['observations'].shape[1]}, "
          f"act_dim={dataset['actions'].shape[1]}")

    trajs                  = split_into_trajectories(dataset, min_len=min_episode_len)
    actions_arr, states_arr = segment_trajectories(trajs, seq_len=seq_len, stride=stride)

    actions = torch.tensor(actions_arr, dtype=torch.float32)
    states  = torch.tensor(states_arr,  dtype=torch.float32)
    return TensorDataset(actions, states)


# ─────────────────────────────────────────────
# Custom numpy array loader
# ─────────────────────────────────────────────

def load_from_numpy(
    actions: np.ndarray,
    states: np.ndarray,
    seq_len: int = 100,
    stride: Optional[int] = None,
    terminals: Optional[np.ndarray] = None,
) -> TensorDataset:
    """
    Load from raw numpy arrays (e.g. Isaac Gym offline data).
    Input shape (B, T, dim) → returned directly.
    Input shape (N_total, dim) → segmented into (B, seq_len, dim).
    """
    if actions.ndim == 3:
        return TensorDataset(
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(states,  dtype=torch.float32),
        )

    if terminals is None:
        terminals       = np.zeros(len(actions), dtype=bool)
        terminals[-1]   = True

    dataset = {
        'observations': states,
        'actions':      actions,
        'rewards':      np.zeros(len(actions)),
        'terminals':    terminals,
    }
    trajs                  = split_into_trajectories(dataset, min_len=seq_len)
    actions_arr, states_arr = segment_trajectories(trajs, seq_len=seq_len, stride=stride)

    return TensorDataset(
        torch.tensor(actions_arr, dtype=torch.float32),
        torch.tensor(states_arr,  dtype=torch.float32),
    )


# ─────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────

def make_synthetic_dataset(
    action_dim: int,
    state_dim: int,
    n_samples: int = 1000,
    seq_len: int = 100,
) -> TensorDataset:
    """Sinusoidal synthetic dataset for quick testing."""
    import math
    T = seq_len
    t = torch.linspace(0, 2 * math.pi, T)

    actions = torch.zeros(n_samples, T, action_dim)
    states  = torch.zeros(n_samples, T, state_dim)

    for i in range(n_samples):
        freqs  = torch.rand(action_dim) * 3.0 + 0.5
        phases = torch.rand(action_dim) * 2 * math.pi
        amps   = torch.rand(action_dim) * 0.5 + 0.5
        for d in range(action_dim):
            actions[i, :, d] = amps[d] * torch.sin(freqs[d] * t + phases[d])

        freqs_s  = torch.rand(state_dim) * 2.0 + 0.3
        phases_s = torch.rand(state_dim) * 2 * math.pi
        for d in range(state_dim):
            states[i, :, d] = torch.sin(freqs_s[d] * t + phases_s[d])

    print(f"Synthetic dataset: {n_samples} samples x {seq_len} steps "
          f"(action_dim={action_dim}, state_dim={state_dim})")
    return TensorDataset(actions, states)