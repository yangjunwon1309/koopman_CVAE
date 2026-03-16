"""
Dataset utilities for Koopman CVAE.
Supports: D4RL (Adroit, MuJoCo), synthetic, and custom numpy arrays.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────
#  Episode splitting
# ─────────────────────────────────────────────

def split_into_trajectories(
    dataset: Dict[str, np.ndarray],
    min_len: int = 50,
) -> List[Dict[str, np.ndarray]]:
    """
    Split flat D4RL dataset into individual episodes.
    Episode boundary is marked by terminals=True or timeouts=True.

    Args:
        dataset: dict from d4rl.qlearning_dataset()
                 keys: observations, actions, rewards, terminals
                 optional: timeouts, next_observations
        min_len:  discard episodes shorter than this

    Returns:
        List of dicts, each with keys: observations, actions, rewards
    """
    obs   = dataset['observations']      # (N, obs_dim)
    acts  = dataset['actions']           # (N, act_dim)
    rews  = dataset['rewards']           # (N,)
    terms = dataset['terminals'].astype(bool)  # (N,)

    # Some D4RL datasets also have 'timeouts' (truncation, not terminal)
    timeouts = dataset.get('timeouts', np.zeros_like(terms, dtype=bool))
    ends = terms | timeouts

    trajectories = []
    start = 0
    for i in range(len(ends)):
        if ends[i] or i == len(ends) - 1:
            end = i + 1
            length = end - start
            if length >= min_len:
                trajectories.append({
                    'observations': obs[start:end],   # (T, obs_dim)
                    'actions':      acts[start:end],  # (T, act_dim)
                    'rewards':      rews[start:end],  # (T,)
                })
            start = end

    print(f"  Extracted {len(trajectories)} episodes "
          f"(min_len={min_len}, total steps={len(obs)})")
    return trajectories


# ─────────────────────────────────────────────
#  Sliding window segmentation
# ─────────────────────────────────────────────

def segment_trajectories(
    trajectories: List[Dict[str, np.ndarray]],
    seq_len: int = 100,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut trajectories into fixed-length segments with sliding window.

    Args:
        trajectories: from split_into_trajectories()
        seq_len:      segment length in timesteps
        stride:       sliding step (default: seq_len // 2 for 50% overlap)

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
        raise ValueError(f"No segments found. Try reducing seq_len (current={seq_len}).")

    actions_arr = np.stack(actions_list)  # (B, T, da)
    states_arr  = np.stack(states_list)   # (B, T, ds)
    print(f"  Segments: {actions_arr.shape[0]} x {seq_len} steps  "
          f"(action_dim={actions_arr.shape[-1]}, state_dim={states_arr.shape[-1]})")
    return actions_arr, states_arr


# ─────────────────────────────────────────────
#  D4RL loader
# ─────────────────────────────────────────────

def load_d4rl_trajectories(
    env_name: str,
    seq_len: int = 100,
    stride: Optional[int] = None,
    min_episode_len: int = 50,
    quality: str = 'expert',       # 'expert', 'medium', 'random', 'human'
) -> TensorDataset:
    """
    Load D4RL offline dataset and return TensorDataset.

    Args:
        env_name:  D4RL gym env name, e.g. 'pen-expert-v1'
                   OR our shorthand, e.g. 'adroit_pen'
        seq_len:   sequence length per sample
        stride:    sliding window stride (default: seq_len//2)
        quality:   only used if env_name is our shorthand

    Returns:
        TensorDataset of (actions (B,T,da), states (B,T,ds))

    Example:
        dataset = load_d4rl_trajectories('pen-expert-v1', seq_len=100)
        dataset = load_d4rl_trajectories('adroit_pen', quality='medium')
    """
    try:
        import d4rl
        import gym
    except ImportError:
        raise ImportError(
            "d4rl not installed. Install with:\n"
            "  pip install d4rl\n"
            "  (requires MuJoCo license and mujoco-py)"
        )

    # Resolve shorthand env names
    from envs.env_configs import D4RL_ENV_MAP
    if env_name in D4RL_ENV_MAP:
        quality_idx = {'expert': 0, 'medium': 1, 'random': 2}
        env_name = D4RL_ENV_MAP[env_name][quality_idx.get(quality, 0)]

    print(f"Loading D4RL dataset: {env_name}")
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    # Dataset structure:
    # {
    #   'observations':      (N, obs_dim)   -- s_t
    #   'actions':           (N, act_dim)   -- a_t
    #   'rewards':           (N,)           -- r_t
    #   'next_observations': (N, obs_dim)   -- s_{t+1}
    #   'terminals':         (N,)           -- episode end (bool)
    #   'timeouts':          (N,)           -- truncation (bool, optional)
    # }
    print(f"  Raw dataset: {len(dataset['observations'])} steps, "
          f"obs_dim={dataset['observations'].shape[1]}, "
          f"act_dim={dataset['actions'].shape[1]}")

    trajs = split_into_trajectories(dataset, min_len=min_episode_len)
    actions_arr, states_arr = segment_trajectories(trajs, seq_len=seq_len, stride=stride)

    actions = torch.tensor(actions_arr, dtype=torch.float32)
    states  = torch.tensor(states_arr,  dtype=torch.float32)
    return TensorDataset(actions, states)


# ─────────────────────────────────────────────
#  Custom numpy array loader
# ─────────────────────────────────────────────

def load_from_numpy(
    actions: np.ndarray,   # (N_total, action_dim) or (B, T, action_dim)
    states: np.ndarray,    # (N_total, state_dim)  or (B, T, state_dim)
    seq_len: int = 100,
    stride: Optional[int] = None,
    terminals: Optional[np.ndarray] = None,
) -> TensorDataset:
    """
    Load from raw numpy arrays (e.g. Isaac Gym offline data).

    If input is already (B, T, dim), returns directly.
    If input is (N_total, dim) flat, segments into (B, seq_len, dim).
    """
    if actions.ndim == 3:
        # Already segmented
        a = torch.tensor(actions, dtype=torch.float32)
        s = torch.tensor(states,  dtype=torch.float32)
        return TensorDataset(a, s)

    # Flat array: need to segment
    if terminals is None:
        # No episode boundaries: treat as one long trajectory
        terminals = np.zeros(len(actions), dtype=bool)
        terminals[-1] = True

    dataset = {
        'observations': states,
        'actions':      actions,
        'rewards':      np.zeros(len(actions)),
        'terminals':    terminals,
    }
    trajs = split_into_trajectories(dataset, min_len=seq_len)
    actions_arr, states_arr = segment_trajectories(trajs, seq_len=seq_len, stride=stride)

    a = torch.tensor(actions_arr, dtype=torch.float32)
    s = torch.tensor(states_arr,  dtype=torch.float32)
    return TensorDataset(a, s)


# ─────────────────────────────────────────────
#  Synthetic data (for testing)
# ─────────────────────────────────────────────

def make_synthetic_dataset(
    action_dim: int,
    state_dim: int,
    n_samples: int = 1000,
    seq_len: int = 100,
) -> TensorDataset:
    """
    Generate synthetic sinusoidal trajectories for quick testing.
    Each sample has random frequencies and phases per dimension.
    """
    T = seq_len
    t = torch.linspace(0, 2 * np.pi, T)

    actions = torch.zeros(n_samples, T, action_dim)
    states  = torch.zeros(n_samples, T, state_dim)

    for i in range(n_samples):
        # Action: superposition of random sinusoids
        freqs  = torch.rand(action_dim) * 3.0 + 0.5
        phases = torch.rand(action_dim) * 2 * np.pi
        amps   = torch.rand(action_dim) * 0.5 + 0.5
        for d in range(action_dim):
            actions[i, :, d] = amps[d] * torch.sin(freqs[d] * t + phases[d])

        # State: different frequencies
        freqs_s  = torch.rand(state_dim) * 2.0 + 0.3
        phases_s = torch.rand(state_dim) * 2 * np.pi
        for d in range(state_dim):
            states[i, :, d] = torch.sin(freqs_s[d] * t + phases_s[d])

    print(f"Synthetic dataset: {n_samples} samples x {seq_len} steps  "
          f"(action_dim={action_dim}, state_dim={state_dim})")
    return TensorDataset(actions, states)