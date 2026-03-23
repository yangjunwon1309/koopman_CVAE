"""
dataset_utils.py
================
Offline dataset loading for KODAC-S.

Supported environments:
  Adroit:          adroit_pen, adroit_hammer, adroit_door, adroit_relocate
  Franka Kitchen:  kitchen_complete, kitchen_partial, kitchen_mixed
  Synthetic:       fallback for quick testing

Franka Kitchen specifics:
  - observation: 60-dim (robot qpos 9 + qvel 9 + object states 42)
  - action:       9-dim joint velocity commands
  - quality tiers: complete (4/4 tasks), partial (2-3/4), mixed (any)
  - episode length: ~280 steps at 12.5Hz (dt=0.08s)
  - multi-task structure: each episode completes a different task subset
    (kettle, microwave, bottom burner, light switch, etc.)

Why Kitchen is more appropriate for KODAC-S:
  - Sequential multi-task structure naturally tests skill segmentation
  - Contact-rich manipulation requires non-trivial Koopman dynamics
  - Human demonstrations (motion capture) provide clean behavior modes
  - Longer episodes give richer temporal context for TCN
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Episode splitting utilities
# ─────────────────────────────────────────────────────────────

def split_into_trajectories(
    dataset: Dict[str, np.ndarray],
    min_len: int = 50,
    use_timeouts: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Split flat D4RL dataset dict into individual episodes.
    Episode boundary: terminals=True OR timeouts=True (if available).
    """
    obs      = dataset['observations']
    acts     = dataset['actions']
    rews     = dataset.get('rewards', np.zeros(len(obs)))
    terms    = dataset['terminals'].astype(bool)
    timeouts = dataset.get('timeouts',
                           np.zeros_like(terms, dtype=bool)) if use_timeouts \
               else np.zeros_like(terms, dtype=bool)
    ends = terms | timeouts

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

    print(f"  Episodes: {len(trajectories)}  "
          f"(min_len={min_len}, total_steps={len(obs)})")
    return trajectories


def split_kitchen_trajectories(
    dataset: Dict[str, np.ndarray],
    min_len: int = 50,
) -> List[Dict[str, np.ndarray]]:
    """
    Franka Kitchen-specific episode splitter.

    Kitchen datasets do not reliably set terminals=True at episode boundaries
    in all versions. Instead, we detect boundaries by checking for large
    discontinuities in the robot qpos (first 9 dims of observation), or
    fall back to fixed-length splits if no discontinuities are found.

    Also extracts task completion info from dataset['infos/tasks_to_complete']
    if available (useful for skill labeling / analysis).
    """
    obs  = dataset['observations']     # (N, 60)
    acts = dataset['actions']          # (N, 9)
    rews = dataset.get('rewards', np.zeros(len(obs)))
    N    = len(obs)

    # 1. Try standard terminal/timeout splitting first
    terms    = dataset['terminals'].astype(bool)
    timeouts = dataset.get('timeouts', np.zeros_like(terms, dtype=bool))
    ends     = terms | timeouts
    n_ends   = ends.sum()

    # 2. If no terminals found, detect via qpos discontinuity
    if n_ends < 2:
        print("  Kitchen: no terminal flags, using qpos discontinuity detection")
        qpos      = obs[:, :9]
        qpos_diff = np.linalg.norm(np.diff(qpos, axis=0), axis=1)  # (N-1,)
        threshold = np.percentile(qpos_diff, 99)   # top 1% jumps = episode boundary
        ends = np.zeros(N, dtype=bool)
        ends[1:] = qpos_diff > threshold
        ends[-1] = True

    # 3. Extract task info if available
    task_info = None
    for key in ['infos/tasks_to_complete', 'infos/goal']:
        if key in dataset:
            task_info = dataset[key]
            break

    trajectories = []
    start = 0
    for i in range(N):
        if ends[i] or i == N - 1:
            end    = i + 1
            length = end - start
            if length >= min_len:
                traj = {
                    'observations': obs[start:end],
                    'actions':      acts[start:end],
                    'rewards':      rews[start:end],
                }
                if task_info is not None:
                    traj['task_info'] = task_info[start:end]
                trajectories.append(traj)
            start = end

    print(f"  Kitchen episodes: {len(trajectories)}  "
          f"(min_len={min_len}, total_steps={N})")
    if trajectories:
        lens = [len(t['observations']) for t in trajectories]
        print(f"  Episode length: min={min(lens)}  "
              f"max={max(lens)}  mean={np.mean(lens):.0f}")
    return trajectories


def segment_trajectories(
    trajectories: List[Dict[str, np.ndarray]],
    seq_len: int = 100,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut trajectories into fixed-length segments via sliding window.
    Returns (actions_arr, states_arr) each shape (B, seq_len, dim).
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

    if not actions_list:
        raise ValueError(
            f"No segments produced. "
            f"Reduce seq_len (current={seq_len}) or min_episode_len."
        )

    actions_arr = np.stack(actions_list).astype(np.float32)
    states_arr  = np.stack(states_list).astype(np.float32)
    print(f"  Segments: {actions_arr.shape[0]} x {seq_len}  "
          f"(action_dim={actions_arr.shape[-1]}, state_dim={states_arr.shape[-1]})")
    return actions_arr, states_arr


# ─────────────────────────────────────────────────────────────
# D4RL environment name maps
# ─────────────────────────────────────────────────────────────

ADROIT_ENV_MAP = {
    'adroit_pen':      {
        'expert': 'pen-expert-v1',
        'human':  'pen-human-v1',
        'cloned': 'pen-cloned-v1',
    },
    'adroit_hammer':   {
        'expert': 'hammer-expert-v1',
        'human':  'hammer-human-v1',
        'cloned': 'hammer-cloned-v1',
    },
    'adroit_door':     {
        'expert': 'door-expert-v1',
        'human':  'door-human-v1',
        'cloned': 'door-cloned-v1',
    },
    'adroit_relocate': {
        'expert': 'relocate-expert-v1',
        'human':  'relocate-human-v1',
        'cloned': 'relocate-cloned-v1',
    },
}

# Franka Kitchen: quality maps to task-completion level
# complete: all 4 tasks completed per episode (hardest, cleanest)
# partial:  2-3 tasks completed
# mixed:    any number of tasks (most diverse, largest dataset)
KITCHEN_ENV_MAP = {
    'kitchen_complete': 'kitchen-complete-v0',
    'kitchen_partial':  'kitchen-partial-v0',
    'kitchen_mixed':    'kitchen-mixed-v0',
}

# Kitchen observation/action dims (fixed in D4RL)
KITCHEN_STATE_DIM  = 60   # robot qpos(9) + qvel(9) + object states(42)
KITCHEN_ACTION_DIM = 9    # joint velocity targets
KITCHEN_DT         = 0.08 # 12.5 Hz control frequency


# ─────────────────────────────────────────────────────────────
# Main loaders
# ─────────────────────────────────────────────────────────────

def load_d4rl_trajectories(
    env_name: str,
    seq_len: int = 100,
    stride: Optional[int] = None,
    min_episode_len: int = 50,
    quality: str = 'human',
) -> TensorDataset:
    """
    Load D4RL offline dataset and return TensorDataset.

    Args:
        env_name:        shorthand key (e.g. 'adroit_pen', 'kitchen_complete')
                         OR full D4RL name (e.g. 'pen-human-v1')
        seq_len:         sequence length per sample
        stride:          sliding window stride (default: seq_len // 2)
        min_episode_len: discard episodes shorter than this
        quality:         for Adroit: 'expert'|'human'|'cloned'
                         for Kitchen: ignored (quality baked into env name)

    Returns:
        TensorDataset of (actions (B,T,da), states (B,T,ds))
    """
    try:
        import d4rl
        import gym
    except ImportError:
        raise ImportError(
            "d4rl not installed.\n"
            "  pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"
        )

    # Resolve env name
    is_kitchen = env_name.startswith('kitchen')
    if env_name in ADROIT_ENV_MAP:
        d4rl_name = ADROIT_ENV_MAP[env_name][quality]
    elif env_name in KITCHEN_ENV_MAP:
        d4rl_name = KITCHEN_ENV_MAP[env_name]
    else:
        d4rl_name = env_name   # assume already a full D4RL name

    print(f"  Loading D4RL env: {d4rl_name}")
    env     = gym.make(d4rl_name)
    dataset = d4rl.qlearning_dataset(env)

    # Print dataset info
    obs_dim = dataset['observations'].shape[-1]
    act_dim = dataset['actions'].shape[-1]
    print(f"  Dataset: {len(dataset['observations'])} steps  "
          f"obs_dim={obs_dim}  act_dim={act_dim}")

    # Split into episodes
    if is_kitchen:
        trajectories = split_kitchen_trajectories(dataset, min_len=min_episode_len)
    else:
        trajectories = split_into_trajectories(dataset, min_len=min_episode_len)

    # Segment into fixed-length windows
    actions_arr, states_arr = segment_trajectories(
        trajectories, seq_len=seq_len, stride=stride
    )

    return TensorDataset(
        torch.FloatTensor(actions_arr),
        torch.FloatTensor(states_arr),
    )


def load_kitchen_all_qualities(
    seq_len: int = 200,
    stride: Optional[int] = None,
    min_episode_len: int = 100,
) -> TensorDataset:
    """
    Load and concatenate all three Kitchen quality tiers.
    Useful for maximum data diversity.
    Returns TensorDataset of (actions, states).
    """
    all_actions, all_states = [], []
    for key in ['kitchen_complete', 'kitchen_partial', 'kitchen_mixed']:
        try:
            ds = load_d4rl_trajectories(
                key, seq_len=seq_len, stride=stride,
                min_episode_len=min_episode_len,
            )
            a, s = ds.tensors
            all_actions.append(a)
            all_states.append(s)
            print(f"  {key}: {a.shape[0]} segments")
        except Exception as e:
            print(f"  {key} failed: {e}")

    if not all_actions:
        raise RuntimeError("All Kitchen quality tiers failed to load.")

    return TensorDataset(
        torch.cat(all_actions, dim=0),
        torch.cat(all_states,  dim=0),
    )


# ─────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────

def make_synthetic_dataset(
    action_dim: int = 9,
    state_dim: int  = 60,
    n_samples: int  = 1000,
    seq_len: int    = 100,
) -> TensorDataset:
    """Random Gaussian dataset for quick testing."""
    actions = torch.randn(n_samples, seq_len, action_dim) * 0.3
    states  = torch.randn(n_samples, seq_len, state_dim)
    return TensorDataset(actions, states)