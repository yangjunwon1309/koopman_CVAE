"""
Crop D4RL Franka Kitchen episodes up to reward 2.

The output is a D4RL-like flat NPZ:
  observations, actions, rewards, terminals
plus episode metadata:
  episode_starts, episode_ends, original_starts, original_ends,
  crop_ts, reward_at_crop, original_lengths, cropped_lengths

Example:
  python crop_episodes.py --quality mixed --target_reward 2 \
      --out data/kitchen_mixed_reward2.npz
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from lqr_koopman import (
    ALL_TASKS,
    detect_completed_tasks_by_reward,
    detect_task_completions,
)


def _reward_trace(rewards: np.ndarray, mode: str) -> np.ndarray:
    """Return the trace used to find the reward threshold crossing."""
    rewards = rewards.astype(np.float32)
    if mode == "raw":
        return rewards
    if mode == "cumsum":
        return np.cumsum(rewards)
    if mode != "auto":
        raise ValueError(f"Unknown reward_mode: {mode}")

    diffs = np.diff(rewards)
    is_monotone = bool(np.all(diffs >= -1e-6))
    looks_cumulative = is_monotone and rewards.max(initial=0.0) > 1.0
    return rewards if looks_cumulative else np.cumsum(rewards)


def _first_threshold_index(trace: np.ndarray, target_reward: float) -> int:
    hits = np.where(trace >= target_reward)[0]
    return int(hits[0]) if len(hits) else -1


def _build_goal_info(obs_ep: np.ndarray, rew_ep: np.ndarray) -> Dict:
    completed = detect_completed_tasks_by_reward(obs_ep, rew_ep)
    if not completed:
        completed = detect_task_completions(obs_ep, ALL_TASKS)
        completed = {k: v for k, v in completed.items() if v >= 0}

    subtask_goals = {
        task: {"obs": obs_ep[t], "timestep": int(t), "completed": True}
        for task, t in completed.items()
    }
    L = len(obs_ep)
    return {
        "final_goal": obs_ep[-1],
        "midpoint_goal": obs_ep[L // 2],
        "subtask_goals": subtask_goals,
        "completions": completed,
        "episode_len": L,
        "n_completed": len(completed),
        "reward_total": float(rew_ep.sum()),
    }


def load_kitchen_episodes(
    quality: str = "mixed",
    min_len: int = 64,
) -> Tuple[List[Dict], np.ndarray]:
    """Load D4RL Kitchen and split into episodes."""
    import d4rl  # noqa: F401
    import gym

    name_map = {
        "mixed": "kitchen-mixed-v0",
        "partial": "kitchen-partial-v0",
        "complete": "kitchen-complete-v0",
    }
    env = gym.make(name_map[quality])
    dataset = env.get_dataset()

    obs = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset.get("rewards", np.zeros(len(obs)))
    terminals = dataset["terminals"].astype(bool)

    ep_ends = list(np.where(terminals)[0])
    ep_starts = [0] + [e + 1 for e in ep_ends[:-1]]
    episodes = []

    for ep_s, ep_e in zip(ep_starts, ep_ends):
        L = ep_e - ep_s + 1
        if L < min_len:
            continue

        obs_ep = obs[ep_s : ep_e + 1]
        acts_ep = actions[ep_s : ep_e + 1]
        rew_ep = rewards[ep_s : ep_e + 1]
        goal_info = _build_goal_info(obs_ep, rew_ep)

        episodes.append(
            {
                "obs": obs_ep,
                "actions": acts_ep,
                "rewards": rew_ep,
                "start_t": ep_s,
                "end_t": ep_e,
                "length": L,
                "tasks": list(goal_info["completions"].keys()),
                "goal_info": goal_info,
            }
        )

    n_tasks = sum(1 for e in episodes if e["tasks"])
    print(f"Episodes: {len(episodes)}  with_tasks={n_tasks}/{len(episodes)}")
    return episodes, obs


def crop_episodes_to_reward(
    episodes: List[Dict],
    target_reward: float = 2.0,
    min_len: int = 16,
    reward_mode: str = "auto",
    keep_no_hit: bool = False,
) -> List[Dict]:
    """Crop each episode through the first timestep where reward reaches target."""
    cropped = []

    for ep_idx, ep in enumerate(episodes):
        trace = _reward_trace(ep["rewards"], reward_mode)
        crop_t = _first_threshold_index(trace, target_reward)

        if crop_t < 0:
            if not keep_no_hit:
                continue
            crop_t = ep["length"] - 1

        new_len = crop_t + 1
        if new_len < min_len:
            continue

        obs_ep = ep["obs"][:new_len]
        acts_ep = ep["actions"][:new_len]
        rew_ep = ep["rewards"][:new_len]
        goal_info = _build_goal_info(obs_ep, rew_ep)

        cropped.append(
            {
                "obs": obs_ep,
                "actions": acts_ep,
                "rewards": rew_ep,
                "start_t": ep["start_t"],
                "end_t": ep["start_t"] + crop_t,
                "length": new_len,
                "tasks": list(goal_info["completions"].keys()),
                "goal_info": goal_info,
                "source_episode": ep_idx,
                "source_start_t": ep["start_t"],
                "source_end_t": ep["end_t"],
                "source_length": ep["length"],
                "crop_t": crop_t,
                "reward_at_crop": float(trace[crop_t]),
            }
        )

    return cropped


def flatten_episodes(episodes: List[Dict]) -> Dict[str, np.ndarray]:
    obs_parts, action_parts, reward_parts, terminal_parts = [], [], [], []
    ep_starts, ep_ends = [], []
    original_starts, original_ends = [], []
    crop_ts, reward_at_crop = [], []
    original_lengths, cropped_lengths = [], []

    cursor = 0
    for ep in episodes:
        L = ep["length"]
        obs_parts.append(ep["obs"])
        action_parts.append(ep["actions"])
        reward_parts.append(ep["rewards"])

        terminals = np.zeros(L, dtype=bool)
        terminals[-1] = True
        terminal_parts.append(terminals)

        ep_starts.append(cursor)
        ep_ends.append(cursor + L - 1)
        cursor += L

        original_starts.append(ep["source_start_t"])
        original_ends.append(ep["source_end_t"])
        crop_ts.append(ep["crop_t"])
        reward_at_crop.append(ep["reward_at_crop"])
        original_lengths.append(ep["source_length"])
        cropped_lengths.append(L)

    if not episodes:
        raise ValueError("No episodes left after cropping. Lower min_len or use --keep_no_hit.")

    return {
        "observations": np.concatenate(obs_parts, axis=0).astype(np.float32),
        "actions": np.concatenate(action_parts, axis=0).astype(np.float32),
        "rewards": np.concatenate(reward_parts, axis=0).astype(np.float32),
        "terminals": np.concatenate(terminal_parts, axis=0).astype(bool),
        "episode_starts": np.asarray(ep_starts, dtype=np.int64),
        "episode_ends": np.asarray(ep_ends, dtype=np.int64),
        "original_starts": np.asarray(original_starts, dtype=np.int64),
        "original_ends": np.asarray(original_ends, dtype=np.int64),
        "crop_ts": np.asarray(crop_ts, dtype=np.int64),
        "reward_at_crop": np.asarray(reward_at_crop, dtype=np.float32),
        "original_lengths": np.asarray(original_lengths, dtype=np.int64),
        "cropped_lengths": np.asarray(cropped_lengths, dtype=np.int64),
    }


def save_cropped_dataset(
    episodes: List[Dict],
    out_path: str,
    args: argparse.Namespace,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays = flatten_episodes(episodes)
    np.savez_compressed(out, **arrays)

    meta = {
        "quality": args.quality,
        "target_reward": args.target_reward,
        "reward_mode": args.reward_mode,
        "min_len": args.min_len,
        "crop_min_len": args.crop_min_len,
        "keep_no_hit": args.keep_no_hit,
        "n_episodes": len(episodes),
        "n_steps": int(arrays["observations"].shape[0]),
        "length_min": int(arrays["cropped_lengths"].min()),
        "length_max": int(arrays["cropped_lengths"].max()),
        "length_mean": float(arrays["cropped_lengths"].mean()),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {out}")
    print(f"Meta:  {meta_path}")
    print(
        "Cropped episodes: "
        f"{meta['n_episodes']}  steps={meta['n_steps']}  "
        f"len=[{meta['length_min']},{meta['length_max']}]  "
        f"mean={meta['length_mean']:.1f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quality", choices=["mixed", "partial", "complete"], default="mixed")
    p.add_argument("--target_reward", type=float, default=2.0)
    p.add_argument("--min_len", type=int, default=64, help="Minimum original episode length.")
    p.add_argument("--crop_min_len", type=int, default=16, help="Minimum cropped episode length.")
    p.add_argument(
        "--reward_mode",
        choices=["auto", "raw", "cumsum"],
        default="auto",
        help="Use raw cumulative rewards, cumulative sum of step rewards, or infer automatically.",
    )
    p.add_argument(
        "--keep_no_hit",
        action="store_true",
        help="Keep episodes that never reach the target reward, cropped at the original end.",
    )
    p.add_argument("--out", default="data/kitchen_mixed_reward2.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    episodes, _ = load_kitchen_episodes(quality=args.quality, min_len=args.min_len)
    cropped = crop_episodes_to_reward(
        episodes,
        target_reward=args.target_reward,
        min_len=args.crop_min_len,
        reward_mode=args.reward_mode,
        keep_no_hit=args.keep_no_hit,
    )
    save_cropped_dataset(cropped, args.out, args)


if __name__ == "__main__":
    main()
