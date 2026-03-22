import numpy as np, torch
from torch.utils.data import TensorDataset
from typing import List, Dict, Optional, Tuple

def split_into_trajectories(dataset, min_len=50):
    obs=dataset['observations']; acts=dataset['actions']; rews=dataset['rewards']
    terms=dataset['terminals'].astype(bool)
    timeouts=dataset.get('timeouts', np.zeros_like(terms,dtype=bool))
    ends=terms|timeouts
    trajectories=[]; start=0
    for i in range(len(ends)):
        if ends[i] or i==len(ends)-1:
            end=i+1; length=end-start
            if length>=min_len:
                trajectories.append({'observations':obs[start:end],'actions':acts[start:end],'rewards':rews[start:end]})
            start=end
    print(f"  Extracted {len(trajectories)} episodes (min_len={min_len})")
    return trajectories

def segment_trajectories(trajectories, seq_len=100, stride=None):
    if stride is None: stride=seq_len//2
    actions_list=[]; states_list=[]
    for traj in trajectories:
        T=len(traj['observations'])
        for start in range(0,T-seq_len+1,stride):
            end=start+seq_len
            states_list.append(traj['observations'][start:end])
            actions_list.append(traj['actions'][start:end])
    if not actions_list: raise ValueError(f"No segments. Reduce seq_len={seq_len}")
    actions_arr=np.stack(actions_list); states_arr=np.stack(states_list)
    print(f"  Segments: {actions_arr.shape[0]} x {seq_len}")
    return actions_arr, states_arr

D4RL_ENV_MAP={
    'adroit_pen':     {'expert':'pen-expert-v1','human':'pen-human-v1','cloned':'pen-cloned-v1'},
    'adroit_hammer':  {'expert':'hammer-expert-v1','human':'hammer-human-v1','cloned':'hammer-cloned-v1'},
    'adroit_door':    {'expert':'door-expert-v1','human':'door-human-v1','cloned':'door-cloned-v1'},
    'adroit_relocate':{'expert':'relocate-expert-v1','human':'relocate-human-v1','cloned':'relocate-cloned-v1'},
}

def load_d4rl_trajectories(env_name, seq_len=100, stride=None, min_episode_len=50, quality='expert'):
    try: import d4rl, gym
    except ImportError: raise ImportError("d4rl not installed. pip install d4rl")
    d4rl_name=D4RL_ENV_MAP[env_name][quality] if env_name in D4RL_ENV_MAP else env_name
    print(f"  Loading D4RL: {d4rl_name}")
    env=gym.make(d4rl_name); dataset=d4rl.qlearning_dataset(env)
    trajectories=split_into_trajectories(dataset,min_len=min_episode_len)
    actions_arr,states_arr=segment_trajectories(trajectories,seq_len,stride)
    return TensorDataset(torch.FloatTensor(actions_arr),torch.FloatTensor(states_arr))

def make_synthetic_dataset(action_dim=6,state_dim=24,n_samples=1000,seq_len=100):
    return TensorDataset(torch.randn(n_samples,seq_len,action_dim)*0.3,torch.randn(n_samples,seq_len,state_dim))