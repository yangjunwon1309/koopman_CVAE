"""
eval_online_policy.py — IQL offline + Online policy 통합 평가 + GIF
"""
import os, sys
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.koopman_cvae import KoopmanCVAE
from kodaq_online import (
    KODAQOnlineTrainer, KoopmanWorldModelWrapper,
    EnvContext, OnlineConfig,
)


def save_gif(frames, path, fps=10):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)
        print(f"    GIF: {path}  ({len(frames)} frames)")
    except ImportError:
        n = min(len(frames), 12)
        step = max(1, len(frames)//n)
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        for i, ax in enumerate(np.array(axes).flatten()):
            ax.imshow(frames[min(i*step, len(frames)-1)]); ax.axis('off')
        plt.tight_layout()
        strip = path.replace('.gif','_strip.png')
        plt.savefig(strip, dpi=80); plt.close()
        print(f"    Strip: {strip}")


def render_frame(env, w=256, h=256):
    try:
        f = env.render(mode='rgb_array', width=w, height=h)
        if f is None: f = env.unwrapped.sim.render(w, h, camera_name='main_cam')
    except Exception:
        try: f = env.unwrapped.sim.render(w, h)
        except Exception: f = np.zeros((h, w, 3), dtype=np.uint8)
    return f


def inspect_info(info):
    if 'num_success' in info: return int(info['num_success']), []
    if 'completed_tasks' in info: return len(info['completed_tasks']), list(info['completed_tasks'])
    if 'goal_achieved' in info: return int(info['goal_achieved']), []
    return 0, []


# ─── Policy Wrapper ────────────────────────────────────────────────────────

class PolicyWrapper:
    """IQL GaussianPolicy / Online pi_lo+pi_hi 통합 인터페이스"""

    def __init__(self, mode, device):
        self.mode = mode; self.device = device
        self.H_lo = 1; self.n_skills = None
        self._pi_iql = None; self._trainer = None; self._cfg = None
        self._hi_timer = 0; self._sid = 0

    @classmethod
    def load_iql(cls, world_ckpt, policy_ckpt, device):
        from iql_koopman import GaussianPolicy, IQLConfig
        dev = device
        print(f"\n[IQL] world: {world_ckpt}")
        wc    = torch.load(world_ckpt, map_location=dev)
        model = KoopmanCVAE(wc['cfg']); model.load_state_dict(wc['model_state'])
        model.eval().to(dev)
        z_dim = model.cfg.koopman_dim; a_dim = model.cfg.action_dim
        print(f"  m={z_dim}  action_dim={a_dim}")
        cfg_iql = IQLConfig()
        pi_iql  = GaussianPolicy(z_dim, a_dim, cfg_iql.hidden_dim, cfg_iql.n_layers)
        pc = torch.load(policy_ckpt, map_location=dev)
        pi_iql.load_state_dict(pc['pi']); pi_iql.eval().to(dev)
        print(f"[IQL] policy: {policy_ckpt}  step={pc.get('step','N/A')}")
        cfg_on = OnlineConfig()
        wm     = KoopmanWorldModelWrapper(model, cfg_on.wm_lr, dev)
        pw = cls('iql', device); pw._pi_iql = pi_iql; pw.H_lo = 1
        return pw, model, wm

    @classmethod
    def load_online(cls, world_ckpt, policy_ckpt, cat_ckpt, device):
        dev = device
        print(f"\n[Online] world: {world_ckpt}")
        wc    = torch.load(world_ckpt, map_location=dev)
        model = KoopmanCVAE(wc['cfg']); model.load_state_dict(wc['model_state'])
        model.eval().to(dev)
        z_dim = model.cfg.koopman_dim; n_skills = model.cfg.num_skills
        a_dim = model.cfg.action_dim
        print(f"  K={n_skills}  m={z_dim}  action_dim={a_dim}")
        cat_head = None
        if cat_ckpt and Path(cat_ckpt).exists():
            from train_reward_head import load_cat_reward_model
            _, cat_head = load_cat_reward_model(cat_ckpt, dev)
            print("  CategoricalRewardHead loaded")
        # H_lo를 checkpoint에서 자동 감지
        pc = torch.load(policy_ckpt, map_location=dev)
        if 'world_model' in pc: model.load_state_dict(pc['world_model']); model.eval()
        pi_lo_w = pc['pi_lo']['mu.weight']          # (H_lo*9, hidden)
        H_lo_ckpt = pi_lo_w.shape[0] // a_dim       # 자동 감지
        print(f"  H_lo detected from checkpoint: {H_lo_ckpt}")

        cfg = OnlineConfig(H_lo=H_lo_ckpt)
        wm  = KoopmanWorldModelWrapper(model, cfg.wm_lr, dev,
                                        cat_head=cat_head, reward_H=8, reward_gamma=0.9)
        trainer = KODAQOnlineTrainer(cfg, wm, z_dim, n_skills, a_dim, dev)
        trainer.pi_hi.load_state_dict(pc['pi_hi'])
        trainer.pi_lo.load_state_dict(pc['pi_lo'])
        trainer.pi_hi.eval(); trainer.pi_lo.eval()
        print(f"[Online] policy: {policy_ckpt}  step={pc.get('step','N/A')}")
        pw = cls('online', device)
        pw._trainer = trainer; pw._cfg = cfg
        pw.H_lo = cfg.H_lo; pw.n_skills = n_skills
        return pw, model, wm

    def reset(self):
        self._hi_timer = 0; self._sid = 0

    @torch.no_grad()
    def act(self, z_t):
        dev = torch.device(self.device)
        if self.mode == 'iql':
            mu, _ = self._pi_iql(z_t)
            a = torch.tanh(mu)
            return a[0].cpu().numpy().reshape(1, -1)  # (1, 9)
        else:
            trainer = self._trainer; cfg = self._cfg
            if self._hi_timer == 0:
                self._sid, _ = trainer.pi_hi.hard_sample(z_t)
                self._hi_timer = cfg.H_hi
            a_seq, _ = trainer.pi_lo.sample(z_t)
            self._hi_timer = max(0, self._hi_timer - cfg.H_lo)
            return a_seq[0].cpu().numpy()  # (H_lo, 9)

    @property
    def skill_id(self): return self._sid if self.mode == 'online' else -1


# ─── Rollout ───────────────────────────────────────────────────────────────

def rollout_episode(env, model, policy, device, cond_len=16,
                    max_steps=280, render=True):
    obs = env.reset()
    ctx = EnvContext(model, device, cond_len); ctx.reset(obs); policy.reset()
    frames=[render_frame(env)] if render else []
    rewards=[]; skill_ids=[]; info_hist=[]; info_keys=set()
    total_r=0.0; n_tasks_max=0; done=False; t=0

    while t < max_steps and not done:
        if ctx.z_t is None:
            a = env.action_space.sample()
            obs, r, done, info = env.step(a); ctx.step(obs, a)
            if render: frames.append(render_frame(env))
            total_r+=r; rewards.append(r); skill_ids.append(-1)
            n_t,_=inspect_info(info); n_tasks_max=max(n_tasks_max,n_t)
            info_hist.append(info); info_keys.update(info.keys()); t+=1; continue

        a_seq = policy.act(ctx.z_t); sid = policy.skill_id
        for k in range(len(a_seq)):
            if done or t >= max_steps: break
            ak = a_seq[k].clip(-1,1)
            obs, r, done, info = env.step(ak); ctx.step(obs, ak)
            if render: frames.append(render_frame(env))
            total_r+=r; rewards.append(r); skill_ids.append(sid)
            n_t,_=inspect_info(info); n_tasks_max=max(n_tasks_max,n_t)
            info_hist.append(info); info_keys.update(info.keys()); t+=1

    return {'frames':frames,'rewards':rewards,'total_reward':total_r,
            'n_tasks':n_tasks_max,'skill_ids':skill_ids,
            'info_keys':info_keys,'n_steps':t}


# ─── Visualize ─────────────────────────────────────────────────────────────

def visualize_episode(result, ep_idx, label, mode, out_path):
    rewards=result['rewards']; skill_ids=result['skill_ids']; T=len(rewards)
    if T==0: return
    n_rows=3 if mode=='online' else 2
    fig,axes=plt.subplots(n_rows,1,figsize=(14,3*n_rows))
    PAL=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    ax=axes[0]
    ax.plot(rewards,color='#1E88E5',lw=1.5,alpha=0.8)
    ax2=ax.twinx(); ax2.plot(np.cumsum(rewards),color='#E53935',lw=1.5,ls='--',alpha=0.7)
    ax2.set_ylabel('Cumulative',color='#E53935',fontsize=8)
    ax.set_title(f'[{mode.upper()}] Ep {ep_idx} {label}  |  '
                 f'total={result["total_reward"]:.3f}  tasks={result["n_tasks"]}  steps={result["n_steps"]}',
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('Step Reward'); ax.spines[['top','right']].set_visible(False)
    ax=axes[1]; r_arr=np.array(rewards)
    ax.bar(range(T),r_arr,color='#43A047',alpha=0.5,width=1.0)
    spikes=np.where(r_arr>0.5)[0]
    if len(spikes): ax.scatter(spikes,r_arr[spikes],color='#E53935',s=40,zorder=5,
                                label=f'task complete ({len(spikes)})'); ax.legend(fontsize=8)
    ax.set_xlabel('Step'); ax.set_ylabel('Reward'); ax.spines[['top','right']].set_visible(False)
    if mode=='online' and n_rows==3:
        ax=axes[2]; valid=np.array(skill_ids)>=0
        if valid.any():
            ts=np.where(valid)[0]; sids=np.array(skill_ids)[valid]
            for k in range(7):
                m=sids==k
                if m.any(): ax.scatter(ts[m],[k]*m.sum(),c=PAL[k%len(PAL)],s=4,alpha=0.7,label=f's{k}')
        ax.set_ylabel('Skill'); ax.set_ylim(-0.5,6.5); ax.set_yticks(range(7))
        ax.legend(fontsize=6,ncol=7); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(out_path,dpi=120,bbox_inches='tight'); plt.close()


def visualize_summary(results, mode, out_path):
    total_rs=[r['total_reward'] for r in results]
    n_tasks=[r['n_tasks'] for r in results]
    n_steps=[r['n_steps'] for r in results]
    ep_ids=list(range(len(results)))
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,vals,title,col in zip(axes,
        [total_rs,n_tasks,n_steps],
        ['Total Reward','Tasks Completed','Steps'],
        ['#1E88E5','#43A047','#FB8C00']):
        ax.bar(ep_ids,vals,color=col,alpha=0.8)
        ax.axhline(np.mean(vals),color='#E53935',ls='--',lw=1.5,label=f'mean={np.mean(vals):.3f}')
        ax.set_xlabel('Episode'); ax.set_title(title,fontsize=10,fontweight='bold')
        ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'[{mode.upper()}] Evaluation ({len(results)} episodes)\n'
                 f'mean_reward={np.mean(total_rs):.4f}  mean_tasks={np.mean(n_tasks):.2f}',
                 fontsize=11,fontweight='bold')
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(out_path,dpi=130,bbox_inches='tight'); plt.close()
    print(f"Summary: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',        choices=['iql','online'], default='online')
    p.add_argument('--world_ckpt',  required=True)
    p.add_argument('--policy_ckpt', required=True)
    p.add_argument('--cat_ckpt',    default=None)
    p.add_argument('--env',         default='kitchen-mixed-v0')
    p.add_argument('--n_ep',        type=int, default=10)
    p.add_argument('--fixed_seed',  type=int, default=42)
    p.add_argument('--fps',         type=int, default=10)
    p.add_argument('--max_steps',   type=int, default=280)
    p.add_argument('--no_gif',      action='store_true')
    p.add_argument('--out_dir',     default='checkpoints/eval')
    p.add_argument('--device',      default='cuda:1' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()

    import gym, d4rl
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode=='iql':
        policy, model, wm = PolicyWrapper.load_iql(
            args.world_ckpt, args.policy_ckpt, args.device)
        cond_len=16
    else:
        policy, model, wm = PolicyWrapper.load_online(
            args.world_ckpt, args.policy_ckpt, args.cat_ckpt, args.device)
        cond_len=OnlineConfig().cond_len

    results=[]; all_info_keys=set()
    print(f"\n{'='*55}\n[{args.mode.upper()}] {args.n_ep} episodes  env={args.env}\n{'='*55}\n")

    for ep_i in range(args.n_ep):
        env=gym.make(args.env)
        if ep_i < args.n_ep//2:
            env.seed(args.fixed_seed+ep_i); label=f'seed{args.fixed_seed+ep_i}'
        else: label='random'
        print(f"  Ep {ep_i:2d} [{label}]  ", end='', flush=True)
        result=rollout_episode(env, model, policy, args.device,
                               cond_len=cond_len, max_steps=args.max_steps,
                               render=not args.no_gif)
        env.close(); all_info_keys.update(result['info_keys']); results.append(result)
        print(f"reward={result['total_reward']:.3f}  tasks={result['n_tasks']}  steps={result['n_steps']}")
        if not args.no_gif and result['frames']:
            save_gif(result['frames'],
                     f"{args.out_dir}/gif/ep{ep_i:02d}_{label}_r{result['total_reward']:.2f}.gif",
                     fps=args.fps)
        visualize_episode(result, ep_i, label, args.mode,
                          f"{args.out_dir}/plots/ep{ep_i:02d}_{label}.png")

    total_rs=[r['total_reward'] for r in results]
    n_tasks=[r['n_tasks'] for r in results]
    print(f"\n{'='*55}\n[{args.mode.upper()}] Results:")
    print(f"  mean_reward: {np.mean(total_rs):.4f} ± {np.std(total_rs):.4f}")
    print(f"  max_reward:  {np.max(total_rs):.4f}")
    print(f"  mean_tasks:  {np.mean(n_tasks):.4f}  max_tasks: {int(np.max(n_tasks))}")
    task_keys=[k for k in all_info_keys if any(w in k.lower()
               for w in ['task','success','complete','solve','goal'])]
    print(f"  info keys: {sorted(all_info_keys)}")
    if task_keys: print(f"  → task candidates: {task_keys}")
    print(f"{'='*55}")
    visualize_summary(results, args.mode, f"{args.out_dir}/summary_{args.mode}.png")
    print(f"\nOutputs → {args.out_dir}/")


if __name__=='__main__':
    main()