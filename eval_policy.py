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
    z_seq_list=[]; action_list=[]
    total_r=0.0; n_tasks_max=0; done=False; t=0

    while t < max_steps and not done:
        if ctx.z_t is None:
            a = env.action_space.sample()
            obs, r, done, info = env.step(a); ctx.step(obs, a)
            if render: frames.append(render_frame(env))
            total_r+=r; rewards.append(r); skill_ids.append(-1)
            action_list.append(a)
            if ctx.z_t is not None: z_seq_list.append(ctx.z_t[0].cpu().numpy())
            n_t,_=inspect_info(info); n_tasks_max=max(n_tasks_max,n_t)
            info_hist.append(info); info_keys.update(info.keys()); t+=1; continue

        a_seq = policy.act(ctx.z_t); sid = policy.skill_id
        for k in range(len(a_seq)):
            if done or t >= max_steps: break
            ak = a_seq[k].clip(-1,1)
            obs, r, done, info = env.step(ak); ctx.step(obs, ak)
            if render: frames.append(render_frame(env))
            total_r+=r; rewards.append(r); skill_ids.append(sid)
            action_list.append(ak)
            if ctx.z_t is not None: z_seq_list.append(ctx.z_t[0].cpu().numpy())
            n_t,_=inspect_info(info); n_tasks_max=max(n_tasks_max,n_t)
            info_hist.append(info); info_keys.update(info.keys()); t+=1

    return {'frames':frames,'rewards':rewards,'total_reward':total_r,
            'n_tasks':n_tasks_max,'skill_ids':skill_ids,
            'info_keys':info_keys,'n_steps':t,
            'z_seq':   z_seq_list,    # (T, m) for reward analysis
            'actions': action_list,   # (T, 9) for action distribution
            'H_lo':    policy.H_lo}


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

    # ─── Analysis (online mode에서 전체 분석 실행) ─────────────────────────
    if args.mode == 'online':
        print("\n" + "="*55 + "\nRunning analysis...")
        res_a = analyze_skill_switching(
            results, f"{args.out_dir}/analysis_skill_switching.png")
        res_b = analyze_action_distribution(
            results, f"{args.out_dir}/analysis_action_dist.png")
        res_c = analyze_reward_distribution(
            results, wm, f"{args.out_dir}/analysis_reward_dist.png", args.device)
        # save text report
        report = "\n\n".join([
            res_a.get('text',''), res_b.get('text',''), res_c.get('text','')])
        rpath = f"{args.out_dir}/analysis_report.txt"
        Path(rpath).write_text(report)
        print(f"  Text report: {rpath}")

    elif args.mode == 'iql':
        # IQL도 action 분포와 reward 분포 분석
        res_b = analyze_action_distribution(
            results, f"{args.out_dir}/analysis_action_dist.png")
        res_c = analyze_reward_distribution(
            results, wm, f"{args.out_dir}/analysis_reward_dist.png", args.device)
        report = "\n\n".join([res_b.get('text',''), res_c.get('text','')])
        rpath = f"{args.out_dir}/analysis_report.txt"
        Path(rpath).write_text(report)
        print(f"  Text report: {rpath}")

    print(f"\nOutputs → {args.out_dir}/")


if __name__=='__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════
# Analysis Module: Skill switching / Action distribution / Reward distribution
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_skill_switching(results_with_extra, out_path: str):
    """
    분석 1: Hi-level skill switching 패턴

    - skill change rate: 매 H_hi step마다 skill이 바뀌는 비율
    - action discontinuity: 연속된 chunk 간 action 차이 (L2 norm)
    - skill duration histogram: 각 skill이 몇 step 유지됐는지

    진동 원인 진단:
      skill이 매 chunk마다 바뀌면 → action 불연속 → 진동
      skill이 유지되면 → action이 일관됨 → 안정적
    """
    import matplotlib.gridspec as gridspec

    all_sid    = []
    all_adiff  = []
    sid_durations = {}  # skill_id → list of durations
    chunk_changes = []  # 0 or 1 per chunk

    for res in results_with_extra:
        sids    = np.array(res['skill_ids'])
        actions = np.array(res['actions'])   # (T, 9)

        # skill change per step
        if len(sids) > 1:
            changes = (sids[1:] != sids[:-1]).astype(float)
            chunk_changes.extend(changes.tolist())

        # consecutive action chunk L2 diff
        H_lo = res.get('H_lo', 4)
        for t in range(H_lo, len(actions) - H_lo, H_lo):
            prev_chunk = actions[t - H_lo:t]
            curr_chunk = actions[t:t + H_lo]
            diff = np.linalg.norm(curr_chunk - prev_chunk, axis=-1).mean()
            all_adiff.append(diff)

        # skill durations
        run_sid, run_len = sids[0] if len(sids) else -1, 0
        for sid in sids:
            if sid == run_sid:
                run_len += 1
            else:
                if run_sid >= 0:
                    sid_durations.setdefault(int(run_sid), []).append(run_len)
                run_sid, run_len = sid, 1
        if run_sid >= 0 and run_len > 0:
            sid_durations.setdefault(int(run_sid), []).append(run_len)

        all_sid.extend(sids[sids >= 0].tolist())

    # Text summary
    change_rate = np.mean(chunk_changes) if chunk_changes else 0.0
    mean_adiff  = np.mean(all_adiff) if all_adiff else 0.0
    text_lines  = [
        "=== Skill Switching Analysis ===",
        f"Skill change rate per step: {change_rate:.3f}  "
        f"(1.0 = every step, 0.0 = never)",
        f"Mean action chunk L2 diff:  {mean_adiff:.4f}",
        f"Diagnosis: {'SEVERE OSCILLATION (skill changes too often)' if change_rate > 0.3 else 'OK' if change_rate < 0.1 else 'MODERATE'}",
        "",
        "Skill usage (steps):",
    ]
    PAL = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    for sid in sorted(sid_durations):
        durs = sid_durations[sid]
        text_lines.append(f"  Skill {sid}: count={len(durs)}  "
                          f"mean_duration={np.mean(durs):.1f}  "
                          f"max={np.max(durs)}")

    print("\n" + "\n".join(text_lines))

    # Plot
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Skill ID histogram
    ax = fig.add_subplot(gs[0, 0])
    if all_sid:
        unique, counts = np.unique(all_sid, return_counts=True)
        ax.bar(unique, counts, color=[PAL[int(s)%len(PAL)] for s in unique], alpha=0.8)
        ax.set_xlabel('Skill ID'); ax.set_ylabel('Step count')
        ax.set_title('Skill Usage Distribution', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # 2. Skill change rate rolling
    ax = fig.add_subplot(gs[0, 1])
    if chunk_changes:
        cc  = np.array(chunk_changes)
        w   = min(50, max(1, len(cc)//10))
        roll= np.convolve(cc, np.ones(w)/w, 'valid')
        ax.plot(roll, color='#E53935', lw=1.5)
        ax.axhline(change_rate, color='k', ls='--', lw=1, alpha=0.6,
                   label=f'mean={change_rate:.3f}')
        ax.set_title('Rolling Skill Change Rate', fontweight='bold')
        ax.set_ylabel('Change rate'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 3. Action chunk discontinuity
    ax = fig.add_subplot(gs[0, 2])
    if all_adiff:
        ax.hist(all_adiff, bins=40, color='#1E88E5', alpha=0.8, edgecolor='white')
        ax.axvline(mean_adiff, color='#E53935', ls='--', lw=1.5,
                   label=f'mean={mean_adiff:.3f}')
        ax.set_title('Action Chunk L2 Discontinuity', fontweight='bold')
        ax.set_xlabel('L2 norm diff'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 4. Skill duration boxplot
    ax = fig.add_subplot(gs[1, :2])
    if sid_durations:
        data  = [sid_durations[k] for k in sorted(sid_durations)]
        labels= [f'Skill {k}' for k in sorted(sid_durations)]
        bp    = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, col in zip(bp['boxes'], PAL):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_ylabel('Duration (steps)')
        ax.set_title('Skill Duration Distribution', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # 5. Text summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ax.text(0.05, 0.95, "\n".join(text_lines[:8]), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle('Skill Switching & Action Continuity Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")

    return {'change_rate': change_rate, 'mean_adiff': mean_adiff,
            'text': "\n".join(text_lines)}


@torch.no_grad()
def analyze_action_distribution(results_with_extra, out_path: str):
    """
    분석 2: Action sequence 분포

    - 각 joint (9-dim)별 action 분포 (histogram)
    - action chunk 내 time consistency (chunk 내에서 action이 얼마나 일관적인지)
    - action magnitude over episode (진동 패턴)
    """
    all_actions = []  # (T_total, 9)
    chunk_std   = []  # per chunk, action std across H_lo steps

    joint_names = [f'J{i}' for i in range(9)]

    for res in results_with_extra:
        acts   = np.array(res['actions'])   # (T, 9)
        H_lo   = res.get('H_lo', 4)
        all_actions.append(acts)
        for t in range(0, len(acts) - H_lo + 1, H_lo):
            chunk = acts[t:t+H_lo]
            chunk_std.append(chunk.std(axis=0))  # (9,)

    if not all_actions:
        return {}

    all_acts  = np.concatenate(all_actions, axis=0)   # (N, 9)
    chunk_std = np.array(chunk_std)                    # (M, 9)

    # Text summary
    text_lines = [
        "=== Action Distribution Analysis ===",
        f"Total action steps: {len(all_acts)}",
        "",
        "Per-joint stats (mean ± std):",
    ]
    for j in range(9):
        text_lines.append(f"  J{j}: mean={all_acts[:,j].mean():.3f}  "
                          f"std={all_acts[:,j].std():.3f}  "
                          f"[{all_acts[:,j].min():.2f}, {all_acts[:,j].max():.2f}]")
    if len(chunk_std):
        text_lines += ["", "Intra-chunk std (consistency within chunk):"]
        for j in range(9):
            text_lines.append(f"  J{j}: {chunk_std[:,j].mean():.4f}")

    print("\n" + "\n".join(text_lines))

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    PAL  = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA',
            '#00ACC1','#FFB300','#607D8B','#795548']

    for j in range(9):
        ax = axes[j]
        ax.hist(all_acts[:, j], bins=50, color=PAL[j], alpha=0.8,
                edgecolor='white', density=True)
        m, s = all_acts[:,j].mean(), all_acts[:,j].std()
        ax.axvline(m, color='k', ls='--', lw=1.5, label=f'μ={m:.3f}')
        ax.axvspan(m-s, m+s, alpha=0.15, color=PAL[j])
        ax.set_title(f'Joint {j}  σ={s:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Action value [-1, 1]')
        ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Action Distribution per Joint', fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()

    # Intra-chunk std plot
    if len(chunk_std):
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        x = np.arange(9)
        ax2.bar(x, chunk_std.mean(axis=0), color=PAL[:9], alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels([f'J{i}' for i in range(9)])
        ax2.set_ylabel('Mean intra-chunk std')
        ax2.set_title('Action Consistency Within Chunk\n'
                      '(high = action changes a lot within one chunk → oscillation)',
                      fontsize=10, fontweight='bold')
        ax2.spines[['top','right']].set_visible(False)
        chunk_path = out_path.replace('.png', '_chunk_std.png')
        plt.tight_layout()
        plt.savefig(chunk_path, dpi=130, bbox_inches='tight'); plt.close()
        print(f"  Saved: {chunk_path}")

    print(f"  Saved: {out_path}")
    return {'mean_per_joint': all_acts.mean(axis=0).tolist(),
            'std_per_joint':  all_acts.std(axis=0).tolist(),
            'text': "\n".join(text_lines)}


@torch.no_grad()
def analyze_reward_distribution(results_with_extra, wm,
                                out_path: str, device: str):
    """
    분석 3: Event / Accumulated reward 분포

    실제 시뮬레이션에서 얻은 z_t에 대해:
    - r_event = BCE head sigmoid(z_t)
    - r_acc   = cat_head.expected_reward(z_t)  if available
    - 두 reward의 분포를 episode 진행에 따라 확인
    - OOD 여부: world model이 본 적 없는 z_t에서 reward가 어떻게 나오는지
    """
    dev   = torch.device(device)
    model = wm.model
    model.eval()

    all_r_event = []
    all_r_acc   = []
    all_r_env   = []
    all_t       = []   # normalized time (0~1)

    for res in results_with_extra:
        z_seq  = np.array(res['z_seq'])    # (T, m)
        r_env  = np.array(res['rewards'])  # (T,)
        T      = len(z_seq)
        if T == 0: continue

        z_t = torch.FloatTensor(z_seq).to(dev)

        # Event reward: BCE head
        r_event_vals = []
        if model.cfg.use_reward_head:
            if hasattr(model.decoder, 'head_reward'):
                logits = model.decoder.head_reward(z_t)
            elif hasattr(model, 'reward_head'):
                logits = model.reward_head(z_t)
            else:
                logits = None
            if logits is not None:
                r_event_vals = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        if not len(r_event_vals):
            r_event_vals = np.zeros(T)

        # Accumulated reward: cat_head
        r_acc_vals = np.zeros(T)
        if wm.cat_head is not None:
            r_acc_vals = wm.cat_head.expected_reward(z_t).cpu().numpy()

        all_r_event.extend(r_event_vals.tolist())
        all_r_acc.extend(r_acc_vals.tolist())
        all_r_env.extend(r_env[:T].tolist())
        all_t.extend((np.arange(T) / max(T-1, 1)).tolist())

    all_r_event = np.array(all_r_event)
    all_r_acc   = np.array(all_r_acc)
    all_r_env   = np.array(all_r_env)
    all_t       = np.array(all_t)

    # Text summary
    text_lines = [
        "=== Reward Head Distribution Analysis ===",
        "",
        f"Event reward (BCE):      mean={all_r_event.mean():.4f}  "
        f"std={all_r_event.std():.4f}  "
        f">0.5: {(all_r_event>0.5).mean()*100:.1f}%",
        f"Accumulated reward (cat): mean={all_r_acc.mean():.4f}  "
        f"std={all_r_acc.std():.4f}  "
        f"max={all_r_acc.max():.2f}",
        f"Env reward (actual):      mean={all_r_env.mean():.4f}  "
        f">0.5: {(all_r_env>0.5).mean()*100:.1f}%",
        "",
        "Diagnosis:",
        f"  r_event saturation: {'HIGH (>0.5 = 30%+) → unreliable' if (all_r_event>0.5).mean()>0.3 else 'OK'}",
        f"  r_acc range: {'SATURATED near max (4)' if all_r_acc.mean()>3.0 else 'OK' if all_r_acc.mean()<1.0 else 'MODERATE'}",
        f"  r_env sparse: {(all_r_env>0.5).mean()*100:.2f}% steps with actual reward",
    ]

    print("\n" + "\n".join(text_lines))

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    PAL = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']

    # 1. r_event histogram
    ax = axes[0, 0]
    ax.hist(all_r_event, bins=50, color=PAL[0], alpha=0.8, edgecolor='white')
    ax.axvline(all_r_event.mean(), color='k', ls='--', lw=1.5,
               label=f'mean={all_r_event.mean():.3f}')
    ax.axvline(0.5, color='orange', ls=':', lw=1.5, label='threshold=0.5')
    ax.set_title('Event Reward (BCE) Distribution', fontweight='bold')
    ax.set_xlabel('r_event'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 2. r_acc histogram
    ax = axes[0, 1]
    ax.hist(all_r_acc, bins=50, color=PAL[1], alpha=0.8, edgecolor='white')
    ax.axvline(all_r_acc.mean(), color='k', ls='--', lw=1.5,
               label=f'mean={all_r_acc.mean():.3f}')
    ax.set_title('Accumulated Reward (Cat) Distribution\nE[R|z_t] ∈ [0,4]',
                 fontweight='bold')
    ax.set_xlabel('r_acc'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # 3. r_env (actual) histogram
    ax = axes[0, 2]
    ax.hist(all_r_env, bins=20, color=PAL[2], alpha=0.8, edgecolor='white')
    ax.set_title(f'Actual Env Reward Distribution\n'
                 f'({(all_r_env>0.5).mean()*100:.2f}% > 0.5)',
                 fontweight='bold')
    ax.set_xlabel('r_env')
    ax.spines[['top','right']].set_visible(False)

    # 4. r_event over normalized episode time
    ax = axes[1, 0]
    if len(all_t) > 100:
        # bin by time
        bins = np.linspace(0, 1, 21)
        bin_means, bin_stds = [], []
        for i in range(len(bins)-1):
            mask = (all_t >= bins[i]) & (all_t < bins[i+1])
            if mask.any():
                bin_means.append(all_r_event[mask].mean())
                bin_stds.append(all_r_event[mask].std())
            else:
                bin_means.append(0); bin_stds.append(0)
        bc = 0.5*(bins[:-1]+bins[1:])
        ax.plot(bc, bin_means, color=PAL[0], lw=1.5)
        ax.fill_between(bc,
                        np.array(bin_means)-np.array(bin_stds),
                        np.array(bin_means)+np.array(bin_stds),
                        alpha=0.2, color=PAL[0])
    ax.set_xlabel('Normalized episode time (0=start, 1=end)')
    ax.set_ylabel('r_event')
    ax.set_title('Event Reward over Episode', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # 5. r_acc over time
    ax = axes[1, 1]
    if len(all_t) > 100:
        bin_means, bin_stds = [], []
        for i in range(len(bins)-1):
            mask = (all_t >= bins[i]) & (all_t < bins[i+1])
            if mask.any():
                bin_means.append(all_r_acc[mask].mean())
                bin_stds.append(all_r_acc[mask].std())
            else:
                bin_means.append(0); bin_stds.append(0)
        ax.plot(bc, bin_means, color=PAL[1], lw=1.5)
        ax.fill_between(bc,
                        np.array(bin_means)-np.array(bin_stds),
                        np.array(bin_means)+np.array(bin_stds),
                        alpha=0.2, color=PAL[1])
    ax.set_xlabel('Normalized episode time')
    ax.set_ylabel('r_acc')
    ax.set_title('Accumulated Reward over Episode', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # 6. Text summary box
    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle('Reward Head Distribution Analysis\n'
                 '(from actual simulation z_t)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")

    return {'r_event_mean': all_r_event.mean(),
            'r_acc_mean':   all_r_acc.mean(),
            'r_env_mean':   all_r_env.mean(),
            'text': "\n".join(text_lines)}