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
    # D4RL kitchen-mixed에서 실제 task completion key
    if 'score' in info:
        # score는 완료된 subtask 수 (0~4)
        return int(round(float(info['score']) * 4)), []
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



# ═══════════════════════════════════════════════════════════════════════════
# Analysis Module: Skill switching / Action distribution / Reward distribution
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()



# ═══════════════════════════════════════════════════════════════════════════
# Analysis: Skill Switching / Action Distribution / Reward Distribution
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_skill_switching(results_with_extra, out_path):
    import matplotlib.gridspec as gridspec
    all_sid=[]; all_adiff=[]; sid_durations={}; chunk_changes=[]
    PAL=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300']
    for res in results_with_extra:
        sids=np.array(res['skill_ids']); actions=np.array(res['actions'])
        if len(sids)>1: chunk_changes.extend((sids[1:]!=sids[:-1]).astype(float).tolist())
        H_lo=res.get('H_lo',4)
        for t in range(H_lo, len(actions)-H_lo, H_lo):
            diff=np.linalg.norm(actions[t:t+H_lo]-actions[t-H_lo:t],axis=-1).mean()
            all_adiff.append(diff)
        run_sid,run_len=sids[0] if len(sids) else -1,0
        for sid in sids:
            if sid==run_sid: run_len+=1
            else:
                if run_sid>=0: sid_durations.setdefault(int(run_sid),[]).append(run_len)
                run_sid,run_len=sid,1
        if run_sid>=0 and run_len>0: sid_durations.setdefault(int(run_sid),[]).append(run_len)
        all_sid.extend(sids[sids>=0].tolist())
    cr=np.mean(chunk_changes) if chunk_changes else 0.0
    ma=np.mean(all_adiff) if all_adiff else 0.0
    lines=[
        "=== Skill Switching Analysis ===",
        f"Skill change rate per step: {cr:.3f}  (1.0=every step, 0.0=never)",
        f"Mean action chunk L2 diff:  {ma:.4f}",
        f"Diagnosis: {'SEVERE OSCILLATION' if cr>0.3 else 'OK' if cr<0.1 else 'MODERATE'}",
        "","Skill usage (steps):",
    ]
    for sid in sorted(sid_durations):
        d=sid_durations[sid]
        lines.append(f"  Skill {sid}: count={len(d)}  mean_dur={np.mean(d):.1f}  max={np.max(d)}")
    print("\n".join(lines))
    fig=plt.figure(figsize=(18,10)); gs=gridspec.GridSpec(2,3,figure=fig)
    ax=fig.add_subplot(gs[0,0])
    if all_sid:
        u,c=np.unique(all_sid,return_counts=True)
        ax.bar(u,c,color=[PAL[int(s)%len(PAL)] for s in u],alpha=0.8)
        ax.set_xlabel('Skill ID'); ax.set_ylabel('Steps')
        ax.set_title('Skill Usage',fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[0,1])
    if chunk_changes:
        cc=np.array(chunk_changes); w=min(50,max(1,len(cc)//10))
        ax.plot(np.convolve(cc,np.ones(w)/w,'valid'),color='#E53935',lw=1.5)
        ax.axhline(cr,color='k',ls='--',lw=1,label=f'mean={cr:.3f}')
        ax.set_title('Rolling Skill Change Rate',fontweight='bold'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[0,2])
    if all_adiff:
        ax.hist(all_adiff,bins=40,color='#1E88E5',alpha=0.8,edgecolor='white')
        ax.axvline(ma,color='#E53935',ls='--',lw=1.5,label=f'mean={ma:.3f}')
        ax.set_title('Action Chunk L2 Discontinuity',fontweight='bold'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[1,:2])
    if sid_durations:
        data=[sid_durations[k] for k in sorted(sid_durations)]
        labels=[f'Skill {k}' for k in sorted(sid_durations)]
        bp=ax.boxplot(data,labels=labels,patch_artist=True)
        for patch,col in zip(bp['boxes'],PAL): patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_ylabel('Duration (steps)'); ax.set_title('Skill Duration',fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[1,2]); ax.axis('off')
    ax.text(0.05,0.95,"\n".join(lines[:8]),transform=ax.transAxes,fontsize=8,va='top',
            fontfamily='monospace',bbox=dict(boxstyle='round',facecolor='#f5f5f5',alpha=0.8))
    fig.suptitle('Skill Switching & Action Continuity',fontsize=13,fontweight='bold')
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(out_path,dpi=130,bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")
    return {'change_rate':cr,'mean_adiff':ma,'text':"\n".join(lines)}


@torch.no_grad()
def analyze_action_distribution(results_with_extra, out_path):
    all_actions=[]; chunk_std=[]
    for res in results_with_extra:
        acts=np.array(res['actions']); H_lo=res.get('H_lo',4)
        all_actions.append(acts)
        for t in range(0,len(acts)-H_lo+1,H_lo):
            chunk_std.append(acts[t:t+H_lo].std(axis=0))
    if not all_actions: return {}
    all_acts=np.concatenate(all_actions,axis=0)
    chunk_std=np.array(chunk_std) if chunk_std else np.zeros((1,9))
    lines=["=== Action Distribution ===",f"Total steps: {len(all_acts)}","","Per-joint:"]
    for j in range(9):
        lines.append(f"  J{j}: mean={all_acts[:,j].mean():.3f}  std={all_acts[:,j].std():.3f}  "
                     f"[{all_acts[:,j].min():.2f},{all_acts[:,j].max():.2f}]")
    lines+=["","Intra-chunk std:"]
    for j in range(9): lines.append(f"  J{j}: {chunk_std[:,j].mean():.4f}")
    print("\n".join(lines))
    PAL=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1','#FFB300','#607D8B','#795548']
    fig,axes=plt.subplots(3,3,figsize=(15,12)); axes=axes.flatten()
    for j in range(9):
        ax=axes[j]; m,s=all_acts[:,j].mean(),all_acts[:,j].std()
        ax.hist(all_acts[:,j],bins=50,color=PAL[j],alpha=0.8,edgecolor='white',density=True)
        ax.axvline(m,color='k',ls='--',lw=1.5,label=f'μ={m:.3f}')
        ax.axvspan(m-s,m+s,alpha=0.15,color=PAL[j])
        ax.set_title(f'Joint {j}  σ={s:.3f}',fontsize=9,fontweight='bold')
        ax.set_xlabel('Action [-1,1]'); ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Action Distribution per Joint',fontsize=13,fontweight='bold')
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(out_path,dpi=130,bbox_inches='tight'); plt.close()
    fig2,ax2=plt.subplots(figsize=(12,5))
    ax2.bar(range(9),chunk_std.mean(axis=0),color=PAL,alpha=0.8)
    ax2.set_xticks(range(9)); ax2.set_xticklabels([f'J{i}' for i in range(9)])
    ax2.set_ylabel('Mean intra-chunk std')
    ax2.set_title('Action Consistency Within Chunk (high=oscillation)',fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path.replace('.png','_chunk_std.png'),dpi=130,bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")
    return {'mean_per_joint':all_acts.mean(axis=0).tolist(),
            'std_per_joint':all_acts.std(axis=0).tolist(),'text':"\n".join(lines)}


@torch.no_grad()
def analyze_reward_distribution(results_with_extra, wm, out_path, device):
    dev=torch.device(device); model=wm.model; model.eval()
    all_re=[]; all_ra=[]; all_rv=[]; all_t=[]
    for res in results_with_extra:
        z_seq=np.array(res['z_seq']); r_env=np.array(res['rewards'])
        T=len(z_seq)
        if T==0: continue
        z_t=torch.FloatTensor(z_seq).to(dev)
        r_event_v=np.zeros(T)
        if model.cfg.use_reward_head:
            if hasattr(model.decoder,'head_reward'): logits=model.decoder.head_reward(z_t)
            elif hasattr(model,'reward_head'): logits=model.reward_head(z_t)
            else: logits=None
            if logits is not None: r_event_v=torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        r_acc_v=np.zeros(T)
        if wm.cat_head is not None: r_acc_v=wm.cat_head.expected_reward(z_t).cpu().numpy()
        all_re.extend(r_event_v.tolist()); all_ra.extend(r_acc_v.tolist())
        all_rv.extend(r_env[:T].tolist()); all_t.extend((np.arange(T)/max(T-1,1)).tolist())
    all_re=np.array(all_re); all_ra=np.array(all_ra)
    all_rv=np.array(all_rv); all_t=np.array(all_t)
    lines=[
        "=== Reward Head Distribution ===","",
        f"Event (BCE):  mean={all_re.mean():.4f}  std={all_re.std():.4f}  >0.5: {(all_re>0.5).mean()*100:.1f}%",
        f"Accum (cat):  mean={all_ra.mean():.4f}  std={all_ra.std():.4f}  max={all_ra.max():.2f}",
        f"Env (actual): mean={all_rv.mean():.4f}  >0.5: {(all_rv>0.5).mean()*100:.1f}%","",
        f"r_event sat: {'HIGH->unreliable' if (all_re>0.5).mean()>0.3 else 'OK'}",
        f"r_acc range: {'SATURATED' if all_ra.mean()>3.0 else 'OK' if all_ra.mean()<1.0 else 'MODERATE'}",
        f"r_env sparse: {(all_rv>0.5).mean()*100:.2f}% steps with reward",
    ]
    print("\n".join(lines))
    PAL=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1']
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for ax,data,title,col,xl in zip(
        axes.flatten()[:3],
        [all_re,all_ra,all_rv],
        ['Event Reward (BCE)','Accum Reward (Cat) E[R|z]','Actual Env Reward'],
        PAL[:3],['r_event','r_acc [0,4]','r_env']):
        ax.hist(data,bins=50,color=col,alpha=0.8,edgecolor='white')
        ax.axvline(data.mean(),color='k',ls='--',lw=1.5,label=f'μ={data.mean():.3f}')
        ax.set_title(title,fontweight='bold'); ax.set_xlabel(xl); ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    bins=np.linspace(0,1,21); bc=0.5*(bins[:-1]+bins[1:])
    for ax_i,(data,col,title) in enumerate(zip(
        [all_re,all_ra],PAL[:2],['Event over Episode','Accum over Episode'])):
        ax=axes[1,ax_i]; bm,bs=[],[]
        for i in range(len(bins)-1):
            mask=(all_t>=bins[i])&(all_t<bins[i+1])
            bm.append(data[mask].mean() if mask.any() else 0)
            bs.append(data[mask].std()  if mask.any() else 0)
        bm,bs=np.array(bm),np.array(bs)
        ax.plot(bc,bm,color=col,lw=1.5)
        ax.fill_between(bc,(bm-bs).clip(0),(bm+bs),alpha=0.2,color=col)
        ax.set_xlabel('Normalized time'); ax.set_title(title,fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    ax=axes[1,2]; ax.axis('off')
    ax.text(0.05,0.95,"\n".join(lines),transform=ax.transAxes,fontsize=8,va='top',
            fontfamily='monospace',bbox=dict(boxstyle='round',facecolor='#f5f5f5',alpha=0.8))
    fig.suptitle('Reward Distribution (from sim z_t)',fontsize=13,fontweight='bold')
    plt.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(out_path,dpi=130,bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path}")
    return {'r_event_mean':all_re.mean(),'r_acc_mean':all_ra.mean(),
            'r_env_mean':all_rv.mean(),'text':"\n".join(lines)}

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