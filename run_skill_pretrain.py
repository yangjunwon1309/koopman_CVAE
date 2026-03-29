"""
run_skill_pretrain.py
=====================
Kitchen mixed 데이터로 TCN-DPM skill pretraining 실행.

서버 실행:
    cd ~/koopman_CVAE
    python run_skill_pretrain.py

결과:
    checkpoints/skill_pretrain/best.pt   — 최적 체크포인트
    checkpoints/skill_pretrain/final.pt  — 최종 체크포인트
    checkpoints/skill_pretrain/labels.npz — skill label 배열
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/koopman_CVAE'))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models.skill_pretrain import SkillPretrainer, SkillPretrainConfig
from data.dataset_utils import load_d4rl_trajectories, make_synthetic_dataset


def main():
    cfg = SkillPretrainConfig(
        # Kitchen
        state_dim   = 60,
        action_dim  = 9,
        # TCN
        tcn_hidden  = 256,
        tcn_layers  = 5,     # RF=63 steps
        tcn_kernel  = 3,
        dropout     = 0.1,
        # Skill latent
        skill_dim   = 32,
        skill_horizon = 20,
        # DPM
        alpha         = 1.0,
        K_init        = 1,
        K_max         = 20,
        kappa0        = 1.0,
        nu0_delta     = 2.0,
        psi_scale     = 1.0,
        birth_thresh  = 0.05,
        birth_min_pts = 10,
        merge_cos     = 0.95,
        # Loss weights
        zeta1 = 1.0,
        zeta2 = 0.5,
        zeta3 = 0.1,
        # Training
        epochs     = 100,
        batch_size = 64,
        lr         = 3e-4,
        device     = 'cuda',
        save_dir   = 'checkpoints/skill_pretrain',
    )

    # ── 데이터 로딩 ───────────────────────────────────────────
    print("Loading kitchen_mixed ...")
    try:
        dataset = load_d4rl_trajectories(
            'kitchen_mixed',
            seq_len=200,
            stride=50,          # 50-step sliding window
            min_episode_len=100,
        )
        print(f"  Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"D4RL load failed ({e}), using synthetic dataset")
        dataset = make_synthetic_dataset(
            action_dim=9, state_dim=60,
            n_samples=2000, seq_len=200,
        )

    # Train / val split (90 / 10)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── 학습 ─────────────────────────────────────────────────
    trainer = SkillPretrainer(cfg)
    trainer.train(train_loader)

    # ── Skill label 생성 및 저장 ──────────────────────────────
    print("\nAssigning skill labels to full dataset ...")
    full_loader = DataLoader(
        dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=2,
    )

    # Hard labels (Koopman CVAE 학습용)
    labels_hard, z_all = trainer.assign_skill_labels(
        full_loader, hard=True)

    # Soft labels (분석용)
    labels_soft, _ = trainer.assign_skill_labels(
        full_loader, hard=False)

    save_path = os.path.join(cfg.save_dir, 'labels.npz')
    np.savez(save_path,
             labels_hard=labels_hard,   # (N, T)      int32
             labels_soft=labels_soft,   # (N, T, K)   float32
             z_all=z_all,               # (N, T, d_z) float32
             K=np.array([trainer.dpm.K]))
    print(f"Labels saved: {save_path}")
    print(f"  labels_hard: {labels_hard.shape}  "
          f"K={trainer.dpm.K}  "
          f"unique={np.unique(labels_hard).tolist()}")


if __name__ == '__main__':
    main()