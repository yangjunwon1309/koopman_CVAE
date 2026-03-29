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
        # ── Encoder ──────────────────────────────────────────
        # HELIOS 원본: GRU + action-only
        # action-only: DPM이 state가 아닌 action pattern을 클러스터링
        encoder_type  = 'gru',
        encoder_input = 'action',
        gru_hidden    = 128,
        gru_layers    = 2,
        # TCN 대안 (encoder_type='tcn'으로 변경 시)
        tcn_hidden  = 128,
        tcn_layers  = 5,
        tcn_kernel  = 3,
        dropout     = 0.1,
        # Skill latent
        skill_dim   = 32,
        skill_horizon = 10,
        # DPM
        alpha         = 2.0,    # higher → more clusters expected
        K_init        = 1,
        K_max         = 20,
        kappa0        = 0.1,    # prior 영향 줄임
        nu0_delta     = 2.0,    # nu0 = d + 2 (minimum valid)
        psi_scale     = 2.0,    # broad initial cov: winner-take-all 방지
        birth_thresh  = 0.3,    # K>1 only; K=1 uses Mahalanobis
        birth_min_pts = 10,
        birth_K_fresh = 4,      # Hughes&Sudderth: 4 sub-clusters per birth
        birth_start_epoch = 8,  # encoder 안정화 후 birth 시작
        merge_cos     = 0.90,
        # Loss weights
        # zeta1 dominant: reconstruction teaches encoder first
        # zeta2 very small early: DPM noisy, don't let it dominate
        zeta1 = 1.0,
        zeta2 = 0.05,
        zeta3 = 0.01,
        # Anti-collapse: z spread 강제
        # spr=0.0이면 z가 collapse → birth 후 act=1 지속 원인
        # 2-Phase: pretrain_epochs 동안 β-VAE만, 그 후 DPM
        pretrain_epochs   = 25,
        zeta_vae_pretrain = 1.0,   # Phase 1: 강한 VAE (β=1)
        # Phase 2 weights
        zeta_spread = 0.5,
        zeta_vae    = 0.5,
        min_z_std   = 0.5,
        birth_warmup_steps = 5,
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
        # SPiRL/HELIOS 방식:
        # skill_horizon=10 step 단위로 잘라서 각각이 하나의 skill sample
        # stride=1: 136937 steps → ~136k samples (DPM에 충분한 데이터)
        # stride=5: ~27k samples (빠른 실험용)
        dataset = load_d4rl_trajectories(
            'kitchen_mixed',
            seq_len=cfg.skill_horizon,   # 10 steps = 1 skill sample
            stride=5,                    # 27k samples (속도/품질 균형)
            min_episode_len=cfg.skill_horizon,
        )
        print(f"  Dataset size: {len(dataset)}  "
              f"(seq_len={cfg.skill_horizon}, stride=5)")
    except Exception as e:
        print(f"D4RL load failed ({e}), using synthetic dataset")
        dataset = make_synthetic_dataset(
            action_dim=9, state_dim=60,
            n_samples=5000, seq_len=cfg.skill_horizon,
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
             labels_hard=labels_hard,   # (N,)     int32   — one label per skill seq
             labels_soft=labels_soft,   # (N, K)   float32
             z_all=z_all,               # (N, d_z) float32
             K=np.array([trainer.dpm.K]),
             skill_horizon=np.array([cfg.skill_horizon]))
    print(f"Labels saved: {save_path}")
    print(f"  labels_hard: {labels_hard.shape}  "
          f"K={trainer.dpm.K}  "
          f"unique={np.unique(labels_hard).tolist()}")
    print(f"  label counts: "
          + str({int(k): int((labels_hard==k).sum())
                 for k in np.unique(labels_hard)}))


if __name__ == '__main__':
    main()