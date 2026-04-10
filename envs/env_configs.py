"""
env_configs.py — KODAQ RSSM-Koopman environment configurations.

KODAQ §1.1:
  x_dim = 2108  (dim_delta_e=2048, dim_delta_p=42, dim_q=9, dim_qdot=9)

  Kitchen (primary target):
    action_dim  = 9     (joint velocity targets)
    state_dim   = 60    (raw D4RL observation; used for GRU auxiliary info)
    koopman_dim = 128   (lifted state d_o; larger than KODAC-S to handle 2108-dim input)
    gru_hidden  = 256   (h_t)
    num_skills  = 8     (K, from EXTRACT)

  Adroit (secondary):
    state_dim varies per task
    x_dim = 60 in state-only mode (no R3M), else 2108

Note: state_dim kept for backward compatibility with env_configs usage.
      x_dim is computed from KoopmanCVAEConfig.x_dim property.
"""

from models.koopman_cvae import KoopmanCVAEConfig

# ── Shared R3M input dims ────────────────────────────────────────────────────
# These match extract_skill_label.py constants; do NOT change independently.
DIM_DELTA_E = 2048
DIM_DELTA_P = 42
DIM_Q_KITCHEN   = 9
DIM_QDOT_KITCHEN = 9
X_DIM_KITCHEN = DIM_DELTA_E + DIM_DELTA_P + DIM_Q_KITCHEN + DIM_QDOT_KITCHEN  # 2108

ENV_CONFIGS = {
    # ── Franka Kitchen (primary KODAQ target) ──────────────────────────────
    # x_t ∈ ℝ^{2108}: R3M(2048) + obj_diff(42) + qpos(9) + qvel(9)
    # action: 9-dim joint velocity  dt: 0.08s (12.5Hz)
    'kitchen_complete': {
        'dim_delta_e':  DIM_DELTA_E,
        'dim_delta_p':  DIM_DELTA_P,
        'dim_q':        DIM_Q_KITCHEN,
        'dim_qdot':     DIM_QDOT_KITCHEN,
        'action_dim':   9,
        'state_dim':    60,      # raw obs for reference
        'koopman_dim':  128,     # d_o: larger for high-dim x_t
        'gru_hidden':   256,     # d_h
        'action_latent': 64,     # d_u
        'num_skills':   8,       # K (EXTRACT)
        'mlp_hidden':   512,     # wider for 2108-dim input
        'enc_layers':   4,
        'dec_layers':   4,
        'dropout':      0.1,
        # Loss weights (§4)
        'lambda1':      1.0,     # L_dyn
        'lambda2':      0.5,     # L_skill
        'lambda3':      0.1,     # L_reg
        'lambda4':      0.01,    # L_stab
        # Reconstruction head weights α_j
        # Δq_t는 episode-first diff → 스케일 작고 안정 → weight 높여도 됨
        # q̇_t는 velocity라 noisy → weight 낮춤
        'alpha_delta_e': 1.0,
        'alpha_delta_p': 2.0,
        'alpha_q':       2.0,   # 절대값 q_t(1.0) → Δq_t는 더 신뢰 가능
        'alpha_qdot':    0.2,   # velocity noisy → 0.5 → 0.2로 낮춤
    },
    'kitchen_partial': {
        'dim_delta_e':  DIM_DELTA_E,
        'dim_delta_p':  DIM_DELTA_P,
        'dim_q':        DIM_Q_KITCHEN,
        'dim_qdot':     DIM_QDOT_KITCHEN,
        'action_dim':   9,
        'state_dim':    60,
        'koopman_dim':  128,
        'gru_hidden':   256,
        'action_latent': 64,
        'num_skills':   8,
        'mlp_hidden':   512,
        'enc_layers':   4,
        'dec_layers':   4,
        'dropout':      0.1,
        'lambda1':      1.0,
        'lambda2':      0.5,
        'lambda3':      0.1,
        'lambda4':      0.01,
        'alpha_delta_e': 1.0,
        'alpha_delta_p': 2.0,
        'alpha_q':       2.0,
        'alpha_qdot':    0.2,
    },
    'kitchen_mixed': {
        'dim_delta_e':  DIM_DELTA_E,
        'dim_delta_p':  DIM_DELTA_P,
        'dim_q':        DIM_Q_KITCHEN,
        'dim_qdot':     DIM_QDOT_KITCHEN,
        'action_dim':   9,
        'state_dim':    60,
        'koopman_dim':  128,
        'gru_hidden':   256,
        'action_latent': 64,
        'num_skills':   8,
        'mlp_hidden':   512,
        'enc_layers':   4,
        'dec_layers':   4,
        'dropout':      0.1,
        'lambda1':      1.0,
        'lambda2':      0.5,
        'lambda3':      0.1,
        'lambda4':      0.01,
        'alpha_delta_e': 1.0,
        'alpha_delta_p': 2.0,
        'alpha_q':       2.0,
        'alpha_qdot':    0.2,
    },

    # ── Adroit (state-only mode: Δe_t = 0, x_dim = 2108 with zero R3M block) ─
    # In state-only mode: x_t = [0(2048), Δp_t, q_t, q̇_t]
    # For Adroit: Δp_t = obj_state_diff (dim varies by task)
    # We keep the same x_dim=2108 interface for model compatibility,
    # padding with zeros where R3M is absent.
    'adroit_pen': {
        'dim_delta_e':  DIM_DELTA_E,
        'dim_delta_p':  DIM_DELTA_P,   # padded to 42
        'dim_q':        DIM_Q_KITCHEN,
        'dim_qdot':     DIM_QDOT_KITCHEN,
        'action_dim':   24,
        'state_dim':    45,
        'koopman_dim':  64,
        'gru_hidden':   128,
        'action_latent': 32,
        'num_skills':   4,
        'mlp_hidden':   256,
        'enc_layers':   3,
        'dec_layers':   3,
        'dropout':      0.1,
        'lambda1':      1.0,
        'lambda2':      0.3,
        'lambda3':      0.05,
        'lambda4':      0.01,
        'alpha_delta_e': 0.0,    # no R3M
        'alpha_delta_p': 1.0,
        'alpha_q':       1.0,
        'alpha_qdot':    0.5,
    },

    # ── Synthetic (quick test) ──────────────────────────────────────────────
    'synthetic': {
        'dim_delta_e':  DIM_DELTA_E,
        'dim_delta_p':  DIM_DELTA_P,
        'dim_q':        DIM_Q_KITCHEN,
        'dim_qdot':     DIM_QDOT_KITCHEN,
        'action_dim':   9,
        'state_dim':    60,
        'koopman_dim':  32,
        'gru_hidden':   64,
        'action_latent': 16,
        'num_skills':   4,
        'mlp_hidden':   128,
        'enc_layers':   2,
        'dec_layers':   2,
        'dropout':      0.0,
        'lambda1':      1.0,
        'lambda2':      0.5,
        'lambda3':      0.1,
        'lambda4':      0.01,
        'alpha_delta_e': 1.0,
        'alpha_delta_p': 1.0,
        'alpha_q':       1.0,
        'alpha_qdot':    1.0,
    },
}

KITCHEN_ENVS = ['kitchen_complete', 'kitchen_partial', 'kitchen_mixed']
ADROIT_ENVS  = ['adroit_pen', 'adroit_hammer', 'adroit_door', 'adroit_relocate']


def build_config(args) -> KoopmanCVAEConfig:
    """
    Build KoopmanCVAEConfig from parsed args + ENV_CONFIGS.
    Args override env defaults for any non-None fields.
    """
    env_key    = getattr(args, 'env', 'synthetic')
    cfg_kwargs = dict(ENV_CONFIGS.get(env_key, ENV_CONFIGS['synthetic']))

    # Allow CLI overrides for all config fields
    override_keys = [
        'dim_delta_e', 'dim_delta_p', 'dim_q', 'dim_qdot',
        'action_dim', 'state_dim',
        'koopman_dim', 'gru_hidden', 'action_latent', 'num_skills',
        'mlp_hidden', 'enc_layers', 'dec_layers', 'dropout',
        'lambda1', 'lambda2', 'lambda3', 'lambda4',
        'alpha_delta_e', 'alpha_delta_p', 'alpha_q', 'alpha_qdot',
        'phase', 'dyn_horizon', 'dyn_alpha',
    ]
    for key in override_keys:
        val = getattr(args, key, None)
        if val is not None:
            cfg_kwargs[key] = val

    # --no_multistep_dyn flag
    if getattr(args, 'no_multistep_dyn', False):
        cfg_kwargs['multistep_dyn'] = False

    return KoopmanCVAEConfig(**cfg_kwargs)