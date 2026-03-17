"""
Environment-specific configurations for Koopman CVAE.
Patch size is chosen so that dt_patch ≈ 100ms across all environments.
"""
from models.koopman_cvae import KoopmanCVAEConfig

ENV_CONFIGS = {
    # ── DMControl (50 Hz, dt=0.02s) ─────────────────────────────
    'dm_reacher':    dict(action_dim=2,  state_dim=11, dt_control=0.02,   patch_size=5),
    'dm_walker':     dict(action_dim=6,  state_dim=24, dt_control=0.02,   patch_size=5),
    'dm_cheetah':    dict(action_dim=6,  state_dim=17, dt_control=0.02,   patch_size=5),
    'dm_cartpole':   dict(action_dim=1,  state_dim=5,  dt_control=0.02,   patch_size=5),
    'dm_humanoid':   dict(action_dim=21, state_dim=67, dt_control=0.02,   patch_size=5),
    'dm_ball_cup':   dict(action_dim=2,  state_dim=8,  dt_control=0.02,   patch_size=5),

    # ── D4RL Adroit Hand (25 Hz, dt=0.04s) ──────────────────────
    'adroit_pen':      dict(action_dim=24, state_dim=45, dt_control=0.04, patch_size=3),
    'adroit_hammer':   dict(action_dim=26, state_dim=46, dt_control=0.04, patch_size=3),
    'adroit_door':     dict(action_dim=28, state_dim=39, dt_control=0.04, patch_size=3),
    'adroit_relocate': dict(action_dim=30, state_dim=39, dt_control=0.04, patch_size=3),

    # ── HumanoidBench (100 Hz, dt=0.01s) ────────────────────────
    'humanoid_stand':  dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_walk':   dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_run':    dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_reach':  dict(action_dim=19, state_dim=132, dt_control=0.01, patch_size=10),

    # ── Isaac Gym (60 Hz, dt=0.0167s) ───────────────────────────
    'isaac_franka':    dict(action_dim=7,  state_dim=23,  dt_control=0.0167, patch_size=6),
    'isaac_allegro':   dict(action_dim=16, state_dim=92,  dt_control=0.0167, patch_size=6),
    'isaac_humanoid':  dict(action_dim=21, state_dim=108, dt_control=0.0167, patch_size=6),
}

# D4RL gym env name mapping
D4RL_ENV_MAP = {
    'adroit_pen':      ['pen-expert-v1',      'pen-medium-v1',      'pen-random-v1'],
    'adroit_hammer':   ['hammer-expert-v1',   'hammer-medium-v1',   'hammer-random-v1'],
    'adroit_door':     ['door-expert-v1',     'door-medium-v1',     'door-random-v1'],
    'adroit_relocate': ['relocate-expert-v1', 'relocate-medium-v1', 'relocate-random-v1'],
}


def build_config(args) -> KoopmanCVAEConfig:
    """Build KoopmanCVAEConfig from args, with env preset override."""
    env_cfg = ENV_CONFIGS.get(args.env, {})
    return KoopmanCVAEConfig(
        action_dim       = env_cfg.get('action_dim',      args.action_dim),
        state_dim        = env_cfg.get('state_dim',       args.state_dim),
        patch_size       = env_cfg.get('patch_size',      args.patch_size),
        dt_control       = env_cfg.get('dt_control',      args.dt_control),
        embed_dim        = args.embed_dim,
        state_embed_dim  = args.state_embed_dim,
        gru_hidden_dim   = args.gru_hidden_dim,
        mlp_hidden_dim   = args.mlp_hidden_dim,
        koopman_dim      = args.koopman_dim,
        beta_kl          = args.beta_kl,
        alpha_pred       = args.alpha_pred,
        gamma_eig        = args.gamma_eig,
        delta_cst        = args.delta_cst,
        dropout          = args.dropout,
    )