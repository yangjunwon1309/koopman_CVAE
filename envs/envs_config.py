"""
Environment-specific configurations for Koopman CVAE.
Patch size is chosen so that dt_patch ≈ 100ms across all environments.
"""

from models.koopman_cvae import KoopmanCVAEConfig

ENV_CONFIGS = {
    # ── DMControl (50 Hz, dt=0.02s) ─────────────────────────────
    'dm_reacher':  dict(action_dim=2,  state_dim=11,  dt_control=0.02, patch_size=5),
    'dm_walker':   dict(action_dim=6,  state_dim=24,  dt_control=0.02, patch_size=5),
    'dm_cheetah':  dict(action_dim=6,  state_dim=17,  dt_control=0.02, patch_size=5),
    'dm_cartpole': dict(action_dim=1,  state_dim=5,   dt_control=0.02, patch_size=5),
    'dm_humanoid': dict(action_dim=21, state_dim=67,  dt_control=0.02, patch_size=5),
    'dm_ball_cup': dict(action_dim=2,  state_dim=8,   dt_control=0.02, patch_size=5),

    # ── D4RL Adroit Hand (25 Hz, dt=0.04s) ──────────────────────
    'adroit_pen':      dict(action_dim=24, state_dim=45, dt_control=0.04, patch_size=3),
    'adroit_hammer':   dict(action_dim=26, state_dim=46, dt_control=0.04, patch_size=3),
    'adroit_door':     dict(action_dim=28, state_dim=39, dt_control=0.04, patch_size=3),
    'adroit_relocate': dict(action_dim=30, state_dim=39, dt_control=0.04, patch_size=3),

    # ── HumanoidBench (100 Hz, dt=0.01s) ────────────────────────
    'humanoid_stand': dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_walk':  dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_run':   dict(action_dim=19, state_dim=76,  dt_control=0.01, patch_size=10),
    'humanoid_reach': dict(action_dim=19, state_dim=132, dt_control=0.01, patch_size=10),

    # ── Isaac Gym (60 Hz, dt=0.0167s) ───────────────────────────
    'isaac_franka':   dict(action_dim=7,  state_dim=23,  dt_control=0.0167, patch_size=6),
    'isaac_allegro':  dict(action_dim=16, state_dim=92,  dt_control=0.0167, patch_size=6),
    'isaac_humanoid': dict(action_dim=21, state_dim=108, dt_control=0.0167, patch_size=6),
}

# ── D4RL gym env name mapping ────────────────────────────────────
# quality keys: 'expert', 'human', 'cloned', 'medium', 'random'
# Not all qualities exist for every task — only listed ones are valid.

D4RL_ENV_MAP = {
    'adroit_pen': {
        'expert': 'pen-expert-v1',
        'human':  'pen-human-v1',
        'cloned': 'pen-cloned-v1',
    },
    'adroit_hammer': {
        'expert': 'hammer-expert-v1',
        'human':  'hammer-human-v1',
        'cloned': 'hammer-cloned-v1',
    },
    'adroit_door': {
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

# Valid quality strings per env (for validation in dataset_utils)
D4RL_VALID_QUALITIES = {
    env: list(qualities.keys())
    for env, qualities in D4RL_ENV_MAP.items()
}


def get_d4rl_env_name(env_name: str, quality: str) -> str:
    """
    Resolve shorthand env + quality → D4RL gym env name.

    Args:
        env_name: e.g. 'adroit_pen'
        quality:  e.g. 'human', 'expert', 'cloned'

    Returns:
        D4RL gym env name, e.g. 'pen-human-v1'

    Raises:
        KeyError if env_name or quality not found.
    """
    if env_name not in D4RL_ENV_MAP:
        raise KeyError(
            f"Unknown env '{env_name}'. "
            f"Available: {list(D4RL_ENV_MAP.keys())}"
        )
    quality_map = D4RL_ENV_MAP[env_name]
    if quality not in quality_map:
        raise KeyError(
            f"Quality '{quality}' not available for '{env_name}'. "
            f"Available qualities: {list(quality_map.keys())}"
        )
    return quality_map[quality]


def build_config(args) -> KoopmanCVAEConfig:
    """Build KoopmanCVAEConfig from args, with env preset override."""
    env_cfg = ENV_CONFIGS.get(args.env, {})
    return KoopmanCVAEConfig(
        action_dim      = env_cfg.get('action_dim',  getattr(args, 'action_dim',  6)),
        state_dim       = env_cfg.get('state_dim',   getattr(args, 'state_dim',   24)),
        patch_size      = env_cfg.get('patch_size',  getattr(args, 'patch_size',  5)),
        dt_control      = env_cfg.get('dt_control',  getattr(args, 'dt_control',  0.02)),
        embed_dim       = getattr(args, 'embed_dim',       128),
        state_embed_dim = getattr(args, 'state_embed_dim', 64),
        gru_hidden_dim  = getattr(args, 'gru_hidden_dim',  256),
        mlp_hidden_dim  = getattr(args, 'mlp_hidden_dim',  256),
        koopman_dim     = getattr(args, 'koopman_dim',     64),
        kl_prior        = getattr(args, 'kl_prior',        'koopman'),
        beta_kl         = getattr(args, 'beta_kl',         0.1),
        alpha_pred      = getattr(args, 'alpha_pred',      1.0),
        gamma_eig       = getattr(args, 'gamma_eig',       0.1),
        delta_cst       = getattr(args, 'delta_cst',       1.0),
        pred_steps      = getattr(args, 'pred_steps',      5),
        dropout         = getattr(args, 'dropout',         0.1),
        temp_contrastive= getattr(args, 'temp_contrastive',0.1),
        delta_pos       = getattr(args, 'delta_pos',       2),
        delta_neg       = getattr(args, 'delta_neg',       4),
    )