"""
env_configs.py — environment configurations for KODAC.
Maps environment names to KoopmanCVAEConfig parameters.
"""

import math
from models.koopman_cvae import KoopmanCVAEConfig

# ─────────────────────────────────────────────────────────────
# Environment registry
# ─────────────────────────────────────────────────────────────

ENV_CONFIGS = {
    # ── D4RL Adroit ──────────────────────────────────────────
    'adroit_pen': {
        'action_dim':  24,
        'state_dim':   45,
        'num_skills':  6,
        'koopman_dim': 64,
        'patch_size':  5,
        'dt_control':  0.02,
    },
    'adroit_hammer': {
        'action_dim':  26,
        'state_dim':   46,
        'num_skills':  6,
        'koopman_dim': 64,
        'patch_size':  5,
        'dt_control':  0.02,
    },
    'adroit_door': {
        'action_dim':  28,
        'state_dim':   39,
        'num_skills':  6,
        'koopman_dim': 64,
        'patch_size':  5,
        'dt_control':  0.02,
    },
    'adroit_relocate': {
        'action_dim':  30,
        'state_dim':   39,
        'num_skills':  8,
        'koopman_dim': 64,
        'patch_size':  5,
        'dt_control':  0.02,
    },

    # ── DMControl (synthetic fallback) ───────────────────────
    'dm_walker': {
        'action_dim':  6,
        'state_dim':   24,
        'num_skills':  4,
        'koopman_dim': 32,
        'patch_size':  5,
        'dt_control':  0.025,
    },
    'dm_cheetah': {
        'action_dim':  6,
        'state_dim':   17,
        'num_skills':  4,
        'koopman_dim': 32,
        'patch_size':  5,
        'dt_control':  0.025,
    },

    # ── Synthetic / default ───────────────────────────────────
    'synthetic': {
        'action_dim':  6,
        'state_dim':   24,
        'num_skills':  4,
        'koopman_dim': 32,
        'patch_size':  5,
        'dt_control':  0.02,
    },
}


def build_config(args) -> KoopmanCVAEConfig:
    """
    Build KoopmanCVAEConfig from parsed args.

    Priority:
        1. Args explicitly passed on CLI
        2. ENV_CONFIGS defaults for the environment
        3. KoopmanCVAEConfig dataclass defaults
    """
    env_key = getattr(args, 'env', 'synthetic')
    env_defaults = ENV_CONFIGS.get(env_key, ENV_CONFIGS['synthetic'])

    # Start from env defaults
    cfg_kwargs = dict(env_defaults)

    # Override with explicit CLI args (non-None values).
    # Keys must exactly match KoopmanCVAEConfig field names.
    for key in [
        # environment
        'action_dim', 'state_dim', 'patch_size', 'dt_control',
        # architecture
        'mlp_hidden_dim', 'gru_hidden_dim', 'embed_dim',
        'koopman_dim', 'num_skills', 'lora_rank', 'dropout',
        # eigenvalue
        'mu_fixed', 'omega_max',
        # loss weights
        'kl_prior', 'pred_steps',
        'alpha_pred', 'alpha_recon_s', 'alpha_recon_a',
        'beta_kl', 'gamma_eig',
        'delta_cst', 'delta_div', 'delta_ent', 'delta_decorr',
        'temp_contrastive', 'freq_repulsion_sigma',
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg_kwargs[key] = val

    return KoopmanCVAEConfig(**cfg_kwargs)