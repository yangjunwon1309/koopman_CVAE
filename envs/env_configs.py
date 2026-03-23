"""
env_configs.py — KODAC-S environment configurations.
"""
from models.koopman_cvae import KoopmanCVAEConfig

ENV_CONFIGS = {
    # ── Adroit ───────────────────────────────────────────────
    'adroit_pen':      {'action_dim': 24, 'state_dim': 45, 'koopman_dim': 64,
                        'patch_size': 5, 'dt_control': 0.02},
    'adroit_hammer':   {'action_dim': 26, 'state_dim': 46, 'koopman_dim': 64,
                        'patch_size': 5, 'dt_control': 0.02},
    'adroit_door':     {'action_dim': 28, 'state_dim': 39, 'koopman_dim': 64,
                        'patch_size': 5, 'dt_control': 0.02},
    'adroit_relocate': {'action_dim': 30, 'state_dim': 39, 'koopman_dim': 64,
                        'patch_size': 5, 'dt_control': 0.02},

    # ── Franka Kitchen ───────────────────────────────────────
    # obs: 60-dim (qpos 9 + qvel 9 + object 42)
    # act: 9-dim joint velocity
    # dt:  0.08s (12.5Hz)
    # patch_size=1 recommended: each step is already ~80ms,
    #   no need for additional temporal aggregation
    'kitchen_complete': {'action_dim':  9, 'state_dim': 60, 'koopman_dim': 64,
                         'patch_size': 1, 'dt_control': 0.08,
                         'tcn_hidden_dim': 256, 'tcn_n_layers': 5,
                         'num_heads': 8,  'lora_rank': 8},
    'kitchen_partial':  {'action_dim':  9, 'state_dim': 60, 'koopman_dim': 64,
                         'patch_size': 1, 'dt_control': 0.08,
                         'tcn_hidden_dim': 256, 'tcn_n_layers': 5,
                         'num_heads': 8,  'lora_rank': 8},
    'kitchen_mixed':    {'action_dim':  9, 'state_dim': 60, 'koopman_dim': 64,
                         'patch_size': 1, 'dt_control': 0.08,
                         'tcn_hidden_dim': 256, 'tcn_n_layers': 5,
                         'num_heads': 8,  'lora_rank': 8},

    # ── DMControl ────────────────────────────────────────────
    'dm_walker':  {'action_dim':  6, 'state_dim': 24, 'koopman_dim': 32,
                   'patch_size': 5, 'dt_control': 0.025},
    'dm_cheetah': {'action_dim':  6, 'state_dim': 17, 'koopman_dim': 32,
                   'patch_size': 5, 'dt_control': 0.025},

    # ── Synthetic ────────────────────────────────────────────
    'synthetic':  {'action_dim':  9, 'state_dim': 60, 'koopman_dim': 32,
                   'patch_size': 1, 'dt_control': 0.08},
}

KITCHEN_ENVS = ['kitchen_complete', 'kitchen_partial', 'kitchen_mixed']
ADROIT_ENVS  = ['adroit_pen', 'adroit_hammer', 'adroit_door', 'adroit_relocate']

def build_config(args) -> KoopmanCVAEConfig:
    env_key    = getattr(args, 'env', 'synthetic')
    cfg_kwargs = dict(ENV_CONFIGS.get(env_key, ENV_CONFIGS['synthetic']))
    for key in [
        'action_dim', 'state_dim', 'patch_size', 'dt_control',
        'mlp_hidden_dim', 'tcn_hidden_dim', 'tcn_n_layers', 'tcn_kernel_size',
        'koopman_dim', 'num_heads', 'lora_rank', 'b_max', 'dropout',
        'eig_target_radius', 'eig_margin', 'eig_div_sigma',
        'alpha_pred', 'alpha_recon', 'gamma_eig', 'gamma_div', 'delta_decorr',
        'pred_steps',
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg_kwargs[key] = val
    return KoopmanCVAEConfig(**cfg_kwargs)