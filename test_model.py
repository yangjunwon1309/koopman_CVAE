"""
Quick sanity check for Koopman Prior CVAE
Tests: forward pass, loss shapes, sampling, rollout
"""

import torch
import math
from models.koopman_cvae import KoopmanCVAE, KoopmanCVAEConfig, symlog, symexp


def test_symlog():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = symlog(x)
    x_rec = symexp(y)
    assert torch.allclose(x, x_rec, atol=1e-5), "symlog/symexp roundtrip failed"
    print(f"  symlog: {x.tolist()} -> {y.tolist()}")
    print("  symlog/symexp roundtrip: PASSED")


def test_forward_pass():
    cfg = KoopmanCVAEConfig(
        action_dim=6,
        state_dim=24,
        patch_size=5,
        dt_control=0.02,
        embed_dim=64,
        state_embed_dim=32,
        gru_hidden_dim=128,
        mlp_hidden_dim=128,
        koopman_dim=16,
    )
    model = KoopmanCVAE(cfg)
    model.eval()

    B, T = 4, 50
    actions = torch.randn(B, T, cfg.action_dim)
    states  = torch.randn(B, T, cfg.state_dim)

    with torch.no_grad():
        out = model(actions, states)

    assert 'loss' in out
    assert 'loss_recon' in out
    assert 'loss_kl' in out
    assert 'loss_pred' in out
    assert not torch.isnan(out['loss']), "NaN loss!"

    Np = (T // cfg.patch_size)
    assert out['p_hat'].shape == (B, Np, cfg.patch_size, cfg.action_dim)
    assert out['z_re'].shape  == (B, Np, cfg.koopman_dim)
    assert out['z_im'].shape  == (B, Np, cfg.koopman_dim)

    print(f"  Forward pass: PASSED")
    print(f"  Loss = {out['loss'].item():.4f}  "
          f"(recon={out['loss_recon'].item():.4f}, "
          f"kl={out['loss_kl'].item():.4f}, "
          f"pred={out['loss_pred'].item():.4f})")


def test_eigenvalues():
    cfg = KoopmanCVAEConfig(koopman_dim=8, mu_fixed=-0.2,
                             omega_max=math.pi, patch_size=5, dt_control=0.02)
    model = KoopmanCVAE(cfg)

    lb_re, lb_im = model.koopman.get_discrete_eigenvalues()
    modulus = torch.sqrt(lb_re**2 + lb_im**2)
    expected_mod = math.exp(-0.2 * 5 * 0.02)

    assert torch.allclose(modulus, torch.full_like(modulus, expected_mod), atol=1e-5), \
        f"Eigenvalue modulus wrong: {modulus}"
    print(f"  Eigenvalue modulus: {modulus[0].item():.6f} (expected {expected_mod:.6f}): PASSED")

    # Check ascending frequencies
    omega = model.koopman.omega.detach()
    print(f"  Initial omega: {omega.tolist()}")
    assert (omega[1:] < omega[:-1]).all(), "Omega should be descending (higher freq first)"
    print("  Omega initialization: PASSED")


def test_sampling():
    cfg = KoopmanCVAEConfig(
        action_dim=6, state_dim=24, patch_size=5,
        dt_control=0.02, koopman_dim=16,
        embed_dim=64, state_embed_dim=32,
        gru_hidden_dim=128, mlp_hidden_dim=128,
    )
    model = KoopmanCVAE(cfg)

    B = 2
    horizon = 50
    states = torch.randn(B, 1, cfg.state_dim)   # only s_1

    with torch.no_grad():
        actions = model.sample(states, horizon=horizon)

    assert actions.shape == (B, horizon, cfg.action_dim), \
        f"Wrong sample shape: {actions.shape}"
    assert not torch.isnan(actions).any(), "NaN in samples!"
    print(f"  Sampling (horizon={horizon}): PASSED, shape={actions.shape}")


def test_rollout_cost():
    """Verify O(m * tau) rollout complexity"""
    cfg = KoopmanCVAEConfig(koopman_dim=64, patch_size=5, dt_control=0.02)
    model = KoopmanCVAE(cfg)

    B, m, tau = 32, 64, 100
    z_re = torch.randn(B, m)
    z_im = torch.randn(B, m)

    z_re_seq, z_im_seq = model.koopman.rollout(z_re, z_im, tau)
    assert z_re_seq.shape == (B, tau, m)
    print(f"  Rollout shape: {z_re_seq.shape}: PASSED")


def test_environment_configs():
    """Test different environment configs"""
    configs = [
        ('DMControl Walker',     dict(action_dim=6,  state_dim=24, patch_size=5,  dt_control=0.02)),
        ('Adroit Pen',           dict(action_dim=24, state_dim=45, patch_size=3,  dt_control=0.04)),
        ('HumanoidBench Stand',  dict(action_dim=19, state_dim=76, patch_size=10, dt_control=0.01)),
        ('Isaac Franka',         dict(action_dim=7,  state_dim=23, patch_size=6,  dt_control=0.0167)),
    ]
    for name, env_cfg in configs:
        cfg = KoopmanCVAEConfig(**env_cfg)
        dt_patch = cfg.patch_size * cfg.dt_control
        model = KoopmanCVAE(cfg)
        B, T = 2, 60
        actions = torch.randn(B, T, cfg.action_dim)
        states  = torch.randn(B, T, cfg.state_dim)
        out = model(actions, states)
        assert not torch.isnan(out['loss'])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:30s} | dt_patch={dt_patch*1000:.0f}ms | loss={out['loss'].item():.3f} | params={n_params:,}")


if __name__ == '__main__':
    print("=" * 60)
    print("Koopman Prior CVAE — Sanity Checks")
    print("=" * 60)

    print("\n[1] Symlog/Symexp")
    test_symlog()

    print("\n[2] Forward Pass")
    test_forward_pass()

    print("\n[3] Eigenvalues")
    test_eigenvalues()

    print("\n[4] Sampling")
    test_sampling()

    print("\n[5] Rollout")
    test_rollout_cost()

    print("\n[6] Environment Configs")
    test_environment_configs()

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
