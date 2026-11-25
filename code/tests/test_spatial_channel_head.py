"""Fast CPU unit tests for variance-decomposed (ANOVA) spatial-channel goal heads."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa_planning import linear_head as lh
from jepa_planning.calibrated_objectives import projected_multi_goal_distance
from jepa_planning.spatial_channel_head import (
    SpatialChannelGoalHead,
    anova_spatial_weights,
    effective_patches,
    fit_anova_spatial_head,
    load_spatial_channel_head,
    save_spatial_channel_head,
    spatial_channel_distance,
    spatial_patch_scale,
)
from jepa_planning.trust_mpc import effective_rank


def test_anova_spatial_weights_matches_exact_pair_enumeration():
    rng = torch.Generator().manual_seed(123)
    # Include unequal group sizes (1, 2, 3, 4, 5) to verify binom(K_g, 2) weighting
    # and singleton handling.
    groups = np.array([10, 20, 20, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 50])
    num_samples, num_patches, feature_dim = len(groups), 12, 8
    features = torch.randn(num_samples, num_patches, feature_dim, generator=rng)

    weights, diag = anova_spatial_weights(features, groups, gamma=1.5, uniform_floor=0.25)

    # Brute-force enumeration over every unordered same-group pair (a < b).
    pair_sq_diffs = []
    for a in range(num_samples):
        for b in range(a + 1, num_samples):
            if groups[a] == groups[b]:
                pair_sq_diffs.append(0.5 * (features[a] - features[b]).square().mean(dim=-1))
    brute_within = torch.stack(pair_sq_diffs, dim=0).mean(dim=0)

    brute_total = features.var(dim=0, unbiased=False).mean(dim=-1)
    brute_between = (brute_total - brute_within).clamp_min(0.0)
    brute_tau = float(brute_total.median().item()) + 1e-8
    brute_score = brute_between / (brute_within + brute_tau)

    torch.testing.assert_close(diag["var_within"], brute_within, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(diag["var_total"], brute_total, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(diag["var_between"], brute_between, atol=1e-6, rtol=1e-5)
    assert diag["tau_var"] == pytest.approx(brute_tau, rel=1e-6)
    torch.testing.assert_close(diag["score"], brute_score, atol=1e-6, rtol=1e-5)

    assert weights.shape == (num_patches,)
    assert float(weights.sum().item()) == pytest.approx(1.0, abs=1e-6)
    assert bool((weights >= 0.0).all())
    assert 1.0 <= diag["effective_patches"] <= float(num_patches)
    assert diag["effective_patches"] == pytest.approx(effective_patches(weights), rel=1e-5)


def test_anova_eliminates_vit_border_token_shortcut_collapse():
    """Synthetic 256-patch (16x16) ViT feature tensor mimicking Sawyer Push-v3 geometry.

    - Border patches (rows 0-1, 32 patches): tiny within-group variance (1e-7)
      and tiny cross-group leakage (1e-5), producing ~100x raw SNR despite
      near-zero physical signal.
    - Upper-arm patches (rows 3-6, 64 patches): huge within-group embodiment
      variance (2.0).
    - Tabletop puck patches (rows 8-9, 32 patches): moderate within-group
      variance (0.1) and strong cross-group object signal (1.0), ~10x raw SNR.
    - Remaining background/table patches: moderate variance (~0.15) with no
      object signal so median total patch variance is non-degenerate.
    """
    num_groups, hands_per_group, num_patches, feature_dim = 32, 4, 256, 16
    num_samples = num_groups * hands_per_group
    groups = np.repeat(np.arange(num_groups), hands_per_group)

    gen = torch.Generator().manual_seed(42)
    features = torch.zeros(num_samples, num_patches, feature_dim, dtype=torch.float32)

    border_idx = torch.arange(0, 32)  # rows 0-1
    arm_idx = torch.arange(48, 112)  # rows 3-6
    puck_idx = torch.arange(128, 160)  # rows 8-9
    used_mask = torch.zeros(num_patches, dtype=torch.bool)
    used_mask[border_idx] = True
    used_mask[arm_idx] = True
    used_mask[puck_idx] = True
    other_idx = torch.nonzero(~used_mask, as_tuple=False).squeeze(-1)

    # Group-level object means and within-group embodiment perturbations.
    border_between = torch.randn(num_groups, len(border_idx), feature_dim, generator=gen) * np.sqrt(1e-5)
    border_within = torch.randn(num_samples, len(border_idx), feature_dim, generator=gen) * np.sqrt(1e-7)
    features[:, border_idx, :] = border_between[groups] + border_within

    arm_within = torch.randn(num_samples, len(arm_idx), feature_dim, generator=gen) * np.sqrt(2.0)
    features[:, arm_idx, :] = arm_within

    puck_between = torch.randn(num_groups, len(puck_idx), feature_dim, generator=gen) * np.sqrt(1.0)
    puck_within = torch.randn(num_samples, len(puck_idx), feature_dim, generator=gen) * np.sqrt(0.1)
    features[:, puck_idx, :] = puck_between[groups] + puck_within

    other_within = torch.randn(num_samples, len(other_idx), feature_dim, generator=gen) * np.sqrt(0.15)
    features[:, other_idx, :] = other_within

    # Pure ANOVA weights (uniform_floor=0.0) and default soft-floored weights.
    w_pure, diag = anova_spatial_weights(features, groups, gamma=1.5, uniform_floor=0.0)
    w_default, _ = anova_spatial_weights(features, groups, gamma=1.5, uniform_floor=0.25)

    # Naive unregularized SNR falls for the border-token shortcut.
    naive_snr = diag["var_between"] / (diag["var_within"] + 1e-12)
    naive_top32 = torch.topk(naive_snr, k=32).indices
    naive_border_hits = int(((naive_top32 >= 0) & (naive_top32 < 32)).sum().item())
    assert naive_border_hits >= 28
    assert float(naive_snr[border_idx].median()) > float(naive_snr[puck_idx].median()) * 3.0

    # Median-regularized ANOVA suppresses border tokens (< 0.01 mass) and focuses on rows 8-9.
    assert float(w_pure[border_idx].sum().item()) < 0.01
    assert float(w_pure[puck_idx].sum().item()) > 0.95
    anova_top32 = torch.topk(w_pure, k=32).indices
    assert bool(((anova_top32 >= 128) & (anova_top32 < 160)).all())

    # Default floor keeps top-32 strictly on rows 8-9 and suppresses border excess mass.
    default_top32 = torch.topk(w_default, k=32).indices
    assert bool(((default_top32 >= 128) & (default_top32 < 160)).all())
    border_floor_mass = 0.25 * (len(border_idx) / num_patches)
    assert float(w_default[border_idx].sum().item()) - border_floor_mass < 0.01
    assert float(w_default[arm_idx].mean().item()) < float(w_default[puck_idx].mean().item()) * 0.25


def test_uniform_spatial_weights_and_exact_distance_identity():
    gen = torch.Generator().manual_seed(7)
    num_samples, num_patches, feature_dim = 16, 16, 12
    groups = np.repeat(np.arange(4), 4)
    features = torch.randn(num_samples, num_patches, feature_dim, generator=gen)

    w_gamma0, _ = anova_spatial_weights(features, groups, gamma=0.0, uniform_floor=0.25)
    w_floor1, _ = anova_spatial_weights(features, groups, gamma=2.0, uniform_floor=1.0)
    expected_uniform = torch.full((num_patches,), 1.0 / num_patches)

    torch.testing.assert_close(w_gamma0, expected_uniform, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(w_floor1, expected_uniform, atol=1e-6, rtol=1e-6)
    assert effective_patches(w_gamma0) == pytest.approx(float(num_patches), rel=1e-5)

    candidates = torch.randn(5, num_patches, feature_dim, generator=gen)
    single_goal = torch.randn(num_patches, feature_dim, generator=gen)
    multi_goals = torch.randn(3, num_patches, feature_dim, generator=gen)
    head = lh.random_head(rank=6, dim=feature_dim, seed=3)

    for goals in (single_goal, multi_goals):
        for exact_mean in (False, True):
            ref = projected_multi_goal_distance(candidates, goals, head, exact_mean=exact_mean)
            d_none = spatial_channel_distance(candidates, goals, head, None, exact_mean=exact_mean)
            d_g0 = spatial_channel_distance(candidates, goals, head, w_gamma0, exact_mean=exact_mean)
            d_f1 = spatial_channel_distance(candidates, goals, head, w_floor1, exact_mean=exact_mean)
            torch.testing.assert_close(d_none, ref, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(d_g0, ref, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(d_f1, ref, atol=1e-6, rtol=1e-6)

    # Arbitrary non-uniform weights satisfy sum_i w_i ||L(a_i - b_i)||^2 identically.
    raw_w = torch.rand(num_patches, generator=gen) + 0.05
    w_norm = raw_w / raw_w.sum()
    explicit = (
        ((candidates - single_goal.unsqueeze(0)).matmul(head.T).square().sum(dim=-1) * w_norm[None, :])
        .sum(dim=-1)
    )
    via_scale = spatial_channel_distance(candidates, single_goal, head, raw_w, exact_mean=False)
    torch.testing.assert_close(via_scale, explicit, atol=1e-6, rtol=1e-5)

    module = SpatialChannelGoalHead(head, raw_w)
    torch.testing.assert_close(module.distance(candidates, single_goal), explicit, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(module.distance(candidates[0], single_goal), explicit[0], atol=1e-6, rtol=1e-5)
    assert module.effective_patches() == pytest.approx(effective_patches(w_norm), rel=1e-5)
    assert module.effective_rank() == pytest.approx(effective_rank(head), rel=1e-5)


@pytest.mark.parametrize("regime", ["fantope", "isotropy", "unregularized"])
def test_fit_anova_spatial_head_regimes_and_pool_hygiene(regime: str):
    gen = torch.Generator().manual_seed(17)
    num_targets, hands_per_target = 12, 4
    num_samples = num_targets * hands_per_target
    num_patches, feature_dim = 16, 12
    groups = np.repeat(np.arange(num_targets), hands_per_target)

    # Signal concentrated in patches 4..7 and channels 0..5.
    target_latents = torch.randn(num_targets, 4, 6, generator=gen)
    features = torch.randn(num_samples, num_patches, feature_dim, generator=gen) * 0.25
    features[:, 4:8, :6] += target_latents[groups] * 1.5

    sampler = lh.AllocationSampler(groups, num_targets=10, hands_per_target=3, seed=5)
    k_fantope = 4.0
    w, head, meta = fit_anova_spatial_head(
        features,
        groups,
        sampler,
        gamma=1.5,
        uniform_floor=0.2,
        regime=regime,
        k_fantope=k_fantope,
        isotropy_weight=0.05,
        steps=20,
        batch_size=8,
        tau=0.2,
        lr=3e-3,
        seed=5,
        rank=6,
    )

    assert w.shape == (num_patches,)
    assert head.shape == (6, feature_dim)
    assert float(w.sum().item()) == pytest.approx(1.0, abs=1e-6)
    assert float(torch.linalg.matrix_norm(head).item()) == pytest.approx(1.0, abs=1e-5)
    assert meta["regime"] == regime
    assert meta["pool_size"] == 30
    assert float(w[4:8].sum().item()) > 0.5
    if regime == "fantope":
        assert effective_rank(head) >= k_fantope - 1e-3

    # Budget/split hygiene: corrupting features outside sampler.images must not alter (w, L).
    perturbed = features.clone()
    unused_mask = np.ones(num_samples, dtype=bool)
    unused_mask[sampler.images] = False
    perturbed[unused_mask] = 999.0
    w2, head2, _ = fit_anova_spatial_head(
        perturbed,
        groups,
        sampler,
        gamma=1.5,
        uniform_floor=0.2,
        regime=regime,
        k_fantope=k_fantope,
        isotropy_weight=0.05,
        steps=20,
        batch_size=8,
        tau=0.2,
        lr=3e-3,
        seed=5,
        rank=6,
    )
    torch.testing.assert_close(w2, w, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(head2, head, atol=1e-6, rtol=1e-6)


def test_save_and_load_spatial_channel_head_and_legacy_fallback(tmp_path):
    head = lh.random_head(rank=8, dim=16, seed=9)
    weights = torch.linspace(1.0, 16.0, 16)
    meta = {"regime": "fantope", "gamma": 1.5, "uniform_floor": 0.25}

    spatial_path = tmp_path / "nested" / "anova_head.pt"
    save_spatial_channel_head(spatial_path, weights, head, meta)

    loaded_w, loaded_head, loaded_meta = load_spatial_channel_head(spatial_path)
    assert loaded_w is not None
    torch.testing.assert_close(loaded_w, weights / weights.sum(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loaded_head, head, atol=1e-6, rtol=1e-6)
    assert loaded_meta["regime"] == "fantope"
    assert loaded_meta["gamma"] == 1.5

    # Legacy lh.load_head also reads the head matrix cleanly from spatial checkpoints.
    legacy_read_head, legacy_read_cfg = lh.load_head(spatial_path)
    torch.testing.assert_close(legacy_read_head, head, atol=1e-6, rtol=1e-6)
    assert legacy_read_cfg["regime"] == "fantope"

    # Legacy lh.save_head file returns spatial_weights=None via load_spatial_channel_head.
    legacy_path = tmp_path / "legacy_head.pt"
    lh.save_head(legacy_path, head, {"rank": 8, "method": "isotropy"})
    fallback_w, fallback_head, fallback_meta = load_spatial_channel_head(legacy_path)
    assert fallback_w is None
    torch.testing.assert_close(fallback_head, head, atol=1e-6, rtol=1e-6)
    assert fallback_meta == {"rank": 8, "method": "isotropy"}


def test_spatial_channel_head_validation_errors():
    feats = torch.randn(4, 4, 6)
    with pytest.raises(ValueError):
        anova_spatial_weights(feats[0], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        anova_spatial_weights(feats, [0, 1, 2])
    with pytest.raises(ValueError):
        anova_spatial_weights(feats, [0, 1, 2, 3])  # no group with >= 2 samples
    with pytest.raises(ValueError):
        anova_spatial_weights(feats, [0, 0, 1, 1], gamma=-0.1)
    with pytest.raises(ValueError):
        anova_spatial_weights(feats, [0, 0, 1, 1], uniform_floor=1.5)

    with pytest.raises(ValueError):
        effective_patches(torch.tensor([-0.1, 1.1]))
    with pytest.raises(ValueError):
        effective_patches(torch.zeros(4))
    with pytest.raises(ValueError):
        spatial_patch_scale(torch.ones(3), num_patches=4)
    with pytest.raises(ValueError):
        spatial_patch_scale(None, num_patches=0)

    sampler = lh.AllocationSampler(np.repeat(np.arange(10), 2), num_targets=8, hands_per_target=2, seed=0)
    with pytest.raises(ValueError):
        fit_anova_spatial_head(torch.randn(20, 4, 6), np.repeat(np.arange(10), 2), sampler, regime="invalid")
