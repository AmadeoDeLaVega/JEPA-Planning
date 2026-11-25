"""Unit tests for calibrated multi-goal, anchor-blended, dense-horizon, and patch objectives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa_planning.calibrated_objectives import (
    anchor_calibrated_blend,
    dense_horizon_head_distance,
    patch_energy_map,
    projected_multi_goal_distance,
)
from jepa_planning.goal_costs import linear_goal_distance, mean_squared_goal_distance
from jepa_planning.linear_head import identity_head, random_head
from jepa_planning.temporal_cost import blend_costs


# ------------------------------------------------ projected_multi_goal_distance


def test_projected_multi_goal_exact_mean_matches_brute_force():
    gen = torch.Generator().manual_seed(7)
    candidates = torch.randn(12, 16, 24, generator=gen, dtype=torch.float64)
    goals = torch.randn(5, 16, 24, generator=gen, dtype=torch.float64)
    head = random_head(rank=6, dim=24, seed=3).double()

    brute_force = torch.stack(
        [linear_goal_distance(candidates, goals[k : k + 1], head) for k in range(len(goals))],
        dim=0,
    ).mean(dim=0)

    exact = projected_multi_goal_distance(candidates, goals, head, exact_mean=True)
    fast = projected_multi_goal_distance(candidates, goals, head, exact_mean=False)

    torch.testing.assert_close(exact, brute_force, atol=1e-12, rtol=1e-12)
    # Fast mode differs only by a candidate-independent variance constant.
    offset = exact - fast
    torch.testing.assert_close(offset, offset[:1].expand_as(offset), atol=1e-12, rtol=1e-12)
    assert int(torch.argmin(fast)) == int(torch.argmin(exact))
    torch.testing.assert_close(torch.argsort(fast), torch.argsort(exact))


def test_projected_multi_goal_accepts_single_unbatched_goal():
    gen = torch.Generator().manual_seed(11)
    candidates = torch.randn(8, 16, 12, generator=gen)
    goal = torch.randn(16, 12, generator=gen)
    head = random_head(rank=4, dim=12, seed=1)

    expected = linear_goal_distance(candidates, goal.unsqueeze(0), head)
    fast = projected_multi_goal_distance(candidates, goal, head, exact_mean=False)
    exact = projected_multi_goal_distance(candidates, goal, head, exact_mean=True)

    torch.testing.assert_close(fast, expected)
    torch.testing.assert_close(exact, expected)


def test_projected_multi_goal_rejects_invalid_inputs():
    candidates = torch.randn(4, 9, 8)
    goals = torch.randn(3, 9, 8)
    head = random_head(rank=4, dim=8, seed=0)

    with pytest.raises(ValueError, match="at least 3 dimensions"):
        projected_multi_goal_distance(torch.randn(4, 8), torch.randn(3, 8), head)
    with pytest.raises(ValueError, match="non-empty"):
        projected_multi_goal_distance(torch.empty(0, 9, 8), goals, head)
    with pytest.raises(ValueError, match="empty"):
        projected_multi_goal_distance(candidates, torch.empty(0, 9, 8), head)
    with pytest.raises(ValueError, match="trailing dimensions"):
        projected_multi_goal_distance(candidates, torch.randn(3, 4, 8), head)
    with pytest.raises(ValueError, match="head must have shape"):
        projected_multi_goal_distance(candidates, goals, torch.randn(4, 7))
    with pytest.raises(ValueError, match="same dtype"):
        projected_multi_goal_distance(candidates, goals.double(), head)


# ------------------------------------------------------ anchor_calibrated_blend


def test_anchor_calibrated_blend_preserves_iia_where_zscore_flips():
    # Counterexample where appending dominated decoy D=(9.0, 10.5) flips z-score winner A -> B.
    primary_base = np.array([[1.0], [2.05], [2.5]])
    secondary_base = np.array([[10.0], [8.0], [11.0]])
    primary_with_decoy = np.array([[1.0], [2.05], [2.5], [9.0]])
    secondary_with_decoy = np.array([[10.0], [8.0], [11.0], [10.5]])

    # Confirm batch z-score blend_costs suffers an IIA reversal on this counterexample.
    z_base = blend_costs(primary_base, secondary_base, weight=0.5)
    z_decoy = blend_costs(primary_with_decoy, secondary_with_decoy, weight=0.5)
    assert int(np.argmin(z_base[:, 0])) == 0
    assert int(np.argmin(z_decoy[:, 0])) == 1

    # Anchor calibration preserves exact scores, ratios, and ordering for existing candidates.
    a_base = anchor_calibrated_blend(primary_base, secondary_base, weight=0.5, anchor_index=0)
    a_decoy = anchor_calibrated_blend(
        primary_with_decoy, secondary_with_decoy, weight=0.5, anchor_index=0
    )

    np.testing.assert_array_equal(a_decoy[:3], a_base)
    assert int(np.argmin(a_base[:, 0])) == 0
    assert int(np.argmin(a_decoy[:, 0])) == 0
    np.testing.assert_allclose(
        a_decoy[:3, 0] / a_decoy[0, 0],
        a_base[:, 0] / a_base[0, 0],
        rtol=0.0,
        atol=0.0,
    )


def test_anchor_calibrated_blend_1d_and_2d_iia_for_arbitrary_decoys():
    rng = np.random.default_rng(42)
    primary = rng.uniform(0.5, 5.0, size=(6, 4))
    secondary = rng.uniform(2.0, 20.0, size=(6, 4))
    decoy_p = np.array([[50.0, 100.0, 75.0, 200.0], [0.1, 999.0, 12.0, 40.0]])
    decoy_s = np.array([[80.0, 60.0, 90.0, 150.0], [500.0, 1.0, 300.0, 20.0]])

    base_2d = anchor_calibrated_blend(primary, secondary, weight=0.35, anchor_index=2)
    ext_2d = anchor_calibrated_blend(
        np.vstack([primary, decoy_p]),
        np.vstack([secondary, decoy_s]),
        weight=0.35,
        anchor_index=2,
    )
    np.testing.assert_array_equal(ext_2d[: len(primary)], base_2d)

    base_1d = anchor_calibrated_blend(primary[:, 0], secondary[:, 0], weight=0.35, anchor_index=2)
    ext_1d = anchor_calibrated_blend(
        np.concatenate([primary[:, 0], decoy_p[:, 0]]),
        np.concatenate([secondary[:, 0], decoy_s[:, 0]]),
        weight=0.35,
        anchor_index=2,
    )
    np.testing.assert_array_equal(ext_1d[: len(primary)], base_1d)
    np.testing.assert_array_equal(base_1d, base_2d[:, 0])


def test_anchor_calibrated_blend_exact_positive_scale_invariance():
    primary = np.array([[1.0, 2.0], [2.5, 4.0], [3.0, 1.5]])
    secondary = np.array([[10.0, 5.0], [8.0, 12.5], [15.0, 7.5]])

    unscaled = anchor_calibrated_blend(primary, secondary, weight=0.4, anchor_index=0)
    scaled = anchor_calibrated_blend(7.5 * primary, 0.03 * secondary, weight=0.4, anchor_index=0)

    np.testing.assert_allclose(scaled, unscaled, rtol=1e-14, atol=1e-15)

    # Power-of-two rescaling is bit-for-bit identical in IEEE-754 arithmetic.
    pow2_scaled = anchor_calibrated_blend(
        8.0 * primary, 0.0625 * secondary, weight=0.4, anchor_index=0
    )
    np.testing.assert_array_equal(pow2_scaled, unscaled)


def test_anchor_calibrated_blend_eps_guards_zero_anchor():
    primary = np.array([0.0, 1.0, 2.0])
    secondary = np.array([0.0, 3.0, 6.0])

    blended = anchor_calibrated_blend(primary, secondary, weight=0.5, anchor_index=0, eps=1e-6)

    assert np.isfinite(blended).all()
    np.testing.assert_allclose(blended, [0.0, 2e6, 4e6])


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_anchor_calibrated_blend_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        anchor_calibrated_blend(np.ones(3), np.ones(3), weight=weight)


@pytest.mark.parametrize("anchor_index", [-1, 3, 10])
def test_anchor_calibrated_blend_rejects_invalid_anchor_index(anchor_index):
    with pytest.raises(ValueError, match="anchor_index"):
        anchor_calibrated_blend(np.ones(3), np.ones(3), weight=0.5, anchor_index=anchor_index)


def test_anchor_calibrated_blend_rejects_bad_shapes_and_empty():
    with pytest.raises(ValueError, match="same shape"):
        anchor_calibrated_blend(np.ones((3, 2)), np.ones((3, 3)), weight=0.5)
    with pytest.raises(ValueError, match="1D"):
        anchor_calibrated_blend(np.ones((2, 2, 2)), np.ones((2, 2, 2)), weight=0.5)
    with pytest.raises(ValueError, match="empty"):
        anchor_calibrated_blend(np.empty((0, 2)), np.empty((0, 2)), weight=0.5)
    with pytest.raises(ValueError, match="eps"):
        anchor_calibrated_blend(np.ones(3), np.ones(3), weight=0.5, eps=0.0)


# -------------------------------------------------- dense_horizon_head_distance


def test_dense_horizon_default_equals_terminal_projected_distance():
    gen = torch.Generator().manual_seed(19)
    rollout = torch.randn(5, 7, 16, 10, generator=gen)
    goals = torch.randn(4, 16, 10, generator=gen)
    head = random_head(rank=5, dim=10, seed=2)

    default_cost = dense_horizon_head_distance(rollout, goals, head)
    explicit_terminal = dense_horizon_head_distance(
        rollout, goals, head, step_weights=[0.0, 0.0, 0.0, 0.0, 3.5]
    )
    expected = projected_multi_goal_distance(rollout[-1], goals, head, exact_mean=False)

    torch.testing.assert_close(default_cost, expected)
    torch.testing.assert_close(explicit_terminal, expected)


def test_dense_horizon_normalizes_weights_and_accepts_single_goal():
    gen = torch.Generator().manual_seed(23)
    rollout = torch.randn(4, 6, 9, 12, generator=gen, dtype=torch.float64)
    goal = torch.randn(9, 12, generator=gen, dtype=torch.float64)
    head = random_head(rank=4, dim=12, seed=4).double()
    raw_weights = [1.0, 2.0, 3.0, 4.0]

    actual = dense_horizon_head_distance(rollout, goal, head, step_weights=raw_weights)
    per_step = torch.stack(
        [linear_goal_distance(rollout[t], goal.unsqueeze(0), head) for t in range(4)],
        dim=0,
    )
    norm_weights = torch.tensor(raw_weights, dtype=torch.float64) / sum(raw_weights)
    expected = (norm_weights[:, None] * per_step).sum(dim=0)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_dense_horizon_rejects_invalid_inputs():
    rollout = torch.randn(3, 4, 9, 8)
    goal = torch.randn(9, 8)
    head = random_head(rank=4, dim=8, seed=0)

    with pytest.raises(ValueError, match="shape"):
        dense_horizon_head_distance(torch.randn(4, 9, 8), goal, head)
    with pytest.raises(ValueError, match="length 3"):
        dense_horizon_head_distance(rollout, goal, head, step_weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="nonnegative"):
        dense_horizon_head_distance(rollout, goal, head, step_weights=[-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive sum"):
        dense_horizon_head_distance(rollout, goal, head, step_weights=[0.0, 0.0, 0.0])


# ------------------------------------------------------------- patch_energy_map


def test_patch_energy_map_mean_matches_mse_and_linear_head_distance():
    gen = torch.Generator().manual_seed(31)
    candidates = torch.randn(5, 16, 12, generator=gen, dtype=torch.float64)
    goal = torch.randn(1, 16, 12, generator=gen, dtype=torch.float64)
    head = random_head(rank=4, dim=12, seed=5).double()

    mse_map = patch_energy_map(candidates, goal)
    assert mse_map.shape == (5, 4, 4)
    torch.testing.assert_close(
        mse_map.mean(dim=(-2, -1)),
        mean_squared_goal_distance(candidates, goal),
        atol=1e-12,
        rtol=1e-12,
    )

    head_map = patch_energy_map(candidates, goal, head=head)
    assert head_map.shape == (5, 4, 4)
    torch.testing.assert_close(
        head_map.mean(dim=(-2, -1)),
        linear_goal_distance(candidates, goal, head),
        atol=1e-12,
        rtol=1e-12,
    )

    # Identity head I/sqrt(D) produces the exact same spatial energy map as plain MSE.
    exact_identity = torch.eye(12, dtype=torch.float64) / (12.0**0.5)
    id_map = patch_energy_map(candidates, goal, head=exact_identity, grid_size=4)
    torch.testing.assert_close(id_map, mse_map, atol=1e-12, rtol=1e-12)

    id_map_f32 = patch_energy_map(
        candidates.float(), goal.float(), head=identity_head(12), grid_size=4
    )
    torch.testing.assert_close(
        id_map_f32, patch_energy_map(candidates.float(), goal.float())
    )


def test_patch_energy_map_preserves_flat_axis_for_non_square_patch_count():
    a = torch.randn(3, 10, 8)
    b = torch.randn(1, 10, 8)

    energy = patch_energy_map(a, b)
    assert energy.shape == (3, 10)
    torch.testing.assert_close(energy.mean(dim=-1), mean_squared_goal_distance(a, b))


def test_patch_energy_map_handles_unbatched_frames_and_rejects_bad_grid():
    a = torch.randn(9, 6)
    b = torch.randn(9, 6)

    energy = patch_energy_map(a, b)
    assert energy.shape == (3, 3)

    with pytest.raises(ValueError, match="incompatible"):
        patch_energy_map(a, b, grid_size=4)
    with pytest.raises(ValueError, match="positive integer"):
        patch_energy_map(a, b, grid_size=0)
    with pytest.raises(ValueError, match="trailing"):
        patch_energy_map(torch.randn(2, 9, 6), torch.randn(2, 9, 5))
