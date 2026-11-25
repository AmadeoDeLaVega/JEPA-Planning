"""Unit tests for Jensen-certified MPC and spectral head fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa_planning import linear_head as lh
from jepa_planning import trust_mpc as tm
from jepa_planning.calibrated_objectives import dense_horizon_head_distance


class _ConvexToyModel:
    """Unimodal quadratic world model where averaging elites strictly helps."""

    def __init__(self, target_action: torch.Tensor):
        self.device = torch.device("cpu")
        self.target_action = target_action
        self.unroll_count = 0

    def encode(self, obs: dict) -> torch.Tensor:
        return torch.zeros(1, 1, 4, 8)

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        residual = (actions[-1] - self.target_action[None, :]).unsqueeze(1).repeat(1, 4, 1)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, : residual.shape[-1]] = residual.reshape(batch_size, 2, 2, -1)
        return {"visual": padded}


class _MultiStepToyModel(_ConvexToyModel):
    """Quadratic toy model emitting per-step residuals across all H rollout steps."""

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        for step in range(horizon):
            residual = (actions[step] - self.target_action[None, :]).unsqueeze(1).repeat(1, 4, 1)
            padded[step + 1, :, 0, :, :, : residual.shape[-1]] = residual.reshape(batch_size, 2, 2, -1)
        return {"visual": padded}


class _BimodalBarrierModel(_ConvexToyModel):
    """Symmetric two-mode world model (+1 or -1 succeeds, 0 hits a high-cost barrier)."""

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        # Minimum distance to either +target or -target; action = 0 sits at the barrier peak!
        d_pos = (actions[0] - self.target_action[None, :]).norm(dim=-1, keepdim=True)
        d_neg = (actions[0] + self.target_action[None, :]).norm(dim=-1, keepdim=True)
        barrier_residual = torch.minimum(d_pos, d_neg).unsqueeze(1).repeat(1, 4, 4)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, :4] = barrier_residual.reshape(batch_size, 2, 2, 4)
        return {"visual": padded}


class _NarrowSubModeBarrierModel(_ConvexToyModel):
    """Four-well model where averaging even aligned elites crosses a sub-barrier."""

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        wells = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
            ],
            dtype=actions.dtype,
        )
        dists = (actions[0, :, None, :] - wells[None, :, :]).norm(dim=-1)
        min_dist = dists.min(dim=-1, keepdim=True).values
        residual = min_dist.unsqueeze(1).repeat(1, 4, 4)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, :4] = residual.reshape(batch_size, 2, 2, 4)
        return {"visual": padded}


def _dummy_obs() -> dict:
    return {"visual": np.zeros((16, 16, 3), dtype=np.uint8), "proprio": np.zeros(4, dtype=np.float32)}


def _synthetic_dominant_dataset() -> tuple[torch.Tensor, lh.AllocationSampler]:
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(12), 2)
    latents = rng.standard_normal((12, 4)).astype(np.float32) * np.array([4.0, 1.0, 0.8, 0.6], dtype=np.float32)
    torch_gen = torch.Generator().manual_seed(11)
    features = torch.zeros(24, 4, 16)
    for row, g in enumerate(groups):
        features[row, :, :4] = torch.from_numpy(latents[g])
        features[row, :, 4:8] = torch.randn(4, 4, generator=torch_gen) * 2.0
    sampler = lh.AllocationSampler(groups, num_targets=12, hands_per_target=2, seed=0)
    return features, sampler


def test_budget_parity_and_scale_invariance():
    target = torch.tensor([0.8, -0.6, 0.4, -0.2], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(0).standard_normal((2, 16, 2, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model_a = _ConvexToyModel(target)
    chunk_a, _, info_a = tm.trust_mpc_plan(model_a, _dummy_obs(), goal, head, normals, hold_chunk)

    model_b = _ConvexToyModel(target)
    chunk_b, _, info_b = tm.trust_mpc_plan(model_b, _dummy_obs(), goal, 6.0 * head, normals, hold_chunk)

    assert model_a.unroll_count == 32 and info_a["unrolls"] == 32
    assert info_a["jensen_violation"] is False
    assert np.allclose(chunk_a, chunk_b)


def test_jensen_barrier_splits_symmetric_bimodal_elites():
    target = torch.tensor([1.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    # Construct symmetric +/- noise so standard elite averaging cancels to 0 (the barrier peak).
    half = np.random.default_rng(2).standard_normal((2, 8, 2, 4)).astype(np.float32)
    normals = np.concatenate([half, -half], axis=1)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model = _BimodalBarrierModel(target)
    chunk, _, info = tm.trust_mpc_plan(model, _dummy_obs(), goal, head, normals, hold_chunk, stay_margin=0.0)
    assert info["jensen_violation"] is True
    assert abs(float(chunk[0])) > 0.4


def test_isotropy_regularizer_prevents_rank_one_collapse():
    features, sampler = _synthetic_dominant_dataset()
    head_iso = tm.fit_isotropy_head(features, sampler, steps=150, batch_size=16, isotropy_weight=0.05, rank=8)
    assert tm.effective_rank(head_iso) > 4.0


def test_project_capped_simplex_invariants_and_validation():
    values = np.array([5.0, 4.0, 0.2, 0.1, -0.5, 0.0], dtype=np.float64)
    projected = tm.project_capped_simplex(values, cap=0.4, total=1.0)

    assert projected.shape == values.shape
    assert np.all(projected >= -1e-12)
    assert np.all(projected <= 0.4 + 1e-12)
    assert np.isclose(projected.sum(), 1.0, atol=1e-12)
    # Top two coordinates saturate at the cap; trailing negative/tiny coordinates hit zero.
    assert np.isclose(projected[0], 0.4, atol=1e-12)
    assert np.isclose(projected[1], 0.4, atol=1e-12)
    assert projected[4] == 0.0

    # Boundary case where cap * n == total forces uniform cap.
    uniform = tm.project_capped_simplex(np.array([10.0, 1.0, 0.0, -3.0]), cap=0.25, total=1.0)
    np.testing.assert_allclose(uniform, np.full(4, 0.25), atol=1e-12)

    # Zero total returns the zero vector.
    zeros = tm.project_capped_simplex(np.array([1.0, 2.0, 3.0]), cap=0.5, total=0.0)
    np.testing.assert_array_equal(zeros, np.zeros(3))

    with pytest.raises(ValueError, match="cap"):
        tm.project_capped_simplex(values, cap=0.0, total=1.0)
    with pytest.raises(ValueError, match="infeasible"):
        tm.project_capped_simplex(np.array([1.0, 2.0, 3.0]), cap=0.2, total=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        tm.project_capped_simplex(np.array([]), cap=0.5, total=1.0)


@pytest.mark.parametrize("k_fantope", [1.0, 2.5, 4.0, 8.0])
def test_project_fantope_mathematical_invariants(k_fantope: float):
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 16, generator=gen, dtype=torch.float64)
    # Strongly spike the first singular direction to test rank-1 collapse recovery.
    weight[0] *= 25.0

    tm.project_fantope_(weight, k_fantope=k_fantope)
    singular_values = torch.linalg.svdvals(weight)
    eigenvalues = singular_values.square()

    assert np.isclose(float(eigenvalues.sum()), 1.0, atol=1e-10)
    assert float(eigenvalues.max()) <= (1.0 / k_fantope) + 1e-10
    assert tm.effective_rank(weight) >= k_fantope - 1e-6


def test_project_fantope_allows_trailing_zero_singular_values_and_validates():
    # Rank-4 matrix embedded in [8, 16]: projecting with k=3 keeps the 4 trailing singular values at 0.
    gen = torch.Generator().manual_seed(13)
    weight = torch.zeros(8, 16, dtype=torch.float64)
    weight[:4] = torch.randn(4, 16, generator=gen, dtype=torch.float64)
    weight[0] *= 10.0

    tm.project_fantope_(weight, k_fantope=3.0)
    singular_values = torch.linalg.svdvals(weight)
    assert np.isclose(float(singular_values.square().sum()), 1.0, atol=1e-10)
    assert float(singular_values.square().max()) <= (1.0 / 3.0) + 1e-10
    assert tm.effective_rank(weight) >= 3.0 - 1e-6
    assert float(singular_values[4:].max()) < 1e-12

    with pytest.raises(ValueError, match="k_fantope"):
        tm.project_fantope_(weight, k_fantope=0.5)
    with pytest.raises(ValueError, match="k_fantope"):
        tm.project_fantope_(weight, k_fantope=9.0)


def test_fit_fantope_head_prevents_rank_one_collapse():
    features, sampler = _synthetic_dominant_dataset()
    k_fantope = 4.0
    head_fantope = tm.fit_fantope_head(
        features,
        sampler,
        k_fantope=k_fantope,
        steps=150,
        batch_size=16,
        rank=8,
        project_every=5,
        seed=0,
    )

    singular_values = torch.linalg.svdvals(head_fantope.float())
    eigenvalues = singular_values.square()

    assert np.isclose(float(eigenvalues.sum()), 1.0, atol=1e-5)
    assert float(eigenvalues.max()) <= (1.0 / k_fantope) + 1e-5
    assert tm.effective_rank(head_fantope) >= k_fantope - 1e-4

    # Signal channels [0:4] should carry much higher column energy than nuisance channels [4:8].
    signal_energy = float(head_fantope[:, :4].square().sum())
    nuisance_energy = float(head_fantope[:, 4:8].square().sum())
    assert signal_energy > nuisance_energy


def test_certified_barrier_split_budget_scale_invariance_and_convex_retention():
    target = torch.tensor([0.8, -0.6, 0.4, -0.2], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(5).standard_normal((2, 16, 2, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model_a = _ConvexToyModel(target)
    chunk_a, _, info_a = tm.trust_mpc_plan(
        model_a,
        _dummy_obs(),
        goal,
        head,
        normals,
        hold_chunk,
        certify_barrier_split=True,
    )

    model_b = _ConvexToyModel(target)
    chunk_b, _, info_b = tm.trust_mpc_plan(
        model_b,
        _dummy_obs(),
        goal,
        6.0 * head,
        normals,
        hold_chunk,
        certify_barrier_split=True,
    )

    assert model_a.unroll_count == 32 and info_a["unrolls"] == 32
    assert model_b.unroll_count == 32 and info_b["unrolls"] == 32
    assert info_a["jensen_violation"] is False
    assert "mode_mean_cost" in info_a
    # Anti-Optimizer's-Curse rule: always retain elite_mean when Jensen convexity holds.
    assert info_a["selected_cost"] == pytest.approx(info_a["elite_mean_cost"])
    assert np.allclose(chunk_a, chunk_b)

    # Even if a single lucky sample lands near the target so mode_mean has lower cost than
    # elite_mean, convex basins (c(elite_mean) <= mean(c_elites)) still keep elite_mean.
    skewed_normals = np.random.default_rng(9).standard_normal((2, 16, 2, 4)).astype(np.float32)
    skewed_normals[1, 1] = 0.0
    model_c = _ConvexToyModel(target)
    _, _, info_c = tm.trust_mpc_plan(
        model_c,
        _dummy_obs(),
        goal,
        head,
        skewed_normals,
        hold_chunk,
        stay_margin=0.0,
        certify_barrier_split=True,
    )
    assert info_c["jensen_violation"] is False
    assert info_c["selected_cost"] == pytest.approx(info_c["elite_mean_cost"])

    with pytest.raises(ValueError, match="samples_per_iter >= 3"):
        tm.trust_mpc_plan(
            _ConvexToyModel(target),
            _dummy_obs(),
            goal,
            head,
            normals[:, :2],
            hold_chunk,
            certify_barrier_split=True,
        )


def test_certified_barrier_split_bimodal_certifies_mode_mean_and_fallback():
    target = torch.tensor([1.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    half = np.random.default_rng(2).standard_normal((2, 8, 2, 4)).astype(np.float32)
    normals = np.concatenate([half, -half], axis=1)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model = _BimodalBarrierModel(target)
    chunk, _, info = tm.trust_mpc_plan(
        model,
        _dummy_obs(),
        goal,
        head,
        normals,
        hold_chunk,
        stay_margin=0.0,
        certify_barrier_split=True,
    )
    assert model.unroll_count == 32 and info["unrolls"] == 32
    assert info["jensen_violation"] is True
    assert "mode_mean_cost" in info
    assert info["mode_mean_cost"] < info["elite_mean_cost"]
    assert abs(float(chunk[0])) > 0.4

    # Construct a 4-well barrier where iter 0 yields mean1 = 0, std1 = 1 on (x, y)
    # and iter 1 places elites near all four corners so both elite_mean (~0, 0) and
    # mode_mean (~1, 0) sit on high-cost barriers, forcing fallback to elite_actions[0].
    well_normals = np.zeros((2, 8, 2, 4), dtype=np.float32)
    well_normals[0, 1, 0, :2] = [1.0, 1.0]
    well_normals[0, 2, 0, :2] = [1.0, -1.0]
    well_normals[0, 3, 0, :2] = [-1.0, 1.0]
    well_normals[0, 4, 0, :2] = [-1.0, -1.0]
    well_normals[1, 1, 0, :2] = [1.0, 1.0]
    well_normals[1, 2, 0, :2] = [1.0, -0.9]
    well_normals[1, 3, 0, :2] = [-1.0, 0.9]
    well_normals[1, 4, 0, :2] = [-1.0, -0.9]

    fallback_model = _NarrowSubModeBarrierModel(target)
    chunk_fb, _, info_fb = tm.trust_mpc_plan(
        fallback_model,
        _dummy_obs(),
        goal,
        head,
        well_normals,
        hold_chunk,
        num_elites=4,
        stay_margin=0.0,
        certify_barrier_split=True,
    )
    assert fallback_model.unroll_count == 16 and info_fb["unrolls"] == 16
    assert info_fb["jensen_violation"] is True
    assert info_fb["mode_mean_cost"] > info_fb["selected_cost"]
    assert np.allclose(chunk_fb[:2], [1.0, 1.0], atol=1e-5)


def test_multi_goal_prototype_and_step_weights_support():
    gen = torch.Generator().manual_seed(21)
    goal_bank = torch.randn(4, 4, 8, generator=gen, dtype=torch.float32)
    goal_centroid = goal_bank.mean(dim=0)
    head = lh.random_head(rank=4, dim=8, seed=3)
    target = torch.tensor([0.5, -0.4, 0.3, -0.2], dtype=torch.float32)
    actions = np.random.default_rng(10).standard_normal((12, 3, 4)).astype(np.float32)

    # rollout_head_costs with [K, P, D] matches centroid [P, D] with zero extra unrolls.
    model_bank = _MultiStepToyModel(target)
    model_centroid = _MultiStepToyModel(target)
    encoded = model_bank.encode(_dummy_obs())
    costs_bank = tm.rollout_head_costs(model_bank, encoded, actions, goal_bank, head, batch_size=5)
    costs_centroid = tm.rollout_head_costs(model_centroid, encoded, actions, goal_centroid, head, batch_size=5)
    assert model_bank.unroll_count == len(actions)
    assert model_centroid.unroll_count == len(actions)
    np.testing.assert_allclose(costs_bank, costs_centroid, rtol=1e-6, atol=1e-6)

    # Dense-horizon step_weights delegates to dense_horizon_head_distance.
    weights = [0.2, 0.3, 0.5]
    model_dense = _MultiStepToyModel(target)
    dense_costs = tm.rollout_head_costs(
        model_dense, encoded, actions, goal_bank, head, batch_size=12, step_weights=weights
    )
    action_tensor = torch.from_numpy(actions).permute(1, 0, 2)
    predicted_visual = _MultiStepToyModel(target).unroll(encoded, action_tensor)["visual"]
    rollout_features = predicted_visual[-3:].reshape(3, len(actions), 4, 8).float()
    expected_dense = dense_horizon_head_distance(
        rollout_features, goal_bank, head, step_weights=weights
    ).numpy()
    np.testing.assert_allclose(dense_costs, expected_dense, rtol=1e-6, atol=1e-6)

    # trust_mpc_plan with [K, P, D] matches centroid [P, D] for both certified and default modes.
    normals = np.random.default_rng(14).standard_normal((2, 12, 3, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)
    for certified in (False, True):
        m_bank = _MultiStepToyModel(target)
        m_cent = _MultiStepToyModel(target)
        c_bank, p_bank, info_bank = tm.trust_mpc_plan(
            m_bank,
            _dummy_obs(),
            goal_bank,
            head,
            normals,
            hold_chunk,
            certify_barrier_split=certified,
            step_weights=weights,
        )
        c_cent, p_cent, info_cent = tm.trust_mpc_plan(
            m_cent,
            _dummy_obs(),
            goal_centroid,
            head,
            normals,
            hold_chunk,
            certify_barrier_split=certified,
            step_weights=weights,
        )
        assert m_bank.unroll_count == 24 and info_bank["unrolls"] == 24
        assert m_cent.unroll_count == 24 and info_cent["unrolls"] == 24
        np.testing.assert_allclose(c_bank, c_cent, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(p_bank, p_cent, rtol=1e-6, atol=1e-6)
