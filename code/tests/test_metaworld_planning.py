"""Unit and simulator tests for Meta-World Sawyer Push planning helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import yaml

from experiments.run_metaworld_trust_mpc import paired_bootstrap_ci, summarize_episodes
from jepa_planning import ROOT
from jepa_planning import goal_groups as gg
from jepa_planning import metaworld_planning as mwp

METAWORLD_ACTION_MEAN = np.array([0.00572358, 0.15735652, -0.13964579, 0.19981939], dtype=np.float32)
METAWORLD_ACTION_STD = np.array([0.73592395, 0.73408043, 0.71829772, 0.74320573], dtype=np.float32)


class _ConvexMetaWorldToyModel:
    """Deterministic quadratic world model tracking per-step action residuals."""

    def __init__(self, target_action: torch.Tensor):
        self.device = torch.device("cpu")
        self.target_action = target_action.float()
        self.unroll_count = 0

    def encode(self, obs: dict) -> torch.Tensor:
        return torch.zeros(1, 1, 4, 8)

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, action_dim = actions.shape
        self.unroll_count += int(batch_size)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        for step in range(horizon):
            residual = actions[step] - self.target_action[None, :]
            padded[step + 1, :, 0, :, :, :action_dim] = residual[:, None, None, :].expand(
                batch_size, 2, 2, action_dim
            )
        return {"visual": padded}


class _BimodalBarrierMetaWorldToyModel(_ConvexMetaWorldToyModel):
    """Symmetric two-mode barrier model where hold_chunk sits on a high-cost peak."""

    def __init__(self, hold_chunk: torch.Tensor, mode_offset: torch.Tensor):
        super().__init__(hold_chunk)
        self.hold_chunk = hold_chunk.float()
        self.mode_offset = mode_offset.float()

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        pos_target = self.hold_chunk + self.mode_offset
        neg_target = self.hold_chunk - self.mode_offset
        d_pos = (actions[0] - pos_target[None, :]).norm(dim=-1, keepdim=True)
        d_neg = (actions[0] - neg_target[None, :]).norm(dim=-1, keepdim=True)
        barrier_residual = torch.minimum(d_pos, d_neg)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, :4] = barrier_residual[:, None, None, :].expand(batch_size, 2, 2, 4)
        return {"visual": padded}


def _dummy_obs() -> dict:
    return {
        "visual": np.zeros((16, 16, 3), dtype=np.uint8),
        "proprio": np.zeros(4, dtype=np.float32),
    }


def test_metaworld_action_bounds_exact_denormalization():
    lower, upper, hold = mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 5)
    assert lower.shape == (20,) and upper.shape == (20,) and hold.shape == (20,)
    assert lower.dtype == np.float32 and upper.dtype == np.float32 and hold.dtype == np.float32

    mean_tiled = np.tile(METAWORLD_ACTION_MEAN, 5)
    std_tiled = np.tile(METAWORLD_ACTION_STD, 5)

    np.testing.assert_allclose(lower * std_tiled + mean_tiled, -1.0, atol=1e-6)
    np.testing.assert_allclose(upper * std_tiled + mean_tiled, 1.0, atol=1e-6)
    np.testing.assert_allclose(hold * std_tiled + mean_tiled, 0.0, atol=1e-6)
    assert np.all(lower <= hold) and np.all(hold <= upper)
    # Normalized 0 drifts substantially because Meta-World's mean action is nonzero.
    assert np.linalg.norm(hold[:4]) > 0.2


def test_metaworld_action_bounds_validation():
    with pytest.raises(ValueError):
        mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 0)
    with pytest.raises(ValueError):
        mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, np.zeros(4), 5)
    with pytest.raises(ValueError):
        mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN[:3], METAWORLD_ACTION_STD, 5)


def test_puck_metrics_cm_accuracy_and_validation():
    goal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    candidates = np.array(
        [
            [0.0, 0.0, 0.0],  # 0.0 cm
            [0.02, 0.0, 0.0],  # 2.0 cm (exact 2 cm boundary)
            [0.0, 0.03, 0.0],  # 3.0 cm (exact 3 cm boundary)
            [0.03, 0.04, 0.0],  # 5.0 cm (exact 5 cm boundary)
            [0.0, 0.0, 0.06],  # 6.0 cm 3D, 0.0 cm XY
        ],
        dtype=np.float64,
    )
    metrics = mwp.puck_metrics_cm(candidates, goal)
    np.testing.assert_allclose(metrics["error_3d_cm"], [0.0, 2.0, 3.0, 5.0, 6.0], atol=1e-9)
    np.testing.assert_allclose(metrics["error_xy_cm"], [0.0, 2.0, 3.0, 5.0, 0.0], atol=1e-9)
    np.testing.assert_array_equal(metrics["success_2cm"], [True, True, False, False, False])
    np.testing.assert_array_equal(metrics["success_3cm"], [True, True, True, False, False])
    np.testing.assert_array_equal(metrics["success_5cm"], [True, True, True, True, False])

    # Single [3] vector input with nonzero goal should work too.
    single = mwp.puck_metrics_cm(np.array([0.13, 0.73, 0.02]), np.array([0.10, 0.70, 0.02]))
    np.testing.assert_allclose(single["error_3d_cm"], np.hypot(3.0, 3.0), atol=1e-9)
    assert bool(single["success_5cm"]) is True
    assert bool(single["success_3cm"]) is False

    with pytest.raises(ValueError):
        mwp.puck_metrics_cm(np.zeros((2, 2)), goal)
    with pytest.raises(ValueError):
        mwp.puck_metrics_cm(candidates, np.zeros(2))
    with pytest.raises(ValueError):
        mwp.puck_metrics_cm(np.empty((0, 3)), goal)


@pytest.mark.parametrize("dual_certify", [True, False])
def test_metaworld_trust_mpc_budget_parity_and_scale_invariance(dual_certify: bool):
    lower, upper, hold = mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 1)
    target = torch.tensor([0.5, -0.4, 0.3, -0.2], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(7).standard_normal((2, 16, 2, 4)).astype(np.float32)

    model_a = _ConvexMetaWorldToyModel(target)
    chunk_a, plan_a, info_a = mwp.metaworld_trust_mpc_plan(
        model_a,
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=8,
        dual_certify=dual_certify,
    )

    model_b = _ConvexMetaWorldToyModel(target)
    chunk_b, plan_b, info_b = mwp.metaworld_trust_mpc_plan(
        model_b,
        _dummy_obs(),
        goal,
        5.0 * head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=8,
        dual_certify=dual_certify,
    )

    assert model_a.unroll_count == 32 and info_a["unrolls"] == 32
    assert model_b.unroll_count == 32 and info_b["unrolls"] == 32
    assert info_a["dual_certify"] is dual_certify
    assert info_a["jensen_violation"] is False
    assert np.all(plan_a >= lower[None, :]) and np.all(plan_a <= upper[None, :])
    np.testing.assert_allclose(chunk_a, chunk_b, atol=1e-6)
    np.testing.assert_allclose(plan_a, plan_b, atol=1e-6)
    np.testing.assert_allclose(info_b["selected_cost"], 25.0 * info_a["selected_cost"], rtol=1e-5)

    # Standard CEM at the same N uses the exact same 2 * N unrolls.
    model_cem = _ConvexMetaWorldToyModel(target)
    _, plan_cem, info_cem = mwp.metaworld_cem_plan(
        model_cem, _dummy_obs(), goal, head, normals, lower, upper, num_elites=4, batch_size=8
    )
    assert model_cem.unroll_count == 32 and info_cem["unrolls"] == 32
    assert np.all(plan_cem >= lower[None, :]) and np.all(plan_cem <= upper[None, :])


@pytest.mark.parametrize("dual_certify", [True, False])
def test_metaworld_trust_mpc_bimodal_barrier_splits_around_hold(dual_certify: bool):
    lower, upper, hold = mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 1)
    mode_offset = torch.tensor([0.9, 0.0, 0.0, 0.0], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)

    # Interleave exact (+/-) mirror pairs after slot 0 so both iter0 and iter1
    # preserve symmetric bimodal elites whose uniform mean sits on the barrier peak.
    rng = np.random.default_rng(11)
    pairs = rng.standard_normal((2, 7, 2, 4)).astype(np.float32)
    normals = np.zeros((2, 16, 2, 4), dtype=np.float32)
    normals[:, 1:15:2] = pairs
    normals[:, 2:15:2] = -pairs

    model = _BimodalBarrierMetaWorldToyModel(torch.from_numpy(hold), mode_offset)
    chunk, _, info = mwp.metaworld_trust_mpc_plan(
        model,
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=6,
        batch_size=8,
        stay_margin=0.0,
        dual_certify=dual_certify,
    )

    assert model.unroll_count == 32 and info["unrolls"] == 32
    assert info["jensen_violation"] is True
    if dual_certify:
        assert info["mode_mean_cost"] < info["elite_mean_cost"]
    assert abs(float(chunk[0] - hold[0])) > 0.35


def test_metaworld_planners_support_multi_goal_prototype_and_step_weights():
    lower, upper, hold = mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 1)
    target = torch.tensor([0.4, -0.3, 0.2, -0.1], dtype=torch.float32)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(19).standard_normal((2, 12, 3, 4)).astype(np.float32)

    # Four equivalent goals whose centroid is zero, with zero-mean nuisance shifts.
    rng = torch.Generator().manual_seed(5)
    nuisance = torch.randn(4, 4, 8, generator=rng)
    nuisance = nuisance - nuisance.mean(dim=0, keepdim=True)
    multi_goal = nuisance  # [K=4, P=4, D=8], centroid == zeros(4, 8)
    centroid_goal = multi_goal.mean(dim=0)  # [P=4, D=8]

    step_weights = (0.2, 0.3, 0.5)
    model_multi = _ConvexMetaWorldToyModel(target)
    chunk_multi, _, info_multi = mwp.metaworld_trust_mpc_plan(
        model_multi,
        _dummy_obs(),
        multi_goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=6,
        step_weights=step_weights,
        dual_certify=True,
    )
    model_centroid = _ConvexMetaWorldToyModel(target)
    chunk_centroid, _, info_centroid = mwp.metaworld_trust_mpc_plan(
        model_centroid,
        _dummy_obs(),
        centroid_goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=6,
        step_weights=step_weights,
        dual_certify=True,
    )

    assert info_multi["unrolls"] == 24 and info_centroid["unrolls"] == 24
    np.testing.assert_allclose(chunk_multi, chunk_centroid, atol=1e-6)
    np.testing.assert_allclose(info_multi["selected_cost"], info_centroid["selected_cost"], atol=1e-6)

    # Verify CEM also matches centroid identically under dense horizon weights.
    cem_multi = mwp.metaworld_cem_plan(
        _ConvexMetaWorldToyModel(target),
        _dummy_obs(),
        multi_goal,
        head,
        normals,
        lower,
        upper,
        num_elites=4,
        batch_size=6,
        step_weights=step_weights,
    )
    cem_centroid = mwp.metaworld_cem_plan(
        _ConvexMetaWorldToyModel(target),
        _dummy_obs(),
        centroid_goal,
        head,
        normals,
        lower,
        upper,
        num_elites=4,
        batch_size=6,
        step_weights=step_weights,
    )
    np.testing.assert_allclose(cem_multi[0], cem_centroid[0], atol=1e-6)


def test_summarize_episodes_and_paired_bootstrap():
    rows = []
    for task in range(6):
        rows.append(
            {
                "head": "identity",
                "planner": "standard_cem",
                "goal_photos": 1,
                "task": task,
                "final_error_3d_cm": 8.0 + 0.5 * task,
                "final_error_xy_cm": 7.5 + 0.5 * task,
                "best_error_3d_cm": 6.0 + 0.5 * task,
                "post_peak_drift_cm": 2.0,
                "success_5cm": 0,
                "success_3cm": 0,
                "success_2cm": 0,
                "native_success": 0,
                "unrolls": 1280,
                "planning_seconds": 1.0,
            }
        )
        rows.append(
            {
                "head": "iso_head",
                "planner": "trust_mpc",
                "goal_photos": 2,
                "task": task,
                "final_error_3d_cm": 3.0 + 0.5 * task,
                "final_error_xy_cm": 2.8 + 0.5 * task,
                "best_error_3d_cm": 2.8 + 0.5 * task,
                "post_peak_drift_cm": 0.2,
                "success_5cm": 1,
                "success_3cm": int(task == 0),
                "success_2cm": 0,
                "native_success": 1,
                "unrolls": 1280,
                "planning_seconds": 1.0,
            }
        )

    ci = paired_bootstrap_ci(np.array([-5.0, -5.0, -5.0, -5.0]), draws=200, seed=0)
    assert ci["mean"] == pytest.approx(-5.0)
    assert ci["ci_lo"] == pytest.approx(-5.0) and ci["ci_hi"] == pytest.approx(-5.0)

    summary = summarize_episodes(rows, bootstrap_draws=500, seed=0)
    assert summary["baseline_condition"] == {"head": "identity", "planner": "standard_cem", "goal_photos": 1}
    assert len(summary["conditions"]) == 2
    improved = [c for c in summary["conditions"] if c["head"] == "iso_head"][0]
    assert improved["paired_vs_baseline"]["final_error_3d_cm"]["mean"] == pytest.approx(-5.0)
    assert improved["paired_vs_baseline"]["final_error_3d_cm"]["ci_hi"] < 0.0
    assert improved["paired_vs_baseline"]["success_5cm"]["mean"] == pytest.approx(1.0)


def test_mujoco_render_puck_sensitivity_pairs():
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    from jepa_planning import metaworld_push as mp

    cfg = yaml.safe_load((ROOT / "code/configs/goal_groups.yaml").read_text())
    env = mp.make_push_env()
    try:
        goal_snap, goal_pose = gg.make_target(env, int(cfg["first_seed"]["test"]), cfg)
        bank = gg.hand_bank(cfg)["test"]
        order = gg.hand_order(cfg, int(cfg["first_seed"]["test"]), len(bank))
        photos, _ = gg.render_group(env, goal_snap, goal_pose, bank, order, 1, cfg)
        assert len(photos) == 1
        assert photos[0]["proprio"].shape == (4,)

        hand_xy = bank[photos[0]["hand_bank_index"]]
        pairs = mwp.render_puck_sensitivity_pairs(env, goal_snap, hand_xy, shifts_m=(0.02, 0.05))

        assert pairs["reference_visual"].shape == (224, 224, 3)
        assert pairs["reference_visual"].dtype == np.uint8
        assert pairs["reference_proprio"].shape == (4,)
        assert pairs["visual"].shape == (2, 224, 224, 3)
        assert pairs["proprio"].shape == (2, 4)

        # Puck x coordinate shifts by exactly 2 cm and 5 cm; y and z stay fixed.
        np.testing.assert_allclose(
            pairs["puck_pose"][:, 0] - pairs["reference_puck_pose"][0],
            [0.02, 0.05],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            pairs["puck_pose"][:, 1:3],
            np.broadcast_to(pairs["reference_puck_pose"][1:3], (2, 2)),
            atol=1e-6,
        )
        # Sawyer hand position and proprioception remain identical across puck shifts.
        np.testing.assert_allclose(
            pairs["hand_pos"],
            np.broadcast_to(pairs["reference_hand_pos"], (2, 3)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            pairs["proprio"],
            np.broadcast_to(pairs["reference_proprio"], (2, 4)),
            atol=1e-6,
        )
        # Perturbed frames visibly differ from the reference frame.
        assert not np.array_equal(pairs["visual"][0], pairs["reference_visual"])
        assert not np.array_equal(pairs["visual"][1], pairs["reference_visual"])
    finally:
        env.close()


class _PatchSelectiveMetaWorldToyModel:
    """Toy model where patch 0 tracks action[0] and patches 1..3 track a distractor target."""

    def __init__(self, puck_target: torch.Tensor, arm_distractor: torch.Tensor):
        self.device = torch.device("cpu")
        self.puck_target = puck_target.float()
        self.arm_distractor = arm_distractor.float()
        self.unroll_count = 0
        self.recorded_action_batches: list[np.ndarray] = []

    def encode(self, obs: dict) -> torch.Tensor:
        return torch.zeros(1, 1, 4, 8)

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, action_dim = actions.shape
        self.unroll_count += int(batch_size)
        self.recorded_action_batches.append(actions.permute(1, 0, 2).detach().cpu().numpy().copy())
        out = torch.zeros(horizon + 1, batch_size, 4, 8)
        for step in range(horizon):
            out[step + 1, :, 0, :action_dim] = actions[step] - self.puck_target[None, :]
            out[step + 1, :, 1:, :action_dim] = (actions[step] - self.arm_distractor[None, :])[:, None, :]
        return {"visual": out}


def test_metaworld_trust_mpc_spatial_weights_and_prior_centered_shift():
    lower, upper, hold = mwp.metaworld_action_bounds(METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, 1)
    puck_target = torch.tensor([0.8, 0.6, -0.5, 0.4], dtype=torch.float32)
    arm_distractor = torch.tensor([-0.9, -0.8, 0.7, -0.6], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)

    horizon = 4
    normals = np.random.default_rng(31).standard_normal((2, 16, horizon, 4)).astype(np.float32)

    # Uniform spatial weights match spatial_weights=None identically (terminal and dense).
    uniform_w = torch.full((4,), 0.25)
    for sw in (None, (0.1, 0.2, 0.3, 0.4)):
        chunk_none, _, info_none = mwp.metaworld_trust_mpc_plan(
            _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor),
            _dummy_obs(),
            goal,
            head,
            normals,
            lower,
            upper,
            hold,
            num_elites=4,
            batch_size=8,
            step_weights=sw,
            spatial_weights=None,
        )
        chunk_unif, _, info_unif = mwp.metaworld_trust_mpc_plan(
            _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor),
            _dummy_obs(),
            goal,
            head,
            normals,
            lower,
            upper,
            hold,
            num_elites=4,
            batch_size=8,
            step_weights=sw,
            spatial_weights=uniform_w,
        )
        np.testing.assert_allclose(chunk_unif, chunk_none, atol=1e-6)
        np.testing.assert_allclose(info_unif["selected_cost"], info_none["selected_cost"], atol=1e-6)

    # Concentrating spatial weights on patch 0 steers the plan toward puck_target
    # instead of the 3-patch arm_distractor.
    puck_w = torch.tensor([0.97, 0.01, 0.01, 0.01], dtype=torch.float32)
    chunk_uniform, _, _ = mwp.metaworld_trust_mpc_plan(
        _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor),
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=16,
        stay_margin=0.0,
        spatial_weights=None,
    )
    chunk_puck, _, _ = mwp.metaworld_trust_mpc_plan(
        _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor),
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        num_elites=4,
        batch_size=16,
        stay_margin=0.0,
        spatial_weights=puck_w,
    )
    puck_np = puck_target.numpy()
    assert float(np.linalg.norm(chunk_puck - puck_np)) < float(np.linalg.norm(chunk_uniform - puck_np))

    # Verify sample_around_prior=True and step_chunks=3 warm-start shifting.
    previous_plan = np.array(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30, 0.30],
            [0.45, -0.35, 0.25, -0.15],
        ],
        dtype=np.float32,
    )
    recording_model = _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor)
    _, plan_prior, info_prior = mwp.metaworld_trust_mpc_plan(
        recording_model,
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        previous_plan=previous_plan,
        num_elites=4,
        batch_size=16,
        stay_margin=0.0,
        spatial_weights=puck_w,
        sample_around_prior=True,
        step_chunks=3,
    )
    assert info_prior["unrolls"] == 32 and recording_model.unroll_count == 32
    assert np.all(plan_prior >= lower[None, :]) and np.all(plan_prior <= upper[None, :])

    # First batch is all 16 iter0 candidates.
    iter0_batch = recording_model.recorded_action_batches[0]
    expected_hold_seq = np.tile(hold[None, :], (horizon, 1))
    expected_shifted = np.vstack([previous_plan[3:], np.tile(hold[None, :], (3, 1))])
    expected_prior_samples = np.clip(normals[0, 2:], lower[None, None, :], upper[None, None, :])

    np.testing.assert_allclose(iter0_batch[0], expected_hold_seq, atol=1e-6)
    np.testing.assert_allclose(iter0_batch[1], expected_shifted, atol=1e-6)
    np.testing.assert_allclose(iter0_batch[2:], expected_prior_samples, atol=1e-6)

    # Confirm default sample_around_prior=False still centers around hold_sequence.
    hold_model = _PatchSelectiveMetaWorldToyModel(puck_target, arm_distractor)
    mwp.metaworld_trust_mpc_plan(
        hold_model,
        _dummy_obs(),
        goal,
        head,
        normals,
        lower,
        upper,
        hold,
        previous_plan=previous_plan,
        num_elites=4,
        batch_size=16,
        sample_around_prior=False,
        step_chunks=3,
    )
    expected_hold_samples = np.clip(
        expected_hold_seq[None] + normals[0, 2:], lower[None, None, :], upper[None, None, :]
    )
    np.testing.assert_allclose(hold_model.recorded_action_batches[0][2:], expected_hold_samples, atol=1e-6)

    # step_chunks validation.
    with pytest.raises(ValueError):
        mwp.metaworld_trust_mpc_plan(
            hold_model,
            _dummy_obs(),
            goal,
            head,
            normals,
            lower,
            upper,
            hold,
            num_elites=4,
            step_chunks=0,
        )


def test_metaworld_trust_mpc_config_and_run_episode_suite_step_chunks_and_spatial_weights(monkeypatch):
    from types import SimpleNamespace

    from experiments.run_metaworld_trust_mpc import run_episode_suite
    from jepa_planning import metaworld_push as mp

    # Verify the calibrated H=6 / step_chunks=3 protocol YAML file.
    cfg_path = ROOT / "code/configs/metaworld_trust_mpc.yaml"
    assert cfg_path.exists()
    yaml_cfg = yaml.safe_load(cfg_path.read_text())
    assert yaml_cfg["seed"] == 60000
    assert yaml_cfg["controls_per_chunk"] == 5
    assert yaml_cfg["cem_horizon"] == 6
    assert yaml_cfg["step_chunks"] == 3
    assert yaml_cfg["replans"] == 3
    assert yaml_cfg["cem_iterations"] == 2
    assert yaml_cfg["cem_samples"] == 64
    assert yaml_cfg["cem_elites"] == 8
    assert yaml_cfg["batch_size"] == 64
    assert yaml_cfg["stay_margin"] == pytest.approx(0.0)
    assert yaml_cfg["sample_around_prior"] is True
    assert yaml_cfg["dual_certify"] is True
    assert yaml_cfg["replicates"] == 1

    # Mock env + patch-selective toy model to test run_episode_suite on CPU.
    controls_per_chunk = 5
    action_dim = 4 * controls_per_chunk

    class _MockPushEnv:
        def __init__(self):
            self.executed_raw_actions: list[np.ndarray] = []
            self.puck_xyz = np.array([0.0, 0.60, 0.02], dtype=np.float64)

        def reset_puck(self):
            self.executed_raw_actions = []
            self.puck_xyz = np.array([0.0, 0.60, 0.02], dtype=np.float64)

        def step(self, raw_action: np.ndarray):
            self.executed_raw_actions.append(np.asarray(raw_action, dtype=np.float32).copy())
            # Move puck slightly proportional to first two control dims so trajectory depends on actions.
            self.puck_xyz = self.puck_xyz + np.array([0.002 * raw_action[0], 0.002 * raw_action[1], 0.0])
            return np.zeros(39, dtype=np.float32), 0.0, False, False, {"success": False}

    class _EpisodeToyModel:
        def __init__(self, puck_target: torch.Tensor, arm_distractor: torch.Tensor):
            self.device = torch.device("cpu")
            self.action_dim = action_dim
            self.preprocessor = SimpleNamespace(
                action_mean=torch.from_numpy(METAWORLD_ACTION_MEAN.copy()),
                action_std=torch.from_numpy(METAWORLD_ACTION_STD.copy()),
            )
            self.puck_target = puck_target.float()
            self.arm_distractor = arm_distractor.float()
            self.unroll_count = 0
            self.iter0_batches: list[np.ndarray] = []

        def encode(self, obs: dict) -> torch.Tensor:
            return torch.zeros(1, 1, 4, 8)

        def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
            horizon, batch_size, _ = actions.shape
            self.unroll_count += int(batch_size)
            if batch_size == 16:
                self.iter0_batches.append(actions.permute(1, 0, 2).detach().cpu().numpy().copy())
            out = torch.zeros(horizon + 1, batch_size, 4, 8)
            # Terminal visual state reflects cumulative mean control across the full horizon and chunk.
            mean_ctrl = actions.reshape(horizon, batch_size, controls_per_chunk, 4).mean(dim=(0, 2))
            out[-1, :, 0, :4] = mean_ctrl - self.puck_target[None, :]
            out[-1, :, 1:, :4] = (mean_ctrl - self.arm_distractor[None, :])[:, None, :]
            return {"visual": out}

    monkeypatch.setattr(mp, "restore", lambda env, snap: env.reset_puck())
    monkeypatch.setattr(mp, "observe", lambda env: _dummy_obs())
    monkeypatch.setattr(mp, "puck_pose", lambda env: np.concatenate([env.puck_xyz, [1.0, 0.0, 0.0, 0.0]]))

    tasks = {
        "group": np.array([0], dtype=np.int64),
        "target_seed": np.array([60000], dtype=np.int64),
        "goal_puck": np.array([[0.05, 0.65, 0.02]], dtype=np.float64),
    }
    goals = torch.zeros((1, 1, 4, 8), dtype=torch.float32)
    head = torch.eye(8) / np.sqrt(8.0)
    puck_target = torch.tensor([0.8, 0.7, 0.0, 0.0], dtype=torch.float32)
    arm_distractor = torch.tensor([-0.8, -0.7, 0.0, 0.0], dtype=torch.float32)
    puck_weights = torch.tensor([0.97, 0.01, 0.01, 0.01], dtype=torch.float32)

    suite_cfg = {
        "seed": 60000,
        "controls_per_chunk": controls_per_chunk,
        "cem_horizon": 6,
        "step_chunks": 3,
        "replans": 3,
        "cem_iterations": 2,
        "cem_samples": 16,
        "cem_elites": 4,
        "batch_size": 16,
        "stay_margin": 0.0,
        "sample_around_prior": True,
        "dual_certify": True,
        "replicates": 1,
    }

    # Trust-MPC with step_chunks=3, sample_around_prior=True, and puck spatial_weights.
    env_puck = _MockPushEnv()
    model_puck = _EpisodeToyModel(puck_target, arm_distractor)
    rows_puck = run_episode_suite(
        env_puck,
        model_puck,
        head,
        "anova_head",
        "trust_mpc",
        1,
        tasks,
        [{}],
        goals,
        suite_cfg,
        spatial_weights=puck_weights,
    )
    assert len(rows_puck) == 1
    assert rows_puck[0]["sim_steps"] == 3 * 3 * controls_per_chunk
    assert len(env_puck.executed_raw_actions) == 45
    assert rows_puck[0]["unrolls"] == 3 * (2 * 16)
    # Verify sample_around_prior=True and step_chunks=3 warm-start padding on replan 1.
    lower_20, upper_20, hold_20 = mwp.metaworld_action_bounds(
        METAWORLD_ACTION_MEAN, METAWORLD_ACTION_STD, controls_per_chunk
    )
    assert len(model_puck.iter0_batches) == 3
    np.testing.assert_allclose(model_puck.iter0_batches[0][0], np.tile(hold_20[None, :], (6, 1)), atol=1e-6)
    # Replan 1 slot 1 has the last 3 chunks padded with hold_20 because step_chunks=3.
    np.testing.assert_allclose(model_puck.iter0_batches[1][1, 3:], np.tile(hold_20[None, :], (3, 1)), atol=1e-6)
    assert np.all(model_puck.iter0_batches[0] >= lower_20[None, None, :])
    assert np.all(model_puck.iter0_batches[0] <= upper_20[None, None, :])

    # Confirm uniform (spatial_weights=None) steers toward the opposite arm distractor.
    env_unif = _MockPushEnv()
    model_unif = _EpisodeToyModel(puck_target, arm_distractor)
    rows_unif = run_episode_suite(
        env_unif,
        model_unif,
        head,
        "uniform_head",
        "trust_mpc",
        1,
        tasks,
        [{}],
        goals,
        suite_cfg,
        spatial_weights=None,
    )
    assert rows_puck[0]["final_error_3d_cm"] < rows_unif[0]["final_error_3d_cm"]

    # Confirm standard_cem also respects step_chunks=3 and spatial_weights.
    env_cem = _MockPushEnv()
    model_cem = _EpisodeToyModel(puck_target, arm_distractor)
    rows_cem = run_episode_suite(
        env_cem,
        model_cem,
        head,
        "anova_head",
        "standard_cem",
        1,
        tasks,
        [{}],
        goals,
        suite_cfg,
        spatial_weights=puck_weights,
    )
    assert rows_cem[0]["sim_steps"] == 45
    assert rows_cem[0]["unrolls"] == 3 * (2 * 16)

    # Confirm default step_chunks=1 stays backwards-compatible (1 chunk = 5 controls per replan).
    cfg_default_chunks = dict(suite_cfg)
    del cfg_default_chunks["step_chunks"]
    env_default = _MockPushEnv()
    rows_default = run_episode_suite(
        env_default,
        _EpisodeToyModel(puck_target, arm_distractor),
        head,
        "anova_head",
        "trust_mpc",
        1,
        tasks,
        [{}],
        goals,
        cfg_default_chunks,
        spatial_weights=puck_weights,
    )
    assert rows_default[0]["sim_steps"] == 3 * 1 * controls_per_chunk
