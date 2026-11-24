"""Tests for PushT head-planning metrics: T polygon, coverage, physical metrics, selection rows, head costs."""

import numpy as np
import torch

from jepa_planning import linear_head as lh
from jepa_planning import pusht_head_planning as hp
from jepa_planning.goal_costs import mean_squared_goal_distance


def test_coverage_is_one_for_same_pose_and_zero_far_away():
    pose = np.array([250.0, 260.0, 0.7])
    assert np.isclose(hp.coverage(pose, pose), 1.0)
    assert hp.coverage(pose + np.array([400.0, 0.0, 0.0]), pose) == 0.0
    assert np.isclose(hp.coverage(pose + np.array([0.0, 0.0, 2 * np.pi]), pose), 1.0)


def test_t_polygon_area_matches_the_env_t():
    # Bar 120 x 30 plus stem 30 x 90 (add_tee with scale 30).
    assert np.isclose(hp.t_polygon(np.zeros(3)).area, 120 * 30 + 30 * 90)


def test_physical_metrics_and_selection_regret():
    target = np.array([200.0, 200.0, 0.0])
    poses = np.array([[230.0, 200.0, 0.0], [205.0, 200.0, 0.0], [200.0, 200.0, np.pi / 4]])
    metrics = hp.physical_metrics(poses, target)
    np.testing.assert_allclose(metrics["position_error"], [30, 5, 0])
    np.testing.assert_array_equal(metrics["success_10px"], [0, 1, 1])
    np.testing.assert_array_equal(metrics["pose_success"], [0, 1, 0])
    row = hp.selection_rows(np.array([0.0, 1.0, 2.0]), metrics)
    assert row["selected"] == 0 and row["regret"] == 30.0
    assert row["combined_regret"] == metrics["combined_error"][0] - metrics["combined_error"].min()


def test_head_costs_with_identity_head_equal_plain_mse():
    generator = torch.Generator().manual_seed(0)
    candidates = torch.randn(5, 4, 8, generator=generator)
    goal = torch.randn(4, 8, generator=generator)
    expected = mean_squared_goal_distance(candidates, goal[None].expand_as(candidates))
    torch.testing.assert_close(hp.head_costs(candidates, goal, lh.identity_head(8)), expected)


def test_head_catalog_lists_saved_heads_and_matched_controls(tmp_path):
    (tmp_path / "heads").mkdir()
    config = {"rank": 4, "compression": 2.0, "B": 16, "M": 8, "K": 2, "resample": 0, "shuffled_labels": False}
    lh.save_head(tmp_path / "heads" / "rank4_B16_M8_K2_r0.pt", lh.random_head(4, 8, 1), config)
    lh.save_head(tmp_path / "heads" / "rank4_B16_M8_K2_r0_shuffled.pt", lh.random_head(4, 8, 2),
                 {**config, "shuffled_labels": True})
    catalog = lh.head_catalog([tmp_path], dim=8, control_seeds=[0, 1])
    methods = [info["method"] for info, _ in catalog]
    assert methods.count("plain_mse") == 1 and methods.count("linear_head") == 1
    assert methods.count("linear_head_shuffled") == 1
    assert methods.count("random_projection") == 2 and methods.count("coordinate_subset") == 2
    assert all(np.isclose(float(torch.linalg.matrix_norm(head)), 1.0) for _, head in catalog)
