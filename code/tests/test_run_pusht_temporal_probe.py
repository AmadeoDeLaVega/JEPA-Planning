"""Tests for the temporal-versus-MSE fixed-bank comparison."""

import numpy as np
import pytest
import torch

from experiments.run_pusht_temporal_probe import evaluate, load_cost, mse_costs, task_record, temporal_costs
from jepa_planning.temporal_cost import TemporalCost


def test_mse_costs_score_every_candidate_against_every_goal():
    candidates = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    goals = torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    np.testing.assert_allclose(mse_costs(candidates, goals), [[0, 4, 2], [2, 2, 4]])


def test_temporal_costs_match_scoring_one_goal_at_a_time():
    torch.manual_seed(0)
    cost = TemporalCost(feature_dim=8, grid=4, width=4, hidden=16).eval()
    candidates, goals = torch.randn(3, 128), torch.randn(2, 128)

    matrix = temporal_costs(cost, candidates, goals)

    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(matrix[:, 1], cost(candidates, goals[1:2]).detach().numpy(), rtol=1e-5)


def test_load_cost_restores_trained_weights(tmp_path):
    cost = TemporalCost(width=4, hidden=16)
    torch.save({"state_dict": cost.state_dict(), "config": {"width": 4, "hidden": 16}}, tmp_path / "cost.pt")

    restored = load_cost(tmp_path / "cost.pt", torch.device("cpu"))

    assert not restored.training
    torch.testing.assert_close(restored.goal_projection.weight, cost.goal_projection.weight)


POSES = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
GOAL = np.zeros(3)


def test_task_record_averages_photos_and_reports_sensitivity():
    costs = np.array([[0.0, 1.0], [1.0, 0.0]])  # photo one picks the best candidate
    other = np.array([[1.0, 1.0], [0.0, 0.0]])

    record = task_record(costs, other, POSES, GOAL)

    assert record["regret"] == 50.0
    assert record["success"] == 0.5
    assert record["photo_agreement"] == 0.0
    assert record["target_response"] == 0.5


def test_constant_cost_cannot_pass_as_a_useful_invariant_cost():
    constant = np.ones((2, 2))

    record = task_record(constant, constant, POSES, GOAL)

    assert record["photo_agreement"] == 1.0
    assert record["target_response"] == 0.0


@pytest.fixture
def setup():
    config = {"tasks": 3, "candidates_per_task": 2, "goals_per_task": 2}
    bank = {"candidate_poses": np.tile(POSES, (3, 1)), "goal_poses": np.zeros((3, 3))}
    return config, bank


def test_evaluate_gives_each_method_its_own_task_and_the_next_target(setup):
    config, bank = setup
    features = torch.arange(6.0)[:, None]
    goals = torch.arange(6.0)[:, None] * 10
    seen = []

    def score(candidates, goal_set):
        seen.append((candidates[0, 0].item(), goal_set[0, 0].item()))
        return np.array([[0.0, 0.0], [1.0, 1.0]])

    rows = evaluate({"a": score, "b": score}, features, goals, bank, config)

    assert [(row["method"], row["task"]) for row in rows] == [
        ("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2),
    ]
    # Task 2 wraps around, so its comparison target comes from task 0.
    assert seen[-2:] == [(4.0, 40.0), (4.0, 0.0)]
