"""Tests for the temporal closed-loop comparison."""

import numpy as np
import pytest
import torch

from experiments.run_pusht_temporal_closed_loop import method_costs, population_costs
from experiments.summarize_pusht_temporal_closed_loop import compare, merge_runs


class FakeModel:
    """Predicts the summed actions as endpoint features."""

    device = torch.device("cpu")

    def unroll(self, encoded, actions):
        features = actions.sum(dim=(0, 2))[:, None].expand(-1, 4)  # actions are [horizon, batch, action]
        return {"visual": features[None]}


def test_population_costs_do_not_depend_on_batch_size():
    actions = np.random.default_rng(0).standard_normal((5, 2, 3)).astype(np.float32)
    goal = torch.zeros(1, 4)
    cost = lambda candidates, goals: (candidates - goals).abs().sum(dim=1)

    whole = population_costs(FakeModel(), {}, actions, goal, cost, batch_size=5)
    split = population_costs(FakeModel(), {}, actions, goal, cost, batch_size=2)

    for left, right in zip(whole, split):
        assert left.shape == (5,)
        np.testing.assert_allclose(left, right, rtol=1e-6)


def test_methods_use_their_own_cost():
    mse, temporal = np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])

    np.testing.assert_array_equal(method_costs("mse", mse, temporal, 0.5), mse)
    np.testing.assert_array_equal(method_costs("temporal", mse, temporal, 0.5), temporal)
    np.testing.assert_allclose(method_costs("blend", mse, temporal, 0.5), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        method_costs("other", mse, temporal, 0.5)


def rows(errors):
    return [{"seed": seed, "episode": episode, "method": method, "final_error": error}
            for (seed, episode, method), error in errors.items()]


def test_merge_requires_mse_to_reproduce_l2_g1():
    temporal = rows({("1", 0, "mse"): 5.0, ("1", 0, "temporal"): 4.0})

    merge_runs(temporal, rows({("1", 0, "l2_g1"): 5.0, ("1", 0, "l2_g4"): 3.0}))
    with pytest.raises(ValueError):
        merge_runs(temporal, rows({("1", 0, "l2_g1"): 6.0, ("1", 0, "l2_g4"): 3.0}))


def test_positive_improvement_favors_the_method():
    errors = {}
    for seed in "12345":
        for episode in range(2):
            errors.update({(seed, episode, "mse"): 10.0, (seed, episode, "l2_g1"): 10.0, (seed, episode, "temporal"): 7.0})
    merged = merge_runs(rows({k: v for k, v in errors.items() if k[2] != "l2_g1"}), rows({k: v for k, v in errors.items() if k[2] == "l2_g1"}))

    result = compare(merged, "temporal", "mse", np.random.default_rng(0), 100)

    assert result["improvement_mean"] == 3.0
    assert result["positive_banks"] == 5 and result["confirmed"]
