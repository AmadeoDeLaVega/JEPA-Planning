"""Fine-grained tests for the PushT multi-goal experiment protocol."""

import numpy as np
import pytest
import torch

from jepa_planning.pusht_multi_goal import (
    cem_standard_normals,
    evaluate_goal_counts,
    goal_subsets,
    physical_errors,
    sample_action_chunks,
)


def test_goal_subsets_exhaust_each_requested_count():
    subsets = goal_subsets(4, (1, 2, 4))

    assert sum(len(subset) == 1 for subset in subsets) == 4
    assert sum(len(subset) == 2 for subset in subsets) == 6
    assert sum(len(subset) == 4 for subset in subsets) == 1
    assert len(set(subsets)) == len(subsets)


@pytest.mark.parametrize("counts", [(), (0,), (5,), (1, 1), (1.5,)])
def test_goal_subsets_reject_invalid_counts(counts):
    with pytest.raises(ValueError):
        goal_subsets(4, counts)


def test_physical_errors_wrap_angles_and_ignore_agent_pose():
    candidates = np.asarray([[10.0, 20.0, 2 * np.pi - 0.1], [13.0, 24.0, 0.5]])
    goal = np.asarray([10.0, 20.0, 0.1])

    position, angle, combined, success = physical_errors(candidates, goal)

    np.testing.assert_allclose(position, [0.0, 5.0])
    np.testing.assert_allclose(angle, [0.2, 0.4])
    np.testing.assert_allclose(combined, np.sqrt(position**2 + (50 * angle) ** 2))
    np.testing.assert_array_equal(success, [True, False])


def test_evaluate_goal_counts_averages_all_goal_subsets():
    candidates = torch.tensor([[0.0], [4.0]])
    goals = torch.tensor([[0.0], [2.0], [4.0]])
    poses = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = np.asarray([0.0, 0.0, 0.0])

    records = evaluate_goal_counts(candidates, goals, poses, target, (1, 3))

    assert len(records) == 4
    assert {row["goal_subset"] for row in records if row["goal_count"] == 1} == {"0", "1", "2"}
    full_set = next(row for row in records if row["goal_count"] == 3)
    assert full_set["goal_subset"] == "0;1;2"
    assert full_set["selected"] == 0
    assert full_set["regret"] == 0.0


def test_action_chunks_are_seeded_bounded_and_smoothly_shaped():
    first = sample_action_chunks(np.random.default_rng(4), 3, 2, 5, 1.5)
    again = sample_action_chunks(np.random.default_rng(4), 3, 2, 5, 1.5)

    np.testing.assert_array_equal(first, again)
    assert first.shape == (3, 2, 5, 2)
    assert np.max(np.abs(first)) <= 1.5


def test_cem_common_noise_is_repeatable_and_replan_specific():
    kwargs = dict(seed=10, episode=2, iterations=2, samples=9, horizon=2, action_dim=10)
    first = cem_standard_normals(replan=1, **kwargs)
    again = cem_standard_normals(replan=1, **kwargs)
    other = cem_standard_normals(replan=2, **kwargs)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (2, 9, 2, 10)
