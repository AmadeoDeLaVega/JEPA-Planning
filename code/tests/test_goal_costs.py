"""Unit tests for feature-space goal-cost computation.

Run this module directly with
``.venv/bin/python -m unittest code/tests/test_goal_costs.py -v`` or run the
complete suite with
``.venv/bin/python -m unittest discover -s code/tests -v``.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from jepa_planning.goal_costs import (  # noqa: E402
    mean_multi_goal_distance,
    mean_squared_goal_distance,
    pairwise_squared_goal_distance,
)


class MeanSquaredGoalDistanceTest(unittest.TestCase):
    """Verify numerical behavior and input validation for goal-feature costs."""

    def test_identical_candidate_and_goal_have_zero_cost(self) -> None:
        candidates = torch.tensor([[1.0, -2.0, 3.0]])

        costs = mean_squared_goal_distance(candidates, candidates.clone())

        torch.testing.assert_close(costs, torch.zeros(1))

    def test_shared_goal_is_broadcast_across_candidates(self) -> None:
        candidates = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        shared_goal = torch.zeros((1, 2))

        costs = mean_squared_goal_distance(candidates, shared_goal)

        torch.testing.assert_close(costs, torch.tensor([5.0, 10.0]))

    def test_candidate_specific_goals_are_applied_pairwise(self) -> None:
        candidates = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        goals = torch.tensor([[0.0, 1.0], [1.0, 2.0]])

        costs = mean_squared_goal_distance(candidates, goals)

        torch.testing.assert_close(costs, torch.tensor([2.5, 2.5]))

    def test_all_non_batch_dimensions_are_averaged(self) -> None:
        candidates = torch.ones((2, 1, 2, 2, 3))
        shared_goal = torch.zeros((1, 1, 2, 2, 3))

        costs = mean_squared_goal_distance(candidates, shared_goal)

        torch.testing.assert_close(costs, torch.ones(2))

    def test_rejects_incompatible_feature_shapes(self) -> None:
        candidates = torch.zeros((2, 3))
        goals = torch.zeros((1, 4))

        with self.assertRaisesRegex(ValueError, "feature dimensions must match"):
            mean_squared_goal_distance(candidates, goals)

    def test_rejects_incompatible_goal_batch(self) -> None:
        candidates = torch.zeros((2, 3))
        goals = torch.zeros((3, 3))

        with self.assertRaisesRegex(ValueError, "one goal or one goal per candidate"):
            mean_squared_goal_distance(candidates, goals)

    def test_rejects_empty_candidate_batch(self) -> None:
        candidates = torch.empty((0, 3))
        shared_goal = torch.zeros((1, 3))

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            mean_squared_goal_distance(candidates, shared_goal)


class MeanMultiGoalDistanceTest(unittest.TestCase):
    """Verify exact averaging and validation for equivalent goal sets."""

    def test_one_goal_matches_single_goal_cost(self) -> None:
        candidates = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        goal = torch.tensor([[0.5, 1.5]])

        expected = mean_squared_goal_distance(candidates, goal)
        actual = mean_multi_goal_distance(candidates, goal)

        torch.testing.assert_close(actual, expected)

    def test_matches_brute_force_candidate_goal_average(self) -> None:
        candidates = torch.tensor([[0.0, 2.0], [4.0, 2.0]])
        goals = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]])

        expected = (candidates[:, None] - goals[None]).square().mean(dim=(1, 2))
        actual = mean_multi_goal_distance(candidates, goals)

        torch.testing.assert_close(actual, expected)

    def test_pairwise_costs_can_be_reused_for_goal_subsets(self) -> None:
        candidates = torch.tensor([[0.0, 2.0], [4.0, 2.0]])
        goals = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]])

        pairwise = pairwise_squared_goal_distance(candidates, goals)
        expected = (candidates[:, None] - goals[None]).square().mean(dim=2)

        torch.testing.assert_close(pairwise, expected)
        torch.testing.assert_close(pairwise[:, [0, 2]].mean(dim=1), torch.tensor([6.0, 6.0]))

    def test_supports_patch_shaped_features(self) -> None:
        candidates = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
        goals = torch.stack((torch.zeros((2, 3, 4)), torch.ones((2, 3, 4))))

        expected = (candidates[:, None] - goals[None]).square().mean(dim=(1, 2, 3, 4))

        torch.testing.assert_close(
            mean_multi_goal_distance(candidates, goals), expected
        )

    def test_rejects_empty_goal_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "goal feature batch must not be empty"):
            mean_multi_goal_distance(torch.zeros((2, 3)), torch.empty((0, 3)))

    def test_rejects_mixed_dtypes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dtype"):
            mean_multi_goal_distance(
                torch.zeros((2, 3), dtype=torch.float32),
                torch.zeros((2, 3), dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
