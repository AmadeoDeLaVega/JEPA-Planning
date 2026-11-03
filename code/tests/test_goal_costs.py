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

from jepa_planning.goal_costs import mean_squared_goal_distance  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
