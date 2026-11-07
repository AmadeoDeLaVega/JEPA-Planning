"""Unit tests for the temporal goal cost, pair sampling, and sensitivity metrics."""

import numpy as np
import pytest
import torch

from jepa_planning.temporal_cost import TemporalCost, one_step_closer, photo_agreement, sample_pairs, target_response


@pytest.fixture
def cost():
    torch.manual_seed(0)
    return TemporalCost(feature_dim=8, grid=4, width=4, hidden=16)


def features(count, seed=0):
    return torch.randn(count, 4 * 4 * 8, generator=torch.Generator().manual_seed(seed))


def test_cost_broadcasts_one_goal_across_candidates(cost):
    candidates, goal = features(3), features(1, seed=1)

    shared = cost(candidates, goal)
    paired = cost(candidates, goal.expand(3, -1))

    assert shared.shape == (3,)
    torch.testing.assert_close(shared, paired)


def test_cost_is_never_negative(cost):
    assert torch.all(cost(features(16), features(16, seed=1)) >= 0)


def test_cost_treats_candidate_and_goal_differently(cost):
    first, second = features(1), features(1, seed=1)

    assert not torch.allclose(cost(first, second), cost(second, first))


def test_cost_rejects_mismatched_goal_batch(cost):
    with pytest.raises(ValueError, match="one goal or one goal per candidate"):
        cost(features(3), features(2))


def test_cost_rejects_grid_that_cannot_be_downsampled():
    with pytest.raises(ValueError, match="divisible"):
        TemporalCost(grid=6)


def test_cost_learns_a_gap_from_features(cost):
    start, near, far = features(1), features(1, seed=1), features(1, seed=2)
    candidates, goals = torch.cat((start, start)), torch.cat((near, far))
    labels = torch.tensor([1.0, 9.0])
    optimizer = torch.optim.Adam(cost.parameters(), lr=1e-2)

    for _ in range(200):
        loss = (cost(candidates, goals) - labels).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.testing.assert_close(cost(candidates, goals), labels, atol=0.2, rtol=0)


def episode_of(frames, counts):
    return np.searchsorted(np.cumsum(counts), frames, side="right")


def test_same_episode_labels_equal_frame_gaps():
    counts = np.array([5, 30, 12])

    starts, goals, labels = sample_pairs(counts, 500, 8, 0.0, np.random.default_rng(0))

    np.testing.assert_array_equal(goals - starts, labels)
    np.testing.assert_array_equal(episode_of(starts, counts), episode_of(goals, counts))
    assert labels.min() == 0
    assert labels.max() == 8


def test_far_pairs_cross_episodes_and_use_the_far_label():
    counts = np.array([5, 30, 12])

    starts, goals, labels = sample_pairs(counts, 200, 8, 1.0, np.random.default_rng(0))

    assert np.all(episode_of(starts, counts) != episode_of(goals, counts))
    assert np.all(labels == 8)


def test_pairs_stay_inside_the_cache_for_single_frame_episodes():
    counts = np.array([1, 1])

    starts, goals, _ = sample_pairs(counts, 50, 8, 0.5, np.random.default_rng(0))

    assert max(starts.max(), goals.max()) <= 1


def test_pairs_repeat_with_the_same_seed():
    first = sample_pairs(np.array([5, 9]), 20, 4, 0.5, np.random.default_rng(3))
    again = sample_pairs(np.array([5, 9]), 20, 4, 0.5, np.random.default_rng(3))

    for left, right in zip(first, again):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "counts,count,max_gap,far",
    [([5], 4, 2, 0.0), ([5, 0], 4, 2, 0.0), ([5, 5], 0, 2, 0.0), ([5, 5], 4, 0, 0.0), ([5, 5], 4, 2, 1.5)],
)
def test_sample_pairs_rejects_invalid_arguments(counts, count, max_gap, far):
    with pytest.raises(ValueError):
        sample_pairs(np.array(counts), count, max_gap, far, np.random.default_rng(0))


def test_closer_frames_are_one_step_nearer_the_same_goal():
    counts = np.array([5, 30, 12])
    starts, goals, _ = sample_pairs(counts, 500, 8, 0.3, np.random.default_rng(0))

    ordered, closer = one_step_closer(starts, goals, counts)

    np.testing.assert_array_equal(closer, starts[ordered] + 1)
    assert np.all(closer <= goals[ordered])
    np.testing.assert_array_equal(episode_of(closer, counts), episode_of(goals[ordered], counts))


def test_far_and_zero_gap_pairs_have_no_closer_frame():
    counts = np.array([4, 4])
    starts, goals = np.array([0, 2, 3, 1]), np.array([5, 2, 3, 3])

    ordered, _ = one_step_closer(starts, goals, counts)

    np.testing.assert_array_equal(ordered, [3])


def test_photo_agreement_counts_matching_photo_pairs():
    costs = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

    assert photo_agreement(costs) == pytest.approx(1 / 3)


def test_photo_agreement_needs_two_photos():
    with pytest.raises(ValueError):
        photo_agreement(np.zeros((3, 1)))


def test_constant_cost_agrees_but_does_not_respond_to_the_target():
    constant = np.ones((4, 3))

    assert photo_agreement(constant) == 1.0
    assert target_response(constant, constant) == 0.0


def test_target_response_counts_changed_selections():
    costs = np.array([[0.0, 0.0], [1.0, 1.0]])
    other = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert target_response(costs, other) == 0.5


def test_target_response_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        target_response(np.zeros((2, 2)), np.zeros((2, 3)))
