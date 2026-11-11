"""Unit tests for the quasimetric goal cost, its objective, and its samplers."""

import numpy as np
import pytest
import torch

from jepa_planning.quasimetric_cost import (
    QuasimetricCost,
    QuasimetricObjective,
    interval_quasimetric,
    sample_frame_pairs,
    sample_transitions,
)


@pytest.fixture
def cost():
    torch.manual_seed(0)
    return QuasimetricCost(feature_dim=8, grid=4, width=8, hidden=32, components=4, component_dim=4)


def features(count, seed=0):
    return torch.randn(count, 4 * 4 * 8, generator=torch.Generator().manual_seed(seed))


def test_interval_distance_is_the_length_of_the_union():
    def distance(start, goal):
        return interval_quasimetric(torch.tensor([[start]]), torch.tensor([[goal]])).item()

    assert distance([0.0, 2.0], [1.0, 3.0]) == pytest.approx(2.0)  # disjoint: [0, 1] and [2, 3]
    assert distance([0.0, 1.0], [2.0, 3.0]) == pytest.approx(3.0)  # overlapping: [0, 3]
    assert distance([0.0, 1.0], [3.0, 2.0]) == pytest.approx(3.0)  # nested: [0, 3]
    assert distance([2.0, 3.0], [1.0, 0.0]) == 0.0  # start above goal everywhere
    assert distance([0.0, 5.0], [1.0, 0.0]) == pytest.approx(1.0)  # only the open coordinate counts


def test_interval_distance_keeps_the_triangle_inequality():
    generator = torch.Generator().manual_seed(0)
    x, y, z = (torch.randn(2000, 3, 5, generator=generator) for _ in range(3))

    direct = interval_quasimetric(x, z)
    detour = interval_quasimetric(x, y) + interval_quasimetric(y, z)

    assert torch.all(direct <= detour + 1e-5)


def test_cost_is_zero_to_itself_and_never_negative(cost):
    candidates, goals = features(16), features(16, seed=1)

    torch.testing.assert_close(cost(candidates, candidates), torch.zeros(16))
    assert torch.all(cost(candidates, goals) >= 0)


def test_cost_keeps_the_triangle_inequality(cost):
    with torch.no_grad():
        x, y, z = features(64), features(64, seed=1), features(64, seed=2)
        assert torch.all(cost(x, z) <= cost(x, y) + cost(y, z) + 1e-5)


def test_cost_can_be_asymmetric(cost):
    first, second = features(1), features(1, seed=1)

    assert not torch.allclose(cost(first, second), cost(second, first))


def test_cost_broadcasts_one_goal_across_candidates(cost):
    candidates, goal = features(3), features(1, seed=1)

    torch.testing.assert_close(cost(candidates, goal), cost(candidates, goal.expand(3, -1)))


def test_cost_rejects_mismatched_goal_batch(cost):
    with pytest.raises(ValueError, match="one goal or one goal per candidate"):
        cost(features(3), features(2))


def test_cost_rejects_grid_that_cannot_be_downsampled():
    with pytest.raises(ValueError, match="divisible"):
        QuasimetricCost(grid=6)


def test_multiplier_grows_only_while_steps_cost_too_much(cost):
    objective = QuasimetricObjective(horizon=10.0)
    frames, goals = features(8), features(8, seed=1)

    def multiplier_gradient(step_length):
        objective.zero_grad()
        with torch.no_grad():  # rescale the latents so a step costs `step_length` on average
            layer = cost.encoder[-1]
            scale = step_length / cost(frames, goals).mean()
            layer.weight.mul_(scale), layer.bias.mul_(scale)
        loss, stats = objective(cost, frames, goals, frames, goals)
        loss.backward()
        assert stats["step_distance"] == pytest.approx(step_length, rel=1e-3)
        return objective.raw_multiplier.grad.item()

    assert multiplier_gradient(5.0) < 0  # steps cost too much: descent raises the multiplier
    assert multiplier_gradient(0.5) > 0  # constraint met: descent lowers it


def test_objective_rejects_invalid_settings():
    with pytest.raises(ValueError):
        QuasimetricObjective(horizon=1.0)
    with pytest.raises(ValueError):
        QuasimetricObjective(horizon=10.0, slack=0.0)


def test_cost_learns_a_shortcut_that_no_single_episode_shows():
    """Two recordings: 0-1-2-3-4-5, and 0-6-5. The fewest steps from 0 to 5 is two.

    Elapsed-time labels from the first recording would put state 5 five steps
    from state 0, further than state 4. Stitched recorded steps put it closer.
    """
    torch.manual_seed(0)
    cost = QuasimetricCost(feature_dim=8, grid=4, width=8, hidden=64, components=8, component_dim=8)
    objective = QuasimetricObjective(horizon=12.0)
    states = features(7)
    frames = states[[0, 1, 2, 3, 4, 5, 0, 6, 5]]
    counts = np.array([6, 3])
    optimizer = torch.optim.Adam([*cost.parameters(), *objective.parameters()], lr=3e-3)
    rng = np.random.default_rng(0)

    for _ in range(1500):
        start, after = sample_transitions(counts, 64, rng)
        sources, goals = sample_frame_pairs(counts, 64, rng)
        loss, _ = objective(cost, frames[start], frames[after], frames[sources], frames[goals])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        d = lambda start, goal: cost(states[[start]], states[[goal]]).item()
        # The slack lets a recorded step cost a little over one, so compare to the
        # learned step lengths along each route rather than to whole numbers.
        shortcut = d(0, 6) + d(6, 5)
        long_way = sum(d(state, state + 1) for state in range(5))
        assert d(0, 5) == pytest.approx(shortcut, rel=0.1)
        assert d(0, 5) < d(0, 4) < long_way + 0.1
        assert d(5, 0) > long_way  # no recording goes from 5 back to 0


def episode_of(frames, counts):
    return np.searchsorted(np.cumsum(counts), frames, side="right")


def test_transitions_are_consecutive_frames_of_one_episode():
    counts = np.array([5, 1, 30, 12])

    frames, after = sample_transitions(counts, 500, np.random.default_rng(0))

    np.testing.assert_array_equal(after, frames + 1)
    np.testing.assert_array_equal(episode_of(frames, counts), episode_of(after, counts))
    assert not np.any(episode_of(frames, counts) == 1)  # the single-frame episode has no transition


def test_transitions_cover_episodes_in_proportion_to_their_length():
    counts = np.array([3, 21])  # 2 and 20 transitions

    frames, _ = sample_transitions(counts, 20000, np.random.default_rng(0))

    assert np.mean(episode_of(frames, counts) == 0) == pytest.approx(2 / 22, abs=0.01)


def test_transitions_need_an_episode_with_two_frames():
    with pytest.raises(ValueError):
        sample_transitions(np.array([1, 1]), 4, np.random.default_rng(0))


def test_frame_pairs_stay_inside_the_cache():
    sources, goals = sample_frame_pairs(np.array([3, 4]), 500, np.random.default_rng(0))

    assert min(sources.min(), goals.min()) == 0
    assert max(sources.max(), goals.max()) == 6


def test_samplers_repeat_with_the_same_seed():
    counts = np.array([5, 9])
    for sampler in (sample_transitions, sample_frame_pairs):
        first = sampler(counts, 20, np.random.default_rng(3))
        again = sampler(counts, 20, np.random.default_rng(3))
        for left, right in zip(first, again):
            np.testing.assert_array_equal(left, right)
