"""CPU checks for the linear goal head, its InfoNCE loss and allocation sampling."""

import numpy as np
import pytest
import torch

from jepa_planning.goal_costs import linear_goal_distance, mean_squared_goal_distance
from jepa_planning.linear_head import (
    NUM_NEGATIVES, AllocationSampler, LinearGoalHead, allocation_grid, coordinate_subset_head,
    head_distance, identity_head, info_nce_loss, load_head, normalized, random_head, save_head,
    train_step,
)


def test_identity_head_reproduces_plain_mse():
    torch.manual_seed(0)
    candidates = torch.randn(5, 1, 16, 16, 384, dtype=torch.float64)
    goal = torch.randn(1, 1, 16, 16, 384, dtype=torch.float64)
    torch.testing.assert_close(
        linear_goal_distance(candidates, goal, identity_head().double()),
        mean_squared_goal_distance(candidates, goal),
    )


def test_linear_goal_distance_matches_definition():
    torch.manual_seed(1)
    candidates, goals, head = torch.randn(3, 4, 6), torch.randn(3, 4, 6), torch.randn(2, 6)
    expected = torch.stack([
        torch.stack([(head @ (candidates[n, i] - goals[n, i])).square().sum() for i in range(4)]).mean()
        for n in range(3)
    ])
    torch.testing.assert_close(linear_goal_distance(candidates, goals, head), expected)
    torch.testing.assert_close(head_distance(candidates, goals, head), expected)


def test_linear_goal_distance_rejects_bad_head():
    with pytest.raises(ValueError, match="shape"):
        linear_goal_distance(torch.zeros(2, 3, 4), torch.zeros(1, 3, 4), torch.zeros(2, 5))


@pytest.mark.parametrize("make", [random_head, coordinate_subset_head])
def test_control_heads_have_unit_norm_and_shape(make):
    head = make(seed=3)
    assert head.shape == (32, 384)
    torch.testing.assert_close(torch.linalg.matrix_norm(head), torch.tensor(1.0))
    torch.testing.assert_close(make(seed=3), head)
    assert not torch.equal(make(seed=4), head)


def test_coordinate_subset_selects_distinct_coordinates():
    head = coordinate_subset_head(seed=0)
    assert torch.count_nonzero(head, dim=1).eq(1).all()
    assert len(torch.nonzero(head)[:, 1].unique()) == 32


def test_normalized_rejects_zero():
    with pytest.raises(ValueError):
        normalized(torch.zeros(2, 3))


def test_info_nce_matches_brute_force():
    torch.manual_seed(2)
    anchor, positive = torch.randn(4, 5, 6), torch.randn(4, 5, 6)
    negatives, head, tau = torch.randn(4, 7, 5, 6), torch.randn(3, 6), 0.7
    expected = 0.0
    for b in range(4):
        dist = lambda other: ((other - anchor[b]) @ head.T).square().sum(-1).mean()  # noqa: E731
        logits = torch.stack([dist(positive[b])] + [dist(n) for n in negatives[b]]) / -tau
        expected += -(logits[0] - torch.logsumexp(logits, 0)) / 4
    torch.testing.assert_close(info_nce_loss(anchor, positive, negatives, head, tau), expected)


def test_train_step_keeps_unit_norm_and_learns():
    torch.manual_seed(3)
    # Signal in coordinate 0 identifies the target; the rest is hand noise.
    target = torch.randn(8, 1, 1, 4) * torch.tensor([3.0, 0, 0, 0])
    def view(group):
        return target[group, 0] + torch.randn(len(group), 2, 4) * torch.tensor([0.0, 1, 1, 1])
    head = LinearGoalHead(random_head(rank=2, dim=4, seed=0))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    rng = np.random.default_rng(0)
    for _ in range(200):
        group = rng.integers(8, size=16)
        negative_groups = (group[:, None] + rng.integers(1, 8, size=(16, 7))) % 8
        batch = {"anchor": view(group), "positive": view(group),
                 "negatives": view(negative_groups.ravel()).reshape(16, 7, 2, 4)}
        _, kept = train_step(head, optimizer, batch, tau=0.5)
        assert kept
        torch.testing.assert_close(torch.linalg.matrix_norm(head.weight.detach()), torch.tensor(1.0))
    column_energy = head.weight.detach().square().sum(0)
    assert column_energy[0] > 0.9


def test_zero_norm_update_is_rejected():
    head = LinearGoalHead(torch.ones(2, 3))
    previous = head.weight.detach().clone()
    with torch.no_grad():
        head.weight.zero_()
    assert not head.renormalize_(previous)
    torch.testing.assert_close(head.weight.detach(), previous)


def test_save_and_load_round_trip(tmp_path):
    head = random_head(seed=5)
    save_head(tmp_path / "L.pt", head, {"M": 8, "K": 8})
    loaded, config = load_head(tmp_path / "L.pt")
    torch.testing.assert_close(loaded, head)
    assert config == {"M": 8, "K": 8}


def test_allocation_grid():
    grid = allocation_grid()
    assert len(grid) == 9
    assert all(budget == m * k for budget, m, k in grid)
    assert (64, 8, 8) in grid and (1024, 512, 2) in grid


def pool(targets=20, hands=8):
    return np.repeat(np.arange(targets), hands)


@pytest.mark.parametrize("budget,targets,hands", allocation_grid())
def test_sampler_uses_exactly_budget_images_with_target_structure(budget, targets, hands):
    groups = pool(512, 8)
    sampler = AllocationSampler(groups, targets, hands, seed=0)
    assert sampler.groups.shape == (targets, hands)
    assert len(np.unique(sampler.images)) == budget
    assert all(len(np.unique(groups[row])) == 1 for row in sampler.groups)
    assert len(np.unique(groups[sampler.groups[:, 0]])) == targets


def test_sampler_is_nested_and_resamples_differ():
    groups = pool(64, 8)
    small, large = AllocationSampler(groups, 8, 2, seed=1), AllocationSampler(groups, 16, 4, seed=1)
    np.testing.assert_array_equal(large.groups[:8, :2], small.groups)
    assert not np.array_equal(AllocationSampler(groups, 8, 2, seed=2).groups, small.groups)


def test_batches_draw_only_from_allocation():
    groups = pool(40, 8)
    sampler = AllocationSampler(groups, 8, 2, seed=0)
    batch = sampler.sample(500, np.random.default_rng(0))
    allowed = set(sampler.images)
    for key in ("anchor", "positive", "negatives"):
        assert set(np.ravel(batch[key])) <= allowed
    assert np.all(batch["anchor"] != batch["positive"])
    assert np.all(groups[batch["anchor"]] == groups[batch["positive"]])
    negative_groups = groups[batch["negatives"]]
    assert negative_groups.shape == (500, NUM_NEGATIVES)
    assert np.all(negative_groups != groups[batch["anchor"]][:, None])
    assert all(len(set(row)) == NUM_NEGATIVES for row in negative_groups)


def test_shuffled_labels_keep_images_but_break_groups():
    groups = pool(64, 8)
    true = AllocationSampler(groups, 32, 8, seed=0)
    shuffled = AllocationSampler(groups, 32, 8, seed=0, shuffle_labels=True)
    np.testing.assert_array_equal(np.sort(true.images), np.sort(shuffled.images))
    assert np.mean([len(np.unique(groups[row])) == 1 for row in shuffled.groups]) < 0.5


def test_sampler_rejects_too_few_targets_or_hands():
    with pytest.raises(ValueError, match="targets"):
        AllocationSampler(pool(), 4, 2, seed=0)
    with pytest.raises(ValueError, match="images"):
        AllocationSampler(np.r_[pool(), [99]], 21, 2, seed=0)
