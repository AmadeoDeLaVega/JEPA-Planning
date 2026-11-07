"""Configuration and training-loop tests for the temporal cost."""

from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from experiments.train_pusht_temporal_cost import losses, strided_frames, train, validate, validate_config
from jepa_planning import ROOT
from jepa_planning.temporal_cost import TemporalCost


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "code/configs/pusht_temporal_cost.yaml").read_text())


def test_default_config_is_valid(config):
    validate_config(config)


def test_unknown_config_field_is_rejected(config):
    with pytest.raises(ValueError):
        validate_config({**config, "typo": 1})


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes", 0),
        ("frame_stride", 0),
        ("max_gap", 0),
        ("max_gap", 1.5),
        ("far_fraction", -0.1),
        ("far_fraction", 1.5),
        ("batch_size", 0),
        ("ordering_weight", -1.0),
        ("ordering_weight", float("nan")),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("seed", 2**32),
    ],
)
def test_invalid_config_is_rejected(config, field, value):
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_strided_frames_start_at_zero_and_stay_in_range():
    np.testing.assert_array_equal(strided_frames(11, 5), [0, 5, 10])
    np.testing.assert_array_equal(strided_frames(3, 5), [0])


@pytest.mark.parametrize("length,stride", [(0, 5), (5, 0)])
def test_strided_frames_reject_invalid_arguments(length, stride):
    with pytest.raises(ValueError):
        strided_frames(length, stride)


@pytest.fixture
def toy():
    torch.manual_seed(0)
    cache = np.random.default_rng(0).standard_normal((12, 4 * 4 * 8)).astype(np.float16)
    return TemporalCost(feature_dim=8, grid=4, width=4, hidden=16), cache


def test_losses_use_requested_rows_and_backpropagate(toy):
    cost, cache = toy
    pairs = (np.array([0, 1]), np.array([2, 3]), np.array([2.0, 2.0], dtype=np.float32))

    regression, ordering, accuracy = losses(cost, cache, np.array([12]), pairs, torch.device("cpu"))
    (regression + ordering).backward()

    assert regression.ndim == ordering.ndim == 0
    assert 0 <= accuracy <= 1
    assert cost.goal_projection.weight.grad.abs().sum() > 0


class FrameGap(torch.nn.Module):
    """A perfect cost for a cache whose features equal the frame index."""

    def __init__(self, sign=1.0):
        super().__init__()
        self.sign = sign

    def forward(self, candidates, goals):
        return self.sign * (goals[:, 0] - candidates[:, 0]).float()


def test_ordering_loss_is_zero_for_a_cost_that_counts_steps():
    cache = np.repeat(np.arange(10, dtype=np.float32)[:, None], 2, axis=1)
    pairs = (np.array([0, 2, 5]), np.array([4, 9, 8]), np.array([4.0, 7.0, 3.0], dtype=np.float32))

    _, ordering, accuracy = losses(FrameGap(), cache, np.array([10]), pairs, torch.device("cpu"))
    _, reversed_ordering, reversed_accuracy = losses(FrameGap(-1.0), cache, np.array([10]), pairs, torch.device("cpu"))

    assert ordering.item() == 0 and accuracy.item() == 1
    assert reversed_ordering.item() == 2 and reversed_accuracy.item() == 0


def test_validation_is_repeatable(toy):
    cost, cache = toy
    pairs = (np.array([0, 1, 6]), np.array([2, 3, 9]), np.array([2.0, 2.0, 3.0], dtype=np.float32))
    counts = np.array([6, 6])

    first = validate(cost, cache, counts, pairs, torch.device("cpu"), chunk=2)

    assert first == validate(cost, cache, counts, pairs, torch.device("cpu"), chunk=2)
    assert set(first) == {"val_loss", "val_ordering", "val_order_accuracy"}


def test_train_updates_the_cost_and_keeps_the_best_weights(toy, config):
    cost, cache = toy
    config.update(train_steps=20, batch_size=4, max_gap=3)
    counts = np.array([6, 6])
    before = cost.candidate_projection.weight.detach().clone()

    history, best_state = train(cost, cache, counts, cache, counts, config, torch.device("cpu"))

    assert len(history) == 20
    assert np.isfinite([row["val_loss"] for row in history]).all()
    assert not torch.equal(before, cost.candidate_projection.weight)
    assert history[0].get("best")
    assert best_state.keys() == cost.state_dict().keys()
    assert best_state["candidate_projection.weight"] is not cost.state_dict()["candidate_projection.weight"]
