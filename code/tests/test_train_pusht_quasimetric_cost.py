"""Configuration, cache and training-loop tests for the quasimetric cost."""

from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from experiments.train_pusht_quasimetric_cost import (
    build_cost,
    load_or_build_cache,
    train,
    validate,
    validate_config,
    validation_samples,
)
from jepa_planning import ROOT
from jepa_planning.quasimetric_cost import QuasimetricCost

FEATURES = 4 * 4 * 8


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "code/configs/pusht_quasimetric_cost.yaml").read_text())


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
        ("components", 0),
        ("component_dim", 1.5),
        ("batch_size", 0),
        ("horizon", 1.0),
        ("horizon", float("inf")),
        ("slack", 0.0),
        ("slack", float("nan")),
        ("learning_rate", 0),
        ("seed", 2**32),
    ],
)
def test_invalid_config_is_rejected(config, field, value):
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_cost_matches_the_config(config):
    cost = build_cost(config)

    assert (cost.components, cost.component_dim) == (config["components"], config["component_dim"])


def fake_build(calls):
    def build(split, path):
        calls.append(split)
        counts = np.array([3, 4]) if split == "train" else np.array([5])
        cache = np.memmap(path, dtype=np.float16, mode="w+", shape=(int(counts.sum()), FEATURES))
        cache[:] = np.arange(len(cache))[:, None]
        cache.flush()
        return counts
    return build


def test_cache_is_built_once_and_then_reused(tmp_path):
    calls, key = [], {"episodes": 2, "frame_stride": 5, "data": "pusht_noise"}

    first = load_or_build_cache(tmp_path / "cache", key, fake_build(calls), feature_size=FEATURES)
    again = load_or_build_cache(tmp_path / "cache", key, fake_build(calls), feature_size=FEATURES)

    assert calls == ["train", "val"]
    for left, right in zip(first, again):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(again[1], [3, 4])
    assert again[0][6, 0] == 6


def test_cache_built_for_other_settings_is_refused(tmp_path):
    key = {"episodes": 2, "frame_stride": 5, "data": "pusht_noise"}
    load_or_build_cache(tmp_path, key, fake_build([]), feature_size=FEATURES)

    with pytest.raises(ValueError, match="built for"):
        load_or_build_cache(tmp_path, {**key, "frame_stride": 1}, fake_build([]), feature_size=FEATURES)


def test_interrupted_cache_is_rebuilt(tmp_path):
    (tmp_path / "train.cache").write_bytes(b"partial")  # no cache.json: the build never finished
    calls = []

    load_or_build_cache(tmp_path, {"episodes": 2}, fake_build(calls), feature_size=FEATURES)

    assert calls == ["train", "val"]


def test_validation_pairs_lie_ahead_in_one_episode():
    counts = np.array([5, 30, 12])

    samples = validation_samples(counts, seed=0)
    starts, goals, gaps = samples["pairs"]
    frames, after = samples["transitions"]

    np.testing.assert_array_equal(goals - starts, gaps)
    assert gaps.min() >= 1
    np.testing.assert_array_equal(after, frames + 1)


class FrameGap(torch.nn.Module):
    """A perfect cost for a cache whose features equal the frame index."""

    def forward(self, candidates, goals):
        return (goals[:, 0] - candidates[:, 0]).float().clamp(min=0)


def test_validation_scores_a_cost_that_counts_steps_exactly():
    counts = np.array([20, 20])
    cache = np.repeat(np.arange(40, dtype=np.float32)[:, None], 2, axis=1)

    scores = validate(FrameGap(), cache, counts, validation_samples(counts, seed=0), torch.device("cpu"), chunk=64)

    assert scores == {"val_step_distance": 1.0, "val_violation": 0.0, "val_gap_ratio": 1.0, "val_order_accuracy": 1.0}


@pytest.fixture
def toy():
    torch.manual_seed(0)
    cache = np.random.default_rng(0).standard_normal((12, FEATURES)).astype(np.float16)
    return QuasimetricCost(feature_dim=8, grid=4, width=4, hidden=16, components=4, component_dim=4), cache


def test_validation_is_repeatable(toy):
    cost, cache = toy
    counts = np.array([6, 6])
    samples = validation_samples(counts, seed=0)

    first = validate(cost, cache, counts, samples, torch.device("cpu"), chunk=100)

    assert first == validate(cost, cache, counts, samples, torch.device("cpu"), chunk=100)
    assert np.isfinite(list(first.values())).all()


def test_train_updates_the_cost_and_logs_its_progress(toy, config):
    cost, cache = toy
    config.update(train_steps=20, batch_size=4)
    counts = np.array([6, 6])
    before = cost.encoder[0].weight.detach().clone()

    history = train(cost, cache, counts, cache, counts, config, torch.device("cpu"))

    assert len(history) == 20
    assert {"push", "violation", "multiplier", "val_gap_ratio", "val_order_accuracy"} <= history[0].keys()
    assert np.isfinite([row["train_loss"] for row in history]).all()
    assert not torch.equal(before, cost.encoder[0].weight)
