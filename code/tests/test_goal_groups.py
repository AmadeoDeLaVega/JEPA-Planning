"""Simulator-free checks for goal-group seeds and the hand bank."""

import copy

import numpy as np
import pytest
import yaml

from jepa_planning import ROOT
from jepa_planning import goal_groups as gg


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "code/configs/goal_groups.yaml").read_text())


def test_default_config_is_valid(cfg):
    gg.validate_config(cfg)


def test_target_seeds_are_disjoint_and_test_matches_baseline(cfg):
    seeds = {split: set(gg.target_seeds(cfg, split)) for split in gg.SPLITS}
    assert not seeds["train"] & seeds["val"] and not seeds["train"] & seeds["test"]
    assert not seeds["val"] & seeds["test"]
    baseline = yaml.safe_load((ROOT / "code/configs/baseline.yaml").read_text())
    assert seeds["test"] == set(range(baseline["first_seed"], baseline["first_seed"] + baseline["num_tasks"]))
    assert (cfg["warmup_steps"], cfg["goal_steps"]) == (baseline["warmup_steps"], baseline["goal_steps"])


def test_hand_bank_is_split_disjoint_inside_box_and_seeded(cfg):
    bank = gg.hand_bank(cfg)
    assert sum(len(v) for v in bank.values()) == cfg["hand_bank"]["size"]
    stacked = np.concatenate(list(bank.values()))
    assert len(np.unique(stacked, axis=0)) == len(stacked)
    assert np.all(stacked >= cfg["hand_bank"]["low"]) and np.all(stacked <= cfg["hand_bank"]["high"])
    np.testing.assert_array_equal(gg.hand_bank(cfg)["train"], bank["train"])


def test_hand_order_depends_on_target(cfg):
    first, second = gg.hand_order(cfg, 100000, 32), gg.hand_order(cfg, 100001, 32)
    assert sorted(first) == list(range(32))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("edit", [
    lambda c: c["first_seed"].update(val=100100),
    lambda c: c["hands_per_target"].update(train=1),
    lambda c: c["hands_per_target"].update(val=17),
    lambda c: c["hand_bank"]["fractions"].update(train=0.6),
    lambda c: c.update(extra=1),
])
def test_invalid_configs(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg)
    with pytest.raises(ValueError):
        gg.validate_config(cfg)
