"""Configuration and aggregation checks without simulator dependencies."""

import pytest
import yaml

from experiments.run_baseline import summarize, validate_config
from jepa_planning import ROOT


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "code/configs/baseline.yaml").read_text())


def test_default_config(cfg):
    validate_config(cfg)


@pytest.mark.parametrize("section,key,value", [
    (None, "num_tasks", 0), (None, "max_steps", 0), (None, "warmup_steps", -1),
    (None, "hand_separation", float("nan")), (None, "max_steps", 500),
    (None, "first_seed", 2**32), ("planner", "num_elites", 1),
    (None, "seed", 2**32), (None, "hand_separation", True),
    ("planner", "num_elites", 301), ("planner", "batch_size", 0),
    ("planner", "num_act_stepped", 7), ("planner", "iterations", 1.5),
    ("planner", "var_scale", 0),
])
def test_invalid_settings(cfg, section, key, value):
    (cfg[section] if section else cfg)[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("section", [None, "planner"])
def test_unknown_settings(cfg, section):
    (cfg[section] if section else cfg)["typo"] = 1
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_summary_preserves_pairing():
    base = {key: 0 for key in ("puck_error_cm", "success_1cm", "success_2cm", "success_5cm",
            "native_success", "native_success_any", "planning_seconds", "episode_seconds", "peak_cuda_mb")}
    rows = [dict(base, goal=goal, puck_error_cm=error, native_success=success)
            for goal, error, success in [("L", 1, True), ("R", 3, False), ("L", 2, True), ("R", 2, True)]]
    result = summarize(rows)
    assert result["num_tasks"] == 2
    assert result["num_episodes"] == 4
    assert result["mean"]["puck_error_cm"] == 2
    assert result["mean"]["native_success"] == 0.75
    assert result["by_goal"]["L"]["puck_error_cm"] == 1.5
    assert result["paired_native_success_changed"] == 1
    assert result["paired_mean_error_change_cm"] == 1
