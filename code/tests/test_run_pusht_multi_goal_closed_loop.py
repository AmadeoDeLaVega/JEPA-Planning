"""Configuration and aggregation tests for paired closed-loop CEM."""

from copy import deepcopy

import pytest
import yaml

from experiments.run_pusht_multi_goal_closed_loop import summarize, validate_config
from jepa_planning import ROOT


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "code/configs/pusht_multi_goal_closed_loop.yaml").read_text())


def test_default_config_is_valid(config):
    validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes", 0),
        ("goals_per_episode", 3),
        ("target_candidates", 0),
        ("cem_samples", 1),
        ("cem_elites", 1),
        ("cem_elites", 64),
        ("cem_iterations", 0),
        ("batch_size", 0),
        ("action_clip", 0),
        ("seed", 2**32),
    ],
)
def test_invalid_config_is_rejected(config, field, value):
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_summary_keeps_methods_separate():
    common = {
        "best_error": 2.0,
        "final_position_error": 2.0,
        "final_angle_error": 0.1,
        "error_reduction": 1.0,
        "ever_success": 1,
    }
    rows = [
        {**common, "method": "l2_g1", "final_error": 5.0},
        {**common, "method": "l2_g1", "final_error": 7.0},
        {**common, "method": "l2_g4", "final_error": 2.0},
        {**common, "method": "l2_g4", "final_error": 4.0},
    ]

    result = summarize(rows)

    assert result[0]["method"] == "l2_g1"
    assert result[0]["final_error"] == 6.0
    assert result[1]["method"] == "l2_g4"
    assert result[1]["final_error"] == 3.0
