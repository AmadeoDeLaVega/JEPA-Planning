"""Configuration and aggregation tests for the fixed-bank experiment."""

from copy import deepcopy

import pytest
import yaml

from experiments.run_pusht_multi_goal_probe import summarize, validate_config
from jepa_planning import ROOT


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "code/configs/pusht_multi_goal_probe.yaml").read_text())


def test_default_config_is_valid(config):
    validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tasks", 0),
        ("candidates_per_task", 0),
        ("goals_per_task", 0),
        ("goal_counts", []),
        ("goal_counts", [0, 1]),
        ("goal_counts", [1, 1]),
        ("goal_counts", [1, 9]),
        ("batch_size", 0),
        ("action_clip", float("nan")),
        ("seed", 2**32),
    ],
)
def test_invalid_config_is_rejected(config, field, value):
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_summary_averages_subsets_before_tasks():
    base = {
        "source": "action_predicted",
        "goal_count": 1,
        "position_error": 0.0,
        "angle_error": 0.0,
        "success": 0,
        "rank_correlation": 0.0,
        "oracle_success_available": 1,
    }
    records = [
        {**base, "task": 0, "goal_subset": "0", "combined_error": 1.0, "regret": 1.0},
        {**base, "task": 0, "goal_subset": "1", "combined_error": 3.0, "regret": 3.0},
        {**base, "task": 1, "goal_subset": "0", "combined_error": 8.0, "regret": 8.0},
    ]

    task_rows, summary = summarize(records)

    assert [row["regret"] for row in task_rows] == [2.0, 8.0]
    assert summary[0]["regret"] == 5.0
    assert summary[0]["tasks"] == 2
