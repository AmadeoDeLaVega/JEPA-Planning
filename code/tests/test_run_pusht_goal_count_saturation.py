"""Configuration and aggregation tests for goal-count saturation."""

from copy import deepcopy

import pytest
import yaml

from experiments.run_pusht_goal_count_saturation import summarize, validate_config
from jepa_planning import ROOT


@pytest.fixture
def config():
    return yaml.safe_load(
        (ROOT / "code/configs/pusht_goal_count_saturation.yaml").read_text()
    )


def test_default_saturation_config_is_valid(config):
    validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tasks", 0),
        ("goals_per_task", 0),
        ("goal_counts", []),
        ("goal_counts", [2, 4, 8, 16, 32]),
        ("goal_counts", [1, 4, 2, 32]),
        ("goal_counts", [1, 2, 4, 8, 16]),
        ("max_subsets_per_count", 0),
        ("batch_size", 0),
        ("action_clip", float("nan")),
        ("seed", 2**32),
    ],
)
def test_invalid_saturation_config_is_rejected(config, field, value):
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_saturation_summary_averages_subsets_before_tasks():
    base = {
        "source": "action_predicted",
        "goal_count": 1,
        "position_error": 0.0,
        "angle_error": 0.0,
        "success": 0,
        "rank_correlation": 0.0,
        "oracle_success_available": 1,
        "prototype_mse_to_full": 0.5,
        "cost_rank_correlation_to_full": 0.5,
        "selection_matches_full": 0,
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
