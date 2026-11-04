"""Paired hierarchical-bootstrap tests for multi-goal summaries."""

from experiments.summarize_pusht_multi_goal import (
    aggregate_closed_loop,
    aggregate_probe,
)


def test_probe_confirmation_uses_paired_task_differences():
    rows = []
    for seed in ("1", "2", "3"):
        for task in range(4):
            common = {
                "seed": seed,
                "source": "action_predicted",
                "task": task,
                "position_error": 1.0,
                "angle_error": 0.1,
                "success": 1.0,
                "rank_correlation": 0.5,
                "oracle_success_available": 1.0,
            }
            rows.extend(
                [
                    {**common, "goal_count": 1, "combined_error": 4.0, "regret": 3.0},
                    {**common, "goal_count": 4, "combined_error": 2.0, "regret": 1.0},
                ]
            )

    _, comparisons = aggregate_probe(rows, bootstrap_samples=200, bootstrap_seed=1)

    comparison = comparisons[0]
    assert comparison["regret_improvement_mean"] == 2.0
    assert comparison["regret_improvement_low"] > 0
    assert comparison["positive_banks"] == 3
    assert comparison["confirmed"]


def test_closed_loop_confirmation_uses_paired_episode_differences():
    rows = []
    for seed in ("1", "2", "3"):
        for episode in range(4):
            common = {
                "seed": seed,
                "episode": episode,
                "best_error": 2.0,
                "final_position_error": 2.0,
                "final_angle_error": 0.1,
                "error_reduction": 1.0,
                "ever_success": 1.0,
            }
            rows.extend(
                [
                    {**common, "method": "l2_g1", "final_error": 5.0},
                    {**common, "method": "l2_g4", "final_error": 2.0},
                ]
            )

    _, comparison = aggregate_closed_loop(rows, bootstrap_samples=200, bootstrap_seed=1)

    assert comparison["final_error_improvement_mean"] == 3.0
    assert comparison["final_error_improvement_low"] > 0
    assert comparison["positive_banks"] == 3
    assert comparison["confirmed"]
