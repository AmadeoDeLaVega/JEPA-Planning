"""Paired aggregation tests for the goal-count saturation study."""

from experiments.summarize_pusht_goal_count_saturation import aggregate


def test_saturation_aggregation_reports_adjacent_and_one_goal_comparisons():
    rows = []
    for seed in ("1", "2", "3"):
        for task in range(3):
            for goal_count, regret in ((1, 6.0), (2, 4.0), (4, 3.0)):
                rows.append(
                    {
                        "seed": seed,
                        "source": "action_predicted",
                        "goal_count": goal_count,
                        "task": task,
                        "position_error": regret,
                        "angle_error": 0.1,
                        "combined_error": regret,
                        "regret": regret,
                        "success": 1.0,
                        "rank_correlation": 0.5,
                        "oracle_success_available": 1.0,
                        "prototype_mse_to_full": 1.0 / goal_count,
                        "cost_rank_correlation_to_full": 0.5,
                        "selection_matches_full": 0.5,
                    }
                )

    summaries, comparisons = aggregate(rows, bootstrap_samples=200, bootstrap_seed=1)

    assert len(summaries) == 3
    keyed = {
        (row["comparison"], row["baseline_goal_count"], row["goal_count"]): row
        for row in comparisons
    }
    assert keyed[("adjacent", 1, 2)]["regret_improvement_mean"] == 2.0
    assert keyed[("adjacent", 2, 4)]["regret_improvement_mean"] == 1.0
    assert keyed[("versus_one", 1, 4)]["regret_improvement_mean"] == 3.0
    assert all(row["positive_banks"] == 3 for row in comparisons)
    assert all(row["confirmed"] for row in comparisons)
