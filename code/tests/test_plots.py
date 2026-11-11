"""Tests for the shared result tables and plots."""

import pandas as pd
import pytest

pytest.importorskip("seaborn")

from jepa_planning import ROOT, plots  # noqa: E402

TABLES = ROOT / "results/tables"


@pytest.mark.parametrize("name", ["method_results.csv", "paired_results.csv", "goal_photo_scaling.csv"])
def test_every_table_row_uses_registered_names(name):
    table = plots.load_table(TABLES / name)

    assert not table.empty
    assert set(table.task) <= set(plots.TASKS)
    assert set(table.evaluation) <= set(plots.EVALUATIONS)
    assert set(table.split) <= {"dev", "eval"}


def test_unregistered_methods_are_rejected(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"method": ["new_cost"], "metric": ["regret"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="new_cost"):
        plots.load_table(path)


def test_every_method_has_a_family_color_and_families_differ():
    assert {family for _, family in plots.METHODS.values()} <= set(plots.FAMILIES)
    assert len(set(plots.FAMILIES.values())) == len(plots.FAMILIES)


@pytest.mark.parametrize("low,positive,banks,expected", [(0.1, 3, 5, True), (0.1, 2, 5, False), (-0.1, 5, 5, False)])
def test_confirmation_rule(low, positive, banks, expected):
    assert plots.passes(low, positive, banks) is expected


def test_labels_follow_registry_order():
    table = pd.DataFrame({"method": ["temporal", "mse", "multi_goal_4"]})

    labelled = plots.label_methods(table)

    assert list(labelled.method) == ["mse", "multi_goal_4", "temporal"]
    assert list(labelled.family) == ["MSE", "Multi-goal averaging", "Temporal cost"]


def test_figures_render_from_the_tables():
    plots.use_style()
    results = plots.load_table(TABLES / "method_results.csv")

    figures = [
        plots.method_comparison(results, [("probe", "regret"), ("closed_loop", "final_error")]),
        plots.paired_comparison(plots.load_table(TABLES / "paired_results.csv")),
        plots.goal_photo_scaling(plots.load_table(TABLES / "goal_photo_scaling.csv")),
    ]

    assert [len([a for a in f.axes if a.get_visible()]) for f in figures] == [2, 2, 1]
