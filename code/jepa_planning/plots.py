"""Shared figure style and project-wide result plots.

The plots read the tidy tables in ``results/tables/`` (see its README), so a new
cost function, task or dataset shows up after its rows are added and the figures
are redrawn with ``code/scripts/plot_results.py``.

Colors follow the entity, never its position: each method belongs to a family,
and the family fixes the color in every figure. Register a new method in
``METHODS`` and, for a new family, add it to ``FAMILIES``. The family colors are
the first slots of a palette checked for color-vision deficiency, so keep the
order and add families at the end.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Family -> color, in palette order. Add new families at the end.
FAMILIES = {
    "MSE": "#2a78d6",
    "Temporal cost": "#eb6834",
    "Multi-goal averaging": "#1baf7a",
    "Learned head (L)": "#eda100",
    "Disagreement-weighted": "#e87ba4",
    "Final-layer adaptation": "#008300",
}
# Method id (as in the tables) -> (label, family). Table order follows this order.
METHODS = {
    "mse": ("MSE, 1 photo", "MSE"),
    "multi_goal_4": ("Multi-goal, 4 photos", "Multi-goal averaging"),
    "multi_goal": ("Multi-goal", "Multi-goal averaging"),
    "temporal_2k": ("Temporal, 2k episodes", "Temporal cost"),
    "temporal": ("Temporal", "Temporal cost"),
    "temporal_ordering": ("Temporal + ordering loss", "Temporal cost"),
    "blend_0.5": ("Blend with MSE, λ = 0.5", "Temporal cost"),
}
# Metric id -> (label, lower is better).
METRICS = {
    "regret": ("Regret", True),
    "final_error": ("Final error", True),
    "success": ("Success rate", False),
    "ever_success": ("Ever succeeded", False),
    "rank_correlation": ("Rank correlation, all candidates", False),
    "rank_near": ("Rank correlation, 16 nearest", False),
    "photo_agreement": ("Photo agreement", False),
}
TASKS = {"pusht": "PushT", "metaworld_push": "Meta-World Push"}
EVALUATIONS = {"probe": "Fixed-bank probe", "closed_loop": "Closed-loop CEM", "branching": "Branching copies"}


def use_style() -> None:
    """Apply the shared seaborn theme: recessive grid, left-aligned bold titles."""
    sns.set_theme(context="notebook", style="whitegrid", font_scale=0.9, rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "grid.color": GRID,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def load_table(path: Path) -> pd.DataFrame:
    """Read a results table and check that every method and metric is registered."""
    table = pd.read_csv(path)
    unknown = set(table.get("method", [])) - set(METHODS) | set(table.get("metric", [])) - set(METRICS)
    if unknown:
        raise ValueError(f"register these in code/jepa_planning/plots.py first: {sorted(unknown)}")
    return table


def label_methods(table: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and families, ordered as in ``METHODS``."""
    order = [METHODS[m][0] for m in METHODS if m in set(table.method)]
    table = table.assign(
        label=pd.Categorical([METHODS[m][0] for m in table.method], categories=order, ordered=True),
        family=[METHODS[m][1] for m in table.method],
    )
    return table.sort_values("label")


def metric_title(metric: str) -> str:
    label, lower = METRICS[metric]
    return f"{label} ({'lower' if lower else 'higher'} is better)"


def _family_legend(figure, families) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f]) for f in FAMILIES if f in families]
    figure.legend(handles, [f for f in FAMILIES if f in families], loc="lower center",
                  ncol=len(handles), bbox_to_anchor=(0.5, -0.04))


def _bars(axis, data: pd.DataFrame, value: str, fmt: str, labels: bool = True) -> list[float]:
    """Draw one bar per method in table order; return the bar centers."""
    data = data.assign(label=data.label.cat.remove_unused_categories()).sort_values("label")
    colors = {row.label: FAMILIES[row.family] for row in data.itertuples()}
    sns.barplot(data=data, x="label", y=value, hue="label", palette=colors, width=0.55, legend=False,
                saturation=1, errorbar=None, ax=axis)
    bars = axis.patches[: len(data)]
    for bar, number in zip(bars, data[value] if labels else []):
        axis.annotate(format(number, fmt), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3 if number >= 0 else -11), textcoords="offset points",
                      ha="center", fontsize=8.5, color=INK)
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=25)
    for tick in axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    return [bar.get_x() + bar.get_width() / 2 for bar in bars]


def passes(improvement_low: float, banks_positive: int, banks: int) -> bool:
    """The project's confirmation rule: interval above zero and at least 3 of 5 banks (60%) favoring."""
    return improvement_low > 0 and banks_positive >= math.ceil(0.6 * banks)


def method_comparison(table: pd.DataFrame, panels: list[tuple[str, str]], split: str = "eval"):
    """Bar chart per (evaluation, metric) panel: one bar per method, colored by family."""
    table = label_methods(table[table.split == split])
    tasks = sorted(table.task.unique())
    figure, axes = plt.subplots(len(tasks), len(panels), figsize=(4.2 * len(panels), 4.4 * len(tasks)),
                                squeeze=False)
    for row, task in enumerate(tasks):
        for column, (evaluation, metric) in enumerate(panels):
            axis = axes[row, column]
            data = table[(table.task == task) & (table.evaluation == evaluation) & (table.metric == metric)]
            if data.empty:
                axis.set_visible(False)
                continue
            _bars(axis, data, "value", ".3g")
            axis.set_ylabel(METRICS[metric][0])
            axis.set_title(f"{TASKS[task]}: {EVALUATIONS[evaluation]}\n{metric_title(metric)}", fontsize=10)
    _family_legend(figure, set(table.family))
    figure.tight_layout()
    return figure


def paired_comparison(table: pd.DataFrame, baseline: str = "mse", split: str = "eval"):
    """Paired improvement over ``baseline`` with 95% intervals, one panel per task and evaluation."""
    table = label_methods(table[(table.split == split) & (table.baseline == baseline)])
    groups = list(table.groupby(["task", "evaluation", "metric"], sort=False))
    figure, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.6), squeeze=False)
    for axis, ((task, evaluation, metric), data) in zip(axes[0], groups):
        data = data.sort_values("label")
        centers = _bars(axis, data, "improvement", "+.2f", labels=False)
        axis.errorbar(centers, data.improvement, yerr=[data.improvement - data.low, data.high - data.improvement],
                      fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
        for center, row in zip(centers, data.itertuples()):
            axis.annotate(f"{row.improvement:+.2f}", (center + 0.3, row.improvement), ha="left", va="center",
                          fontsize=8.5, color=INK)
        ticks = [
            f"{row.label}\n{row.banks_positive}/{row.banks} banks, "
            f"{'passes' if passes(row.low, row.banks_positive, row.banks) else 'fails'}"
            for row in data.itertuples()
        ]
        axis.set_xticks(centers, ticks)
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.set_ylabel(f"Improvement in {METRICS[metric][0].lower()} over {METHODS[baseline][0]}")
        axis.set_title(f"{TASKS[task]}: {EVALUATIONS[evaluation]}, 95% intervals\n"
                       f"Above 0 is better than {METHODS[baseline][0]}", fontsize=10)
    _family_legend(figure, set(table.family))
    figure.tight_layout()
    return figure


def goal_photo_scaling(table: pd.DataFrame, metric: str = "regret"):
    """Metric against the number of equivalent goal photos, with 95% interval bands."""
    table = label_methods(table[table.metric == metric])
    figure, axis = plt.subplots(figsize=(6, 4.2))
    for (task, label, family), data in table.groupby(["task", "label", "family"], observed=True):
        name = f"{label} ({TASKS[task]})"
        axis.fill_between(data.photos, data.low, data.high, color=FAMILIES[family], alpha=0.15, linewidth=0)
        axis.plot(data.photos, data.value, marker="o", color=FAMILIES[family], linewidth=2, label=name)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(table.photos.unique()), [str(p) for p in sorted(table.photos.unique())])
    axis.set_xlabel("Equivalent goal photos averaged")
    axis.set_ylabel(METRICS[metric][0])
    axis.set_title(f"More goal photos per target\n{metric_title(metric)}, band: 95% interval")
    axis.legend(loc="upper right")
    return figure
