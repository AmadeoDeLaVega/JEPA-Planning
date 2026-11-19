#!/usr/bin/env python3
"""Draw every results figure from the tables in results/tables/.

Writes project-wide comparisons to results/figures/overview/, learned-cost
diagnostics (temporal and quasimetric) to results/figures/pusht_temporal_cost/,
and quasimetric training curves to results/figures/pusht_quasimetric_cost/.
Re-run after adding rows to the tables; register new methods and metrics in
code/jepa_planning/plots.py.

From the repository root::

    .venv/bin/python code/scripts/plot_results.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # files only; notebooks keep their inline backend
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from jepa_planning import plots  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"

# Diagnostics compare the learned costs with MSE. Each uses its family color; the
# ordering-loss model is a temporal variant, so it takes a darker orange.
VARIANTS = {"MSE, 1 photo": plots.FAMILIES["MSE"], "Temporal": plots.FAMILIES["Temporal cost"],
            "Temporal + ordering loss": "#a8441a", "Quasimetric": plots.FAMILIES["Quasimetric cost"]}
LEARNED = {"temporal": "Temporal", "temporal_ordering": "Temporal + ordering loss", "quasimetric": "Quasimetric"}
SUBSETS = {"All 64 candidates": "#86b6ef", "16 nearest the goal": "#1c5cab"}  # one-hue ramp, light to dark


def overview() -> dict:
    results = plots.load_table(TABLES / "method_results.csv")
    paired = plots.load_table(TABLES / "paired_results.csv")
    scaling = plots.load_table(TABLES / "goal_photo_scaling.csv")
    panels = [("probe", "regret"), ("probe", "success"), ("closed_loop", "final_error"), ("closed_loop", "ever_success")]
    dev_panels = [("probe", "regret"), ("probe", "success"), ("probe", "photo_agreement"), ("branching", "rank_near")]
    return {
        "overview/method_comparison.png": plots.method_comparison(results, panels),
        "overview/method_comparison_dev.png": plots.method_comparison(results, dev_panels, split="dev"),
        "overview/paired_vs_mse.png": plots.paired_comparison(paired),
        "overview/goal_photo_scaling.png": plots.goal_photo_scaling(scaling),
    }


def step_counting():
    table = pd.read_csv(TABLES / "pusht_temporal_cost/step_counting.csv")
    table["episodes"] = table.episodes.map({"train": "Training episodes", "val": "Validation episodes (unseen)"})
    colors = dict(zip(table.episodes.unique(), (plots.FAMILIES["MSE"], plots.FAMILIES["Temporal cost"])))
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    for episodes, data in table.groupby("episodes"):
        axis.fill_between(data.gap, data["mean"] - data.sd, data["mean"] + data.sd, color=colors[episodes],
                          alpha=0.12, linewidth=0)
    sns.lineplot(data=table, x="gap", y="mean", hue="episodes", palette=colors, marker="o", linewidth=2, ax=axis)
    axis.plot([0, 24], [0, 24], color=plots.MUTED, linestyle="--", linewidth=1, label="Perfect prediction")
    axis.set_xlabel("True gap (model steps)")
    axis.set_ylabel("Predicted gap (band: ±1 sd)")
    axis.set_title("Temporal cost: step counting on recorded episodes")
    axis.legend(title=None, loc="upper left")
    return figure


def diagnostics_table() -> pd.DataFrame:
    table = plots.label_methods(plots.load_table(TABLES / "method_results.csv"))
    return table[table.method.isin(["mse", *LEARNED])]


def near_goal_ranking():
    table = diagnostics_table()
    table = table[(table.metric == "rank_near") & (table.split == "dev")]
    table = table.assign(setting=table.evaluation.map({"probe": "Probe, dev banks", "branching": "Branching copies"}),
                         label=table.label.astype(str))
    figure, axis = plt.subplots(figsize=(7, 4.2))
    sns.barplot(data=table, x="setting", y="value", hue="label", palette=VARIANTS, hue_order=list(VARIANTS),
                order=["Probe, dev banks", "Branching copies"], width=0.75, saturation=1, errorbar=None, ax=axis)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", padding=2, fontsize=8.5, color=plots.INK)
    axis.set_xlabel("")
    axis.set_ylabel("Rank correlation, 16 nearest candidates")
    axis.set_ylim(0, 0.8)
    axis.set_title("Separating the best candidates (higher is better)", pad=28)
    axis.legend(title=None, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    return figure


def ranking():
    table = diagnostics_table()
    table = table[table.metric.isin(["rank_correlation", "rank_near"]) & (table.split == "dev")]
    table = table.assign(
        setting=table.evaluation.map({"probe": "Probe, dev banks", "branching": "Branching copies"}),
        subset=table.metric.map({"rank_correlation": "All 64 candidates", "rank_near": "16 nearest the goal"}),
        label=table.label.astype(str),
    )
    grid = sns.catplot(data=table, kind="bar", x="label", y="value", hue="subset", col="setting",
                       col_order=["Probe, dev banks", "Branching copies"], order=list(VARIANTS),
                       hue_order=list(SUBSETS), palette=SUBSETS, height=4.2, aspect=1.3, width=0.7,
                       saturation=1, errorbar=None, legend_out=False)
    for axis in grid.axes.flat:
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f", padding=2, fontsize=8.5, color=plots.INK)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=15)
    grid.set_titles("{col_name}")
    grid.set_ylabels("Spearman correlation of cost with T error")
    grid.set(ylim=(0, 0.8))
    grid.legend.set_title(None)
    grid.figure.suptitle("Ranking candidates by physical error (higher is better)", x=0.02, ha="left",
                         fontweight="bold", y=1.04)
    return grid.figure


def blend_sweep():
    table = pd.read_csv(TABLES / "pusht_temporal_cost/blend_sweep.csv")
    table["learned_cost"] = table.learned_cost.map(LEARNED)
    table["metric_title"] = table.metric.map(plots.metric_title)
    mse = table[table["lambda"] == 0].drop_duplicates("metric").set_index("metric_title").value
    grid = sns.relplot(data=table, kind="line", x="lambda", y="value", hue="learned_cost", col="metric_title",
                       col_wrap=2, palette={k: VARIANTS[k] for k in table.learned_cost.unique()}, marker="o",
                       linewidth=2, height=3.4, aspect=1.35, facet_kws={"sharey": False, "legend_out": False})
    lambdas = sorted(table["lambda"].unique())
    for title, axis in grid.axes_dict.items():
        axis.axhline(mse[title], color=VARIANTS["MSE, 1 photo"], linestyle="--", linewidth=1.5, label="MSE alone (λ = 0)")
        axis.axvline(0.5, color=plots.MUTED, linestyle=":", linewidth=1.2, label="Frozen temporal λ = 0.5")
        axis.set_xticks(lambdas, [f"{value:g}" for value in lambdas])
        axis.set_title(title, fontsize=10)
    grid.set_xlabels("λ in (1 − λ)·z(MSE) + λ·z(learned cost)")
    grid.set_ylabels("")
    handles, labels = grid.axes.flat[0].get_legend_handles_labels()
    if grid.legend is not None:
        grid.legend.remove()
    grid.figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    grid.figure.suptitle("Blend sweep on the dev banks. Temporal λ = 0.5 was frozen and then failed on the evaluation seeds",
                         x=0.02, ha="left", fontweight="bold", y=1.02)
    return grid.figure


def quasimetric_training():
    """Training curves of every quasimetric run, on validation episodes unless noted."""
    table = pd.read_csv(TABLES / "pusht_quasimetric_cost/training.csv")
    slack = 0.25
    panels = [
        ("val_step_distance", "Mean recorded-step cost", (1.0, 1 + slack), False),
        ("val_step_p90", "90th-percentile recorded-step cost", (1.0, 1 + slack), False),
        ("val_violation", "Step-rule violation (log scale)", None, True),
        ("multiplier", "Multiplier on the step rule", None, False),
        ("val_gap_ratio", "Distance ÷ recorded gap", None, False),
        ("val_order_accuracy", "Next frame scored closer", None, False),
    ]
    runs = list(dict.fromkeys(table.run))  # in table order: first run, then later ones
    styles = [{"color": "#9085e9", "linestyle": "--"}, {"color": "#6250d6", "linestyle": "-"},
              {"color": "#3f2fa8", "linestyle": ":"}]  # violet steps, one per run
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for axis, (column, title, band, log) in zip(axes.flat, panels):
        if band:
            axis.axhspan(*band, color=plots.GRID, label="target, 1 to 1 + slack")
        if column == "val_violation":
            axis.axhline(slack**2, color=plots.INK, linestyle=":", linewidth=1.2, label="limit, slack² = 0.0625")
        if column == "val_order_accuracy":
            axis.axhline(0.804, color=plots.FAMILIES["Temporal cost"], linestyle="--", linewidth=1.5,
                         label="temporal cost")
        for run, style in zip(runs, styles):
            data = table[(table.run == run) & table[column].notna()]
            if not data.empty:
                axis.plot(data.step, data[column], marker="o", markersize=4, linewidth=2, label=run, **style)
        if log:
            axis.set_yscale("log")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Training step")
    handles, labels = {}, {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    last = table.groupby("run").step.max()
    note = ", ".join(f"{run}: to step {last[run]:,}" for run in runs)
    figure.suptitle(f"Quasimetric cost training, validation episodes ({note})", x=0.01, ha="left", fontweight="bold")
    figure.tight_layout()
    return figure


def main() -> None:
    plots.use_style()
    figures = overview()
    figures |= {
        "pusht_temporal_cost/step_counting.png": step_counting(),
        "pusht_temporal_cost/near_goal_ranking.png": near_goal_ranking(),
        "pusht_temporal_cost/ranking.png": ranking(),
        "pusht_temporal_cost/blend_sweep.png": blend_sweep(),
        "pusht_quasimetric_cost/training.png": quasimetric_training(),
    }
    for name, figure in figures.items():
        path = FIGURES / name
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
