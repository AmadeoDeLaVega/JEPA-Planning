#!/usr/bin/env python3
"""Figures for the PushT linear goal head report (docs/pusht_linear_head.md).

Reads the generated results (git-ignored) and writes PNGs to ``docs/figures/``:

  pusht_head_closed_loop.png   final T error per rank-32 allocation vs plain MSE (step 6.4)
  pusht_head_bank_regret.png   fixed-bank regret per cost, actual vs predicted endpoints (6.2/6.3)
  pusht_head_sensitivity.png   distance growth when the T is moved, per cost (6.1)
  pusht_head_rank.png          predicted-endpoint regret against head rank (compression sweep)

Error bars are 95% percentile intervals from a task-level bootstrap (all rows of
one task averaged first), the same procedure as ``summarize_pusht_heads.py``.
Re-run after more closed-loop heads finish; heads not yet run are left out.

Usage (repository root):
  .venv/bin/python code/experiments/plot_pusht_head_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_planning import ROOT  # noqa: E402
from summarize_pusht_heads import bootstrap  # noqa: E402

# Validated categorical slots 1-4 (light surface) and text/grid tokens.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#8a8984"  # reference lines (plain MSE)

LABELS = {"plain_mse": "Plain MSE", "linear_head": "Learned head", "random_projection": "Random projection",
          "coordinate_subset": "Coordinate subset", "linear_head_shuffled": "Shuffled-label head"}


def style() -> None:
    """Recessive axes and grid, plain text ink, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def task_ci(frame: pd.DataFrame, metric: str, draws: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap mean and 95% interval of ``metric`` over tasks (rows of a task averaged first)."""
    per_task = frame.groupby("task")[metric].mean().to_numpy()
    return bootstrap(per_task, draws, np.random.default_rng(seed))


def dot(ax, x, y, lo, hi, color, horizontal=True) -> None:
    """Point estimate with a 95% interval: 2px whisker, 8px dot with a surface ring."""
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=2)
    else:
        ax.plot([x, x], [lo, hi], color=color, linewidth=2, zorder=2)
    ax.scatter([x], [y], s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)


def closed_loop(results: Path, out: Path) -> None:
    episodes = sorted((results / "pusht_head_closed_loop/test/episodes").glob("*.csv"))
    frame = pd.concat([pd.read_csv(p) for p in episodes])
    mse = task_ci(frame[frame.method == "plain_mse"], "final_position_error")
    heads = frame[(frame.method == "linear_head") & (frame["rank"] == 32)]
    random = frame[(frame.method == "random_projection") & (frame["rank"] == 32)]
    allocations = sorted(heads.groupby(["B", "M", "K"]).groups, key=lambda a: (a[0], a[2]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = []
    for i, (b, m, k) in enumerate(allocations):
        sub = heads[(heads.B == b) & (heads.M == m) & (heads.K == k)]
        rows.append((f"B={int(b)}  M={int(m)}  K={int(k)}", task_ci(sub, "final_position_error"),
                     sub["head"].nunique()))
    if len(random):
        rows.append(("Random projection (rank 32)", task_ci(random, "final_position_error"), random["head"].nunique()))
    y = np.arange(len(rows))[::-1]
    for yi, (label, ci, _) in zip(y, rows):
        color = ORANGE if label.startswith("Random") else BLUE
        dot(ax, ci["mean"], yi, ci["mean_lo"], ci["mean_hi"], color)
        ax.text(ci["mean_hi"] + 0.8, yi, f"{ci['mean']:.1f}", va="center", color=INK_2, fontsize=9)
    ax.axvspan(mse["mean_lo"], mse["mean_hi"], color=NEUTRAL, alpha=0.12, linewidth=0, zorder=1)
    ax.axvline(mse["mean"], color=NEUTRAL, linewidth=1.5, zorder=1)
    ax.text(mse["mean"], len(rows) - 0.45, f"plain MSE {mse['mean']:.1f}", ha="center", va="bottom",
            color=INK_2, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlim(0, max(r[1]["mean_hi"] for r in rows) + 6)
    ax.grid(axis="y", visible=False)
    resamples = sorted({r[2] for r in rows})
    ax.set_xlabel("Mean final T position error (px), lower is better")
    ax.set_title("Closed loop, held-out PushT targets (rank-32 heads)", pad=18)
    fig.text(0.01, 0.01, f"100 test tasks × 2 goal photos per head; {'/'.join(map(str, resamples))} resample(s) "
             "per allocation so far.\nWhiskers: 95% task-bootstrap interval. Gray line and band: plain MSE and its "
             "interval.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "pusht_head_closed_loop.png", dpi=160)
    plt.close(fig)


def bank_regret(results: Path, out: Path) -> None:
    records = pd.read_csv(results / "pusht_head_bank/test/records.csv", low_memory=False)
    methods = ["plain_mse", "linear_head", "coordinate_subset", "random_projection", "linear_head_shuffled"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(methods))[::-1]
    offsets = {"actual": 0.14, "predicted": -0.14}
    colors = {"actual": BLUE, "predicted": ORANGE}
    for source in ("actual", "predicted"):
        for yi, method in zip(y, methods):
            ci = task_ci(records[(records.source == source) & (records.method == method)], "regret")
            dot(ax, ci["mean"], yi + offsets[source], ci["mean_lo"], ci["mean_hi"], colors[source])
            ax.text(ci["mean_hi"] + 0.7, yi + offsets[source], f"{ci['mean']:.1f}", va="center",
                    color=INK_2, fontsize=8)
    ax.set_yticks(y, [LABELS[m] for m in methods])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    ax.set_xlabel("Regret vs best candidate in the bank (px), lower is better")
    ax.set_title("Fixed candidate bank on held-out targets (all ranks pooled)")
    handles = [plt.Line2D([], [], marker="o", color=colors[s], markersize=7, linewidth=2,
                          label=f"{s} endpoints") for s in ("actual", "predicted")]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.text(0.01, 0.01, "100 test tasks × 64 candidates × 2 goal photos. Predicted = chosen from world-model "
             "predictions (what the planner sees).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "pusht_head_bank_regret.png", dpi=160)
    plt.close(fig)


def sensitivity(results: Path, out: Path) -> None:
    test = pd.read_csv(results / "pusht_linear_head_mac/test_eval/test.csv")
    series = [("linear_head", 32, BLUE), ("linear_head_shuffled", 32, AQUA),
              ("random_projection", 32, YELLOW), ("plain_mse", 384, ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, kind, amounts, unit in ((axes[0], "shift_px", (5, 10, 20), "px"),
                                    (axes[1], "rotation_deg", (5, 10, 20), "°")):
        for method, rank, color in series:
            sub = test[(test.method == method) & (test["rank"] == rank)]
            values = [sub[f"sensitivity_{kind}_{a}"].mean() for a in amounts]
            ax.plot(amounts, values, color=color, marker="o", markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            if ax is axes[1]:
                label = LABELS[method] + (" (rank 32)" if method != "plain_mse" else "")
                # Random projection and plain MSE nearly coincide: label one above, one below.
                dy = {"random_projection": 7, "plain_mse": -7}.get(method, 0)
                ax.annotate(label, (amounts[-1], values[-1]), xytext=(6, dy), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
        ax.axhline(1.0, color=NEUTRAL, linewidth=1)
        ax.set_xticks(amounts, [f"{a}{unit}" for a in amounts])
        ax.set_xlabel("T moved" if kind == "shift_px" else "T rotated")
    axes[0].text(5, 1.05, "1 = spread between photos of one target", fontsize=8, color=INK_2, va="bottom")
    axes[0].set_ylabel("Distance ratio (higher = notices)")
    fig.suptitle("Does the cost notice when the T moves?", x=0.01, ha="left", fontweight="bold", fontsize=11)
    axes[1].set_xlim(3.5, 33)
    fig.text(0.01, 0.01, "Held-out test photos. Ratio = distance to the photo with the T moved / median distance "
             "between photos of the same target.\nMedian per head, averaged over fitted heads (or control seeds). "
             "Random projection and plain MSE overlap.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "pusht_head_sensitivity.png", dpi=160)
    plt.close(fig)


def rank_sweep(results: Path, out: Path) -> None:
    records = pd.read_csv(results / "pusht_head_bank/test/records.csv", low_memory=False)
    pred = records[records.source == "predicted"]
    mse = task_ci(pred[pred.method == "plain_mse"], "regret")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method, color in (("linear_head", BLUE), ("random_projection", YELLOW)):
        ranks = sorted(pred[pred.method == method]["rank"].unique())
        cis = [task_ci(pred[(pred.method == method) & (pred["rank"] == r)], "regret") for r in ranks]
        ax.plot(ranks, [c["mean"] for c in cis], color=color, zorder=2)
        for r, c in zip(ranks, cis):
            dot(ax, r, c["mean"], c["mean_lo"], c["mean_hi"], color, horizontal=False)
        ax.annotate(LABELS[method], (ranks[-1], cis[-1]["mean"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.axhspan(mse["mean_lo"], mse["mean_hi"], color=NEUTRAL, alpha=0.12, linewidth=0)
    ax.axhline(mse["mean"], color=NEUTRAL, linewidth=1.5)
    ax.annotate(f"plain MSE (rank 384) {mse['mean']:.1f}", (4, mse["mean"]), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 32, 128], ["4\n(96×)", "16\n(24×)", "32\n(12×)", "128\n(3×)"])
    ax.set_xlim(3, 260)
    ax.set_ylim(0, None)
    ax.set_xlabel("Head rank r (compression 384 / r)")
    ax.set_ylabel("Predicted-endpoint regret (px)")
    ax.set_title("Compression sweep: fixed bank, held-out targets")
    fig.tight_layout()
    fig.savefig(out / "pusht_head_rank.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Write all four figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    closed_loop(args.results, args.output)
    bank_regret(args.results, args.output)
    sensitivity(args.results, args.output)
    rank_sweep(args.results, args.output)
    print(f"wrote figures to {args.output}")


if __name__ == "__main__":
    main()
