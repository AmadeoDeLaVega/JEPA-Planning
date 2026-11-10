#!/usr/bin/env python3
"""Summarize goal-head planning results with task-level bootstrap intervals (plan step 6.5).

Every row for one task (its two goal photos and, for fitted heads, the three
calibration resamples or control seeds) is averaged first. Tasks are then
resampled with replacement, so the related photos of a task stay together
(Agarwal et al.). Reported per condition: mean, interquartile mean (IQM) and
95% percentile intervals. Also reported: the paired difference to plain MSE on
the same tasks. A negative difference means lower error or regret than plain MSE.

For the fixed bank, also reports the pilot's goal-photo intervention: the share of
tasks where the selected candidate, or its native success, changes between
photo 0 and photo 1.

Usage (repository root):
  .venv/bin/python code/experiments/summarize_pusht_heads.py --kind bank \
    --input results/pusht_head_bank/test --output results/pusht_head_bank/test/summary.csv
  .venv/bin/python code/experiments/summarize_pusht_heads.py --kind closed-loop \
    --input results/pusht_head_closed_loop/test --output results/pusht_head_closed_loop/test/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION = ["method", "rank", "B", "M", "K"]
BANK_METRICS = ["position_error", "regret", "combined_regret", "success_10px", "success_20px", "success_50px",
                "pose_success", "native_success", "photo_selection_changed", "photo_native_changed"]
LOOP_METRICS = ["final_position_error", "final_combined_error", "final_success_10px", "final_success_20px",
                "final_success_50px", "final_pose_success", "final_native_success", "ever_native_success",
                "ever_pose_success", "planning_seconds"]
PAIRED = {"bank": ["position_error", "regret", "native_success"],
          "closed-loop": ["final_position_error", "final_combined_error", "final_native_success"]}


def iqm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean along ``axis`` (mean of the middle 50% of sorted values)."""
    values = np.sort(values, axis=axis)
    n = values.shape[axis]
    # Keep sorted positions [25%, 75%); max(..., low + 1) keeps at least one value for tiny n.
    low, high = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.take(values, range(low, max(high, low + 1)), axis=axis).mean(axis=axis)


def bootstrap(per_task: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """Mean and IQM of per-task values with 95% percentile intervals from ``draws`` task resamples."""
    # Each row of `index` is one bootstrap sample of tasks, drawn with replacement.
    index = rng.integers(0, len(per_task), size=(draws, len(per_task)))
    samples = per_task[index]
    means, iqms = samples.mean(axis=1), iqm(samples, axis=1)
    return {"mean": float(per_task.mean()), "mean_lo": float(np.percentile(means, 2.5)),
            "mean_hi": float(np.percentile(means, 97.5)), "iqm": float(iqm(per_task)),
            "iqm_lo": float(np.percentile(iqms, 2.5)), "iqm_hi": float(np.percentile(iqms, 97.5))}


def add_photo_changes(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (source, head, task): did photo 0 and photo 1 select different candidates / native outcomes?"""
    keys = ["source", "head", "task"]
    # One row per (source, head, task) with photo 0 and photo 1 side by side.
    pivot = frame.pivot_table(index=keys, columns="photo", values=["selected", "native_success"], aggfunc="first")
    changes = pd.DataFrame({
        "photo_selection_changed": (pivot["selected"][0] != pivot["selected"][1]).astype(float),
        "photo_native_changed": (pivot["native_success"][0] != pivot["native_success"][1]).astype(float),
    }).reset_index()
    return frame.merge(changes, on=keys)


def summarize(frame: pd.DataFrame, metrics: list[str], paired: list[str], by: list[str], draws: int,
              seed: int) -> pd.DataFrame:
    """One row per condition: bootstrap stats per metric and the paired difference to plain MSE.

    Rows of one task (photos, resamples, control seeds) are averaged first, so
    the bootstrap resamples whole tasks.
    """
    rng = np.random.default_rng(seed)
    for column in CONDITION:
        if column not in frame:
            frame[column] = np.nan
    # groupby drops NaN keys, so plain MSE / controls (no B, M, K) get the placeholder -1.
    frame[CONDITION[1:]] = frame[CONDITION[1:]].astype(float).fillna(-1)  # -1: not a fitted head
    keys = by + CONDITION
    # Average photos, resamples and control seeds within a task first: the task is the unit.
    per_task = frame.groupby(keys + ["task"], dropna=False)[metrics].mean().reset_index()
    head_counts = frame.groupby(keys, dropna=False)["head"].nunique()
    rows = []
    for condition, group in per_task.groupby(keys, dropna=False):
        group = group.sort_values("task")
        row = dict(zip(keys, condition))
        row["tasks"] = len(group)
        row["heads"] = int(head_counts.loc[condition])
        same = np.ones(len(per_task), dtype=bool)
        for key in by:
            same &= (per_task[key] == row[key]).to_numpy()
        reference = per_task[same & (per_task.method == "plain_mse").to_numpy()].set_index("task")
        for metric in metrics:
            stats = bootstrap(group[metric].to_numpy(), draws, rng)
            row.update({f"{metric}_{name}": value for name, value in stats.items()})
            if metric in paired and row["method"] != "plain_mse" and len(reference):
                # Paired: same tasks, so task difficulty cancels in the difference.
                difference = group[metric].to_numpy() - reference.loc[group["task"], metric].to_numpy()
                stats = bootstrap(difference, draws, rng)
                row.update({f"{metric}_vs_mse": stats["mean"], f"{metric}_vs_mse_lo": stats["mean_lo"],
                            f"{metric}_vs_mse_hi": stats["mean_hi"]})
        rows.append(row)
    return pd.DataFrame(rows).replace({column: {-1: np.nan} for column in CONDITION[2:]})


def main() -> None:
    """Read records (bank) or episodes (closed loop), summarize, and write a CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("bank", "closed-loop"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "bank":
        frame = add_photo_changes(pd.read_csv(args.input / "records.csv"))
        summary = summarize(frame, BANK_METRICS, PAIRED["bank"], ["source"], args.draws, args.seed)
    else:
        frame = pd.concat([pd.read_csv(path) for path in sorted((args.input / "episodes").glob("*.csv"))])
        summary = summarize(frame, LOOP_METRICS, PAIRED["closed-loop"], [], args.draws, args.seed)
    summary.to_csv(args.output, index=False)
    print(f"wrote {len(summary)} conditions to {args.output}")


if __name__ == "__main__":
    main()
