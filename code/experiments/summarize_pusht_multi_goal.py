#!/usr/bin/env python3
"""Aggregate multi-seed PushT multi-goal runs with a hierarchical bootstrap.

The bootstrap first resamples independently generated seed banks, then tasks or
paired episodes within each sampled bank. Probe comparisons report the regret
improvement of each goal count over one goal. Closed-loop comparison reports
``final_error(l2_g1) - final_error(l2_g4)``. Positive values favor multi-goal.

Examples::

    .venv/bin/python code/experiments/summarize_pusht_multi_goal.py \
      --kind probe --input results/pusht_multi_goal_probe \
      --output results/pusht_multi_goal_probe/aggregate

    .venv/bin/python code/experiments/summarize_pusht_multi_goal.py \
      --kind closed-loop --input results/pusht_multi_goal_closed_loop \
      --output results/pusht_multi_goal_closed_loop/aggregate
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


PROBE_METRICS = (
    "position_error",
    "angle_error",
    "combined_error",
    "regret",
    "success",
    "rank_correlation",
    "oracle_success_available",
)
CLOSED_LOOP_METRICS = (
    "final_error",
    "best_error",
    "final_position_error",
    "final_angle_error",
    "error_reduction",
    "ever_success",
)


def _run_seed(path: Path) -> str:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata next to {path}")
    metadata = json.loads(metadata_path.read_text())
    return str(metadata["config"]["seed"])


def load_probe_rows(input_dir: Path) -> list[dict]:
    """Load task-level rows from all immediate run subdirectories."""
    paths = sorted(input_dir.glob("*/task_rows.csv"))
    if not paths:
        raise FileNotFoundError(f"no */task_rows.csv under {input_dir}")
    rows = []
    for path in paths:
        seed = _run_seed(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "seed": seed,
                        "source": row["source"],
                        "goal_count": int(row["goal_count"]),
                        "task": int(row["task"]),
                        **{metric: float(row[metric]) for metric in PROBE_METRICS},
                    }
                )
    return rows


def load_closed_loop_rows(input_dir: Path) -> list[dict]:
    """Load episode-level rows from all immediate run subdirectories."""
    paths = sorted(input_dir.glob("*/episodes.csv"))
    if not paths:
        raise FileNotFoundError(f"no */episodes.csv under {input_dir}")
    rows = []
    for path in paths:
        seed = _run_seed(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "seed": seed,
                        "episode": int(row["episode"]),
                        "method": row["method"],
                        **{metric: float(row[metric]) for metric in CLOSED_LOOP_METRICS},
                    }
                )
    return rows


def hierarchical_interval(
    rows: list[dict],
    metric: str,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    """Return observed mean and percentile interval over banks then units."""
    if not rows or samples < 1:
        raise ValueError("rows and bootstrap samples must be non-empty")
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(float(row[metric]))
    seeds = sorted(by_seed)
    observed = float(np.mean([value for seed in seeds for value in by_seed[seed]]))
    draws = np.empty(samples)
    for index in range(samples):
        sampled_values = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            values = np.asarray(by_seed[seed])
            sampled_values.extend(rng.choice(values, size=len(values), replace=True))
        draws[index] = np.mean(sampled_values)
    low, high = np.percentile(draws, [2.5, 97.5])
    return observed, float(low), float(high)


def _intervals(rows: list[dict], metrics: tuple[str, ...], rng, samples) -> dict:
    output = {}
    for metric in metrics:
        mean, low, high = hierarchical_interval(rows, metric, rng, samples)
        output.update({f"{metric}_mean": mean, f"{metric}_low": low, f"{metric}_high": high})
    return output


def _confirmation(differences: list[dict], low: float) -> dict:
    seed_values = defaultdict(list)
    for row in differences:
        seed_values[row["seed"]].append(row["improvement"])
    bank_means = [float(np.mean(values)) for values in seed_values.values()]
    positive = sum(value > 0 for value in bank_means)
    return {
        "positive_banks": positive,
        "zero_banks": sum(value == 0 for value in bank_means),
        "negative_banks": sum(value < 0 for value in bank_means),
        "confirmed": bool(low > 0 and positive >= 3),
    }


def aggregate_probe(
    rows: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261008,
) -> tuple[list[dict], list[dict]]:
    """Summarize probe metrics and paired regret improvements over one goal."""
    rng = np.random.default_rng(bootstrap_seed)
    summaries = []
    groups = sorted({(row["source"], row["goal_count"]) for row in rows})
    for source, goal_count in groups:
        selected = [row for row in rows if row["source"] == source and row["goal_count"] == goal_count]
        summaries.append(
            {
                "source": source,
                "goal_count": goal_count,
                "seeds": len({row["seed"] for row in selected}),
                "tasks": len(selected),
                **_intervals(selected, PROBE_METRICS, rng, bootstrap_samples),
            }
        )

    lookup = {(row["seed"], row["source"], row["goal_count"], row["task"]): row for row in rows}
    comparisons = []
    for source, goal_count in groups:
        if goal_count == 1:
            continue
        differences = []
        for seed, row_source, row_count, task in sorted(lookup):
            if row_source == source and row_count == goal_count:
                baseline = lookup[(seed, source, 1, task)]
                candidate = lookup[(seed, source, goal_count, task)]
                differences.append({"seed": seed, "improvement": baseline["regret"] - candidate["regret"]})
        mean, low, high = hierarchical_interval(differences, "improvement", rng, bootstrap_samples)
        comparisons.append(
            {
                "source": source,
                "goal_count": goal_count,
                "baseline_goal_count": 1,
                "regret_improvement_mean": mean,
                "regret_improvement_low": low,
                "regret_improvement_high": high,
                **_confirmation(differences, low),
            }
        )
    return summaries, comparisons


def aggregate_closed_loop(
    rows: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261108,
) -> tuple[list[dict], dict]:
    """Summarize paired closed-loop metrics and the primary final-error test."""
    rng = np.random.default_rng(bootstrap_seed)
    summaries = []
    for method in ("l2_g1", "l2_g4"):
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            raise ValueError(f"missing rows for {method}")
        summaries.append(
            {
                "method": method,
                "seeds": len({row["seed"] for row in selected}),
                "episodes": len(selected),
                **_intervals(selected, CLOSED_LOOP_METRICS, rng, bootstrap_samples),
            }
        )
    lookup = {(row["seed"], row["episode"], row["method"]): row for row in rows}
    differences = []
    for seed, episode, method in sorted(lookup):
        if method == "l2_g4":
            baseline = lookup[(seed, episode, "l2_g1")]
            candidate = lookup[(seed, episode, "l2_g4")]
            differences.append(
                {
                    "seed": seed,
                    "improvement": baseline["final_error"] - candidate["final_error"],
                    "best_error_improvement": baseline["best_error"] - candidate["best_error"],
                    "success_improvement": candidate["ever_success"] - baseline["ever_success"],
                }
            )
    comparison = {}
    for metric in ("improvement", "best_error_improvement", "success_improvement"):
        mean, low, high = hierarchical_interval(differences, metric, rng, bootstrap_samples)
        label = "final_error_improvement" if metric == "improvement" else metric
        comparison.update({f"{label}_mean": mean, f"{label}_low": low, f"{label}_high": high})
    comparison.update(_confirmation(differences, comparison["final_error_improvement_low"]))
    return summaries, comparison


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("probe", "closed-loop"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    if args.kind == "probe":
        summaries, comparisons = aggregate_probe(
            load_probe_rows(args.input), args.bootstrap_samples
        )
        write_csv(args.output / "summary.csv", summaries)
        write_csv(args.output / "paired_comparisons.csv", comparisons)
    else:
        summaries, comparison = aggregate_closed_loop(
            load_closed_loop_rows(args.input), args.bootstrap_samples
        )
        write_csv(args.output / "summary.csv", summaries)
        write_csv(args.output / "paired_comparison.csv", [comparison])
    print(json.dumps({"summary": summaries, "comparison": comparisons if args.kind == "probe" else comparison}, indent=2))


if __name__ == "__main__":
    main()
