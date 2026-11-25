#!/usr/bin/env python3
"""Aggregate PushT goal-count saturation runs with paired bootstraps.

The report includes metric intervals for every goal count and paired physical-
regret improvements for adjacent counts. Counts above two are also compared
with the one-goal baseline. Positive improvements favor the larger goal set.

Example::

    .venv/bin/python code/experiments/summarize_pusht_goal_count_saturation.py \
      --input results/pusht_goal_count_saturation \
      --output results/pusht_goal_count_saturation/aggregate
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS = (
    "position_error",
    "angle_error",
    "combined_error",
    "regret",
    "success",
    "rank_correlation",
    "oracle_success_available",
    "prototype_mse_to_full",
    "cost_rank_correlation_to_full",
    "selection_matches_full",
)


def load_rows(input_dir: Path) -> list[dict]:
    """Load task-level rows and attach each run's bank seed."""
    paths = sorted(input_dir.glob("*/task_rows.csv"))
    if not paths:
        raise FileNotFoundError(f"no */task_rows.csv under {input_dir}")
    rows = []
    for path in paths:
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing metadata next to {path}")
        seed = str(json.loads(metadata_path.read_text())["config"]["seed"])
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "seed": seed,
                        "source": row["source"],
                        "goal_count": int(row["goal_count"]),
                        "task": int(row["task"]),
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
    return rows


def _balanced_intervals(
    rows: list[dict],
    metrics: tuple[str, ...],
    rng: np.random.Generator,
    samples: int,
) -> dict:
    """Vectorize a bank-then-task bootstrap for a complete balanced design."""
    if not rows or samples < 1:
        raise ValueError("rows and bootstrap samples must be non-empty")
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    lengths = {len(seed_rows) for seed_rows in by_seed.values()}
    if len(lengths) != 1:
        raise ValueError("every seed bank must contain the same number of tasks")
    banks = np.asarray(
        [
            [[float(row[metric]) for metric in metrics] for row in by_seed[seed]]
            for seed in sorted(by_seed)
        ]
    )
    bank_count, task_count, _ = banks.shape
    chosen_banks = np.broadcast_to(
        rng.integers(bank_count, size=(samples, bank_count, 1)),
        (samples, bank_count, task_count),
    )
    chosen_tasks = rng.integers(task_count, size=(samples, bank_count, task_count))
    draws = banks[chosen_banks, chosen_tasks].mean(axis=(1, 2))
    quantiles = np.quantile(draws, [0.025, 0.975], axis=0)
    observed = banks.mean(axis=(0, 1))
    output = {}
    for index, metric in enumerate(metrics):
        output.update(
            {
                f"{metric}_mean": float(observed[index]),
                f"{metric}_low": float(quantiles[0, index]),
                f"{metric}_high": float(quantiles[1, index]),
            }
        )
    return output


def _bank_signs(differences: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in differences:
        grouped[row["seed"]].append(row["improvement"])
    means = [float(np.mean(values)) for values in grouped.values()]
    return {
        "positive_banks": sum(value > 0 for value in means),
        "zero_banks": sum(value == 0 for value in means),
        "negative_banks": sum(value < 0 for value in means),
    }


def aggregate(
    rows: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261121,
) -> tuple[list[dict], list[dict]]:
    """Return count summaries and adjacent/one-goal paired comparisons."""
    if not rows:
        raise ValueError("rows must not be empty")
    rng = np.random.default_rng(bootstrap_seed)
    groups = sorted({(row["source"], row["goal_count"]) for row in rows})
    summaries = []
    for source, goal_count in groups:
        selected = [
            row for row in rows if row["source"] == source and row["goal_count"] == goal_count
        ]
        summaries.append(
            {
                "source": source,
                "goal_count": goal_count,
                "seeds": len({row["seed"] for row in selected}),
                "tasks": len(selected),
                **_balanced_intervals(selected, METRICS, rng, bootstrap_samples),
            }
        )

    lookup = {
        (row["seed"], row["source"], row["goal_count"], row["task"]): row
        for row in rows
    }
    comparisons = []
    for source in sorted({row["source"] for row in rows}):
        counts = sorted({row["goal_count"] for row in rows if row["source"] == source})
        if not counts or counts[0] != 1:
            raise ValueError(f"source {source} is missing the one-goal baseline")
        pairs = [
            (counts[index - 1], count, "adjacent")
            for index, count in enumerate(counts[1:], 1)
        ]
        pairs.extend((1, count, "versus_one") for count in counts[2:])
        for baseline_count, goal_count, comparison in pairs:
            differences = []
            for seed, row_source, row_count, task in sorted(lookup):
                if row_source == source and row_count == goal_count:
                    baseline = lookup[(seed, source, baseline_count, task)]
                    candidate = lookup[(seed, source, goal_count, task)]
                    differences.append(
                        {
                            "seed": seed,
                            "improvement": baseline["regret"] - candidate["regret"],
                        }
                    )
            interval = _balanced_intervals(
                differences, ("improvement",), rng, bootstrap_samples
            )
            signs = _bank_signs(differences)
            comparisons.append(
                {
                    "source": source,
                    "comparison": comparison,
                    "goal_count": goal_count,
                    "baseline_goal_count": baseline_count,
                    "regret_improvement_mean": interval["improvement_mean"],
                    "regret_improvement_low": interval["improvement_low"],
                    "regret_improvement_high": interval["improvement_high"],
                    **signs,
                    "confirmed": bool(
                        interval["improvement_low"] > 0 and signs["positive_banks"] >= 3
                    ),
                }
            )
    return summaries, comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    summaries, comparisons = aggregate(
        load_rows(args.input), bootstrap_samples=args.bootstrap_samples
    )
    args.output.mkdir(parents=True)
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
