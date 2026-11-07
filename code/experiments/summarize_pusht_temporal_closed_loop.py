#!/usr/bin/env python3
"""Compare temporal closed-loop runs with MSE and with the multi-goal planners.

Reads ``*/episodes.csv`` from the temporal runs and from the multi-goal runs on
the same seeds. The temporal runs' ``mse`` rows must reproduce ``l2_g1`` exactly,
which confirms both used the same episodes. Improvements are final-error
reductions, so a positive value favors the method. ``confirmed`` uses the
multi-goal rule: a 95% interval above zero and at least 3 of 5 banks positive.

From the repository root::

    .venv/bin/python code/experiments/summarize_pusht_temporal_closed_loop.py \
      --input results/pusht_temporal_closed_loop \
      --reference results/pusht_multi_goal_closed_loop \
      --output results/pusht_temporal_closed_loop/aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.summarize_pusht_multi_goal import (  # noqa: E402
    _confirmation,
    hierarchical_interval,
    load_closed_loop_rows,
    write_csv,
)

COMPARISONS = (("temporal", "mse"), ("blend", "mse"), ("temporal", "l2_g4"), ("blend", "l2_g4"), ("l2_g4", "mse"))


def merge_runs(temporal_rows: list[dict], reference_rows: list[dict]) -> dict:
    """Index rows by (seed, episode, method) after checking that mse reproduces l2_g1."""
    rows = {(row["seed"], row["episode"], row["method"]): row for row in temporal_rows + reference_rows}
    mse = [key for key in rows if key[2] == "mse"]
    for seed, episode, _ in mse:
        reference = rows.get((seed, episode, "l2_g1"))
        if reference is None or not np.isclose(rows[(seed, episode, "mse")]["final_error"], reference["final_error"]):
            raise ValueError("mse rows do not reproduce l2_g1; the runs did not use the same episodes")
    if not mse:
        raise ValueError("no mse rows to check against l2_g1")
    return rows


def compare(rows: dict, method: str, baseline: str, rng: np.random.Generator, samples: int) -> dict:
    """Return the paired final-error improvement of ``method`` over ``baseline``."""
    keys = sorted({key[:2] for key in rows if key[2] == method})
    differences = [
        {"seed": seed, "improvement": rows[(seed, episode, baseline)]["final_error"] - rows[(seed, episode, method)]["final_error"]}
        for seed, episode in keys
    ]
    mean, low, high = hierarchical_interval(differences, "improvement", rng, samples)
    return {"method": method, "baseline": baseline, "improvement_mean": mean, "improvement_low": low,
            "improvement_high": high, **_confirmation(differences, low)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")

    rows = merge_runs(load_closed_loop_rows(args.input), load_closed_loop_rows(args.reference))
    rng = np.random.default_rng(20261108)
    comparisons = [compare(rows, method, baseline, rng, args.bootstrap_samples) for method, baseline in COMPARISONS]
    means = [
        {"method": method, **{metric: float(np.mean([row[metric] for key, row in rows.items() if key[2] == method]))
                              for metric in ("final_error", "best_error", "ever_success")}}
        for method in ("mse", "l2_g4", "temporal", "blend")
    ]
    args.output.mkdir(parents=True)
    write_csv(args.output / "summary.csv", means)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    print(json.dumps({"summary": means, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
