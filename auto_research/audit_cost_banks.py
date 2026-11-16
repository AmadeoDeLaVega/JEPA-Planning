"""Check normalization sensitivity on real, saved candidate cost banks.

Run: python audit_cost_banks.py banks.npz --weight 0.5

Input arrays (no pickle):
  costs: [tasks, candidates, 2], MSE and temporal costs for ONE goal per task.
  calibration: [calibration candidates, 2], disjoint development-only costs.
  physical_error: [tasks, candidates], independently measured physical error.

Costs and physical errors must align by task and candidate. Keep checkpoint,
head, split and action budget fixed before export. Do not fit calibration on
these evaluation tasks. This script does not reconstruct missing model runs.
It prints per-task rows to stdout; redirect to a new CSV if desired.
"""

import argparse
import numpy as np
import pandas as pd


def scores(costs, weight, reference):
    spread = reference.std(axis=0)
    spread = np.where(spread > 0, spread, 1.)
    scaled = (costs - reference.mean(axis=0)) / spread
    return scaled @ np.array([1 - weight, weight])


def audit(costs, calibration, physical, weight):
    rows = []
    for task, (bank, error) in enumerate(zip(costs, physical)):
        for method in ["current bank", "frozen calibration"]:
            reference = bank if method == "current bank" else calibration
            chosen = int(scores(bank, weight, reference).argmin())
            for axis in [0, 1]:
                # A dominated artificial cost row is a sensitivity probe only.
                # It has no physical rollout and is never assigned a fake outcome.
                spread = np.maximum(bank.std(axis=0), 1e-12)
                excess = np.ones(2)
                excess[axis] = 50.
                added = bank.max(axis=0) + spread * excess
                extended = np.vstack([bank, added])
                new_reference = extended if method == "current bank" else calibration
                changed = int(scores(extended, weight, new_reference).argmin())
                rows.append({
                    "task": task, "scaling": method, "outlier_axis": axis,
                    "before": chosen, "after": changed,
                    "changed": changed != chosen,
                    "physical_before": error[chosen],
                    "physical_after": error[changed] if changed < len(bank) else np.nan,
                    "regret_before": error[chosen] - error.min(),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--weight", type=float, default=0.5)
    args = parser.parse_args()
    with np.load(args.path, allow_pickle=False) as data:
        costs = np.asarray(data["costs"], dtype=float)
        calibration = np.asarray(data["calibration"], dtype=float)
        physical = np.asarray(data["physical_error"], dtype=float)
    # Brief input checks keep a malformed export from looking like a result.
    if costs.ndim != 3 or costs.shape[-1] != 2 or physical.shape != costs.shape[:2]:
        raise ValueError("costs must be [tasks, candidates, 2]; physical_error must align")
    if calibration.ndim != 2 or calibration.shape[1] != 2 or len(calibration) < 2:
        raise ValueError("calibration must have at least two rows and two costs")
    if min(costs.shape[:2]) < 1 or not 0 <= args.weight <= 1:
        raise ValueError("need nonempty banks and weight in [0, 1]")
    if not all(np.isfinite(x).all() for x in [costs, calibration, physical]):
        raise ValueError("input arrays must be finite")
    print(audit(costs, calibration, physical, args.weight).to_csv(index=False), end="")
