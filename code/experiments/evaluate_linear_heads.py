#!/usr/bin/env python3
"""Score saved linear heads and their controls on held-out test groups.

Photo-level evaluation (plan step 6.1) on the test split, after every choice
(τ, lr, steps) was fixed on validation by train_linear_head.py. Nothing is
refitted or tuned here. For each head: top-1 retrieval, AUROC, within/between
medians, and the sensitivity ratio for photos with the T moved slightly.

Writes test.csv and metadata.json to a new output directory.

Usage (repository root):
  .venv/bin/python code/experiments/evaluate_linear_heads.py \
    --runs results/pusht_linear_head_mac/rank* --test results/pusht_goal_groups/test \
    --output results/pusht_linear_head_mac/test_eval --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from train_linear_head import evaluate, load_groups, write_csv  # noqa: E402


def main() -> None:
    """Score all saved heads and controls on the test groups and write ``test.csv``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new --output")

    started = perf_counter()
    test = load_groups(args.test, args.device)
    # Plain MSE, every saved head (all ranks, allocations, resamples, shuffled) and the
    # random / coordinate-subset controls. Nothing is refitted or retuned on test.
    catalog = lh.head_catalog(args.runs, dim=test["features"].shape[-1])
    rows = []
    for index, (info, head) in enumerate(catalog):
        rows.append({**info, **evaluate(head, test)})
        if index % 50 == 0:
            print(f"{index}/{len(catalog)} {info['head']}", flush=True)
    args.output.mkdir(parents=True)
    write_csv(args.output / "test.csv", rows)
    metadata = {
        "command": sys.argv,
        "runs": [str(run) for run in args.runs],
        "test": str(args.test),
        "test_images": len(test["groups"]),
        "test_groups_dropped_short": test["dropped_groups"],
        "heads": len(catalog),
        "device": args.device,
        "elapsed_seconds": perf_counter() - started,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
