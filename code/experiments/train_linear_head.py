#!/usr/bin/env python3
"""Fit the linear goal head for every (B, M, K) allocation on cached features.

Inputs are two output directories of make_pusht_goal_groups.py (or any
directory with images.npz "group" and features.pt "groups"): the training pool
and the validation groups. Test groups are never read here.

Everything is repeated for each head rank r in the config's ``ranks``
(compression ratio D / r, e.g. 384 / 32 = 12x), to see how compressing the goal
features changes the outcome:

1. Grid search: for every (tau, lr) in the grid, fit every allocation on
   resample 0 and score validation AUROC; keep the pair with the best mean
   over allocations (one pair per rank, shared by its allocations).
2. With that pair, fit every allocation on every resample (and, as control 4,
   with shuffled pair labels).
3. Score untrained controls of the same rank (random projections, coordinate
   subsets) and plain MSE (full rank, no compression) on the same validation
   groups.

Ranks can be split across jobs with --ranks.

Writes heads/rank{r}_B{B}_M{M}_K{K}_r{resample}[_shuffled].pt, grid.csv, curves.csv,
validation.csv (retrieval and sensitivity per head) and metadata.json to a new
output directory.

Usage (repository root):
  .venv/bin/python code/experiments/train_linear_head.py --device cuda \
    --train results/pusht_goal_groups/train --val results/pusht_goal_groups/val \
    --output results/pusht_linear_head/seed0
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402


def load_groups(directory: Path, device: str, min_images: int = 0):
    """Features [n, P, D] (fp16 on device), group ids, and sensitivity data if present.

    Groups with fewer than ``min_images`` photos (targets short of agents) are dropped.
    """
    groups = np.load(directory / "images.npz")["group"]
    payload = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    features = payload["groups"]
    if len(features) != len(groups):
        raise ValueError(f"{directory}: {len(features)} features for {len(groups)} images")
    # Training needs K images of every target for the largest K, so targets that
    # lost photos to rejections are dropped whole rather than half-used.
    ids, counts = np.unique(groups, return_counts=True)
    keep = np.isin(groups, ids[counts >= min_images])
    data = {"features": features[torch.from_numpy(keep)].to(device), "groups": groups[keep],
            "dropped_groups": int(np.sum(counts < min_images))}
    if "sensitivity" in payload:
        meta = np.load(directory / "sensitivity.npz")
        data["sensitivity"] = {"features": payload["sensitivity"].to(device), "group": meta["group"],
                               "kind": meta["kind"], "amount": meta["amount"]}
    return data


def evaluate(head: torch.Tensor, val: dict) -> dict:
    """Validation retrieval metrics and, if present, median sensitivity ratios per object move."""
    head = head.to(val["features"].device)
    metrics = lh.retrieval_metrics(val["features"], val["groups"], head)
    sens = val.get("sensitivity")
    if sens is not None:
        # Reference = the group's first photo (same agent as the moved-object photos),
        # so only the T differs between the two photos of each pair.
        first = {g: np.flatnonzero(val["groups"] == g)[0] for g in np.unique(val["groups"])}
        usable = np.array([g in first for g in sens["group"]])
        reference = val["features"][torch.as_tensor([first[g] for g in sens["group"][usable]])]
        ratios = lh.sensitivity_ratios(reference, sens["features"][torch.from_numpy(usable)], head,
                                       metrics["within_median"])
        for kind, amount in sorted(set(zip(sens["kind"][usable], sens["amount"][usable]))):
            mask = (sens["kind"][usable] == kind) & (sens["amount"][usable] == amount)
            metrics[f"sensitivity_{kind}_{amount:g}"] = float(np.median(ratios[mask]))  # e.g. sensitivity_shift_px_20
    return metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` (dicts, union of keys as columns) to ``path``."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the τ/lr grid on val per rank, then all allocations × resamples and the controls."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/linear_head_training.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranks", type=int, nargs="+", help="override the config's ranks (e.g. one per Slurm job)")
    parser.add_argument("--smoke", action="store_true", help="tiny settings to check the pipeline")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(steps=5, eval_every=0, resamples=1, budgets=[16], hands_per_target=[2],
                   grid={"tau": [0.1], "lr": [0.01]}, control_seeds=[0], ranks=[4, 32])
    if args.ranks:
        cfg["ranks"] = args.ranks
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new --output")

    started = perf_counter()
    allocations = lh.allocation_grid(cfg["budgets"], cfg["hands_per_target"])
    train = load_groups(args.train, args.device, min_images=max(cfg["hands_per_target"]))
    val = load_groups(args.val, args.device)
    dim = train["features"].shape[-1]
    if any(not 1 <= rank <= dim for rank in cfg["ranks"]):
        raise ValueError(f"ranks must be between 1 and the feature dimension {dim}")

    def fit(rank, budget, targets, hands, resample, tau, lr, shuffled=False):
        """Fit one head of ``rank`` on a B = M × K allocation; returns ``(head, curve)``."""
        # The seed depends only on the resample, so resample r uses the same image
        # selection and initial L for every rank and every tau/lr setting.
        sampler = lh.AllocationSampler(train["groups"], targets, hands, seed=cfg["seed"] * 1000 + resample,
                                       shuffle_labels=shuffled)
        assert len(sampler.images) == budget  # exactly B = M * K images, as the proposal requires
        return lh.fit_head(train["features"], sampler, steps=cfg["steps"], batch_size=cfg["batch_size"], tau=tau,
                           lr=lr, seed=cfg["seed"] * 1000 + resample, rank=rank,
                           validation=(val["features"], val["groups"]), eval_every=cfg["eval_every"])

    def size(rank):
        """Rank and compression (384 / rank) columns for an output row."""
        return {"rank": rank, "compression": dim / rank}

    (args.output / "heads").mkdir(parents=True)
    grid_rows, curve_rows, validation_rows, chosen = [], [], [], {}
    validation_rows.append({"method": "plain_mse", "head": "identity", **size(dim),
                            **evaluate(lh.identity_head(dim), val)})
    for rank in cfg["ranks"]:
        # 1. (tau, lr) chosen on validation for this rank, shared by its allocations (resample 0).
        fitted, means = {}, {}
        for tau, lr in itertools.product(cfg["grid"]["tau"], cfg["grid"]["lr"]):
            scores = []
            for budget, targets, hands in allocations:
                head, curve = fit(rank, budget, targets, hands, 0, tau, lr)
                fitted[(tau, lr, budget, targets, hands)] = (head, curve)
                scores.append(curve[-1]["val_auroc"])
                grid_rows.append({**size(rank), "tau": tau, "lr": lr, "B": budget, "M": targets, "K": hands,
                                  "val_auroc": scores[-1], "val_top1": curve[-1]["val_top1"]})
            means[(tau, lr)] = float(np.mean(scores))
            print(f"rank={rank} tau={tau} lr={lr} mean val AUROC={means[(tau, lr)]:.4f}", flush=True)
        tau, lr = max(means, key=means.get)  # highest mean val AUROC over the 9 allocations
        chosen[rank] = {"tau": tau, "lr": lr, "mean_val_auroc": means[(tau, lr)]}

        # 2. All allocations x resamples (+ shuffled labels) with the chosen pair.
        variants = [False, True] if cfg["shuffled_control"] else [False]
        for (budget, targets, hands), resample, shuffled in itertools.product(
                allocations, range(cfg["resamples"]), variants):
            if resample == 0 and not shuffled:
                # Already fitted during the grid search with identical settings and seed.
                head, curve = fitted[(tau, lr, budget, targets, hands)]
            else:
                head, curve = fit(rank, budget, targets, hands, resample, tau, lr, shuffled)
            name = f"rank{rank}_B{budget}_M{targets}_K{hands}_r{resample}" + ("_shuffled" if shuffled else "")
            config = {**size(rank), "B": budget, "M": targets, "K": hands, "resample": resample,
                      "shuffled_labels": shuffled, "tau": tau, "lr": lr, "steps": cfg["steps"],
                      "batch_size": cfg["batch_size"]}
            lh.save_head(args.output / "heads" / f"{name}.pt", head, config)
            method = "linear_head_shuffled" if shuffled else "linear_head"
            curve_rows.extend({"head": name, **size(rank), **row} for row in curve)
            validation_rows.append({"method": method, "head": name, **config, **evaluate(head, val)})

        # 3. Untrained controls of the same size on the same validation groups.
        for seed in cfg["control_seeds"]:
            validation_rows.append({"method": "random_projection", "head": f"rank{rank}_random_s{seed}", **size(rank),
                                    "seed": seed, **evaluate(lh.random_head(rank, dim, seed), val)})
            validation_rows.append({"method": "coordinate_subset", "head": f"rank{rank}_coords_s{seed}", **size(rank),
                                    "seed": seed, **evaluate(lh.coordinate_subset_head(rank, dim, seed), val)})

    write_csv(args.output / "grid.csv", grid_rows)
    write_csv(args.output / "curves.csv", curve_rows)
    write_csv(args.output / "validation.csv", validation_rows)
    metadata = {
        "command": sys.argv,
        "config": cfg,
        "chosen": chosen,
        "train": str(args.train),
        "val": str(args.val),
        "train_images": len(train["groups"]),
        "train_groups_dropped_short": train["dropped_groups"],
        "device": args.device,
        "elapsed_seconds": perf_counter() - started,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "slurm": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST")},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "chosen": metadata["chosen"]}, indent=2))


if __name__ == "__main__":
    main()
