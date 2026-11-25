#!/usr/bin/env python3
"""Unified CLI head trainer and spectral auditor for calibrated linear goal heads.

Fits low-rank linear goal heads L in R^{r x D} (||L||_F = 1) on cached DINO-WM
goal-group features under three training regimes:

  - unregularized : standard trace-normalized InfoNCE (linear_head.fit_head)
  - isotropy      : InfoNCE + log-det row-isotropy regularization
  - fantope       : InfoNCE + periodic exact k-Fantope spectral projection

Works identically on PushT and Meta-World goal-group directories containing
``images.npz`` ("group"), ``features.pt`` ("groups", optional "sensitivity"),
and optional ``sensitivity.npz`` ("group", "kind", "amount").

Saves every fitted head via ``linear_head.save_head`` under ``<output>/heads/*.pt``
and writes ``<output>/spectral_audit.csv`` and ``<output>/spectral_audit.json``.

Usage (repository root):
  .venv/bin/python code/experiments/train_calibrated_heads.py \
    --train-groups results/pusht_goal_groups/train \
    --val-groups results/pusht_goal_groups/val \
    --regimes unregularized isotropy fantope \
    --budgets 64 256 1024 --hands 4 --ranks 32 \
    --device cpu --output results/calibrated_heads_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import trust_mpc as tm  # noqa: E402

REGIMES = ("unregularized", "isotropy", "fantope")


def load_goal_groups(directory: Path, device: str, min_images: int = 0) -> dict:
    """Load cached features [n, P, D], group ids, and optional sensitivity pairs.

    Compatible with both PushT and Meta-World goal-group directories. Groups with
    fewer than ``min_images`` photos are excluded.
    """
    directory = Path(directory)
    images_meta = np.load(directory / "images.npz")
    groups = np.asarray(images_meta["group"], dtype=np.int64)
    payload = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    features = payload["groups"]
    if len(features) != len(groups):
        raise ValueError(f"{directory}: {len(features)} features for {len(groups)} images")

    ids, counts = np.unique(groups, return_counts=True)
    keep_mask = np.isin(groups, ids[counts >= min_images])
    data: dict = {
        "features": features[torch.from_numpy(keep_mask)].to(device),
        "groups": groups[keep_mask],
        "dropped_groups": int(np.sum(counts < min_images)),
    }

    sens_path = directory / "sensitivity.npz"
    if "sensitivity" in payload and sens_path.exists():
        sens_meta = np.load(sens_path)
        data["sensitivity"] = {
            "features": payload["sensitivity"].to(device),
            "group": np.asarray(sens_meta["group"], dtype=np.int64),
            "kind": np.asarray(sens_meta["kind"]),
            "amount": np.asarray(sens_meta["amount"], dtype=np.float64),
        }
    return data


def spectral_summary(head: torch.Tensor) -> dict[str, float]:
    """Compute spectral concentration statistics for W = L^T L.

    Returns participation-ratio ``effective_rank``, Roy-Vetterli
    ``entropy_rank``, leading normalized eigenvalue ``top1_eig``, sum of the
    top-4 normalized eigenvalues ``top4_eig_sum``, and ``frobenius_norm``.
    """
    if head.ndim != 2 or min(head.shape) == 0:
        raise ValueError("head must be a non-empty 2D tensor [r, D]")

    matrix = head.detach().float()
    singular_values = torch.linalg.svdvals(matrix)
    eigenvalues = singular_values.square()
    total = eigenvalues.sum().clamp_min(1e-12)
    probs = eigenvalues / total

    eff_rank = float(1.0 / probs.square().sum().clamp_min(1e-12))
    positive = probs[probs > 1e-12]
    entropy = float(-(positive * positive.log()).sum())
    ent_rank = float(math.exp(entropy))
    top1 = float(probs[0])
    top4 = float(probs[: min(4, len(probs))].sum())
    frob = float(torch.linalg.matrix_norm(matrix))

    return {
        "effective_rank": eff_rank,
        "entropy_rank": ent_rank,
        "top1_eig": top1,
        "top4_eig_sum": top4,
        "frobenius_norm": frob,
    }


def evaluate_head(head: torch.Tensor, val_data: dict) -> dict[str, float]:
    """Score validation retrieval metrics and optional object-shift sensitivity ratios."""
    val_features = val_data["features"]
    head_device = head.to(device=val_features.device, dtype=torch.float32)
    metrics = lh.retrieval_metrics(val_features, val_data["groups"], head_device)

    sens = val_data.get("sensitivity")
    if sens is not None and len(sens["group"]) > 0:
        first_index = {
            int(g): int(np.flatnonzero(val_data["groups"] == g)[0])
            for g in np.unique(val_data["groups"])
        }
        usable = np.array([int(g) in first_index for g in sens["group"]], dtype=bool)
        if usable.any():
            ref_indices = torch.as_tensor(
                [first_index[int(g)] for g in sens["group"][usable]],
                dtype=torch.long,
                device=val_features.device,
            )
            reference = torch.index_select(val_features, 0, ref_indices)
            perturbed = sens["features"][torch.from_numpy(usable).to(sens["features"].device)]
            ratios = lh.sensitivity_ratios(reference, perturbed, head_device, metrics["within_median"])

            kinds = sens["kind"][usable]
            amounts = sens["amount"][usable]
            pairs = sorted({(str(k), float(a)) for k, a in zip(kinds, amounts)})
            for kind, amount in pairs:
                mask = (kinds.astype(str) == kind) & np.isclose(amounts, amount)
                if mask.any():
                    metrics[f"sensitivity_{kind}_{amount:g}"] = float(np.median(ratios[mask]))
    return metrics


def make_head_name(
    regime: str,
    rank: int,
    budget: int,
    hands: int,
    resample: int,
    *,
    isotropy_weight: float = 0.015,
    fantope_k: float = 4.0,
) -> str:
    """Deterministic stem for a fitted head checkpoint."""
    if regime == "isotropy":
        tag = f"isotropy_w{isotropy_weight:g}"
    elif regime == "fantope":
        tag = f"fantope_k{fantope_k:g}"
    elif regime == "unregularized":
        tag = "unregularized"
    else:
        raise ValueError(f"unsupported regime: {regime}")
    return f"{tag}_r{rank}_B{budget}_K{hands}_s{resample}"


def train_and_audit_head(
    train_data: dict,
    val_data: dict,
    *,
    regime: str,
    budget: int,
    hands: int = 4,
    rank: int = lh.HEAD_RANK,
    resample: int = 0,
    steps: int = 2000,
    batch_size: int = 64,
    tau: float = 0.3,
    lr: float = 0.003,
    isotropy_weight: float = 0.015,
    fantope_k: float = 4.0,
    project_every: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, dict, dict]:
    """Fit one linear head under ``regime`` and return ``(head, config, audit_row)``."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if budget <= 0 or hands < 2 or budget % hands != 0:
        raise ValueError(f"budget ({budget}) must be a positive multiple of hands ({hands} >= 2)")

    train_features = train_data["features"]
    feature_dim = int(train_features.shape[-1])
    if not 1 <= rank <= feature_dim:
        raise ValueError(f"rank must be between 1 and {feature_dim}, got {rank}")

    targets = budget // hands
    run_seed = int(seed) * 1000 + int(resample)
    sampler = lh.AllocationSampler(
        train_data["groups"],
        num_targets=targets,
        hands_per_target=hands,
        seed=run_seed,
    )
    if len(sampler.images) != budget:
        raise RuntimeError(f"expected {budget} sampled images, got {len(sampler.images)}")

    started = perf_counter()
    if regime == "unregularized":
        head, _ = lh.fit_head(
            train_features,
            sampler,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            seed=run_seed,
            rank=rank,
        )
    elif regime == "isotropy":
        head = tm.fit_isotropy_head(
            train_features,
            sampler,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            isotropy_weight=isotropy_weight,
            seed=run_seed,
            rank=rank,
        )
    else:
        head = tm.fit_fantope_head(
            train_features,
            sampler,
            k_fantope=fantope_k,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            seed=run_seed,
            rank=rank,
            project_every=project_every,
        )
    elapsed = perf_counter() - started

    head = lh.normalized(head.detach().cpu().float())
    head_name = make_head_name(
        regime,
        rank,
        budget,
        hands,
        resample,
        isotropy_weight=isotropy_weight,
        fantope_k=fantope_k,
    )
    spec = spectral_summary(head)
    val_metrics = evaluate_head(head, val_data)

    config = {
        "regime": regime,
        "rank": int(rank),
        "feature_dim": feature_dim,
        "compression": float(feature_dim / rank),
        "B": int(budget),
        "M": int(targets),
        "K": int(hands),
        "resample": int(resample),
        "seed": run_seed,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "tau": float(tau),
        "lr": float(lr),
        "isotropy_weight": float(isotropy_weight) if regime == "isotropy" else 0.0,
        "fantope_k": float(fantope_k) if regime == "fantope" else None,
        "project_every": int(project_every) if regime == "fantope" else None,
        "shuffled_labels": False,
    }
    audit_row = {
        "head": head_name,
        "head_path": f"heads/{head_name}.pt",
        "regime": regime,
        "rank": int(rank),
        "compression": float(feature_dim / rank),
        "B": int(budget),
        "M": int(targets),
        "K": int(hands),
        "resample": int(resample),
        "tau": float(tau),
        "lr": float(lr),
        "steps": int(steps),
        "isotropy_weight": float(isotropy_weight) if regime == "isotropy" else "",
        "fantope_k": float(fantope_k) if regime == "fantope" else "",
        **spec,
        **val_metrics,
        "elapsed_seconds": float(elapsed),
    }
    return head, config, audit_row


def write_audit_csv(path: Path, rows: list[dict]) -> None:
    """Write spectral and validation audit rows to CSV."""
    if not rows:
        raise ValueError("rows must be non-empty")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for calibrated head training and auditing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-groups", type=Path, required=True, help="Training goal-group directory")
    parser.add_argument("--val-groups", type=Path, required=True, help="Validation goal-group directory")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=list(REGIMES),
        help="Spectral training regimes to fit",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[64, 256, 1024],
        help="Calibration image budgets B = M * K",
    )
    parser.add_argument(
        "--hands",
        type=int,
        nargs="+",
        default=[4],
        help="Photos per target K (one or more integers, default: 4)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[32],
        help="Head ranks r (one or more integers, default: 32)",
    )
    parser.add_argument(
        "--isotropy-weight",
        type=float,
        default=0.015,
        help="Log-det isotropy penalty weight for the isotropy regime",
    )
    parser.add_argument(
        "--fantope-k",
        type=float,
        default=4.0,
        help="Target k-Fantope spectral floor for the fantope regime",
    )
    parser.add_argument(
        "--project-every",
        type=int,
        default=5,
        help="Projection cadence in optimization steps for the fantope regime",
    )
    parser.add_argument("--resamples", type=int, default=1, help="Number of allocation resamples")
    parser.add_argument("--steps", type=int, default=2000, help="Adam optimization steps per head")
    parser.add_argument("--batch-size", type=int, default=64, help="InfoNCE mini-batch size")
    parser.add_argument("--tau", type=float, default=0.3, help="InfoNCE temperature tau")
    parser.add_argument("--lr", type=float, default=0.003, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for allocation and initialization")
    parser.add_argument(
        "--include-identity",
        action="store_true",
        help="Also record the full-rank identity (plain MSE) reference row in the audit",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, mps, cuda)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for heads and audit")
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    """Run calibrated head training and spectral auditing across requested grids."""
    args = build_parser().parse_args(argv)
    if args.resamples < 1:
        raise ValueError("--resamples must be >= 1")

    started = perf_counter()
    train_data = load_goal_groups(args.train_groups, args.device, min_images=max(args.hands))
    val_data = load_goal_groups(args.val_groups, args.device)
    feature_dim = int(train_data["features"].shape[-1])

    heads_dir = args.output / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.include_identity:
        id_head = lh.identity_head(feature_dim)
        rows.append(
            {
                "head": "identity",
                "head_path": "",
                "regime": "identity",
                "rank": feature_dim,
                "compression": 1.0,
                "B": "",
                "M": "",
                "K": "",
                "resample": "",
                "tau": "",
                "lr": "",
                "steps": 0,
                "isotropy_weight": "",
                "fantope_k": "",
                **spectral_summary(id_head),
                **evaluate_head(id_head, val_data),
                "elapsed_seconds": 0.0,
            }
        )

    for rank in args.ranks:
        for budget in args.budgets:
            for hands in args.hands:
                for regime in args.regimes:
                    for resample in range(args.resamples):
                        head, config, audit_row = train_and_audit_head(
                            train_data,
                            val_data,
                            regime=regime,
                            budget=budget,
                            hands=hands,
                            rank=rank,
                            resample=resample,
                            steps=args.steps,
                            batch_size=args.batch_size,
                            tau=args.tau,
                            lr=args.lr,
                            isotropy_weight=args.isotropy_weight,
                            fantope_k=args.fantope_k,
                            project_every=args.project_every,
                            seed=args.seed,
                        )
                        lh.save_head(heads_dir / f"{audit_row['head']}.pt", head, config)
                        rows.append(audit_row)
                        print(
                            f"{audit_row['head']} | r_eff={audit_row['effective_rank']:.2f} "
                            f"top1_eig={audit_row['top1_eig']:.3f} | "
                            f"val_top1={audit_row['top1']:.3f} val_auroc={audit_row['auroc']:.4f}",
                            flush=True,
                        )

    csv_path = args.output / "spectral_audit.csv"
    json_path = args.output / "spectral_audit.json"
    write_audit_csv(csv_path, rows)

    payload = {
        "train_groups": str(args.train_groups),
        "val_groups": str(args.val_groups),
        "regimes": list(args.regimes),
        "budgets": [int(b) for b in args.budgets],
        "hands": [int(k) for k in args.hands],
        "ranks": [int(r) for r in args.ranks],
        "isotropy_weight": float(args.isotropy_weight),
        "fantope_k": float(args.fantope_k),
        "project_every": int(args.project_every),
        "resamples": int(args.resamples),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "tau": float(args.tau),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "device": args.device,
        "feature_dim": feature_dim,
        "train_images": int(len(train_data["groups"])),
        "val_images": int(len(val_data["groups"])),
        "train_groups_dropped_short": int(train_data["dropped_groups"]),
        "elapsed_seconds": float(perf_counter() - started),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(rows)} audit rows to {csv_path}", flush=True)
    return rows


if __name__ == "__main__":
    main()
