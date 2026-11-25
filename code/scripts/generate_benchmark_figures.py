#!/usr/bin/env python3
"""Generate publication-grade benchmark figures from saved heads and evaluation summaries.

Supports three independent figure products (generated whenever the matching
inputs are supplied):

  1. Eigenspectra comparison (``--heads`` and/or ``--heads-dir``)
  2. Closed-loop forest plot (``--closed-loop-json`` or ``--episodes-csv``)
  3. Calibration budget scaling curve (``--scaling-json`` or ``--scaling-csv``,
     optionally joined with ``--spectral-audit`` for ``effective_rank``)

Usage (repository root):
  .venv/bin/python code/scripts/generate_benchmark_figures.py \
    --heads-dir results/calibrated_heads/heads \
    --closed-loop-json results/metaworld_trust_mpc/summary.json \
    --scaling-csv results/budget_scaling.csv \
    --theme light --pdf --output results/figures
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning.benchmark_figures import (  # noqa: E402
    THEMES,
    plot_budget_scaling_curve,
    plot_closed_loop_forest,
    plot_eigenspectra_comparison,
)


def _iqm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    ordered = np.sort(values, axis=axis)
    n = ordered.shape[axis]
    low, high = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.take(ordered, range(low, max(high, low + 1)), axis=axis).mean(axis=axis)


def _bootstrap_stats(values: np.ndarray, draws: int = 4000, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(draws, len(arr)))
    samples = arr[idx]
    means = samples.mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "iqm": float(_iqm(arr)),
    }


def load_heads_from_args(
    head_specs: list[str],
    heads_dir: Path | None,
    include_identity: bool = False,
) -> dict[str, torch.Tensor]:
    """Load named ``[r, D]`` heads from ``LABEL=PATH`` specs and/or a directory."""
    heads: dict[str, torch.Tensor] = {}
    if heads_dir is not None:
        for path in sorted(Path(heads_dir).glob("*.pt")):
            matrix, _ = lh.load_head(path)
            heads[path.stem] = matrix
    for spec in head_specs:
        if "=" in spec:
            label, raw_path = spec.split("=", 1)
            path = Path(raw_path)
        else:
            path = Path(spec)
            label = path.stem
        matrix, _ = lh.load_head(path)
        heads[label] = matrix
    if include_identity and heads:
        dim = int(next(iter(heads.values())).shape[-1])
        heads["Plain MSE (identity)"] = lh.identity_head(dim)
    return heads


def load_closed_loop_payload(
    json_path: Path | None,
    episodes_csv: Path | None,
    *,
    unit: str | None = None,
    metric: str | None = None,
) -> tuple[list[dict], list[dict], str, str]:
    """Parse closed-loop summary rows and paired contrasts from JSON or episodes CSV."""
    if json_path is not None:
        raw = json.loads(Path(json_path).read_text())
        if "summary_rows" in raw:
            resolved_unit = unit or str(raw.get("unit", "px"))
            resolved_title = str(raw.get("title", "Closed-loop planning benchmark"))
            return list(raw["summary_rows"]), list(raw.get("paired_contrasts", [])), resolved_unit, resolved_title

        if "conditions" in raw:
            # Compatible with run_metaworld_trust_mpc.py summary.json
            metric_key = metric or "final_error_3d_cm"
            resolved_unit = unit or ("cm" if "cm" in metric_key else "px")
            baseline = raw.get("baseline_condition", {})
            summary_rows: list[dict] = []
            paired_contrasts: list[dict] = []
            for cond in raw["conditions"]:
                m_stats = cond["metrics"][metric_key]
                k_tag = f" (K={cond['goal_photos']})" if "goal_photos" in cond else ""
                label = f"{cond['head']} + {cond['planner']}{k_tag}"
                is_base = (
                    cond.get("head") == baseline.get("head")
                    and cond.get("planner") == baseline.get("planner")
                    and cond.get("goal_photos") == baseline.get("goal_photos")
                )
                summary_rows.append(
                    {
                        "label": label,
                        "mean": float(m_stats["mean"]),
                        "ci_lo": float(m_stats["ci_lo"]),
                        "ci_hi": float(m_stats["ci_hi"]),
                        "iqm": float(m_stats["iqm"]),
                        "is_baseline": is_base,
                    }
                )
                paired = cond.get("paired_vs_baseline", {}).get(metric_key)
                if paired is not None:
                    paired_contrasts.append(
                        {
                            "label": label,
                            "diff": float(paired["mean"]),
                            "ci_lo": float(paired["ci_lo"]),
                            "ci_hi": float(paired["ci_hi"]),
                        }
                    )
            title = (
                "Meta-World Sawyer Push-v3 closed-loop benchmark"
                if resolved_unit == "cm"
                else "PushT closed-loop planning benchmark"
            )
            return summary_rows, paired_contrasts, resolved_unit, title
        raise ValueError(f"Unrecognized closed-loop JSON schema in {json_path}")

    if episodes_csv is None:
        raise ValueError("Provide --closed-loop-json or --episodes-csv")

    with Path(episodes_csv).open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{episodes_csv} contains no rows")

    if metric is not None:
        metric_key = metric
    elif "final_error_3d_cm" in rows[0]:
        metric_key = "final_error_3d_cm"
    elif "final_position_error" in rows[0]:
        metric_key = "final_position_error"
    else:
        raise ValueError(f"Cannot infer error column from {list(rows[0].keys())}")

    resolved_unit = unit or ("cm" if "cm" in metric_key else "px")
    has_goal_photos = "goal_photos" in rows[0]

    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k_val = str(row["goal_photos"]) if has_goal_photos else ""
        cond_key = (str(row["head"]), str(row["planner"]), k_val)
        grouped[cond_key][int(row["task"])].append(float(row[metric_key]))

    per_task = {
        key: {task_id: float(np.mean(vals)) for task_id, vals in task_map.items()}
        for key, task_map in grouped.items()
    }
    ordered_keys = sorted(per_task.keys(), key=lambda k: (k[0] != "identity", k[1] != "standard_cem", k))
    baseline_key = next(
        (k for k in ordered_keys if k[0] == "identity" and k[1] == "standard_cem"),
        ordered_keys[0],
    )

    summary_rows = []
    paired_contrasts = []
    base_map = per_task[baseline_key]

    for idx, key in enumerate(ordered_keys):
        head_name, planner_name, k_val = key
        label = f"{head_name} + {planner_name}" + (f" (K={k_val})" if k_val else "")
        task_ids = sorted(per_task[key].keys())
        values = np.array([per_task[key][t] for t in task_ids], dtype=np.float64)
        stats = _bootstrap_stats(values, seed=idx)
        summary_rows.append({"label": label, **stats, "is_baseline": key == baseline_key})

        if key != baseline_key:
            shared = [t for t in task_ids if t in base_map]
            if shared:
                diffs = np.array([per_task[key][t] - base_map[t] for t in shared], dtype=np.float64)
                p_stats = _bootstrap_stats(diffs, seed=1000 + idx)
                wins = int(np.sum(diffs < 0.0))
                losses = int(np.sum(diffs > 0.0))
                ties = int(len(diffs) - wins - losses)
                paired_contrasts.append(
                    {
                        "label": label,
                        "diff": p_stats["mean"],
                        "ci_lo": p_stats["ci_lo"],
                        "ci_hi": p_stats["ci_hi"],
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "total": len(shared),
                    }
                )

    title = (
        "Meta-World Sawyer Push-v3 closed-loop benchmark"
        if resolved_unit == "cm"
        else "PushT closed-loop planning benchmark"
    )
    return summary_rows, paired_contrasts, resolved_unit, title


def load_scaling_rows(
    scaling_path: Path,
    spectral_audit_path: Path | None = None,
) -> list[dict]:
    """Load scaling rows from CSV or JSON, optionally joining ``effective_rank`` from an audit file."""
    path = Path(scaling_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        rows = list(payload["rows"]) if isinstance(payload, dict) and "rows" in payload else list(payload)
    else:
        with path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))

    audit_rank: dict[tuple[str, int], list[float]] = defaultdict(list)
    if spectral_audit_path is not None:
        audit_path = Path(spectral_audit_path)
        if audit_path.suffix.lower() == ".json":
            audit_payload = json.loads(audit_path.read_text())
            audit_rows = audit_payload.get("rows", audit_payload)
        else:
            with audit_path.open("r", newline="") as handle:
                audit_rows = list(csv.DictReader(handle))
        for arow in audit_rows:
            if arow.get("regime") and arow.get("B") not in (None, ""):
                audit_rank[(str(arow["regime"]), int(float(arow["B"])))].append(
                    float(arow["effective_rank"])
                )

    normalized: list[dict] = []
    for row in rows:
        item = dict(row)
        regime = str(item.get("regime") or item.get("label") or item.get("method") or "method")
        budget = int(float(item.get("B") if item.get("B") not in (None, "") else item["budget"]))
        item["regime"] = regime
        item["B"] = budget
        if not item.get("effective_rank") and not item.get("r_eff"):
            ranks = audit_rank.get((regime, budget))
            if ranks:
                item["effective_rank"] = float(np.mean(ranks))
        normalized.append(item)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for publication benchmark figure generation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--heads",
        nargs="*",
        default=[],
        help="Head checkpoints as PATH or LABEL=PATH for the eigenspectra figure",
    )
    parser.add_argument(
        "--heads-dir",
        type=Path,
        default=None,
        help="Directory of saved .pt heads for the eigenspectra figure",
    )
    parser.add_argument(
        "--include-identity",
        action="store_true",
        help="Include full-rank identity reference in the eigenspectra figure",
    )
    parser.add_argument("--closed-loop-json", type=Path, default=None, help="Closed-loop summary JSON")
    parser.add_argument("--episodes-csv", type=Path, default=None, help="Closed-loop episodes CSV")
    parser.add_argument("--metric", type=str, default=None, help="Error metric column override")
    parser.add_argument("--scaling-input", type=Path, default=None, help="Budget scaling CSV or JSON")
    parser.add_argument(
        "--spectral-audit",
        type=Path,
        default=None,
        help="Optional spectral_audit.csv/json to supply effective_rank for budget scaling",
    )
    parser.add_argument("--baseline-error", type=float, default=None, help="Optional baseline error line")
    parser.add_argument("--unit", choices=("px", "cm"), default=None, help="Error unit override (px or cm)")
    parser.add_argument("--title", type=str, default=None, help="Optional title override for forest plot")
    parser.add_argument("--theme", choices=tuple(THEMES), default="light", help="Color theme (light or dark)")
    parser.add_argument("--dpi", type=int, default=300, help="Raster export DPI")
    parser.add_argument("--pdf", action="store_true", help="Also write companion vector PDF files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for generated figures")
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    """Generate all requested benchmark figures and return the list of written paths."""
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.heads or args.heads_dir is not None:
        heads = load_heads_from_args(args.heads, args.heads_dir, include_identity=args.include_identity)
        if heads:
            out_path = args.output / "eigenspectra_comparison.png"
            plot_eigenspectra_comparison(
                heads,
                out_path,
                theme=args.theme,
                dpi=args.dpi,
                also_pdf=args.pdf,
            )
            written.append(out_path)
            print(f"wrote {out_path}", flush=True)

    if args.closed_loop_json is not None or args.episodes_csv is not None:
        summary_rows, paired_contrasts, unit, default_title = load_closed_loop_payload(
            args.closed_loop_json,
            args.episodes_csv,
            unit=args.unit,
            metric=args.metric,
        )
        out_path = args.output / "closed_loop_forest.png"
        plot_closed_loop_forest(
            summary_rows,
            paired_contrasts,
            unit=unit,
            title=args.title or default_title,
            output_path=out_path,
            theme=args.theme,
            dpi=args.dpi,
            also_pdf=args.pdf,
        )
        written.append(out_path)
        print(f"wrote {out_path}", flush=True)

    if args.scaling_input is not None:
        scaling_rows = load_scaling_rows(args.scaling_input, args.spectral_audit)
        out_path = args.output / "budget_scaling_curve.png"
        plot_budget_scaling_curve(
            scaling_rows,
            out_path,
            unit=args.unit or "px",
            theme=args.theme,
            baseline_error=args.baseline_error,
            dpi=args.dpi,
            also_pdf=args.pdf,
        )
        written.append(out_path)
        print(f"wrote {out_path}", flush=True)

    if not written:
        raise ValueError(
            "No figure inputs provided. Pass at least one of --heads/--heads-dir, "
            "--closed-loop-json/--episodes-csv, or --scaling-input."
        )
    return written


if __name__ == "__main__":
    main()
