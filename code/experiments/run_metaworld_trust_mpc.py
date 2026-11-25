#!/usr/bin/env python3
"""Paired closed-loop Meta-World Sawyer Push-v3 evaluation for linear heads and Trust-MPC.

Runs paired episodes across goal heads (``identity`` plain MSE plus any saved
``--head-files``), planners (``standard_cem``, ``trust_mpc``), and multi-goal
prototype sizes (``--goal-photos``, e.g. ``1 2 4``) using identical start
snapshots, exact ``2 * N`` rollout budget parity, and common standard-normal
streams per ``(seed, target_seed, replan)``.

Writes ``episodes.csv`` and ``summary.json`` (including task-level paired
bootstrap 95% confidence intervals) to ``--output``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import metaworld_planning as mwp  # noqa: E402
from jepa_planning.pusht_multi_goal import cem_standard_normals  # noqa: E402

SUMMARY_METRICS = (
    "final_error_3d_cm",
    "final_error_xy_cm",
    "best_error_3d_cm",
    "post_peak_drift_cm",
    "success_5cm",
    "success_3cm",
    "success_2cm",
    "native_success",
    "unrolls",
    "planning_seconds",
)
PAIRED_METRICS = (
    "final_error_3d_cm",
    "final_error_xy_cm",
    "success_5cm",
    "success_3cm",
    "success_2cm",
    "native_success",
)


def load_tasks_from_groups(
    groups_dir: Path,
    required_photos: int,
    max_tasks: int | None = None,
) -> tuple[dict, torch.Tensor, dict]:
    """Load Meta-World target seeds, puck goals, and cached features for one split."""
    meta = json.loads((groups_dir / "metadata.json").read_text())
    group_cfg = meta["config"]
    images = np.load(groups_dir / "images.npz")

    groups = np.asarray(images["group"], dtype=np.int64)
    slots = np.asarray(images["slot"], dtype=np.int64)
    target_seeds = np.asarray(images["target_seed"], dtype=np.int64)
    puck_poses = np.asarray(images["puck_pose"], dtype=np.float64)

    num_groups = int(groups.max()) + 1 if len(groups) else 0
    photo_index = np.full((num_groups, required_photos), -1, dtype=np.int64)
    group_seed = np.full(num_groups, -1, dtype=np.int64)
    group_puck = np.zeros((num_groups, 3), dtype=np.float64)

    for row_idx, (grp, slot) in enumerate(zip(groups, slots)):
        if group_seed[grp] < 0:
            group_seed[grp] = target_seeds[row_idx]
            group_puck[grp] = puck_poses[row_idx, :3]
        if 0 <= slot < required_photos:
            photo_index[grp, slot] = row_idx

    keep = np.flatnonzero((photo_index >= 0).all(axis=1))
    if max_tasks is not None:
        keep = keep[:max_tasks]
    if len(keep) == 0:
        raise ValueError(f"No goal groups in {groups_dir} have at least {required_photos} photos")

    feature_bank = torch.load(groups_dir / "features.pt", map_location="cpu", weights_only=True)["groups"]
    kept_indices = photo_index[keep]
    goals = feature_bank[torch.from_numpy(kept_indices.ravel())].reshape(
        *kept_indices.shape, *feature_bank.shape[1:]
    )
    tasks = {
        "group": keep,
        "target_seed": group_seed[keep],
        "goal_puck": group_puck[keep],
    }
    return tasks, goals, group_cfg


def build_start_snapshots(env, target_seeds: np.ndarray, warmup_steps: int) -> list[dict]:
    """Build deterministic post-warmup start snapshots for every task once."""
    from jepa_planning import metaworld_push as mp

    snapshots = []
    for seed in target_seeds:
        obs = mp.reset_task(env, int(seed))
        mp.run_expert(env, obs, warmup_steps)
        snapshots.append(mp.snapshot(env))
    return snapshots


def run_episode_suite(
    env,
    model,
    head: torch.Tensor,
    head_name: str,
    planner_name: str,
    goal_photos: int,
    tasks: dict,
    start_snapshots: list[dict],
    goals: torch.Tensor,
    cfg: dict,
) -> list[dict]:
    """Run paired closed-loop episodes for one ``(head, planner, goal_photos)`` tuple."""
    from jepa_planning import metaworld_push as mp

    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    controls_per_chunk = int(cfg["controls_per_chunk"])
    lower, upper, hold_chunk = mwp.metaworld_action_bounds(
        action_mean, action_std, controls_per_chunk
    )
    head_device = head.to(model.device).float()
    step_weights = cfg.get("step_weights")
    replicates = int(cfg.get("replicates", 1))
    rows = []

    for task_idx in range(len(tasks["target_seed"])):
        target_seed = int(tasks["target_seed"][task_idx])
        goal_puck = tasks["goal_puck"][task_idx]
        start_snap = start_snapshots[task_idx]

        max_reps = min(replicates, goals.shape[1] // goal_photos)
        for rep_idx in range(max_reps):
            photo_slice = slice(rep_idx * goal_photos, (rep_idx + 1) * goal_photos)
            goal_tensor = goals[task_idx, photo_slice].to(model.device).float()

            mp.restore(env, start_snap)
            start_metrics = {
                k: float(v)
                for k, v in mwp.puck_metrics_cm(mp.puck_pose(env)[:3], goal_puck).items()
            }
            best_error_3d = start_metrics["error_3d_cm"]
            ever_5cm = bool(start_metrics["success_5cm"])
            ever_3cm = bool(start_metrics["success_3cm"])
            ever_2cm = bool(start_metrics["success_2cm"])
            step_metrics = start_metrics

            previous_plan = None
            unrolls = 0
            sim_steps = 0
            jensen_violations = 0
            stay_gate_triggers = 0
            native_successes: list[bool] = []
            done = False
            started = perf_counter()

            for replan in range(int(cfg["replans"])):
                if done:
                    break
                observation = mp.observe(env)
                normals = cem_standard_normals(
                    int(cfg["seed"]),
                    target_seed,
                    replan,
                    int(cfg["cem_iterations"]),
                    int(cfg["cem_samples"]),
                    int(cfg["cem_horizon"]),
                    model.action_dim,
                )
                if planner_name == "standard_cem":
                    action_chunk, _, info_plan = mwp.metaworld_cem_plan(
                        model,
                        observation,
                        goal_tensor,
                        head_device,
                        normals,
                        lower,
                        upper,
                        num_elites=int(cfg["cem_elites"]),
                        batch_size=int(cfg["batch_size"]),
                        step_weights=step_weights,
                    )
                elif planner_name == "trust_mpc":
                    action_chunk, previous_plan, info_plan = mwp.metaworld_trust_mpc_plan(
                        model,
                        observation,
                        goal_tensor,
                        head_device,
                        normals,
                        lower,
                        upper,
                        hold_chunk,
                        previous_plan,
                        num_elites=int(cfg["cem_elites"]),
                        batch_size=int(cfg["batch_size"]),
                        stay_margin=float(cfg.get("stay_margin", 0.02)),
                        step_weights=step_weights,
                        dual_certify=bool(cfg.get("dual_certify", True)),
                    )
                    jensen_violations += int(info_plan["jensen_violation"])
                    stay_gate_triggers += int(info_plan["used_stay_gate"])
                else:
                    raise ValueError(f"Unknown planner: {planner_name}")

                unrolls += int(info_plan["unrolls"])

                for control in action_chunk.reshape(controls_per_chunk, 4):
                    raw_action = np.clip(control * action_std + action_mean, -1.0, 1.0).astype(np.float32)
                    _, _, terminated, truncated, step_info = env.step(raw_action)
                    sim_steps += 1
                    native_successes.append(bool(step_info.get("success", False)))
                    if terminated or truncated:
                        done = True
                        break

                step_metrics = {
                    k: float(v)
                    for k, v in mwp.puck_metrics_cm(mp.puck_pose(env)[:3], goal_puck).items()
                }
                best_error_3d = min(best_error_3d, step_metrics["error_3d_cm"])
                ever_5cm = ever_5cm or bool(step_metrics["success_5cm"])
                ever_3cm = ever_3cm or bool(step_metrics["success_3cm"])
                ever_2cm = ever_2cm or bool(step_metrics["success_2cm"])

            rows.append(
                {
                    "head": head_name,
                    "planner": planner_name,
                    "goal_photos": int(goal_photos),
                    "replicate": int(rep_idx),
                    "task": int(task_idx),
                    "group": int(tasks["group"][task_idx]),
                    "target_seed": target_seed,
                    "initial_error_3d_cm": start_metrics["error_3d_cm"],
                    "initial_error_xy_cm": start_metrics["error_xy_cm"],
                    "final_error_3d_cm": step_metrics["error_3d_cm"],
                    "final_error_xy_cm": step_metrics["error_xy_cm"],
                    "best_error_3d_cm": best_error_3d,
                    "post_peak_drift_cm": step_metrics["error_3d_cm"] - best_error_3d,
                    "success_5cm": int(bool(step_metrics["success_5cm"])),
                    "success_3cm": int(bool(step_metrics["success_3cm"])),
                    "success_2cm": int(bool(step_metrics["success_2cm"])),
                    "ever_success_5cm": int(ever_5cm),
                    "ever_success_3cm": int(ever_3cm),
                    "ever_success_2cm": int(ever_2cm),
                    "native_success": int(native_successes[-1]) if native_successes else 0,
                    "native_success_any": int(any(native_successes)),
                    "unrolls": unrolls,
                    "sim_steps": sim_steps,
                    "jensen_violations": jensen_violations,
                    "stay_gate_triggers": stay_gate_triggers,
                    "planning_seconds": perf_counter() - started,
                }
            )
    return rows


def _iqm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean along ``axis``."""
    ordered = np.sort(values, axis=axis)
    n = ordered.shape[axis]
    low, high = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.take(ordered, range(low, max(high, low + 1)), axis=axis).mean(axis=axis)


def paired_bootstrap_ci(
    values: np.ndarray,
    draws: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean, IQM, and 95% percentile bootstrap CI over task-level values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(draws, len(arr)))
    samples = arr[indices]
    means = samples.mean(axis=1)
    iqms = _iqm(samples, axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "iqm": float(_iqm(arr)),
        "iqm_ci_lo": float(np.percentile(iqms, 2.5)),
        "iqm_ci_hi": float(np.percentile(iqms, 97.5)),
    }


def summarize_episodes(
    rows: list[dict],
    *,
    bootstrap_draws: int = 5000,
    seed: int = 0,
) -> dict:
    """Summarize episodes by ``(head, planner, goal_photos)`` with paired 95% CIs."""
    if not rows:
        raise ValueError("rows must be non-empty")

    grouped: dict[tuple[str, str, int], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["head"]), str(row["planner"]), int(row["goal_photos"]))
        grouped[key][int(row["task"])].append(row)

    per_condition_task: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for key, by_task in grouped.items():
        per_condition_task[key] = {}
        for task_id, task_rows in by_task.items():
            per_condition_task[key][task_id] = {
                metric: float(np.mean([r[metric] for r in task_rows])) for metric in SUMMARY_METRICS
            }

    sorted_keys = sorted(per_condition_task.keys(), key=lambda k: (k[2], k[0] != "identity", k[0], k[1]))
    baseline_key = next(
        (k for k in sorted_keys if k[0] == "identity" and k[1] == "standard_cem"),
        sorted_keys[0],
    )

    conditions = []
    for cond_idx, key in enumerate(sorted_keys):
        head_name, planner_name, goal_k = key
        task_map = per_condition_task[key]
        task_ids = sorted(task_map.keys())
        metrics_summary = {}
        for m_idx, metric in enumerate(SUMMARY_METRICS):
            vals = np.array([task_map[t][metric] for t in task_ids], dtype=np.float64)
            metrics_summary[metric] = paired_bootstrap_ci(
                vals, draws=bootstrap_draws, seed=seed + cond_idx * 100 + m_idx
            )

        paired_vs_baseline = {}
        if key != baseline_key:
            base_map = per_condition_task[baseline_key]
            shared_tasks = [t for t in task_ids if t in base_map]
            if shared_tasks:
                for m_idx, metric in enumerate(PAIRED_METRICS):
                    diffs = np.array(
                        [task_map[t][metric] - base_map[t][metric] for t in shared_tasks],
                        dtype=np.float64,
                    )
                    paired_vs_baseline[metric] = paired_bootstrap_ci(
                        diffs, draws=bootstrap_draws, seed=seed + 10_000 + cond_idx * 100 + m_idx
                    )

        conditions.append(
            {
                "head": head_name,
                "planner": planner_name,
                "goal_photos": goal_k,
                "tasks": len(task_ids),
                "episodes": sum(len(grouped[key][t]) for t in task_ids),
                "metrics": metrics_summary,
                "paired_vs_baseline": paired_vs_baseline,
            }
        )

    return {
        "baseline_condition": {
            "head": baseline_key[0],
            "planner": baseline_key[1],
            "goal_photos": baseline_key[2],
        },
        "bootstrap_draws": int(bootstrap_draws),
        "conditions": conditions,
    }


def _parse_step_weights(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    weights = [float(token) for token in raw.replace(",", " ").split() if token]
    return weights if weights else None


def main() -> None:
    """CLI entrypoint for paired Meta-World closed-loop Trust-MPC evaluation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=Path, required=True, help="Encoded Meta-World goal-group directory")
    parser.add_argument("--head-files", type=Path, nargs="*", default=[], help="Saved .pt linear head files")
    parser.add_argument("--planners", nargs="+", choices=("standard_cem", "trust_mpc"),
                        default=["standard_cem", "trust_mpc"])
    parser.add_argument("--goal-photos", type=int, nargs="+", default=[1],
                        help="Multi-goal prototype sizes K to evaluate (e.g. 1 2 4)")
    parser.add_argument("--replicates", type=int, default=1, help="Disjoint photo-set replicates per task")
    parser.add_argument("--config", type=Path, help="Optional YAML config overriding planner defaults")
    parser.add_argument("--tasks", type=int, default=20, help="Number of held-out tasks to evaluate")
    parser.add_argument("--replans", type=int, default=10, help="Closed-loop replans per episode")
    parser.add_argument("--horizon", type=int, default=4, help="CEM planning horizon in action chunks")
    parser.add_argument("--samples", type=int, default=64, help="Samples per CEM iteration (total budget = 2 * N)")
    parser.add_argument("--elites", type=int, default=8, help="Number of CEM elites")
    parser.add_argument("--batch-size", type=int, default=32, help="World-model unroll batch size")
    parser.add_argument("--stay-margin", type=float, default=0.02, help="Relative hold-cost gate margin")
    parser.add_argument("--step-weights", type=str, default=None, help="Optional horizon weights (comma/space list)")
    parser.add_argument("--no-dual-certify", action="store_true", help="Certify only elite_mean instead of dual mode")
    parser.add_argument("--seed", type=int, default=20261124, help="Base seed for common CEM random numbers")
    parser.add_argument("--bootstrap-draws", type=int, default=5000, help="Task-level bootstrap draws for 95%% CIs")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = {
        "seed": args.seed,
        "controls_per_chunk": 5,
        "cem_horizon": args.horizon,
        "replans": args.replans,
        "cem_samples": args.samples,
        "cem_elites": args.elites,
        "cem_iterations": 2,
        "batch_size": args.batch_size,
        "stay_margin": args.stay_margin,
        "dual_certify": not args.no_dual_certify,
        "step_weights": _parse_step_weights(args.step_weights),
        "replicates": args.replicates,
    }
    if args.config is not None:
        cfg.update(yaml.safe_load(args.config.read_text()))

    required_photos = max(args.goal_photos) * int(cfg["replicates"])
    tasks, goals, group_cfg = load_tasks_from_groups(args.groups, required_photos, args.tasks)

    heads: list[tuple[str, torch.Tensor]] = [("identity", lh.identity_head(int(goals.shape[-1])))]
    for path in args.head_files:
        matrix, _ = lh.load_head(path)
        heads.append((path.stem, matrix))

    from jepa_planning import metaworld_push as mp
    from jepa_planning.model import load_world_model

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_world_model("metaworld", args.device).requires_grad_(False).eval()
    if model.action_dim != 4 * int(cfg["controls_per_chunk"]):
        raise ValueError("controls_per_chunk does not match the Meta-World checkpoint action dimension")

    env = mp.make_push_env()
    all_rows: list[dict] = []
    try:
        start_snapshots = build_start_snapshots(env, tasks["target_seed"], int(group_cfg["warmup_steps"]))
        for goal_k in args.goal_photos:
            for head_name, matrix in heads:
                for planner_name in args.planners:
                    rows = run_episode_suite(
                        env,
                        model,
                        matrix,
                        head_name,
                        planner_name,
                        int(goal_k),
                        tasks,
                        start_snapshots,
                        goals,
                        cfg,
                    )
                    all_rows.extend(rows)
                    mean_err = float(np.mean([r["final_error_3d_cm"] for r in rows]))
                    succ_5 = float(np.mean([r["success_5cm"] for r in rows]))
                    print(
                        f"K={goal_k} | {head_name} + {planner_name}: "
                        f"mean 3D error {mean_err:.2f} cm, success<=5cm {100.0 * succ_5:.1f}%",
                        flush=True,
                    )
    finally:
        env.close()

    with (args.output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = summarize_episodes(all_rows, bootstrap_draws=args.bootstrap_draws, seed=int(cfg["seed"]))
    summary["config"] = cfg
    summary["groups"] = str(args.groups)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(all_rows)} episodes and summary to {args.output}")


if __name__ == "__main__":
    main()
