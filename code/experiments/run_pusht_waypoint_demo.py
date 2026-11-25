#!/usr/bin/env python3
"""Paired 2-stage PushT waypoint experiment: Plain MSE CEM vs. Calibrated Trust-MPC.

Evaluates purely visual subgoal chaining across seeded 2-stage PushT maneuvers
(Stage 1 intermediate pose -> Stage 2 final target pose). Each subgoal photo is
rendered with an independent distractor blue-pusher position so goal photos
never leak pusher hints. Both planners share identical start states, subgoal
photos, random CEM normal streams, and exact ``2 * N`` unroll budgets per
replan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import pusht_head_planning as hp  # noqa: E402
from jepa_planning import trust_mpc as tm  # noqa: E402
from jepa_planning.pusht_multi_goal import cem_standard_normals, state_from  # noqa: E402
from jepa_planning.studio import standard_cem_plan  # noqa: E402
from jepa_planning.waypoint_mpc import (  # noqa: E402
    WaypointTracker,
    compose_waypoint_gif_bytes,
    encode_visual_features,
    waypoint_trust_mpc_step,
)


def make_environment():
    """Construct the headless PushT environment matching the frozen DINO-WM checkpoint."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


def _sample_distractor_pusher(
    rng: np.random.Generator,
    block_xy: np.ndarray,
    avoid_points: list[np.ndarray],
    *,
    min_block_dist: float = 115.0,
    min_avoid_dist: float = 110.0,
) -> np.ndarray:
    """Sample a blue-pusher distractor position well separated from the T-block and reference points."""
    best_candidate = np.array([75.0, 75.0], dtype=np.float32)
    best_clearance = -1.0
    for _ in range(128):
        cand = rng.uniform(58.0, 454.0, size=2).astype(np.float64)
        block_dist = float(np.linalg.norm(cand - block_xy))
        if block_dist < min_block_dist:
            continue
        avoid_min = min(
            (float(np.linalg.norm(cand - pt)) for pt in avoid_points),
            default=float("inf"),
        )
        if avoid_min >= min_avoid_dist:
            return cand.astype(np.float32)
        score = min(block_dist, avoid_min)
        if score > best_clearance:
            best_clearance = score
            best_candidate = cand.astype(np.float32)
    return best_candidate


def sample_two_stage_pusht_tasks(num_tasks: int, seed: int = 20261124) -> list[dict[str, object]]:
    """Generate deterministic 2-stage around-the-corner PushT tasks with independent distractors.

    Each task specifies:
    - ``initial_agent`` and ``initial_block``: start state
    - ``stage1_block`` and ``stage1_goal_agent``: intermediate subgoal photo
    - ``stage2_block`` and ``stage2_goal_agent``: final target subgoal photo
    """
    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks < 1:
        raise ValueError("num_tasks must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    tasks: list[dict[str, object]] = []

    for task_id in range(num_tasks):
        # Place Stage 1 near the central arena so both legs stay comfortably inside bounds.
        s1_xy = rng.uniform(210.0, 302.0, size=2)
        heading1 = float(rng.uniform(0.0, 2.0 * math.pi))
        leg1_len = float(rng.uniform(48.0, 68.0))
        init_xy = np.clip(
            s1_xy - leg1_len * np.array([math.cos(heading1), math.sin(heading1)]),
            120.0,
            392.0,
        )

        # Turn 60..115 degrees (left or right) at Stage 1 for a multi-stage corner maneuver.
        turn_sign = -1.0 if (task_id % 2 == 0) else 1.0
        turn_angle = turn_sign * float(rng.uniform(math.radians(60.0), math.radians(115.0)))
        heading2 = heading1 + turn_angle
        leg2_len = float(rng.uniform(48.0, 68.0))
        s2_xy = np.clip(
            s1_xy + leg2_len * np.array([math.cos(heading2), math.sin(heading2)]),
            120.0,
            392.0,
        )

        init_theta = float(rng.uniform(0.0, 2.0 * math.pi))
        s1_theta = float((init_theta + turn_sign * rng.uniform(0.20, 0.50)) % (2.0 * math.pi))
        s2_theta = float((s1_theta + turn_sign * rng.uniform(0.25, 0.60)) % (2.0 * math.pi))

        # Start pusher behind the T-block along leg 1 so Stage 1 is directly reachable.
        pusher_offset = float(rng.uniform(86.0, 102.0))
        init_agent = np.clip(
            init_xy - pusher_offset * np.array([math.cos(heading1), math.sin(heading1)]),
            58.0,
            454.0,
        ).astype(np.float32)

        stage1_distractor = _sample_distractor_pusher(
            rng, s1_xy, [init_agent.astype(np.float64), init_xy]
        )
        stage2_distractor = _sample_distractor_pusher(
            rng, s2_xy, [init_agent.astype(np.float64), stage1_distractor.astype(np.float64), s1_xy]
        )

        tasks.append(
            {
                "task_id": task_id,
                "initial_agent": init_agent.tolist(),
                "initial_block": [float(init_xy[0]), float(init_xy[1]), init_theta],
                "stage1_block": [float(s1_xy[0]), float(s1_xy[1]), s1_theta],
                "stage1_goal_agent": stage1_distractor.tolist(),
                "stage2_block": [float(s2_xy[0]), float(s2_xy[1]), s2_theta],
                "stage2_goal_agent": stage2_distractor.tolist(),
            }
        )

    return tasks


def _render_pose(env, agent_xy: list[float] | np.ndarray, block_pose: list[float] | np.ndarray) -> dict:
    env.reset_to_state = state_from(
        np.asarray(agent_xy, dtype=np.float32),
        np.asarray(block_pose, dtype=np.float32),
    )
    obs, _ = env.reset()
    return {
        "visual": np.asarray(obs["visual"], dtype=np.uint8).copy(),
        "proprio": np.asarray(obs["proprio"], dtype=np.float32).copy(),
    }


def run_waypoint_episode(
    env,
    model,
    head: torch.Tensor,
    planner_name: str,
    task: dict[str, object],
    waypoint_features: list[torch.Tensor],
    shared_normals: list[np.ndarray],
    *,
    controls_per_chunk: int = 5,
    cem_elites: int = 8,
    action_clip: float = 2.0,
    batch_size: int = 64,
    stay_margin: float = 0.02,
    arrival_ratio: float = 0.35,
    allow_stay_gate_advance: bool = True,
    stage1_threshold_px: float = 35.0,
) -> tuple[dict[str, object], list[np.ndarray], list[int]]:
    """Run one 2-stage visual waypoint episode under either standard CEM or certified Trust-MPC."""
    if planner_name not in ("standard_cem", "waypoint_trust_mpc"):
        raise ValueError(f"Unsupported planner_name: {planner_name}")

    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is not None and hasattr(preprocessor, "action_mean") and hasattr(preprocessor, "action_std"):
        action_mean = np.asarray(preprocessor.action_mean.detach().cpu().numpy(), dtype=np.float32)
        action_std = np.asarray(preprocessor.action_std.detach().cpu().numpy(), dtype=np.float32)
    else:
        action_mean = np.zeros(2, dtype=np.float32)
        action_std = np.ones(2, dtype=np.float32)

    hold_chunk = tm.zero_displacement_chunk(action_mean, action_std, controls_per_chunk, action_clip)
    head_device = head.to(device=model.device, dtype=torch.float32)

    tracker = WaypointTracker(
        waypoint_features,
        head_device,
        arrival_ratio=arrival_ratio,
        allow_stay_gate_advance=bool(allow_stay_gate_advance and planner_name == "waypoint_trust_mpc"),
    )

    initial_state = state_from(
        np.asarray(task["initial_agent"], dtype=np.float32),
        np.asarray(task["initial_block"], dtype=np.float32),
    )
    stage1_target = np.asarray(task["stage1_block"], dtype=np.float64)
    stage2_target = np.asarray(task["stage2_block"], dtype=np.float64)

    env.reset_to_state = initial_state.copy()
    observation, _ = env.reset()

    frames: list[np.ndarray] = [np.asarray(observation["visual"], dtype=np.uint8).copy()]
    frame_stages: list[int] = [0]

    init_s1 = float(hp.physical_metrics(initial_state[None, 2:5], stage1_target)["position_error"][0])
    init_s2 = float(hp.physical_metrics(initial_state[None, 2:5], stage2_target)["position_error"][0])
    best_stage1_error = init_s1
    best_stage2_error = init_s2

    cur_block_pose = initial_state[2:5].astype(np.float64)
    handoff_step: int | None = None
    handoff_stage1_error: float | None = None
    handoff_trigger: str | None = None

    previous_plan: np.ndarray | None = None
    total_unrolls = 0
    sim_steps = 0
    jensen_violations = 0
    stay_gate_triggers = 0
    warm_start_resets = 0
    started = perf_counter()

    for replan_idx, normals in enumerate(shared_normals):
        if planner_name == "standard_cem":
            current_features = encode_visual_features(model, observation)
            t_info = tracker.update(current_features, used_stay_gate=False)
            action_chunk, p_info = standard_cem_plan(
                model,
                observation,
                tracker.active_waypoint,
                head_device,
                normals,
                num_elites=cem_elites,
                action_clip=action_clip,
                batch_size=batch_size,
            )
            step_diag = {
                **p_info,
                **t_info,
                "jensen_violation": False,
                "used_stay_gate": False,
                "warm_start_reset": bool(t_info["advanced"]),
            }
        else:
            action_chunk, previous_plan, step_diag = waypoint_trust_mpc_step(
                model,
                observation,
                tracker,
                normals,
                hold_chunk,
                previous_plan,
                num_elites=cem_elites,
                action_clip=action_clip,
                batch_size=batch_size,
                stay_margin=stay_margin,
                certify_barrier_split=True,
            )

        if step_diag["advanced"] and handoff_step is None:
            handoff_step = replan_idx
            handoff_stage1_error = float(
                hp.physical_metrics(cur_block_pose[None, :], stage1_target)["position_error"][0]
            )
            handoff_trigger = str(step_diag["trigger"])

        total_unrolls += int(step_diag["unrolls"])
        if step_diag.get("jensen_violation"):
            jensen_violations += 1
        if step_diag.get("used_stay_gate"):
            stay_gate_triggers += 1
        if step_diag.get("warm_start_reset"):
            warm_start_resets += 1

        active_stage = int(tracker.stage_index)
        for control in action_chunk.reshape(controls_per_chunk, 2):
            raw_action = control * action_std + action_mean
            observation, _, _, step_info = env.step(raw_action)
            sim_steps += 1
            frames.append(np.asarray(observation["visual"], dtype=np.uint8).copy())
            frame_stages.append(active_stage)

        cur_block_pose = np.asarray(step_info["state"], dtype=np.float64)[2:5]
        s1_err = float(hp.physical_metrics(cur_block_pose[None, :], stage1_target)["position_error"][0])
        s2_metrics = {
            k: float(v[0]) for k, v in hp.physical_metrics(cur_block_pose[None, :], stage2_target).items()
        }
        best_stage1_error = min(best_stage1_error, s1_err)
        best_stage2_error = min(best_stage2_error, s2_metrics["position_error"])

    final_s2_metrics = {
        k: float(v[0]) for k, v in hp.physical_metrics(cur_block_pose[None, :], stage2_target).items()
    }
    stage1_completed = bool(best_stage1_error <= stage1_threshold_px)
    handoff_triggered = bool(handoff_step is not None)
    handoff_accurate = bool(
        handoff_triggered
        and handoff_stage1_error is not None
        and handoff_stage1_error <= stage1_threshold_px
    )

    row: dict[str, object] = {
        "task_id": int(task["task_id"]),
        "planner": planner_name,
        "initial_stage1_error_px": init_s1,
        "initial_stage2_error_px": init_s2,
        "best_stage1_error_px": best_stage1_error,
        "stage1_completed": int(stage1_completed),
        "handoff_triggered": int(handoff_triggered),
        "handoff_step": -1 if handoff_step is None else int(handoff_step),
        "handoff_trigger": handoff_trigger or "none",
        "handoff_stage1_error_px": (
            float("nan") if handoff_stage1_error is None else float(handoff_stage1_error)
        ),
        "handoff_accurate": int(handoff_accurate),
        "final_stage2_error_px": final_s2_metrics["position_error"],
        "best_stage2_error_px": best_stage2_error,
        "final_stage2_angle_deg": final_s2_metrics["angle_error_deg"],
        "final_stage2_combined_error": final_s2_metrics["combined_error"],
        "final_stage2_coverage": final_s2_metrics["coverage"],
        "stage2_success_20px": int(final_s2_metrics["success_20px"]),
        "final_stage_index": int(tracker.stage_index),
        "jensen_violations": int(jensen_violations),
        "stay_gate_triggers": int(stay_gate_triggers),
        "warm_start_resets": int(warm_start_resets),
        "unrolls": int(total_unrolls),
        "sim_steps": int(sim_steps),
        "planning_seconds": perf_counter() - started,
    }
    return row, frames, frame_stages


def summarize_waypoint_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate Stage-1 completion, handoff accuracy, and final Stage-2 error per label."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        label = str(row.get("method", row["planner"]))
        grouped.setdefault(label, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for label, group in grouped.items():
        summary[label] = {
            "episodes": float(len(group)),
            "stage1_completion_rate": float(np.mean([r["stage1_completed"] for r in group])),
            "handoff_rate": float(np.mean([r["handoff_triggered"] for r in group])),
            "handoff_accuracy": float(np.mean([r["handoff_accurate"] for r in group])),
            "mean_final_stage2_error_px": float(np.mean([r["final_stage2_error_px"] for r in group])),
            "mean_best_stage2_error_px": float(np.mean([r["best_stage2_error_px"] for r in group])),
            "stage2_success_20px_rate": float(np.mean([r["stage2_success_20px"] for r in group])),
            "mean_unrolls": float(np.mean([r["unrolls"] for r in group])),
        }
    return summary


def run_paired_waypoint_suite(
    env,
    model,
    calibrated_head: torch.Tensor,
    *,
    calibrated_head_name: str = "calibrated_head",
    num_tasks: int = 8,
    replans: int = 8,
    seed: int = 20261124,
    arrival_ratio: float = 0.35,
    stage1_threshold_px: float = 35.0,
    cfg: dict | None = None,
    gif_task_index: int = 0,
    export_gif_path: Path | None = None,
    gif_fps: int = 4,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    """Run paired 2-stage PushT waypoint trials comparing Plain MSE CEM vs. Calibrated Trust-MPC."""
    config = {
        "controls_per_chunk": 5,
        "cem_horizon": 2,
        "cem_samples": 64,
        "cem_elites": 8,
        "cem_iterations": 2,
        "batch_size": 64,
        "action_clip": 2.0,
        "stay_margin": 0.02,
    }
    if cfg:
        config.update(cfg)

    tasks = sample_two_stage_pusht_tasks(num_tasks=num_tasks, seed=seed)
    feature_dim = int(calibrated_head.shape[1])
    identity_head = lh.identity_head(feature_dim)
    action_dim = int(getattr(model, "action_dim", config["controls_per_chunk"] * 2))

    rows: list[dict[str, object]] = []
    gif_payload: tuple[list[np.ndarray], list[np.ndarray], list[int]] | None = None

    for task in tasks:
        task_id = int(task["task_id"])
        obs_s1 = _render_pose(env, task["stage1_goal_agent"], task["stage1_block"])
        obs_s2 = _render_pose(env, task["stage2_goal_agent"], task["stage2_block"])

        waypoint_features = [
            encode_visual_features(model, obs_s1).cpu(),
            encode_visual_features(model, obs_s2).cpu(),
        ]
        waypoint_frames = [obs_s1["visual"], obs_s2["visual"]]

        shared_normals = [
            cem_standard_normals(
                seed,
                task_id,
                replan_idx,
                int(config["cem_iterations"]),
                int(config["cem_samples"]),
                int(config["cem_horizon"]),
                action_dim,
            )
            for replan_idx in range(replans)
        ]

        mse_row, _, _ = run_waypoint_episode(
            env,
            model,
            identity_head,
            "standard_cem",
            task,
            waypoint_features,
            shared_normals,
            controls_per_chunk=int(config["controls_per_chunk"]),
            cem_elites=int(config["cem_elites"]),
            action_clip=float(config["action_clip"]),
            batch_size=int(config["batch_size"]),
            stay_margin=float(config["stay_margin"]),
            arrival_ratio=arrival_ratio,
            allow_stay_gate_advance=False,
            stage1_threshold_px=stage1_threshold_px,
        )
        mse_row["head"] = "identity"
        mse_row["method"] = "plain_mse_waypoint_cem"
        rows.append(mse_row)

        trust_row, trust_frames, trust_stages = run_waypoint_episode(
            env,
            model,
            calibrated_head,
            "waypoint_trust_mpc",
            task,
            waypoint_features,
            shared_normals,
            controls_per_chunk=int(config["controls_per_chunk"]),
            cem_elites=int(config["cem_elites"]),
            action_clip=float(config["action_clip"]),
            batch_size=int(config["batch_size"]),
            stay_margin=float(config["stay_margin"]),
            arrival_ratio=arrival_ratio,
            allow_stay_gate_advance=True,
            stage1_threshold_px=stage1_threshold_px,
        )
        trust_row["head"] = calibrated_head_name
        trust_row["method"] = "calibrated_waypoint_trust_mpc"
        rows.append(trust_row)

        if export_gif_path is not None and task_id == gif_task_index:
            gif_payload = (trust_frames, waypoint_frames, trust_stages)

    if export_gif_path is not None and gif_payload is not None:
        export_gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_bytes = compose_waypoint_gif_bytes(
            gif_payload[0],
            gif_payload[1],
            gif_payload[2],
            title_prefix=f"Trust-MPC ({calibrated_head_name})",
            fps=gif_fps,
        )
        export_gif_path.write_bytes(gif_bytes)

    return rows, summarize_waypoint_rows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the 2-stage PushT waypoint demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-file",
        type=Path,
        required=True,
        help="Path to a saved calibrated linear head (.pt)",
    )
    parser.add_argument("--tasks", type=int, default=8, help="Number of seeded 2-stage PushT tasks")
    parser.add_argument("--replans", type=int, default=8, help="Closed-loop replanning steps per episode")
    parser.add_argument("--seed", type=int, default=20261124, help="Random seed for tasks and CEM streams")
    parser.add_argument(
        "--arrival-ratio",
        type=float,
        default=0.35,
        help="Scale-invariant latent arrival threshold D_L(z_t, G_s) <= ratio * D_L(z_entry, G_s)",
    )
    parser.add_argument(
        "--stage1-threshold-px",
        type=float,
        default=35.0,
        help="Physical Stage-1 T-block tolerance (px) for completion and handoff accuracy",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "code/configs/pusht_head_closed_loop.yaml",
        help="Base PushT planner YAML config",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for frozen DINO-WM (default: cpu)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory for CSV and JSON")
    parser.add_argument(
        "--export-gif",
        type=Path,
        default=None,
        help="Optional path to write a [Active Subgoal | Live Rollout] animated GIF",
    )
    parser.add_argument("--gif-task", type=int, default=0, help="Task index to render when --export-gif is set")
    parser.add_argument("--gif-fps", type=int, default=4, help="Frames per second for exported GIF")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text()) if args.config.is_file() else {}
    calibrated_head, _ = lh.load_head(args.head_file)

    from jepa_planning.model import load_world_model

    model = load_world_model("pusht", args.device).requires_grad_(False).eval()
    env = make_environment()
    try:
        rows, summary = run_paired_waypoint_suite(
            env,
            model,
            calibrated_head,
            calibrated_head_name=args.head_file.stem,
            num_tasks=args.tasks,
            replans=args.replans,
            seed=args.seed,
            arrival_ratio=args.arrival_ratio,
            stage1_threshold_px=args.stage1_threshold_px,
            cfg=cfg,
            gif_task_index=args.gif_task,
            export_gif_path=args.export_gif,
            gif_fps=args.gif_fps,
        )
    finally:
        env.close()

    for label, stats in summary.items():
        print(
            f"{label}: "
            f"Stage1={100.0 * stats['stage1_completion_rate']:.1f}% | "
            f"HandoffAcc={100.0 * stats['handoff_accuracy']:.1f}% "
            f"(triggered {100.0 * stats['handoff_rate']:.1f}%) | "
            f"FinalStage2={stats['mean_final_stage2_error_px']:.2f} px | "
            f"Unrolls={stats['mean_unrolls']:.0f}",
            flush=True,
        )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
