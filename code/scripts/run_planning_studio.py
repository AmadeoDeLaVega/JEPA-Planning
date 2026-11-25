#!/usr/bin/env python3
"""Launch the Interactive Visual World-Model Planning Studio or export a duel GIF.

Usage (from repository root):
  # Start interactive web studio on localhost:8765
  .venv/bin/python code/scripts/run_planning_studio.py --device mps

  # Start with 3D Meta-World Sawyer Push-v3 enabled
  MUJOCO_GL=glfw .venv/bin/python code/scripts/run_planning_studio.py --device mps --enable-metaworld

  # Headless PushT demo GIF export (no browser needed)
  .venv/bin/python code/scripts/run_planning_studio.py --device mps --export-demo-gif demo_duel.gif

  # Headless 3D Meta-World Sawyer Push-v3 duel GIF export
  MUJOCO_GL=glfw .venv/bin/python code/scripts/run_planning_studio.py --device mps --export-metaworld-gif mw_duel.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning.studio import (  # noqa: E402
    PUSHT_PRESETS,
    MetaWorldStudioSession,
    PlanningStudioBackend,
    PushTStudioSession,
    create_studio_server,
    load_heads_directory,
)


def _preset_by_id(preset_id: str) -> dict[str, object]:
    for preset in PUSHT_PRESETS:
        if preset["id"] == preset_id:
            return preset
    available = [str(p["id"]) for p in PUSHT_PRESETS]
    raise ValueError(f"Unknown preset '{preset_id}'. Available: {available}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the planning studio launcher."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host address")
    parser.add_argument("--port", type=int, default=8765, help="Bind HTTP port")
    parser.add_argument("--device", default="mps", help="Torch device (mps, cuda, cpu)")
    parser.add_argument(
        "--heads-dir",
        type=Path,
        default=ROOT / "results" / "pusht_calibrated_heads",
        help="Directory containing saved PushT .pt linear heads",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "code" / "configs" / "pusht_head_closed_loop.yaml",
        help="PushT closed-loop planner YAML config",
    )
    parser.add_argument(
        "--enable-metaworld",
        action="store_true",
        help="Load the 3D Meta-World Sawyer Push-v3 world model and inspector/duel tab",
    )
    parser.add_argument(
        "--metaworld-heads-dir",
        type=Path,
        default=ROOT / "results" / "metaworld_calibrated_heads",
        help="Directory containing saved Meta-World .pt linear heads",
    )
    parser.add_argument(
        "--export-demo-gif",
        type=Path,
        default=None,
        help="Headless mode: run one preset PushT duel and write an animated GIF to PATH",
    )
    parser.add_argument(
        "--export-metaworld-gif",
        type=Path,
        default=None,
        help="Headless mode: run one 3D Meta-World Sawyer Push-v3 duel and write an animated GIF to PATH",
    )
    parser.add_argument(
        "--preset",
        default="corner_distractor",
        choices=[str(p["id"]) for p in PUSHT_PRESETS],
        help="Preset scenario used with --export-demo-gif",
    )
    parser.add_argument("--head", default=None, help="Head name override for headless GIF export")
    parser.add_argument("--replans", type=int, default=None, help="Replans override for headless GIF export")
    parser.add_argument("--fps", type=int, default=4, help="Frame rate for exported GIFs")
    parser.add_argument("--mw-seed", type=int, default=60000, help="Task seed for --export-metaworld-gif")
    parser.add_argument(
        "--mw-angle",
        type=float,
        default=45.0,
        help="Distractor gripper offset angle in degrees for --export-metaworld-gif",
    )
    parser.add_argument("--mw-horizon", type=int, default=2, help="Planning horizon for --export-metaworld-gif")
    parser.add_argument("--mw-samples", type=int, default=64, help="CEM samples per iteration for Meta-World")
    parser.add_argument("--mw-elites", type=int, default=8, help="CEM elites for Meta-World")
    parser.add_argument("--mw-batch-size", type=int, default=32, help="World-model unroll batch size for Meta-World")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load world models and heads, then export a duel GIF or serve the studio UI."""
    args = build_parser().parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    need_metaworld = bool(args.enable_metaworld or args.export_metaworld_gif is not None)
    if need_metaworld and sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    from jepa_planning.model import load_world_model

    if args.export_metaworld_gif is not None and args.export_demo_gif is None:
        print(f"Loading Meta-World DINO-WM on {args.device}...", flush=True)
        mw_model = load_world_model("metaworld", args.device).requires_grad_(False).eval()
        mw_heads, default_mw_head = load_heads_directory(args.metaworld_heads_dir)
        print(f"Loaded {len(mw_heads)} Meta-World head(s); default: {default_mw_head}", flush=True)
        mw_session = MetaWorldStudioSession(
            mw_model,
            mw_heads,
            cem_samples=int(args.mw_samples),
            cem_elites=int(args.mw_elites),
            batch_size=int(args.mw_batch_size),
            default_head=default_mw_head,
        )
        try:
            mw_replans = int(args.replans) if args.replans is not None else 6
            print(
                f"Running headless 3D Sawyer Push-v3 duel (seed={args.mw_seed}, "
                f"angle={args.mw_angle} deg, head={args.head or default_mw_head}, "
                f"horizon={args.mw_horizon}, replans={mw_replans})...",
                flush=True,
            )
            result = mw_session.run_duel(
                seed=int(args.mw_seed),
                hand_offset_angle_deg=float(args.mw_angle),
                head_name=args.head,
                horizon=int(args.mw_horizon),
                replans=mw_replans,
            )
            gif_bytes = mw_session.get_latest_duel_gif_bytes(fps=args.fps)
            args.export_metaworld_gif.parent.mkdir(parents=True, exist_ok=True)
            args.export_metaworld_gif.write_bytes(gif_bytes)
            base_err = result["baseline"]["summary"]["final_error_3d_cm"]
            trust_err = result["trust_mpc"]["summary"]["final_error_3d_cm"]
            print(
                f"Wrote {args.export_metaworld_gif} ({len(gif_bytes)} bytes) | "
                f"Plain MSE+CEM: {base_err:.2f} cm vs Trust-MPC: {trust_err:.2f} cm",
                flush=True,
            )
        finally:
            mw_session.close()
        return

    cfg = yaml.safe_load(args.config.read_text())

    print(f"Loading PushT DINO-WM on {args.device}...", flush=True)
    pusht_model = load_world_model("pusht", args.device).requires_grad_(False).eval()
    pusht_heads, default_pusht_head = load_heads_directory(args.heads_dir)
    print(f"Loaded {len(pusht_heads)} PushT head(s); default: {default_pusht_head}", flush=True)

    pusht_session = PushTStudioSession(
        pusht_model,
        pusht_heads,
        controls_per_chunk=int(cfg["controls_per_chunk"]),
        cem_horizon=int(cfg["cem_horizon"]),
        cem_samples=int(cfg["cem_samples"]),
        cem_elites=int(cfg["cem_elites"]),
        cem_iterations=int(cfg["cem_iterations"]),
        action_clip=float(cfg["action_clip"]),
        batch_size=int(cfg["batch_size"]),
        stay_margin=float(cfg.get("stay_margin", 0.02)),
        default_head=default_pusht_head,
    )

    mw_session: MetaWorldStudioSession | None = None
    if need_metaworld:
        print(f"Loading Meta-World DINO-WM on {args.device}...", flush=True)
        mw_model = load_world_model("metaworld", args.device).requires_grad_(False).eval()
        mw_heads, default_mw_head = load_heads_directory(args.metaworld_heads_dir)
        mw_session = MetaWorldStudioSession(
            mw_model,
            mw_heads,
            cem_samples=int(args.mw_samples),
            cem_elites=int(args.mw_elites),
            batch_size=int(args.mw_batch_size),
            default_head=default_mw_head,
        )
        print(f"Loaded {len(mw_heads)} Meta-World head(s); default: {default_mw_head}", flush=True)

    backend = PlanningStudioBackend(pusht_session, mw_session)

    try:
        if args.export_demo_gif is not None:
            preset = _preset_by_id(args.preset)
            replans = int(args.replans) if args.replans is not None else int(preset["replans"])
            print(
                f"Running headless duel for preset '{preset['id']}' "
                f"(head={args.head or default_pusht_head}, replans={replans})...",
                flush=True,
            )
            result = backend.pusht.run_duel(
                initial_agent=preset["initial_agent"],
                initial_block=preset["initial_block"],
                goal_agent=preset["goal_agent"],
                goal_block=preset["goal_block"],
                head_name=args.head,
                replans=replans,
                seed=int(preset["seed"]),
            )
            gif_bytes = backend.get_duel_gif_bytes(fps=args.fps)
            args.export_demo_gif.parent.mkdir(parents=True, exist_ok=True)
            args.export_demo_gif.write_bytes(gif_bytes)
            base_err = result["baseline"]["summary"]["final_position_error"]
            trust_err = result["trust_mpc"]["summary"]["final_position_error"]
            print(
                f"Wrote {args.export_demo_gif} ({len(gif_bytes)} bytes) | "
                f"Plain MSE+CEM: {base_err:.1f} px vs Trust-MPC: {trust_err:.1f} px",
                flush=True,
            )
            if args.export_metaworld_gif is None:
                return

        if args.export_metaworld_gif is not None and backend.metaworld is not None:
            mw_replans = int(args.replans) if args.replans is not None else 6
            result = backend.metaworld.run_duel(
                seed=int(args.mw_seed),
                hand_offset_angle_deg=float(args.mw_angle),
                head_name=args.head,
                horizon=int(args.mw_horizon),
                replans=mw_replans,
            )
            gif_bytes = backend.get_metaworld_duel_gif_bytes(fps=args.fps)
            args.export_metaworld_gif.parent.mkdir(parents=True, exist_ok=True)
            args.export_metaworld_gif.write_bytes(gif_bytes)
            base_err = result["baseline"]["summary"]["final_error_3d_cm"]
            trust_err = result["trust_mpc"]["summary"]["final_error_3d_cm"]
            print(
                f"Wrote {args.export_metaworld_gif} ({len(gif_bytes)} bytes) | "
                f"Plain MSE+CEM: {base_err:.2f} cm vs Trust-MPC: {trust_err:.2f} cm",
                flush=True,
            )
            return

        server = create_studio_server(args.host, args.port, backend)
        bound_host, bound_port = server.server_address
        print(f"Visual World-Model Planning Studio listening at http://{bound_host}:{bound_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down studio server.", flush=True)
        finally:
            server.server_close()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
