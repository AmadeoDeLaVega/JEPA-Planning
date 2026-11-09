#!/usr/bin/env python3
"""Score goal heads on a fixed PushT candidate bank (plan steps 6.2 and 6.3).

For each held-out target of a goal-group split (val = development, test =
held out), the task starts from the target's own initial state. The script runs
``candidates_per_task`` shared action chunks in the simulator and encodes the
actual endpoints. It also predicts the same endpoints with the frozen DINO-WM.
Every head (plain MSE, fitted heads, shuffled-label heads, random projections,
coordinate subsets) then picks the min-cost candidate for each of the task's two
goal photos. The photos show the same T pose with different held-out agents.

Metrics follow proposal Eq. 5 (see ``pusht_head_planning``): T position error
to the photographed target, regret against the best candidate in the bank,
success within 10/20/50 px, pose success and native (coverage) success. The
goal-photo intervention from the pilot is also reported: does the selected
candidate, or its native success, change between photo 0 and photo 1?

Writes to a new output directory:
  records.csv     one row per (source, head, task, photo)
  bank.npz        initial states, target poses, actions, endpoint states
  metadata.json   configs, checkpoint hash, timing, git revision

Usage (repository root):
  .venv/bin/python code/experiments/run_pusht_head_bank.py --split val \
    --runs results/pusht_linear_head_mac/rank* --output results/pusht_head_bank/val --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from jepa_planning import pusht_goal_groups as pg  # noqa: E402
from jepa_planning import pusht_head_planning as hp  # noqa: E402
from jepa_planning.pusht_multi_goal import sample_action_chunks, sample_initial_state  # noqa: E402


def make_environment():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


def load_tasks(groups_dir: Path, split: str, photos: int) -> dict:
    """Targets, their initial states and the indices of their first ``photos`` group photos."""
    group_cfg = json.loads((groups_dir / "metadata.json").read_text())["config"]
    images = np.load(groups_dir / "images.npz")
    targets = images["target_poses"]
    photo_index = np.full((len(targets), photos), -1)
    for index, (group, slot) in enumerate(zip(images["group"], images["slot"])):
        if slot < photos:
            photo_index[group, slot] = index
    keep = np.flatnonzero((photo_index >= 0).all(axis=1))
    initial = np.stack([sample_initial_state(pg.target_rng(group_cfg, split, int(i))) for i in keep])
    return {"group_cfg": group_cfg, "group": keep, "target": targets[keep], "initial": initial,
            "photo_index": photo_index[keep], "photo_object_pose": images["object_pose"][photo_index[keep]]}


def verify_targets(env, tasks: dict, split: str, action_mean, action_std) -> float:
    """Rebuild every target from its seed; return the largest pose difference."""
    worst = 0.0
    for task, group in enumerate(tasks["group"]):
        rng = pg.target_rng(tasks["group_cfg"], split, int(group))
        pose = pg.sample_target(env, rng, tasks["group_cfg"], action_mean, action_std)
        worst = max(worst, float(np.abs(pose - tasks["target"][task]).max()))
    return worst


def run_bank(env, tasks: dict, cfg: dict, split: str, action_mean, action_std) -> dict:
    """Execute the shared chunks; keep initial and endpoint observations and states."""
    out = {key: [] for key in ("initial_visual", "initial_proprio", "actions", "end_visual", "end_proprio",
                               "end_state")}
    for group, initial in zip(tasks["group"], tasks["initial"]):
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], pg.SPLIT_STREAM[split], int(group)]))
        chunks = sample_action_chunks(rng, cfg["candidates_per_task"], cfg["horizon"], cfg["controls_per_chunk"],
                                      cfg["action_clip"])
        env.reset_to_state = initial
        observation, _ = env.reset()
        out["initial_visual"].append(observation["visual"])
        out["initial_proprio"].append(observation["proprio"])
        out["actions"].append(chunks.reshape(len(chunks), cfg["horizon"], -1))
        for chunk in chunks:
            env.reset_to_state = initial
            env.reset()
            for normalized_action in chunk.reshape(-1, 2):
                observation, _, _, info = env.step(normalized_action * action_std + action_mean)
            out["end_visual"].append(observation["visual"])
            out["end_proprio"].append(observation["proprio"])
            out["end_state"].append(np.asarray(info["state"], dtype=np.float32))
    return {key: np.stack(value) for key, value in out.items()}


@torch.inference_mode()
def predict(model, bank: dict, batch_size: int) -> torch.Tensor:
    """Predicted endpoint features ``[tasks * candidates, P, D]`` (fp16, CPU)."""
    out = []
    for task in range(len(bank["initial_visual"])):
        visual = torch.from_numpy(bank["initial_visual"][task]).permute(2, 0, 1)[None, None]
        proprio = torch.from_numpy(bank["initial_proprio"][task]).float()[None, None]
        encoded = model.encode({"visual": visual, "proprio": proprio})
        actions = bank["actions"][task]
        for start in range(0, len(actions), batch_size):
            batch = torch.from_numpy(actions[start : start + batch_size]).permute(1, 0, 2).to(model.device)
            features = model.unroll(encoded, batch)["visual"][-1]
            out.append(features.reshape(features.shape[0], -1, features.shape[-1]).half().cpu())
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--groups", type=Path, help="goal-group directory (default results/pusht_goal_groups/<split>)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="train_linear_head.py outputs")
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_head_bank.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=int, help="use only the first N tasks (smoke runs)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    groups_dir = args.groups or ROOT / "results" / "pusht_goal_groups" / args.split
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new --output")

    from jepa_planning.model import load_world_model
    from encode_goal_groups import encode as encode_frames
    from scripts.download_checkpoint import sha256

    started = perf_counter()
    tasks = load_tasks(groups_dir, args.split, cfg["goal_photos"])
    if args.tasks:
        tasks = {key: value[: args.tasks] if key != "group_cfg" else value for key, value in tasks.items()}
    model = load_world_model("pusht", args.device).requires_grad_(False).eval()
    action_mean = model.preprocessor.action_mean.cpu().numpy()
    action_std = model.preprocessor.action_std.cpu().numpy()
    env = make_environment()
    try:
        target_error = verify_targets(env, tasks, args.split, action_mean, action_std)
        if target_error > 1e-3:
            raise RuntimeError(f"rebuilt targets differ from the stored ones by {target_error}")
        sim_started = perf_counter()
        bank = run_bank(env, tasks, cfg, args.split, action_mean, action_std)
        sim_seconds = perf_counter() - sim_started
    finally:
        env.close()

    model_started = perf_counter()
    sources = {"actual": encode_frames(model, bank["end_visual"], bank["end_proprio"], cfg["batch_size"]),
               "predicted": predict(model, bank, cfg["batch_size"])}
    model_seconds = perf_counter() - model_started
    goal_features = torch.load(groups_dir / "features.pt", map_location="cpu", weights_only=True)["groups"]
    goals = goal_features[torch.from_numpy(tasks["photo_index"].ravel())].reshape(
        *tasks["photo_index"].shape, *goal_features.shape[1:])

    count = cfg["candidates_per_task"]
    metrics = [hp.physical_metrics(bank["end_state"][t * count : (t + 1) * count, 2:5], tasks["target"][t])
               for t in range(len(tasks["group"]))]
    catalog = lh.head_catalog(args.runs)
    device = torch.device(args.device)
    rows = []
    for source, features in sources.items():
        features = features.to(device).float()
        for info, head in catalog:
            head = head.to(device)
            for task, group in enumerate(tasks["group"]):
                candidates = features[task * count : (task + 1) * count]
                for photo in range(cfg["goal_photos"]):
                    costs = hp.head_costs(candidates, goals[task, photo].to(device).float(), head).cpu().numpy()
                    rows.append({"source": source, **info, "task": task, "group": int(group), "photo": photo,
                                 **hp.selection_rows(costs, metrics[task])})
        print(f"{source}: scored {len(catalog)} heads", flush=True)

    args.output.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "bank.npz", group=tasks["group"], initial=tasks["initial"],
                        target=tasks["target"], photo_index=tasks["photo_index"], actions=bank["actions"],
                        end_state=bank["end_state"])
    checkpoint = ROOT / "checkpoints/dino_wm_pusht.pth.tar"
    metadata = {
        "command": sys.argv,
        "config": cfg,
        "split": args.split,
        "groups": str(groups_dir),
        "goal_group_config": tasks["group_cfg"],
        "runs": [str(run) for run in args.runs],
        "tasks": len(tasks["group"]),
        "heads": len(catalog),
        "max_rebuilt_target_error": target_error,
        "photo_target_matches": bool(np.allclose(tasks["photo_object_pose"], tasks["target"][:, None], atol=1e-4)),
        "simulation_seconds": sim_seconds,
        "model_seconds": model_seconds,
        "elapsed_seconds": perf_counter() - started,
        "device": args.device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ("tasks", "heads", "max_rebuilt_target_error",
                                                     "simulation_seconds", "model_seconds", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
