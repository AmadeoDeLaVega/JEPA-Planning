#!/usr/bin/env python3
"""Goal-photo intervention on Meta-World Push (proposal, Section 5).

How much does the goal photograph change the model's choice? Two goal photos
show the same puck pose with the hand in different places. Each candidate
action sequence is scored against both photos with the plain DINO-WM cost, and
we check whether the selected candidate changes.

Per task seed:
  1. Start state: reset Push and run the scripted expert for 40 steps.
  2. Goal: the puck pose after 30 more expert steps (6 model steps x 5).
     Photos L and R keep that puck pose and move the hand +/- 7 cm along a
     random horizontal direction.
  3. Candidates: 128 copies of the expert's actions plus Gaussian noise, with
     the noise scale going from 0 to 1.5 (normalised action units).
  4. Endpoints: run every candidate in the simulator (actual) and in the
     world model (predicted).
  5. Cost: mean squared difference of the visual features (no proprio term).

Puck error is the distance from the candidate's final puck position to the
puck position in the goal photos.

Usage:
  MUJOCO_GL=egl .venv/bin/python code/experiments/goal_photo_sensitivity.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from jepa_planning import ROOT, metaworld_push as mp  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402

WARMUP_STEPS = 40
HORIZON = 6  # world-model steps; each is FRAMESKIP env steps
NUM_CANDIDATES = 128
HAND_SEPARATION = 0.14  # metres between the hands in photos L and R
MAX_NOISE = 1.5
BATCH = 32


@torch.no_grad()
def encode_frames(model, frames: np.ndarray) -> torch.Tensor:
    """[B, H, W, 3] uint8 frames -> [B, 1, 16, 16, 384] features."""
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(1)
    return torch.cat([model.encode(visual[i : i + BATCH]) for i in range(0, len(visual), BATCH)])[:, -1]


@torch.no_grad()
def predict_endpoints(model, z_start, actions: torch.Tensor) -> torch.Tensor:
    """[HORIZON, B, 20] model actions -> predicted final features [B, 1, 16, 16, 384]."""
    actions = actions.to(model.device)
    return torch.cat(
        [model.unroll(z_start, actions[:, i : i + BATCH])["visual"][-1] for i in range(0, actions.shape[1], BATCH)]
    )


def cost(endpoints: torch.Tensor, goal: torch.Tensor) -> np.ndarray:
    return (endpoints - goal).pow(2).mean(dim=(1, 2, 3, 4)).cpu().numpy()


def goal_photos(env, goal_snap: dict, rng: np.random.Generator):
    """Photos L and R: same puck pose, hands about HAND_SEPARATION apart and off the puck."""
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        offset = HAND_SEPARATION / 2 * np.array([np.cos(angle), np.sin(angle), 0.0])
        photos, hands = [], []
        for sign in (1, -1):
            mp.restore(env, goal_snap)
            mp.move_hand_keep_puck(env, sign * offset)
            photos.append(env.render())
            hands.append(mp.hand_pos(env))
        puck_xy = mp.puck_pose(env)[:2]
        if all(np.linalg.norm(h[:2] - puck_xy) > 0.06 for h in hands):
            return photos, float(np.linalg.norm(hands[0] - hands[1]))


def run_task(env, model, seed: int, rng: np.random.Generator, photo_dir: Path) -> dict:
    pre = model.preprocessor
    env_steps = HORIZON * mp.FRAMESKIP

    # 1. Start state.
    obs = mp.reset_task(env, seed)
    obs, _ = mp.run_expert(env, obs, WARMUP_STEPS)
    start = mp.snapshot(env)
    z_start = model.encode(mp.observe(env))

    # 2. Goal puck position and the two goal photos.
    _, expert_actions = mp.run_expert(env, obs, env_steps)
    goal_puck = mp.puck_pose(env)[:3]
    photos, hand_separation = goal_photos(env, mp.snapshot(env), rng)
    Image.fromarray(np.concatenate(photos, axis=1)).save(photo_dir / f"seed{seed}_L_R.png")
    goals = encode_frames(model, np.stack(photos))

    # 3. Candidates as env actions [K, 30, 4] and as model actions [HORIZON, K, 20];
    # one model action is FRAMESKIP env actions concatenated.
    expert = pre.normalize_actions(torch.tensor(np.stack(expert_actions)))
    noise_scale = torch.linspace(0, MAX_NOISE, NUM_CANDIDATES)[:, None, None]
    noisy = expert + noise_scale * torch.randn(NUM_CANDIDATES, env_steps, 4)
    env_actions = pre.denormalize_actions(noisy).clamp(-1, 1)
    model_actions = pre.normalize_actions(env_actions).reshape(NUM_CANDIDATES, HORIZON, -1)
    model_actions = model_actions.permute(1, 0, 2).float()

    # 4. Actual endpoints (simulator) and predicted endpoints (world model).
    puck_end, success, frames = [], [], []
    for actions in env_actions.numpy():
        mp.restore(env, start)
        for action in actions:
            *_, info = env.step(action)
        puck_end.append(mp.puck_pose(env)[:3])
        success.append(info["success"])
        frames.append(env.render())
    error_cm = 100 * np.linalg.norm(np.stack(puck_end) - goal_puck, axis=1)
    endpoints = {
        "predicted": predict_endpoints(model, z_start, model_actions),
        "actual": encode_frames(model, np.stack(frames)),
    }

    # 5. Compare the candidates selected with photo L and with photo R.
    row = {"seed": seed, "hand_separation_cm": 100 * hand_separation}
    for name, features in endpoints.items():
        sel_l = int(np.argmin(cost(features, goals[0:1])))
        sel_r = int(np.argmin(cost(features, goals[1:2])))
        row |= {
            f"{name}_selection_changed": sel_l != sel_r,
            f"{name}_success_changed": bool(success[sel_l] != success[sel_r]),
            f"{name}_error_change_cm": abs(error_cm[sel_l] - error_cm[sel_r]),
            f"{name}_regret_cm": (error_cm[sel_l] + error_cm[sel_r]) / 2 - error_cm.min(),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-seed", type=int, default=58000)
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "goal_photo_sensitivity")
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    photo_dir = args.output / "goal_photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    model = load_world_model("metaworld", device=args.device)
    env = mp.make_push_env()

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.num_tasks):
        rows.append(run_task(env, model, seed, rng, photo_dir))
        print(rows[-1], flush=True)

    with (args.output / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_tasks": len(rows), "hand_separation_cm": np.mean([r["hand_separation_cm"] for r in rows])}
    for name in ("predicted", "actual"):
        summary[name] = {
            "selection_changed": sum(r[f"{name}_selection_changed"] for r in rows),
            "success_changed": sum(r[f"{name}_success_changed"] for r in rows),
            "mean_error_change_cm": np.mean([r[f"{name}_error_change_cm"] for r in rows]),
            "mean_regret_cm": np.mean([r[f"{name}_regret_cm"] for r in rows]),
        }
    text = json.dumps(summary, indent=2, default=float)
    (args.output / "summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
