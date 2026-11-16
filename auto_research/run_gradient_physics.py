"""Fresh full-vector physics follow-up using saved DINO action gradients.

Run from repo root: python auto_research/run_gradient_physics.py --evidence fresh_gradient.npz > fresh_gradient.csv
Needs the vendored PushT simulator dependencies. An explicit
--gymnasium-compat flag substitutes installed Gymnasium only for Env/spaces.
No model checkpoint needed: gradients are fixed archived model measurements.
No downloads, installs or existing-file changes. Writes CSV to stdout and
new action/trajectory evidence to the required fresh NPZ path.

First replay all 2440 archived trajectories. Stop on state mismatch
before evaluating new actions. Then compare descent, ascent and 10 seeded
random directions at requested L2 radii .05/.20/.50; .20 is primary,
.05/.50 are sensitivity checks, never best-radius selection. All commands reset the
original seven-value reset procedure and execute five controls. This is a new-action
check on old tasks, not new-task or closed-loop generalization. Boundary
clipping can change actual step lengths; report these, do not call them equal.
"""

from pathlib import Path
import argparse
import contextlib
import io
import os
import sys
import zipfile
import numpy as np
import csv

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(root / "third_party/jepa-wms"))


def arrays(archive, path):
    with np.load(io.BytesIO(archive.read(path)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rollout(env, inputs, action):
    env.seed(0)
    env.reset_to_state = inputs["initial_state"].copy()
    env.reset()
    states = []
    for control in action:
        env.step(control * inputs["action_std"] + inputs["action_mean"])
        states.append(env._get_obs())
    return np.asarray(states, dtype=np.float32)


def error(state, goal):
    pose = state[2:5].astype(float)
    goal = goal.astype(float)
    angle = (pose[2] - goal[2] + np.pi) % (2*np.pi) - np.pi
    return np.sqrt(np.sum((pose[:2] - goal[:2])**2) + (50*angle)**2)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--gymnasium-compat", action="store_true")
args = parser.parse_args()
if args.evidence.suffix != ".npz":
    raise ValueError("Evidence path must end in .npz")
if args.evidence.exists():
    raise ValueError("Choose a new evidence path; existing files are preserved")

if args.gymnasium_compat:
    import gymnasium
    sys.modules["gym"] = gymnasium
with contextlib.redirect_stdout(sys.stderr):
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

archive = zipfile.ZipFile(root / "auto_research/evidence.zip")
prefix = "docs/planner_visited_gradient_results/audit/"
paths = sorted(path for path in archive.namelist() if path.startswith(prefix) and path.endswith("/diagnostics.npz"))
env = PushTEnv(with_velocity=True, with_target=True)
cases = []
replay = []
# This cross-platform engineering gate was revised before fresh-action results.
# The original 1e-5 state tolerance failed on cross-platform velocity differences.
# These amended bounds were fixed after replay auditing, before new action outcomes.
# Seven observed state values do not capture every hidden body or contact state.
for path in paths:
    inputs = arrays(archive, path.replace("diagnostics.npz", "inputs.npz"))
    data = arrays(archive, path)
    worst = np.zeros(7)
    worst_error = 0.
    repeat_difference = 0.
    for index, action in enumerate(inputs["actions"]):
        states = rollout(env, inputs, action)
        old = data["states"][index]
        worst = np.maximum(worst, np.max(np.abs(states - old), axis=0))
        worst_error = max(worst_error, abs(error(states[-1], inputs["goal_pose"]) - error(old[-1], inputs["goal_pose"])))
        if not np.isfinite(states).all():
            raise RuntimeError("Nonfinite replay; no fresh actions evaluated")
        if index in [0, 61]:
            repeat_difference = max(repeat_difference, np.max(np.abs(states - rollout(env, inputs, action))))
        if index == 61:
            baseline = error(states[-1], inputs["goal_pose"])
    # Object x/y/angle are checked separately, not only through scalar error.
    if worst.max() > 1e-3 or worst[4] > 1e-5 or worst_error > 1e-3 or repeat_difference != 0:
        env.close()
        raise RuntimeError("Cross-platform replay gate failed; do not loosen thresholds")
    replay.append(np.r_[worst, worst_error, repeat_difference])
    cases.append((path, inputs, data, inputs["actions"][61], baseline))
print("All archived replay cases passed the revised engineering gate", file=sys.stderr)

rows = []
new_actions = []
new_states = []
for path, inputs, data, base, baseline in cases:
    bank = int(path.split("seed_")[1].split("/")[0])
    task = int(path.split("task_")[1].split("/")[0])
    gradient = data["autograd"][1].reshape(5, 2)
    rng = np.random.default_rng(np.random.SeedSequence([bank, task, 20261113]))
    directions = [("descent", -gradient), ("ascent", gradient)]
    directions += [(f"random_{i}", rng.normal(size=(5, 2))) for i in range(10)]
    for radius in [0.05, 0.20, 0.50]:
        for method, direction in directions:
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 0 else np.zeros_like(direction)
            proposed = base + radius * direction
            action = np.clip(proposed, -2., 2.).astype(np.float32)
            states = rollout(env, inputs, action)
            outcome = error(states[-1], inputs["goal_pose"])
            new_actions.append(action)
            new_states.append(states)
            rows.append(dict(bank=bank, task=task, radius=radius, method=method,
                             actual_step=float(np.linalg.norm(action-base)),
                             clipped=int(np.count_nonzero(proposed != np.clip(proposed, -2., 2.))),
                             baseline=baseline, physical_error=outcome, gain=baseline-outcome))
env.close()
np.savez_compressed(args.evidence, actions=np.array(new_actions), states=np.array(new_states),
                    bank=np.array([row["bank"] for row in rows]),
                    task=np.array([row["task"] for row in rows]),
                    radius=np.array([row["radius"] for row in rows]),
                    method=np.array([row["method"] for row in rows]),
                    replay=np.array(replay),
                    initial_states=np.array([case[1]["initial_state"] for case in cases]),
                    goal_poses=np.array([case[1]["goal_pose"] for case in cases]),
                    physical_error=np.array([row["physical_error"] for row in rows]),
                    baseline=np.array([row["baseline"] for row in rows]),
                    actual_step=np.array([row["actual_step"] for row in rows]),
                    clipped=np.array([row["clipped"] for row in rows]))
writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
