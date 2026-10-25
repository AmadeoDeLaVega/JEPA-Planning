"""Meta-World Push helpers: seeded tasks, simulator snapshots, and hand moves.

Rendering uses the upstream JEPA-WMs wrapper, so images match the ones the
released DINO-WM checkpoint was trained on (corner2 camera, 224x224).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from metaworld.policies import SawyerPushV3Policy
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.metaworld import make_env

# Env steps per world-model action (frameskip in the checkpoint's data config).
FRAMESKIP = 5
# The puck is a free joint: qpos[9:12] is its position, qpos[12:16] its rotation.
PUCK_QPOS = slice(9, 16)
# Full MuJoCo integration state, so replaying actions from a snapshot is deterministic.
SIM_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def make_push_env():
    cfg = OmegaConf.create(
        {
            "meta": {"seed": 0},
            "task_specification": {
                "task": "mw-push",
                "img_size": 224,
                "max_episode_steps": 500,
                "env": {"freeze_rand_vec": False},
            },
        }
    )
    # Skip upstream's TimeLimit wrapper; snapshots restore the step count instead.
    return make_env(cfg).env


def reset_task(env, seed: int) -> np.ndarray:
    """Reset to the puck start and target given by `seed`."""
    env.unwrapped.seeded_rand_vec = True
    env.unwrapped.seed(seed)
    obs, _ = env.reset()
    return obs


def run_expert(env, obs: np.ndarray, steps: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run Meta-World's scripted push policy; return the last obs and its actions."""
    policy = SawyerPushV3Policy()
    actions = []
    for _ in range(steps):
        action = np.clip(policy.get_action(obs), -1, 1).astype(np.float32)
        obs, *_ = env.step(action)
        actions.append(action)
    return obs, actions


def snapshot(env) -> dict:
    u = env.unwrapped
    sim = np.empty(mujoco.mj_stateSize(u.model, SIM_STATE))
    mujoco.mj_getState(u.model, u.data, sim, SIM_STATE)
    return {"sim": sim, "prev_obs": u._prev_obs.copy(), "steps": u.curr_path_length}


def restore(env, snap: dict) -> None:
    u = env.unwrapped
    mujoco.mj_setState(u.model, u.data, snap["sim"], SIM_STATE)
    mujoco.mj_forward(u.model, u.data)
    u._prev_obs = snap["prev_obs"].copy()
    u.curr_path_length = snap["steps"]


def puck_pose(env) -> np.ndarray:
    return env.unwrapped.data.qpos[PUCK_QPOS].copy()


def hand_pos(env) -> np.ndarray:
    """Gripper centre, the hand position in obs[:3]."""
    return env.unwrapped.tcp_center.copy()


def move_hand_keep_puck(env, offset: np.ndarray, steps: int = 50) -> None:
    """Move the hand by `offset`, then put the puck back in its exact pose."""
    u = env.unwrapped
    puck = puck_pose(env)
    target = u.data.mocap_pos[0] + offset
    for _ in range(steps):
        u.data.mocap_pos[0] = target
        u.data.mocap_quat[0] = [1, 0, 1, 0]
        u.do_simulation([-1, 1], u.frame_skip)
    qpos, qvel = u.data.qpos.copy(), u.data.qvel.copy()
    qpos[PUCK_QPOS] = puck
    qvel[9:15] = 0
    u.set_state(qpos, qvel)


def observe(env) -> dict[str, torch.Tensor]:
    """World-model input: visual [1, 1, 3, 224, 224] uint8 and proprio [1, 1, 4] (obs[:4])."""
    frame = env.render().copy()
    proprio = env.unwrapped._get_curr_obs_combined_no_goal()[:4].astype(np.float32)
    return {
        "visual": torch.from_numpy(frame).permute(2, 0, 1)[None, None],
        "proprio": torch.from_numpy(proprio)[None, None],
    }
