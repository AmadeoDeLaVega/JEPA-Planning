#!/usr/bin/env python3
"""CPU smoke test for the narrowed JEPA-WMs Push-T environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
cache_dir = Path(__file__).resolve().parents[2] / ".tools" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import decord  # noqa: E402
import numpy as np  # noqa: E402
import pymunk  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402


def main() -> None:
    env = PushTEnv(render_size=64, with_velocity=False, with_target=True)
    env.seed(123)
    observation, state = env.reset()
    frame = observation["visual"]
    next_observation, reward, done, info = env.step(np.array([0.05, -0.05]))
    result = {
        "python_smoke": "ok",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_on_current_node": torch.cuda.is_available(),
        "decord": decord.__version__,
        "pymunk": getattr(pymunk, "version", "unknown"),
        "initial_state_shape": list(state.shape),
        "frame_shape": list(frame.shape),
        "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "step_frame_shape": list(next_observation["visual"].shape),
        "step_reward": float(reward),
        "step_done": bool(done),
        "step_state_shape": list(info["state"].shape),
    }
    assert frame.shape == (64, 64, 3)
    assert next_observation["visual"].shape == (64, 64, 3)
    assert np.isfinite(frame).all()
    assert np.isfinite(next_observation["visual"]).all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
