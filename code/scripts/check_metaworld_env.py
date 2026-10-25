#!/usr/bin/env python3
"""CPU smoke test for Meta-World Push: rendering and deterministic snapshot replay.

Rendering needs an OpenGL backend: MUJOCO_GL=egl on headless Linux (the
default here), MUJOCO_GL=glfw on macOS.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from jepa_planning import metaworld_push  # noqa: E402


def main() -> None:
    env = metaworld_push.make_push_env()
    metaworld_push.reset_task(env, seed=123)
    frame = env.render()
    assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8

    # The same actions from the same snapshot must reach the same state.
    start = metaworld_push.snapshot(env)
    actions = np.random.default_rng(0).uniform(-1, 1, (10, 4)).astype(np.float32)
    ends = []
    for _ in range(2):
        metaworld_push.restore(env, start)
        for action in actions:
            obs, *_ = env.step(action)
        ends.append(obs)
    assert np.array_equal(ends[0], ends[1]), "snapshot replay is not deterministic"
    print("Meta-World Push check: ok")


if __name__ == "__main__":
    main()
