#!/usr/bin/env python3
"""Load a released DINO-WM checkpoint and run one CPU encode + predict step.

Checks that the encoder, action-conditioned predictor, and proprioception
encoder load and produce [B, T, 1, 16, 16, 384] patch features.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa_planning.model import load_world_model  # noqa: E402


def pusht_observation() -> dict[str, torch.Tensor]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(7)
    env.reset_to_state = np.array([80.0, 300.0, 250.0, 250.0, 0.35, 0.0, 0.0])
    observation, _ = env.reset()
    frame = env.render(mode="rgb_array")
    env.close()
    return {
        "visual": torch.from_numpy(frame).permute(2, 0, 1)[None, None],
        "proprio": torch.from_numpy(observation["proprio"])[None, None],
    }


def metaworld_observation() -> dict[str, torch.Tensor]:
    from jepa_planning import metaworld_push

    env = metaworld_push.make_push_env()
    metaworld_push.reset_task(env, seed=7)
    return metaworld_push.observe(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["pusht", "metaworld"], default="pusht")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    model = load_world_model(args.env, device=args.device)
    load_seconds = time.perf_counter() - started

    observation = pusht_observation() if args.env == "pusht" else metaworld_observation()
    with torch.inference_mode():
        encoded = model.encode(observation)
        # One zero-action step checks the predictor too. A model action is
        # 5 env actions (the frameskip) concatenated.
        actions = torch.zeros((1, 1, model.action_dim), dtype=torch.float32, device=model.device)
        predicted = model.unroll(encoded, actions)

    result = {
        "env": args.env,
        "device": str(model.device),
        "load_seconds": round(load_seconds, 3),
        "encoded_shape": list(encoded["visual"].shape),
        "predicted_shape": list(predicted["visual"].shape),
        "encoded_mean": float(encoded["visual"].mean()),
        "encoded_std": float(encoded["visual"].std()),
        "predicted_last_mean": float(predicted["visual"][-1].mean()),
        "predicted_last_std": float(predicted["visual"][-1].std()),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
