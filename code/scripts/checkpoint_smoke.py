#!/usr/bin/env python3
"""Load the released DINO-WM PushT checkpoint and run one CPU inference.

This deliberately disables the optional RGB decoder head: encoding and latent
prediction use the released backbone/predictor weights, while avoiding a large
decoder download that is irrelevant to the feature-space experiments.
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
import yaml


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "jepa-wms"
# Keep runtime caches project-local as well as the Python environment. This
# avoids silently reading from or writing to a user's Conda/home environment.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
sys.path.insert(0, str(UPSTREAM))


def build_model(checkpoint: Path, device: str = "cpu"):
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from evals.simu_env_planning.eval import init_module

    config_path = (
        UPSTREAM
        / "configs/evals/simu_env_planning/pt/dino-wm/"
        / "pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
    )
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = cfg["model_kwargs"]
    data_cfg = model_cfg["data"]
    aug_cfg = model_cfg["data_aug"]
    pretrain_cfg = model_cfg["pretrain_kwargs"]

    # The decoder is used only for visualizations, not encoding or planning.
    pretrain_cfg["heads_cfg"] = {"architectures": {}}

    stats = get_data_stats("pusht")
    img_size = data_cfg["img_size"]
    transform = make_transforms(
        img_size=img_size,
        normalize=aug_cfg["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **aug_cfg)
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )
    model = init_module(
        folder=str(checkpoint.parent),
        checkpoint=checkpoint.name,
        module_name=model_cfg["module_name"],
        model_kwargs=pretrain_cfg,
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
        cfgs_data=data_cfg,
        device=torch.device(device),
        action_dim=stats["action_dim"],
        proprio_dim=stats["proprio_dim"],
        preprocessor=preprocessor,
    )
    return model


def render_frame() -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(7)
    env.reset_to_state = np.array([80.0, 300.0, 250.0, 250.0, 0.35, 0.0, 0.0])
    observation, _ = env.reset()
    frame = env.render(mode="rgb_array")
    proprio = observation["proprio"]
    env.close()
    return frame, proprio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "dino_wm_pusht.pth.tar",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    started = time.perf_counter()
    model = build_model(checkpoint, device=args.device)
    load_seconds = time.perf_counter() - started

    frame, proprio = render_frame()
    visual = torch.from_numpy(frame).permute(2, 0, 1)[None, None]
    observation = {
        "visual": visual,
        "proprio": torch.from_numpy(proprio)[None, None],
    }
    with torch.inference_mode():
        encoded = model.encode(observation)
        # One zero-action step verifies the released transition model too.
        # PushT has 2-D controls, grouped over the configured frameskip of 5.
        actions = torch.zeros(
            (1, 1, model.action_dim), dtype=torch.float32, device=model.device
        )
        predicted = model.unroll(encoded, actions)

    result = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
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
