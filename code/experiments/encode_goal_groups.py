#!/usr/bin/env python3
"""Encode rendered goal groups with the frozen DINO-WM encoder (plan step 2).

Reads images.npz (and sensitivity.npz if present) from a directory written by
make_pusht_goal_groups.py and adds features.pt: fp16 [n, P, D] under "groups"
and, if present, "sensitivity". Also adds encode.json with timing and the
checkpoint hash. Refuses to overwrite an existing features.pt.

Usage (repository root):
  .venv/bin/python code/experiments/encode_goal_groups.py results/pusht_goal_groups/val --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402


@torch.inference_mode()
def encode(model, visual: np.ndarray, proprio: np.ndarray, batch_size: int) -> torch.Tensor:
    """Frames [n, H, W, 3] uint8 -> fp16 CPU features [n, P, D] (last encoded time step)."""
    out = []
    for start in range(0, len(visual), batch_size):
        frames = torch.from_numpy(visual[start : start + batch_size]).permute(0, 3, 1, 2)[:, None]
        state = torch.from_numpy(proprio[start : start + batch_size]).float()[:, None]
        features = model.encode({"visual": frames, "proprio": state})["visual"][:, -1]
        out.append(features.reshape(len(frames), -1, features.shape[-1]).half().cpu())
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--env", default="pusht", choices=["pusht", "metaworld"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    target = args.directory / "features.pt"
    if target.exists():
        raise FileExistsError(f"{target} exists")

    from jepa_planning.model import load_world_model
    from scripts.download_checkpoint import sha256

    started = perf_counter()
    model = load_world_model(args.env, args.device).requires_grad_(False).eval()
    payload = {}
    for key, name in [("groups", "images.npz"), ("sensitivity", "sensitivity.npz")]:
        if (args.directory / name).exists():
            data = np.load(args.directory / name)
            payload[key] = encode(model, data["visual"], data["proprio"].astype(np.float32), args.batch_size)
            print(f"{key}: {tuple(payload[key].shape)}", flush=True)
    torch.save(payload, target)
    checkpoint = ROOT / "checkpoints" / f"dino_wm_{args.env}.pth.tar"
    info = {"device": args.device, "seconds": perf_counter() - started, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "shapes": {k: list(v.shape) for k, v in payload.items()}}
    (args.directory / "encode.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
