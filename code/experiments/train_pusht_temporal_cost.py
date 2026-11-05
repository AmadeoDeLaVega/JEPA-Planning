#!/usr/bin/env python3
"""Train the temporal goal cost on recorded PushT trajectories.

Frames are encoded once with the frozen DINO-WM encoder, then pairs of cached
features train :class:`TemporalCost` to predict their separation in model steps.
Only the cost is trained; the world model stays frozen.

Run from the repository root after downloading ``pusht_noise`` into ``data/``::

    .venv/bin/python code/experiments/train_pusht_temporal_cost.py \
      --device cuda --output results/pusht_temporal_cost/seed_20261201

Use --smoke for a small pipeline check; it is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning import ROOT  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.temporal_cost import TemporalCost, sample_pairs  # noqa: E402

FEATURE_SIZE = 16 * 16 * 384
CONFIG_FIELDS = {
    "seed",
    "episodes",
    "frame_stride",
    "max_gap",
    "far_fraction",
    "width",
    "hidden",
    "batch_size",
    "train_steps",
    "learning_rate",
    "encode_batch_size",
}


def validate_config(config: dict) -> None:
    """Reject incomplete or invalid training configurations."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError(f"config fields must be {sorted(CONFIG_FIELDS)}")
    for field in CONFIG_FIELDS - {"seed", "far_fraction", "learning_rate"}:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit uint32")
    if not 0 <= config["far_fraction"] <= 1:
        raise ValueError("far_fraction must be in [0, 1]")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")


def strided_frames(length: int, stride: int) -> np.ndarray:
    """Return the frame indices kept from an episode of ``length`` frames."""
    if length < 1 or stride < 1:
        raise ValueError("length and stride must be positive")
    return np.arange(0, length, stride)


@torch.inference_mode()
def cache_features(model, split: Path, episodes: int, config: dict, path: Path) -> np.ndarray:
    """Encode strided frames to a float16 memmap and return per-episode frame counts."""
    import decord

    with (split / "seq_lengths.pkl").open("rb") as handle:
        lengths = list(pickle.load(handle))[:episodes]
    frames = [strided_frames(int(length), config["frame_stride"]) for length in lengths]
    counts = np.asarray([len(indices) for indices in frames])
    cache = np.memmap(path, dtype=np.float16, mode="w+", shape=(int(counts.sum()), FEATURE_SIZE))
    row = 0
    for episode, indices in enumerate(frames):
        reader = decord.VideoReader(str(split / "obses" / f"episode_{episode:03d}.mp4"), num_threads=1)
        video = torch.from_numpy(reader.get_batch(indices.tolist()).asnumpy())
        for start in range(0, len(video), config["encode_batch_size"]):
            batch = video[start : start + config["encode_batch_size"]].permute(0, 3, 1, 2)[:, None]
            features = model.encode(batch)[:, -1].flatten(start_dim=1).half().cpu().numpy()
            cache[row : row + len(features)] = features
            row += len(features)
    cache.flush()
    return counts


def pair_loss(cost: TemporalCost, cache: np.ndarray, pairs: tuple, device: torch.device) -> torch.Tensor:
    """Return the Huber regression loss for sampled start, goal, and label arrays."""
    starts, goals, labels = pairs
    candidates = torch.from_numpy(np.asarray(cache[starts])).to(device)
    targets = torch.from_numpy(np.asarray(cache[goals])).to(device)
    return functional.smooth_l1_loss(cost(candidates, targets), torch.from_numpy(labels).to(device))


def train(cost: TemporalCost, cache, counts, val_cache, val_counts, config: dict, device) -> list[dict]:
    """Fit the cost and return periodic training and validation losses."""
    rng = np.random.default_rng(config["seed"])
    optimizer = torch.optim.AdamW(cost.parameters(), lr=config["learning_rate"])
    sample = (config["batch_size"], config["max_gap"], config["far_fraction"], rng)
    history = []
    for step in range(1, config["train_steps"] + 1):
        loss = pair_loss(cost, cache, sample_pairs(counts, *sample), device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, config["train_steps"] // 20) == 0:
            with torch.no_grad():
                val_loss = pair_loss(cost, val_cache, sample_pairs(val_counts, *sample), device)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss.item()})
            print(json.dumps(history[-1]), flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_temporal_cost.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/pusht_noise")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pusht_temporal_cost")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.smoke:
        config.update(episodes=4, train_steps=20, batch_size=4)
    validate_config(config)
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")

    started = perf_counter()
    torch.manual_seed(config["seed"])
    args.output.mkdir(parents=True)
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    counts = cache_features(model, args.data / "train", config["episodes"], config, args.output / "train.cache")
    val_counts = cache_features(model, args.data / "val", config["episodes"], config, args.output / "val.cache")
    del model
    cache = np.memmap(args.output / "train.cache", dtype=np.float16, mode="r", shape=(int(counts.sum()), FEATURE_SIZE))
    val_cache = np.memmap(args.output / "val.cache", dtype=np.float16, mode="r", shape=(int(val_counts.sum()), FEATURE_SIZE))

    cost = TemporalCost(width=config["width"], hidden=config["hidden"]).to(device)
    history = train(cost, cache, counts, val_cache, val_counts, config, device)
    torch.save({"state_dict": cost.state_dict(), "config": config}, args.output / "temporal_cost.pt")
    for path in (args.output / "train.cache", args.output / "val.cache"):
        path.unlink()

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    metadata = {
        "config": config,
        "git_revision": revision,
        "git_dirty": dirty,
        "torch": torch.__version__,
        "command": sys.argv,
        "elapsed_seconds": perf_counter() - started,
        "history": history,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
