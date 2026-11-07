#!/usr/bin/env python3
"""Train the temporal goal cost on recorded PushT trajectories.

Frames are encoded once with the frozen DINO-WM encoder, then pairs of cached
features train :class:`TemporalCost` to predict their separation in model steps.
A second loss, weighted by ``ordering_weight``, asks the cost to score a frame
at least one step lower than the frame before it, for the same goal. Planning
compares candidates that are about one step apart, and the regression alone was
imprecise at that scale. Only the cost is trained, the world model stays frozen.

The weights saved are those with the best score on a fixed sample of validation
pairs, which comes from separate scenes.

Run from the repository root after downloading ``pusht_noise`` into ``data/``::

    .venv/bin/python code/experiments/train_pusht_temporal_cost.py \
      --device cuda --output results/pusht_temporal_cost/seed_20261201

Use --smoke for a small pipeline check; it is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
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
from jepa_planning.temporal_cost import TemporalCost, one_step_closer, sample_pairs  # noqa: E402

FEATURE_SIZE = 16 * 16 * 384
VALIDATION_PAIRS = 4096
CONFIG_FIELDS = {
    "seed",
    "episodes",
    "frame_stride",
    "max_gap",
    "far_fraction",
    "ordering_weight",
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
    for field in CONFIG_FIELDS - {"seed", "far_fraction", "ordering_weight", "learning_rate"}:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit uint32")
    if not 0 <= config["far_fraction"] <= 1:
        raise ValueError("far_fraction must be in [0, 1]")
    if not math.isfinite(config["ordering_weight"]) or config["ordering_weight"] < 0:
        raise ValueError("ordering_weight must be finite and non-negative")
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


def losses(cost: TemporalCost, cache: np.ndarray, counts: np.ndarray, pairs: tuple, device: torch.device) -> tuple:
    """Return the regression loss, the one-step ordering loss, and ordering accuracy."""
    starts, goals, labels = pairs
    rows = lambda frames: torch.from_numpy(np.asarray(cache[frames])).to(device)
    targets = rows(goals)
    predicted = cost(rows(starts), targets)
    regression = functional.smooth_l1_loss(predicted, torch.from_numpy(labels).to(device))
    ordered, closer = one_step_closer(starts, goals, counts)
    if not len(ordered):
        zero = predicted.sum() * 0
        return regression, zero, zero
    margin = predicted[ordered] - cost(rows(closer), targets[ordered])
    return regression, functional.relu(1 - margin).mean(), (margin > 0).float().mean()


@torch.no_grad()
def validate(cost: TemporalCost, cache: np.ndarray, counts: np.ndarray, pairs: tuple, device, chunk: int = 512) -> dict:
    """Average the losses over fixed validation pairs, in equal chunks."""
    parts = [
        losses(cost, cache, counts, tuple(array[start : start + chunk] for array in pairs), device)
        for start in range(0, len(pairs[0]), chunk)
    ]
    names = ("val_loss", "val_ordering", "val_order_accuracy")
    return {name: float(np.mean([part[index].item() for part in parts])) for index, name in enumerate(names)}


def train(cost: TemporalCost, cache, counts, val_cache, val_counts, config: dict, device) -> tuple[list[dict], dict]:
    """Fit the cost. Return periodic losses and the weights with the best validation score."""
    rng = np.random.default_rng(config["seed"])
    sample = (config["batch_size"], config["max_gap"], config["far_fraction"], rng)
    val_rng = np.random.default_rng(config["seed"] + 1)
    val_pairs = sample_pairs(val_counts, VALIDATION_PAIRS, config["max_gap"], config["far_fraction"], val_rng)
    weight = config["ordering_weight"]
    optimizer = torch.optim.AdamW(cost.parameters(), lr=config["learning_rate"])
    history, best_score, best_state = [], math.inf, None
    for step in range(1, config["train_steps"] + 1):
        regression, ordering, _ = losses(cost, cache, counts, sample_pairs(counts, *sample), device)
        optimizer.zero_grad()
        (regression + weight * ordering).backward()
        optimizer.step()
        if step % max(1, config["train_steps"] // 20) == 0:
            row = {"step": step, "train_loss": regression.item(), "train_ordering": ordering.item()}
            row.update(validate(cost, val_cache, val_counts, val_pairs, device))
            score = row["val_loss"] + weight * row["val_ordering"]
            if score < best_score:
                best_score, best_state = score, deepcopy(cost.state_dict())
                row["best"] = True
            history.append(row)
            print(json.dumps(row), flush=True)
    return history, best_state


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
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    torch.manual_seed(config["seed"])
    args.output.mkdir(parents=True)
    model = load_world_model("pusht", str(device)).requires_grad_(False).eval()
    counts = cache_features(model, args.data / "train", config["episodes"], config, args.output / "train.cache")
    val_counts = cache_features(model, args.data / "val", config["episodes"], config, args.output / "val.cache")
    del model
    cache = np.memmap(args.output / "train.cache", dtype=np.float16, mode="r", shape=(int(counts.sum()), FEATURE_SIZE))
    val_cache = np.memmap(args.output / "val.cache", dtype=np.float16, mode="r", shape=(int(val_counts.sum()), FEATURE_SIZE))

    cost = TemporalCost(width=config["width"], hidden=config["hidden"]).to(device)
    history, best_state = train(cost, cache, counts, val_cache, val_counts, config, device)
    torch.save({"state_dict": best_state, "config": config}, args.output / "temporal_cost.pt")
    for path in (args.output / "train.cache", args.output / "val.cache"):
        path.unlink()

    metadata = {
        "config": config,
        "git_revision": revision,
        "git_dirty": dirty,
        "torch": torch.__version__,
        "command": sys.argv,
        "elapsed_seconds": perf_counter() - started,
        "best_step": max(row["step"] for row in history if row.get("best")),
        "history": history,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
