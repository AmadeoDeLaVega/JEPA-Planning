#!/usr/bin/env python3
"""Train the quasimetric (minimum-step) goal cost on recorded PushT trajectories.

Frames are encoded once with the frozen DINO-WM encoder, as for the temporal
cost. Each training step draws a batch of recorded transitions, which must cost
at most one step, and a batch of random frame pairs, whose distances are pushed
up (``QuasimetricObjective``). Only the cost is trained.

The saved weights are the final ones. Validation numbers on separate scenes are
logged but not used to pick a checkpoint: the temporal study showed that
step-counting scores along a recording did not predict planning, and the
branching check and dev banks are the tests that count.

Run from the repository root after downloading ``pusht_noise`` into ``data/``::

    .venv/bin/python code/experiments/train_pusht_quasimetric_cost.py \
      --device cuda --cache artifacts/pusht_feature_cache \
      --output results/pusht_quasimetric_cost/seed_20261201

``--cache`` keeps the encoded features (about 90 GB for all episodes) for later
runs with the same episodes and frame stride. Without it the cache is written to
the output directory and deleted at the end. Use --smoke for a small pipeline
check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_pusht_temporal_cost import FEATURE_SIZE, cache_features  # noqa: E402
from jepa_planning import ROOT  # noqa: E402
from jepa_planning.model import load_world_model  # noqa: E402
from jepa_planning.quasimetric_cost import (  # noqa: E402
    QuasimetricCost,
    QuasimetricObjective,
    sample_frame_pairs,
    sample_transitions,
)
from jepa_planning.temporal_cost import one_step_closer, sample_pairs  # noqa: E402

VALIDATION_SAMPLES = 4096
VALIDATION_MAX_GAP = 24  # model steps, as in the temporal cost's validation pairs
CONFIG_FIELDS = {
    "seed",
    "episodes",
    "frame_stride",
    "horizon",
    "slack",
    "components",
    "component_dim",
    "width",
    "hidden",
    "batch_size",
    "train_steps",
    "learning_rate",
    "multiplier_learning_rate",
    "encode_batch_size",
}
INTEGER_FIELDS = CONFIG_FIELDS - {"seed", "horizon", "slack", "learning_rate", "multiplier_learning_rate"}


def validate_config(config: dict) -> None:
    """Reject incomplete or invalid training configurations."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError(f"config fields must be {sorted(CONFIG_FIELDS)}")
    for field in INTEGER_FIELDS:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit uint32")
    if not math.isfinite(config["horizon"]) or config["horizon"] <= 1:
        raise ValueError("horizon must be finite and exceed one step")
    if not math.isfinite(config["slack"]) or config["slack"] <= 0:
        raise ValueError("slack must be finite and positive")
    for field in ("learning_rate", "multiplier_learning_rate"):
        if not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be finite and positive")


def build_cost(config: dict) -> QuasimetricCost:
    """Construct the cost described by a training config."""
    return QuasimetricCost(
        width=config["width"],
        hidden=config["hidden"],
        components=config["components"],
        component_dim=config["component_dim"],
    )


def load_or_build_cache(
    directory: Path,
    key: dict,
    build: Callable[[str, Path], np.ndarray],
    feature_size: int = FEATURE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return read-only train and validation caches with their episode frame counts.

    A cache is reused only if it was built with the same ``key`` (episodes, frame
    stride, data). Otherwise ``build(split, path)`` encodes each split and returns
    its frame counts. ``cache.json`` is written last, so an interrupted build is
    rebuilt rather than reused.
    """
    manifest = directory / "cache.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["key"] != key:
            raise ValueError(f"{directory} was built for {saved['key']}, not {key}; choose another --cache")
        counts = {split: np.asarray(saved["counts"][split]) for split in ("train", "val")}
    else:
        directory.mkdir(parents=True, exist_ok=True)
        counts = {split: np.asarray(build(split, directory / f"{split}.cache")) for split in ("train", "val")}
        manifest.write_text(json.dumps({"key": key, "counts": {s: c.tolist() for s, c in counts.items()}}) + "\n")
    caches = {
        split: np.memmap(directory / f"{split}.cache", dtype=np.float16, mode="r",
                         shape=(int(counts[split].sum()), feature_size))
        for split in ("train", "val")
    }
    return caches["train"], counts["train"], caches["val"], counts["val"]


def rows(cache: np.ndarray, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(cache[frames])).to(device)


def validation_samples(counts: np.ndarray, seed: int) -> dict[str, tuple]:
    """Fixed validation transitions and same-episode (start, goal, gap) pairs."""
    rng = np.random.default_rng(seed)
    starts, goals, gaps = sample_pairs(counts, VALIDATION_SAMPLES, VALIDATION_MAX_GAP, 0.0, rng)
    ahead = gaps > 0
    return {
        "transitions": sample_transitions(counts, VALIDATION_SAMPLES, rng),
        "pairs": (starts[ahead], goals[ahead], gaps[ahead]),
    }


@torch.no_grad()
def validate(cost: QuasimetricCost, cache, counts, samples: dict, device, chunk: int = 512) -> dict:
    """Check the step rule and the recorded-gap bound on unseen episodes.

    - ``val_step_distance`` and ``val_violation``: recorded steps should cost
      about one, as on training episodes. ``val_step_p10/p50/p90`` show the
      spread, which the mean hides: every step should be near one, not just
      the average.
    - ``val_gap_ratio``: mean distance over recorded gap for same-episode pairs.
      With every step near one it should be at most about ``1 + slack``; below
      one means the cost found shorter routes than the recording took.
    - ``val_order_accuracy``: how often the next recorded frame is scored nearer
      the same goal ahead.
    """
    def distances(starts, goals):
        return torch.cat([
            cost(rows(cache, starts[i : i + chunk], device), rows(cache, goals[i : i + chunk], device))
            for i in range(0, len(starts), chunk)
        ])

    frames, after = samples["transitions"]
    step = distances(frames, after)
    starts, goals, gaps = samples["pairs"]
    to_goal = distances(starts, goals)
    ordered, closer = one_step_closer(starts, goals, counts)
    nearer = distances(closer, goals[ordered])
    p10, p50, p90 = torch.quantile(step.float(), torch.tensor([0.1, 0.5, 0.9], device=step.device)).tolist()
    return {
        "val_step_distance": step.mean().item(),
        "val_step_p10": p10,
        "val_step_p50": p50,
        "val_step_p90": p90,
        "val_violation": torch.nn.functional.relu(step - 1).square().mean().item(),
        "val_gap_ratio": (to_goal.cpu() / torch.from_numpy(gaps)).mean().item(),
        "val_order_accuracy": (nearer < to_goal[ordered]).float().mean().item(),
    }


def train(cost: QuasimetricCost, cache, counts, val_cache, val_counts, config: dict, device,
          checkpoints: Path | None = None) -> list[dict]:
    """Fit the cost and return periodic training and validation numbers.

    With ``checkpoints``, the weights at every logged step are saved there as
    ``step_NNNNN.pt``, so states before an overshoot can still be analysed.
    """
    rng = np.random.default_rng(config["seed"])
    samples = validation_samples(val_counts, config["seed"] + 1)
    objective = QuasimetricObjective(config["horizon"], config["slack"]).to(device)
    # The multiplier has its own learning rate: under Adam its raw value moves by
    # about that much per step, so a small one caps how fast the step rule can be
    # enforced. It gets no weight decay, which would bias it toward zero.
    optimizer = torch.optim.AdamW([
        {"params": cost.parameters()},
        {"params": objective.parameters(), "weight_decay": 0.0, "lr": config["multiplier_learning_rate"]},
    ], lr=config["learning_rate"])
    history = []
    for step in range(1, config["train_steps"] + 1):
        frames, after = sample_transitions(counts, config["batch_size"], rng)
        sources, goals = sample_frame_pairs(counts, config["batch_size"], rng)
        loss, stats = objective(cost, *(rows(cache, f, device) for f in (frames, after, sources, goals)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, config["train_steps"] // 20) == 0:
            row = {"step": step, "train_loss": loss.item(), **stats}
            row.update(validate(cost, val_cache, val_counts, samples, device))
            history.append(row)
            print(json.dumps(row), flush=True)
            if checkpoints is not None:
                checkpoints.mkdir(exist_ok=True)
                torch.save({"kind": "quasimetric", "state_dict": cost.state_dict(), "config": config, "step": step},
                           checkpoints / f"step_{step:05d}.pt")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/pusht_quasimetric_cost.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/pusht_noise")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pusht_quasimetric_cost")
    parser.add_argument("--cache", type=Path, help="keep encoded features here and reuse them in later runs")
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

    model = {}

    def build(split: str, path: Path) -> np.ndarray:
        if "world" not in model:
            model["world"] = load_world_model("pusht", str(device)).requires_grad_(False).eval()
        return cache_features(model["world"], args.data / split, config["episodes"], config, path)

    cache_dir = args.cache or args.output / "cache"
    key = {"episodes": config["episodes"], "frame_stride": config["frame_stride"], "data": str(args.data.resolve())}
    cache, counts, val_cache, val_counts = load_or_build_cache(cache_dir, key, build)
    model.clear()

    cost = build_cost(config).to(device)
    history = train(cost, cache, counts, val_cache, val_counts, config, device, checkpoints=args.output / "checkpoints")
    torch.save({"kind": "quasimetric", "state_dict": cost.state_dict(), "config": config},
               args.output / "quasimetric_cost.pt")
    if args.cache is None:
        del cache, val_cache
        shutil.rmtree(cache_dir)

    metadata = {
        "config": config,
        "git_revision": revision,
        "git_dirty": dirty,
        "torch": torch.__version__,
        "command": sys.argv,
        "feature_cache": str(args.cache) if args.cache else None,
        "elapsed_seconds": perf_counter() - started,
        "history": history,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
