"""Goal-image groups for fitting the linear goal head (plan step 1).

A group is several photos of one puck target that differ only in hand
position. Targets come from the same construction as the evaluation goals
(`baseline.prepare_task`): reset a seeded Push task, run the expert for
`warmup_steps`, then `goal_steps` more, and take that puck pose. Train, val
and test targets use disjoint seed ranges; test is the baseline's range.

Hand positions come from one fixed bank of absolute xy positions, sampled
uniformly over a workspace box independently of the targets and split into
disjoint train/val/test hands. The hand keeps its height at the goal snapshot
(xy-only bank). For each target, bank hands are tried in a seeded order and a
photo is kept only if the puck pose is unchanged, the hand reached the bank
position, and the hand is at least `min_hand_puck_distance` from the puck.
Rejections are counted per reason, because they make hands and targets
slightly dependent.

The pure helpers (seeds, bank, orders) need no simulator; the simulator code
imports `metaworld_push` lazily, as `baseline.py` does.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

SPLITS = ("train", "val", "test")


def validate_config(cfg: dict) -> None:
    """Raise ``ValueError`` if the goal-group config is missing fields or has bad values."""
    fields = {"seed", "warmup_steps", "goal_steps", "first_seed", "num_targets", "hands_per_target",
              "hand_bank", "min_hand_puck_distance", "max_reach_error"}
    if set(cfg) != fields:
        raise ValueError(f"Config fields must be {sorted(fields)}")
    for name in ("first_seed", "num_targets", "hands_per_target"):
        if set(cfg[name]) != set(SPLITS):
            raise ValueError(f"{name} needs an entry for each of {SPLITS}")
    bank = cfg["hand_bank"]
    if set(bank) != {"size", "fractions", "low", "high"} or set(bank["fractions"]) != set(SPLITS):
        raise ValueError("hand_bank needs size, fractions (per split), low and high")
    if not np.all(np.less(bank["low"], bank["high"])) or len(bank["low"]) != 2 or len(bank["high"]) != 2:
        raise ValueError("hand_bank low/high must be xy boxes with low < high")
    if abs(sum(bank["fractions"].values()) - 1) > 1e-9:
        raise ValueError("hand_bank fractions must sum to 1")
    sizes = bank_split_sizes(bank)
    for split in SPLITS:
        if cfg["hands_per_target"][split] < 2 or cfg["hands_per_target"][split] > sizes[split]:
            raise ValueError(f"{split}: need 2 <= hands_per_target <= {sizes[split]} bank hands")
    ranges = sorted((cfg["first_seed"][s], cfg["first_seed"][s] + cfg["num_targets"][s]) for s in SPLITS)
    if any(prev_end > start for (_, prev_end), (start, _) in zip(ranges, ranges[1:])):
        raise ValueError("train/val/test target seed ranges must be disjoint")


def target_seeds(cfg: dict, split: str) -> np.ndarray:
    """Task seeds of one split (a contiguous, split-specific range)."""
    start = cfg["first_seed"][split]
    return np.arange(start, start + cfg["num_targets"][split])


def bank_split_sizes(bank: dict) -> dict[str, int]:
    """Number of bank hands per split; train gets the remainder after val/test rounding."""
    sizes = {split: int(round(bank["size"] * bank["fractions"][split])) for split in SPLITS[1:]}
    sizes["train"] = bank["size"] - sum(sizes.values())  # so the split sizes always sum to size
    return sizes


def hand_bank(cfg: dict) -> dict[str, np.ndarray]:
    """Absolute hand xy positions per split; ``{split: [n, 2]}``, disjoint by construction."""
    bank = cfg["hand_bank"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0]))
    # One draw for the whole bank, then consecutive slices: no hand position is in two splits.
    positions = rng.uniform(bank["low"], bank["high"], size=(bank["size"], 2))
    sizes = bank_split_sizes(bank)
    out, start = {}, 0
    for split in SPLITS:
        out[split] = positions[start : start + sizes[split]]
        start += sizes[split]
    return out


def hand_order(cfg: dict, target_seed: int, bank_size: int) -> np.ndarray:
    """Seeded order in which a target tries the bank hands of its split."""
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1, int(target_seed)])).permutation(bank_size)


# ------------------------------------------------------------------- simulator


def make_target(env, seed: int, cfg: dict):
    """Goal snapshot and puck pose [7], built exactly as `baseline.prepare_task`."""
    from jepa_planning import metaworld_push as mp

    obs = mp.reset_task(env, seed)
    obs, _ = mp.run_expert(env, obs, cfg["warmup_steps"])
    mp.run_expert(env, obs, cfg["goal_steps"])
    return mp.snapshot(env), mp.puck_pose(env)


def render_group(env, goal_snap: dict, goal_pose: np.ndarray, bank_xy: np.ndarray, order: np.ndarray,
                 needed: int, cfg: dict) -> tuple[list[dict], Counter]:
    """Up to `needed` accepted photos of one target, trying bank hands in `order`."""
    from jepa_planning import metaworld_push as mp

    accepted, rejected = [], Counter()
    for index in order:
        if len(accepted) == needed:
            break
        target_xy = bank_xy[index]
        if np.linalg.norm(target_xy - goal_pose[:2]) < cfg["min_hand_puck_distance"]:
            rejected["near_puck"] += 1  # skip the simulation; the bank point itself is too close
            continue
        mp.restore(env, goal_snap)  # every photo starts from the same goal state
        # Move the hand to the absolute bank position (xy only; height unchanged).
        offset = np.array([*(target_xy - mp.hand_pos(env)[:2]), 0.0])
        mp.move_hand_keep_puck(env, offset)
        hand = mp.hand_pos(env)
        # Reject photos where moving the hand disturbed the puck or the goal is not met.
        if not np.array_equal(mp.puck_pose(env), goal_pose):
            rejected["puck_moved"] += 1
        elif np.linalg.norm(hand[:2] - target_xy) > cfg["max_reach_error"]:
            rejected["unreached"] += 1
        elif np.linalg.norm(hand[:2] - goal_pose[:2]) < cfg["min_hand_puck_distance"]:
            rejected["near_puck"] += 1
        else:
            accepted.append({"frame": env.render().copy(), "hand_bank_index": int(index), "hand_pos": hand})
    return accepted, rejected
