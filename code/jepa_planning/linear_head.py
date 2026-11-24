"""Learned linear goal head L (proposal Sections 3-4).

The frozen DINO-WM stays unchanged. One matrix ``L`` of shape ``[r, D]``
(``r = 32``, ``D = 384``) is shared across all patches, kept at Frobenius norm
1, and used only in the goal cost:

    D_L(g, h) = mean_i ||L (g_i - h_i)||^2

It is fitted with InfoNCE on goal-image groups: two photos of the same puck
target with different hand positions are a positive pair, photos of other
targets are negatives.

Everything here works on cached features of shape ``[..., P, D]`` (``P``
patches), so fitting never touches the world model or the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HEAD_RANK = 32
FEATURE_DIM = 384
NUM_NEGATIVES = 7
BUDGETS = (64, 256, 1024)
HANDS_PER_TARGET = (2, 4, 8)


# ---------------------------------------------------------------- head matrices


def normalized(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``matrix / ||matrix||_F``; reject zero or non-finite matrices."""
    if matrix.ndim != 2:
        raise ValueError("head must be a matrix of shape [r, D]")
    norm = torch.linalg.matrix_norm(matrix)
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("head must have a finite, non-zero Frobenius norm")
    return matrix / norm


def random_head(rank: int = HEAD_RANK, dim: int = FEATURE_DIM, seed: int = 0) -> torch.Tensor:
    """Seeded Gaussian matrix with unit Frobenius norm (initialisation and control 2)."""
    generator = torch.Generator().manual_seed(seed)
    return normalized(torch.randn(rank, dim, generator=generator))


def coordinate_subset_head(rank: int = HEAD_RANK, dim: int = FEATURE_DIM, seed: int = 0) -> torch.Tensor:
    """Select ``rank`` distinct feature coordinates (control 3), unit norm."""
    if not 1 <= rank <= dim:
        raise ValueError("rank must be between 1 and dim")
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.randperm(dim, generator=generator)[:rank]
    head = torch.zeros(rank, dim)
    head[torch.arange(rank), coordinates] = 1
    return normalized(head)


def identity_head(dim: int = FEATURE_DIM) -> torch.Tensor:
    """``I_D / sqrt(D)``: the head under which D_L equals the plain feature MSE."""
    # ||I/sqrt(D) x||^2 = ||x||^2 / D, i.e. the mean over feature dims; with the
    # mean over patches in head_distance this is exactly the plain MSE cost.
    return torch.eye(dim) / dim**0.5


# --------------------------------------------------------------- distance, loss


def project(features: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Apply L to every patch: ``[..., P, D] -> [..., P, r]``."""
    if head.ndim != 2 or features.shape[-1] != head.shape[1]:
        raise ValueError("feature dimension must match the head's columns")
    return features.matmul(head.transpose(0, 1))


def head_distance(first: torch.Tensor, second: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """D_L for broadcastable ``[..., P, D]`` inputs; returns ``[...]``."""
    # Project the difference patch-wise, squared norm over the r outputs, mean over patches.
    return project(first - second, head).square().sum(dim=-1).mean(dim=-1)


def info_nce_logits(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    head: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Logits ``-D_L / tau``, column 0 = positive.

    Shapes: anchor and positive ``[B, P, D]``, negatives ``[B, K, P, D]``;
    returns ``[B, 1 + K]``.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if anchor.shape != positive.shape or negatives.shape[0] != anchor.shape[0] or negatives.shape[2:] != anchor.shape[1:]:
        raise ValueError("anchor, positive and negatives have incompatible shapes")
    # Project once and compare in the r-dim space ([B, 1, P, r] vs [B, 1+K, P, r]);
    # L is linear, so L(a) - L(c) = L(a - c) and this equals head_distance.
    projected_anchor = project(anchor, head).unsqueeze(1)
    candidates = project(torch.cat([positive.unsqueeze(1), negatives], dim=1), head)
    distances = (candidates - projected_anchor).square().sum(dim=-1).mean(dim=-1)
    return -distances / tau


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    head: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Mean InfoNCE loss with one positive and K negatives per anchor (Eq. 4)."""
    logits = info_nce_logits(anchor, positive, negatives, head, tau)
    # The positive is column 0, so the target class is 0 for every anchor.
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


class LinearGoalHead(torch.nn.Module):
    """Trainable ``L`` with ``||L||_F = 1`` restored after every update."""

    def __init__(self, initial: torch.Tensor):
        """Start from ``initial`` (any shape ``[r, D]``), rescaled to unit Frobenius norm."""
        super().__init__()
        self.weight = torch.nn.Parameter(normalized(initial.detach().clone().float()))

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Head distance ``mean_i ||L (first_i - second_i)||²`` (Eq. 4)."""
        return head_distance(first, second, self.weight)

    @torch.no_grad()
    def renormalize_(self, previous: torch.Tensor) -> bool:
        """Rescale to unit norm; if the norm is zero or not finite, restore ``previous``.

        Returns whether the update was kept.
        """
        norm = torch.linalg.matrix_norm(self.weight)
        if not torch.isfinite(norm) or norm == 0:
            # Proposal: skip the step if the norm is 0 (cannot be rescaled to 1).
            self.weight.copy_(previous)
            return False
        self.weight.div_(norm)
        return True


def train_step(head: LinearGoalHead, optimizer: torch.optim.Optimizer, batch: dict, tau: float) -> tuple[float, bool]:
    """One gradient step on InfoNCE followed by renormalisation.

    ``batch`` holds ``anchor``, ``positive`` and ``negatives`` feature tensors.
    Returns the loss before the step and whether the step was kept.
    """
    previous = head.weight.detach().clone()  # restored if the step must be skipped
    optimizer.zero_grad(set_to_none=True)
    loss = info_nce_loss(batch["anchor"], batch["positive"], batch["negatives"], head.weight, tau)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), head.renormalize_(previous)


def save_head(path: Path, head: torch.Tensor, config: dict) -> None:
    """Save a head and its training config (rank, B, M, K, resample, τ, lr, ...)."""
    torch.save({"head": head.detach().cpu(), "config": config}, path)


def load_head(path: Path) -> tuple[torch.Tensor, dict]:
    """Load a saved head (renormalized to unit norm) and its config."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return normalized(payload["head"]), payload["config"]


# ------------------------------------------------------------------ allocations


def allocation_grid(budgets=BUDGETS, hands_per_target=HANDS_PER_TARGET) -> list[tuple[int, int, int]]:
    """``(B, M, K)`` for every budget ``B = M * K``."""
    grid = []
    for budget in budgets:
        for hands in hands_per_target:
            if budget % hands:
                raise ValueError(f"budget {budget} is not divisible by K={hands}")
            grid.append((budget, budget // hands, hands))
    return grid


class AllocationSampler:
    """Exactly ``B = M * K`` training images and InfoNCE batches drawn from them.

    ``pool_groups`` gives the target-group id of every image in the training
    pool. For a fixed ``seed`` the selection is nested: the M targets are the
    first M of one seeded target order, and the K images of each target are the
    first K of a seeded per-target order, so larger allocations contain smaller
    ones. Different seeds are the independent resamples.

    With ``shuffle_labels=True`` (control 4) the same B images are randomly
    regrouped into M groups of K, destroying the target structure while keeping
    the data and the batch layout.
    """

    def __init__(self, pool_groups: np.ndarray, num_targets: int, hands_per_target: int, seed: int,
                 shuffle_labels: bool = False):
        """Select the B images; see the class docstring for the nesting rule."""
        pool_groups = np.asarray(pool_groups)
        if pool_groups.ndim != 1:
            raise ValueError("pool_groups must be one group id per image")
        if hands_per_target < 2:
            raise ValueError("need at least two images per target for a positive pair")
        if num_targets < NUM_NEGATIVES + 1:
            raise ValueError(f"need at least {NUM_NEGATIVES + 1} targets for {NUM_NEGATIVES} negatives")
        group_ids = np.unique(pool_groups)
        if num_targets > len(group_ids):
            raise ValueError(f"pool has {len(group_ids)} targets, {num_targets} requested")

        # One seeded target order per resample; taking its first M keeps allocations
        # nested (M=8 targets are a subset of M=16, ...). SeedSequence sub-streams
        # [seed, 0], [seed, 1, group], [seed, 2] keep the draws independent.
        order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(group_ids)
        groups = []
        for group in order[:num_targets]:
            members = np.flatnonzero(pool_groups == group)
            if len(members) < hands_per_target:
                raise ValueError(f"target {group} has {len(members)} images, {hands_per_target} requested")
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1, int(group)]))
            groups.append(rng.permutation(members)[:hands_per_target])  # first K agents of this target
        groups = np.stack(groups)
        if shuffle_labels:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            # Same B images, random regrouping: "positives" no longer share a target.
            groups = rng.permutation(groups.ravel()).reshape(groups.shape)
        self.groups = groups  # [M, K] pool image indices
        self.target_ids = order[:num_targets]

    @property
    def images(self) -> np.ndarray:
        """The B pool indices this allocation may use."""
        return self.groups.ravel()

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Pool indices for ``batch_size`` (anchor, positive, 7 negatives) triples.

        The group is uniform; anchor and positive are two distinct images of it;
        each negative is a random image from a distinct other group.
        """
        num_targets, hands = self.groups.shape
        group = rng.integers(num_targets, size=batch_size)
        anchor_slot = rng.integers(hands, size=batch_size)
        # Offset in [1, K-1] mod K: a uniformly random slot that is never the anchor's.
        positive_slot = (anchor_slot + rng.integers(1, hands, size=batch_size)) % hands
        # 7 distinct other groups per row: random sort keys, the anchor's group pushed
        # to the end with +inf, then take the 7 smallest.
        keys = rng.random((batch_size, num_targets))
        keys[np.arange(batch_size), group] = np.inf
        negative_groups = np.argsort(keys, axis=1)[:, :NUM_NEGATIVES]
        negative_slots = rng.integers(hands, size=negative_groups.shape)
        return {
            "anchor": self.groups[group, anchor_slot],
            "positive": self.groups[group, positive_slot],
            "negatives": self.groups[negative_groups, negative_slots],
        }


# ------------------------------------------------------- photo-level evaluation


@torch.no_grad()
def pairwise_head_distance(first: torch.Tensor, second: torch.Tensor, head: torch.Tensor,
                           batch_size: int = 256) -> torch.Tensor:
    """D_L between every row of ``first [n, P, D]`` and ``second [m, P, D]``: ``[n, m]``."""
    def projected(features):
        """Project features in batches and flatten patches: ``[n, P * r]``."""
        return torch.cat([project(features[i : i + batch_size].to(head), head).flatten(1)
                          for i in range(0, len(features), batch_size)])
    a, b = projected(first), projected(second)
    patches = first.shape[-2]
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 for all pairs at once (one matmul);
    # clamp removes tiny negatives from rounding, / patches gives the patch mean.
    squared = a.square().sum(1, keepdim=True) - 2 * a @ b.T + b.square().sum(1)[None]
    return squared.clamp_min(0) / patches


def auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """P(positive score > negative score), ties counted as 1/2 (Mann-Whitney)."""
    from scipy.stats import rankdata

    scores = np.concatenate([positive_scores, negative_scores])
    ranks = rankdata(scores)  # average ranks, so ties count 1/2
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    # Mann-Whitney U of the positives divided by the number of (pos, neg) pairs.
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def retrieval_metrics(features: torch.Tensor, groups: np.ndarray, head: torch.Tensor) -> dict[str, float]:
    """Do photos of the same target lie closer under D_L than photos of other targets?

    ``top1``: fraction of photos whose nearest other photo shows the same target.
    ``auroc``: probability that a same-target pair is closer than a
    different-target pair. ``within_median`` is the median same-target
    distance, the scale used by :func:`sensitivity_ratios`.
    """
    groups = np.asarray(groups)
    distance = pairwise_head_distance(features, features, head).cpu().double().numpy()
    same = groups[:, None] == groups[None, :]
    off_diagonal = ~np.eye(len(groups), dtype=bool)
    np.fill_diagonal(distance, np.inf)  # a photo is not its own nearest neighbour
    top1 = float(np.mean(same[np.arange(len(groups)), distance.argmin(1)]))
    upper = np.triu(off_diagonal, 1)  # each unordered pair once
    within, between = distance[upper & same], distance[upper & ~same]
    return {
        "top1": top1,
        # Smaller distance should mean "same target", so score = -distance.
        "auroc": auroc(-within, -between),
        "within_median": float(np.median(within)),
        "between_median": float(np.median(between)),
    }


@torch.no_grad()
def sensitivity_ratios(reference: torch.Tensor, perturbed: torch.Tensor, head: torch.Tensor,
                       within_median: float) -> np.ndarray:
    """D_L(reference_j, perturbed_j) / median same-target distance, per pair.

    Values well above 1 mean the head still separates a slightly moved target
    from the original even though it ignores the agent; values near 0 mean the
    head collapsed the target change.
    """
    distance = head_distance(reference.to(head), perturbed.to(head), head)
    return (distance / within_median).cpu().numpy()


# --------------------------------------------------------------------- fitting


def fit_head(features: torch.Tensor, sampler: AllocationSampler, *, steps: int, batch_size: int, tau: float,
             lr: float, seed: int, rank: int = HEAD_RANK, validation=None, eval_every: int = 0):
    """Fit L on one allocation with Adam, renormalising after every step.

    ``features`` is the whole training pool ``[n, P, D]`` (any float dtype,
    on the training device); the sampler restricts which rows are used.
    ``validation`` is an optional ``(features, groups)`` pair scored with
    :func:`retrieval_metrics` every ``eval_every`` steps and at the end.

    Returns ``(head_matrix, curve)`` where ``curve`` lists dicts with step,
    mean training loss since the last record, skipped steps and validation
    metrics.
    """
    device = features.device
    # Random unit-norm start, seeded per resample, so fits differ only by seed.
    head = LinearGoalHead(random_head(rank, features.shape[-1], seed)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    curve, losses, skipped = [], [], 0

    def record(step):
        """Append the mean loss since the last record (and val metrics) to the curve."""
        row = {"step": step, "loss": float(np.mean(losses)) if losses else float("nan"), "skipped": skipped}
        if validation is not None:
            row |= {f"val_{k}": v for k, v in retrieval_metrics(validation[0], validation[1], head.weight.detach()).items()}
        curve.append(row)
        losses.clear()

    for step in range(1, steps + 1):
        index = sampler.sample(batch_size, rng)
        # index_select is identical to features[index] but much faster on MPS.
        batch = {key: torch.index_select(features, 0, torch.as_tensor(value.ravel(), device=device))
                 .reshape(*value.shape, *features.shape[1:]).float() for key, value in index.items()}
        loss, kept = train_step(head, optimizer, batch, tau)
        losses.append(loss)
        skipped += not kept  # steps undone because the norm became 0 / non-finite
        if eval_every and step % eval_every == 0 and step != steps:
            record(step)
    record(steps)
    return head.weight.detach().cpu(), curve


# ---------------------------------------------------------------------- catalog


def head_catalog(run_directories: list[Path], dim: int = FEATURE_DIM,
                 control_seeds=range(5)) -> list[tuple[dict, torch.Tensor]]:
    """Every head to evaluate: plain MSE, saved fitted heads, and per-rank controls.

    ``run_directories`` are train_linear_head.py outputs (each has ``heads/``).
    Returns ``(info, head)`` pairs; ``info`` has ``method``, ``head`` and the
    head's config (rank, compression, B, M, K, resample, ...).
    """
    catalog = [({"method": "plain_mse", "head": "identity", "rank": dim, "compression": 1.0},
                identity_head(dim))]
    ranks = []
    for directory in run_directories:
        for path in sorted((Path(directory) / "heads").glob("*.pt")):
            head, config = load_head(path)
            method = "linear_head_shuffled" if config["shuffled_labels"] else "linear_head"
            catalog.append(({"method": method, "head": path.stem, **config}, head))
            ranks.append(int(config["rank"]))
    # Controls get the same rank as the fitted heads they are compared with.
    for rank in sorted(set(ranks)):
        size = {"rank": rank, "compression": dim / rank}
        for seed in control_seeds:
            catalog.append(({"method": "random_projection", "head": f"rank{rank}_random_s{seed}", **size,
                             "seed": seed}, random_head(rank, dim, seed)))
            catalog.append(({"method": "coordinate_subset", "head": f"rank{rank}_coords_s{seed}", **size,
                             "seed": seed}, coordinate_subset_head(rank, dim, seed)))
    return catalog
