"""Variance-decomposed (ANOVA) spatial-channel factored goal head ``(w, L)``.

In 3D fixed-camera manipulation (such as the ``corner2`` view of 7-DoF Sawyer
``Push-v3``), roughly 75% of embodiment feature variance originates from
upper-arm patches while the 4 cm puck occupies roughly 15-40 tabletop patches.
Optimizing spatial patch weights ``w in Delta^{P-1}`` end-to-end with InfoNCE
suffers from *ViT border-token shortcut collapse*: static peripheral patches
where the arm never enters have near-zero within-group variance
``sigma^2_within(i) ~ 0`` alongside tiny global self-attention leakage
``sigma^2_between(i) = eps > 0``, producing huge raw SNR ``eps / 0`` despite
negligible physical rollout sensitivity.

Decomposing per-patch feature variance via the Law of Total Variance across
multi-embodiment goal groups,

    sigma^2_obj(i) = max(sigma^2_total(i) - sigma^2_within(i), 0),

and regularizing the denominator with the patch-median total variance
``tau_var = median_j(sigma^2_total(j)) + eps`` eliminates border-token
shortcuts in closed form without segmentation masks or 3D labels. Scaling patch
tokens by ``s_i = sqrt(P * w_i)`` then reduces spatially weighted projected
distance to standard uniform-patch distance,

    sum_i w_i ||L (a_i - b_i)||^2 == mean_i ||L (s_i a_i - s_i b_i)||^2,

allowing spectrally calibrated channel projections ``L`` (``k``-Fantope or
log-det row isotropy) to be fitted directly on the spatially reweighted pool.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import torch

from jepa_planning import linear_head as lh
from jepa_planning.calibrated_objectives import projected_multi_goal_distance
from jepa_planning.trust_mpc import (
    effective_rank,
    fit_fantope_head,
    fit_isotropy_head,
)

_VALID_REGIMES = ("fantope", "isotropy", "unregularized")


def _validate_and_normalize_weights(
    weights: torch.Tensor,
    *,
    num_patches: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Validate 1D nonnegative spatial weights with positive sum and normalize."""
    if not isinstance(weights, torch.Tensor):
        raise TypeError("spatial_weights must be a torch.Tensor")
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("spatial_weights must be a non-empty 1D tensor")
    if num_patches is not None and weights.shape != (num_patches,):
        raise ValueError(
            f"spatial_weights must have shape ({num_patches},), got {tuple(weights.shape)}"
        )

    target_device = weights.device if device is None else torch.device(device)
    w = weights.detach().to(device=target_device, dtype=dtype)
    if not torch.isfinite(w).all():
        raise ValueError("spatial_weights must contain only finite values")
    if bool((w < 0.0).any()):
        raise ValueError("spatial_weights must be nonnegative")
    total = w.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("spatial_weights must have a strictly positive sum")
    return w / total


def effective_patches(weights: torch.Tensor) -> float:
    """Inverse Simpson participation ratio ``1 / sum_i w_i^2`` in ``[1, P]``."""
    w = _validate_and_normalize_weights(weights, dtype=torch.float32)
    denom = float(w.square().sum().item())
    return float(1.0 / max(denom, 1e-12))


def spatial_patch_scale(
    spatial_weights: torch.Tensor | None,
    num_patches: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return patch multiplier ``s_i = sqrt(P * w_i)`` (or ones when ``None``).

    Re-normalizes ``w`` onto the probability simplex so that

        sum_i w_i ||L (a_i - b_i)||^2 == mean_i ||L (s_i a_i - s_i b_i)||^2

    holds identically for every linear head ``L``.
    """
    if (
        isinstance(num_patches, bool)
        or not isinstance(num_patches, (int, np.integer))
        or int(num_patches) < 1
    ):
        raise ValueError("num_patches must be a positive integer")
    patches = int(num_patches)
    if spatial_weights is None:
        return torch.ones(patches, device=device, dtype=dtype)

    w_norm = _validate_and_normalize_weights(
        spatial_weights,
        num_patches=patches,
        device=device,
        dtype=dtype,
    )
    return torch.sqrt(float(patches) * w_norm)


def anova_spatial_weights(
    features: torch.Tensor,
    groups: np.ndarray | Sequence[int],
    *,
    gamma: float = 1.5,
    uniform_floor: float = 0.25,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Compute closed-form variance-decomposed spatial patch weights ``w in Delta^{P-1}``.

    For each patch ``i`` across ``N`` cached feature maps ``[N, P, D]`` labeled
    by multi-embodiment goal group ids ``groups``:

    1. Exact pairwise within-group variance ``var_within[i]`` equals
       ``0.5 * E_{same g, a != b}[mean_d (z_{a,i,d} - z_{b,i,d})^2]``, computed
       by weighting each group's unbiased within-group variance by
       ``binom(K_g, 2)``.
    2. Total variance ``var_total[i] = var_n(z_{n,i})`` (population variance
       averaged over channels ``D``) and object-signal variance
       ``var_between[i] = max(var_total[i] - var_within[i], 0)``.
    3. Median-regularized SNR ``score[i] = var_between[i] / (var_within[i] + tau_var)``
       with ``tau_var = median_j(var_total[j]) + eps`` suppresses zero-variance
       border tokens.
    4. Soft power shaping with exponent ``gamma`` and uniform floor
       ``uniform_floor`` yields simplex weights ``w`` summing to 1.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 3 or min(features.shape) == 0:
        raise ValueError("features must be a non-empty 3D tensor of shape [N, P, D]")
    num_samples, num_patches, feature_dim = features.shape

    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or len(groups_arr) != num_samples:
        raise ValueError(f"groups must be a 1D sequence of length {num_samples}")
    if isinstance(gamma, bool) or not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise ValueError("gamma must be a finite nonnegative number")
    if (
        isinstance(uniform_floor, bool)
        or not math.isfinite(float(uniform_floor))
        or not 0.0 <= float(uniform_floor) <= 1.0
    ):
        raise ValueError("uniform_floor must be a finite number in [0, 1]")
    if isinstance(eps, bool) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive number")

    x = features.float()
    if not torch.isfinite(x).all():
        raise ValueError("features must contain only finite values")

    unique_ids, inverse, counts = np.unique(groups_arr, return_inverse=True, return_counts=True)
    valid_mask = counts >= 2
    if not np.any(valid_mask):
        raise ValueError("groups must contain at least one group with at least 2 samples")

    device = x.device
    num_groups = len(unique_ids)
    inv_idx = torch.as_tensor(inverse, dtype=torch.long, device=device)
    counts_t = torch.as_tensor(counts, dtype=torch.float32, device=device)

    group_sums = torch.zeros((num_groups, num_patches, feature_dim), device=device, dtype=torch.float32)
    group_sums.index_add_(0, inv_idx, x)
    group_means = group_sums / counts_t[:, None, None]

    sq_dev = (x - group_means.index_select(0, inv_idx)).square().mean(dim=-1)
    group_sse = torch.zeros((num_groups, num_patches), device=device, dtype=torch.float32)
    group_sse.index_add_(0, inv_idx, sq_dev)

    valid_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    valid_counts = counts_t[valid_t]
    # Exact binom(K_g, 2)-weighted average of group_sse[g] / (K_g - 1):
    # sum_g (K_g / 2) * group_sse[g] / sum_g (K_g * (K_g - 1) / 2).
    total_ordered_pairs = float((valid_counts * (valid_counts - 1.0)).sum().item())
    var_within = (valid_counts[:, None] * group_sse[valid_t]).sum(dim=0) / total_ordered_pairs

    var_total = x.var(dim=0, unbiased=False).mean(dim=-1)
    var_between = (var_total - var_within).clamp_min(0.0)
    tau_var = float(var_total.median().item()) + float(eps)
    score = var_between / (var_within + tau_var)

    gamma_f = float(gamma)
    floor_f = float(uniform_floor)
    if gamma_f == 0.0 or float(score.max().item()) <= 0.0:
        w_peaked = torch.full((num_patches,), 1.0 / num_patches, device=device, dtype=torch.float32)
    else:
        powered = score.clamp_min(1e-12).pow(gamma_f)
        w_peaked = powered / powered.sum()

    w = (1.0 - floor_f) * w_peaked + (floor_f / num_patches)
    w = w / w.sum()

    diagnostics: dict[str, torch.Tensor | float] = {
        "var_total": var_total,
        "var_within": var_within,
        "var_between": var_between,
        "score": score,
        "tau_var": tau_var,
        "effective_patches": float(1.0 / (w.square().sum().item() + 1e-12)),
    }
    return w, diagnostics


def spatial_channel_distance(
    candidates: torch.Tensor,
    goals: torch.Tensor,
    head: torch.Tensor,
    spatial_weights: torch.Tensor | None = None,
    *,
    exact_mean: bool = False,
) -> torch.Tensor:
    """Evaluate spatially weighted projected distance ``sum_i w_i ||L(z_i - g_i)||^2``."""
    if spatial_weights is None:
        return projected_multi_goal_distance(candidates, goals, head, exact_mean=exact_mean)

    if candidates.ndim < 3 or candidates.shape[0] == 0 or any(d == 0 for d in candidates.shape[1:]):
        raise ValueError("candidates must be non-empty with shape [N, P, D]")
    num_patches = candidates.shape[-2]
    if goals.ndim not in (candidates.ndim - 1, candidates.ndim):
        raise ValueError("goals must have shape [P, D] or [K, P, D] matching candidates")
    if goals.shape[-2] != num_patches:
        raise ValueError("goals patch count must match candidates")

    scale = spatial_patch_scale(
        spatial_weights,
        num_patches,
        device=candidates.device,
        dtype=candidates.dtype,
    )
    cand_scale = scale.view(*([1] * (candidates.ndim - 2)), num_patches, 1)
    goal_scale = scale.view(*([1] * (goals.ndim - 2)), num_patches, 1)
    return projected_multi_goal_distance(
        candidates * cand_scale,
        goals * goal_scale,
        head,
        exact_mean=exact_mean,
    )


class SpatialChannelGoalHead(torch.nn.Module):
    """Factored spatial-channel goal head wrapping ``L [r, D]`` and ``w [P]``."""

    def __init__(self, head: torch.Tensor, spatial_weights: torch.Tensor) -> None:
        super().__init__()
        self.head = torch.nn.Parameter(lh.normalized(head.detach().clone().float()))
        w = _validate_and_normalize_weights(spatial_weights, dtype=torch.float32)
        self.register_buffer("spatial_weights", w)

    def distance(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        exact_mean: bool = False,
    ) -> torch.Tensor:
        """Compute spatially weighted projected distance between ``first`` and ``second``."""
        head = self.head.to(device=first.device, dtype=first.dtype)
        weights = self.spatial_weights.to(device=first.device, dtype=first.dtype)
        if first.ndim == 2:
            return spatial_channel_distance(
                first.unsqueeze(0),
                second,
                head,
                weights,
                exact_mean=exact_mean,
            ).squeeze(0)
        return spatial_channel_distance(
            first,
            second,
            head,
            weights,
            exact_mean=exact_mean,
        )

    def effective_patches(self) -> float:
        """Inverse Simpson effective patch count of ``self.spatial_weights``."""
        return effective_patches(self.spatial_weights)

    def effective_rank(self) -> float:
        """Participation-ratio effective rank of ``L^T L``."""
        return effective_rank(self.head.detach())


def fit_anova_spatial_head(
    features: torch.Tensor,
    groups: np.ndarray,
    sampler: lh.AllocationSampler,
    *,
    gamma: float = 1.5,
    uniform_floor: float = 0.25,
    regime: str = "fantope",
    k_fantope: float = 6.0,
    isotropy_weight: float = 0.015,
    steps: int = 1200,
    batch_size: int = 32,
    tau: float = 0.1,
    lr: float = 3e-3,
    seed: int = 0,
    rank: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit factored ``(w, L)`` using only the sampled allocation pool ``sampler.images``."""
    if regime not in _VALID_REGIMES:
        raise ValueError(f"regime must be one of {_VALID_REGIMES}, got {regime!r}")
    if not isinstance(features, torch.Tensor) or features.ndim != 3 or min(features.shape) == 0:
        raise ValueError("features must be a non-empty 3D tensor of shape [N, P, D]")

    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or len(groups_arr) != features.shape[0]:
        raise ValueError(f"groups must have shape ({features.shape[0]},)")

    pool_indices = sampler.images
    pool_idx_tensor = torch.as_tensor(pool_indices, dtype=torch.long, device=features.device)
    pool_features = features.index_select(0, pool_idx_tensor)
    w, anova_info = anova_spatial_weights(
        pool_features,
        groups_arr[pool_indices],
        gamma=gamma,
        uniform_floor=uniform_floor,
    )

    patch_scale = spatial_patch_scale(
        w,
        features.shape[1],
        device=features.device,
        dtype=torch.float32,
    )
    scaled_features = features.float() * patch_scale[None, :, None]

    if regime == "fantope":
        head = fit_fantope_head(
            scaled_features,
            sampler,
            k_fantope=k_fantope,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            seed=seed,
            rank=rank,
        )
    elif regime == "isotropy":
        head = fit_isotropy_head(
            scaled_features,
            sampler,
            isotropy_weight=isotropy_weight,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            seed=seed,
            rank=rank,
        )
    else:
        head, _ = lh.fit_head(
            scaled_features,
            sampler,
            steps=steps,
            batch_size=batch_size,
            tau=tau,
            lr=lr,
            seed=seed,
            rank=rank,
        )

    w_cpu = w.detach().cpu()
    head_cpu = head.detach().cpu()
    metadata = {
        "regime": regime,
        "gamma": float(gamma),
        "uniform_floor": float(uniform_floor),
        "k_fantope": float(k_fantope) if regime == "fantope" else None,
        "isotropy_weight": float(isotropy_weight) if regime == "isotropy" else None,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "tau": float(tau),
        "lr": float(lr),
        "seed": int(seed),
        "rank": int(rank),
        "num_patches": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "pool_size": int(len(pool_indices)),
        "tau_var": float(anova_info["tau_var"]),
        "effective_patches": float(anova_info["effective_patches"]),
        "effective_rank": float(effective_rank(head_cpu)),
        "anova_score_max": float(anova_info["score"].max().item()),
        "anova_score_median": float(anova_info["score"].median().item()),
    }
    return w_cpu, head_cpu, metadata


def save_spatial_channel_head(
    path: Path,
    spatial_weights: torch.Tensor,
    head: torch.Tensor,
    metadata: dict | None = None,
) -> None:
    """Save a spatial-channel head checkpoint ``(w, L, metadata)``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_norm = _validate_and_normalize_weights(spatial_weights, device="cpu", dtype=torch.float32)
    head_norm = lh.normalized(head.detach().cpu().float())
    meta = dict(metadata) if metadata is not None else {}
    torch.save(
        {
            "head": head_norm,
            "spatial_weights": w_norm,
            "config": meta,
            "metadata": meta,
        },
        out_path,
    )


def load_spatial_channel_head(path: Path) -> tuple[torch.Tensor | None, torch.Tensor, dict]:
    """Load either a spatial-channel checkpoint or a legacy linear-head checkpoint.

    Returns ``(spatial_weights, head, metadata)``, where ``spatial_weights`` is
    ``None`` for standard ``linear_head.save_head`` files.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    head = lh.normalized(payload["head"].float())
    raw_meta = payload.get("metadata")
    if raw_meta is None:
        raw_meta = payload.get("config")
    metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    raw_weights = payload.get("spatial_weights")
    if raw_weights is None:
        return None, head, metadata
    weights = _validate_and_normalize_weights(raw_weights, device="cpu", dtype=torch.float32)
    return weights, head, metadata
