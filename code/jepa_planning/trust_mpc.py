"""Jensen-certified MPC and spectral-isotropy head fitting for frozen DINO-WM.

Consolidates two empirical findings from the PushT development split:
1. At large contrastive budgets (B = 1024), trace-normalized InfoNCE collapses
   effective rank toward 1 (top eigenvalue > 0.84), blinding closed-loop control
   to orthogonal pose coordinates. A log-det spectral isotropy penalty preserves
   multi-dimensional pose sensitivity.
2. Comparing 127 noisy Gaussian rollouts directly against 1 denoised elite mean
   suffers from Optimizer's Curse, while returning an unscored elite mean fails
   across non-convex contact barriers. Evaluating the final elite mean on the
   128th rollout and checking Jensen's inequality (c(mean) <= mean(c_elites))
   retains elite-mean denoising on unimodal basins and splits onto the dominant
   elite mode only when averaging crosses a barrier.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jepa_planning import linear_head as lh
from jepa_planning.goal_costs import linear_goal_distance


def zero_displacement_chunk(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    controls_per_chunk: int,
    action_clip: float,
) -> np.ndarray:
    """Normalized action chunk corresponding to zero raw controller displacement."""
    single_step_hold = np.clip(-action_mean / action_std, -action_clip, action_clip)
    return np.tile(single_step_hold.astype(np.float32), controls_per_chunk)


def effective_rank(head: torch.Tensor) -> float:
    """Participation-ratio effective rank 1 / sum_k lambda_k^2 of W = L^T L."""
    singular_values = torch.linalg.svdvals(head.float())
    eigenvalues = singular_values.square()
    eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    return float(1.0 / eigenvalues.square().sum())


def fit_isotropy_head(
    features: torch.Tensor,
    sampler: lh.AllocationSampler,
    *,
    steps: int = 2000,
    batch_size: int = 64,
    tau: float = 0.3,
    lr: float = 0.003,
    isotropy_weight: float = 0.05,
    seed: int = 0,
    rank: int = lh.HEAD_RANK,
) -> torch.Tensor:
    """Fit linear head L with InfoNCE plus a log-det row-isotropy regularizer."""
    device = features.device
    pool_indices = sampler.images
    subset = features[torch.as_tensor(pool_indices, dtype=torch.long, device=device)].float()

    local_map = np.empty(len(features), dtype=np.int64)
    local_map[pool_indices] = np.arange(len(pool_indices), dtype=np.int64)

    head = lh.LinearGoalHead(lh.random_head(rank, features.shape[-1], seed)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))

    patches = subset.shape[1]
    eye_jitter = torch.eye(rank, device=device) * 1e-4
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(steps):
        draw = sampler.sample(batch_size, rng)
        anc_idx = torch.as_tensor(local_map[draw["anchor"]], dtype=torch.long, device=device)
        pos_idx = torch.as_tensor(local_map[draw["positive"]], dtype=torch.long, device=device)
        neg_idx = torch.as_tensor(local_map[draw["negatives"]].ravel(), dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        projected = subset.matmul(head.weight.transpose(0, 1))
        anchor = torch.index_select(projected, 0, anc_idx).unsqueeze(1)
        positive = torch.index_select(projected, 0, pos_idx).unsqueeze(1)
        negatives = torch.index_select(projected, 0, neg_idx).reshape(batch_size, -1, patches, rank)

        candidates = torch.cat([positive, negatives], dim=1)
        distances = (candidates - anchor).square().sum(dim=-1).mean(dim=-1)
        nce_loss = F.cross_entropy(-distances / tau, targets)

        gram = rank * head.weight.matmul(head.weight.transpose(0, 1)) + eye_jitter
        isotropy_loss = -torch.linalg.slogdet(gram)[1] / rank
        loss = nce_loss + isotropy_weight * isotropy_loss

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            head.weight.div_(torch.linalg.matrix_norm(head.weight).clamp_min(1e-12))

    return head.weight.detach().cpu()


@torch.inference_mode()
def rollout_head_costs(
    model,
    encoded_state,
    candidate_actions: np.ndarray,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict rollouts for candidate_actions [N, H, A] and score endpoints under head L."""
    goal = goal_features[None].to(device=model.device, dtype=torch.float32)
    head_device = head.to(device=model.device, dtype=torch.float32)
    feature_dim = goal_features.shape[-1]
    all_costs = []

    for start in range(0, len(candidate_actions), batch_size):
        chunk_batch = candidate_actions[start : start + batch_size]
        action_tensor = torch.from_numpy(chunk_batch).permute(1, 0, 2).to(model.device)
        endpoint = model.unroll(encoded_state, action_tensor)["visual"][-1]
        terminal_features = endpoint.reshape(len(chunk_batch), -1, feature_dim).float()
        cost = linear_goal_distance(terminal_features, goal, head_device)
        all_costs.append(cost.cpu().numpy())

    return np.concatenate(all_costs)


@torch.inference_mode()
def trust_mpc_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    normals: np.ndarray,
    hold_chunk: np.ndarray,
    previous_plan: np.ndarray | None = None,
    *,
    num_elites: int = 8,
    action_clip: float = 2.0,
    batch_size: int = 64,
    stay_margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Jensen-certified mode-consistent CEM under an exact 2 * N rollout budget.

    Iteration 0 evaluates N candidates (anchoring slot 0 at zero-displacement
    hold and slot 1 at the shifted previous plan when available). Iteration 1
    evaluates N - 1 elite-centered samples and uses rollout #2N to certify the
    final elite mean. If Jensen's inequality holds (c(elite_mean) <= mean(c_elites)),
    the denoised elite mean is kept; otherwise elites straddle a contact barrier
    and only the dominant directional mode is averaged.
    """
    visual = torch.from_numpy(observation["visual"]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
    encoded_state = model.encode({"visual": visual, "proprio": proprio})

    _, samples_per_iter, horizon, action_dim = normals.shape
    hold_sequence = np.tile(hold_chunk[None, :], (horizon, 1)).astype(np.float32)

    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones_like(mean)

    iter0_actions = np.clip(mean[None] + std[None] * normals[0], -action_clip, action_clip).astype(np.float32)
    iter0_actions[0] = hold_sequence
    if previous_plan is not None and samples_per_iter > 1:
        shifted_plan = np.vstack([previous_plan[1:], hold_chunk[None, :]]).astype(np.float32)
        iter0_actions[1] = np.clip(shifted_plan, -action_clip, action_clip)

    iter0_costs = rollout_head_costs(model, encoded_state, iter0_actions, goal_features, head, batch_size)
    hold_cost = float(iter0_costs[0])

    iter0_elites = np.argsort(iter0_costs)[:num_elites]
    mean1 = iter0_actions[iter0_elites].mean(axis=0)
    std1 = iter0_actions[iter0_elites].std(axis=0)

    iter1_count = samples_per_iter - 1
    iter1_actions = np.clip(
        mean1[None] + std1[None] * normals[1, :iter1_count],
        -action_clip,
        action_clip,
    ).astype(np.float32)
    iter1_actions[0] = mean1
    iter1_costs = rollout_head_costs(model, encoded_state, iter1_actions, goal_features, head, batch_size)

    final_elites = np.argsort(iter1_costs)[:num_elites]
    elite_actions = iter1_actions[final_elites]
    elite_costs = iter1_costs[final_elites]
    elite_mean = np.clip(elite_actions.mean(axis=0), -action_clip, action_clip).astype(np.float32)

    mean_cost = float(
        rollout_head_costs(model, encoded_state, elite_mean[None, :, :], goal_features, head, batch_size)[0]
    )

    jensen_violation = mean_cost > float(elite_costs.mean())
    if not jensen_violation:
        selected_plan = elite_mean
        selected_cost = mean_cost
    else:
        lead_direction = elite_actions[0, 0] - hold_chunk
        alignment = np.sum((elite_actions[:, 0] - hold_chunk[None, :]) * lead_direction[None, :], axis=1)
        dominant_mode = elite_actions[alignment >= 0.0]
        selected_plan = (
            dominant_mode[0]
            if len(dominant_mode) <= 1
            else np.clip(dominant_mode.mean(axis=0), -action_clip, action_clip).astype(np.float32)
        )
        selected_cost = float(elite_costs[0])

    used_stay_gate = False
    if stay_margin > 0.0 and selected_cost > (1.0 - stay_margin) * hold_cost and float(elite_costs[0]) > (1.0 - stay_margin) * hold_cost:
        selected_plan = hold_sequence
        selected_cost = hold_cost
        used_stay_gate = True

    diagnostics = {
        "unrolls": int(len(iter0_actions) + len(iter1_actions) + 1),
        "hold_cost": hold_cost,
        "elite_mean_cost": mean_cost,
        "selected_cost": selected_cost,
        "jensen_violation": bool(jensen_violation),
        "used_stay_gate": used_stay_gate,
    }
    return selected_plan[0], selected_plan, diagnostics
