"""Unit tests for Jensen-certified MPC and spectral-isotropy head fitting."""

from __future__ import annotations

import numpy as np
import torch

from jepa_planning import linear_head as lh
from jepa_planning import trust_mpc as tm


class _ConvexToyModel:
    """Unimodal quadratic world model where averaging elites strictly helps."""

    def __init__(self, target_action: torch.Tensor):
        self.device = torch.device("cpu")
        self.target_action = target_action
        self.unroll_count = 0

    def encode(self, obs: dict) -> torch.Tensor:
        return torch.zeros(1, 1, 4, 8)

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        residual = (actions[-1] - self.target_action[None, :]).unsqueeze(1).repeat(1, 4, 1)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, : residual.shape[-1]] = residual.reshape(batch_size, 2, 2, -1)
        return {"visual": padded}


class _BimodalBarrierModel(_ConvexToyModel):
    """Symmetric two-mode world model (+1 or -1 succeeds, 0 hits a high-cost barrier)."""

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        # Minimum distance to either +target or -target; action = 0 sits at the barrier peak!
        d_pos = (actions[0] - self.target_action[None, :]).norm(dim=-1, keepdim=True)
        d_neg = (actions[0] + self.target_action[None, :]).norm(dim=-1, keepdim=True)
        barrier_residual = torch.minimum(d_pos, d_neg).unsqueeze(1).repeat(1, 4, 4)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, :4] = barrier_residual.reshape(batch_size, 2, 2, 4)
        return {"visual": padded}


def _dummy_obs() -> dict:
    return {"visual": np.zeros((16, 16, 3), dtype=np.uint8), "proprio": np.zeros(4, dtype=np.float32)}


def test_budget_parity_and_scale_invariance():
    target = torch.tensor([0.8, -0.6, 0.4, -0.2], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(0).standard_normal((2, 16, 2, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model_a = _ConvexToyModel(target)
    chunk_a, _, info_a = tm.trust_mpc_plan(model_a, _dummy_obs(), goal, head, normals, hold_chunk)

    model_b = _ConvexToyModel(target)
    chunk_b, _, info_b = tm.trust_mpc_plan(model_b, _dummy_obs(), goal, 6.0 * head, normals, hold_chunk)

    assert model_a.unroll_count == 32 and info_a["unrolls"] == 32
    assert info_a["jensen_violation"] is False
    assert np.allclose(chunk_a, chunk_b)


def test_jensen_barrier_splits_symmetric_bimodal_elites():
    target = torch.tensor([1.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    # Construct symmetric +/- noise so standard elite averaging cancels to 0 (the barrier peak).
    half = np.random.default_rng(2).standard_normal((2, 8, 2, 4)).astype(np.float32)
    normals = np.concatenate([half, -half], axis=1)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model = _BimodalBarrierModel(target)
    chunk, _, info = tm.trust_mpc_plan(model, _dummy_obs(), goal, head, normals, hold_chunk, stay_margin=0.0)
    assert info["jensen_violation"] is True
    assert abs(float(chunk[0])) > 0.4


def test_isotropy_regularizer_prevents_rank_one_collapse():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(12), 2)
    # Construct features with one dominant target axis plus three weaker target axes.
    latents = rng.standard_normal((12, 4)).astype(np.float32) * np.array([4.0, 1.0, 0.8, 0.6], dtype=np.float32)
    features = torch.zeros(24, 4, 16)
    for row, g in enumerate(groups):
        features[row, :, :4] = torch.from_numpy(latents[g])
        features[row, :, 4:8] = torch.randn(4, 4) * 2.0  # nuisance channels

    sampler = lh.AllocationSampler(groups, num_targets=12, hands_per_target=2, seed=0)
    head_iso = tm.fit_isotropy_head(features, sampler, steps=150, batch_size=16, isotropy_weight=0.05, rank=8)
    assert tm.effective_rank(head_iso) > 4.0
