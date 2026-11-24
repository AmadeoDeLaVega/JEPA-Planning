"""Unit tests for trust-regularized MPC and local-sensitivity calibration."""

from __future__ import annotations

import numpy as np
import torch

from jepa_planning import trust_mpc as tm


class _ToyWorldModel:
    """Minimal deterministic stand-in for EncPredWM with countable unrolls."""

    def __init__(self, target_action: torch.Tensor):
        self.device = torch.device("cpu")
        self.target_action = target_action
        self.unroll_count = 0

    def encode(self, obs: dict) -> torch.Tensor:
        return torch.zeros(1, 1, 4, 8)

    def unroll(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # actions has shape [H, B, A]; endpoint feature encodes residual to target_action.
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        residual = (actions[-1] - self.target_action[None, :]).unsqueeze(1).repeat(1, 4, 1)
        padded = torch.zeros(horizon + 1, batch_size, 1, 2, 2, 8)
        padded[-1, :, 0, :, :, : residual.shape[-1]] = residual.reshape(batch_size, 2, 2, -1)
        return {"visual": padded}


def _dummy_observation() -> dict:
    return {
        "visual": np.zeros((16, 16, 3), dtype=np.uint8),
        "proprio": np.zeros(4, dtype=np.float32),
    }


def test_budget_parity_and_scale_invariance():
    target = torch.tensor([0.8, -0.6, 0.4, -0.2], dtype=torch.float32)
    goal = torch.zeros(4, 8)
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(0).standard_normal((2, 16, 2, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)

    model_a = _ToyWorldModel(target)
    chunk_a, _, info_a = tm.trust_mpc_plan(model_a, _dummy_observation(), goal, head, normals, hold_chunk)

    model_b = _ToyWorldModel(target)
    chunk_b, _, info_b = tm.trust_mpc_plan(model_b, _dummy_observation(), goal, 7.5 * head, normals, hold_chunk)

    # 2 iterations x 16 budget = 32 exact unrolls, invariant to scaling head L.
    assert model_a.unroll_count == 32
    assert info_a["unrolls"] == 32 and info_b["unrolls"] == 32
    assert np.allclose(chunk_a, chunk_b)


def test_hysteresis_stay_gate_holds_near_goal():
    # Target equals hold_chunk = 0, so no noisy sample can beat hold_cost by stay_margin.
    model = _ToyWorldModel(torch.zeros(4))
    goal = torch.ones(4, 8) * 0.1
    head = torch.eye(8) / np.sqrt(8.0)
    normals = np.random.default_rng(3).standard_normal((2, 12, 2, 4)).astype(np.float32)
    hold_chunk = np.zeros(4, dtype=np.float32)

    chosen_chunk, _, info = tm.trust_mpc_plan(
        model, _dummy_observation(), goal, head, normals, hold_chunk, stay_margin=0.05
    )
    assert info["used_stay_gate"] is True
    assert np.allclose(chosen_chunk, hold_chunk)


def test_select_calibrated_hyperparameters_prefers_local_sensitivity():
    rows = [
        {"tau": 0.3, "lr": 0.003, "top1": 0.995, "auroc": 1.0, "sensitivity_shift_px_10": 0.42},
        {"tau": 0.05, "lr": 0.01, "top1": 0.992, "auroc": 1.0, "sensitivity_shift_px_10": 1.15},
        {"tau": 0.01, "lr": 0.01, "top1": 0.850, "auroc": 0.99, "sensitivity_shift_px_10": 1.40},
    ]
    assert tm.select_calibrated_hyperparameters(rows, min_top1=0.98) == (0.05, 0.01)
