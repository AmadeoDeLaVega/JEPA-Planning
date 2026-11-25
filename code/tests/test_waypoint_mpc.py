"""Unit and integration tests for multi-stage visual waypoint Trust-MPC."""

from __future__ import annotations

import io
import json
import os
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.run_pusht_waypoint_demo import (
    build_parser,
    run_paired_waypoint_suite,
    sample_two_stage_pusht_tasks,
)
from jepa_planning import linear_head as lh
from jepa_planning.studio import (
    PUSHT_WAYPOINT_PRESETS,
    PlanningStudioBackend,
    PushTStudioSession,
    create_studio_server,
)
from jepa_planning.waypoint_mpc import (
    WaypointTracker,
    compose_waypoint_gif_bytes,
    waypoint_trust_mpc_step,
)


def _make_synthetic_feature(
    task_vec: list[float],
    nuisance_vec: list[float],
    *,
    num_patches: int = 8,
) -> torch.Tensor:
    """Build deterministic ``[P, D]`` features with task channels ``0:4`` and nuisance ``4:12``."""
    feat = torch.zeros(num_patches, 12, dtype=torch.float32)
    task_t = torch.tensor(task_vec, dtype=torch.float32)
    nuis_t = torch.tensor(nuisance_vec, dtype=torch.float32)
    # Slight patch-dependent modulation keeps patches non-degenerate while staying deterministic.
    patch_weights = torch.linspace(0.85, 1.15, num_patches, dtype=torch.float32).unsqueeze(1)
    feat[:, :4] = patch_weights * task_t.unsqueeze(0)
    feat[:, 4:] = patch_weights * nuis_t.unsqueeze(0)
    return feat


def _task_aligned_head(feature_dim: int = 12) -> torch.Tensor:
    """Normalized rank-4 linear head isolating task channels ``0:4``."""
    head = torch.zeros(4, feature_dim, dtype=torch.float32)
    head[0, 0] = 1.0
    head[1, 1] = 1.0
    head[2, 2] = 1.0
    head[3, 3] = 1.0
    return lh.normalized(head)


class _LinearLatentWorldModel:
    """Deterministic multi-step world model that records unrolled candidate batches.

    The state is stored in ``observation["proprio"][:4]`` (task coordinates)
    and ``observation["proprio"][4:12]`` (nuisance coordinates). Applying an
    action chunk advances the task channels linearly while injecting a
    nuisance perturbation.
    """

    def __init__(self, num_patches: int = 8, feature_dim: int = 12) -> None:
        self.device = torch.device("cpu")
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.unroll_count = 0
        self.recorded_action_batches: list[np.ndarray] = []

    def _state_to_tokens(self, state_vec: torch.Tensor) -> torch.Tensor:
        patch_weights = torch.linspace(0.85, 1.15, self.num_patches, dtype=torch.float32).unsqueeze(1)
        return patch_weights * state_vec.unsqueeze(0)  # [P, D]

    def encode(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        proprio = obs["proprio"][0, -1].float()  # [D]
        tokens = self._state_to_tokens(proprio)  # [P, D]
        return {"visual": tokens[None, None]}  # [1, 1, P, D]

    def unroll(self, encoded_state: dict[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # actions: [H, N, A] with A = 4 matching task channels 0:4
        horizon, batch_size, action_dim = actions.shape
        self.unroll_count += int(batch_size)
        self.recorded_action_batches.append(actions.detach().cpu().numpy().copy())

        base = encoded_state["visual"][0, -1]  # [P, D]
        out = base[None, None].expand(horizon + 1, batch_size, self.num_patches, self.feature_dim).clone()
        patch_weights = torch.linspace(0.85, 1.15, self.num_patches, dtype=torch.float32)[None, :, None]
        cur_delta = torch.zeros(batch_size, action_dim, dtype=actions.dtype)
        for step in range(horizon):
            cur_delta = cur_delta + 0.5 * actions[step]
            out[step + 1, :, :, :action_dim] = (
                base[None, :, :action_dim] + patch_weights * cur_delta[:, None, :]
            )
            # Nuisance channels shift with action norm to stress-test nuisance invariance.
            out[step + 1, :, :, action_dim:] = (
                base[None, :, action_dim:] + 0.2 * cur_delta[:, :1].unsqueeze(1)
            )
        return {"visual": out}


def _latent_obs(task_vec: np.ndarray, nuisance_vec: np.ndarray) -> dict[str, np.ndarray]:
    proprio = np.concatenate([task_vec, nuisance_vec]).astype(np.float32)
    return {
        "visual": np.zeros((16, 16, 3), dtype=np.uint8),
        "proprio": proprio,
    }


class _MockPushTWorldModel:
    """Fast CPU patch-feature world model compatible with real PushT 224x224 RGB observations."""

    def __init__(self, feature_dim: int = 16, controls_per_chunk: int = 2) -> None:
        self.device = torch.device("cpu")
        self.feature_dim = feature_dim
        self.action_dim = controls_per_chunk * 2
        self.unroll_count = 0
        self.preprocessor = SimpleNamespace(
            action_mean=torch.zeros(2, dtype=torch.float32),
            action_std=torch.full((2,), 20.0, dtype=torch.float32),
        )

    def encode(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = obs["visual"]
        batch_size, time_steps, _, _, _ = visual.shape
        flat = visual.reshape(batch_size * time_steps, 3, visual.shape[-2], visual.shape[-1]).float() / 255.0
        pooled = torch.nn.functional.adaptive_avg_pool2d(flat, (16, 16))
        r, g, b = pooled[:, 0], pooled[:, 1], pooled[:, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        blue_excess = (b - 0.5 * (r + g)).clamp_min(0.0)

        feats = torch.zeros(batch_size * time_steps, 16, 16, self.feature_dim, dtype=torch.float32)
        feats[..., 0] = blue_excess * 4.0
        feats[..., 1] = (b - r) * 2.0
        feats[..., 8] = luma * 3.0
        feats[..., 9] = (1.0 - luma) * 3.0
        feats[..., 10] = (g - 0.5 * b) * 2.0
        feats[..., 11] = torch.abs(luma - luma.mean(dim=(-2, -1), keepdim=True)) * 4.0
        return {"visual": feats.reshape(batch_size, time_steps, 256, self.feature_dim)}

    def unroll(self, encoded_state: dict[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        base = encoded_state["visual"][0, -1]
        out = base[None, None].expand(horizon + 1, batch_size, 256, self.feature_dim).clone()
        delta_x = actions.mean(dim=0)[:, 0].clamp(-2.0, 2.0)
        delta_y = actions.mean(dim=0)[:, 1].clamp(-2.0, 2.0)
        out[-1, :, :, 8] += 0.25 * delta_x[:, None]
        out[-1, :, :, 9] += 0.25 * delta_y[:, None]
        out[-1, :, :, 0] += 0.15 * (delta_x + delta_y)[:, None]
        return {"visual": out}


def _pusht_structure_head(feature_dim: int = 16) -> torch.Tensor:
    head = torch.zeros(4, feature_dim, dtype=torch.float32)
    head[0, 8] = 1.0
    head[1, 9] = 1.0
    head[2, 10] = 1.0
    head[3, 11] = 1.0
    return lh.normalized(head)


def test_distractor_stall_vs_invariant_handoff_theorem():
    """Plain MSE stalls on Stage 0 under distractor mismatch; calibrated head hands off cleanly."""
    # Start pose: object at [0, 0, 0, 0], pusher nuisance at A
    z_start = _make_synthetic_feature(
        task_vec=[0.0, 0.0, 0.0, 0.0],
        nuisance_vec=[2.0, -2.0, 2.0, -2.0, 1.5, -1.5, 1.5, -1.5],
    )
    # Waypoint 1 photo: object at [1.5, 0, 0, 0], distractor pusher at B (far from A and C)
    g_stage1 = _make_synthetic_feature(
        task_vec=[1.5, 0.0, 0.0, 0.0],
        nuisance_vec=[-2.5, 2.5, -2.5, 2.5, -2.0, 2.0, -2.0, 2.0],
    )
    # Waypoint 2 photo: object at [1.5, 2.0, 0, 0], distractor pusher at D
    g_stage2 = _make_synthetic_feature(
        task_vec=[1.5, 2.0, 0.0, 0.0],
        nuisance_vec=[0.5, 0.5, -0.5, -0.5, 2.5, -2.5, 0.0, 0.0],
    )
    # Arrival state: object reaches Waypoint 1 EXACTLY, while pusher sits at C != B.
    z_reached_w1 = _make_synthetic_feature(
        task_vec=[1.5, 0.0, 0.0, 0.0],
        nuisance_vec=[2.2, -2.2, 2.2, -2.2, 1.8, -1.8, 1.8, -1.8],
    )

    identity_head = lh.identity_head(12)
    aligned_head = _task_aligned_head(12)

    mse_tracker = WaypointTracker([g_stage1, g_stage2], identity_head, arrival_ratio=0.35)
    head_tracker = WaypointTracker([g_stage1, g_stage2], aligned_head, arrival_ratio=0.35)

    # Step 0: register stage-0 entry costs.
    mse_init = mse_tracker.update(z_start)
    head_init = head_tracker.update(z_start)
    assert mse_init["stage_index"] == 0 and mse_init["advanced"] is False
    assert head_init["stage_index"] == 0 and head_init["advanced"] is False

    # Step 1: object reaches Waypoint 1 with a different pusher nuisance vector.
    mse_step1 = mse_tracker.update(z_reached_w1)
    head_step1 = head_tracker.update(z_reached_w1)

    # Plain MSE remains dominated by the distractor nuisance mismatch and stalls on Stage 0.
    assert mse_step1["cost_ratio"] > 0.35
    assert mse_step1["advanced"] is False
    assert mse_tracker.stage_index == 0

    # Calibrated task-aligned head drops to ~0 and advances cleanly to Stage 1.
    assert head_step1["previous_stage_cost"] < 1e-6
    assert head_step1["advanced"] is True
    assert head_step1["trigger"] == "arrival_ratio"
    assert head_tracker.stage_index == 1
    assert head_tracker.is_final_stage is True
    assert head_step1["entry_cost"] == pytest.approx(
        head_tracker.current_waypoint_cost(z_reached_w1, stage_index=1),
        abs=1e-6,
    )
    assert len(head_tracker.transitions) == 1
    assert head_tracker.transitions[0]["from_stage"] == 0
    assert head_tracker.transitions[0]["to_stage"] == 1


def test_waypoint_tracker_multi_goal_bank_and_stay_gate_handoff():
    """Supports [K, P, D] waypoint banks and stay-gate handoff after positive progress."""
    aligned_head = _task_aligned_head(12)
    # Multi-goal bank [K=3, P=8, D=12] for Stage 1 with varying nuisance channels.
    g_stage1_bank = torch.stack(
        [
            _make_synthetic_feature([2.0, 0.0, 0.0, 0.0], [float(k), -float(k), 1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
            for k in range(3)
        ],
        dim=0,
    )
    g_stage2 = _make_synthetic_feature([2.0, 2.0, 0.0, 0.0], [0.0] * 8)

    tracker = WaypointTracker(
        [g_stage1_bank, g_stage2],
        aligned_head,
        arrival_ratio=0.35,
        allow_stay_gate_advance=True,
    )
    z0 = _make_synthetic_feature([0.0, 0.0, 0.0, 0.0], [3.0] * 8)
    tracker.update(z0)

    # No positive progress (cost ratio = 1.0 >= 0.85): stay gate must NOT advance.
    no_progress = tracker.update(z0, used_stay_gate=True)
    assert no_progress["advanced"] is False
    assert tracker.stage_index == 0

    # Moderate progress (task = 0.7 -> residual 1.3 vs 2.0 -> ratio = 1.69/4 = 0.4225 in (0.35, 0.85)):
    z_mid = _make_synthetic_feature([0.7, 0.0, 0.0, 0.0], [-4.0] * 8)
    without_stay = tracker.update(z_mid, used_stay_gate=False)
    assert without_stay["advanced"] is False
    assert tracker.stage_index == 0

    with_stay = tracker.update(z_mid, used_stay_gate=True)
    assert with_stay["advanced"] is True
    assert with_stay["trigger"] == "stay_gate"
    assert tracker.stage_index == 1


def test_exact_positive_scale_invariance_multi_step_rollout():
    """Multiplying head by 9.3 produces bit-for-bit identical transitions and actions."""
    base_head = _task_aligned_head(12)
    scaled_head = 9.3 * base_head

    waypoints = [
        _make_synthetic_feature([1.0, 0.0, 0.0, 0.0], [-2.0] * 8),
        _make_synthetic_feature([1.0, 1.5, 0.0, 0.0], [2.5] * 8),
    ]
    rng = np.random.default_rng(77)
    normals_seq = [rng.standard_normal((2, 16, 2, 4)).astype(np.float32) for _ in range(5)]
    hold_chunk = np.zeros(4, dtype=np.float32)

    def _run_rollout(head_matrix: torch.Tensor):
        model = _LinearLatentWorldModel(num_patches=8, feature_dim=12)
        tracker = WaypointTracker(waypoints, head_matrix, arrival_ratio=0.35, allow_stay_gate_advance=True)
        task_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        nuisance_state = np.array([1.2, -1.2, 0.8, -0.8, 1.5, -1.5, 0.4, -0.4], dtype=np.float32)
        previous_plan = None
        actions: list[np.ndarray] = []
        stages: list[int] = []
        advanced_flags: list[bool] = []
        triggers: list[str | None] = []
        stay_flags: list[bool] = []

        for step_idx, normals in enumerate(normals_seq):
            obs = _latent_obs(task_state, nuisance_state)
            chunk, previous_plan, diag = waypoint_trust_mpc_step(
                model,
                obs,
                tracker,
                normals,
                hold_chunk,
                previous_plan,
                num_elites=4,
                action_clip=2.0,
                batch_size=16,
                stay_margin=0.02,
                certify_barrier_split=True,
            )
            assert diag["unrolls"] == 32
            actions.append(chunk.copy())
            stages.append(int(diag["stage_index"]))
            advanced_flags.append(bool(diag["advanced"]))
            triggers.append(diag["trigger"])
            stay_flags.append(bool(diag["used_stay_gate"]))
            # Advance synthetic environment along the executed action.
            task_state = task_state + 0.65 * chunk
            nuisance_state = nuisance_state + np.float32(0.3 * (step_idx + 1))

        assert model.unroll_count == 5 * 32
        return actions, stages, advanced_flags, triggers, stay_flags

    actions_a, stages_a, adv_a, trig_a, stay_a = _run_rollout(base_head)
    actions_b, stages_b, adv_b, trig_b, stay_b = _run_rollout(scaled_head)

    # Confirm a real Stage 0 -> Stage 1 transition occurred during the rollout.
    assert stages_a[0] == 0
    assert stages_a[-1] == 1
    assert any(adv_a)

    # Bit-for-bit identical stage trajectory, transition step, flags, and actions.
    assert stages_a == stages_b
    assert adv_a == adv_b
    assert trig_a == trig_b
    assert stay_a == stay_b
    for chunk_a, chunk_b in zip(actions_a, actions_b):
        assert np.array_equal(chunk_a, chunk_b)


def test_warm_start_reset_on_handoff_and_preserved_within_stage():
    """previous_plan is preserved within the same stage and cleared on stage handoff."""
    head = _task_aligned_head(12)
    waypoints = [
        _make_synthetic_feature([1.0, 0.0, 0.0, 0.0], [0.0] * 8),
        _make_synthetic_feature([1.0, 2.0, 0.0, 0.0], [0.0] * 8),
    ]
    tracker = WaypointTracker(waypoints, head, arrival_ratio=0.35)
    model = _LinearLatentWorldModel(num_patches=8, feature_dim=12)

    hold_chunk = np.zeros(4, dtype=np.float32)
    normals = np.random.default_rng(19).standard_normal((2, 10, 2, 4)).astype(np.float32)

    # Step 0: far from Waypoint 1 -> registers entry cost, stays on Stage 0.
    obs_start = _latent_obs(np.array([0.0, 0.0, 0.0, 0.0]), np.zeros(8))
    _, _, diag0 = waypoint_trust_mpc_step(
        model,
        obs_start,
        tracker,
        normals,
        hold_chunk,
        previous_plan=None,
        num_elites=3,
        batch_size=10,
    )
    assert diag0["advanced"] is False
    assert diag0["warm_start_reset"] is False
    assert diag0["used_previous_plan"] is False

    # Step 1: still far from Waypoint 1 -> previous_plan MUST be injected into iter0_actions[1].
    distinctive_plan = np.array(
        [[1.4, -1.3, 0.9, -0.8], [1.7, -1.6, 1.1, -1.2]],
        dtype=np.float32,
    )
    expected_shifted = np.vstack([distinctive_plan[1:], hold_chunk[None, :]])

    model.recorded_action_batches.clear()
    obs_mid = _latent_obs(np.array([0.2, 0.0, 0.0, 0.0]), np.zeros(8))
    _, next_plan_same_stage, diag1 = waypoint_trust_mpc_step(
        model,
        obs_mid,
        tracker,
        normals,
        hold_chunk,
        previous_plan=distinctive_plan,
        num_elites=3,
        batch_size=10,
    )
    assert diag1["advanced"] is False
    assert diag1["warm_start_reset"] is False
    assert diag1["used_previous_plan"] is True
    # Recorded batch 0 is iter0_actions with shape [H, N, A]; candidate index 1 is [:, 1, :].
    iter0_slot1_same = model.recorded_action_batches[0][:, 1, :]
    np.testing.assert_allclose(iter0_slot1_same, expected_shifted, atol=1e-6)
    assert next_plan_same_stage.shape == (2, 4)

    # Step 2: arrive at Waypoint 1 -> advanced=True MUST clear previous_plan before planning!
    model.recorded_action_batches.clear()
    obs_arrived = _latent_obs(np.array([1.0, 0.0, 0.0, 0.0]), np.full(8, 3.0))
    _, next_plan_handoff, diag2 = waypoint_trust_mpc_step(
        model,
        obs_arrived,
        tracker,
        normals,
        hold_chunk,
        previous_plan=distinctive_plan,
        num_elites=3,
        batch_size=10,
    )
    assert diag2["advanced"] is True
    assert diag2["stage_index"] == 1
    assert diag2["warm_start_reset"] is True
    assert diag2["used_previous_plan"] is False
    iter0_slot1_reset = model.recorded_action_batches[0][:, 1, :]
    expected_fresh_sample = np.clip(normals[0, 1], -2.0, 2.0)
    np.testing.assert_allclose(iter0_slot1_reset, expected_fresh_sample, atol=1e-6)
    assert not np.allclose(iter0_slot1_reset, expected_shifted)
    assert next_plan_handoff.shape == (2, 4)


def test_compose_waypoint_gif_bytes_valid_gif89a_and_stage_switch():
    """compose_waypoint_gif_bytes emits valid GIF89a with a visible mid-episode subgoal switch."""
    # Distinct subgoal colors (Stage 0 = blue-tinted, Stage 1 = green-tinted).
    wp0 = np.zeros((224, 224, 3), dtype=np.uint8)
    wp0[..., 2] = 220
    wp1 = np.zeros((224, 224, 3), dtype=np.uint8)
    wp1[..., 1] = 220
    frames = [np.full((224, 224, 3), 50 + 25 * i, dtype=np.uint8) for i in range(6)]
    active_stages = [0, 0, 0, 1, 1, 1]

    gif_bytes = compose_waypoint_gif_bytes(
        frames,
        [wp0, wp1],
        active_stages,
        title_prefix="Trust-MPC",
        fps=5,
    )
    assert gif_bytes.startswith(b"GIF89a")

    with Image.open(io.BytesIO(gif_bytes)) as gif:
        assert gif.format == "GIF"
        assert getattr(gif, "n_frames", 1) == 6
        # 2 panels (224px) + gutter (8px) + margins (16px) = 472; header (34) + 224 + footer (26) = 284
        assert gif.size == (472, 284)

        gif.seek(0)
        frame0 = np.asarray(gif.convert("RGB"))
        gif.seek(4)
        frame4 = np.asarray(gif.convert("RGB"))

        # Left panel interior (subgoal photo) should visibly switch from blue (Stage 0) to green (Stage 1).
        left0 = frame0[80:180, 40:180].astype(np.float32)
        left4 = frame4[80:180, 40:180].astype(np.float32)
        assert float(left0[..., 2].mean()) > float(left0[..., 1].mean()) + 100.0
        assert float(left4[..., 1].mean()) > float(left4[..., 2].mean()) + 100.0

    with pytest.raises(ValueError, match="active_stages"):
        compose_waypoint_gif_bytes(frames, [wp0, wp1], [0, 1], "Bad")
    with pytest.raises(ValueError, match="out of range"):
        compose_waypoint_gif_bytes(frames[:2], [wp0, wp1], [0, 2], "Bad")


def test_demo_cli_tasks_paired_suite_and_studio_waypoint_endpoint(tmp_path):
    """Verify 2-stage PushT demo CLI helpers, paired suite GIF export, and Studio HTTP endpoint."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    tasks_a = sample_two_stage_pusht_tasks(num_tasks=3, seed=42)
    tasks_b = sample_two_stage_pusht_tasks(num_tasks=3, seed=42)
    assert tasks_a == tasks_b
    for task in tasks_a:
        s1_block = np.asarray(task["stage1_block"][:2])
        s1_agent = np.asarray(task["stage1_goal_agent"])
        s2_block = np.asarray(task["stage2_block"][:2])
        s2_agent = np.asarray(task["stage2_goal_agent"])
        assert float(np.linalg.norm(s1_agent - s1_block)) >= 85.0
        assert float(np.linalg.norm(s2_agent - s2_block)) >= 85.0

    parsed = build_parser().parse_args(
        ["--head-file", str(tmp_path / "head.pt"), "--tasks", "2", "--export-gif", str(tmp_path / "wp.gif")]
    )
    assert parsed.tasks == 2
    assert parsed.export_gif == tmp_path / "wp.gif"

    model = _MockPushTWorldModel(feature_dim=16, controls_per_chunk=2)
    head = _pusht_structure_head(16)
    session = PushTStudioSession(
        model,
        {"identity": lh.identity_head(16), "calibrated_iso": head},
        controls_per_chunk=2,
        cem_horizon=2,
        cem_samples=6,
        cem_elites=2,
        cem_iterations=2,
        batch_size=6,
        default_head="calibrated_iso",
    )
    backend = PlanningStudioBackend(session, metaworld_session=None)
    server = create_studio_server("127.0.0.1", 0, backend, quiet=True)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # Paired demo suite with real PushT env and GIF export.
        gif_path = tmp_path / "demo_waypoint.gif"
        rows, summary = run_paired_waypoint_suite(
            session.env,
            model,
            head,
            calibrated_head_name="calibrated_iso",
            num_tasks=1,
            replans=2,
            seed=20261124,
            cfg={
                "controls_per_chunk": 2,
                "cem_horizon": 2,
                "cem_samples": 6,
                "cem_elites": 2,
                "cem_iterations": 2,
                "batch_size": 6,
            },
            export_gif_path=gif_path,
            gif_fps=4,
        )
        assert len(rows) == 2
        assert rows[0]["unrolls"] == rows[1]["unrolls"] == 24
        assert "plain_mse_waypoint_cem" in summary
        assert "calibrated_waypoint_trust_mpc" in summary
        assert gif_path.is_file() and gif_path.read_bytes().startswith(b"GIF89a")

        # Studio GET /api/waypoint.gif before running returns 404.
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}/api/waypoint.gif", timeout=5)
        assert exc_info.value.code == 404

        # Studio POST /api/waypoint_chain + GET /api/waypoint.gif roundtrip.
        preset = PUSHT_WAYPOINT_PRESETS[0]
        body = json.dumps(
            {
                "initial_agent": preset["initial_agent"],
                "initial_block": preset["initial_block"],
                "waypoints": preset["waypoints"],
                "head_name": "calibrated_iso",
                "replans": 2,
                "arrival_ratio": 0.35,
                "seed": 20261124,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/waypoint_chain",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["gif_ready"] is True
            assert data["num_waypoints"] == 2
            assert len(data["waypoint_images"]) == 2
            assert data["baseline"]["summary"]["unrolls"] == data["trust_mpc"]["summary"]["unrolls"] == 24

        with urllib.request.urlopen(f"{base_url}/api/waypoint.gif?fps=4", timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/gif"
            wp_gif = resp.read()
            assert wp_gif.startswith(b"GIF89a")
            with Image.open(io.BytesIO(wp_gif)) as img:
                assert getattr(img, "n_frames", 1) == 1 + 2 * 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()
