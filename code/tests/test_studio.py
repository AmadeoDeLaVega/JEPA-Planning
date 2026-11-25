"""Fast CPU unit and HTTP integration tests for the Planning Studio."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import pytest
import torch

from jepa_planning import linear_head as lh
from jepa_planning.studio import (
    MetaWorldStudioSession,
    PlanningStudioBackend,
    PushTStudioSession,
    compose_duel_gif_bytes,
    create_studio_server,
    encode_png_base64,
    load_heads_directory,
    render_patch_heatmap_overlay,
)


class _MockPatchWorldModel:
    """Fast deterministic CPU world model operating on real 224x224 RGB frames.

    Downsamples RGB frames to a ``16x16`` patch grid with ``feature_dim = 16``:
    channels ``0:8`` emphasize chromatic/blue pusher signal while channels
    ``8:16`` capture luminance/structure. This lets tests verify real simulator
    renders, spatial patch heatmaps, calibrated-head selectivity, and exact
    rollout parity without loading DINO-WM checkpoints.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        controls_per_chunk: int = 2,
        step_action_dim: int = 2,
    ) -> None:
        self.device = torch.device("cpu")
        self.feature_dim = feature_dim
        self.action_dim = controls_per_chunk * step_action_dim
        self.unroll_count = 0
        if step_action_dim == 4:
            self.preprocessor = SimpleNamespace(
                action_mean=torch.tensor([0.1, -0.05, 0.0, 0.0], dtype=torch.float32),
                action_std=torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32),
            )
        else:
            self.preprocessor = SimpleNamespace(
                action_mean=torch.zeros(step_action_dim, dtype=torch.float32),
                action_std=torch.full((step_action_dim,), 20.0, dtype=torch.float32),
            )

    def encode(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = obs["visual"]  # [B, T, 3, H, W]
        batch_size, time_steps, _, _, _ = visual.shape
        flat = visual.reshape(batch_size * time_steps, 3, visual.shape[-2], visual.shape[-1]).float() / 255.0
        pooled = torch.nn.functional.adaptive_avg_pool2d(flat, (16, 16))  # [B*T, 3, 16, 16]
        r, g, b = pooled[:, 0], pooled[:, 1], pooled[:, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        blue_excess = (b - 0.5 * (r + g)).clamp_min(0.0)
        red_excess = (r - 0.5 * (g + b)).clamp_min(0.0)

        feats = torch.zeros(batch_size * time_steps, 16, 16, self.feature_dim, dtype=torch.float32)
        # Nuisance-heavy channels (blue pusher / colored arm chroma)
        feats[..., 0] = blue_excess * 4.0
        feats[..., 1] = (b - r) * 2.0
        feats[..., 2] = red_excess * 3.0
        feats[..., 3] = (r - g) * 2.0
        # Structure / object channels (grayscale contrast and spatial gradients)
        feats[..., 8] = luma * 3.0
        feats[..., 9] = (1.0 - luma) * 3.0
        feats[..., 10] = (g - 0.5 * b) * 2.0
        feats[..., 11] = torch.abs(luma - luma.mean(dim=(-2, -1), keepdim=True)) * 4.0

        patch_tokens = feats.reshape(batch_size, time_steps, 256, self.feature_dim)
        return {"visual": patch_tokens}

    def unroll(self, encoded_state: dict[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon, batch_size, _ = actions.shape
        self.unroll_count += int(batch_size)
        base = encoded_state["visual"][0, -1]  # [256, D]
        out = base[None, None].expand(horizon + 1, batch_size, 256, self.feature_dim).clone()
        # Smooth deterministic action-dependent latent displacement on task channels.
        delta_x = actions.mean(dim=0)[:, 0].clamp(-2.0, 2.0)  # [N]
        delta_y = actions.mean(dim=0)[:, 1].clamp(-2.0, 2.0)  # [N]
        out[-1, :, :, 8] += 0.25 * delta_x[:, None]
        out[-1, :, :, 9] += 0.25 * delta_y[:, None]
        out[-1, :, :, 0] += 0.15 * (delta_x + delta_y)[:, None]
        return {"visual": out}


def _structure_head(feature_dim: int = 16) -> torch.Tensor:
    """Mock calibrated linear head focusing on task/structure channels 8..11."""
    head = torch.zeros(4, feature_dim, dtype=torch.float32)
    head[0, 8] = 1.0
    head[1, 9] = 1.0
    head[2, 10] = 1.0
    head[3, 11] = 1.0
    return lh.normalized(head)


def _decode_data_uri(uri: str) -> Image.Image:
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def test_encode_png_base64_and_patch_heatmap_overlay():
    rgb = np.full((224, 224, 3), 180, dtype=np.uint8)
    uri = encode_png_base64(rgb)
    decoded = _decode_data_uri(uri)
    assert decoded.size == (224, 224)
    assert decoded.mode == "RGB"

    energy = np.zeros((16, 16), dtype=np.float32)
    energy[4:8, 10:14] = 10.0
    overlay = render_patch_heatmap_overlay(rgb, energy, alpha=0.6)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8

    # High-energy region should be warmer (higher red channel) than zero-energy background.
    hot_patch = overlay[70:98, 154:182, :].astype(np.float32)
    cold_patch = overlay[14:42, 14:42, :].astype(np.float32)
    assert float(hot_patch[..., 0].mean()) > float(cold_patch[..., 0].mean()) + 60.0

    # Zero-energy grid and explicit vmax work cleanly without NaNs.
    zero_overlay = render_patch_heatmap_overlay(rgb, np.zeros((16, 16), dtype=np.float32), vmax=2.5)
    assert zero_overlay.shape == (224, 224, 3)


def test_compose_duel_gif_bytes_valid_multiframe_gif():
    goal = np.full((224, 224, 3), 210, dtype=np.uint8)
    baseline_frames = [np.full((224, 224, 3), 40 + 30 * i, dtype=np.uint8) for i in range(4)]
    trust_frames = [np.full((224, 224, 3), 60 + 20 * i, dtype=np.uint8) for i in range(5)]

    gif_bytes = compose_duel_gif_bytes(baseline_frames, trust_frames, goal, fps=5)
    assert gif_bytes.startswith(b"GIF89a")

    with Image.open(io.BytesIO(gif_bytes)) as gif:
        assert gif.format == "GIF"
        assert getattr(gif, "n_frames", 1) == 5
        # 3 panels (224px each) + 2 gutters (6px) + 2 margins (8px) = 700 x 262
        assert gif.size == (700, 262)


def test_pusht_studio_session_inspect_and_duel_with_real_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    model = _MockPatchWorldModel(feature_dim=16, controls_per_chunk=2)
    heads = {
        "identity": lh.identity_head(16),
        "calibrated_iso": _structure_head(16),
    }
    session = PushTStudioSession(
        model,
        heads,
        controls_per_chunk=2,
        cem_horizon=2,
        cem_samples=8,
        cem_elites=3,
        cem_iterations=2,
        batch_size=8,
        default_head="calibrated_iso",
    )
    try:
        rgb, proprio = session.render_pusht_pose([145.0, 340.0], [230.0, 290.0, 0.35])
        assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
        assert proprio.shape == (4,) and proprio.dtype == np.float32

        inspected = session.inspect_patch_geometry(
            cur_agent=[145.0, 340.0],
            cur_block=[230.0, 290.0, 0.35],
            goal_agent=[435.0, 95.0],
            goal_block=[265.0, 250.0, 0.65],
            head_name="calibrated_iso",
        )
        assert inspected["head_name"] == "calibrated_iso"
        assert inspected["grid_shape"] == [16, 16]
        assert inspected["effective_rank"] == pytest.approx(4.0, abs=1e-3)
        assert inspected["plain_mse"]["nuisance_energy"] > 0.0
        assert inspected["plain_mse"]["task_energy"] > 0.0
        assert inspected["learned_head"]["task_to_nuisance_ratio"] > inspected["plain_mse"]["task_to_nuisance_ratio"]
        for key in ("cur_vs_goal", "nuisance", "task_shift"):
            assert _decode_data_uri(inspected["plain_mse"]["overlays"][key]).size == (224, 224)
            assert _decode_data_uri(inspected["learned_head"]["overlays"][key]).size == (224, 224)

        model.unroll_count = 0
        duel = session.run_duel(
            initial_agent=[145.0, 340.0],
            initial_block=[230.0, 290.0, 0.35],
            goal_agent=[435.0, 95.0],
            goal_block=[265.0, 250.0, 0.65],
            head_name="calibrated_iso",
            replans=2,
            seed=20261124,
        )
        assert duel["gif_ready"] is True
        expected_unrolls_per_planner = 2 * 2 * 8  # replans * cem_iterations * cem_samples
        assert duel["baseline"]["summary"]["unrolls"] == expected_unrolls_per_planner
        assert duel["trust_mpc"]["summary"]["unrolls"] == expected_unrolls_per_planner
        assert model.unroll_count == 2 * expected_unrolls_per_planner
        assert len(duel["baseline"]["frames"]) == 1 + 2 * 2
        assert len(duel["trust_mpc"]["frames"]) == 1 + 2 * 2
        assert len(duel["trust_mpc"]["replan_telemetry"]) == 2
        for row in duel["trust_mpc"]["replan_telemetry"]:
            for field in (
                "hold_cost",
                "elite_mean_cost",
                "selected_cost",
                "jensen_violation",
                "used_stay_gate",
                "position_error",
                "angle_error_deg",
            ):
                assert field in row

        gif_bytes = session.get_latest_duel_gif_bytes(fps=4)
        assert gif_bytes.startswith(b"GIF89a")
    finally:
        session.close()


def test_studio_http_server_roundtrip():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    model = _MockPatchWorldModel(feature_dim=16, controls_per_chunk=2)
    heads = {
        "identity": lh.identity_head(16),
        "calibrated_iso": _structure_head(16),
    }
    session = PushTStudioSession(
        model,
        heads,
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
        # GET /
        with urllib.request.urlopen(f"{base_url}/", timeout=5) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Visual World-Model Planning Studio" in html
            assert "/api/duel.gif" in html

        # GET /api/state
        with urllib.request.urlopen(f"{base_url}/api/state", timeout=5) as resp:
            assert resp.status == 200
            state_data = json.loads(resp.read().decode("utf-8"))
            assert state_data["pusht"]["default_head"] == "calibrated_iso"
            assert state_data["metaworld"]["enabled"] is False
            assert len(state_data["pusht"]["presets"]) >= 3

        # GET /api/duel.gif before any duel returns 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}/api/duel.gif", timeout=5)
        assert exc_info.value.code == 404

        # POST /api/inspect
        inspect_body = json.dumps(
            {
                "cur_agent": [145.0, 340.0],
                "cur_block": [230.0, 290.0, 0.35],
                "goal_agent": [435.0, 95.0],
                "goal_block": [265.0, 250.0, 0.65],
                "head_name": "calibrated_iso",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/inspect",
            data=inspect_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            inspect_data = json.loads(resp.read().decode("utf-8"))
            assert inspect_data["learned_head"]["overlays"]["cur_vs_goal"].startswith("data:image/png;base64,")

        # POST /api/duel
        duel_body = json.dumps(
            {
                "initial_agent": [145.0, 340.0],
                "initial_block": [230.0, 290.0, 0.35],
                "goal_agent": [435.0, 95.0],
                "goal_block": [265.0, 250.0, 0.65],
                "head_name": "calibrated_iso",
                "replans": 2,
                "seed": 20261124,
            }
        ).encode("utf-8")
        req_duel = urllib.request.Request(
            f"{base_url}/api/duel",
            data=duel_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_duel, timeout=10) as resp:
            assert resp.status == 200
            duel_data = json.loads(resp.read().decode("utf-8"))
            assert duel_data["gif_ready"] is True
            assert duel_data["baseline"]["summary"]["unrolls"] == duel_data["trust_mpc"]["summary"]["unrolls"]

        # GET /api/duel.gif after duel returns valid animated GIF
        with urllib.request.urlopen(f"{base_url}/api/duel.gif?fps=4", timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/gif"
            gif_bytes = resp.read()
            assert gif_bytes.startswith(b"GIF89a")
            with Image.open(io.BytesIO(gif_bytes)) as gif:
                assert getattr(gif, "n_frames", 1) == 5

        # POST /api/metaworld_inspect when disabled returns 503
        req_mw = urllib.request.Request(
            f"{base_url}/api/metaworld_inspect",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as mw_err:
            urllib.request.urlopen(req_mw, timeout=5)
        assert mw_err.value.code == 503

        # POST /api/metaworld_duel and GET /api/metaworld_duel.gif when disabled return 503
        req_mw_duel = urllib.request.Request(
            f"{base_url}/api/metaworld_duel",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as mw_duel_err:
            urllib.request.urlopen(req_mw_duel, timeout=5)
        assert mw_duel_err.value.code == 503

        with pytest.raises(urllib.error.HTTPError) as mw_gif_err:
            urllib.request.urlopen(f"{base_url}/api/metaworld_duel.gif", timeout=5)
        assert mw_gif_err.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()


def test_metaworld_studio_session_inspect_with_real_mujoco():
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    model = _MockPatchWorldModel(feature_dim=16, controls_per_chunk=2)
    heads = {
        "identity": lh.identity_head(16),
        "calibrated_iso": _structure_head(16),
    }
    mw_session = MetaWorldStudioSession(
        model,
        heads,
        warmup_steps=5,
        goal_steps=5,
        default_head="calibrated_iso",
    )
    try:
        result = mw_session.inspect_metaworld_task(
            seed=1000,
            hand_offset_angle_deg=135.0,
            shift_cm=5.0,
            head_name="calibrated_iso",
        )
        assert result["head_name"] == "calibrated_iso"
        assert result["effective_rank"] == pytest.approx(4.0, abs=1e-3)
        assert result["plain_mse"]["nuisance_energy"] > 0.0
        assert result["plain_mse"]["task_energy"] > 0.0
        assert _decode_data_uri(result["frames"]["goal"]).size == (224, 224)
        assert _decode_data_uri(result["learned_head"]["overlays"]["task_shift"]).size == (224, 224)
    finally:
        mw_session.close()


def test_metaworld_studio_duel_and_http_roundtrip_with_real_mujoco():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    pusht_model = _MockPatchWorldModel(feature_dim=16, controls_per_chunk=2, step_action_dim=2)
    mw_model = _MockPatchWorldModel(feature_dim=16, controls_per_chunk=2, step_action_dim=4)
    heads = {
        "identity": lh.identity_head(16),
        "calibrated_iso": _structure_head(16),
    }
    pusht_session = PushTStudioSession(
        pusht_model,
        heads,
        controls_per_chunk=2,
        cem_horizon=2,
        cem_samples=6,
        cem_elites=2,
        cem_iterations=2,
        batch_size=6,
        default_head="calibrated_iso",
    )
    mw_session = MetaWorldStudioSession(
        mw_model,
        heads,
        warmup_steps=5,
        goal_steps=5,
        controls_per_chunk=2,
        cem_samples=6,
        cem_elites=2,
        cem_iterations=2,
        batch_size=6,
        default_head="calibrated_iso",
    )
    backend = PlanningStudioBackend(pusht_session, metaworld_session=mw_session)
    server = create_studio_server("127.0.0.1", 0, backend, quiet=True)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    expected_unrolls = 1 * 2 * 6  # replans * cem_iterations * cem_samples
    client_errors: list[BaseException] = []

    def _run_http_checks() -> None:
        try:
            # GET /api/metaworld_duel.gif before any duel returns 404
            with pytest.raises(urllib.error.HTTPError) as pre_err:
                urllib.request.urlopen(f"{base_url}/api/metaworld_duel.gif", timeout=5)
            assert pre_err.value.code == 404

            # HTTP POST /api/metaworld_duel
            payload = json.dumps(
                {
                    "seed": 60000,
                    "hand_offset_angle_deg": 45.0,
                    "head_name": "calibrated_iso",
                    "horizon": 2,
                    "replans": 1,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/metaworld_duel",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                assert resp.status == 200
                duel_data = json.loads(resp.read().decode("utf-8"))
                assert duel_data["gif_ready"] is True
                assert duel_data["head_name"] == "calibrated_iso"
                assert duel_data["effective_rank"] == pytest.approx(4.0, abs=1e-3)
                assert duel_data["baseline"]["summary"]["unrolls"] == expected_unrolls
                assert duel_data["trust_mpc"]["summary"]["unrolls"] == expected_unrolls
                assert duel_data["trust_mpc"]["summary"]["final_error_3d_cm"] >= 0.0
                assert len(duel_data["baseline"]["frames"]) == 1 + 2
                assert len(duel_data["trust_mpc"]["frames"]) == 1 + 2
                assert _decode_data_uri(duel_data["goal_image"]).size == (224, 224)
                assert _decode_data_uri(duel_data["trust_mpc"]["frames"][-1]).size == (224, 224)
                for field in (
                    "error_3d_cm",
                    "error_xy_cm",
                    "hold_cost",
                    "elite_mean_cost",
                    "selected_cost",
                    "jensen_violation",
                    "used_stay_gate",
                ):
                    assert field in duel_data["trust_mpc"]["replan_telemetry"][0]

            # HTTP GET /api/metaworld_duel.gif returns valid animated GIF
            with urllib.request.urlopen(f"{base_url}/api/metaworld_duel.gif?fps=4", timeout=5) as resp:
                assert resp.status == 200
                assert resp.headers.get("Content-Type") == "image/gif"
                gif_bytes = resp.read()
                assert gif_bytes.startswith(b"GIF89a")
                with Image.open(io.BytesIO(gif_bytes)) as gif:
                    assert gif.format == "GIF"
                    assert getattr(gif, "n_frames", 1) == 3
                    assert gif.size == (700, 262)

            # GET /api/state reflects enabled Meta-World and cached duel GIF
            with urllib.request.urlopen(f"{base_url}/api/state", timeout=5) as resp:
                state_data = json.loads(resp.read().decode("utf-8"))
                assert state_data["metaworld"]["enabled"] is True
                assert state_data["metaworld"]["has_duel_gif"] is True
        except BaseException as exc:  # pragma: no cover - propagated to main thread
            client_errors.append(exc)
        finally:
            server.shutdown()

    try:
        # Serve on the main thread so macOS GLFW renders on its owning thread.
        client_thread = threading.Thread(target=_run_http_checks, daemon=True)
        client_thread.start()
        server.serve_forever(poll_interval=0.05)
        client_thread.join(timeout=15)
        assert not client_errors, f"HTTP client check failed: {client_errors[0]!r}"

        # Also verify direct session invocation and exact unroll budget accounting.
        mw_model.unroll_count = 0
        direct = mw_session.run_duel(
            seed=60000,
            hand_offset_angle_deg=45.0,
            head_name="calibrated_iso",
            horizon=2,
            replans=1,
        )
        assert direct["gif_ready"] is True
        assert mw_model.unroll_count == 2 * expected_unrolls
        assert mw_session.get_latest_duel_gif_bytes(fps=4).startswith(b"GIF89a")
    finally:
        server.server_close()
        backend.close()


def test_load_heads_directory_and_cli_parser(tmp_path):
    from scripts.run_planning_studio import build_parser

    saved = _structure_head(16)
    lh.save_head(tmp_path / "iso005_demo.pt", saved, {"rank": 4})
    heads, default_head = load_heads_directory(tmp_path, feature_dim=16)
    assert "identity" in heads
    assert "iso005_demo" in heads
    assert default_head == "iso005_demo"

    args = build_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--export-demo-gif",
            str(tmp_path / "out.gif"),
            "--export-metaworld-gif",
            str(tmp_path / "mw_out.gif"),
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert args.export_demo_gif == tmp_path / "out.gif"
    assert args.export_metaworld_gif == tmp_path / "mw_out.gif"
    assert args.metaworld_heads_dir.name == "metaworld_calibrated_heads"

