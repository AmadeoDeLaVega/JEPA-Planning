"""Interactive visual world-model planning studio for PushT and Meta-World Push.

Provides dependency-injected sessions, spatial 16x16 patch-energy heatmap
rendering, side-by-side animated GIF export, and a zero-dependency HTTP server
serving a single-page web interface for exploring frozen DINO-WM representations
and Jensen-certified Trust-MPC closed-loop duels.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from jepa_planning import linear_head as lh
from jepa_planning import metaworld_planning as mwp
from jepa_planning import pusht_head_planning as hp
from jepa_planning import trust_mpc as tm
from jepa_planning.calibrated_objectives import patch_energy_map
from jepa_planning.pusht_multi_goal import cem_standard_normals, state_from
from jepa_planning.waypoint_mpc import (
    WaypointTracker,
    compose_waypoint_gif_bytes,
    encode_visual_features,
    waypoint_trust_mpc_step,
)

# --------------------------------------------------------------------- presets

PUSHT_ARENA_SIZE = 512.0

PUSHT_PRESETS: list[dict[str, object]] = [
    {
        "id": "corner_distractor",
        "name": "Corner Distractor Trap",
        "description": (
            "Goal photo places the blue pusher in the opposite corner while the "
            "gray T-block sits nearby. Plain MSE chases the distractor pusher; "
            "the calibrated head stays locked on the T-block."
        ),
        "initial_agent": [145.0, 340.0],
        "initial_block": [230.0, 290.0, 0.35],
        "goal_agent": [435.0, 95.0],
        "goal_block": [265.0, 250.0, 0.65],
        "replans": 4,
        "seed": 20261124,
    },
    {
        "id": "bimodal_stem_split",
        "name": "Bimodal Stem Split",
        "description": (
            "Pusher starts directly below the T stem where left and right sweeps "
            "form competing modes. Trust-MPC detects Jensen barrier violations "
            "instead of averaging across the stem."
        ),
        "initial_agent": [256.0, 395.0],
        "initial_block": [256.0, 295.0, 0.0],
        "goal_agent": [105.0, 115.0],
        "goal_block": [256.0, 220.0, 1.57],
        "replans": 4,
        "seed": 20261124,
    },
    {
        "id": "already_solved",
        "name": "Already Solved (Stay-Gate Test)",
        "description": (
            "The T-block is already aligned with the target while only the "
            "distractor pusher sits across the arena, testing whether the "
            "stay-gate suppresses post-peak drift."
        ),
        "initial_agent": [120.0, 130.0],
        "initial_block": [256.0, 256.0, 0.78],
        "goal_agent": [410.0, 390.0],
        "goal_block": [256.0, 256.0, 0.78],
        "replans": 4,
        "seed": 20261124,
    },
]

PUSHT_WAYPOINT_PRESETS: list[dict[str, object]] = [
    {
        "id": "corner_waypoint_chain",
        "name": "L-Turn Corner Chain",
        "description": (
            "Two-stage around-the-corner push with independent distractor pushers "
            "in each subgoal photo. Tests purely visual Stage 1 -> Stage 2 handoff "
            "and warm-start reset without simulator coordinates."
        ),
        "initial_agent": [145.0, 355.0],
        "initial_block": [210.0, 315.0, 0.25],
        "waypoints": [
            {"block": [255.0, 265.0, 0.65], "goal_agent": [425.0, 105.0]},
            {"block": [315.0, 285.0, 1.15], "goal_agent": [95.0, 115.0]},
        ],
        "replans": 6,
        "arrival_ratio": 0.35,
        "seed": 20261124,
    },
    {
        "id": "distractor_relay_chain",
        "name": "Distractor Relay Chain",
        "description": (
            "Stage 1 and Stage 2 place distractor pushers in opposite corners so "
            "plain MSE stalls chasing the Stage-1 distractor while the calibrated "
            "head hands off smoothly to Stage 2."
        ),
        "initial_agent": [370.0, 360.0],
        "initial_block": [305.0, 305.0, 2.40],
        "waypoints": [
            {"block": [256.0, 256.0, 1.85], "goal_agent": [95.0, 415.0]},
            {"block": [205.0, 200.0, 1.25], "goal_agent": [420.0, 95.0]},
        ],
        "replans": 6,
        "arrival_ratio": 0.35,
        "seed": 20261125,
    },
]

# Warm dark-slate -> amber -> orange -> crimson -> gold colormap control points.
_HEATMAP_STOPS = np.array([0.0, 0.25, 0.55, 0.80, 1.0], dtype=np.float32)
_HEATMAP_COLORS = np.array(
    [
        [24.0, 26.0, 30.0],
        [118.0, 46.0, 18.0],
        [218.0, 118.0, 26.0],
        [224.0, 50.0, 44.0],
        [255.0, 222.0, 108.0],
    ],
    dtype=np.float32,
)


# --------------------------------------------------------- image & gif helpers


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous ``[H, W, 3]`` uint8 RGB array."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected RGB image of shape [H, W, 3], got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def encode_png_base64(rgb_uint8: np.ndarray) -> str:
    """Encode an ``[H, W, 3]`` uint8 RGB image as a base64 PNG data URI."""
    arr = _as_rgb_uint8(rgb_uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _apply_warm_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map ``[H, W]`` intensities in ``[0, 1]`` to float32 RGB in ``[0, 255]``."""
    flat = np.clip(normalized, 0.0, 1.0).ravel()
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, _HEATMAP_STOPS, _HEATMAP_COLORS[:, channel])
    return rgb.reshape(*normalized.shape, 3)


def render_patch_heatmap_overlay(
    rgb_uint8: np.ndarray,
    energy_grid: np.ndarray,
    alpha: float = 0.58,
    vmax: float | None = None,
) -> np.ndarray:
    """Overlay a patch-energy grid on an RGB frame with crisp patch boundaries.

    Args:
        rgb_uint8: Base RGB image of shape ``[H, W, 3]`` (typically ``224x224``).
        energy_grid: Nonnegative per-patch energies of shape ``[G, G]`` (typically
            ``[16, 16]``) or flat square ``[P]``.
        alpha: Maximum blending strength in ``[0, 1]``.
        vmax: Optional upper scale for normalization; defaults to ``max(energy_grid)``.

    Returns:
        Blended ``[H, W, 3]`` uint8 RGB array.
    """
    base_rgb = _as_rgb_uint8(rgb_uint8)
    height, width, _ = base_rgb.shape
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")

    grid = np.asarray(energy_grid, dtype=np.float32)
    if grid.ndim == 1:
        side = math.isqrt(grid.size)
        if side * side != grid.size or side == 0:
            raise ValueError("1D energy_grid must have a positive square length")
        grid = grid.reshape(side, side)
    elif grid.ndim != 2 or min(grid.shape) == 0:
        raise ValueError(f"energy_grid must be 2D [G, G], got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("energy_grid must contain only finite values")

    clipped_grid = np.maximum(grid, 0.0)
    if vmax is not None:
        if isinstance(vmax, bool) or not np.isfinite(vmax) or float(vmax) <= 0.0:
            raise ValueError("vmax must be a positive finite number when provided")
        scale = float(vmax)
    else:
        scale = float(clipped_grid.max())

    if scale > 0.0:
        norm_grid = np.clip(clipped_grid / scale, 0.0, 1.0)
    else:
        norm_grid = np.zeros_like(clipped_grid, dtype=np.float32)

    # Smoothly upscale patch intensities to image resolution.
    grid_img = Image.fromarray((norm_grid * 255.0).astype(np.uint8), mode="L")
    upscaled_norm = (
        np.asarray(grid_img.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )
    heat_rgb = _apply_warm_colormap(upscaled_norm)

    # Desaturate and slightly dim the base frame so warm energy peaks pop clearly.
    base_f = base_rgb.astype(np.float32)
    luma = 0.299 * base_f[..., 0] + 0.587 * base_f[..., 1] + 0.114 * base_f[..., 2]
    dimmed_base = (0.72 * luma[..., None] + 0.28 * base_f) * 0.72

    local_alpha = np.clip(float(alpha) * (0.20 + 0.80 * upscaled_norm), 0.0, 1.0)[..., None]
    blended = dimmed_base * (1.0 - local_alpha) + heat_rgb * local_alpha

    # Draw subtle patch grid lines (14x14 px cells for 16x16 patches on 224x224).
    rows, cols = grid.shape
    line_color = np.array([46.0, 50.0, 56.0], dtype=np.float32)
    line_blend = 0.42
    for row in range(1, rows):
        y = int(round(row * height / rows))
        if 0 <= y < height:
            blended[y, :, :] = blended[y, :, :] * (1.0 - line_blend) + line_color * line_blend
    for col in range(1, cols):
        x = int(round(col * width / cols))
        if 0 <= x < width:
            blended[:, x, :] = blended[:, x, :] * (1.0 - line_blend) + line_color * line_blend

    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def compose_duel_gif_bytes(
    baseline_frames: list[np.ndarray],
    trust_frames: list[np.ndarray],
    goal_frame: np.ndarray,
    fps: int = 4,
) -> bytes:
    """Stitch ``[Goal | Plain MSE + CEM | Calibrated Head + Trust-MPC]`` into a GIF."""
    if not baseline_frames or not trust_frames:
        raise ValueError("baseline_frames and trust_frames must be non-empty")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")

    goal_arr = _as_rgb_uint8(goal_frame)
    panel_size = 224
    goal_pil = Image.fromarray(goal_arr, mode="RGB").resize(
        (panel_size, panel_size), resample=Image.Resampling.BILINEAR
    )

    total_frames = max(len(baseline_frames), len(trust_frames))
    header_h = 30
    gutter = 6
    margin = 8
    canvas_w = margin * 2 + panel_size * 3 + gutter * 2
    canvas_h = header_h + panel_size + margin

    bg_color = (30, 32, 34)
    header_border = (54, 58, 64)
    titles = [
        ("Goal Photo", (166, 176, 190)),
        ("Plain MSE + CEM", (224, 112, 96)),
        ("Calibrated Head + Trust-MPC", (98, 206, 148)),
    ]
    font = ImageFont.load_default()

    stitched_frames: list[Image.Image] = []
    for idx in range(total_frames):
        base_arr = _as_rgb_uint8(baseline_frames[min(idx, len(baseline_frames) - 1)])
        trust_arr = _as_rgb_uint8(trust_frames[min(idx, len(trust_frames) - 1)])

        base_pil = Image.fromarray(base_arr, mode="RGB").resize(
            (panel_size, panel_size), resample=Image.Resampling.BILINEAR
        )
        trust_pil = Image.fromarray(trust_arr, mode="RGB").resize(
            (panel_size, panel_size), resample=Image.Resampling.BILINEAR
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, header_h - 1), (canvas_w, header_h - 1)], fill=header_border, width=1)

        panels = [goal_pil, base_pil, trust_pil]
        for col, (panel, (label, text_color)) in enumerate(zip(panels, titles)):
            x0 = margin + col * (panel_size + gutter)
            canvas.paste(panel, (x0, header_h))
            draw.text((x0 + 4, 9), label, fill=text_color, font=font)

        step_badge = f"t={idx}/{total_frames - 1}"
        draw.text((canvas_w - margin - 52, 9), step_badge, fill=(154, 162, 172), font=font)
        stitched_frames.append(canvas)

    buffer = io.BytesIO()
    duration_ms = max(20, int(round(1000.0 / fps)))
    stitched_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=stitched_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()


# ----------------------------------------------------------- shared model ops


def _encode_patch_features(model, rgb: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
    """Encode a single RGB + proprio observation into CPU float32 ``[P, D]`` features."""
    visual = torch.from_numpy(_as_rgb_uint8(rgb)).permute(2, 0, 1)[None, None]
    proprio_t = torch.from_numpy(np.asarray(proprio, dtype=np.float32))[None, None]
    with torch.inference_mode():
        encoded = model.encode({"visual": visual, "proprio": proprio_t})
    vis = encoded["visual"] if isinstance(encoded, dict) else encoded
    if vis.ndim >= 4:
        vis = vis[0, -1]
    elif vis.ndim == 3:
        vis = vis[0]
    return vis.reshape(-1, vis.shape[-1]).detach().cpu().float()


@torch.inference_mode()
def standard_cem_plan(
    model,
    observation: dict,
    goal_features: torch.Tensor,
    head: torch.Tensor,
    normals: np.ndarray,
    *,
    num_elites: int = 8,
    action_clip: float = 2.0,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Standard CEM planner under head ``L`` matching the ``2 * N`` unroll budget."""
    visual = torch.from_numpy(_as_rgb_uint8(observation["visual"])).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
    encoded = model.encode({"visual": visual, "proprio": proprio})

    iterations, samples_per_iter, horizon, action_dim = normals.shape
    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones_like(mean)
    best_cost = float("nan")
    elite_mean_cost = float("nan")
    total_unrolls = 0

    for iteration in range(iterations):
        actions = np.clip(
            mean[None] + std[None] * normals[iteration],
            -action_clip,
            action_clip,
        ).astype(np.float32)
        actions[0] = mean
        costs = tm.rollout_head_costs(model, encoded, actions, goal_features, head, batch_size=batch_size)
        total_unrolls += len(actions)
        elites = np.argsort(costs)[:num_elites]
        mean = actions[elites].mean(axis=0).astype(np.float32)
        std = actions[elites].std(axis=0).astype(np.float32)
        best_cost = float(costs[elites[0]])
        elite_mean_cost = float(costs[elites].mean())

    return np.clip(mean[0], -action_clip, action_clip).astype(np.float32), {
        "unrolls": int(total_unrolls),
        "best_cost": best_cost,
        "elite_mean_cost": elite_mean_cost,
    }


def load_heads_directory(
    heads_dir: Path | None,
    feature_dim: int = lh.FEATURE_DIM,
) -> tuple[dict[str, torch.Tensor], str]:
    """Load saved ``.pt`` goal heads from ``heads_dir`` alongside plain-MSE identity."""
    heads: dict[str, torch.Tensor] = {"identity": lh.identity_head(feature_dim)}
    if heads_dir is not None:
        directory = Path(heads_dir)
        search_dirs = [directory]
        if (directory / "heads").is_dir():
            search_dirs.append(directory / "heads")
        for folder in search_dirs:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.pt")):
                matrix, _ = lh.load_head(path)
                heads[path.stem] = matrix.float()

    preferred = "iso005_tau03_2k_B1024_M256_K4_r0"
    if preferred in heads:
        default_head = preferred
    else:
        iso_heads = [name for name in heads if name != "identity" and "iso" in name]
        non_identity = [name for name in heads if name != "identity"]
        default_head = iso_heads[0] if iso_heads else (non_identity[0] if non_identity else "identity")
    return heads, default_head


def _head_catalog_metadata(heads: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    """Return serializable rank and shape summaries for every loaded head."""
    rows: list[dict[str, object]] = []
    for name, matrix in heads.items():
        rows.append(
            {
                "name": name,
                "rank": int(matrix.shape[0]),
                "feature_dim": int(matrix.shape[1]),
                "effective_rank": round(tm.effective_rank(matrix), 2),
            }
        )
    return rows


# -------------------------------------------------------------- PushT session


def _make_pusht_env():
    """Construct the headless PushT environment matching the DINO-WM checkpoint."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    env = PushTEnv(render_size=224, with_target=True, with_velocity=True)
    env.seed(0)
    return env


def _choose_counterfactual_agent(goal_agent: np.ndarray, goal_block: np.ndarray) -> np.ndarray:
    """Pick a well-separated counterfactual pusher position clear of the T-block."""
    reflected = np.array([PUSHT_ARENA_SIZE - goal_agent[0], PUSHT_ARENA_SIZE - goal_agent[1]], dtype=np.float64)
    candidates = [
        np.clip(reflected, 55.0, 457.0),
        np.array([75.0, 75.0]),
        np.array([435.0, 75.0]),
        np.array([435.0, 435.0]),
        np.array([75.0, 435.0]),
    ]
    best = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        dist_agent = float(np.linalg.norm(candidate - goal_agent))
        dist_block = float(np.linalg.norm(candidate - goal_block[:2]))
        if dist_block < 95.0:
            continue
        score = min(dist_agent, 250.0) + 0.25 * min(dist_block, 200.0)
        if score > best_score:
            best_score = score
            best = candidate
    return best.astype(np.float32)


def _choose_shifted_block(goal_block: np.ndarray, goal_agent: np.ndarray, shift_px: float = 20.0) -> np.ndarray:
    """Shift the T-block by ``shift_px`` while avoiding bounds and the goal pusher."""
    offsets = [
        np.array([shift_px, 0.0]),
        np.array([-shift_px, 0.0]),
        np.array([0.0, shift_px]),
        np.array([0.0, -shift_px]),
    ]
    for delta in offsets:
        candidate_xy = goal_block[:2] + delta
        if 80.0 <= candidate_xy[0] <= 432.0 and 80.0 <= candidate_xy[1] <= 432.0:
            if float(np.linalg.norm(candidate_xy - goal_agent)) > 80.0:
                return np.array([candidate_xy[0], candidate_xy[1], goal_block[2]], dtype=np.float32)
    candidate_xy = np.clip(goal_block[:2] + offsets[0], 80.0, 432.0)
    return np.array([candidate_xy[0], candidate_xy[1], goal_block[2]], dtype=np.float32)


class PushTStudioSession:
    """Interactive 2D PushT inspection and paired Trust-MPC duel backend."""

    def __init__(
        self,
        model,
        heads: dict[str, torch.Tensor],
        env=None,
        *,
        controls_per_chunk: int = 5,
        cem_horizon: int = 2,
        cem_samples: int = 64,
        cem_elites: int = 8,
        cem_iterations: int = 2,
        action_clip: float = 2.0,
        batch_size: int = 64,
        stay_margin: float = 0.02,
        default_head: str | None = None,
    ) -> None:
        if not heads:
            raise ValueError("heads dictionary must contain at least one head")
        self.model = model
        self.heads = {name: mat.detach().cpu().float() for name, mat in heads.items()}
        first_head = next(iter(self.heads.values()))
        self.feature_dim = int(first_head.shape[1])
        if "identity" not in self.heads:
            self.heads["identity"] = lh.identity_head(self.feature_dim)

        if default_head is not None and default_head in self.heads:
            self.default_head = default_head
        else:
            non_identity = [name for name in self.heads if name != "identity"]
            self.default_head = non_identity[0] if non_identity else "identity"

        self._owns_env = env is None
        self.env = env if env is not None else _make_pusht_env()
        self.controls_per_chunk = int(controls_per_chunk)
        self.cem_horizon = int(cem_horizon)
        self.cem_samples = int(cem_samples)
        self.cem_elites = int(cem_elites)
        self.cem_iterations = int(cem_iterations)
        self.action_clip = float(action_clip)
        self.batch_size = int(batch_size)
        self.stay_margin = float(stay_margin)
        self.action_dim = int(getattr(model, "action_dim", self.controls_per_chunk * 2))

        preprocessor = getattr(model, "preprocessor", None)
        if preprocessor is not None and hasattr(preprocessor, "action_mean") and hasattr(preprocessor, "action_std"):
            self.action_mean = np.asarray(preprocessor.action_mean.detach().cpu().numpy(), dtype=np.float32)
            self.action_std = np.asarray(preprocessor.action_std.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.action_mean = np.zeros(2, dtype=np.float32)
            self.action_std = np.ones(2, dtype=np.float32)

        self._lock = threading.Lock()
        self._latest_duel_frames: tuple[list[np.ndarray], list[np.ndarray], np.ndarray] | None = None
        self._latest_waypoint_gif_data: tuple[list[np.ndarray], list[np.ndarray], list[int], str] | None = None

    def close(self) -> None:
        """Release simulator resources owned by this session."""
        if self._owns_env and self.env is not None:
            self.env.close()

    def _resolve_head(self, head_name: str | None) -> tuple[str, torch.Tensor]:
        name = self.default_head if not head_name else str(head_name)
        if name not in self.heads:
            raise KeyError(f"Unknown PushT head '{name}'. Available: {sorted(self.heads)}")
        return name, self.heads[name]

    def _render_pusht_pose_unlocked(
        self,
        agent_xy: Sequence[float],
        block_xytheta: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        agent = np.asarray(agent_xy, dtype=np.float32)
        block = np.asarray(block_xytheta, dtype=np.float32)
        if agent.shape != (2,) or block.shape != (3,):
            raise ValueError("agent_xy must have shape [2] and block_xytheta must have shape [3]")
        if not np.all(np.isfinite(agent)) or not np.all(np.isfinite(block)):
            raise ValueError("PushT pose coordinates must be finite")

        clamped_agent = np.clip(agent, 30.0, 482.0)
        clamped_block = np.array(
            [
                float(np.clip(block[0], 70.0, 442.0)),
                float(np.clip(block[1], 70.0, 442.0)),
                float(block[2] % (2.0 * np.pi)),
            ],
            dtype=np.float32,
        )
        self.env.reset_to_state = state_from(clamped_agent, clamped_block)
        observation, _ = self.env.reset()
        rgb = _as_rgb_uint8(observation["visual"]).copy()
        proprio = np.asarray(observation["proprio"], dtype=np.float32).copy()
        return rgb, proprio

    def render_pusht_pose(
        self,
        agent_xy: Sequence[float],
        block_xytheta: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render any custom PushT pose into ``(rgb [224,224,3], proprio [4])``."""
        with self._lock:
            return self._render_pusht_pose_unlocked(agent_xy, block_xytheta)

    def inspect_patch_geometry(
        self,
        cur_agent: Sequence[float],
        cur_block: Sequence[float],
        goal_agent: Sequence[float],
        goal_block: Sequence[float],
        head_name: str | None = None,
    ) -> dict[str, object]:
        """Compare 16x16 patch energies under Plain MSE and a learned linear head."""
        resolved_name, head_matrix = self._resolve_head(head_name)
        cur_agent_arr = np.asarray(cur_agent, dtype=np.float32)
        cur_block_arr = np.asarray(cur_block, dtype=np.float32)
        goal_agent_arr = np.asarray(goal_agent, dtype=np.float32)
        goal_block_arr = np.asarray(goal_block, dtype=np.float32)

        nuisance_agent = _choose_counterfactual_agent(goal_agent_arr, goal_block_arr)
        shifted_block = _choose_shifted_block(goal_block_arr, goal_agent_arr, shift_px=20.0)

        with self._lock:
            cur_rgb, cur_prop = self._render_pusht_pose_unlocked(cur_agent_arr, cur_block_arr)
            goal_rgb, goal_prop = self._render_pusht_pose_unlocked(goal_agent_arr, goal_block_arr)
            nuis_rgb, nuis_prop = self._render_pusht_pose_unlocked(nuisance_agent, goal_block_arr)
            shift_rgb, shift_prop = self._render_pusht_pose_unlocked(goal_agent_arr, shifted_block)

            f_cur = _encode_patch_features(self.model, cur_rgb, cur_prop)
            f_goal = _encode_patch_features(self.model, goal_rgb, goal_prop)
            f_nuis = _encode_patch_features(self.model, nuis_rgb, nuis_prop)
            f_shift = _encode_patch_features(self.model, shift_rgb, shift_prop)

        mse_cur_grid = patch_energy_map(f_cur, f_goal, head=None).numpy()
        mse_nuis_grid = patch_energy_map(f_goal, f_nuis, head=None).numpy()
        mse_task_grid = patch_energy_map(f_goal, f_shift, head=None).numpy()

        head_cur_grid = patch_energy_map(f_cur, f_goal, head=head_matrix).numpy()
        head_nuis_grid = patch_energy_map(f_goal, f_nuis, head=head_matrix).numpy()
        head_task_grid = patch_energy_map(f_goal, f_shift, head=head_matrix).numpy()

        mse_nuis_mean = float(mse_nuis_grid.mean())
        mse_task_mean = float(mse_task_grid.mean())
        mse_ratio = float(mse_task_mean / max(mse_nuis_mean, 1e-12))

        head_nuis_mean = float(head_nuis_grid.mean())
        head_task_mean = float(head_task_grid.mean())
        head_ratio = float(head_task_mean / max(head_nuis_mean, 1e-12))

        # Share vmax across each metric's nuisance/task pair for fair visual contrast.
        mse_pair_vmax = max(float(mse_nuis_grid.max()), float(mse_task_grid.max()), 1e-12)
        head_pair_vmax = max(float(head_nuis_grid.max()), float(head_task_grid.max()), 1e-12)

        phys = hp.physical_metrics(cur_block_arr[None, :], goal_block_arr)

        return {
            "head_name": resolved_name,
            "effective_rank": float(tm.effective_rank(head_matrix)),
            "grid_shape": list(mse_cur_grid.shape),
            "physical": {
                "position_error": float(phys["position_error"][0]),
                "angle_error_deg": float(phys["angle_error_deg"][0]),
                "combined_error": float(phys["combined_error"][0]),
                "coverage": float(phys["coverage"][0]),
                "agent_distance_px": float(np.linalg.norm(cur_agent_arr - goal_agent_arr)),
                "nuisance_agent_shift_px": float(np.linalg.norm(nuisance_agent - goal_agent_arr)),
                "task_block_shift_px": float(np.linalg.norm(shifted_block[:2] - goal_block_arr[:2])),
            },
            "frames": {
                "current": encode_png_base64(cur_rgb),
                "goal": encode_png_base64(goal_rgb),
                "nuisance": encode_png_base64(nuis_rgb),
                "task_shift": encode_png_base64(shift_rgb),
            },
            "plain_mse": {
                "effective_rank": float(self.feature_dim),
                "cur_to_goal_energy": float(mse_cur_grid.mean()),
                "nuisance_energy": mse_nuis_mean,
                "task_energy": mse_task_mean,
                "task_to_nuisance_ratio": mse_ratio,
                "overlays": {
                    "cur_vs_goal": encode_png_base64(render_patch_heatmap_overlay(cur_rgb, mse_cur_grid)),
                    "nuisance": encode_png_base64(
                        render_patch_heatmap_overlay(nuis_rgb, mse_nuis_grid, vmax=mse_pair_vmax)
                    ),
                    "task_shift": encode_png_base64(
                        render_patch_heatmap_overlay(shift_rgb, mse_task_grid, vmax=mse_pair_vmax)
                    ),
                },
            },
            "learned_head": {
                "head_name": resolved_name,
                "effective_rank": float(tm.effective_rank(head_matrix)),
                "cur_to_goal_energy": float(head_cur_grid.mean()),
                "nuisance_energy": head_nuis_mean,
                "task_energy": head_task_mean,
                "task_to_nuisance_ratio": head_ratio,
                "ratio_gain_vs_mse": float(head_ratio / max(mse_ratio, 1e-12)),
                "overlays": {
                    "cur_vs_goal": encode_png_base64(render_patch_heatmap_overlay(cur_rgb, head_cur_grid)),
                    "nuisance": encode_png_base64(
                        render_patch_heatmap_overlay(nuis_rgb, head_nuis_grid, vmax=head_pair_vmax)
                    ),
                    "task_shift": encode_png_base64(
                        render_patch_heatmap_overlay(shift_rgb, head_task_grid, vmax=head_pair_vmax)
                    ),
                },
            },
        }

    def run_duel(
        self,
        initial_agent: Sequence[float],
        initial_block: Sequence[float],
        goal_agent: Sequence[float],
        goal_block: Sequence[float],
        head_name: str | None = None,
        replans: int = 4,
        seed: int = 20261124,
    ) -> dict[str, object]:
        """Run paired closed-loop episodes: Plain MSE + CEM vs. Calibrated Head + Trust-MPC."""
        if isinstance(replans, bool) or not isinstance(replans, int) or not 1 <= replans <= 16:
            raise ValueError("replans must be an integer between 1 and 16")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        resolved_name, head_matrix = self._resolve_head(head_name)
        init_agent_arr = np.asarray(initial_agent, dtype=np.float32)
        init_block_arr = np.asarray(initial_block, dtype=np.float32)
        goal_agent_arr = np.asarray(goal_agent, dtype=np.float32)
        goal_block_arr = np.asarray(goal_block, dtype=np.float32)

        initial_state = state_from(
            np.clip(init_agent_arr, 30.0, 482.0),
            np.array(
                [
                    float(np.clip(init_block_arr[0], 70.0, 442.0)),
                    float(np.clip(init_block_arr[1], 70.0, 442.0)),
                    float(init_block_arr[2] % (2.0 * np.pi)),
                ],
                dtype=np.float32,
            ),
        )
        target_pose = np.array(
            [
                float(np.clip(goal_block_arr[0], 70.0, 442.0)),
                float(np.clip(goal_block_arr[1], 70.0, 442.0)),
                float(goal_block_arr[2] % (2.0 * np.pi)),
            ],
            dtype=np.float64,
        )

        hold_chunk = tm.zero_displacement_chunk(
            self.action_mean, self.action_std, self.controls_per_chunk, self.action_clip
        )
        identity_head = self.heads["identity"]

        with self._lock:
            goal_rgb, goal_prop = self._render_pusht_pose_unlocked(goal_agent_arr, target_pose)
            goal_features = _encode_patch_features(self.model, goal_rgb, goal_prop)

            shared_normals = [
                cem_standard_normals(
                    seed,
                    0,
                    replan_idx,
                    self.cem_iterations,
                    self.cem_samples,
                    self.cem_horizon,
                    self.action_dim,
                )
                for replan_idx in range(replans)
            ]

            baseline_result, baseline_rgb = self._run_single_episode_unlocked(
                initial_state=initial_state,
                target_pose=target_pose,
                goal_features=goal_features,
                head=identity_head,
                planner="standard_cem",
                shared_normals=shared_normals,
                hold_chunk=hold_chunk,
            )
            trust_result, trust_rgb = self._run_single_episode_unlocked(
                initial_state=initial_state,
                target_pose=target_pose,
                goal_features=goal_features,
                head=head_matrix,
                planner="trust_mpc",
                shared_normals=shared_normals,
                hold_chunk=hold_chunk,
            )
            self._latest_duel_frames = (baseline_rgb, trust_rgb, goal_rgb)

        return {
            "head_name": resolved_name,
            "effective_rank": float(tm.effective_rank(head_matrix)),
            "replans": replans,
            "controls_per_chunk": self.controls_per_chunk,
            "seed": seed,
            "goal_image": encode_png_base64(goal_rgb),
            "baseline": baseline_result,
            "trust_mpc": trust_result,
            "gif_ready": True,
        }

    def _run_single_episode_unlocked(
        self,
        *,
        initial_state: np.ndarray,
        target_pose: np.ndarray,
        goal_features: torch.Tensor,
        head: torch.Tensor,
        planner: str,
        shared_normals: list[np.ndarray],
        hold_chunk: np.ndarray,
    ) -> tuple[dict[str, object], list[np.ndarray]]:
        self.env.reset_to_state = initial_state.copy()
        observation, _ = self.env.reset()

        first_rgb = _as_rgb_uint8(observation["visual"]).copy()
        raw_frames: list[np.ndarray] = [first_rgb]
        encoded_frames: list[str] = [encode_png_base64(first_rgb)]

        init_metrics = {
            k: float(v[0]) for k, v in hp.physical_metrics(initial_state[None, 2:5], target_pose).items()
        }
        best_pos_err = init_metrics["position_error"]
        replan_telemetry: list[dict[str, object]] = []
        previous_plan: np.ndarray | None = None
        total_unrolls = 0
        sim_steps = 0
        step_metrics = init_metrics

        for replan_idx, normals in enumerate(shared_normals):
            if planner == "standard_cem":
                action_chunk, plan_info = standard_cem_plan(
                    self.model,
                    observation,
                    goal_features,
                    head,
                    normals,
                    num_elites=self.cem_elites,
                    action_clip=self.action_clip,
                    batch_size=self.batch_size,
                )
                extra_diag: dict[str, object] = {
                    "best_cost": float(plan_info["best_cost"]),
                    "elite_mean_cost": float(plan_info["elite_mean_cost"]),
                }
            else:
                action_chunk, previous_plan, plan_info = tm.trust_mpc_plan(
                    self.model,
                    observation,
                    goal_features,
                    head,
                    normals,
                    hold_chunk,
                    previous_plan,
                    num_elites=self.cem_elites,
                    action_clip=self.action_clip,
                    batch_size=self.batch_size,
                    stay_margin=self.stay_margin,
                )
                extra_diag = {
                    "hold_cost": float(plan_info["hold_cost"]),
                    "elite_mean_cost": float(plan_info["elite_mean_cost"]),
                    "selected_cost": float(plan_info["selected_cost"]),
                    "jensen_violation": bool(plan_info["jensen_violation"]),
                    "used_stay_gate": bool(plan_info["used_stay_gate"]),
                }

            total_unrolls += int(plan_info["unrolls"])

            for control in action_chunk.reshape(self.controls_per_chunk, 2):
                raw_action = control * self.action_std + self.action_mean
                observation, _, _, step_info = self.env.step(raw_action)
                sim_steps += 1
                frame = _as_rgb_uint8(observation["visual"]).copy()
                raw_frames.append(frame)
                encoded_frames.append(encode_png_base64(frame))

            cur_state = np.asarray(step_info["state"], dtype=np.float64)
            step_metrics = {
                k: float(v[0]) for k, v in hp.physical_metrics(cur_state[None, 2:5], target_pose).items()
            }
            best_pos_err = min(best_pos_err, step_metrics["position_error"])

            replan_telemetry.append(
                {
                    "replan": replan_idx + 1,
                    "frame_index": len(raw_frames) - 1,
                    "position_error": step_metrics["position_error"],
                    "angle_error_deg": step_metrics["angle_error_deg"],
                    "combined_error": step_metrics["combined_error"],
                    "coverage": step_metrics["coverage"],
                    "unrolls": int(plan_info["unrolls"]),
                    **extra_diag,
                }
            )

        summary: dict[str, object] = {
            "initial_position_error": init_metrics["position_error"],
            "final_position_error": step_metrics["position_error"],
            "best_position_error": best_pos_err,
            "post_peak_drift": max(0.0, step_metrics["position_error"] - best_pos_err),
            "final_angle_error_deg": step_metrics["angle_error_deg"],
            "final_combined_error": step_metrics["combined_error"],
            "final_coverage": step_metrics["coverage"],
            "success_20px": bool(step_metrics["success_20px"]),
            "pose_success": bool(step_metrics["pose_success"]),
            "unrolls": int(total_unrolls),
            "sim_steps": int(sim_steps),
        }
        if planner == "trust_mpc":
            summary["jensen_violations"] = int(sum(1 for row in replan_telemetry if row["jensen_violation"]))
            summary["stay_gate_triggers"] = int(sum(1 for row in replan_telemetry if row["used_stay_gate"]))

        return {
            "planner": planner,
            "frames": encoded_frames,
            "replan_telemetry": replan_telemetry,
            "summary": summary,
        }, raw_frames

    def has_latest_duel_gif(self) -> bool:
        """Whether at least one duel has been run and cached for GIF export."""
        with self._lock:
            return self._latest_duel_frames is not None

    def get_latest_duel_gif_bytes(self, fps: int = 4) -> bytes:
        """Encode the most recent duel as a side-by-side animated GIF."""
        with self._lock:
            if self._latest_duel_frames is None:
                raise LookupError("No duel has been executed yet")
            baseline_rgb, trust_rgb, goal_rgb = self._latest_duel_frames
        return compose_duel_gif_bytes(baseline_rgb, trust_rgb, goal_rgb, fps=fps)

    def run_waypoint_chain(
        self,
        initial_agent: Sequence[float],
        initial_block: Sequence[float],
        waypoints: Sequence[dict[str, Sequence[float]]],
        head_name: str | None = None,
        replans: int = 6,
        arrival_ratio: float = 0.35,
        seed: int = 20261124,
        stage1_threshold_px: float = 35.0,
    ) -> dict[str, object]:
        """Run paired multi-stage visual waypoint chain episodes under exact rollout parity."""
        if isinstance(replans, bool) or not isinstance(replans, int) or not 1 <= replans <= 24:
            raise ValueError("replans must be an integer between 1 and 24")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(arrival_ratio, bool) or not math.isfinite(float(arrival_ratio)):
            raise ValueError("arrival_ratio must be a finite number in [0, 1]")
        if not 0.0 <= float(arrival_ratio) <= 1.0:
            raise ValueError("arrival_ratio must lie in [0, 1]")
        if not waypoints or len(waypoints) < 2:
            raise ValueError("waypoints must contain at least 2 subgoal entries")

        resolved_name, head_matrix = self._resolve_head(head_name)
        init_agent_arr = np.asarray(initial_agent, dtype=np.float32)
        init_block_arr = np.asarray(initial_block, dtype=np.float32)
        if init_agent_arr.shape != (2,) or init_block_arr.shape != (3,):
            raise ValueError("initial_agent must have shape [2] and initial_block must have shape [3]")

        initial_state = state_from(
            np.clip(init_agent_arr, 30.0, 482.0),
            np.array(
                [
                    float(np.clip(init_block_arr[0], 70.0, 442.0)),
                    float(np.clip(init_block_arr[1], 70.0, 442.0)),
                    float(init_block_arr[2] % (2.0 * np.pi)),
                ],
                dtype=np.float32,
            ),
        )

        waypoint_poses: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, entry in enumerate(waypoints):
            if not isinstance(entry, dict) or "block" not in entry or "goal_agent" not in entry:
                raise ValueError(f"waypoints[{idx}] must be an object with 'block' and 'goal_agent'")
            wp_agent = np.clip(np.asarray(entry["goal_agent"], dtype=np.float32), 30.0, 482.0)
            wp_block_raw = np.asarray(entry["block"], dtype=np.float32)
            if wp_agent.shape != (2,) or wp_block_raw.shape != (3,):
                raise ValueError(f"waypoints[{idx}] must specify goal_agent [2] and block [3]")
            wp_block = np.array(
                [
                    float(np.clip(wp_block_raw[0], 70.0, 442.0)),
                    float(np.clip(wp_block_raw[1], 70.0, 442.0)),
                    float(wp_block_raw[2] % (2.0 * np.pi)),
                ],
                dtype=np.float64,
            )
            waypoint_poses.append((wp_agent, wp_block))

        hold_chunk = tm.zero_displacement_chunk(
            self.action_mean, self.action_std, self.controls_per_chunk, self.action_clip
        )
        identity_head = self.heads["identity"]

        with self._lock:
            waypoint_rgbs: list[np.ndarray] = []
            waypoint_features: list[torch.Tensor] = []
            for wp_agent, wp_block in waypoint_poses:
                wp_rgb, wp_prop = self._render_pusht_pose_unlocked(wp_agent, wp_block)
                waypoint_rgbs.append(wp_rgb)
                waypoint_features.append(_encode_patch_features(self.model, wp_rgb, wp_prop))

            shared_normals = [
                cem_standard_normals(
                    seed,
                    0,
                    replan_idx,
                    self.cem_iterations,
                    self.cem_samples,
                    self.cem_horizon,
                    self.action_dim,
                )
                for replan_idx in range(replans)
            ]

            target_blocks = [wp_block for _, wp_block in waypoint_poses]
            baseline_result, _, _ = self._run_single_waypoint_episode_unlocked(
                initial_state=initial_state,
                waypoint_blocks=target_blocks,
                waypoint_features=waypoint_features,
                head=identity_head,
                planner="standard_cem",
                shared_normals=shared_normals,
                hold_chunk=hold_chunk,
                arrival_ratio=float(arrival_ratio),
                stage1_threshold_px=float(stage1_threshold_px),
            )
            trust_result, trust_rgb, trust_frame_stages = self._run_single_waypoint_episode_unlocked(
                initial_state=initial_state,
                waypoint_blocks=target_blocks,
                waypoint_features=waypoint_features,
                head=head_matrix,
                planner="waypoint_trust_mpc",
                shared_normals=shared_normals,
                hold_chunk=hold_chunk,
                arrival_ratio=float(arrival_ratio),
                stage1_threshold_px=float(stage1_threshold_px),
            )
            self._latest_waypoint_gif_data = (
                trust_rgb,
                waypoint_rgbs,
                trust_frame_stages,
                f"Trust-MPC ({resolved_name})",
            )

        return {
            "head_name": resolved_name,
            "effective_rank": float(tm.effective_rank(head_matrix)),
            "replans": replans,
            "controls_per_chunk": self.controls_per_chunk,
            "arrival_ratio": float(arrival_ratio),
            "seed": seed,
            "num_waypoints": len(waypoint_rgbs),
            "waypoint_images": [encode_png_base64(img) for img in waypoint_rgbs],
            "baseline": baseline_result,
            "trust_mpc": trust_result,
            "gif_ready": True,
        }

    def _run_single_waypoint_episode_unlocked(
        self,
        *,
        initial_state: np.ndarray,
        waypoint_blocks: list[np.ndarray],
        waypoint_features: list[torch.Tensor],
        head: torch.Tensor,
        planner: str,
        shared_normals: list[np.ndarray],
        hold_chunk: np.ndarray,
        arrival_ratio: float,
        stage1_threshold_px: float,
    ) -> tuple[dict[str, object], list[np.ndarray], list[int]]:
        self.env.reset_to_state = initial_state.copy()
        observation, _ = self.env.reset()

        first_rgb = _as_rgb_uint8(observation["visual"]).copy()
        raw_frames: list[np.ndarray] = [first_rgb]
        encoded_frames: list[str] = [encode_png_base64(first_rgb)]
        frame_stages: list[int] = [0]

        tracker = WaypointTracker(
            waypoint_features,
            head.to(device=self.model.device, dtype=torch.float32),
            arrival_ratio=arrival_ratio,
            allow_stay_gate_advance=(planner == "waypoint_trust_mpc"),
        )

        stage1_pose = waypoint_blocks[0]
        final_pose = waypoint_blocks[-1]
        cur_block_pose = initial_state[2:5].astype(np.float64)
        best_stage1_err = float(hp.physical_metrics(cur_block_pose[None, :], stage1_pose)["position_error"][0])
        best_final_err = float(hp.physical_metrics(cur_block_pose[None, :], final_pose)["position_error"][0])
        final_metrics = {
            k: float(v[0]) for k, v in hp.physical_metrics(cur_block_pose[None, :], final_pose).items()
        }

        handoff_step: int | None = None
        handoff_stage1_err: float | None = None
        replan_telemetry: list[dict[str, object]] = []
        previous_plan: np.ndarray | None = None
        total_unrolls = 0
        sim_steps = 0

        for replan_idx, normals in enumerate(shared_normals):
            if planner == "standard_cem":
                current_features = encode_visual_features(self.model, observation)
                t_info = tracker.update(current_features, used_stay_gate=False)
                action_chunk, p_info = standard_cem_plan(
                    self.model,
                    observation,
                    tracker.active_waypoint,
                    head,
                    normals,
                    num_elites=self.cem_elites,
                    action_clip=self.action_clip,
                    batch_size=self.batch_size,
                )
                step_diag: dict[str, object] = {
                    **p_info,
                    **t_info,
                    "jensen_violation": False,
                    "used_stay_gate": False,
                    "warm_start_reset": bool(t_info["advanced"]),
                }
            else:
                action_chunk, previous_plan, step_diag = waypoint_trust_mpc_step(
                    self.model,
                    observation,
                    tracker,
                    normals,
                    hold_chunk,
                    previous_plan,
                    num_elites=self.cem_elites,
                    action_clip=self.action_clip,
                    batch_size=self.batch_size,
                    stay_margin=self.stay_margin,
                    certify_barrier_split=True,
                )

            if step_diag["advanced"] and handoff_step is None:
                handoff_step = replan_idx
                handoff_stage1_err = float(
                    hp.physical_metrics(cur_block_pose[None, :], stage1_pose)["position_error"][0]
                )

            total_unrolls += int(step_diag["unrolls"])
            active_stage = int(tracker.stage_index)
            for control in action_chunk.reshape(self.controls_per_chunk, 2):
                raw_action = control * self.action_std + self.action_mean
                observation, _, _, step_info = self.env.step(raw_action)
                sim_steps += 1
                frame = _as_rgb_uint8(observation["visual"]).copy()
                raw_frames.append(frame)
                encoded_frames.append(encode_png_base64(frame))
                frame_stages.append(active_stage)

            cur_block_pose = np.asarray(step_info["state"], dtype=np.float64)[2:5]
            s1_err = float(hp.physical_metrics(cur_block_pose[None, :], stage1_pose)["position_error"][0])
            active_err = float(
                hp.physical_metrics(cur_block_pose[None, :], waypoint_blocks[active_stage])["position_error"][0]
            )
            final_metrics = {
                k: float(v[0]) for k, v in hp.physical_metrics(cur_block_pose[None, :], final_pose).items()
            }
            best_stage1_err = min(best_stage1_err, s1_err)
            best_final_err = min(best_final_err, final_metrics["position_error"])

            replan_telemetry.append(
                {
                    "replan": replan_idx + 1,
                    "frame_index": len(raw_frames) - 1,
                    "stage_index": active_stage,
                    "advanced": bool(step_diag["advanced"]),
                    "trigger": step_diag["trigger"],
                    "cost_ratio": float(step_diag["cost_ratio"]),
                    "warm_start_reset": bool(step_diag["warm_start_reset"]),
                    "active_stage_error_px": active_err,
                    "final_stage_error_px": final_metrics["position_error"],
                    "jensen_violation": bool(step_diag.get("jensen_violation", False)),
                    "used_stay_gate": bool(step_diag.get("used_stay_gate", False)),
                    "unrolls": int(step_diag["unrolls"]),
                }
            )

        handoff_triggered = handoff_step is not None
        handoff_accurate = bool(
            handoff_triggered
            and handoff_stage1_err is not None
            and handoff_stage1_err <= stage1_threshold_px
        )
        summary: dict[str, object] = {
            "best_stage1_error_px": best_stage1_err,
            "stage1_completed": bool(best_stage1_err <= stage1_threshold_px),
            "handoff_triggered": handoff_triggered,
            "handoff_step": handoff_step,
            "handoff_stage1_error_px": handoff_stage1_err,
            "handoff_accurate": handoff_accurate,
            "final_stage_index": int(tracker.stage_index),
            "final_position_error": final_metrics["position_error"],
            "best_position_error": best_final_err,
            "final_angle_error_deg": final_metrics["angle_error_deg"],
            "final_combined_error": final_metrics["combined_error"],
            "final_coverage": final_metrics["coverage"],
            "success_20px": bool(final_metrics["success_20px"]),
            "jensen_violations": int(sum(1 for r in replan_telemetry if r["jensen_violation"])),
            "stay_gate_triggers": int(sum(1 for r in replan_telemetry if r["used_stay_gate"])),
            "warm_start_resets": int(sum(1 for r in replan_telemetry if r["warm_start_reset"])),
            "unrolls": int(total_unrolls),
            "sim_steps": int(sim_steps),
        }
        return {
            "planner": planner,
            "frames": encoded_frames,
            "frame_stages": frame_stages,
            "replan_telemetry": replan_telemetry,
            "transitions": list(tracker.transitions),
            "summary": summary,
        }, raw_frames, frame_stages

    def has_latest_waypoint_gif(self) -> bool:
        """Whether at least one multi-stage waypoint chain has been cached for GIF export."""
        with self._lock:
            return self._latest_waypoint_gif_data is not None

    def get_latest_waypoint_gif_bytes(self, fps: int = 4) -> bytes:
        """Encode the most recent multi-stage waypoint rollout as an animated GIF."""
        with self._lock:
            if self._latest_waypoint_gif_data is None:
                raise LookupError("No waypoint chain has been executed yet")
            rollout_rgb, waypoint_rgbs, frame_stages, title_prefix = self._latest_waypoint_gif_data
        return compose_waypoint_gif_bytes(
            rollout_rgb,
            waypoint_rgbs,
            frame_stages,
            title_prefix=title_prefix,
            fps=fps,
        )


# --------------------------------------------------------- Meta-World session


class MetaWorldStudioSession:
    """Interactive 3D Meta-World Sawyer Push-v3 counterfactual geometry inspector."""

    def __init__(
        self,
        model,
        heads: dict[str, torch.Tensor],
        env=None,
        *,
        warmup_steps: int = 15,
        goal_steps: int = 25,
        default_head: str | None = None,
    ) -> None:
        if not heads:
            raise ValueError("heads dictionary must contain at least one head")
        self.model = model
        self.heads = {name: mat.detach().cpu().float() for name, mat in heads.items()}
        first_head = next(iter(self.heads.values()))
        self.feature_dim = int(first_head.shape[1])
        if "identity" not in self.heads:
            self.heads["identity"] = lh.identity_head(self.feature_dim)

        if default_head is not None and default_head in self.heads:
            self.default_head = default_head
        else:
            non_identity = [name for name in self.heads if name != "identity"]
            self.default_head = non_identity[0] if non_identity else "identity"

        self._owns_env = env is None
        if env is None:
            from jepa_planning import metaworld_push as mp

            self.env = mp.make_push_env()
        else:
            self.env = env
        self.warmup_steps = int(warmup_steps)
        self.goal_steps = int(goal_steps)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Release MuJoCo simulator resources owned by this session."""
        if self._owns_env and self.env is not None:
            self.env.close()

    def _resolve_head(self, head_name: str | None) -> tuple[str, torch.Tensor]:
        name = self.default_head if not head_name else str(head_name)
        if name not in self.heads:
            raise KeyError(f"Unknown Meta-World head '{name}'. Available: {sorted(self.heads)}")
        return name, self.heads[name]

    def inspect_metaworld_task(
        self,
        seed: int = 1000,
        hand_offset_angle_deg: float = 135.0,
        shift_cm: float = 5.0,
        head_name: str | None = None,
    ) -> dict[str, object]:
        """Inspect 3D Sawyer Push-v3 patch energies under arm nuisance vs. puck shift."""
        from jepa_planning import metaworld_push as mp

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not np.isfinite(hand_offset_angle_deg):
            raise ValueError("hand_offset_angle_deg must be finite")
        if not np.isfinite(shift_cm) or float(shift_cm) <= 0.0:
            raise ValueError("shift_cm must be positive and finite")

        resolved_name, head_matrix = self._resolve_head(head_name)
        shift_m = float(shift_cm) / 100.0
        angle_rad = math.radians(float(hand_offset_angle_deg))

        with self._lock:
            obs = mp.reset_task(self.env, seed)
            obs, _ = mp.run_expert(self.env, obs, self.warmup_steps)
            mp.run_expert(self.env, obs, self.goal_steps)
            goal_snap = mp.snapshot(self.env)
            goal_puck = mp.puck_pose(self.env)

            base_hand_xy = np.array(
                [
                    float(np.clip(goal_puck[0] + 0.085, -0.18, 0.18)),
                    float(np.clip(goal_puck[1] - 0.045, 0.48, 0.86)),
                ],
                dtype=np.float64,
            )
            pairs = mwp.render_puck_sensitivity_pairs(
                self.env,
                goal_snap,
                base_hand_xy,
                shifts_m=(shift_m,),
            )
            goal_rgb = _as_rgb_uint8(pairs["reference_visual"]).copy()
            goal_prop = np.asarray(pairs["reference_proprio"], dtype=np.float32).copy()
            ref_hand = np.asarray(pairs["reference_hand_pos"], dtype=np.float64)

            shift_rgb = _as_rgb_uint8(pairs["visual"][0]).copy()
            shift_prop = np.asarray(pairs["proprio"][0], dtype=np.float32).copy()

            nuis_hand_xy = np.array(
                [
                    float(np.clip(goal_puck[0] + 0.11 * math.cos(angle_rad), -0.18, 0.18)),
                    float(np.clip(goal_puck[1] + 0.11 * math.sin(angle_rad), 0.48, 0.86)),
                ],
                dtype=np.float64,
            )
            mp.restore(self.env, goal_snap)
            offset = np.array(
                [
                    nuis_hand_xy[0] - mp.hand_pos(self.env)[0],
                    nuis_hand_xy[1] - mp.hand_pos(self.env)[1],
                    0.0,
                ],
                dtype=np.float64,
            )
            mp.move_hand_keep_puck(self.env, offset)
            nuis_rgb = _as_rgb_uint8(self.env.render()).copy()
            nuis_prop = self.env.unwrapped._get_curr_obs_combined_no_goal()[:4].astype(np.float32)
            nuis_hand = mp.hand_pos(self.env)

            f_goal = _encode_patch_features(self.model, goal_rgb, goal_prop)
            f_nuis = _encode_patch_features(self.model, nuis_rgb, nuis_prop)
            f_shift = _encode_patch_features(self.model, shift_rgb, shift_prop)

        mse_nuis_grid = patch_energy_map(f_goal, f_nuis, head=None).numpy()
        mse_task_grid = patch_energy_map(f_goal, f_shift, head=None).numpy()
        head_nuis_grid = patch_energy_map(f_goal, f_nuis, head=head_matrix).numpy()
        head_task_grid = patch_energy_map(f_goal, f_shift, head=head_matrix).numpy()

        mse_nuis_mean = float(mse_nuis_grid.mean())
        mse_task_mean = float(mse_task_grid.mean())
        mse_ratio = float(mse_task_mean / max(mse_nuis_mean, 1e-12))

        head_nuis_mean = float(head_nuis_grid.mean())
        head_task_mean = float(head_task_grid.mean())
        head_ratio = float(head_task_mean / max(head_nuis_mean, 1e-12))

        mse_vmax = max(float(mse_nuis_grid.max()), float(mse_task_grid.max()), 1e-12)
        head_vmax = max(float(head_nuis_grid.max()), float(head_task_grid.max()), 1e-12)

        return {
            "seed": seed,
            "hand_offset_angle_deg": float(hand_offset_angle_deg),
            "shift_cm": float(shift_cm),
            "head_name": resolved_name,
            "effective_rank": float(tm.effective_rank(head_matrix)),
            "physical": {
                "hand_displacement_cm": float(100.0 * np.linalg.norm(nuis_hand[:2] - ref_hand[:2])),
                "puck_shift_cm": float(shift_cm),
            },
            "frames": {
                "goal": encode_png_base64(goal_rgb),
                "nuisance": encode_png_base64(nuis_rgb),
                "task_shift": encode_png_base64(shift_rgb),
            },
            "plain_mse": {
                "effective_rank": float(self.feature_dim),
                "nuisance_energy": mse_nuis_mean,
                "task_energy": mse_task_mean,
                "task_to_nuisance_ratio": mse_ratio,
                "overlays": {
                    "nuisance": encode_png_base64(
                        render_patch_heatmap_overlay(nuis_rgb, mse_nuis_grid, vmax=mse_vmax)
                    ),
                    "task_shift": encode_png_base64(
                        render_patch_heatmap_overlay(shift_rgb, mse_task_grid, vmax=mse_vmax)
                    ),
                },
            },
            "learned_head": {
                "head_name": resolved_name,
                "effective_rank": float(tm.effective_rank(head_matrix)),
                "nuisance_energy": head_nuis_mean,
                "task_energy": head_task_mean,
                "task_to_nuisance_ratio": head_ratio,
                "ratio_gain_vs_mse": float(head_ratio / max(mse_ratio, 1e-12)),
                "overlays": {
                    "nuisance": encode_png_base64(
                        render_patch_heatmap_overlay(nuis_rgb, head_nuis_grid, vmax=head_vmax)
                    ),
                    "task_shift": encode_png_base64(
                        render_patch_heatmap_overlay(shift_rgb, head_task_grid, vmax=head_vmax)
                    ),
                },
            },
        }


# ------------------------------------------------------------- unified backend


class PlanningStudioBackend:
    """Facade combining 2D PushT and optional 3D Meta-World studio sessions."""

    def __init__(
        self,
        pusht_session: PushTStudioSession,
        metaworld_session: MetaWorldStudioSession | None = None,
    ) -> None:
        self.pusht = pusht_session
        self.metaworld = metaworld_session

    def get_state(self) -> dict[str, object]:
        """Return available heads, presets, and environment flags for the UI."""
        return {
            "pusht": {
                "heads": _head_catalog_metadata(self.pusht.heads),
                "default_head": self.pusht.default_head,
                "presets": PUSHT_PRESETS,
                "waypoint_presets": PUSHT_WAYPOINT_PRESETS,
                "config": {
                    "controls_per_chunk": self.pusht.controls_per_chunk,
                    "cem_horizon": self.pusht.cem_horizon,
                    "cem_samples": self.pusht.cem_samples,
                    "cem_elites": self.pusht.cem_elites,
                    "cem_iterations": self.pusht.cem_iterations,
                    "stay_margin": self.pusht.stay_margin,
                },
                "has_duel_gif": self.pusht.has_latest_duel_gif(),
                "has_waypoint_gif": self.pusht.has_latest_waypoint_gif(),
            },
            "metaworld": {
                "enabled": self.metaworld is not None,
                "heads": _head_catalog_metadata(self.metaworld.heads) if self.metaworld is not None else [],
                "default_head": self.metaworld.default_head if self.metaworld is not None else None,
            },
        }

    def inspect_pusht(self, payload: dict[str, object]) -> dict[str, object]:
        """Validate a JSON request payload and run PushT patch-geometry inspection."""
        return self.pusht.inspect_patch_geometry(
            cur_agent=payload["cur_agent"],
            cur_block=payload["cur_block"],
            goal_agent=payload["goal_agent"],
            goal_block=payload["goal_block"],
            head_name=payload.get("head_name"),
        )

    def run_pusht_duel(self, payload: dict[str, object]) -> dict[str, object]:
        """Validate a JSON request payload and run a paired PushT closed-loop duel."""
        return self.pusht.run_duel(
            initial_agent=payload["initial_agent"],
            initial_block=payload["initial_block"],
            goal_agent=payload["goal_agent"],
            goal_block=payload["goal_block"],
            head_name=payload.get("head_name"),
            replans=int(payload.get("replans", 4)),
            seed=int(payload.get("seed", 20261124)),
        )

    def get_duel_gif_bytes(self, fps: int = 4) -> bytes:
        """Return the latest PushT duel as an animated GIF byte buffer."""
        return self.pusht.get_latest_duel_gif_bytes(fps=fps)

    def run_pusht_waypoint_chain(self, payload: dict[str, object]) -> dict[str, object]:
        """Validate a JSON request payload and run a paired multi-stage PushT waypoint chain."""
        return self.pusht.run_waypoint_chain(
            initial_agent=payload["initial_agent"],
            initial_block=payload["initial_block"],
            waypoints=payload["waypoints"],
            head_name=payload.get("head_name"),
            replans=int(payload.get("replans", 6)),
            arrival_ratio=float(payload.get("arrival_ratio", 0.35)),
            seed=int(payload.get("seed", 20261124)),
            stage1_threshold_px=float(payload.get("stage1_threshold_px", 35.0)),
        )

    def get_waypoint_gif_bytes(self, fps: int = 4) -> bytes:
        """Return the latest multi-stage waypoint chain as an animated GIF byte buffer."""
        return self.pusht.get_latest_waypoint_gif_bytes(fps=fps)

    def inspect_metaworld(self, payload: dict[str, object]) -> dict[str, object]:
        """Run 3D Meta-World Sawyer Push-v3 inspection if enabled."""
        if self.metaworld is None:
            raise RuntimeError("Meta-World session is not enabled. Launch with --enable-metaworld.")
        return self.metaworld.inspect_metaworld_task(
            seed=int(payload.get("seed", 1000)),
            hand_offset_angle_deg=float(payload.get("hand_offset_angle_deg", 135.0)),
            shift_cm=float(payload.get("shift_cm", 5.0)),
            head_name=payload.get("head_name"),
        )

    def close(self) -> None:
        """Close underlying simulator environments."""
        self.pusht.close()
        if self.metaworld is not None:
            self.metaworld.close()


# --------------------------------------------------------------- embedded HTML

STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Visual World-Model Planning Studio</title>
<style>
  :root {
    --bg: #1e2022;
    --surface: #26292c;
    --surface-inset: #191b1d;
    --border: #35393f;
    --text: #eceff3;
    --muted: #98a1ac;
    --amber: #e69d3c;
    --crimson: #e05347;
    --emerald: #42b87c;
    --blue: #5da4e8;
    --radius: 8px;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.45;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 22px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .brand h1 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .brand p {
    margin: 2px 0 0;
    color: var(--muted);
    font-size: 12px;
  }
  .header-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .tabs {
    display: inline-flex;
    background: var(--surface-inset);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px;
  }
  .tab-btn {
    border: 0;
    background: transparent;
    color: var(--muted);
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }
  .tab-btn.active {
    background: var(--surface);
    color: var(--text);
  }
  select, input[type="number"] {
    background: var(--surface-inset);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 12px;
    font-family: var(--mono);
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 8px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 11px;
    background: var(--surface-inset);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.amber { border-color: rgba(230, 157, 60, 0.5); color: var(--amber); }
  .badge.emerald { border-color: rgba(66, 184, 124, 0.5); color: var(--emerald); }
  .badge.crimson { border-color: rgba(224, 83, 71, 0.5); color: var(--crimson); }
  main {
    max-width: 1360px;
    margin: 0 auto;
    padding: 18px 22px 40px;
  }
  .hidden { display: none !important; }
  .grid-top {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 1020px) {
    .grid-top { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
  }
  .panel-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 0 10px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
  }
  .canvas-wrap {
    display: flex;
    justify-content: center;
    background: var(--surface-inset);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
  }
  canvas#arena {
    width: 330px;
    height: 330px;
    cursor: grab;
    touch-action: none;
  }
  canvas#arena:active { cursor: grabbing; }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted);
  }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .preset-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 10px 0;
  }
  .btn {
    border: 1px solid var(--border);
    background: var(--surface-inset);
    color: var(--text);
    padding: 7px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: border-color 0.12s ease, background 0.12s ease;
  }
  .btn:hover { border-color: #555c66; background: #222528; }
  .btn.primary { background: var(--amber); color: #16181a; border-color: var(--amber); font-weight: 600; }
  .btn.primary:hover { filter: brightness(1.06); }
  .btn.emerald { background: rgba(66, 184, 124, 0.16); border-color: rgba(66, 184, 124, 0.6); color: var(--emerald); }
  .btn.small { padding: 5px 8px; font-size: 11px; }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .slider-row {
    display: grid;
    grid-template-columns: 112px 1fr 48px;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 12px;
  }
  .slider-row output { text-align: right; font-family: var(--mono); color: var(--muted); }
  input[type="range"] { width: 100%; accent-color: var(--amber); }
  .inline-fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 8px 0 12px;
  }
  .field label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 3px; }
  .field input { width: 100%; }
  .action-bar { display: flex; gap: 8px; flex-wrap: wrap; }
  .preset-desc {
    margin: 4px 0 10px;
    padding: 8px 10px;
    background: var(--surface-inset);
    border-left: 2px solid var(--amber);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11.5px;
  }
  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }
  @media (max-width: 720px) { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
  .kpi {
    background: var(--surface-inset);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 9px 11px;
  }
  .kpi-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-value { font-family: var(--mono); font-size: 17px; font-weight: 600; margin-top: 2px; }
  .kpi-sub { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 2px; }
  .heatmap-section { margin-bottom: 10px; }
  .heatmap-row-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 8px 0 6px;
    font-size: 12px;
    font-weight: 500;
  }
  .heatmap-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .frame-card {
    background: var(--surface-inset);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .frame-card img {
    display: block;
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    background: #131517;
  }
  .frame-caption {
    padding: 6px 8px;
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    gap: 6px;
    border-top: 1px solid var(--border);
  }
  .duel-panel { margin-top: 16px; }
  .duel-stage {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 12px 0;
  }
  .scrubber-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    padding: 10px 12px;
    background: var(--surface-inset);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 12px;
  }
  .scrubber-bar input[type="range"] { flex: 1; min-width: 140px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
  }
  th, td {
    text-align: right;
    padding: 7px 8px;
    border-bottom: 1px solid var(--border);
  }
  th:first-child, td:first-child { text-align: left; }
  th {
    color: var(--muted);
    font-weight: 500;
    font-size: 11px;
    background: var(--surface-inset);
  }
  .status-line {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--muted);
  }
  .mw-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: flex-end;
    margin-bottom: 14px;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Visual World-Model Planning Studio</h1>
    <p>Frozen DINO-WM 16&times;16 Patch Geometry &middot; Counterfactual Distractors &middot; Jensen-Certified Trust-MPC</p>
  </div>
  <div class="header-controls">
    <div class="tabs">
      <button class="tab-btn active" data-tab="pusht">2D PushT Studio</button>
      <button class="tab-btn" data-tab="waypoints">Multi-Stage Waypoint Chain</button>
      <button class="tab-btn" data-tab="metaworld">3D Sawyer Push-v3</button>
    </div>
    <label for="head-select" style="color: var(--muted); font-size: 12px;">Head</label>
    <select id="head-select"></select>
    <span id="rank-badge" class="badge amber">eff. rank --</span>
  </div>
</header>

<main>
  <!-- 2D PushT Tab -->
  <section id="tab-pusht">
    <div class="grid-top">
      <div class="panel">
        <div class="panel-title">
          <span>Interactive PushT Arena</span>
          <span class="status-line">drag T or pusher</span>
        </div>
        <div class="canvas-wrap">
          <canvas id="arena" width="330" height="330"></canvas>
        </div>
        <div class="legend">
          <span><i class="dot" style="background:#9ca4b0"></i>Initial T</span>
          <span><i class="dot" style="background:#5da4e8"></i>Initial Pusher</span>
          <span><i class="dot" style="background:#e69d3c"></i>Target T</span>
          <span><i class="dot" style="border:2px solid #5da4e8;background:transparent"></i>Goal Distractor</span>
        </div>

        <div class="preset-row" id="preset-buttons"></div>
        <div class="preset-desc" id="preset-desc">Drag any element on the arena or pick a diagnostic preset.</div>

        <div class="slider-row">
          <label for="init-rot">Initial T Angle</label>
          <input id="init-rot" type="range" min="0" max="359" value="20" />
          <output id="init-rot-out">20&deg;</output>
        </div>
        <div class="slider-row">
          <label for="goal-rot">Target T Angle</label>
          <input id="goal-rot" type="range" min="0" max="359" value="37" />
          <output id="goal-rot-out">37&deg;</output>
        </div>

        <div class="inline-fields">
          <div class="field">
            <label for="replans-input">Closed-Loop Replans</label>
            <input id="replans-input" type="number" min="1" max="12" value="4" />
          </div>
          <div class="field">
            <label for="seed-input">Shared CEM Seed</label>
            <input id="seed-input" type="number" value="20261124" />
          </div>
        </div>

        <div class="action-bar">
          <button id="btn-inspect" class="btn">Inspect 16&times;16 Patch Geometry</button>
          <button id="btn-duel" class="btn primary">Run Head-to-Head Duel</button>
        </div>
        <div id="pusht-status" class="status-line" style="margin-top:8px;">Ready.</div>
      </div>

      <div class="panel">
        <div class="panel-title">
          <span>16&times;16 Patch-Energy Geometry</span>
          <span id="geom-subtitle" class="status-line">Awaiting inspection</span>
        </div>

        <div class="kpi-strip">
          <div class="kpi">
            <div class="kpi-label">Plain MSE Ratio</div>
            <div class="kpi-value" id="kpi-mse-ratio">--</div>
            <div class="kpi-sub" id="kpi-mse-sub">task / nuisance</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Learned Head Ratio</div>
            <div class="kpi-value" style="color:var(--emerald)" id="kpi-head-ratio">--</div>
            <div class="kpi-sub" id="kpi-head-sub">task / nuisance</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Selectivity Gain</div>
            <div class="kpi-value" style="color:var(--amber)" id="kpi-gain">--</div>
            <div class="kpi-sub" id="kpi-rank-sub">eff. rank --</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Initial T Gap</div>
            <div class="kpi-value" id="kpi-phys-gap">--</div>
            <div class="kpi-sub" id="kpi-phys-sub">angle --</div>
          </div>
        </div>

        <div class="heatmap-section">
          <div class="heatmap-row-label">
            <span>Plain MSE (Identity Head)</span>
            <span id="mse-energy-line" class="status-line">--</span>
          </div>
          <div class="heatmap-grid">
            <div class="frame-card">
              <img id="img-mse-cur" alt="Plain MSE current vs goal" />
              <div class="frame-caption"><span>Current vs Goal</span><span>MSE</span></div>
            </div>
            <div class="frame-card">
              <img id="img-mse-nuis" alt="Plain MSE nuisance" />
              <div class="frame-caption"><span>Nuisance (Moved Pusher)</span><span id="cap-mse-nuis">--</span></div>
            </div>
            <div class="frame-card">
              <img id="img-mse-task" alt="Plain MSE task shift" />
              <div class="frame-caption"><span>Task (+20px T-Shift)</span><span id="cap-mse-task">--</span></div>
            </div>
          </div>
        </div>

        <div class="heatmap-section">
          <div class="heatmap-row-label">
            <span id="head-row-title">Calibrated Linear Head</span>
            <span id="head-energy-line" class="status-line">--</span>
          </div>
          <div class="heatmap-grid">
            <div class="frame-card">
              <img id="img-head-cur" alt="Learned head current vs goal" />
              <div class="frame-caption"><span>Current vs Goal</span><span>Head L</span></div>
            </div>
            <div class="frame-card">
              <img id="img-head-nuis" alt="Learned head nuisance" />
              <div class="frame-caption"><span>Nuisance (Moved Pusher)</span><span id="cap-head-nuis">--</span></div>
            </div>
            <div class="frame-card">
              <img id="img-head-task" alt="Learned head task shift" />
              <div class="frame-caption"><span>Task (+20px T-Shift)</span><span id="cap-head-task">--</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel duel-panel">
      <div class="panel-title">
        <span>Paired Closed-Loop Duel (Exact Rollout Parity)</span>
        <div style="display:flex;gap:8px;align-items:center;">
          <span id="duel-parity-badge" class="badge">unrolls: --</span>
          <a id="btn-download-gif" class="btn small emerald hidden" href="/api/duel.gif" download="planning_duel.gif">Download Side-by-Side GIF</a>
        </div>
      </div>

      <div class="kpi-strip">
        <div class="kpi">
          <div class="kpi-label">Plain MSE + CEM Final</div>
          <div class="kpi-value" style="color:var(--crimson)" id="duel-mse-final">--</div>
          <div class="kpi-sub" id="duel-mse-drift">drift --</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Head + Trust-MPC Final</div>
          <div class="kpi-value" style="color:var(--emerald)" id="duel-trust-final">--</div>
          <div class="kpi-sub" id="duel-trust-drift">drift --</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Jensen Barrier Splits</div>
          <div class="kpi-value" style="color:var(--amber)" id="duel-jensen-count">--</div>
          <div class="kpi-sub">non-convex mode splits</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Stay-Gate Triggers</div>
          <div class="kpi-value" style="color:var(--blue)" id="duel-stay-count">--</div>
          <div class="kpi-sub">hold-cost certifications</div>
        </div>
      </div>

      <div class="duel-stage">
        <div class="frame-card">
          <img id="duel-img-goal" alt="Goal photo" />
          <div class="frame-caption"><span>Goal Photo (with Distractor)</span><span class="status-line">Target</span></div>
        </div>
        <div class="frame-card">
          <img id="duel-img-base" alt="Plain MSE + CEM trajectory" />
          <div class="frame-caption"><span>Plain MSE + Standard CEM</span><span id="duel-cap-base">--</span></div>
        </div>
        <div class="frame-card">
          <img id="duel-img-trust" alt="Calibrated Head + Trust-MPC trajectory" />
          <div class="frame-caption"><span>Calibrated Head + Trust-MPC</span><span id="duel-cap-trust">--</span></div>
        </div>
      </div>

      <div class="scrubber-bar">
        <button id="btn-play" class="btn small" disabled>Play</button>
        <input id="duel-scrubber" type="range" min="0" max="0" value="0" disabled />
        <span id="duel-frame-label" class="status-line">Step 0 / 0</span>
      </div>

      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Replan</th>
              <th>MSE Err (px)</th>
              <th>Trust Err (px)</th>
              <th>Trust Angle (&deg;)</th>
              <th>Hold Cost</th>
              <th>Elite Mean Cost</th>
              <th>Selected Cost</th>
              <th>Jensen Gate</th>
              <th>Stay Gate</th>
              <th>Unrolls</th>
            </tr>
          </thead>
          <tbody id="telemetry-body">
            <tr><td colspan="10" style="text-align:center;color:var(--muted);">Run a head-to-head duel to inspect per-replan telemetry.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Multi-Stage Waypoint Chain Tab -->
  <section id="tab-waypoints" class="hidden">
    <div class="panel">
      <div class="panel-title">
        <span>Multi-Stage Waypoint Chain (Visual Latent Handoff)</span>
        <div style="display:flex;gap:8px;align-items:center;">
          <span id="wp-parity-badge" class="badge">unrolls: --</span>
          <a id="btn-wp-download-gif" class="btn small emerald hidden" href="/api/waypoint.gif" download="waypoint_chain.gif">Download Waypoint GIF</a>
        </div>
      </div>
      <div class="preset-row" id="wp-preset-buttons"></div>
      <div id="wp-preset-desc" class="preset-desc">Select a 2-stage preset to compare purely visual latent handoff under Plain MSE vs. a calibrated head.</div>
      <div class="mw-controls">
        <div class="field">
          <label for="wp-replans">Replans</label>
          <input id="wp-replans" type="number" min="2" max="24" value="6" />
        </div>
        <div class="field">
          <label for="wp-arrival-ratio">Arrival Ratio</label>
          <input id="wp-arrival-ratio" type="number" step="0.05" min="0.05" max="0.95" value="0.35" />
        </div>
        <div class="field">
          <label for="wp-seed">Shared Seed</label>
          <input id="wp-seed" type="number" value="20261124" />
        </div>
        <button id="btn-wp-run" class="btn primary">Run Waypoint Chain Duel</button>
      </div>
      <div id="wp-status" class="status-line" style="margin-bottom:12px;">Ready.</div>

      <div class="kpi-strip">
        <div class="kpi">
          <div class="kpi-label">Plain MSE Stage Handoff</div>
          <div class="kpi-value" style="color:var(--crimson)" id="wp-mse-handoff">--</div>
          <div class="kpi-sub" id="wp-mse-final">Stage-2 final --</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Trust-MPC Stage Handoff</div>
          <div class="kpi-value" style="color:var(--emerald)" id="wp-trust-handoff">--</div>
          <div class="kpi-sub" id="wp-trust-final">Stage-2 final --</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Warm-Start Resets</div>
          <div class="kpi-value" style="color:var(--amber)" id="wp-resets">--</div>
          <div class="kpi-sub">cleared on stage switch</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Stay-Gate / Jensen</div>
          <div class="kpi-value" style="color:var(--blue)" id="wp-gates">--</div>
          <div class="kpi-sub">stay / barrier splits</div>
        </div>
      </div>

      <div class="duel-stage">
        <div class="frame-card">
          <img id="wp-img-subgoal" alt="Active subgoal photo" />
          <div class="frame-caption"><span>Active Subgoal Photo</span><span id="wp-stage-badge" class="badge amber">Stage 1/2</span></div>
        </div>
        <div class="frame-card">
          <img id="wp-img-base" alt="Plain MSE waypoint rollout" />
          <div class="frame-caption"><span>Plain MSE + Waypoint CEM</span><span id="wp-cap-base">--</span></div>
        </div>
        <div class="frame-card">
          <img id="wp-img-trust" alt="Calibrated Trust-MPC waypoint rollout" />
          <div class="frame-caption"><span>Calibrated Head + Waypoint Trust-MPC</span><span id="wp-cap-trust">--</span></div>
        </div>
      </div>

      <div class="scrubber-bar">
        <button id="btn-wp-play" class="btn small" disabled>Play</button>
        <input id="wp-scrubber" type="range" min="0" max="0" value="0" disabled />
        <span id="wp-frame-label" class="status-line">Step 0 / 0</span>
      </div>

      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Replan</th>
              <th>MSE Stage</th>
              <th>Trust Stage</th>
              <th>Cost Ratio</th>
              <th>Warm Reset</th>
              <th>Trust Stage-2 Err (px)</th>
              <th>Jensen Gate</th>
              <th>Stay Gate</th>
              <th>Unrolls</th>
            </tr>
          </thead>
          <tbody id="wp-telemetry-body">
            <tr><td colspan="9" style="text-align:center;color:var(--muted);">Run a Multi-Stage Waypoint Chain duel to inspect latent handoff telemetry.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- 3D Meta-World Tab -->
  <section id="tab-metaworld" class="hidden">
    <div class="panel">
      <div class="panel-title">
        <span>3D Meta-World Sawyer Push-v3 Counterfactual Inspector</span>
        <span id="mw-status-badge" class="badge">checking...</span>
      </div>
      <div id="mw-disabled-notice" class="preset-desc hidden">
        Meta-World Sawyer Push-v3 backend is not active. Restart the studio with <code>--enable-metaworld</code> to inspect live 3D MuJoCo renders and patch heatmaps.
      </div>
      <div id="mw-controls" class="mw-controls">
        <div class="field">
          <label for="mw-seed">Task Seed</label>
          <input id="mw-seed" type="number" value="1000" />
        </div>
        <div class="field">
          <label for="mw-angle">Distractor Arm Angle (&deg;)</label>
          <input id="mw-angle" type="number" value="135" min="0" max="360" />
        </div>
        <div class="field">
          <label for="mw-shift">Puck Shift (cm)</label>
          <input id="mw-shift" type="number" value="5.0" step="0.5" min="1.0" max="12.0" />
        </div>
        <div class="field">
          <label for="mw-head-select">Meta-World Head</label>
          <select id="mw-head-select"></select>
        </div>
        <button id="btn-mw-inspect" class="btn primary">Inspect 3D Patch Geometry</button>
      </div>
      <div id="mw-status" class="status-line" style="margin-bottom:12px;">Ready.</div>

      <div class="kpi-strip">
        <div class="kpi">
          <div class="kpi-label">Plain MSE Ratio</div>
          <div class="kpi-value" id="mw-mse-ratio">--</div>
          <div class="kpi-sub">puck shift / arm move</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Learned Head Ratio</div>
          <div class="kpi-value" style="color:var(--emerald)" id="mw-head-ratio">--</div>
          <div class="kpi-sub">puck shift / arm move</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Selectivity Gain</div>
          <div class="kpi-value" style="color:var(--amber)" id="mw-gain">--</div>
          <div class="kpi-sub" id="mw-rank">eff. rank --</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Arm Displacement</div>
          <div class="kpi-value" id="mw-arm-cm">--</div>
          <div class="kpi-sub" id="mw-puck-cm">puck shift --</div>
        </div>
      </div>

      <div class="heatmap-grid" style="margin-bottom:12px;">
        <div class="frame-card">
          <img id="mw-img-goal" alt="3D goal frame" />
          <div class="frame-caption"><span>Reference Goal</span><span>3D RGB</span></div>
        </div>
        <div class="frame-card">
          <img id="mw-img-mse-nuis" alt="3D MSE arm nuisance" />
          <div class="frame-caption"><span>Plain MSE: Moved Arm</span><span id="mw-cap-mse-nuis">--</span></div>
        </div>
        <div class="frame-card">
          <img id="mw-img-mse-task" alt="3D MSE puck shift" />
          <div class="frame-caption"><span>Plain MSE: Shifted Puck</span><span id="mw-cap-mse-task">--</span></div>
        </div>
      </div>

      <div class="heatmap-grid">
        <div class="frame-card">
          <img id="mw-img-nuis-raw" alt="3D moved arm frame" />
          <div class="frame-caption"><span>Counterfactual Arm Pose</span><span>3D RGB</span></div>
        </div>
        <div class="frame-card">
          <img id="mw-img-head-nuis" alt="3D Head arm nuisance" />
          <div class="frame-caption"><span>Head L: Moved Arm</span><span id="mw-cap-head-nuis">--</span></div>
        </div>
        <div class="frame-card">
          <img id="mw-img-head-task" alt="3D Head puck shift" />
          <div class="frame-caption"><span>Head L: Shifted Puck</span><span id="mw-cap-head-task">--</span></div>
        </div>
      </div>
    </div>
  </section>
</main>

<script>
(() => {
  const ARENA = 512;
  const canvas = document.getElementById("arena");
  const ctx = canvas.getContext("2d");

  const state = {
    heads: [],
    presets: [],
    waypoint_presets: [],
    active_waypoint_preset: null,
    cur_agent: [145, 340],
    cur_block: [230, 290, 0.35],
    goal_agent: [435, 95],
    goal_block: [265, 250, 0.65],
    dragging: null,
    duel: null,
    playTimer: null,
    frameIdx: 0,
    waypointChain: null,
    wpPlayTimer: null,
    wpFrameIdx: 0,
  };

  function toCanvas(v) { return (v / ARENA) * canvas.width; }
  function toArena(v) { return Math.max(35, Math.min(477, (v / canvas.width) * ARENA)); }

  function drawT(x, y, angle, fillStyle, strokeStyle, dashed) {
    ctx.save();
    ctx.translate(toCanvas(x), toCanvas(y));
    ctx.rotate(angle);
    const s = canvas.width / ARENA;
    ctx.beginPath();
    ctx.rect(-60 * s, 0, 120 * s, 30 * s);
    ctx.rect(-15 * s, 30 * s, 30 * s, 90 * s);
    ctx.fillStyle = fillStyle;
    ctx.fill();
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawArena() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#17191b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle 16x16 patch grid matching ViT patches
    ctx.strokeStyle = "rgba(255,255,255,0.045)";
    ctx.lineWidth = 1;
    const step = canvas.width / 16;
    for (let i = 1; i < 16; i++) {
      const p = Math.round(i * step) + 0.5;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(canvas.width, p); ctx.stroke();
    }

    // Target T & goal distractor pusher
    drawT(state.goal_block[0], state.goal_block[1], state.goal_block[2], "rgba(230,157,60,0.22)", "#e69d3c", true);
    drawT(state.cur_block[0], state.cur_block[1], state.cur_block[2], "rgba(156,164,176,0.82)", "#cfd5dd", false);

    // Goal distractor ring
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(toCanvas(state.goal_agent[0]), toCanvas(state.goal_agent[1]), toCanvas(15), 0, Math.PI * 2);
    ctx.strokeStyle = "#5da4e8";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    // Initial pusher disc
    ctx.beginPath();
    ctx.arc(toCanvas(state.cur_agent[0]), toCanvas(state.cur_agent[1]), toCanvas(15), 0, Math.PI * 2);
    ctx.fillStyle = "#5da4e8";
    ctx.fill();
    ctx.strokeStyle = "#d9ecff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function syncSliders() {
    const initDeg = Math.round(((state.cur_block[2] * 180 / Math.PI) % 360 + 360) % 360);
    const goalDeg = Math.round(((state.goal_block[2] * 180 / Math.PI) % 360 + 360) % 360);
    document.getElementById("init-rot").value = initDeg;
    document.getElementById("init-rot-out").textContent = initDeg + "\u00b0";
    document.getElementById("goal-rot").value = goalDeg;
    document.getElementById("goal-rot-out").textContent = goalDeg + "\u00b0";
  }

  function pointerCoords(ev) {
    const rect = canvas.getBoundingClientRect();
    return [
      ((ev.clientX - rect.left) / rect.width) * canvas.width,
      ((ev.clientY - rect.top) / rect.height) * canvas.height,
    ];
  }

  canvas.addEventListener("pointerdown", (ev) => {
    const [cx, cy] = pointerCoords(ev);
    const handles = [
      { key: "cur_agent", x: toCanvas(state.cur_agent[0]), y: toCanvas(state.cur_agent[1]) },
      { key: "goal_agent", x: toCanvas(state.goal_agent[0]), y: toCanvas(state.goal_agent[1]) },
      { key: "cur_block", x: toCanvas(state.cur_block[0]), y: toCanvas(state.cur_block[1]) },
      { key: "goal_block", x: toCanvas(state.goal_block[0]), y: toCanvas(state.goal_block[1]) },
    ];
    let best = null;
    let bestDist = 34;
    for (const h of handles) {
      const d = Math.hypot(cx - h.x, cy - h.y);
      if (d < bestDist) { bestDist = d; best = h.key; }
    }
    if (best) {
      state.dragging = best;
      canvas.setPointerCapture(ev.pointerId);
    }
  });

  canvas.addEventListener("pointermove", (ev) => {
    if (!state.dragging) return;
    const [cx, cy] = pointerCoords(ev);
    const ax = toArena(cx);
    const ay = toArena(cy);
    if (state.dragging === "cur_agent") state.cur_agent = [ax, ay];
    if (state.dragging === "goal_agent") state.goal_agent = [ax, ay];
    if (state.dragging === "cur_block") state.cur_block = [Math.max(80, Math.min(432, ax)), Math.max(80, Math.min(432, ay)), state.cur_block[2]];
    if (state.dragging === "goal_block") state.goal_block = [Math.max(80, Math.min(432, ax)), Math.max(80, Math.min(432, ay)), state.goal_block[2]];
    drawArena();
  });

  canvas.addEventListener("pointerup", () => { state.dragging = null; });

  document.getElementById("init-rot").addEventListener("input", (e) => {
    const deg = Number(e.target.value);
    state.cur_block[2] = (deg * Math.PI) / 180;
    document.getElementById("init-rot-out").textContent = deg + "\u00b0";
    drawArena();
  });
  document.getElementById("goal-rot").addEventListener("input", (e) => {
    const deg = Number(e.target.value);
    state.goal_block[2] = (deg * Math.PI) / 180;
    document.getElementById("goal-rot-out").textContent = deg + "\u00b0";
    drawArena();
  });

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.tab;
      document.getElementById("tab-pusht").classList.toggle("hidden", target !== "pusht");
      document.getElementById("tab-waypoints").classList.toggle("hidden", target !== "waypoints");
      document.getElementById("tab-metaworld").classList.toggle("hidden", target !== "metaworld");
    });
  });

  function updateRankBadge() {
    const sel = document.getElementById("head-select").value;
    const item = state.heads.find((h) => h.name === sel);
    document.getElementById("rank-badge").textContent = item ? `eff. rank ${item.effective_rank}` : "eff. rank --";
  }
  document.getElementById("head-select").addEventListener("change", updateRankBadge);

  function fmt(v, digits = 3) {
    if (v === undefined || v === null || Number.isNaN(v)) return "--";
    return Number(v).toFixed(digits);
  }

  function applyPreset(preset) {
    state.cur_agent = [...preset.initial_agent];
    state.cur_block = [...preset.initial_block];
    state.goal_agent = [...preset.goal_agent];
    state.goal_block = [...preset.goal_block];
    document.getElementById("replans-input").value = preset.replans;
    document.getElementById("seed-input").value = preset.seed;
    document.getElementById("preset-desc").textContent = preset.description;
    syncSliders();
    drawArena();
  }

  async function runInspect() {
    const status = document.getElementById("pusht-status");
    status.textContent = "Computing 16x16 patch energies...";
    try {
      const resp = await fetch("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cur_agent: state.cur_agent,
          cur_block: state.cur_block,
          goal_agent: state.goal_agent,
          goal_block: state.goal_block,
          head_name: document.getElementById("head-select").value,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Inspect request failed");

      document.getElementById("geom-subtitle").textContent = `Head: ${data.head_name}`;
      document.getElementById("head-row-title").textContent = `Calibrated Head (${data.head_name})`;
      document.getElementById("kpi-mse-ratio").textContent = `${fmt(data.plain_mse.task_to_nuisance_ratio, 2)}x`;
      document.getElementById("kpi-head-ratio").textContent = `${fmt(data.learned_head.task_to_nuisance_ratio, 2)}x`;
      document.getElementById("kpi-gain").textContent = `${fmt(data.learned_head.ratio_gain_vs_mse, 2)}x`;
      document.getElementById("kpi-rank-sub").textContent = `eff. rank ${fmt(data.effective_rank, 2)}`;
      document.getElementById("kpi-phys-gap").textContent = `${fmt(data.physical.position_error, 1)} px`;
      document.getElementById("kpi-phys-sub").textContent = `angle ${fmt(data.physical.angle_error_deg, 1)}\u00b0`;

      document.getElementById("mse-energy-line").textContent =
        `nuisance=${fmt(data.plain_mse.nuisance_energy, 4)} | shift=${fmt(data.plain_mse.task_energy, 4)}`;
      document.getElementById("head-energy-line").textContent =
        `nuisance=${fmt(data.learned_head.nuisance_energy, 4)} | shift=${fmt(data.learned_head.task_energy, 4)}`;

      document.getElementById("img-mse-cur").src = data.plain_mse.overlays.cur_vs_goal;
      document.getElementById("img-mse-nuis").src = data.plain_mse.overlays.nuisance;
      document.getElementById("img-mse-task").src = data.plain_mse.overlays.task_shift;
      document.getElementById("cap-mse-nuis").textContent = fmt(data.plain_mse.nuisance_energy, 4);
      document.getElementById("cap-mse-task").textContent = fmt(data.plain_mse.task_energy, 4);

      document.getElementById("img-head-cur").src = data.learned_head.overlays.cur_vs_goal;
      document.getElementById("img-head-nuis").src = data.learned_head.overlays.nuisance;
      document.getElementById("img-head-task").src = data.learned_head.overlays.task_shift;
      document.getElementById("cap-head-nuis").textContent = fmt(data.learned_head.nuisance_energy, 4);
      document.getElementById("cap-head-task").textContent = fmt(data.learned_head.task_energy, 4);

      status.textContent = "Patch geometry updated.";
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    }
  }

  function showDuelFrame(idx) {
    if (!state.duel) return;
    const maxIdx = Math.max(0, state.duel.trust_mpc.frames.length - 1);
    state.frameIdx = Math.max(0, Math.min(maxIdx, idx));
    document.getElementById("duel-img-base").src = state.duel.baseline.frames[state.frameIdx];
    document.getElementById("duel-img-trust").src = state.duel.trust_mpc.frames[state.frameIdx];
    document.getElementById("duel-scrubber").value = state.frameIdx;
    document.getElementById("duel-frame-label").textContent = `Step ${state.frameIdx} / ${maxIdx}`;
  }

  function stopPlayback() {
    if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
    document.getElementById("btn-play").textContent = "Play";
  }

  async function runDuel() {
    stopPlayback();
    const status = document.getElementById("pusht-status");
    const btn = document.getElementById("btn-duel");
    btn.disabled = true;
    status.textContent = "Running paired closed-loop duel...";
    try {
      const resp = await fetch("/api/duel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_agent: state.cur_agent,
          initial_block: state.cur_block,
          goal_agent: state.goal_agent,
          goal_block: state.goal_block,
          head_name: document.getElementById("head-select").value,
          replans: Number(document.getElementById("replans-input").value),
          seed: Number(document.getElementById("seed-input").value),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Duel request failed");

      state.duel = data;
      const bSum = data.baseline.summary;
      const tSum = data.trust_mpc.summary;

      document.getElementById("duel-img-goal").src = data.goal_image;
      document.getElementById("duel-mse-final").textContent = `${fmt(bSum.final_position_error, 1)} px`;
      document.getElementById("duel-mse-drift").textContent = `best ${fmt(bSum.best_position_error, 1)} | drift ${fmt(bSum.post_peak_drift, 1)}`;
      document.getElementById("duel-trust-final").textContent = `${fmt(tSum.final_position_error, 1)} px`;
      document.getElementById("duel-trust-drift").textContent = `best ${fmt(tSum.best_position_error, 1)} | drift ${fmt(tSum.post_peak_drift, 1)}`;
      document.getElementById("duel-jensen-count").textContent = `${tSum.jensen_violations} / ${data.replans}`;
      document.getElementById("duel-stay-count").textContent = `${tSum.stay_gate_triggers} / ${data.replans}`;
      document.getElementById("duel-parity-badge").textContent = `unrolls: ${bSum.unrolls} vs ${tSum.unrolls}`;
      document.getElementById("duel-cap-base").textContent = `final ${fmt(bSum.final_position_error, 1)} px`;
      document.getElementById("duel-cap-trust").textContent = `final ${fmt(tSum.final_position_error, 1)} px`;

      const maxIdx = Math.max(0, data.trust_mpc.frames.length - 1);
      const scrubber = document.getElementById("duel-scrubber");
      scrubber.max = maxIdx;
      scrubber.disabled = false;
      document.getElementById("btn-play").disabled = false;
      const dl = document.getElementById("btn-download-gif");
      dl.classList.remove("hidden");
      dl.href = `/api/duel.gif?ts=${Date.now()}`;
      showDuelFrame(maxIdx);

      const tbody = document.getElementById("telemetry-body");
      tbody.innerHTML = "";
      for (let i = 0; i < data.trust_mpc.replan_telemetry.length; i++) {
        const bRow = data.baseline.replan_telemetry[i];
        const tRow = data.trust_mpc.replan_telemetry[i];
        const tr = document.createElement("tr");
        const jensenBadge = tRow.jensen_violation
          ? '<span class="badge crimson">SPLIT</span>'
          : '<span class="badge">UNIMODAL</span>';
        const stayBadge = tRow.used_stay_gate
          ? '<span class="badge emerald">HOLD</span>'
          : '<span class="badge">MOVE</span>';
        tr.innerHTML = `
          <td>#${tRow.replan}</td>
          <td>${fmt(bRow.position_error, 1)}</td>
          <td>${fmt(tRow.position_error, 1)}</td>
          <td>${fmt(tRow.angle_error_deg, 1)}</td>
          <td>${fmt(tRow.hold_cost, 4)}</td>
          <td>${fmt(tRow.elite_mean_cost, 4)}</td>
          <td>${fmt(tRow.selected_cost, 4)}</td>
          <td>${jensenBadge}</td>
          <td>${stayBadge}</td>
          <td>${tRow.unrolls}</td>
        `;
        tbody.appendChild(tr);
      }

      status.textContent = "Duel complete.";
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  document.getElementById("btn-inspect").addEventListener("click", runInspect);
  document.getElementById("btn-duel").addEventListener("click", runDuel);
  document.getElementById("duel-scrubber").addEventListener("input", (e) => {
    stopPlayback();
    showDuelFrame(Number(e.target.value));
  });
  document.getElementById("btn-play").addEventListener("click", () => {
    if (!state.duel) return;
    if (state.playTimer) { stopPlayback(); return; }
    const maxIdx = state.duel.trust_mpc.frames.length - 1;
    if (state.frameIdx >= maxIdx) showDuelFrame(0);
    document.getElementById("btn-play").textContent = "Pause";
    state.playTimer = setInterval(() => {
      if (state.frameIdx >= maxIdx) { stopPlayback(); return; }
      showDuelFrame(state.frameIdx + 1);
    }, 240);
  });

  // Multi-Stage Waypoint Chain
  function applyWaypointPreset(preset) {
    state.active_waypoint_preset = preset;
    document.getElementById("wp-replans").value = preset.replans;
    document.getElementById("wp-arrival-ratio").value = preset.arrival_ratio;
    document.getElementById("wp-seed").value = preset.seed;
    document.getElementById("wp-preset-desc").textContent = preset.description;
  }

  function showWaypointFrame(idx) {
    if (!state.waypointChain) return;
    const data = state.waypointChain;
    const maxIdx = Math.max(0, data.trust_mpc.frames.length - 1);
    state.wpFrameIdx = Math.max(0, Math.min(maxIdx, idx));
    const stageIdx = data.trust_mpc.frame_stages[state.wpFrameIdx] || 0;
    document.getElementById("wp-img-subgoal").src = data.waypoint_images[stageIdx];
    document.getElementById("wp-stage-badge").textContent = `Stage ${stageIdx + 1}/${data.num_waypoints}`;
    document.getElementById("wp-img-base").src = data.baseline.frames[state.wpFrameIdx];
    document.getElementById("wp-img-trust").src = data.trust_mpc.frames[state.wpFrameIdx];
    document.getElementById("wp-scrubber").value = state.wpFrameIdx;
    document.getElementById("wp-frame-label").textContent = `Step ${state.wpFrameIdx} / ${maxIdx}`;
  }

  function stopWaypointPlayback() {
    if (state.wpPlayTimer) { clearInterval(state.wpPlayTimer); state.wpPlayTimer = null; }
    document.getElementById("btn-wp-play").textContent = "Play";
  }

  async function runWaypointChain() {
    if (!state.active_waypoint_preset) return;
    stopWaypointPlayback();
    const status = document.getElementById("wp-status");
    const btn = document.getElementById("btn-wp-run");
    btn.disabled = true;
    status.textContent = "Running paired Multi-Stage Waypoint Chain...";
    try {
      const preset = state.active_waypoint_preset;
      const resp = await fetch("/api/waypoint_chain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_agent: preset.initial_agent,
          initial_block: preset.initial_block,
          waypoints: preset.waypoints,
          head_name: document.getElementById("head-select").value,
          replans: Number(document.getElementById("wp-replans").value),
          arrival_ratio: Number(document.getElementById("wp-arrival-ratio").value),
          seed: Number(document.getElementById("wp-seed").value),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Waypoint chain request failed");

      state.waypointChain = data;
      const bSum = data.baseline.summary;
      const tSum = data.trust_mpc.summary;
      document.getElementById("wp-mse-handoff").textContent = bSum.handoff_triggered ? `Step ${bSum.handoff_step + 1}` : "STALLED";
      document.getElementById("wp-mse-final").textContent = `Stage-2 final ${fmt(bSum.final_position_error, 1)} px`;
      document.getElementById("wp-trust-handoff").textContent = tSum.handoff_triggered ? `Step ${tSum.handoff_step + 1}` : "STALLED";
      document.getElementById("wp-trust-final").textContent = `Stage-2 final ${fmt(tSum.final_position_error, 1)} px`;
      document.getElementById("wp-resets").textContent = `${tSum.warm_start_resets}`;
      document.getElementById("wp-gates").textContent = `${tSum.stay_gate_triggers} / ${tSum.jensen_violations}`;
      document.getElementById("wp-parity-badge").textContent = `unrolls: ${bSum.unrolls} vs ${tSum.unrolls}`;
      document.getElementById("wp-cap-base").textContent = `final ${fmt(bSum.final_position_error, 1)} px`;
      document.getElementById("wp-cap-trust").textContent = `final ${fmt(tSum.final_position_error, 1)} px`;

      const maxIdx = Math.max(0, data.trust_mpc.frames.length - 1);
      const scrubber = document.getElementById("wp-scrubber");
      scrubber.max = maxIdx;
      scrubber.disabled = false;
      document.getElementById("btn-wp-play").disabled = false;
      const dl = document.getElementById("btn-wp-download-gif");
      dl.classList.remove("hidden");
      dl.href = `/api/waypoint.gif?ts=${Date.now()}`;
      showWaypointFrame(maxIdx);

      const tbody = document.getElementById("wp-telemetry-body");
      tbody.innerHTML = "";
      for (let i = 0; i < data.trust_mpc.replan_telemetry.length; i++) {
        const bRow = data.baseline.replan_telemetry[i];
        const tRow = data.trust_mpc.replan_telemetry[i];
        const tr = document.createElement("tr");
        const resetBadge = tRow.warm_start_reset ? '<span class="badge amber">RESET</span>' : '<span class="badge">KEEP</span>';
        const jensenBadge = tRow.jensen_violation ? '<span class="badge crimson">SPLIT</span>' : '<span class="badge">UNIMODAL</span>';
        const stayBadge = tRow.used_stay_gate ? '<span class="badge emerald">HOLD</span>' : '<span class="badge">MOVE</span>';
        tr.innerHTML = `
          <td>#${tRow.replan}</td>
          <td>Stage ${bRow.stage_index + 1}</td>
          <td>Stage ${tRow.stage_index + 1}</td>
          <td>${fmt(tRow.cost_ratio, 3)}</td>
          <td>${resetBadge}</td>
          <td>${fmt(tRow.final_stage_error_px, 1)}</td>
          <td>${jensenBadge}</td>
          <td>${stayBadge}</td>
          <td>${tRow.unrolls}</td>
        `;
        tbody.appendChild(tr);
      }
      status.textContent = "Multi-Stage Waypoint Chain complete.";
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  document.getElementById("btn-wp-run").addEventListener("click", runWaypointChain);
  document.getElementById("wp-scrubber").addEventListener("input", (e) => {
    stopWaypointPlayback();
    showWaypointFrame(Number(e.target.value));
  });
  document.getElementById("btn-wp-play").addEventListener("click", () => {
    if (!state.waypointChain) return;
    if (state.wpPlayTimer) { stopWaypointPlayback(); return; }
    const maxIdx = state.waypointChain.trust_mpc.frames.length - 1;
    if (state.wpFrameIdx >= maxIdx) showWaypointFrame(0);
    document.getElementById("btn-wp-play").textContent = "Pause";
    state.wpPlayTimer = setInterval(() => {
      if (state.wpFrameIdx >= maxIdx) { stopWaypointPlayback(); return; }
      showWaypointFrame(state.wpFrameIdx + 1);
    }, 240);
  });

  // 3D Meta-World inspection
  document.getElementById("btn-mw-inspect").addEventListener("click", async () => {
    const mwStatus = document.getElementById("mw-status");
    mwStatus.textContent = "Rendering 3D Sawyer Push-v3 counterfactuals...";
    try {
      const resp = await fetch("/api/metaworld_inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: Number(document.getElementById("mw-seed").value),
          hand_offset_angle_deg: Number(document.getElementById("mw-angle").value),
          shift_cm: Number(document.getElementById("mw-shift").value),
          head_name: document.getElementById("mw-head-select").value,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Meta-World request failed");

      document.getElementById("mw-mse-ratio").textContent = `${fmt(data.plain_mse.task_to_nuisance_ratio, 2)}x`;
      document.getElementById("mw-head-ratio").textContent = `${fmt(data.learned_head.task_to_nuisance_ratio, 2)}x`;
      document.getElementById("mw-gain").textContent = `${fmt(data.learned_head.ratio_gain_vs_mse, 2)}x`;
      document.getElementById("mw-rank").textContent = `eff. rank ${fmt(data.effective_rank, 2)}`;
      document.getElementById("mw-arm-cm").textContent = `${fmt(data.physical.hand_displacement_cm, 1)} cm`;
      document.getElementById("mw-puck-cm").textContent = `puck shift ${fmt(data.physical.puck_shift_cm, 1)} cm`;

      document.getElementById("mw-img-goal").src = data.frames.goal;
      document.getElementById("mw-img-nuis-raw").src = data.frames.nuisance;
      document.getElementById("mw-img-mse-nuis").src = data.plain_mse.overlays.nuisance;
      document.getElementById("mw-img-mse-task").src = data.plain_mse.overlays.task_shift;
      document.getElementById("mw-img-head-nuis").src = data.learned_head.overlays.nuisance;
      document.getElementById("mw-img-head-task").src = data.learned_head.overlays.task_shift;
      document.getElementById("mw-cap-mse-nuis").textContent = fmt(data.plain_mse.nuisance_energy, 4);
      document.getElementById("mw-cap-mse-task").textContent = fmt(data.plain_mse.task_energy, 4);
      document.getElementById("mw-cap-head-nuis").textContent = fmt(data.learned_head.nuisance_energy, 4);
      document.getElementById("mw-cap-head-task").textContent = fmt(data.learned_head.task_energy, 4);
      mwStatus.textContent = "3D patch geometry updated.";
    } catch (err) {
      mwStatus.textContent = `Error: ${err.message}`;
    }
  });

  async function init() {
    const resp = await fetch("/api/state");
    const info = await resp.json();
    state.heads = info.pusht.heads;
    state.presets = info.pusht.presets;
    state.waypoint_presets = info.pusht.waypoint_presets || [];

    const headSelect = document.getElementById("head-select");
    headSelect.innerHTML = "";
    for (const h of state.heads) {
      const opt = document.createElement("option");
      opt.value = h.name;
      opt.textContent = `${h.name} (r=${h.effective_rank})`;
      if (h.name === info.pusht.default_head) opt.selected = true;
      headSelect.appendChild(opt);
    }
    updateRankBadge();

    const presetRow = document.getElementById("preset-buttons");
    presetRow.innerHTML = "";
    for (const preset of state.presets) {
      const btn = document.createElement("button");
      btn.className = "btn small";
      btn.textContent = preset.name;
      btn.addEventListener("click", () => applyPreset(preset));
      presetRow.appendChild(btn);
    }
    if (state.presets.length > 0) applyPreset(state.presets[0]);

    const wpPresetRow = document.getElementById("wp-preset-buttons");
    wpPresetRow.innerHTML = "";
    for (const wpPreset of state.waypoint_presets) {
      const btn = document.createElement("button");
      btn.className = "btn small";
      btn.textContent = wpPreset.name;
      btn.addEventListener("click", () => applyWaypointPreset(wpPreset));
      wpPresetRow.appendChild(btn);
    }
    if (state.waypoint_presets.length > 0) applyWaypointPreset(state.waypoint_presets[0]);

    const mwBadge = document.getElementById("mw-status-badge");
    if (info.metaworld.enabled) {
      mwBadge.textContent = "active";
      mwBadge.classList.add("emerald");
      const mwSel = document.getElementById("mw-head-select");
      mwSel.innerHTML = "";
      for (const h of info.metaworld.heads) {
        const opt = document.createElement("option");
        opt.value = h.name;
        opt.textContent = `${h.name} (r=${h.effective_rank})`;
        if (h.name === info.metaworld.default_head) opt.selected = true;
        mwSel.appendChild(opt);
      }
    } else {
      mwBadge.textContent = "disabled";
      document.getElementById("mw-disabled-notice").classList.remove("hidden");
      document.getElementById("btn-mw-inspect").disabled = true;
    }

    drawArena();
    runInspect();
  }

  init();
})();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- HTTP server


def create_studio_server(
    host: str,
    port: int,
    backend: PlanningStudioBackend,
    *,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    """Create a multi-threaded HTTP server exposing the studio UI and JSON API."""

    class StudioRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            if not quiet:
                super().log_message(format, *args)

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self._send_bytes(status, raw, "application/json; charset=utf-8")

        def _read_json_body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON request body: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("JSON request body must be an object")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path

            if route == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    STUDIO_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if route == "/api/state":
                self._send_json(HTTPStatus.OK, backend.get_state())
                return

            if route == "/api/duel.gif":
                query = parse_qs(parsed.query)
                fps_raw = query.get("fps", ["4"])[0]
                try:
                    fps = int(fps_raw)
                    gif_bytes = backend.get_duel_gif_bytes(fps=fps)
                except LookupError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_bytes(
                    HTTPStatus.OK,
                    gif_bytes,
                    "image/gif",
                    extra_headers={"Content-Disposition": 'attachment; filename="planning_duel.gif"'},
                )
                return

            if route == "/api/waypoint.gif":
                query = parse_qs(parsed.query)
                fps_raw = query.get("fps", ["4"])[0]
                try:
                    fps = int(fps_raw)
                    gif_bytes = backend.get_waypoint_gif_bytes(fps=fps)
                except LookupError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_bytes(
                    HTTPStatus.OK,
                    gif_bytes,
                    "image/gif",
                    extra_headers={"Content-Disposition": 'attachment; filename="waypoint_chain.gif"'},
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                payload = self._read_json_body()
                if route == "/api/inspect":
                    self._send_json(HTTPStatus.OK, backend.inspect_pusht(payload))
                elif route == "/api/duel":
                    self._send_json(HTTPStatus.OK, backend.run_pusht_duel(payload))
                elif route == "/api/waypoint_chain":
                    self._send_json(HTTPStatus.OK, backend.run_pusht_waypoint_chain(payload))
                elif route == "/api/metaworld_inspect":
                    self._send_json(HTTPStatus.OK, backend.inspect_metaworld(payload))
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {route}"})
            except (KeyError, ValueError, TypeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), StudioRequestHandler)
    server.daemon_threads = True
    return server
