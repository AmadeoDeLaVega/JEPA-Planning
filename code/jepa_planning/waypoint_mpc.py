"""Multi-stage visual subgoal tracking and waypoint Trust-MPC for frozen DINO-WM.

Single-horizon (H = 2 chunks) MPC struggles on long-displacement or
around-the-corner maneuvers. Given an ordered sequence of visual subgoal
photographs ``[G_0, G_1, ..., G_{S-1}]`` (each ``[P, D]`` or ``[K, P, D]`` with
independent distractor pusher poses), this module advances across stages purely
from visual latent distances ``D_L(z_t, G_s)`` without reading simulator
coordinates:

- :class:`WaypointTracker` tracks the active subgoal ``s`` and triggers stage
  handoff ``s -> s + 1`` using the scale-invariant ratio rule
  ``D_L(z_t, G_s) <= arrival_ratio * D_L(z_entry, G_s)`` (or when the Trust-MPC
  stay gate triggers after positive progress
  ``D_L(z_t, G_s) < stay_gate_progress_ratio * D_L(z_entry, G_s)``).
- :func:`waypoint_trust_mpc_step` combines observation encoding, visual stage
  tracking, warm-start reset on stage handoff, and certified barrier-split
  Trust-MPC planning toward the active subgoal.
- :func:`compose_waypoint_gif_bytes` renders a side-by-side animated GIF
  ``[Active Subgoal Photo (Stage s/S) | Live Rollout]`` highlighting visual
  stage transitions as they occur.
"""

from __future__ import annotations

from collections.abc import Sequence
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from jepa_planning import trust_mpc as tm
from jepa_planning.calibrated_objectives import projected_multi_goal_distance


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous ``[H, W, 3]`` uint8 RGB array."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected RGB image of shape [H, W, 3], got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@torch.inference_mode()
def encode_visual_features(model, observation: dict) -> torch.Tensor:
    """Encode a single observation dict into float32 patch features ``[P, D]``."""
    visual = torch.from_numpy(_as_rgb_uint8(observation["visual"])).permute(2, 0, 1)[None, None]
    if "proprio" in observation and observation["proprio"] is not None:
        proprio = torch.from_numpy(np.asarray(observation["proprio"], dtype=np.float32))[None, None]
        encoded = model.encode({"visual": visual, "proprio": proprio})
    else:
        encoded = model.encode(visual)
    vis = (
        encoded["visual"]
        if isinstance(encoded, dict) or (hasattr(encoded, "keys") and "visual" in encoded.keys())
        else encoded
    )
    if vis.ndim >= 4:
        vis = vis[0, -1]
    elif vis.ndim == 3:
        vis = vis[0]
    if vis.ndim < 2:
        raise ValueError(f"Expected encoded visual features with at least 2 dims, got {tuple(vis.shape)}")
    return vis.reshape(-1, vis.shape[-1]).float()


class WaypointTracker:
    """Stateful, positive-scale-invariant visual subgoal state machine.

    Advances through ``waypoints = [G_0, ..., G_{S-1}]`` purely from projected
    visual distance ``D_L(z_t, G_s)``. Multiplying ``head`` by any positive
    scalar ``alpha > 0`` scales both ``current_cost`` and ``entry_cost`` by
    ``alpha**2``, preserving exact stage-transition invariance.
    """

    def __init__(
        self,
        waypoints: Sequence[torch.Tensor],
        head: torch.Tensor,
        *,
        arrival_ratio: float = 0.35,
        allow_stay_gate_advance: bool = True,
        stay_gate_progress_ratio: float = 0.85,
    ) -> None:
        if not waypoints:
            raise ValueError("waypoints must be a non-empty sequence of feature tensors")
        if head.ndim != 2 or head.shape[0] == 0 or head.shape[1] == 0:
            raise ValueError("head must have shape [r, D] with positive dimensions")
        if isinstance(arrival_ratio, bool) or not math.isfinite(float(arrival_ratio)):
            raise ValueError("arrival_ratio must be a finite number in [0, 1]")
        if not 0.0 <= float(arrival_ratio) <= 1.0:
            raise ValueError(f"arrival_ratio must lie in [0, 1], got {arrival_ratio}")
        if not isinstance(allow_stay_gate_advance, bool):
            raise ValueError("allow_stay_gate_advance must be a boolean")
        if isinstance(stay_gate_progress_ratio, bool) or not math.isfinite(
            float(stay_gate_progress_ratio)
        ):
            raise ValueError("stay_gate_progress_ratio must be a finite number in (0, 1]")
        if not 0.0 < float(stay_gate_progress_ratio) <= 1.0:
            raise ValueError(
                f"stay_gate_progress_ratio must lie in (0, 1], got {stay_gate_progress_ratio}"
            )

        self.head = head.detach().float()
        self.device = self.head.device
        self.feature_dim = int(self.head.shape[1])

        normalized_waypoints: list[torch.Tensor] = []
        expected_patches: int | None = None
        for idx, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, torch.Tensor):
                raise TypeError(f"waypoint[{idx}] must be a torch.Tensor")
            if waypoint.ndim not in (2, 3) or any(dim == 0 for dim in waypoint.shape):
                raise ValueError(
                    f"waypoint[{idx}] must have non-empty shape [P, D] or [K, P, D], got {tuple(waypoint.shape)}"
                )
            patches, dim = int(waypoint.shape[-2]), int(waypoint.shape[-1])
            if dim != self.feature_dim:
                raise ValueError(
                    f"waypoint[{idx}] feature dim {dim} does not match head dim {self.feature_dim}"
                )
            if expected_patches is None:
                expected_patches = patches
            elif patches != expected_patches:
                raise ValueError(
                    f"waypoint[{idx}] patch count {patches} does not match expected {expected_patches}"
                )
            normalized_waypoints.append(waypoint.detach().to(device=self.device, dtype=torch.float32))

        self.num_patches = int(expected_patches)
        self.waypoints: tuple[torch.Tensor, ...] = tuple(normalized_waypoints)
        self.arrival_ratio = float(arrival_ratio)
        self.allow_stay_gate_advance = allow_stay_gate_advance
        self.stay_gate_progress_ratio = float(stay_gate_progress_ratio)

        self.stage_index: int = 0
        self.entry_cost: float | None = None
        self.last_cost: float | None = None
        self.last_used_stay_gate: bool = False
        self.step_count: int = 0
        self.transitions: list[dict[str, object]] = []

    @property
    def num_waypoints(self) -> int:
        """Total number of ordered visual subgoals."""
        return len(self.waypoints)

    @property
    def active_waypoint(self) -> torch.Tensor:
        """Feature tensor ``[P, D]`` or ``[K, P, D]`` of the active subgoal."""
        return self.waypoints[self.stage_index]

    @property
    def is_final_stage(self) -> bool:
        """Whether the tracker is currently targeting the final subgoal."""
        return self.stage_index >= self.num_waypoints - 1

    def reset(self) -> None:
        """Reset stage state to waypoint 0."""
        self.stage_index = 0
        self.entry_cost = None
        self.last_cost = None
        self.last_used_stay_gate = False
        self.step_count = 0
        self.transitions.clear()

    def _prepare_current_features(self, current_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(current_features, torch.Tensor):
            raise TypeError("current_features must be a torch.Tensor")
        if current_features.ndim == 2:
            feat = current_features.unsqueeze(0)
        elif current_features.ndim == 3 and current_features.shape[0] == 1:
            feat = current_features
        else:
            raise ValueError(
                f"current_features must have shape [P, D] or [1, P, D], got {tuple(current_features.shape)}"
            )
        if feat.shape[1] != self.num_patches or feat.shape[2] != self.feature_dim:
            raise ValueError(
                f"current_features trailing shape {tuple(feat.shape[1:])} does not match "
                f"expected ({self.num_patches}, {self.feature_dim})"
            )
        return feat.to(device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def current_waypoint_cost(
        self,
        current_features: torch.Tensor,
        stage_index: int | None = None,
    ) -> float:
        """Return projected visual distance ``D_L(z_t, G_s)`` to the active (or specified) waypoint."""
        idx = self.stage_index if stage_index is None else int(stage_index)
        if not 0 <= idx < self.num_waypoints:
            raise IndexError(f"stage_index {idx} out of range for {self.num_waypoints} waypoints")
        feat = self._prepare_current_features(current_features)
        dist = projected_multi_goal_distance(feat, self.waypoints[idx], self.head, exact_mean=False)
        return float(dist[0].item())

    @torch.inference_mode()
    def update(
        self,
        current_features: torch.Tensor,
        used_stay_gate: bool = False,
    ) -> dict[str, object]:
        """Observe current visual features and advance to the next subgoal if arrival triggers.

        Args:
            current_features: Current patch features ``[P, D]`` (or ``[1, P, D]``).
            used_stay_gate: Whether the previous Trust-MPC step activated the
                hold-cost stay gate.

        Returns:
            Dictionary containing ``stage_index``, ``previous_stage``,
            ``num_waypoints``, ``advanced``, ``trigger``, ``current_cost``,
            ``entry_cost``, ``cost_ratio``, ``previous_stage_cost``,
            ``previous_stage_entry_cost``, and ``is_final_stage``.
        """
        cost = self.current_waypoint_cost(current_features)
        if self.entry_cost is None:
            self.entry_cost = cost

        previous_stage = self.stage_index
        cost_before_advance = cost
        entry_before_advance = self.entry_cost
        if entry_before_advance > 0.0:
            cost_ratio = cost_before_advance / entry_before_advance
        else:
            cost_ratio = 0.0 if cost_before_advance <= 0.0 else float("inf")

        advanced = False
        trigger: str | None = None

        if self.stage_index < self.num_waypoints - 1:
            ratio_reached = cost_before_advance <= self.arrival_ratio * entry_before_advance
            stay_reached = (
                self.allow_stay_gate_advance
                and bool(used_stay_gate)
                and cost_before_advance < self.stay_gate_progress_ratio * entry_before_advance
            )
            if ratio_reached or stay_reached:
                trigger = "arrival_ratio" if ratio_reached else "stay_gate"
                self.stage_index += 1
                advanced = True
                cost = self.current_waypoint_cost(current_features)
                self.entry_cost = cost
                cost_ratio = 1.0 if self.entry_cost > 0.0 else 0.0
                self.transitions.append(
                    {
                        "step": self.step_count,
                        "from_stage": previous_stage,
                        "to_stage": self.stage_index,
                        "trigger": trigger,
                        "previous_entry_cost": float(entry_before_advance),
                        "arrival_cost": float(cost_before_advance),
                        "arrival_ratio_observed": (
                            float(cost_before_advance / entry_before_advance)
                            if entry_before_advance > 0.0
                            else 0.0
                        ),
                        "new_entry_cost": float(self.entry_cost),
                    }
                )

        self.last_cost = cost
        self.step_count += 1
        return {
            "stage_index": self.stage_index,
            "previous_stage": previous_stage,
            "num_waypoints": self.num_waypoints,
            "advanced": advanced,
            "trigger": trigger,
            "current_cost": float(cost),
            "entry_cost": float(self.entry_cost),
            "cost_ratio": float(cost_ratio),
            "previous_stage_cost": float(cost_before_advance),
            "previous_stage_entry_cost": float(entry_before_advance),
            "is_final_stage": self.is_final_stage,
        }


@torch.inference_mode()
def waypoint_trust_mpc_step(
    model,
    observation: dict,
    tracker: WaypointTracker,
    normals: np.ndarray,
    hold_chunk: np.ndarray,
    previous_plan: np.ndarray | None = None,
    *,
    head: torch.Tensor | None = None,
    used_stay_gate: bool | None = None,
    num_elites: int = 8,
    action_clip: float = 2.0,
    batch_size: int = 64,
    stay_margin: float = 0.02,
    certify_barrier_split: bool = True,
    step_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Encode observation, update visual subgoal state, and run certified Trust-MPC.

    When ``tracker.update(...)`` advances to the next waypoint, the incoming
    ``previous_plan`` is cleared before planning so warm-starting never carries
    stale momentum from the completed stage into the new subgoal.
    """
    current_features = encode_visual_features(model, observation)
    prev_stay = tracker.last_used_stay_gate if used_stay_gate is None else bool(used_stay_gate)
    tracker_info = tracker.update(current_features, used_stay_gate=prev_stay)

    plan_for_planner = None if tracker_info["advanced"] else previous_plan
    active_head = tracker.head if head is None else head

    action_chunk, next_previous_plan, plan_info = tm.trust_mpc_plan(
        model,
        observation,
        tracker.active_waypoint,
        active_head,
        normals,
        hold_chunk,
        plan_for_planner,
        num_elites=num_elites,
        action_clip=action_clip,
        batch_size=batch_size,
        stay_margin=stay_margin,
        certify_barrier_split=certify_barrier_split,
        step_weights=step_weights,
    )
    tracker.last_used_stay_gate = bool(plan_info["used_stay_gate"])

    step_diagnostics: dict[str, object] = {
        **plan_info,
        **tracker_info,
        "warm_start_reset": bool(tracker_info["advanced"]),
        "used_previous_plan": bool(plan_for_planner is not None),
    }
    return action_chunk, next_previous_plan, step_diagnostics


def compose_waypoint_gif_bytes(
    frames: Sequence[np.ndarray],
    waypoint_frames: Sequence[np.ndarray],
    active_stages: Sequence[int],
    title_prefix: str = "Waypoint Trust-MPC",
    fps: int = 4,
) -> bytes:
    """Render ``[Active Subgoal Photo (Stage s/S) | Live Rollout]`` as an animated GIF.

    Visually highlights the active subgoal panel and stage-progress footer so
    mid-episode latent handoffs between subgoals are immediately clear.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    if not waypoint_frames:
        raise ValueError("waypoint_frames must be non-empty")
    if len(active_stages) != len(frames):
        raise ValueError(
            f"active_stages length ({len(active_stages)}) must match frames length ({len(frames)})"
        )
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")

    num_stages = len(waypoint_frames)
    panel_size = 224
    prepared_waypoints = [
        Image.fromarray(_as_rgb_uint8(wp), mode="RGB").resize(
            (panel_size, panel_size), resample=Image.Resampling.BILINEAR
        )
        for wp in waypoint_frames
    ]

    stage_indices: list[int] = []
    for idx, stage in enumerate(active_stages):
        if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
            raise ValueError(f"active_stages[{idx}] must be an integer")
        stage_int = int(stage)
        if not 0 <= stage_int < num_stages:
            raise ValueError(
                f"active_stages[{idx}]={stage_int} out of range for {num_stages} waypoint_frames"
            )
        stage_indices.append(stage_int)

    header_h = 34
    footer_h = 26
    gutter = 8
    margin = 8
    canvas_w = margin * 2 + panel_size * 2 + gutter
    canvas_h = header_h + panel_size + footer_h

    bg_color = (26, 28, 31)
    border_color = (56, 60, 68)
    amber = (230, 157, 60)
    emerald = (78, 198, 134)
    text_light = (232, 236, 242)
    text_muted = (150, 158, 168)
    font = ImageFont.load_default()

    total_frames = len(frames)
    stitched: list[Image.Image] = []

    for frame_idx, (raw_frame, stage_idx) in enumerate(zip(frames, stage_indices)):
        rollout_pil = Image.fromarray(_as_rgb_uint8(raw_frame), mode="RGB").resize(
            (panel_size, panel_size), resample=Image.Resampling.BILINEAR
        )
        subgoal_pil = prepared_waypoints[stage_idx]
        is_final = stage_idx == num_stages - 1
        accent = emerald if is_final else amber

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, header_h - 1), (canvas_w, header_h - 1)], fill=border_color, width=1)

        left_x = margin
        right_x = margin + panel_size + gutter
        panel_y = header_h

        canvas.paste(subgoal_pil, (left_x, panel_y))
        canvas.paste(rollout_pil, (right_x, panel_y))

        # Highlight border around active subgoal panel to emphasize stage switch.
        draw.rectangle(
            [left_x, panel_y, left_x + panel_size - 1, panel_y + panel_size - 1],
            outline=accent,
            width=3,
        )
        draw.rectangle(
            [right_x, panel_y, right_x + panel_size - 1, panel_y + panel_size - 1],
            outline=border_color,
            width=1,
        )

        left_title = f"Active Subgoal (Stage {stage_idx + 1}/{num_stages})"
        right_title = f"{title_prefix} | Live"
        draw.text((left_x + 4, 10), left_title, fill=accent, font=font)
        draw.text((right_x + 4, 10), right_title, fill=text_light, font=font)

        step_badge = f"t={frame_idx}/{total_frames - 1}"
        draw.text((canvas_w - margin - 54, 10), step_badge, fill=text_muted, font=font)

        # Stage progress bar in the footer.
        footer_y0 = header_h + panel_size + 5
        footer_y1 = canvas_h - 5
        bar_gap = 6
        usable_w = canvas_w - 2 * margin - bar_gap * (num_stages - 1)
        seg_w = max(24, usable_w // num_stages)
        for s in range(num_stages):
            sx0 = margin + s * (seg_w + bar_gap)
            sx1 = sx0 + seg_w
            if s < stage_idx:
                fill = (42, 92, 68)
                outline = emerald
                label_col = (196, 236, 212)
                tag = f"Stage {s + 1} [done]"
            elif s == stage_idx:
                fill = (52, 74, 58) if is_final else (78, 56, 26)
                outline = accent
                label_col = accent
                tag = f"Stage {s + 1} [active]"
            else:
                fill = (36, 39, 44)
                outline = border_color
                label_col = text_muted
                tag = f"Stage {s + 1}"
            draw.rectangle([sx0, footer_y0, sx1, footer_y1], fill=fill, outline=outline, width=1)
            draw.text((sx0 + 6, footer_y0 + 3), tag, fill=label_col, font=font)

        stitched.append(canvas)

    buffer = io.BytesIO()
    duration_ms = max(20, int(round(1000.0 / fps)))
    stitched[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=stitched[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()
