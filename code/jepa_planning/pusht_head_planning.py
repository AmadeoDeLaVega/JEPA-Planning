"""Planning evaluation of goal heads on PushT (plan step 6.2-6.5).

Tasks are the held-out goal-group targets: task ``i`` of a split starts from
the initial state that ``pusht_goal_groups.sample_target`` drew for target
``i`` (the first draw of its seeded stream), and its goal photos are that
group's photos, which show the same T pose with different held-out agents.
So the planner never sees a target or an agent position used to fit a head.

Physical metrics follow proposal Eq. 5 with q = the T position in the goal
photo: ``c_q`` is the T position error in px and regret is relative to the best
candidate in the shared set. Also reported: success within 10/20/50 px
(PushT stand-ins for 1/2/5 cm), pose success (20 px and 20 degrees, as in the
multi-goal experiments) and native success, which uses the env's own rule (T-area
coverage >= 0.95) applied to the photographed target rather than to the env's
fixed built-in goal.
"""

from __future__ import annotations

import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from shapely.ops import unary_union
import torch

from jepa_planning.goal_costs import linear_goal_distance
from jepa_planning.pusht_goal_groups import T_RECTANGLES
from jepa_planning.pusht_multi_goal import angle_error

POSITION_THRESHOLDS = (10.0, 20.0, 50.0)
NATIVE_COVERAGE = 0.95
# The T in its own frame, built from the same rectangles as the env's add_tee.
_T_LOCAL = unary_union([sg.box(x0, y0, x1, y1) for x0, x1, y0, y1 in T_RECTANGLES])


def t_polygon(pose: np.ndarray):
    """Shapely polygon of the T at ``pose = (x, y, angle)`` in env pixels."""
    x, y, angle = (float(v) for v in pose)
    # Rotate about the block's origin, then move it there: world = position + R(angle) @ local.
    rotated = sa.rotate(_T_LOCAL, angle, origin=(0, 0), use_radians=True)
    return sa.translate(rotated, x, y)


def coverage(pose: np.ndarray, target: np.ndarray) -> float:
    """Fraction of the target T's area covered by the T at ``pose`` (the env's reward rule)."""
    goal = t_polygon(target)
    return float(goal.intersection(t_polygon(pose)).area / goal.area)


def physical_metrics(poses: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pose metrics against the photographed target pose ``q`` (``poses`` is ``[N, 3]``)."""
    poses = np.asarray(poses, dtype=np.float64)
    position = np.linalg.norm(poses[:, :2] - np.asarray(target)[None, :2], axis=1)
    angle = angle_error(poses[:, 2], target[2])
    out = {
        "position_error": position,
        "angle_error_deg": np.rad2deg(angle),
        # 50 px per radian, the same weighting as the multi-goal experiments.
        "combined_error": np.sqrt(position**2 + (50.0 * angle) ** 2),
        "pose_success": ((position < 20.0) & (angle < np.pi / 9)).astype(int),  # 20 px and 20 degrees
        "coverage": np.array([coverage(pose, target) for pose in poses]),
    }
    out["native_success"] = (out["coverage"] >= NATIVE_COVERAGE).astype(int)
    for threshold in POSITION_THRESHOLDS:
        out[f"success_{threshold:g}px"] = (position < threshold).astype(int)
    return out


def selection_rows(costs: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """Metrics of the min-cost candidate, plus Eq. 5 regret on position and on combined error."""
    selected = int(np.argmin(costs))
    row = {"selected": selected, **{name: float(values[selected]) for name, values in metrics.items()}}
    # Eq. 5: error of the chosen candidate minus the best error available in the same set.
    row["regret"] = row["position_error"] - float(metrics["position_error"].min())
    row["combined_regret"] = row["combined_error"] - float(metrics["combined_error"].min())
    row["oracle_native_available"] = int(metrics["native_success"].any())  # could any candidate succeed?
    return row


def head_costs(candidates: torch.Tensor, goal: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Eq. 3 for candidates ``[N, P, D]`` and one goal ``[P, D]``; returns ``[N]``."""
    head = head.to(device=candidates.device, dtype=candidates.dtype)
    return linear_goal_distance(candidates, goal[None].to(candidates), head)  # goal broadcast over N
