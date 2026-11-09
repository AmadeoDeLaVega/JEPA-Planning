"""PushT goal-group logic checked against a pymunk-free stand-in environment."""

import copy

import numpy as np
import pytest
import yaml

from jepa_planning import ROOT
from jepa_planning import pusht_goal_groups as pg


class FakePushT:
    """Reset places agent/object exactly unless the agent is within 60 px of the object,
    in which case the object is pushed 3 px. Each step moves the object by the action."""

    def __init__(self):
        self.reset_to_state = None
        self.resets = 0

    def reset(self):
        self.resets += 1
        state = np.array(self.reset_to_state, dtype=np.float64)
        if np.linalg.norm(state[:2] - state[2:4]) < 60:
            state[2] += 3
        self.state = state
        return {"visual": np.full((4, 4, 3), self.resets % 255, np.uint8), "proprio": state[[0, 1, 5, 6]]}, state

    def step(self, action):
        self.state[2:4] += action
        return None, 0.0, False, {"state": self.state.copy()}


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "code/configs/pusht_goal_groups.yaml").read_text())


TOL = {"position": 0.5, "angle": 0.005}


def test_default_config_is_valid(cfg):
    pg.validate_config(cfg)


@pytest.mark.parametrize("edit", [
    lambda c: c["agents_per_target"].update(train=1),
    lambda c: c["agents_per_target"].update(val=17),
    lambda c: c["agent_bank"]["fractions"].update(train=0.9),
    lambda c: c["pose_tolerance"].update(position=0),
    lambda c: c.update(extra=1),
    lambda c: c["target"].pop("horizon"),
    lambda c: c.update(min_agent_clearance=-1),
])
def test_invalid_configs(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg)
    with pytest.raises(ValueError):
        pg.validate_config(cfg)


def test_target_closed_loop_construction_matches(cfg):
    closed_loop = yaml.safe_load((ROOT / "code/configs/pusht_multi_goal_closed_loop.yaml").read_text())
    assert cfg["target"] == {"target_candidates": closed_loop["target_candidates"], "horizon": closed_loop["cem_horizon"],
                             "controls_per_chunk": closed_loop["controls_per_chunk"],
                             "action_clip": closed_loop["action_clip"]}


def test_agent_bank_is_disjoint_inside_board_and_seeded(cfg):
    bank = pg.agent_bank(cfg)
    stacked = np.concatenate([bank[s] for s in pg.SPLITS])
    assert len(stacked) == cfg["agent_bank"]["size"] and len(np.unique(stacked, axis=0)) == len(stacked)
    assert stacked.min() >= 45 and stacked.max() <= 467
    np.testing.assert_array_equal(pg.agent_bank(cfg)["val"], bank["val"])


def test_target_streams_and_orders_differ_by_split_and_index(cfg):
    draws = {(s, i): pg.target_rng(cfg, s, i).random() for s in pg.SPLITS for i in range(3)}
    assert len(set(draws.values())) == len(draws)
    assert not np.array_equal(pg.agent_order(cfg, "train", 0, 32), pg.agent_order(cfg, "train", 1, 32))


def test_sample_target_returns_farthest_endpoint(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["target"]["target_candidates"] = 6
    env = FakePushT()
    pose = pg.sample_target(env, np.random.default_rng(0), cfg, np.zeros(2), np.ones(2))
    assert pose.shape == (3,)
    assert env.resets == 6


def test_render_group_rejects_agent_on_object():
    pose = np.array([200.0, 200.0, 1.0])
    bank = np.array([[210.0, 200.0], [400.0, 400.0], [205.0, 205.0], [60.0, 60.0], [300.0, 100.0]])
    photos, rejected = pg.render_group(FakePushT(), pose, bank, np.arange(5), needed=2, tolerance=TOL)
    assert [p["agent_bank_index"] for p in photos] == [1, 3]
    assert rejected == {"agent_on_object": 2}  # indices 0 and 2 overlap the T; index 4 is never tried


def test_render_group_rejects_when_reset_moves_object():
    # 40 px below the bar: geometrically clear, but the stand-in physics pushes within 60 px.
    pose = np.array([200.0, 200.0, 0.0])
    bank = np.array([[200.0, 160.0], [400.0, 400.0]])
    photos, rejected = pg.render_group(FakePushT(), pose, bank, np.arange(2), needed=2, tolerance=TOL)
    assert len(photos) == 1 and rejected == {"object_moved": 1}


def test_render_group_rejects_overlap_before_rendering():
    pose = np.array([200.0, 200.0, 0.0])
    bank = np.array([[200.0, 210.0], [400.0, 400.0], [60.0, 60.0]])
    env = FakePushT()
    photos, rejected = pg.render_group(env, pose, bank, np.arange(3), needed=2, tolerance=TOL, min_clearance=5.0)
    assert [p["agent_bank_index"] for p in photos] == [1, 2]
    assert rejected == {"agent_on_object": 1} and env.resets == 2


def test_render_group_reports_shortfall():
    pose = np.array([200.0, 200.0, 1.0])
    bank = np.array([[210.0, 200.0], [400.0, 400.0]])
    photos, rejected = pg.render_group(FakePushT(), pose, bank, np.arange(2), needed=2, tolerance=TOL)
    assert len(photos) == 1 and rejected["agent_on_object"] == 1


@pytest.mark.parametrize("agent,angle,expected", [
    ((200.0, 215.0), 0.0, -15.0),            # inside the bar
    ((200.0, 300.0), 0.0, -15.0),            # inside the stem
    ((200.0, -20.0 + 200.0), 0.0, 5.0),      # 20 px below the bar
    ((300.0, 215.0), 0.0, 25.0),             # 40 px right of the bar end
    ((200.0, 340.0), 0.0, 5.0),              # 20 px past the stem tip
    ((200.0 - 340.0 + 200.0, 200.0), np.pi / 2, 5.0),  # stem rotated to point along -x
])
def test_agent_clearance(agent, angle, expected):
    assert pg.agent_clearance(np.array(agent), np.array([200.0, 200.0, angle])) == pytest.approx(expected)


def test_perturbed_poses_have_requested_sizes(cfg):
    pose = np.array([250.0, 250.0, 6.2])
    moved = pg.perturbed_poses(pose, cfg, np.random.default_rng(0))
    assert len(moved) == 6
    for kind, amount, new in moved:
        if kind == "shift_px":
            assert np.isclose(np.linalg.norm(new[:2] - pose[:2]), amount) and new[2] == pose[2]
        else:
            wrapped = abs((new[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi)
            assert np.isclose(np.rad2deg(wrapped), amount) and np.array_equal(new[:2], pose[:2])
            assert 0 <= new[2] < 2 * np.pi
