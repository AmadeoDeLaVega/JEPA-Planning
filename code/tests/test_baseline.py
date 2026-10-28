"""CPU checks for bounded CEM, goal construction, and episode execution."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import jepa_planning
from jepa_planning import ROOT
from jepa_planning.baseline import VisualCEMPlanner, prepare_task, run_episode


class Preprocessor:
    action_mean = torch.tensor([0.2, -0.1, 0.3, 0.0])
    action_std = torch.tensor([0.5, 0.25, 0.7, 1.0])

    def normalize_actions(self, actions):
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        return actions * self.action_std + self.action_mean


class ToyModel:
    device = torch.device("cpu")
    action_dim = 20
    preprocessor = Preprocessor()

    def __init__(self):
        self.batches = []

    def encode(self, obs):
        if isinstance(obs, dict):
            return {"visual": obs["visual"].reshape(1, 1, 1, 1, 1, 1)}
        return obs.float().mean().reshape(1, 1, 1, 1, 1, 1) / 255

    def unroll(self, context, act_suffix):
        actions = act_suffix
        horizon, batch, _ = actions.shape
        self.batches.append(batch)
        raw = self.preprocessor.denormalize_actions(actions.reshape(horizon, batch, 5, 4))
        delta = (raw[..., 0].sum(-1) * 0.01).cumsum(0).reshape(horizon, batch, 1, 1, 1, 1)
        start = context["visual"][:, -1].expand(batch, 1, 1, 1, 1)
        return {"visual": torch.cat([start[None], start[None] + delta])}


@pytest.fixture
def cfg():
    config = yaml.safe_load((ROOT / "code/configs/baseline.yaml").read_text())
    config.update(num_tasks=1, warmup_steps=0, goal_steps=1, max_steps=7)
    config["planner"].update(horizon=2, num_act_stepped=2, iterations=5, num_samples=48, num_elites=6, batch_size=7)
    return config


@pytest.fixture
def context():
    return {"visual": torch.zeros(1, 1, 1, 1, 1, 1)}


def test_cost_uses_bounded_actions_and_terminal_visual_features(cfg, context):
    model = ToyModel()
    planner = VisualCEMPlanner(model, torch.full((1, 1, 1, 1, 1), 0.1), **cfg["planner"])
    actions = torch.full((2, 17, 20), 1000.0)
    cost = planner.cost_function(actions, context)
    assert model.batches == [7, 7, 3]
    torch.testing.assert_close(actions[0, 0], planner.upper)
    torch.testing.assert_close(cost, torch.zeros(17), atol=1e-12, rtol=0)
    actions.fill_(-1000)
    planner.cost_function(actions, context)
    torch.testing.assert_close(actions[0, 0], planner.lower)


def test_chunking_preserves_costs(cfg, context):
    model = ToyModel()
    goal = torch.zeros(1, 1, 1, 1, 1)
    actions = torch.randn(2, 17, 20, generator=torch.Generator().manual_seed(4))
    chunked = VisualCEMPlanner(model, goal, **cfg["planner"])
    full = VisualCEMPlanner(model, goal, **{**cfg["planner"], "batch_size": 17})
    torch.testing.assert_close(
        chunked.cost_function(actions.clone(), context), full.cost_function(actions.clone(), context)
    )


def test_cem_improves_cost_and_repeats_with_same_seed(cfg, context):
    model = ToyModel()
    goal = torch.full((1, 1, 1, 1, 1), 0.06)
    planners = [VisualCEMPlanner(model, goal, seed=13, **cfg["planner"]) for _ in range(2)]
    initial = planners[0].cost_function(torch.zeros(2, 1, 20), context)
    results = [planner.plan(context) for planner in planners]
    torch.testing.assert_close(results[0].actions, results[1].actions)
    final = planners[0].cost_function(results[0].actions[:, None], context)
    assert final.item() < initial.item() / 10
    assert results[0].actions.device.type == "cpu"
    assert torch.all(results[0].actions >= planners[0].lower)
    assert torch.all(results[0].actions <= planners[0].upper)


class FakeEnv:
    def __init__(self, terminate_at=500):
        self.position = np.zeros(3)
        self.steps = 0
        self.terminate_at = terminate_at

    def step(self, action):
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1)
        self.position[0] += action[0] * 0.01
        self.steps += 1
        return None, 0.0, self.steps >= self.terminate_at, False, {"success": self.steps >= 3}


@pytest.fixture
def helpers(monkeypatch):
    def snapshot(env):
        return {"position": env.position.copy(), "steps": env.steps}

    def restore(env, snap):
        env.position, env.steps = snap["position"].copy(), snap["steps"]

    def expert(env, obs, steps):
        for _ in range(steps):
            env.step(np.ones(4))
        return obs, []

    def reset(env, seed):
        env.position, env.steps = np.zeros(3), 0

    def move_hand(env, offset):
        env.hand = np.array([0, 0.3, 0]) + offset

    mp = SimpleNamespace(
        FRAMESKIP=5, snapshot=snapshot, restore=restore, reset_task=reset,
        run_expert=expert, puck_pose=lambda env: np.r_[env.position, 1, 0, 0, 0],
        observe=lambda env: {"visual": torch.tensor(env.position[0], dtype=torch.float32)},
        move_hand_keep_puck=move_hand, hand_pos=lambda env: env.hand.copy(),
    )
    monkeypatch.setattr(jepa_planning, "metaworld_push", mp, raising=False)
    return mp


@pytest.mark.parametrize("terminate_at,expected", [(500, 7), (3, 3)])
def test_episode_caps_actions_and_restores_paired_start(cfg, helpers, terminate_at, expected):
    cfg["planner"]["num_act_stepped"] = 1
    model = ToyModel()
    env = FakeEnv(terminate_at)
    start = helpers.snapshot(env)
    photo = np.full((1, 1, 3), 15, dtype=np.uint8)
    runs = [run_episode(env, model, start, photo, np.array([0.06, 0, 0]), cfg, 11) for _ in range(2)]
    metrics, trace = runs[0]
    assert metrics["steps"] == expected
    assert metrics["replans"] == (expected + 4) // 5
    assert metrics["native_success"]
    assert trace["actions"].shape == (expected, 4)
    assert trace["puck_positions"].shape == (expected + 1, 3)
    assert metrics["planning_seconds"] >= 0
    assert metrics["episode_seconds"] >= metrics["planning_seconds"]
    np.testing.assert_array_equal(trace["actions"], runs[1][1]["actions"])
    np.testing.assert_allclose(
        metrics["puck_error_cm"], 100 * np.linalg.norm(trace["puck_positions"][-1] - [0.06, 0, 0])
    )


def test_goal_photos_keep_puck_and_repeat_per_task(cfg, helpers):
    env = FakeEnv()
    env.render = lambda: np.full((1, 1, 3), env.hand[0] + 1)
    first, second = (prepare_task(env, 60000, cfg) for _ in range(2))
    np.testing.assert_array_equal(first[1], [0.01, 0, 0])
    np.testing.assert_array_equal(first[3], second[3])
    assert len(first[2]) == 2
    np.testing.assert_allclose(np.linalg.norm(first[3][0] - first[3][1]), cfg["hand_separation"])


def test_goal_photos_reject_puck_motion(cfg, helpers):
    env = FakeEnv()
    helpers.move_hand_keep_puck = lambda env, offset: env.position.__setitem__(0, 100)
    with pytest.raises(AssertionError):
        prepare_task(env, 60000, cfg)


def test_goal_photo_retries_are_bounded(cfg, helpers):
    env = FakeEnv()
    env.render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)
    helpers.hand_pos = lambda env: env.position.copy()
    with pytest.raises(RuntimeError, match="60000"):
        prepare_task(env, 60000, cfg)
