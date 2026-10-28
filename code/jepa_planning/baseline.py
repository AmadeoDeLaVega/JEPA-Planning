"""Visual-only CEM planning and closed-loop evaluation for Meta-World Push."""

from time import perf_counter

import numpy as np
import torch

from evals.simu_env_planning.planning.planning.planner import CEMPlanner


class VisualCEMPlanner(CEMPlanner):
    def __init__(self, model, goal, seed=0, batch_size=32, **settings):
        super().__init__(
            unroll=model.unroll,
            action_dim=model.action_dim,
            local_generator=torch.Generator(device=model.device).manual_seed(seed),
            **settings,
        )
        self.model, self.goal, self.batch_size = model, goal, batch_size
        self.device = torch.device(model.device)
        pre = model.preprocessor
        repeats = model.action_dim // pre.action_mean.numel()
        self.lower = pre.normalize_actions(-torch.ones_like(pre.action_mean)).repeat(repeats).to(self.device)
        self.upper = pre.normalize_actions(torch.ones_like(pre.action_mean)).repeat(repeats).to(self.device)

    def cost_function(self, actions, context):
        actions.clamp_(self.lower, self.upper)
        costs = []
        for i in range(0, actions.shape[1], self.batch_size):
            endpoint = self.unroll(context, actions[:, i : i + self.batch_size])["visual"][-1]
            costs.append((endpoint - self.goal).square().flatten(1).mean(1))
        return torch.cat(costs)


def prepare_task(env, seed, cfg):
    from jepa_planning import metaworld_push as mp

    obs = mp.reset_task(env, seed)
    obs, _ = mp.run_expert(env, obs, cfg["warmup_steps"])
    start = mp.snapshot(env)
    mp.run_expert(env, obs, cfg["goal_steps"])
    goal_pose = mp.puck_pose(env)
    goal_snap = mp.snapshot(env)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], seed]))
    for _ in range(100):
        angle = rng.uniform(0, 2 * np.pi)
        offset = cfg["hand_separation"] / 2 * np.array([np.cos(angle), np.sin(angle), 0])
        photos, hands = [], []
        for sign in (1, -1):
            mp.restore(env, goal_snap)
            mp.move_hand_keep_puck(env, sign * offset)
            np.testing.assert_array_equal(mp.puck_pose(env), goal_pose)
            photos.append(env.render().copy())
            hands.append(mp.hand_pos(env))
        if all(np.linalg.norm(hand[:2] - goal_pose[:2]) > 0.06 for hand in hands):
            return start, goal_pose[:3], photos, np.stack(hands)
    raise RuntimeError(f"Could not construct goal photos for task {seed}")


@torch.inference_mode()
def run_episode(env, model, start, photo, goal_puck, cfg, seed):
    from jepa_planning import metaworld_push as mp

    mp.restore(env, start)
    device = torch.device(model.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    goal = model.encode(torch.from_numpy(photo).permute(2, 0, 1)[None, None])[:, -1]
    planner = VisualCEMPlanner(model, goal, seed=seed, **cfg["planner"])
    actions, puck_positions, successes, times, losses = [], [mp.puck_pose(env)[:3]], [], [], []
    done = False
    while len(actions) < cfg["max_steps"] and not done:
        context = model.encode(mp.observe(env))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = perf_counter()
        remaining = cfg["max_steps"] - len(actions)
        plan = planner.plan(context, steps_left=(remaining + mp.FRAMESKIP - 1) // mp.FRAMESKIP)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(perf_counter() - tick)
        losses.append(plan.losses.squeeze(-1).cpu().numpy())
        env_actions = model.preprocessor.denormalize_actions(plan.actions.cpu().reshape(-1, 4)).clamp(-1, 1)
        for action in env_actions[:remaining].numpy():
            _, _, terminated, truncated, info = env.step(action)
            actions.append(action.copy())
            puck_positions.append(mp.puck_pose(env)[:3])
            successes.append(bool(info["success"]))
            done = bool(terminated or truncated)
            if done:
                break
    error_cm = float(100 * np.linalg.norm(puck_positions[-1] - goal_puck))
    metrics = {
        "planner_seed": seed,
        "steps": len(actions),
        "replans": len(times),
        "puck_error_cm": error_cm,
        "success_1cm": error_cm <= 1,
        "success_2cm": error_cm <= 2,
        "success_5cm": error_cm <= 5,
        "native_success": successes[-1],
        "native_success_any": any(successes),
        "planning_seconds": sum(times),
        "episode_seconds": perf_counter() - started,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
    }
    trace = {
        "actions": np.stack(actions),
        "puck_positions": np.stack(puck_positions),
        "native_success": np.array(successes),
        "planning_seconds": np.array(times),
        "cem_losses": np.stack(losses),
    }
    return metrics, trace
