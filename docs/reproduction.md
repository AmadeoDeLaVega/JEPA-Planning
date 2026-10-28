# Reproduction

## Scope

One environment supports released DINO-WM inference on PushT and on the
project task, Meta-World Push. It uses the vendored JEPA-WMs source at upstream
commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` and excludes the D4RL,
dm-control, PyBullet, and real-robot dependency stacks.

Meta-World is pinned to `3.0.0` because Meta-World's current `master` requires
Python >= 3.11 while JEPA-WMs requires Python 3.10.

## Isolation

The setup uses uv-managed Python 3.10.19 and creates `.venv/` in the repository
root. The Python runtime, dependency cache, Torch Hub cache, and Matplotlib
cache remain under `.tools/`. Neither directory is tracked by Git.

An active Conda environment is not used for package installation. The
environment checker fails if Python resolves outside this repository's
`.venv/`.

## Create the environment

Install uv 0.9.4, or point `UV_BIN` to that executable, and run:

```bash
bash code/scripts/setup_env.sh
```

The setup script installs a uv-managed Python rather than selecting a system or
Conda interpreter. It then installs the hash-pinned Linux x86-64 + CUDA 12.6
dependency set from `requirements/project.lock` and installs the vendored
JEPA-WMs package in editable mode.

The setup concludes with two CPU-only simulator smoke tests:
`check_pusht_env.py` and `check_metaworld_env.py`. The Meta-World check also
verifies that restoring a simulator snapshot and replaying the same actions is
deterministic, which the candidate evaluations rely on. GPU availability is
reported but is not required.

Meta-World rendering needs an OpenGL backend: `MUJOCO_GL=egl` on headless Linux
nodes (the default in the scripts), `MUJOCO_GL=glfw` on macOS.

## Download and verify the checkpoints

Checkpoints are intentionally excluded from Git. Download the official DINO-WM
checkpoints with:

```bash
.venv/bin/python code/scripts/download_checkpoint.py --env pusht
.venv/bin/python code/scripts/download_checkpoint.py --env metaworld
```

| Env | Path | Bytes | SHA-256 |
|---|---|---|---|
| PushT | `checkpoints/dino_wm_pusht.pth.tar` | `274877419` | `8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f` |
| Meta-World | `checkpoints/dino_wm_metaworld.pth.tar` | `280902699` | `eab6e241f732db986f98ad224115ed7949ab4ff3f6cc8bc8f95b273cdb9e5d13` |

The downloader refuses to overwrite an existing file with an unexpected hash.

## Validate the environment and models

```bash
.venv/bin/python code/scripts/check_env.py
.venv/bin/python code/scripts/checkpoint_smoke.py --env pusht \
  --output artifacts/checkpoint_smoke/dino_wm_pusht_cpu.json
MUJOCO_GL=egl .venv/bin/python code/scripts/checkpoint_smoke.py --env metaworld \
  --output artifacts/checkpoint_smoke/dino_wm_metaworld_cpu.json
```

The environment check reports interpreter and dependency versions and verifies
that Python, caches, vendored source, and checkpoint paths are project-local.
The model smoke test loads the released encoder, action-conditioned predictor,
action encoder, and proprioception encoder; the latter three should log
`All keys matched successfully`. It disables only the optional decoder heads
used for visualization.

Expected latent shapes are `[1, 1, 1, 16, 16, 384]` after encoding and
`[2, 1, 1, 16, 16, 384]` after one zero-action prediction step. The first run
may also download the frozen DINOv2 backbone into `.tools/torch/`.

## Regenerate the lockfile

```bash
uv pip compile requirements/project.in --torch-backend cu126 --generate-hashes \
  --python-version 3.10 --python-platform x86_64-manylinux_2_28 \
  --output-file requirements/project.lock
```

## Closed-loop Meta-World Push baseline

Frozen released DINO-WM with upstream CEM and terminal visual-feature MSE.
Proprioception conditions prediction but contributes nothing to the cost.
Decoder heads and per-iteration visualizations are disabled.

After environment setup and checkpoint download, run from the repository root
on a Linux/WSL2 desktop or inside an allocated GPU session:

```bash
# Small check: one task, two photos, five simulator steps each.
MUJOCO_GL=egl .venv/bin/python code/experiments/run_baseline.py \
  --smoke --output results/baseline_smoke

# Full baseline: 100 tasks, two goal photos per task.
MUJOCO_GL=egl .venv/bin/python code/experiments/run_baseline.py \
  --output results/baseline
```

For Slurm, use `sbatch code/scripts/run_baseline.slurm --smoke --output
results/baseline_smoke`. The launcher requests one GPU, four CPU cores, 32 GB
RAM, and four hours; adjust account/partition and wall time for your cluster
after profiling. `--device cpu` supports a small CPU smoke check. On macOS,
set `MUJOCO_GL=glfw` and use a separately compatible environment.

`--config` selects a YAML configuration. `--first-seed` and `--num-tasks`
partition tasks across runs; use a new output directory for each run. Existing
output directories are rejected. For an 8 GB GPU, start with
`planner.batch_size: 8` and profile memory use. This preserves the search budget.

### Protocol

`code/configs/baseline.yaml` defines the settings. Seeds 60000-60099 are separate
from the sensitivity development seeds. After 40 expert warmup steps, save the
start state; another 30 expert steps define the photographed puck goal. Two
photos move the hand in opposite horizontal directions while verifying the
same puck position and orientation. Both episodes restore the same start and
use the same planner seed.

CEM uses 300 candidates, 15 iterations, 10 elites, horizon 6, and executes 3
model actions before observing and replanning. Each model action concatenates
five 4D simulator actions. Candidates are bounded in normalized space so
predicted and executed simulator actions lie in [-1, 1]. Predictions are scored
in batches of 32 by default. Each replan encodes the current observation;
unrolling uses the checkpoint's two-step context window. Evaluation lasts up
to 100 simulator steps, stopping early only on termination/truncation.

The photographed puck target can differ from the native simulator target, so
report both photo placement success within 1/2/5 cm and native success. Smoke
runs reduce the search budget and are excluded from baseline results. This
protocol does not reproduce the upstream Reach benchmark.

### Outputs and tests

Outputs are ignored by Git:

- `manifest.json`: effective config, smoke flag, device, versions, Git revision,
  dirty status, source/checkpoint hashes, and model load time.
- `episodes.csv`: final puck error, photo success thresholds, final/any-step
  native success, steps, replans, timing, and peak allocated GPU memory.
- `summary.json`: episode means, means per goal photo, and paired changes.
- `seed*/task.npz` and `goals_L_R.png`: start integration state, previous
  observation, step count, photo/native targets, hand positions, and goal images.
- `seed*/L.npz`, `R.npz`: executed actions, puck trajectories, native success,
  per-replan CEM times, and per-iteration losses.

Planning time covers CEM. Episode time also includes encoding and simulation,
excluding task/goal construction and output writes. GPU memory measures
allocated memory, not reserved memory. CSV rows and traces are saved after
each episode; the summary is written on normal completion. Rates are fractions.

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest code/tests -q
```

Tests use a small CPU model and simulator double to check bounded actions,
batching, CEM optimization, paired replay, goal invariance, step limits, config
validation, and aggregation. The smoke command requires the actual simulator
and checkpoint; that validation remains pending.
