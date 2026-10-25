# Reproduction environment

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
