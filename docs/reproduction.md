# PushT reproduction environment

## Scope

This environment supports the released DINO-WM PushT inference path. It uses
the vendored JEPA-WMs source at upstream commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` and deliberately excludes the
larger MetaWorld and real-robot dependency stacks.

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
bash code/scripts/setup_pusht_env.sh
```

The setup script installs a uv-managed Python rather than selecting a system or
Conda interpreter. It then installs the hash-pinned CUDA 12.6 dependency set
from `requirements/pusht.lock` and installs the vendored JEPA-WMs package in
editable mode.

The setup concludes with a CPU-only PushT simulator smoke test. GPU availability
is reported but is not required for that test.

## Download and verify the checkpoint

Checkpoints are intentionally excluded from Git. Download the official
DINO-WM PushT checkpoint with:

```bash
.venv/bin/python code/scripts/download_pusht_checkpoint.py
```

Expected output path:
`checkpoints/dino_wm_pusht.pth.tar`

Expected size: `274877419` bytes

Expected SHA-256:
`8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f`

The downloader refuses to overwrite an existing file with an unexpected hash.

## Validate the environment and model

```bash
.venv/bin/python code/scripts/check_env.py
.venv/bin/python code/scripts/checkpoint_smoke.py \
  --device cpu \
  --output artifacts/checkpoint_smoke/dino_wm_pusht_cpu.json
```

The environment check reports interpreter and dependency versions and verifies
that Python, caches, vendored source, and checkpoint paths are project-local.
The model smoke test loads the released encoder, action-conditioned predictor,
action encoder, and proprioception encoder. It disables only the optional image
decoder used for visualization.

Expected latent shapes are `[1, 1, 1, 16, 16, 384]` after encoding and
`[2, 1, 1, 16, 16, 384]` after one zero-action prediction step. The first run
may also download the frozen DINOv2 backbone into `.tools/torch/`.
