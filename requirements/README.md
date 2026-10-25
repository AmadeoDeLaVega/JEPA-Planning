# Requirements

`project.in` lists the direct dependencies for the whole project (released
DINO-WM inference on PushT and Meta-World Push). `project.lock` is the fully
resolved, hash-pinned Linux x86-64 + CUDA 12.6 environment generated with:

```bash
uv pip compile requirements/project.in \
  --torch-backend cu126 \
  --generate-hashes \
  --python-version 3.10 \
  --python-platform x86_64-manylinux_2_28 \
  --output-file requirements/project.lock
```

The lockfile was tested with uv 0.9.4 and Python 3.10.19. The project setup
creates `.venv/` and a uv-managed Python installation under `.tools/`; it does
not install packages into Conda `base` or use an active Conda environment.
