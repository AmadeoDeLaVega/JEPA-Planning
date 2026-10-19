# Requirements

`pusht.in` lists the direct dependencies for the released PushT evaluation
path. `pusht.lock` is the fully resolved, hash-pinned CUDA 12.6 environment
generated with:

```bash
uv pip compile requirements/pusht.in \
  --torch-backend cu126 \
  --generate-hashes \
  --output-file requirements/pusht.lock
```

The lockfile was tested with uv 0.9.4 and Python 3.10.19. The project setup
creates `.venv/` and a uv-managed Python installation under `.tools/`; it does
not install packages into Conda `base` or use an active Conda environment.
