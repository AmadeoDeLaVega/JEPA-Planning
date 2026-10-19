# JEPA-Planning

Course project investigating planning with frozen joint-embedding predictive
world models. The repository separates team-authored code from proposals,
documentation, generated outputs, and the external DINO-WM implementation.

## Repository layout

```text
proposal/              Original proposal and presentation
docs/                  Project notes and reports
code/jepa_planning/    Reusable team-authored modules
code/experiments/      Experiment entry points
code/configs/          Experiment configurations
code/scripts/          Setup and cluster scripts
code/tests/            Automated tests
requirements/          Reproducible dependency specifications
third_party/           Pinned external source snapshots and attribution
```

## DINO-WM baseline

Our reference implementation is
[`facebookresearch/jepa-wms`](https://github.com/facebookresearch/jepa-wms) at
commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`. A clean snapshot is included at
`third_party/jepa-wms/` so the baseline source remains reproducible and clearly
separated from team-authored code. See `third_party/README.md` for provenance
and licensing information.

Model checkpoints, datasets, environments, and generated experiment outputs
are intentionally excluded from version control.

## PushT environment smoke test

The first reproducible path uses a dedicated project-local `.venv` with Python
3.10.19. It does not install into Conda or reuse Conda packages.

```bash
bash code/scripts/setup_pusht_env.sh
.venv/bin/python code/scripts/download_pusht_checkpoint.py
.venv/bin/python code/scripts/check_env.py
.venv/bin/python code/scripts/checkpoint_smoke.py --device cpu
```

See `docs/reproduction.md` for prerequisites, expected outputs, and checkpoint
integrity information.

## Collaboration

Keep changes focused and reviewable. Each experiment should record its config,
checkpoint, dataset or task split, seed, and evaluation metric. Durable findings
belong in `docs/`; large or generated artifacts do not belong in Git.
