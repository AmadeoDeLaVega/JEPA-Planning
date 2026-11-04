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

## Environment

One project-local `.venv` (Python 3.10.19, from `requirements/project.lock`)
serves both PushT and the project task, Meta-World Push. It does not install
into Conda or reuse Conda packages.

```bash
bash code/scripts/setup_env.sh
.venv/bin/python code/scripts/download_checkpoint.py --env pusht
.venv/bin/python code/scripts/download_checkpoint.py --env metaworld
.venv/bin/python code/scripts/check_env.py
.venv/bin/python code/scripts/checkpoint_smoke.py --env pusht
MUJOCO_GL=egl .venv/bin/python code/scripts/checkpoint_smoke.py --env metaworld
```

Goal-photo intervention (proposal, Section 5):

```bash
MUJOCO_GL=egl .venv/bin/python code/experiments/goal_photo_sensitivity.py --device cuda
```

See `docs/reproduction.md` for prerequisites, expected outputs, and checkpoint
integrity information, and `docs/goal_photo_sensitivity.md` for results.

The equivalent-goal PushT experiments compare terminal feature MSE to one goal
photo against the mean over several photos with the same object target. See
`docs/pusht_multi_goal.md` for the hypothesis, fixed-bank and closed-loop
protocols, commands, and full results. In a five-seed reproduction, four-goal
averaging reduced fixed-bank predicted-latent regret from 5.309 to 3.270
(paired improvement 2.039, 95% interval [1.208, 2.980]) and closed-loop final
physical error from 43.932 to 30.786 (paired improvement 13.147, 95% interval
[2.109, 26.784]). Both primary comparisons were positive in all five seed
banks.

The closed-loop Meta-World Push baseline runs through
`code/experiments/run_baseline.py`. Its protocol, commands, outputs, and tests
are documented in `docs/reproduction.md`.

## Coding Requirements

All team-authored code should satisfy the following standards.

### Documentation and readability

- Provide a user guide when a component requires setup or non-obvious usage.
- Begin every substantive source file with a descriptive header or module
  docstring explaining its purpose and providing usage information when
  appropriate.
- Document important inputs, outputs, assumptions, invariants, and design
  decisions close to the relevant code.
- Use clear control flow, descriptive names, and self-documenting interfaces.
  Comments should explain intent and non-obvious behavior rather than merely
  restating individual statements.

### Design and portability

- Keep components organized, clean, loosely coupled, and extensible.
- Place reusable logic in `code/jepa_planning/`; keep experiment orchestration
  in `code/experiments/` and environment or cluster helpers in `code/scripts/`.
- Avoid user-specific absolute paths and undocumented machine assumptions.
  Expose system-dependent values through configuration, command-line options,
  or environment variables.
- Pin dependencies and document platform-specific requirements and supported
  alternatives.

### Unit tests and validation

- Add fine-grained unit tests for the individual functions, classes, and other
  meaningful units of code.
- When applicable, provide approximately one unit-test module in `code/tests/`
  for every substantive team-authored source file.
- Cover normal behavior as well as important boundary conditions and failure
  cases.
- Treat end-to-end, integration, and smoke tests as additional validation; a
  holistic test of the entire system does not replace unit tests.
- Document the commands needed to run tests and reproduce validation results.

## Collaboration

Keep changes focused and reviewable. Each experiment should record its config,
checkpoint, dataset or task split, seed, and evaluation metric. Durable findings
belong in `docs/`; large or generated artifacts do not belong in Git.
