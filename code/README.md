# Code

This directory contains code authored or adapted by the JEPA-Planning team.
The unmodified DINO-WM reference snapshot is kept separately under
`third_party/jepa-wms/` with its original license and attribution.

- `jepa_planning/`: reusable project modules;
- `experiments/`: executable experiment entry points;
- `configs/`: explicit experiment configurations;
- `scripts/`: environment, data, and Slurm helpers;
- `tests/`: automated tests.

Experiment scripts should avoid hidden defaults and write generated outputs to
an ignored directory such as `results/<experiment_name>/`.
