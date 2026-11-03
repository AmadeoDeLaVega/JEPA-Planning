# Tests

Add fast unit tests for reusable modules and small smoke tests for experiment
entry points. Substantive GPU validation must run through Slurm.

Run the complete suite from the repository root with pytest in the project
environment (install `pytest` first):

```bash
.venv/bin/python -m pytest code/tests -q
```

The goal-feature cost tests also use the standard-library test runner and can
be run independently without pytest:

```bash
.venv/bin/python -m unittest code/tests/test_goal_costs.py -v
```

Baseline tests use CPU toy dynamics; see `docs/reproduction.md` for the
checkpoint-backed smoke run.
