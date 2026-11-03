# Tests

Add fast unit tests for reusable modules and small smoke tests for experiment
entry points. Substantive GPU validation must run through Slurm.

Run `python -m pytest code/tests -q` in the project environment (install `pytest`
first). Baseline tests use CPU toy dynamics; see `docs/reproduction.md` for the
checkpoint-backed smoke run.
