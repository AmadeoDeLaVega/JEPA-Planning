# Tests

Add fast unit tests for reusable modules and small smoke tests for experiment
entry points. Substantive GPU validation must run through Slurm.

Run the fine-grained unit-test suite from the repository root with:

```bash
.venv/bin/python -m unittest discover -s code/tests -v
```
