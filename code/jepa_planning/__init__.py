"""Reusable components for the JEPA-Planning course project."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "jepa-wms"

# Keep runtime caches project-local, and make the vendored upstream importable
# (`app`, `evals`, `src`) without installing it.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
