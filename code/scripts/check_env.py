#!/usr/bin/env python3
"""Verify that JEPA-Planning is using only its project-local Python environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PREFIX = (ROOT / ".venv").resolve()
EXPECTED_BASE = (ROOT / ".tools/python/cpython-3.10.19-linux-x86_64-gnu").resolve()
os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools/torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))

import decord  # noqa: E402
import gym  # noqa: E402
import numpy  # noqa: E402
import pygame  # noqa: E402
import pymunk  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    executable = Path(sys.executable).resolve()
    errors = []
    if prefix != EXPECTED_PREFIX:
        errors.append(f"sys.prefix is {prefix}, expected {EXPECTED_PREFIX}")
    if base_prefix != EXPECTED_BASE:
        errors.append(f"sys.base_prefix is {base_prefix}, expected {EXPECTED_BASE}")
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python is {sys.version_info.major}.{sys.version_info.minor}, expected 3.10")
    if torch.__version__ != "2.7.0+cu126":
        errors.append(f"torch is {torch.__version__}, expected 2.7.0+cu126")
    if Path(torch.hub.get_dir()).resolve() != (ROOT / ".tools/torch/hub").resolve():
        errors.append(f"torch hub cache escaped project: {torch.hub.get_dir()}")
    required = [
        ROOT / "requirements/pusht.lock",
        ROOT / "checkpoints/dino_wm_pusht.pth.tar",
        ROOT / "third_party/jepa-wms",
    ]
    errors.extend(f"missing required path: {path}" for path in required if not path.exists())
    result = {
        "ok": not errors,
        "project_root": str(ROOT),
        "python_executable": str(executable),
        "venv_prefix": str(prefix),
        "base_python_prefix": str(base_prefix),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_hub": torch.hub.get_dir(),
        "conda_prefix_in_shell": os.environ.get("CONDA_PREFIX"),
        "versions": {
            "decord": decord.__version__,
            "gym": gym.__version__,
            "numpy": numpy.__version__,
            "pygame": pygame.version.ver,
            "pymunk": pymunk.version,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
