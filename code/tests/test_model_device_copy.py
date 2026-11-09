"""Proprio normalization must land on the model device synchronously (MPS async-copy bug)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_planning.model import _synchronous_proprio_copy  # noqa: E402


class _Preprocessor:
    proprio_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    proprio_std = torch.tensor([2.0, 2.0, 4.0, 4.0])

    def normalize_proprios(self, proprio):
        return (proprio - self.proprio_mean) / self.proprio_std


def test_cpu_device_leaves_preprocessor_unchanged():
    pre = _Preprocessor()
    original = pre.normalize_proprios
    _synchronous_proprio_copy(pre, torch.device("cpu"))
    assert pre.normalize_proprios == original


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_normalized_proprio_matches_cpu():
    pre = _Preprocessor()
    proprio = torch.randn(64, 1, 4)
    expected = _Preprocessor().normalize_proprios(proprio)
    _synchronous_proprio_copy(pre, torch.device("mps"))
    out = pre.normalize_proprios(proprio)
    assert out.device.type == "mps" and out.dtype == torch.float32
    # upstream's follow-up copy is now a no-op and must not corrupt the values
    moved = out.to("mps", dtype=torch.float32, non_blocking=True)
    torch.testing.assert_close(moved.cpu(), expected)
