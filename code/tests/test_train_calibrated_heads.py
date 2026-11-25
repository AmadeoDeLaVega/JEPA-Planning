"""Fast CPU unit tests for calibrated linear goal head training and spectral auditing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import train_calibrated_heads as tch
from jepa_planning import linear_head as lh


def _write_synthetic_goal_groups(
    directory: Path,
    *,
    num_targets: int = 12,
    hands_per_target: int = 2,
    patches: int = 4,
    feature_dim: int = 16,
    seed: int = 0,
    with_sensitivity: bool = True,
    sensitivity_kind: str = "shift_px",
    sensitivity_amounts: tuple[float, ...] = (5.0, 20.0),
) -> None:
    """Create a tiny PushT/Meta-World compatible goal-group cache on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_gen = torch.Generator().manual_seed(seed + 101)

    groups = np.repeat(np.arange(num_targets, dtype=np.int64), hands_per_target)
    slots = np.tile(np.arange(hands_per_target, dtype=np.int64), num_targets)
    n_images = len(groups)

    # First 4 channels carry multi-axis target pose signal (with a dominant first axis
    # so unregularized training tends to concentrate), channels 4:8 carry hand nuisance.
    latents = rng.standard_normal((num_targets, 4)).astype(np.float32) * np.array(
        [4.0, 1.2, 0.9, 0.7], dtype=np.float32
    )
    features = torch.zeros((n_images, patches, feature_dim), dtype=torch.float32)
    for row_idx, grp in enumerate(groups):
        features[row_idx, :, :4] = torch.from_numpy(latents[grp])
        features[row_idx, :, 4:8] = torch.randn((patches, 4), generator=t_gen) * 1.5
        features[row_idx, :, 8:] = torch.randn((patches, feature_dim - 8), generator=t_gen) * 0.1

    np.savez(directory / "images.npz", group=groups, slot=slots)

    payload: dict[str, torch.Tensor] = {"groups": features}
    if with_sensitivity:
        sens_features = []
        sens_groups = []
        sens_kinds = []
        sens_amounts = []
        for grp in range(min(num_targets, 6)):
            ref_idx = int(np.flatnonzero(groups == grp)[0])
            base_feat = features[ref_idx]
            for amount in sensitivity_amounts:
                perturbed = base_feat.clone()
                perturbed[:, :4] = perturbed[:, :4] + float(amount) * 0.15
                sens_features.append(perturbed)
                sens_groups.append(grp)
                sens_kinds.append(sensitivity_kind)
                sens_amounts.append(float(amount))

        payload["sensitivity"] = torch.stack(sens_features, dim=0)
        np.savez(
            directory / "sensitivity.npz",
            group=np.asarray(sens_groups, dtype=np.int64),
            kind=np.asarray(sens_kinds),
            amount=np.asarray(sens_amounts, dtype=np.float64),
        )

    torch.save(payload, directory / "features.pt")


def test_spectral_summary_exact_invariants():
    # Rank-1 unit-norm matrix has effective_rank == 1, entropy_rank == 1, top1_eig == 1.
    rank1 = torch.zeros(6, 12, dtype=torch.float32)
    rank1[0, 0] = 1.0
    s1 = tch.spectral_summary(rank1)
    assert s1["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert s1["entropy_rank"] == pytest.approx(1.0, abs=1e-6)
    assert s1["top1_eig"] == pytest.approx(1.0, abs=1e-6)
    assert s1["top4_eig_sum"] == pytest.approx(1.0, abs=1e-6)
    assert s1["frobenius_norm"] == pytest.approx(1.0, abs=1e-6)

    # Orthonormal rank-4 matrix rescaled to unit Frobenius norm has r_eff == 4, entropy_rank == 4.
    iso4 = torch.zeros(8, 12, dtype=torch.float32)
    iso4[:4, :4] = torch.eye(4) * 0.5
    s4 = tch.spectral_summary(iso4)
    assert s4["effective_rank"] == pytest.approx(4.0, abs=1e-5)
    assert s4["entropy_rank"] == pytest.approx(4.0, abs=1e-5)
    assert s4["top1_eig"] == pytest.approx(0.25, abs=1e-6)
    assert s4["top4_eig_sum"] == pytest.approx(1.0, abs=1e-6)


def test_train_calibrated_heads_end_to_end_all_regimes_with_sensitivity(tmp_path: Path):
    train_dir = tmp_path / "train_groups"
    val_dir = tmp_path / "val_groups"
    out_dir = tmp_path / "audit_out"

    _write_synthetic_goal_groups(train_dir, num_targets=12, hands_per_target=2, seed=1, with_sensitivity=False)
    _write_synthetic_goal_groups(
        val_dir,
        num_targets=10,
        hands_per_target=2,
        seed=2,
        with_sensitivity=True,
        sensitivity_kind="shift_px",
        sensitivity_amounts=(5.0, 20.0),
    )

    fantope_k = 4.0
    rows = tch.main(
        [
            "--train-groups",
            str(train_dir),
            "--val-groups",
            str(val_dir),
            "--regimes",
            "unregularized",
            "isotropy",
            "fantope",
            "--budgets",
            "16",
            "--hands",
            "2",
            "--ranks",
            "8",
            "--isotropy-weight",
            "0.05",
            "--fantope-k",
            str(fantope_k),
            "--resamples",
            "1",
            "--steps",
            "35",
            "--batch-size",
            "16",
            "--tau",
            "0.3",
            "--device",
            "cpu",
            "--output",
            str(out_dir),
        ]
    )

    assert len(rows) == 3
    heads_dir = out_dir / "heads"
    saved_pts = sorted(heads_dir.glob("*.pt"))
    assert len(saved_pts) == 3

    csv_path = out_dir / "spectral_audit.csv"
    json_path = out_dir / "spectral_audit.json"
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open("r", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3

    required_cols = {
        "head",
        "regime",
        "effective_rank",
        "entropy_rank",
        "top1_eig",
        "top4_eig_sum",
        "top1",
        "auroc",
        "within_median",
        "sensitivity_shift_px_5",
        "sensitivity_shift_px_20",
    }
    assert required_cols.issubset(set(csv_rows[0].keys()))

    by_regime = {row["regime"]: row for row in csv_rows}
    assert set(by_regime.keys()) == {"unregularized", "isotropy", "fantope"}

    # Verify saved checkpoints load cleanly via lh.load_head and match spectral audit values.
    for row in csv_rows:
        ckpt_path = out_dir / row["head_path"]
        assert ckpt_path.exists()
        head, cfg = lh.load_head(ckpt_path)
        assert head.shape == (8, 16)
        assert cfg["regime"] == row["regime"]
        assert cfg["B"] == 16
        assert cfg["K"] == 2
        spec = tch.spectral_summary(head)
        assert float(row["effective_rank"]) == pytest.approx(spec["effective_rank"], rel=1e-5, abs=1e-5)
        assert 0.0 <= float(row["top1"]) <= 1.0
        assert 0.0 <= float(row["auroc"]) <= 1.0
        assert float(row["within_median"]) >= 0.0
        assert float(row["sensitivity_shift_px_20"]) > 0.0

    # Fantope projection guarantees effective_rank >= fantope_k and top1_eig <= 1 / fantope_k.
    fantope_row = by_regime["fantope"]
    assert float(fantope_row["effective_rank"]) >= fantope_k - 1e-4
    assert float(fantope_row["top1_eig"]) <= (1.0 / fantope_k) + 1e-4

    # JSON output matches CSV count and includes metadata.
    payload = json.loads(json_path.read_text())
    assert len(payload["rows"]) == 3
    assert payload["feature_dim"] == 16
    assert payload["fantope_k"] == pytest.approx(fantope_k)


def test_train_calibrated_heads_without_sensitivity_and_with_identity(tmp_path: Path):
    train_dir = tmp_path / "mw_train"
    val_dir = tmp_path / "mw_val"
    out_dir = tmp_path / "mw_out"

    _write_synthetic_goal_groups(train_dir, num_targets=10, hands_per_target=2, seed=7, with_sensitivity=False)
    _write_synthetic_goal_groups(val_dir, num_targets=8, hands_per_target=2, seed=8, with_sensitivity=False)

    rows = tch.main(
        [
            "--train-groups",
            str(train_dir),
            "--val-groups",
            str(val_dir),
            "--regimes",
            "fantope",
            "--budgets",
            "16",
            "--hands",
            "2",
            "--ranks",
            "6",
            "--fantope-k",
            "3.0",
            "--steps",
            "20",
            "--batch-size",
            "8",
            "--include-identity",
            "--device",
            "cpu",
            "--output",
            str(out_dir),
        ]
    )

    assert len(rows) == 2
    assert rows[0]["regime"] == "identity"
    assert rows[0]["effective_rank"] == pytest.approx(16.0, abs=1e-4)
    assert rows[1]["regime"] == "fantope"
    assert rows[1]["effective_rank"] >= 3.0 - 1e-4
    assert not any(k.startswith("sensitivity_") for k in rows[1])


def _write_spatially_localized_goal_groups(
    directory: Path,
    *,
    num_targets: int = 12,
    hands_per_target: int = 2,
    patches: int = 16,
    feature_dim: int = 16,
    seed: int = 0,
) -> None:
    """Create synthetic groups where patches 0..1 carry object signal and 2..7 carry arm noise."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_gen = torch.Generator().manual_seed(seed + 303)

    groups = np.repeat(np.arange(num_targets, dtype=np.int64), hands_per_target)
    slots = np.tile(np.arange(hands_per_target, dtype=np.int64), num_targets)
    n_images = len(groups)

    obj_latents = rng.standard_normal((num_targets, 4)).astype(np.float32) * 3.0
    features = torch.zeros((n_images, patches, feature_dim), dtype=torch.float32)
    for row_idx, grp in enumerate(groups):
        # Object patches: strong between-group signal, low within-group arm noise
        features[row_idx, :2, :4] = torch.from_numpy(obj_latents[grp])
        features[row_idx, :2, 4:] = torch.randn((2, feature_dim - 4), generator=t_gen) * 0.1
        # Arm nuisance patches: high within-group variance, zero between-group signal
        features[row_idx, 2:8, :] = torch.randn((6, feature_dim), generator=t_gen) * 2.5
        # Static background/border patches: near-zero variance with tiny leakage
        features[row_idx, 8:, :1] = torch.from_numpy(obj_latents[grp, :1]) * 1e-3
        features[row_idx, 8:, 1:] = torch.randn((patches - 8, feature_dim - 1), generator=t_gen) * 1e-4

    np.savez(directory / "images.npz", group=groups, slot=slots)
    torch.save({"groups": features}, directory / "features.pt")


def test_train_calibrated_heads_anova_regimes_and_default_preservation(tmp_path: Path):
    from jepa_planning import spatial_channel_head as sch

    # Default --regimes must remain unchanged for backwards compatibility.
    parsed_defaults = tch.build_parser().parse_args(
        ["--train-groups", "a", "--val-groups", "b", "--output", "c"]
    )
    assert parsed_defaults.regimes == ["unregularized", "isotropy", "fantope"]
    assert parsed_defaults.anova_gamma == pytest.approx(1.5)
    assert parsed_defaults.anova_floor == pytest.approx(0.25)

    train_dir = tmp_path / "anova_train"
    val_dir = tmp_path / "anova_val"
    out_dir = tmp_path / "anova_out"

    _write_spatially_localized_goal_groups(train_dir, num_targets=12, hands_per_target=2, patches=16, seed=11)
    _write_spatially_localized_goal_groups(val_dir, num_targets=10, hands_per_target=2, patches=16, seed=12)

    fantope_k = 3.0
    rows = tch.main(
        [
            "--train-groups",
            str(train_dir),
            "--val-groups",
            str(val_dir),
            "--regimes",
            "anova_fantope",
            "anova_isotropy",
            "--budgets",
            "16",
            "--hands",
            "2",
            "--ranks",
            "8",
            "--fantope-k",
            str(fantope_k),
            "--isotropy-weight",
            "0.03",
            "--anova-gamma",
            "1.5",
            "--anova-floor",
            "0.25",
            "--steps",
            "30",
            "--batch-size",
            "16",
            "--tau",
            "0.2",
            "--device",
            "cpu",
            "--output",
            str(out_dir),
        ]
    )

    assert len(rows) == 2
    by_regime = {row["regime"]: row for row in rows}
    assert set(by_regime.keys()) == {"anova_fantope", "anova_isotropy"}

    csv_path = out_dir / "spectral_audit.csv"
    json_path = out_dir / "spectral_audit.json"
    with csv_path.open("r", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert "effective_patches" in csv_rows[0]

    payload = json.loads(json_path.read_text())
    assert payload["anova_gamma"] == pytest.approx(1.5)
    assert payload["anova_floor"] == pytest.approx(0.25)
    assert len(payload["rows"]) == 2

    for row, csv_row in zip(rows, csv_rows):
        ckpt_path = out_dir / row["head_path"]
        assert ckpt_path.exists()
        weights, head, meta = sch.load_spatial_channel_head(ckpt_path)
        assert weights is not None
        assert weights.shape == (16,)
        assert float(weights.sum().item()) == pytest.approx(1.0, abs=1e-5)
        # Object patches (0..1) receive substantially higher weight than uniform (2/16 = 0.125).
        assert float(weights[:2].sum().item()) > 0.5
        # Effective patches matches CSV and JSON audit values and lies strictly below 16.
        p_eff = sch.effective_patches(weights)
        assert 1.0 <= p_eff < 16.0
        assert float(csv_row["effective_patches"]) == pytest.approx(p_eff, rel=1e-5, abs=1e-5)
        assert float(row["effective_patches"]) == pytest.approx(p_eff, rel=1e-5, abs=1e-5)
        assert meta["regime"] == row["regime"]
        assert meta["anova_gamma"] == pytest.approx(1.5)
        assert meta["anova_floor"] == pytest.approx(0.25)
        assert head.shape == (8, 16)

    assert float(by_regime["anova_fantope"]["effective_rank"]) >= fantope_k - 1e-4
