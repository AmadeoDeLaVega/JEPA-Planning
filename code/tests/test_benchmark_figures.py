"""Fast CPU unit tests for publication benchmark figure builders and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from jepa_planning import benchmark_figures as bf  # noqa: E402
from jepa_planning import linear_head as lh  # noqa: E402
from scripts import generate_benchmark_figures as gbf  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PDF_MAGIC = b"%PDF-"


def _synthetic_heads(rank: int = 32, dim: int = 64) -> dict[str, torch.Tensor]:
    """Build synthetic rank-1 collapsed, isotropy, and k-Fantope heads."""
    gen = torch.Generator().manual_seed(0)
    u, _, vh = torch.linalg.svd(torch.randn(rank, dim, generator=gen), full_matrices=False)

    # Collapsed spectrum (top eigenvalue ~0.88)
    s_unreg = torch.sqrt(torch.tensor([0.88] + [0.12 / (rank - 1)] * (rank - 1)))
    head_unreg = (u * s_unreg.unsqueeze(0)).matmul(vh)

    # Smooth isotropy spectrum
    raw_iso = torch.exp(-0.18 * torch.arange(rank, dtype=torch.float32))
    s_iso = torch.sqrt(raw_iso / raw_iso.sum())
    head_iso = (u * s_iso.unsqueeze(0)).matmul(vh)

    # Exact 4-Fantope spectrum (4 flat leading eigenvalues at 0.25, trailing zeros)
    eigs_fan = torch.zeros(rank, dtype=torch.float32)
    eigs_fan[:4] = 0.25
    head_fan = (u * torch.sqrt(eigs_fan).unsqueeze(0)).matmul(vh)

    return {
        "Unregularized (B=1024)": head_unreg,
        "Isotropy (w=0.015)": head_iso,
        "Fantope (k=4)": head_fan,
    }


def _assert_nonempty_png(path: Path) -> None:
    assert path.exists(), f"missing PNG: {path}"
    data = path.read_bytes()
    assert len(data) > 1000
    assert data.startswith(PNG_MAGIC)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_plot_eigenspectra_comparison_writes_png_without_warnings(tmp_path: Path, theme: str):
    out_png = tmp_path / f"eigenspectra_{theme}.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = bf.plot_eigenspectra_comparison(
            _synthetic_heads(),
            out_png,
            theme=theme,
            dpi=150,
            also_pdf=(theme == "light"),
        )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    _assert_nonempty_png(out_png)
    if theme == "light":
        pdf_path = out_png.with_suffix(".pdf")
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(PDF_MAGIC)


@pytest.mark.parametrize(
    ("unit", "title", "scale"),
    [
        ("px", "PushT closed-loop control (held-out targets)", 1.0),
        ("cm", "Meta-World Sawyer Push-v3 closed-loop control", 0.12),
    ],
)
def test_plot_closed_loop_forest_pusht_and_metaworld(
    tmp_path: Path,
    unit: str,
    title: str,
    scale: float,
):
    summary_rows = [
        {
            "label": "Plain MSE + Standard CEM",
            "mean": 48.0 * scale,
            "ci_lo": 42.0 * scale,
            "ci_hi": 54.0 * scale,
            "iqm": 45.5 * scale,
            "is_baseline": True,
        },
        {
            "label": "Unregularized + Standard CEM",
            "mean": 51.5 * scale,
            "ci_lo": 45.0 * scale,
            "ci_hi": 58.0 * scale,
            "iqm": 49.0 * scale,
        },
        {
            "label": "Isotropy + Trust-MPC",
            "mean": 31.2 * scale,
            "ci_lo": 26.5 * scale,
            "ci_hi": 36.1 * scale,
            "iqm": 29.4 * scale,
        },
        {
            "label": "Fantope (k=4) + Trust-MPC",
            "mean": 27.8 * scale,
            "ci_lo": 23.1 * scale,
            "ci_hi": 32.6 * scale,
            "iqm": 25.9 * scale,
        },
    ]
    paired_contrasts = [
        {
            "label": "Unregularized + CEM",
            "diff": 3.5 * scale,
            "ci_lo": -1.2 * scale,
            "ci_hi": 8.4 * scale,
            "wins": 9,
            "losses": 15,
            "total": 24,
        },
        {
            "label": "Isotropy + Trust-MPC",
            "diff": -16.8 * scale,
            "ci_lo": -22.5 * scale,
            "ci_hi": -11.0 * scale,
            "wins": 19,
            "losses": 5,
            "total": 24,
        },
        {
            "label": "Fantope + Trust-MPC",
            "diff": -20.2 * scale,
            "ci_lo": -26.0 * scale,
            "ci_hi": -14.3 * scale,
            "wins": 21,
            "losses": 3,
            "total": 24,
        },
    ]

    out_png = tmp_path / f"forest_{unit}.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = bf.plot_closed_loop_forest(
            summary_rows,
            paired_contrasts,
            unit=unit,
            title=title,
            output_path=out_png,
            theme="dark" if unit == "cm" else "light",
            dpi=150,
        )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    _assert_nonempty_png(out_png)


@pytest.mark.parametrize("unit", ["px", "cm"])
def test_plot_budget_scaling_curve_px_and_cm(tmp_path: Path, unit: str):
    scale = 1.0 if unit == "px" else 0.12
    scaling_rows = [
        {"regime": "unregularized", "B": 64, "error": 38.0 * scale, "ci_lo": 34.0 * scale, "ci_hi": 42.0 * scale, "effective_rank": 6.4},
        {"regime": "unregularized", "B": 256, "error": 41.5 * scale, "ci_lo": 37.0 * scale, "ci_hi": 46.0 * scale, "effective_rank": 2.3},
        {"regime": "unregularized", "B": 1024, "error": 52.0 * scale, "ci_lo": 47.0 * scale, "ci_hi": 57.0 * scale, "effective_rank": 1.18},
        {"regime": "isotropy", "B": 64, "error": 36.2 * scale, "ci_lo": 32.5 * scale, "ci_hi": 40.0 * scale, "effective_rank": 14.2},
        {"regime": "isotropy", "B": 256, "error": 31.8 * scale, "ci_lo": 28.0 * scale, "ci_hi": 35.5 * scale, "effective_rank": 12.1},
        {"regime": "isotropy", "B": 1024, "error": 29.1 * scale, "ci_lo": 25.5 * scale, "ci_hi": 32.8 * scale, "effective_rank": 10.6},
        {"regime": "fantope", "B": 64, "error": 35.0 * scale, "ci_lo": 31.0 * scale, "ci_hi": 39.0 * scale, "effective_rank": 4.4},
        {"regime": "fantope", "B": 256, "error": 29.8 * scale, "ci_lo": 26.2 * scale, "ci_hi": 33.4 * scale, "effective_rank": 4.1},
        {"regime": "fantope", "B": 1024, "error": 27.4 * scale, "ci_lo": 23.9 * scale, "ci_hi": 31.0 * scale, "effective_rank": 4.02},
    ]

    out_png = tmp_path / f"budget_scaling_{unit}.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = bf.plot_budget_scaling_curve(
            scaling_rows,
            out_png,
            unit=unit,
            baseline_error=48.0 * scale,
            theme="light",
            dpi=150,
        )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    _assert_nonempty_png(out_png)


def test_generate_benchmark_figures_cli_end_to_end(tmp_path: Path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    for name, mat in _synthetic_heads(rank=16, dim=32).items():
        stem = name.lower().split()[0]
        lh.save_head(heads_dir / f"{stem}.pt", mat, {"regime": stem, "rank": 16})

    # Meta-World style summary.json
    mw_summary_path = tmp_path / "summary.json"
    mw_summary = {
        "baseline_condition": {"head": "identity", "planner": "standard_cem", "goal_photos": 1},
        "conditions": [
            {
                "head": "identity",
                "planner": "standard_cem",
                "goal_photos": 1,
                "metrics": {"final_error_3d_cm": {"mean": 6.2, "ci_lo": 5.1, "ci_hi": 7.3, "iqm": 5.9}},
                "paired_vs_baseline": {},
            },
            {
                "head": "fantope",
                "planner": "trust_mpc",
                "goal_photos": 1,
                "metrics": {"final_error_3d_cm": {"mean": 3.8, "ci_lo": 3.0, "ci_hi": 4.6, "iqm": 3.6}},
                "paired_vs_baseline": {
                    "final_error_3d_cm": {"mean": -2.4, "ci_lo": -3.4, "ci_hi": -1.4, "iqm": -2.2}
                },
            },
        ],
    }
    mw_summary_path.write_text(json.dumps(mw_summary))

    # Scaling CSV + spectral_audit.csv join
    scaling_csv = tmp_path / "scaling.csv"
    with scaling_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "B", "error"])
        writer.writeheader()
        writer.writerow({"regime": "unregularized", "B": 64, "error": 5.5})
        writer.writerow({"regime": "unregularized", "B": 256, "error": 6.4})
        writer.writerow({"regime": "fantope", "B": 64, "error": 4.4})
        writer.writerow({"regime": "fantope", "B": 256, "error": 3.8})

    audit_csv = tmp_path / "spectral_audit.csv"
    with audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "B", "effective_rank"])
        writer.writeheader()
        writer.writerow({"regime": "unregularized", "B": 64, "effective_rank": 5.1})
        writer.writerow({"regime": "unregularized", "B": 256, "effective_rank": 1.3})
        writer.writerow({"regime": "fantope", "B": 64, "effective_rank": 4.2})
        writer.writerow({"regime": "fantope", "B": 256, "effective_rank": 4.0})

    out_dir = tmp_path / "figures"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = gbf.main(
            [
                "--heads-dir",
                str(heads_dir),
                "--include-identity",
                "--closed-loop-json",
                str(mw_summary_path),
                "--scaling-input",
                str(scaling_csv),
                "--spectral-audit",
                str(audit_csv),
                "--unit",
                "cm",
                "--theme",
                "dark",
                "--dpi",
                "150",
                "--output",
                str(out_dir),
            ]
        )

    assert len(written) == 3
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    for path in written:
        _assert_nonempty_png(path)


def _synthetic_spatial_anatomy(grid_size: int = 16) -> dict[str, "np.ndarray"]:
    import numpy as np

    rng = np.random.default_rng(42)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[:, :, 0] = 48
    rgb[:, :, 1] = 64
    rgb[:, :, 2] = 84
    # Tabletop region (rows ~8..12) and bright puck blob near row 9, col 8
    rgb[112:182, 28:196] = np.array([175, 165, 150], dtype=np.uint8)
    rgb[126:148, 108:132] = np.array([220, 60, 50], dtype=np.uint8)

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    gt_puck = np.exp(-0.5 * (((yy - 9.0) / 0.85) ** 2 + ((xx - 8.0) / 1.4) ** 2))
    gt_puck = gt_puck / gt_puck.sum()

    # Shortcut map concentrated in top-2 static border rows
    shortcut = np.full((grid_size, grid_size), 1e-4, dtype=np.float64)
    shortcut[0, :4] = 0.20
    shortcut[1, 12:] = 0.04
    shortcut = shortcut / shortcut.sum()

    # Closed-form ANOVA map localized on the tabletop puck region with soft floor
    anova = 0.75 * gt_puck + 0.25 / float(grid_size * grid_size) + rng.uniform(0.0, 2e-4, size=(grid_size, grid_size))
    anova = anova / anova.sum()

    return {
        "rgb_frame": rgb,
        "gt_puck_map": gt_puck,
        "shortcut_weights": shortcut,
        "anova_weights": anova,
    }


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_plot_spatial_anatomy_comparison_light_dark_and_1d_weights(tmp_path: Path, theme: str):
    data = _synthetic_spatial_anatomy(16)
    out_png = tmp_path / f"spatial_anatomy_{theme}.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = bf.plot_spatial_anatomy_comparison(
            data["rgb_frame"],
            data["gt_puck_map"],
            data["shortcut_weights"].ravel(),  # verify 1D [256] support
            data["anova_weights"].ravel(),
            out_png,
            theme=theme,
            dpi=150,
            also_pdf=(theme == "light"),
        )

    assert returned == out_png
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    _assert_nonempty_png(out_png)
    if theme == "light":
        pdf_path = out_png.with_suffix(".pdf")
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(PDF_MAGIC)


@pytest.mark.parametrize("ext", [".pt", ".npz", ".json"])
def test_generate_benchmark_figures_cli_spatial_anatomy_formats(tmp_path: Path, ext: str):
    import numpy as np

    data = _synthetic_spatial_anatomy(16)
    src_path = tmp_path / f"anatomy{ext}"
    if ext == ".pt":
        torch.save({k: torch.from_numpy(v) for k, v in data.items()}, src_path)
    elif ext == ".npz":
        np.savez(src_path, **data)
    else:
        src_path.write_text(json.dumps({k: v.tolist() for k, v in data.items()}))

    out_dir = tmp_path / f"figs_{ext.lstrip('.')}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = gbf.main(
            [
                "--spatial-anatomy",
                str(src_path),
                "--theme",
                "light",
                "--dpi",
                "150",
                "--pdf",
                "--output",
                str(out_dir),
            ]
        )

    expected_png = out_dir / "spatial_anatomy_comparison.png"
    assert written == [expected_png]
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
    _assert_nonempty_png(expected_png)
    expected_pdf = expected_png.with_suffix(".pdf")
    assert expected_pdf.exists()
    assert expected_pdf.read_bytes().startswith(PDF_MAGIC)
