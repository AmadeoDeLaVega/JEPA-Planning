"""Publication-grade Matplotlib figure builders for calibrated goal-head benchmarks.

Provides light- and dark-theme configurable figure functions that return
``matplotlib.figure.Figure`` instances and optionally export high-DPI PNG/PDF
files without requiring an interactive display backend:

  - :func:`plot_eigenspectra_comparison`: two-panel linear/log eigenspectrum
    comparison with effective-rank annotations and uniform reference line.
  - :func:`plot_closed_loop_forest`: two-panel forest plot showing closed-loop
    mean ± 95% CI, IQM diamonds, paired bootstrap contrasts, and task win counts
    for either 2D PushT (``unit="px"``) or 3D Meta-World Sawyer Push (``unit="cm"``).
  - :func:`plot_budget_scaling_curve`: calibration budget ``B`` on a log-x axis
    against closed-loop error and spectral effective rank ``r_eff``.
  - :func:`plot_spatial_anatomy_comparison`: four-panel 16x16 spatial patch
    anatomy comparing the Sawyer camera view, true puck-shift energy,
    end-to-end InfoNCE border-token shortcut collapse, and the closed-form
    ANOVA spatial prior.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

THEMES: dict[str, dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "panel": "#ffffff",
        "ink": "#111214",
        "ink_muted": "#52545c",
        "grid": "#e3e5e8",
        "spine": "#c9cdd4",
        "neutral": "#7b808a",
        "zero": "#2d3138",
        "positive": "#169c6c",
        "negative": "#d1434b",
        "palette": [
            "#2a78d6",  # blue
            "#e8612c",  # orange
            "#169c6c",  # teal
            "#7c4dff",  # purple
            "#d99100",  # amber
            "#c93b4b",  # crimson
        ],
    },
    "dark": {
        "surface": "#0e131b",
        "panel": "#151c27",
        "ink": "#f1f5f9",
        "ink_muted": "#9fb0c7",
        "grid": "#263245",
        "spine": "#384861",
        "neutral": "#8c9db5",
        "zero": "#d4dfed",
        "positive": "#34d399",
        "negative": "#f87171",
        "palette": [
            "#5aa9ff",  # blue
            "#ff8250",  # orange
            "#34d399",  # teal
            "#a78bfa",  # purple
            "#fbbf24",  # amber
            "#f87171",  # crimson
        ],
    },
}

REGIME_MARKERS = {
    "unregularized": "s",
    "isotropy": "o",
    "fantope": "^",
    "identity": "D",
}

REGIME_LINESTYLES = {
    "unregularized": "--",
    "isotropy": "-",
    "fantope": "-",
    "identity": ":",
}


def _theme_tokens(theme: str) -> dict[str, object]:
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {tuple(THEMES)}, got {theme!r}")
    return THEMES[theme]


def _semantic_color(label: str, index: int, tokens: dict[str, object]) -> str:
    """Assign a consistent, colorblind-safe color based on regime keywords."""
    palette: list[str] = tokens["palette"]  # type: ignore[assignment]
    lower = label.lower()
    if "fantope" in lower:
        return palette[2]  # teal
    if "isotropy" in lower or "iso" in lower:
        return palette[0]  # blue
    if "unreg" in lower or "vanilla" in lower or "collapse" in lower:
        return palette[1]  # orange
    if "identity" in lower or "plain mse" in lower or "plain_mse" in lower:
        return str(tokens["neutral"])
    if "shuffled" in lower or "random" in lower:
        return palette[4]  # amber
    return palette[index % len(palette)]


def _style_axes(ax, tokens: dict[str, object], *, grid_axis: str = "both") -> None:
    """Apply clean publication styling to a Matplotlib Axes object."""
    ax.set_facecolor(tokens["panel"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(tokens["spine"])
    ax.spines["bottom"].set_color(tokens["spine"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors=tokens["ink_muted"], labelsize=9, length=3.5, width=0.8)
    ax.xaxis.label.set_color(tokens["ink"])
    ax.yaxis.label.set_color(tokens["ink"])
    ax.title.set_color(tokens["ink"])
    ax.grid(True, axis=grid_axis, color=tokens["grid"], linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)


def _save_figure(
    fig: Figure,
    output_path: Path | str | None,
    *,
    dpi: int = 300,
    also_pdf: bool = False,
) -> None:
    """Render and write ``fig`` using the Agg canvas without touching GUI backends."""
    FigureCanvasAgg(fig)
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    if also_pdf and path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())


def _normalized_eigenvalues(head: torch.Tensor) -> tuple[np.ndarray, float]:
    """Return sorted normalized eigenvalues of W = L^T L and participation effective rank."""
    matrix = torch.as_tensor(head, dtype=torch.float32)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("each head must be a non-empty 2D tensor [r, D]")
    singular_values = torch.linalg.svdvals(matrix).detach().cpu().numpy().astype(np.float64)
    eigenvalues = np.square(singular_values)
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("head must have finite, non-zero Frobenius norm")
    probs = eigenvalues / total
    r_eff = float(1.0 / np.sum(np.square(probs)))
    return probs, r_eff


def _first_present(row: Mapping[str, object], keys: Sequence[str], default=None):
    for key in keys:
        if key in row and row[key] is not None and row[key] != "":
            return row[key]
    return default


def _format_value(value: float, unit: str, signed: bool = False) -> str:
    decimals = 2 if unit == "cm" else 1
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}"


def plot_eigenspectra_comparison(
    heads: dict[str, torch.Tensor],
    output_path: Path | str | None = None,
    *,
    top_k: int = 12,
    theme: str = "light",
    title: str = "Goal-head eigenspectra and effective rank",
    dpi: int = 300,
    also_pdf: bool = False,
) -> Figure:
    """Two-panel eigenspectrum plot comparing spectral collapse against regularized heads.

    Panel A displays the top ``top_k`` (default 12) normalized eigenvalues on a
    linear scale. Panel B shows the full rank spectrum on a log y-axis with a
    uniform ``1 / r`` reference line and ``r_eff`` legend labels.
    """
    if not heads:
        raise ValueError("heads mapping must be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    tokens = _theme_tokens(theme)
    fig = Figure(figsize=(10.6, 4.4), dpi=dpi, facecolor=tokens["surface"])
    ax_lin, ax_log = fig.subplots(1, 2)
    _style_axes(ax_lin, tokens)
    _style_axes(ax_log, tokens)

    spectra: list[tuple[str, np.ndarray, float, str]] = []
    for idx, (label, matrix) in enumerate(heads.items()):
        probs, r_eff = _normalized_eigenvalues(matrix)
        color = _semantic_color(label, idx, tokens)
        spectra.append((label, probs, r_eff, color))

    max_lin_k = max(min(top_k, len(probs)) for _, probs, _, _ in spectra)
    max_eig = 0.0
    max_rank = 1
    low_ranks = [len(probs) for label, probs, _, _ in spectra if "identity" not in label.lower()]
    ref_rank = min(low_ranks) if low_ranks else min(len(probs) for _, probs, _, _ in spectra)

    log_floor = 1e-6
    min_positive_eig = 1.0

    for idx, (label, probs, r_eff, color) in enumerate(spectra):
        lower = label.lower()
        marker = "o"
        linestyle = "-"
        for key, m_val in REGIME_MARKERS.items():
            if key in lower:
                marker = m_val
                linestyle = REGIME_LINESTYLES[key]
                break

        k_slice = min(top_k, len(probs))
        x_lin = np.arange(1, k_slice + 1)
        y_lin = probs[:k_slice]
        max_eig = max(max_eig, float(y_lin.max()))
        legend_label = f"{label} ($r_{{\\mathrm{{eff}}}}={r_eff:.1f}$)"

        ax_lin.plot(
            x_lin,
            y_lin,
            color=color,
            linewidth=2.1,
            linestyle=linestyle,
            marker=marker,
            markersize=5.2,
            markeredgecolor=tokens["panel"],
            markeredgewidth=0.9,
            label=legend_label,
            zorder=3 + idx * 0.1,
        )

        x_full = np.arange(1, len(probs) + 1)
        max_rank = max(max_rank, len(probs))
        positive = probs[probs > log_floor]
        if len(positive):
            min_positive_eig = min(min_positive_eig, float(positive.min()))
        y_log = np.maximum(probs, log_floor)

        ax_log.plot(
            x_full,
            y_log,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            marker=marker if len(probs) <= 40 else None,
            markersize=4.0,
            markeredgecolor=tokens["panel"],
            markeredgewidth=0.7,
            label=legend_label,
            zorder=3 + idx * 0.1,
        )

    # Panel A formatting (linear top-k)
    ax_lin.set_title(f"A. Top-{max_lin_k} normalized eigenvalues (linear)", loc="left", fontsize=10.5, fontweight="bold")
    ax_lin.set_xlabel("Eigenvalue index $k$", fontsize=9.5)
    ax_lin.set_ylabel("Normalized eigenvalue $\\lambda_k$", fontsize=9.5)
    ax_lin.set_xticks(np.arange(1, max_lin_k + 1))
    ax_lin.set_xlim(0.6, max_lin_k + 0.4)
    ax_lin.set_ylim(-0.02, min(1.06, max(0.2, max_eig * 1.16)))

    # Panel B formatting (full rank log scale + uniform reference)
    uniform_val = 1.0 / float(ref_rank)
    ax_log.axhline(
        uniform_val,
        color=tokens["neutral"],
        linestyle=":",
        linewidth=1.5,
        label=f"Uniform $1/{ref_rank} = {uniform_val:.3f}$",
        zorder=2,
    )
    ax_log.set_yscale("log")
    ax_log.set_title("B. Full-rank eigenspectrum (log scale)", loc="left", fontsize=10.5, fontweight="bold")
    ax_log.set_xlabel("Eigenvalue index $k$", fontsize=9.5)
    ax_log.set_ylabel("Normalized eigenvalue $\\lambda_k$ (log)", fontsize=9.5)
    ax_log.set_xlim(0.7, max_rank + max(0.5, 0.03 * max_rank))
    lower_bound = max(log_floor, min(min_positive_eig * 0.4, uniform_val * 0.1))
    ax_log.set_ylim(lower_bound, 1.45)

    legend = ax_log.legend(
        loc="lower left",
        frameon=True,
        facecolor=tokens["panel"],
        edgecolor=tokens["grid"],
        fontsize=8.3,
    )
    for text in legend.get_texts():
        text.set_color(tokens["ink"])

    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=12, fontweight="bold", color=tokens["ink"])
    fig.text(
        0.02,
        0.015,
        "Eigenvalues of $W = L^\\top L$ with $\\mathrm{tr}(W) = 1$; "
        "$r_{\\mathrm{eff}} = 1 / \\sum_k \\lambda_k^2$.",
        fontsize=8.2,
        color=tokens["ink_muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.84, bottom=0.17, wspace=0.24)
    _save_figure(fig, output_path, dpi=dpi, also_pdf=also_pdf)
    return fig


def plot_closed_loop_forest(
    summary_rows: list[dict],
    paired_contrasts: list[dict],
    unit: str,
    title: str,
    output_path: Path | str | None = None,
    *,
    theme: str = "light",
    metric_label: str | None = None,
    dpi: int = 300,
    also_pdf: bool = False,
) -> Figure:
    """Two-panel forest plot for closed-loop control error and paired contrasts.

    Panel A plots condition mean ± 95% CI horizontal whiskers together with IQM
    diamond markers. Panel B plots paired task-level bootstrap difference CIs
    against zero with optional win-count annotations. Works for PushT
    (``unit="px"``) and 3D Meta-World Sawyer Push (``unit="cm"``).
    """
    if not summary_rows:
        raise ValueError("summary_rows must be non-empty")

    tokens = _theme_tokens(theme)
    n_left = len(summary_rows)
    n_right = len(paired_contrasts)
    height = max(3.9, 0.56 * max(n_left, n_right, 3) + 1.65)

    fig = Figure(figsize=(15.8, height), dpi=dpi, facecolor=tokens["surface"])
    ax_abs, ax_diff = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})
    _style_axes(ax_abs, tokens, grid_axis="x")
    _style_axes(ax_diff, tokens, grid_axis="x")

    # ---- Panel A: Absolute error mean ± 95% CI + IQM diamonds ----
    y_abs = np.arange(n_left)[::-1].astype(np.float64)
    abs_labels: list[str] = []
    abs_hi_max = 0.0
    abs_lo_min = float("inf")
    has_iqm = False
    baseline_row = None

    for idx, (yi, row) in enumerate(zip(y_abs, summary_rows)):
        label = str(
            _first_present(
                row,
                ("label", "name"),
                default=f"{row.get('head', 'head')} + {row.get('planner', 'planner')}",
            )
        )
        mean = float(_first_present(row, ("mean", "error")))
        ci_lo = float(_first_present(row, ("ci_lo", "mean_lo"), default=mean))
        ci_hi = float(_first_present(row, ("ci_hi", "mean_hi"), default=mean))
        iqm_val = _first_present(row, ("iqm",))
        color = str(row["color"]) if row.get("color") else _semantic_color(label, idx, tokens)

        if bool(row.get("is_baseline")) or (
            baseline_row is None and ("plain mse" in label.lower() or "identity" in label.lower())
        ):
            baseline_row = (mean, ci_lo, ci_hi)

        ax_abs.plot([ci_lo, ci_hi], [yi, yi], color=color, linewidth=2.3, solid_capstyle="round", zorder=3)
        ax_abs.scatter(
            [mean],
            [yi],
            s=58,
            color=color,
            edgecolor=tokens["panel"],
            linewidth=1.4,
            zorder=4,
        )
        if iqm_val is not None:
            has_iqm = True
            ax_abs.scatter(
                [float(iqm_val)],
                [yi],
                s=52,
                marker="D",
                facecolor=tokens["panel"],
                edgecolor=color,
                linewidth=1.6,
                zorder=5,
            )

        abs_labels.append(label)
        abs_hi_max = max(abs_hi_max, ci_hi, mean, float(iqm_val) if iqm_val is not None else mean)
        abs_lo_min = min(abs_lo_min, ci_lo, mean, float(iqm_val) if iqm_val is not None else mean)

    if baseline_row is not None:
        b_mean, b_lo, b_hi = baseline_row
        ax_abs.axvspan(b_lo, b_hi, color=tokens["neutral"], alpha=0.12, linewidth=0, zorder=1)
        ax_abs.axvline(b_mean, color=tokens["neutral"], linestyle="--", linewidth=1.2, zorder=2)

    span_abs = max(abs_hi_max - min(0.0, abs_lo_min), 1.0)
    for yi, row in zip(y_abs, summary_rows):
        mean = float(_first_present(row, ("mean", "error")))
        ci_hi = float(_first_present(row, ("ci_hi", "mean_hi"), default=mean))
        iqm_val = _first_present(row, ("iqm",))
        anchor_x = max(ci_hi, float(iqm_val) if iqm_val is not None else ci_hi)
        text = _format_value(mean, unit)
        if iqm_val is not None:
            text += f" ({_format_value(float(iqm_val), unit)})"
        ax_abs.text(
            anchor_x + 0.022 * span_abs,
            yi,
            text,
            va="center",
            ha="left",
            fontsize=8.3,
            color=tokens["ink_muted"],
        )

    ax_abs.set_yticks(y_abs)
    ax_abs.set_yticklabels(abs_labels, fontsize=8.8, color=tokens["ink"])
    ax_abs.set_ylim(-1.05, n_left - 0.3)
    ax_abs.set_xlim(min(0.0, abs_lo_min - 0.05 * span_abs), abs_hi_max + 0.36 * span_abs)
    abs_metric = metric_label or ("Final 3D puck error" if unit == "cm" else "Final T position error")
    ax_abs.set_xlabel(f"{abs_metric} ({unit}), lower is better", fontsize=9.2)
    ax_abs.set_title("A. Closed-loop error (mean ± 95% CI, IQM)", loc="left", fontsize=10.2, fontweight="bold")

    legend_handles = [
        Line2D(
            [],
            [],
            color=tokens["ink_muted"],
            marker="o",
            markersize=6.5,
            linewidth=2.0,
            markerfacecolor=tokens["ink_muted"],
            markeredgecolor=tokens["panel"],
            label="Mean ± 95% CI",
        )
    ]
    if has_iqm:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="none",
                marker="D",
                markersize=6.0,
                markerfacecolor=tokens["panel"],
                markeredgecolor=tokens["ink_muted"],
                markeredgewidth=1.5,
                label="IQM",
            )
        )
    leg_abs = ax_abs.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor=tokens["panel"],
        edgecolor=tokens["grid"],
        fontsize=8.0,
    )
    for text in leg_abs.get_texts():
        text.set_color(tokens["ink"])

    # ---- Panel B: Paired bootstrap differences + win counts ----
    ax_diff.set_title("B. Paired bootstrap contrasts vs. baseline", loc="left", fontsize=10.2, fontweight="bold")
    ax_diff.axvline(0.0, color=tokens["zero"], linestyle="--", linewidth=1.3, zorder=2)

    if not paired_contrasts:
        ax_diff.text(
            0.5,
            0.5,
            "No paired contrasts provided",
            transform=ax_diff.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=tokens["ink_muted"],
        )
        ax_diff.set_yticks([])
        ax_diff.set_xlabel(f"Paired difference ({unit}), negative is better", fontsize=9.2)
    else:
        y_diff = np.arange(n_right)[::-1].astype(np.float64)
        diff_labels: list[str] = []
        lo_all = 0.0
        hi_all = 0.0

        parsed_contrasts: list[tuple[str, float, float, float, str, str]] = []
        for row in paired_contrasts:
            label = str(_first_present(row, ("label", "name"), default="contrast"))
            diff = float(_first_present(row, ("diff", "mean", "delta")))
            ci_lo = float(_first_present(row, ("ci_lo", "mean_lo"), default=diff))
            ci_hi = float(_first_present(row, ("ci_hi", "mean_hi"), default=diff))

            if ci_hi < 0.0:
                color = str(tokens["positive"])
            elif ci_lo > 0.0:
                color = str(tokens["negative"])
            else:
                color = str(tokens["palette"][0])  # type: ignore[index]

            win_text = row.get("win_text")
            if not win_text and row.get("wins") is not None:
                wins = int(row["wins"])
                total = _first_present(row, ("total", "n_tasks", "tasks"))
                if total is None and row.get("losses") is not None:
                    total = wins + int(row["losses"]) + int(row.get("ties", 0))
                if total is not None and int(total) > 0:
                    win_text = f"{wins}/{int(total)} wins"
                else:
                    win_text = f"{wins} wins"

            annot = f"{_format_value(diff, unit, signed=True)} {unit}"
            if win_text:
                annot += f"  [{win_text}]"

            parsed_contrasts.append((label, diff, ci_lo, ci_hi, color, annot))
            diff_labels.append(label)
            lo_all = min(lo_all, ci_lo, diff)
            hi_all = max(hi_all, ci_hi, diff)

        span_diff = max(hi_all - lo_all, 1.0)
        left_lim = lo_all - 0.12 * span_diff
        right_lim = hi_all + 0.52 * span_diff
        ax_diff.axvspan(left_lim, 0.0, color=tokens["positive"], alpha=0.055, linewidth=0, zorder=1)

        for yi, (_, diff, ci_lo, ci_hi, color, annot) in zip(y_diff, parsed_contrasts):
            ax_diff.plot([ci_lo, ci_hi], [yi, yi], color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
            ax_diff.scatter(
                [diff],
                [yi],
                s=60,
                color=color,
                edgecolor=tokens["panel"],
                linewidth=1.4,
                zorder=4,
            )
            ax_diff.text(
                max(ci_hi, diff) + 0.025 * span_diff,
                yi,
                annot,
                va="center",
                ha="left",
                fontsize=8.2,
                color=tokens["ink_muted"],
            )

        ax_diff.set_yticks(y_diff)
        ax_diff.set_yticklabels(diff_labels, fontsize=8.6, color=tokens["ink"])
        ax_diff.set_ylim(-0.7, n_right - 0.3)
        ax_diff.set_xlim(left_lim, right_lim)
        ax_diff.set_xlabel(f"Paired difference vs. baseline ({unit}), negative improves", fontsize=9.2)

    fig.suptitle(title, x=0.02, y=0.97, ha="left", fontsize=12, fontweight="bold", color=tokens["ink"])
    fig.text(
        0.02,
        0.015,
        "Whiskers denote task-level 95% percentile bootstrap intervals; "
        "values in parentheses in Panel A report IQM.",
        fontsize=8.2,
        color=tokens["ink_muted"],
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.84, bottom=0.17, wspace=0.92)
    _save_figure(fig, output_path, dpi=dpi, also_pdf=also_pdf)
    return fig


def plot_budget_scaling_curve(
    scaling_rows: list[dict],
    output_path: Path | str | None = None,
    *,
    unit: str = "px",
    theme: str = "light",
    title: str = "Calibration budget scaling: spectral collapse vs. regularization",
    baseline_error: float | None = None,
    baseline_label: str = "Plain MSE",
    dpi: int = 300,
    also_pdf: bool = False,
) -> Figure:
    """Two-panel scaling figure over calibration budget ``B`` (log-x axis).

    Panel A plots closed-loop error vs. calibration budget ``B``. Panel B plots
    the corresponding spectral ``effective_rank`` vs. ``B``, highlighting rank-1
    collapse under unregularized InfoNCE versus stability under spectral
    regularization.
    """
    if not scaling_rows:
        raise ValueError("scaling_rows must be non-empty")

    tokens = _theme_tokens(theme)
    grouped: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    all_budgets: set[float] = set()

    for row in scaling_rows:
        regime = str(_first_present(row, ("regime", "label", "method"), default="method"))
        budget = float(_first_present(row, ("B", "budget")))
        if budget <= 0:
            raise ValueError(f"calibration budget B must be positive, got {budget}")
        grouped[regime][budget].append(row)
        all_budgets.add(budget)

    fig = Figure(figsize=(10.6, 4.4), dpi=dpi, facecolor=tokens["surface"])
    ax_err, ax_rank = fig.subplots(1, 2)
    _style_axes(ax_err, tokens)
    _style_axes(ax_rank, tokens)

    sorted_budgets = sorted(all_budgets)
    regime_order = sorted(
        grouped.keys(),
        key=lambda r: (
            0 if "unreg" in r.lower() else 1 if "iso" in r.lower() else 2 if "fantope" in r.lower() else 3,
            r,
        ),
    )

    err_min, err_max = float("inf"), float("-inf")
    rank_max = 1.0

    for idx, regime in enumerate(regime_order):
        color = _semantic_color(regime, idx, tokens)
        lower = regime.lower()
        marker = "o"
        linestyle = "-"
        for key, m_val in REGIME_MARKERS.items():
            if key in lower:
                marker = m_val
                linestyle = REGIME_LINESTYLES[key]
                break

        xs: list[float] = []
        err_means: list[float] = []
        err_los: list[float] = []
        err_his: list[float] = []
        rank_means: list[float] = []
        rank_los: list[float] = []
        rank_his: list[float] = []

        for budget in sorted(grouped[regime].keys()):
            bucket = grouped[regime][budget]
            errs = [
                float(
                    _first_present(
                        r,
                        ("error", "closed_loop_error", "mean", "final_position_error", "final_error_3d_cm"),
                    )
                )
                for r in bucket
            ]
            ranks = [float(_first_present(r, ("effective_rank", "r_eff"))) for r in bucket]

            e_mean = float(np.mean(errs))
            e_lo_explicit = _first_present(bucket[0], ("ci_lo", "error_lo", "mean_lo")) if len(bucket) == 1 else None
            e_hi_explicit = _first_present(bucket[0], ("ci_hi", "error_hi", "mean_hi")) if len(bucket) == 1 else None
            e_lo = float(e_lo_explicit) if e_lo_explicit is not None else float(np.min(errs))
            e_hi = float(e_hi_explicit) if e_hi_explicit is not None else float(np.max(errs))

            r_mean = float(np.mean(ranks))
            r_lo_explicit = _first_present(bucket[0], ("effective_rank_lo", "r_eff_lo")) if len(bucket) == 1 else None
            r_hi_explicit = _first_present(bucket[0], ("effective_rank_hi", "r_eff_hi")) if len(bucket) == 1 else None
            r_lo = float(r_lo_explicit) if r_lo_explicit is not None else float(np.min(ranks))
            r_hi = float(r_hi_explicit) if r_hi_explicit is not None else float(np.max(ranks))

            xs.append(budget)
            err_means.append(e_mean)
            err_los.append(e_lo)
            err_his.append(e_hi)
            rank_means.append(r_mean)
            rank_los.append(r_lo)
            rank_his.append(r_hi)

            err_min = min(err_min, e_lo, e_mean)
            err_max = max(err_max, e_hi, e_mean)
            rank_max = max(rank_max, r_hi, r_mean)

        x_arr = np.asarray(xs, dtype=np.float64)
        ax_err.plot(
            x_arr,
            err_means,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            marker=marker,
            markersize=6.2,
            markeredgecolor=tokens["panel"],
            markeredgewidth=1.0,
            label=regime,
            zorder=3,
        )
        if any(hi > lo for lo, hi in zip(err_los, err_his)):
            ax_err.fill_between(x_arr, err_los, err_his, color=color, alpha=0.14, linewidth=0, zorder=2)

        ax_rank.plot(
            x_arr,
            rank_means,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            marker=marker,
            markersize=6.2,
            markeredgecolor=tokens["panel"],
            markeredgewidth=1.0,
            label=regime,
            zorder=3,
        )
        if any(hi > lo for lo, hi in zip(rank_los, rank_his)):
            ax_rank.fill_between(x_arr, rank_los, rank_his, color=color, alpha=0.14, linewidth=0, zorder=2)

    if baseline_error is not None:
        ax_err.axhline(
            float(baseline_error),
            color=tokens["neutral"],
            linestyle=":",
            linewidth=1.6,
            label=f"{baseline_label} ({_format_value(float(baseline_error), unit)} {unit})",
            zorder=2,
        )
        err_min = min(err_min, float(baseline_error))
        err_max = max(err_max, float(baseline_error))

    ax_rank.axhline(
        1.0,
        color=tokens["negative"],
        linestyle=":",
        linewidth=1.3,
        alpha=0.85,
        label="Rank-1 collapse ($r_{\\mathrm{eff}}=1$)",
        zorder=2,
    )

    for ax in (ax_err, ax_rank):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted_budgets)
        ax.set_xticklabels([f"{int(b)}" if float(b).is_integer() else f"{b:g}" for b in sorted_budgets])
        if len(sorted_budgets) >= 2:
            ax.set_xlim(sorted_budgets[0] * 0.72, sorted_budgets[-1] * 1.38)
        else:
            ax.set_xlim(sorted_budgets[0] * 0.5, sorted_budgets[0] * 2.0)
        ax.set_xlabel("Calibration budget $B$ (log scale)", fontsize=9.5)

    pad_err = max((err_max - err_min) * 0.16, 1.0 if unit == "px" else 0.3)
    ax_err.set_ylim(max(0.0, err_min - pad_err), err_max + pad_err)
    ax_err.set_ylabel(f"Closed-loop error ({unit}), lower is better", fontsize=9.5)
    ax_err.set_title("A. Closed-loop error vs. budget $B$", loc="left", fontsize=10.5, fontweight="bold")

    ax_rank.set_ylim(0.5, max(rank_max * 1.18, 2.5))
    ax_rank.set_ylabel("Effective rank $r_{\\mathrm{eff}}$", fontsize=9.5)
    ax_rank.set_title("B. Spectral effective rank vs. budget $B$", loc="left", fontsize=10.5, fontweight="bold")

    leg_err = ax_err.legend(
        loc="best",
        frameon=True,
        facecolor=tokens["panel"],
        edgecolor=tokens["grid"],
        fontsize=8.3,
    )
    leg_rank = ax_rank.legend(
        loc="best",
        frameon=True,
        facecolor=tokens["panel"],
        edgecolor=tokens["grid"],
        fontsize=8.3,
    )
    for legend in (leg_err, leg_rank):
        for text in legend.get_texts():
            text.set_color(tokens["ink"])

    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=12, fontweight="bold", color=tokens["ink"])
    fig.text(
        0.02,
        0.015,
        "Unregularized InfoNCE collapses toward $r_{\\mathrm{eff}} \\approx 1$ at large $B$, "
        "whereas spectral regularization preserves multi-axis pose sensitivity.",
        fontsize=8.2,
        color=tokens["ink_muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.84, bottom=0.17, wspace=0.24)
    _save_figure(fig, output_path, dpi=dpi, also_pdf=also_pdf)
    return fig


def _to_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _normalize_rgb_frame(rgb_frame: np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate an RGB frame and return float32 ``[H, W, 3]`` in ``[0, 1]``."""
    raw = _to_numpy(rgb_frame)
    if raw.ndim != 3 or raw.shape[2] != 3 or min(raw.shape[:2]) == 0:
        raise ValueError(f"rgb_frame must have shape [H, W, 3], got {tuple(raw.shape)}")
    arr = raw.astype(np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("rgb_frame must contain finite values")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_patch_grid(values: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert a ``[G, G]`` or ``[G*G]`` patch array into a finite float64 ``[G, G]`` grid."""
    raw = _to_numpy(values).astype(np.float64)
    if raw.ndim == 1:
        side = int(round(math.sqrt(raw.size)))
        if side < 1 or side * side != raw.size:
            raise ValueError(f"{name} 1D array length ({raw.size}) must be a positive perfect square")
        grid = raw.reshape(side, side)
    elif raw.ndim == 2 and min(raw.shape) > 0:
        grid = raw
    else:
        raise ValueError(f"{name} must be a non-empty 2D grid or 1D square vector, got shape {tuple(raw.shape)}")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite values")
    return grid


def _normalize_simplex_grid(grid: np.ndarray) -> np.ndarray:
    clamped = np.maximum(grid, 0.0)
    total = float(clamped.sum())
    if total <= 0.0:
        return np.full_like(clamped, 1.0 / clamped.size, dtype=np.float64)
    return clamped / total


def _pearson_r(first: np.ndarray, second: np.ndarray) -> float:
    x = first.ravel().astype(np.float64)
    y = second.ravel().astype(np.float64)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(dx, dy) / denom)


def plot_spatial_anatomy_comparison(
    rgb_frame: np.ndarray,
    gt_puck_map: np.ndarray,
    shortcut_weights: np.ndarray,
    anova_weights: np.ndarray,
    output_path: Path,
    *,
    theme: str = "light",
    dpi: int = 300,
    also_pdf: bool = False,
) -> Path:
    """Four-panel 16x16 spatial anatomy & ViT border-token shortcut comparison figure.

    Panels (left to right):
      1. ``Sawyer Push-v3 (corner2, 224x224)`` RGB view with subtle ``16x16`` patch grid.
      2. ``True 5 cm Puck Shift Energy`` highlighting tabletop row-8..10 concentration.
      3. ``End-to-End InfoNCE Spatial w`` showing border-token shortcut mass in top rows
         and inverse-Simpson effective patch count ``P_eff``.
      4. ``Closed-Form ANOVA Spatial Prior`` showing tabletop puck localization, ``P_eff``,
         and Pearson ``r`` against the ground-truth puck-shift map.
    """
    tokens = _theme_tokens(theme)
    rgb = _normalize_rgb_frame(rgb_frame)
    gt_grid = _to_patch_grid(gt_puck_map, "gt_puck_map")
    sc_grid = _to_patch_grid(shortcut_weights, "shortcut_weights")
    an_grid = _to_patch_grid(anova_weights, "anova_weights")

    if sc_grid.shape != gt_grid.shape or an_grid.shape != gt_grid.shape:
        raise ValueError(
            "gt_puck_map, shortcut_weights, and anova_weights must share the same grid shape"
        )

    grid_h, grid_w = gt_grid.shape
    gt_prob = _normalize_simplex_grid(gt_grid)
    sc_prob = _normalize_simplex_grid(sc_grid)
    an_prob = _normalize_simplex_grid(an_grid)

    p_eff_sc = float(1.0 / max(float(np.sum(sc_prob * sc_prob)), 1e-12))
    p_eff_an = float(1.0 / max(float(np.sum(an_prob * an_prob)), 1e-12))
    r_sc = _pearson_r(sc_prob, gt_prob)
    r_an = _pearson_r(an_prob, gt_prob)

    r_lo = min(8, grid_h - 1)
    r_hi = min(10, grid_h - 1)
    top_rows = min(2, grid_h)

    gt_band_pct = 100.0 * float(gt_prob[r_lo : r_hi + 1, :].sum())
    sc_top_pct = 100.0 * float(sc_prob[:top_rows, :].sum())
    an_band_pct = 100.0 * float(an_prob[r_lo : r_hi + 1, :].sum())
    peak_row_gt = int(np.argmax(gt_prob.sum(axis=1)))

    def _display_norm(mat: np.ndarray) -> np.ndarray:
        peak = float(mat.max())
        return mat / peak if peak > 1e-12 else np.zeros_like(mat)

    gt_disp = _display_norm(gt_prob)
    sc_disp = _display_norm(sc_prob)
    an_disp = _display_norm(an_prob)

    fig = Figure(figsize=(13.5, 3.6), dpi=dpi, facecolor=tokens["surface"])
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055], wspace=0.14)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_an = fig.add_subplot(gs[0, 3])
    cax = fig.add_subplot(gs[0, 4])

    # ---- Panel 1: RGB camera view with 16x16 patch grid ----
    img_h, img_w = rgb.shape[:2]
    ax_rgb.imshow(rgb, aspect="equal")
    for idx in range(1, grid_w):
        ax_rgb.axvline(idx * (img_w / grid_w) - 0.5, color="#ffffff", linewidth=0.45, alpha=0.32)
    for idx in range(1, grid_h):
        ax_rgb.axhline(idx * (img_h / grid_h) - 0.5, color="#ffffff", linewidth=0.45, alpha=0.32)
    ax_rgb.set_xlim(-0.5, img_w - 0.5)
    ax_rgb.set_ylim(img_h - 0.5, -0.5)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    for spine in ax_rgb.spines.values():
        spine.set_color(tokens["spine"])
        spine.set_linewidth(0.9)
    ax_rgb.set_title(
        f"Sawyer Push-v3 (corner2, {img_h}x{img_w})",
        fontsize=9.6,
        fontweight="bold",
        color=tokens["ink"],
        pad=7,
    )
    ax_rgb.set_xlabel(
        f"{grid_h}x{grid_w} ViT patch grid",
        fontsize=8.3,
        color=tokens["ink_muted"],
        labelpad=6,
    )

    # Tick positions for patch grids (0, 4, 8, 12, 15 for 16x16)
    step = max(1, grid_w // 4)
    tick_pos = list(range(0, grid_w, step))
    if tick_pos[-1] != grid_w - 1:
        tick_pos.append(grid_w - 1)
    y_step = max(1, grid_h // 4)
    y_tick_pos = list(range(0, grid_h, y_step))
    if y_tick_pos[-1] != grid_h - 1:
        y_tick_pos.append(grid_h - 1)

    heatmap_specs = [
        (
            ax_gt,
            gt_disp,
            "True 5 cm Puck Shift Energy",
            f"Rows {r_lo}-{r_hi}: {gt_band_pct:.0f}% mass (peak row {peak_row_gt})",
            (r_lo, r_hi, str(tokens["palette"][0])),  # type: ignore[index]
            True,
        ),
        (
            ax_sc,
            sc_disp,
            "End-to-End InfoNCE Spatial w",
            f"$P_{{\\mathrm{{eff}}}}={p_eff_sc:.1f}$  |  top-{top_rows} rows: {sc_top_pct:.0f}%  |  $r={r_sc:+.2f}$",
            (0, top_rows - 1, str(tokens["negative"])),
            False,
        ),
        (
            ax_an,
            an_disp,
            "Closed-Form ANOVA Spatial Prior",
            f"$P_{{\\mathrm{{eff}}}}={p_eff_an:.1f}$  |  rows {r_lo}-{r_hi}: {an_band_pct:.0f}%  |  $r={r_an:+.2f}$",
            (r_lo, r_hi, str(tokens["positive"])),
            False,
        ),
    ]

    im_handle = None
    for ax, mat, panel_title, footer, (band_lo, band_hi, band_color), show_y_labels in heatmap_specs:
        im_handle = ax.imshow(
            mat,
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.add_patch(
            Rectangle(
                (-0.5, band_lo - 0.5),
                grid_w,
                band_hi - band_lo + 1,
                fill=False,
                edgecolor=band_color,
                linewidth=1.55,
                linestyle="--",
                zorder=4,
            )
        )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(t) for t in tick_pos], fontsize=7.8, color=tokens["ink_muted"])
        ax.set_yticks(y_tick_pos)
        if show_y_labels:
            ax.set_yticklabels([str(t) for t in y_tick_pos], fontsize=7.8, color=tokens["ink_muted"])
            ax.set_ylabel("Patch row", fontsize=8.4, color=tokens["ink"], labelpad=3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(colors=tokens["ink_muted"], length=2.8, width=0.75)
        for spine in ax.spines.values():
            spine.set_color(tokens["spine"])
            spine.set_linewidth(0.9)
        ax.set_title(panel_title, fontsize=9.6, fontweight="bold", color=tokens["ink"], pad=7)
        ax.set_xlabel(footer, fontsize=8.2, color=tokens["ink_muted"], labelpad=6)

    assert im_handle is not None
    cbar = fig.colorbar(im_handle, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])
    cbar.ax.tick_params(colors=tokens["ink_muted"], labelsize=8.0, length=2.5, width=0.75)
    cbar.set_label("Relative patch weight", fontsize=8.4, color=tokens["ink"], labelpad=5)
    cbar.outline.set_edgecolor(tokens["spine"])
    cbar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.025, right=0.935, top=0.86, bottom=0.17, wspace=0.14)
    out_path = Path(output_path)
    _save_figure(fig, out_path, dpi=dpi, also_pdf=also_pdf)
    return out_path
