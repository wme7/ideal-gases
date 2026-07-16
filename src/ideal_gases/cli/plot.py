# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Matplotlib visualization for Riemann solution profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ideal_gases.riemann import RiemannResult

PlotKind = Literal["classical", "quantum"]
PlotLayout = Literal["panels", "comparison", "both"]

STATISTIC_COLORS = {"BE": "r", "MB": "b", "FD": "m"}
STATISTICS = ("FD", "MB", "BE")

CLASSICAL_FIELDS = (
    ("rho", r"$\rho$"),
    ("ux", r"$u_x$"),
    ("p", r"$p$"),
    ("e", r"$e$"),
    ("mach", r"$\mathrm{Ma}$"),
    ("entropy", r"$s$"),
)

QUANTUM_FIELDS = (
    ("rho", r"$\rho$"),
    ("ux", r"$u_x$"),
    ("p", r"$p$"),
    ("e", r"$e$"),
    ("t", r"$\theta$"),
    ("z", r"$z$"),
)

QUANTUM_PANEL_FIELDS = (
    ("rho", r"$\rho$"),
    ("ux", r"$u_x$"),
    ("p", r"$p$"),
    ("e", r"$e$"),
    ("t", r"$\theta$"),
    ("z", r"$z$"),
)

COMPARISON_FIELDS = (
    ("rho", r"$\rho$"),
    ("ux", r"$u_x$"),
    ("e", r"$e$"),
    ("p", r"$p$"),
    ("z", r"$z$"),
    ("t", r"$\theta$"),
)


def _require_matplotlib(*, show: bool):
    try:
        import matplotlib
    except ImportError as exc:
        msg = (
            "Plotting requires matplotlib. Install with: pip install ideal-gases[plot]"
        )
        raise RuntimeError(msg) from exc

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def configure_matplotlib(*, show: bool) -> None:
    plt = _require_matplotlib(show=show)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "axes.grid": False,
        }
    )


def figure_path_with_suffix(base: Path, suffix: str) -> Path:
    if base.suffix.lower() == ".png":
        return base.with_name(f"{base.stem}_{suffix}{base.suffix}")
    return base.parent / f"{base.name}_{suffix}.png"


def _finalize_figure(fig, *, output: Path | None, show: bool) -> None:
    plt = _require_matplotlib(show=show)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single(
    result: RiemannResult,
    *,
    title: str,
    output: Path | None,
    show: bool,
    kind: PlotKind,
) -> Path | None:
    """Render a single-solution 2x3 panel figure."""
    plt = _require_matplotlib(show=show)
    configure_matplotlib(show=show)
    fields = QUANTUM_FIELDS if kind == "quantum" else CLASSICAL_FIELDS
    x = result.x

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (attr, label) in zip(axes_flat, fields, strict=True):
        ax.plot(x, getattr(result, attr), color="b")
        ax.set_ylabel(label)
        ax.set_xlabel("$x$")

    fig.suptitle(title, fontsize=11)
    _finalize_figure(fig, output=output, show=show)
    return output


def plot_statistics_panels(
    results: dict[str, RiemannResult],
    *,
    title: str,
    output: Path | None,
    show: bool,
) -> Path | None:
    """Render a 3x6 grid with one row per statistic."""
    plt = _require_matplotlib(show=show)
    configure_matplotlib(show=show)
    x = next(iter(results.values())).x

    fig, axes = plt.subplots(3, 6, figsize=(8, 5), constrained_layout=True)

    for row, statistic in enumerate(STATISTICS):
        result = results[statistic]
        color = STATISTIC_COLORS[statistic]
        for col, (attr, label) in enumerate(QUANTUM_PANEL_FIELDS):
            ax = axes[row, col]
            ax.plot(x, getattr(result, attr), color=color)
            if row == 0:
                ax.set_title(f"{label}-{statistic}")
            if row == 2:
                ax.set_xlabel("$x$")
            ax.autoscale(enable=True, axis="both", tight=True)

    fig.suptitle(title, fontsize=11)
    _finalize_figure(fig, output=output, show=show)
    return output


def plot_statistics_comparison(
    results: dict[str, RiemannResult],
    *,
    title: str,
    output: Path | None,
    show: bool,
) -> Path | None:
    """Render overlay panels comparing FD, MB, and BE on shared axes."""
    plt = _require_matplotlib(show=show)
    configure_matplotlib(show=show)
    x = next(iter(results.values())).x

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (attr, label) in zip(axes_flat, COMPARISON_FIELDS, strict=True):
        for statistic in ("BE", "MB", "FD"):
            result = results[statistic]
            ax.plot(
                x,
                getattr(result, attr),
                color=STATISTIC_COLORS[statistic],
                label=statistic,
            )
        ax.set_ylabel(label)
        ax.set_xlabel("$x$")
        ax.legend(loc="best")

    fig.suptitle(title, fontsize=11)
    _finalize_figure(fig, output=output, show=show)
    return output


def plot_all_statistics(
    results: dict[str, RiemannResult],
    *,
    title: str,
    figure: Path,
    layout: PlotLayout,
    show: bool,
) -> list[Path]:
    """Render all-statistics figure(s) according to layout."""
    saved: list[Path] = []

    if layout in ("panels", "both"):
        panels_path = figure_path_with_suffix(figure, "panels")
        plot_statistics_panels(
            results,
            title=f"{title} (panels)",
            output=panels_path,
            show=show,
        )
        saved.append(panels_path)

    if layout in ("comparison", "both"):
        comparison_path = figure_path_with_suffix(figure, "comparison")
        plot_statistics_comparison(
            results,
            title=f"{title} (comparison)",
            output=comparison_path,
            show=show,
        )
        saved.append(comparison_path)

    return saved
