#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Plot fractional polylogarithms for visual verification.

Ports the workflow of ``matlab/PlotPolyLog.m`` using :func:`ideal_gases.polylog`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ideal_gases import polylog

ROOT = Path(__file__).resolve().parents[1]

# MATLAB color list: k, r, m, b, c, g, y (1-based indices 1..7).
MATLAB_COLORS = ("k", "r", "m", "b", "c", "g", "y")


@dataclass(frozen=True)
class CurveSpec:
    order: float
    linestyle: str
    color_index: int
    label: str


CURVES: tuple[CurveSpec, ...] = (
    CurveSpec(7 / 2, "-", 7, r"$\mathrm{Li}_{7/2}(z)$"),
    CurveSpec(3.0, "-", 6, r"$\mathrm{Li}_{3}(z)$"),
    CurveSpec(5 / 2, "-", 5, r"$\mathrm{Li}_{5/2}(z)$"),
    CurveSpec(2.0, "-", 4, r"$\mathrm{Li}_{2}(z)$"),
    CurveSpec(3 / 2, "-", 3, r"$\mathrm{Li}_{3/2}(z)$"),
    CurveSpec(1.0, "-", 2, r"$\mathrm{Li}_{1}(z)$"),
    CurveSpec(1 / 2, "-", 1, r"$\mathrm{Li}_{1/2}(z)$"),
    CurveSpec(0.0, "-.", 1, r"$\mathrm{Li}_{0}(z)$"),
    CurveSpec(-1 / 2, "--", 1, r"$\mathrm{Li}_{-1/2}(z)$"),
    CurveSpec(-1.0, "--", 2, r"$\mathrm{Li}_{-1}(z)$"),
    CurveSpec(-3 / 2, "--", 3, r"$\mathrm{Li}_{-3/2}(z)$"),
    CurveSpec(-2.0, "--", 4, r"$\mathrm{Li}_{-2}(z)$"),
    CurveSpec(-5 / 2, "--", 5, r"$\mathrm{Li}_{-5/2}(z)$"),
    CurveSpec(-3.0, "--", 6, r"$\mathrm{Li}_{-3}(z)$"),
    CurveSpec(-7 / 2, "--", 7, r"$\mathrm{Li}_{-7/2}(z)$"),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "legend.fontsize": 14,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "axes.linewidth": 1.5,
            "axes.grid": True,
        }
    )


def _z_grid(z_min: float, z_max: float, dz: float) -> NDArray[np.float64]:
    count = int(np.floor((z_max - z_min) / dz + 0.5)) + 1
    return np.linspace(z_min, z_max, count, dtype=np.float64)


def plot_polylogarithms(
    z: NDArray[np.float64],
    *,
    output_path: Path,
    show: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for curve in CURVES:
        values = polylog(curve.order, z)
        ax.plot(
            z,
            values,
            linestyle=curve.linestyle,
            color=MATLAB_COLORS[curve.color_index - 1],
            label=curve.label,
        )

    ax.axhline(0.0, color="k", linewidth=1.5)
    ax.axvline(0.0, color="k", linewidth=1.5)
    ax.set_xlim(-2.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(r"PolyLogarithms of $z$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mathrm{Li}_n(z)$")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "PolyLogPlot.png",
        help="Output PNG path (default: repo root / PolyLogPlot.png).",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=-2.0,
        help="Minimum z value (MATLAB default: -2).",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=0.99,
        help="Maximum z value (MATLAB default: 0.99).",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.01,
        help="Grid spacing in z (MATLAB default: 0.01).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    z = _z_grid(args.z_min, args.z_max, args.dz)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_path = plot_polylogarithms(z, output_path=args.output, show=args.show)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
