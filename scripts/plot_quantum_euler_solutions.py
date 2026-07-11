#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Plot quantum Euler Riemann solutions for visual verification.

Ports the workflow of ``matlab/PlotQEuler.m`` using :func:`euler.quantum_euler_gas`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from euler.riemann import RiemannResult, quantum_euler_gas

ROOT = Path(__file__).resolve().parents[1]

COLORS = {"BE": "r", "MB": "b", "FD": "m"}
STATISTICS = ("FD", "MB", "BE")


@dataclass(frozen=True)
class ExampleConfig:
    name: str
    rho_l: float
    u_l: float
    t_l: float
    rho_r: float
    u_r: float
    t_r: float
    t_end: float
    n: float
    h: float
    h_fd: float | None = None


EXAMPLES: dict[int, ExampleConfig] = {
    1: ExampleConfig(
        name="Sod shock tube (classical regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=1.0,
    ),
    2: ExampleConfig(
        name="Sod shock tube (quantum h = 2)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=2.0,
    ),
    3: ExampleConfig(
        name="Sod shock tube (degenerate FD, h_FD = 6)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=2.65,
        h_fd=6.0,
    ),
    4: ExampleConfig(
        name="Hu and Jin (classical regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,
        h=1.0,
    ),
    5: ExampleConfig(
        name="Hu and Jin (degenerate regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,
        h=3.3,
        h_fd=6.0,
    ),
    6: ExampleConfig(
        name="Yang and Shi (degenerate regime)",
        rho_l=3.086455,
        u_l=0.0,
        t_l=8.053324,
        rho_r=3.084272,
        u_r=0.0,
        t_r=8.067390,
        t_end=0.10,
        n=3.0,
        h=6.0,
    ),
    7: ExampleConfig(
        name="Filbet and Jing (2010), classical",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,
        h=0.1,
    ),
    8: ExampleConfig(
        name="Filbet and Jing (2010), quantum",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,
        h=3.0,
    ),
}


def _configure_matplotlib() -> None:
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


def _solve_example(
    config: ExampleConfig,
    *,
    dx: float,
    x0: float,
) -> tuple[NDArray[np.float64], dict[str, RiemannResult]]:
    x = np.arange(0.0, 1.0 + 0.5 * dx, dx, dtype=np.float64)
    solutions: dict[str, RiemannResult] = {}

    for statistic in STATISTICS:
        h = config.h_fd if statistic == "FD" and config.h_fd is not None else config.h
        solutions[statistic] = quantum_euler_gas(
            rho_l=config.rho_l,
            u_l=config.u_l,
            t_l=config.t_l,
            rho_r=config.rho_r,
            u_r=config.u_r,
            t_r=config.t_r,
            t_end=config.t_end,
            n=config.n,
            h=h,
            statistic=statistic,
            x=x,
            x0=x0,
            dx=dx,
        )

    return x, solutions


def _plot_individual_panels(
    x: np.ndarray,
    solutions: dict[str, RiemannResult],
    *,
    example_id: int,
    output_dir: Path,
    show: bool,
) -> Path:
    fields = [
        ("rho", r"$\rho$"),
        ("ux", r"$u_x$"),
        ("p", r"$p$"),
        ("e", r"$e$"),
        ("t", r"$\theta$"),
        ("z", r"$z$"),
    ]

    fig, axes = plt.subplots(3, 6, figsize=(14, 6), constrained_layout=True)

    for row, statistic in enumerate(STATISTICS):
        result = solutions[statistic]
        color = COLORS[statistic]
        for col, (attr, label) in enumerate(fields):
            ax = axes[row, col]
            y = getattr(result, attr)
            ax.plot(x, y, color=color)
            if row == 0:
                ax.set_title(f"{label}-{statistic}")
            if row == 2:
                ax.set_xlabel("$x$")
            ax.autoscale(enable=True, axis="both", tight=True)

    fig.suptitle(f"Quantum Euler example {example_id}", fontsize=11)
    output_path = output_dir / f"QEuler_Eg{example_id}_AllPlots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_comparison_panels(
    x: np.ndarray,
    solutions: dict[str, RiemannResult],
    *,
    example_id: int,
    output_dir: Path,
    show: bool,
) -> Path:
    panels = [
        ("rho", r"$\rho$"),
        ("ux", r"$u_x$"),
        ("e", r"$e$"),
        ("p", r"$p$"),
        ("z", r"$z$"),
        ("t", r"$\theta$"),
    ]
    subplot_positions = [1, 2, 3, 4, 5, 6]

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (attr, label), position in zip(axes_flat, panels, subplot_positions, strict=True):
        for statistic in ("BE", "MB", "FD"):
            result = solutions[statistic]
            ax.plot(x, getattr(result, attr), color=COLORS[statistic], label=statistic)
        ax.set_ylabel(label)
        ax.set_xlabel("$x$")
        ax.legend(loc="best")

    fig.suptitle(f"Quantum Euler comparison example {example_id}", fontsize=11)
    output_path = output_dir / f"QEuler_Eg{example_id}_TogetherPlots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        type=int,
        default=7,
        choices=sorted(EXAMPLES),
        help="Example case number (default: 7, Filbet and Jing classical).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for saved PNG figures.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.002,
        help="Spatial grid spacing (MATLAB default: 0.002).",
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=0.5,
        help="Initial discontinuity location.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    config = EXAMPLES[args.example]
    print(f"Running example {args.example}: {config.name}")

    x, solutions = _solve_example(config, dx=args.dx, x0=args.x0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_plots = _plot_individual_panels(
        x,
        solutions,
        example_id=args.example,
        output_dir=args.output_dir,
        show=args.show,
    )
    together_plots = _plot_comparison_panels(
        x,
        solutions,
        example_id=args.example,
        output_dir=args.output_dir,
        show=args.show,
    )

    print(f"Saved {all_plots}")
    print(f"Saved {together_plots}")


if __name__ == "__main__":
    main()
