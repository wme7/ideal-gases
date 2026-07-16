# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Shared matplotlib widget helpers for interactive Riemann explorers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

SAVE_DPI = 240


def ylim_from_lines(*lines: Line2D) -> tuple[float, float] | None:
    ymin = np.inf
    ymax = -np.inf
    for line in lines:
        if not line.get_visible():
            continue
        y = np.asarray(line.get_ydata(), dtype=np.float64)
        ymin = min(ymin, float(np.min(y)))
        ymax = max(ymax, float(np.max(y)))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None
    if ymax <= ymin:
        margin = max(abs(ymax), 1.0) * 0.05
        return ymax - margin, ymax + margin
    margin = 0.05 * (ymax - ymin)
    return ymin - margin, ymax + margin


def autoscale_axes(
    fig: Figure,
    axis_lines: list[tuple[Axes, tuple[Line2D, ...]]],
) -> None:
    for axis, lines in axis_lines:
        limits = ylim_from_lines(*lines)
        if limits is not None:
            axis.set_ylim(limits)
    fig.canvas.draw_idle()


def piecewise_ic(
    x: np.ndarray,
    x0: float,
    left_value: float,
    right_value: float,
) -> np.ndarray:
    return left_value * (x <= x0) + right_value * (x > x0)
