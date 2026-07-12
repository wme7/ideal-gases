# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Spatial grid helpers for CLI solvers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ideal_gases.riemann import DEFAULT_DX, DEFAULT_X0


def build_grid(
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    x0: float = DEFAULT_X0,
    dx: float = DEFAULT_DX,
    nx: int | None = None,
) -> NDArray[np.float64]:
    """Return a uniform spatial grid on ``[x_min, x_max]``."""
    if x_max <= x_min:
        msg = f"x_max ({x_max}) must be greater than x_min ({x_min})."
        raise ValueError(msg)
    if nx is not None:
        if nx < 2:
            msg = f"nx must be at least 2 (got {nx})."
            raise ValueError(msg)
        return np.linspace(x_min, x_max, nx, dtype=np.float64)

    if dx <= 0.0:
        msg = f"dx must be positive (got {dx})."
        raise ValueError(msg)

    count = int(np.floor((x_max - x_min) / dx)) + 1
    x = x_min + dx * np.arange(count, dtype=np.float64)
    if x[-1] < x_max:
        x = np.append(x, x_max)
    return x


def validate_x0(*, x0: float, x_min: float, x_max: float) -> None:
    if not x_min <= x0 <= x_max:
        msg = f"x0 ({x0}) must lie in [x_min, x_max] = [{x_min}, {x_max}]."
        raise ValueError(msg)
