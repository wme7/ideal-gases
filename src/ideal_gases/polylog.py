# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Fractional PolyLogarithm function: PolyLog(n, z).

This module provides a vectorized implementation of the fractional
PolyLogarithm function: ``PolyLog(n, z)``.

The C++ kernel uses Fukushima minimax Fermi-Dirac / Bose-Einstein
integrals for orders ``n`` in ``{-1/2, 1/2, 1, 3/2, ..., 9/2}`` on
``z < 0`` and ``0 < z < 1``. Other arguments and orders use the
Bhagat/Kuhnert approximations from ``matlab/PolyLog.m`` and integer
analytic branches.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ideal_gases._polylog import polylog as _polylog_scalar
from ideal_gases._polylog import polylog_1d as _polylog_array_1d

__all__ = ["polylog"]


@overload
def polylog(n: float, z: float) -> float: ...


@overload
def polylog(n: float, z: ArrayLike) -> NDArray[np.float64]: ...


def polylog(n: float, z: ArrayLike | float) -> NDArray[np.float64] | float:
    """Evaluate PolyLog(n, z) using the fast C++ implementation.

    Parameters
    ----------
    n:
        Polylogarithm order. Fukushima minimax integrals are used for
        ``n`` in ``{-1/2, 1/2, 1, 3/2, ..., 9/2}`` when ``z < 0`` or
        ``0 < z < 1``. Integer orders (including negatives) and other
        arguments use analytic branches or the Bhagat/Kuhnert
        approximations from ``matlab/PolyLog.m``.
    z:
        Scalar or array-like argument.

    Returns
    -------
    float or ndarray
        Real polylogarithm ``Li_n(z)``.
    """
    if np.isscalar(z):
        z_scalar = float(np.asarray(z, dtype=np.float64))
        return float(_polylog_scalar(n, z_scalar))

    z_array = np.asarray(z, dtype=np.float64)
    flat = np.ascontiguousarray(z_array.ravel())
    result = _polylog_array_1d(n, flat)
    return result.reshape(z_array.shape)
