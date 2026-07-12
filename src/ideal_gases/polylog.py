# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Fractional PolyLogarithm function: PolyLog(n, z).

This module provides a vectorized implementation of the fractional 
PolyLogarithm function: ``PolyLog(n, z)``.

The implementation is based on the Bhagat/Kuhnert approximations from 
``matlab/PolyLog.m`` (Diaz, 2014).
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
        Polylogarithm order. Integer values (including negative integers) use the
        analytic branch; non-integer values use the Bhagat/Kuhnert
        approximations from ``matlab/PolyLog.m``.
    z:
        Scalar or array-like argument.

    Returns
    -------
    float or ndarray
        ``real(polylog(n, z))`` for integer ``n`` (positive or negative),
        otherwise the fractional approximation from the MATLAB reference
        implementation.
    """
    if np.isscalar(z):
        return float(_polylog_scalar(n, float(z)))

    z_array = np.asarray(z, dtype=np.float64)
    flat = np.ascontiguousarray(z_array.ravel())
    result = _polylog_array_1d(n, flat)
    return result.reshape(z_array.shape)