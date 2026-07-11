"""Shared fixtures and reference implementations for tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from mpmath import fp
from numpy.typing import ArrayLike, NDArray


def _mpmath_polylog_real(n: float, z: float) -> float:
    value = fp.polylog(n, z)
    if hasattr(value, "real"):
        return float(value.real)
    return float(value)


@pytest.fixture(scope="session")
def mpmath_polylog() -> np.ufunc:
    return np.frompyfunc(_mpmath_polylog_real, 2, 1)


@pytest.fixture(scope="session")
def mpmath_reference(
    mpmath_polylog: np.ufunc,
) -> Callable[[float, ArrayLike | float], NDArray[np.float64] | float]:
    def _reference(n: float, z: ArrayLike | float) -> NDArray[np.float64] | float:
        if np.isscalar(z):
            return float(mpmath_polylog(n, float(z)))

        z_array = np.asarray(z, dtype=np.float64)
        reference = mpmath_polylog(n, z_array).astype(np.float64)
        return reference.reshape(z_array.shape)

    return _reference
