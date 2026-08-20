# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Unified C++ polylog kernel versus mpmath."""

from __future__ import annotations

import numpy as np
import pytest

from ideal_gases import polylog

FUKUSHIMA_ORDERS = (-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)

# (n, z, rtol, atol). Tight tolerance where Fukushima or integer analytic
# branches apply; looser for Bhagat fallback fractionals.
_MPMATH_CASES: list[tuple[float, float, float, float]] = [
    (0.0, 0.5, 1e-9, 1e-12),
    (1.0, 0.5, 1e-9, 1e-12),
    (2.0, 0.5, 1e-9, 1e-12),
    (2.0, 2.0, 1e-9, 1e-12),
    (2.0, -0.3, 1e-9, 1e-12),
    (2.0, -10.0, 1e-9, 1e-12),
    (3.0, -10.0, 1e-9, 1e-12),
    (4.0, -100.0, 1e-9, 1e-12),
    (-1.0, 0.5, 1e-9, 1e-12),
    (-2.0, 0.5, 1e-9, 1e-12),
    (-2.0, 10.0, 1e-9, 1e-12),
    (-3.0, -0.3, 1e-9, 1e-12),
    (-3.0, 2.0, 1e-9, 1e-12),
    (-1.5, -0.5, 1e-6, 1e-3),
    (5.5, 0.3, 1e-6, 1e-3),
]
for _n in FUKUSHIMA_ORDERS:
    for _z in (-0.05, -0.5, -50.0, -901.284):
        _MPMATH_CASES.append((_n, _z, 1e-9, 1e-12))
    for _z in (1e-4, 0.2, 0.55, 0.99, 0.999999):
        _MPMATH_CASES.append((_n, _z, 1e-9, 1e-12))


@pytest.mark.parametrize(("n", "z", "rtol", "atol"), _MPMATH_CASES)
def test_polylog_matches_mpmath(
    n: float, z: float, rtol: float, atol: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(n, z) == pytest.approx(expected, rel=rtol, abs=atol)


@pytest.mark.parametrize(("m", "z"), [(2, 10.0), (3, 2.0)])
def test_polylog_negative_integer_inversion_identity(m: int, z: float) -> None:
    n = -m
    result = polylog(float(n), z) + ((-1) ** m) * polylog(float(n), 1.0 / z)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_polylog_integer_smooth_across_former_branch_cut() -> None:
    for n in (2, 3):
        values = [polylog(float(n), z) for z in (-0.751, -0.749)]
        assert abs(values[1] - values[0]) < 0.002


def test_polylog_array_matches_scalar_loop() -> None:
    n = 2.5
    z = np.linspace(0.1, 0.9, 25)
    vectorized = polylog(n, z)
    expected = np.array([polylog(n, value) for value in z], dtype=np.float64)
    np.testing.assert_allclose(vectorized, expected, rtol=0.0, atol=0.0)


def test_polylog_preserves_input_shape() -> None:
    z = np.arange(12, dtype=np.float64).reshape(3, 4) / 20.0
    result = polylog(3.0, z)
    assert result.shape == z.shape
