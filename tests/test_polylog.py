# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

import numpy as np
import pytest

from ideal_gases import polylog


INTEGER_SCALAR_CASES = [
    (0, 0.5),
    (1, 0.5),
    (2, 0.2),
    (2, 0.5),
    (2, 0.9),
    (2, 2.0),
    (2, -0.3),
    (2, -10.0),
    (3, -10.0),
    (4, -100.0),
]

NEGATIVE_INTEGER_SCALAR_CASES = [
    (-1, 0.5),
    (-2, 0.5),
    (-2, 0.9),
    (-2, 10.0),
    (-3, -0.3),
    (-3, 2.0),
]

FRACTIONAL_SCALAR_CASES = [
    (1.5, 0.2),
    (1.5, 0.5),
    (1.5, 0.55),
    (1.5, 0.7),
    (1.5, 0.9),
    (1.5, 0.95),
    (2.5, 0.2),
    (2.5, 0.3),
    (2.5, 0.5),
    (2.5, 0.55),
    (2.5, 0.7),
    (2.5, 0.9),
    (2.5, 0.95),
    (3.5, 0.2),
    (3.5, 0.5),
    (3.5, 0.55),
    (3.5, 0.7),
    (3.5, 0.95),
]


@pytest.mark.parametrize(("n", "z"), [(2, -0.76), (2, -0.85), (3, -0.76), (3, -0.9)])
def test_polylog_integer_negative_z_uses_convergent_series(
    n: int, z: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(float(n), z) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_polylog_integer_smooth_across_former_branch_cut() -> None:
    for n in (2, 3):
        values = [polylog(float(n), z) for z in (-0.751, -0.749)]
        assert abs(values[1] - values[0]) < 0.002


@pytest.mark.parametrize(("n", "z"), INTEGER_SCALAR_CASES)
def test_polylog_integer_scalar_matches_mpmath(
    n: int, z: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(float(n), z) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("n", "z"), NEGATIVE_INTEGER_SCALAR_CASES)
def test_polylog_negative_integer_scalar_matches_mpmath(
    n: int, z: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(float(n), z) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("m", "z"), [(2, 10.0), (3, 2.0)])
def test_polylog_negative_integer_inversion_identity(m: int, z: float) -> None:
    n = -m
    result = polylog(float(n), z) + ((-1) ** m) * polylog(float(n), 1.0 / z)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_polylog_negative_integer_array_matches_mpmath(mpmath_reference) -> None:
    n = -2.0
    z = np.linspace(0.2, 0.9, 25)
    result = polylog(n, z)
    expected = mpmath_reference(n, z)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-6)


def test_polylog_negative_integer_array_outside_unit_disk(mpmath_reference) -> None:
    n = -2.0
    z = np.array([2.0, 5.0, 10.0])
    result = polylog(n, z)
    expected = mpmath_reference(n, z)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-9)


FRACTIONAL_NEGATIVE_Z_CASES = [
    (-0.5, -0.1),
    (-0.5, -0.3),
    (-0.5, -0.5),
    (-1.5, -0.1),
    (-1.5, -0.3),
    (-1.5, -0.9),
    (-2.5, -0.3),
    (-2.5, -0.5),
    (-3.5, -0.5),
    (-3.5, -0.9),
]


def _matlab_range3_reference(n: float, z: float) -> float:
    """Replicate ``matlab/PolyLog.m`` Range 3 (FD rational approximation)."""

    def power(base: float, exponent: float) -> float:
        if base == 0.0:
            return 0.0
        return float(np.exp(exponent * np.log(base)))

    def fd(nu: float, zz: float, terms: int) -> float:
        total = 0.0
        for l in range(1, terms + 1):
            total += ((-1) ** (l - 1)) * (zz**l) / (l**nu)
        return total

    coeffs = (6435, 27456, 48048, 44352, 23100, 6720, 1008, 64)
    bases = (9, 8, 7, 6, 5, 4, 3, 2)
    z3 = abs(z)
    numerator = 0.0
    denominator = 0.0
    z_power = 1.0
    for i, (coeff, base) in enumerate(zip(coeffs, bases, strict=True)):
        base_power = power(float(base), n)
        numerator += coeff * base_power * z_power * fd(n, z3, 8 - i)
        denominator += coeff * base_power * z_power
        z_power *= z3
    denominator += z3**8
    return -numerator / denominator


@pytest.mark.parametrize(("n", "z"), FRACTIONAL_NEGATIVE_Z_CASES)
def test_polylog_fractional_negative_z_matches_matlab_range3(n: float, z: float) -> None:
    expected = _matlab_range3_reference(n, z)
    assert polylog(n, z) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("n", "z"), FRACTIONAL_SCALAR_CASES)
def test_polylog_fractional_scalar_matches_mpmath(
    n: float, z: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(n, z) == pytest.approx(expected, rel=1e-6, abs=1e-3)


def test_mpmath_reference_uses_frompyfunc(mpmath_polylog, mpmath_reference) -> None:
    n = 2.5
    z = np.linspace(0.2, 0.9, 7)
    vectorized = mpmath_polylog(n, z).astype(np.float64)
    expected = np.array([mpmath_reference(n, value) for value in z], dtype=np.float64)
    np.testing.assert_allclose(vectorized, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("n", "rtol", "atol"),
    [
        (2.0, 1e-8, 1e-7),
        (2.5, 1e-6, 1e-3),
        (3.5, 1e-6, 1e-3),
    ],
)
def test_polylog_array_matches_mpmath_reference(
    n: float, rtol: float, atol: float, mpmath_reference
) -> None:
    z = np.linspace(0.15, 0.95, 33)
    result = polylog(n, z)
    expected = mpmath_reference(n, z)
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)


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
