import numpy as np
import pytest

from euler import polylog


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


@pytest.mark.parametrize(("n", "z"), INTEGER_SCALAR_CASES)
def test_polylog_integer_scalar_matches_mpmath(
    n: int, z: float, mpmath_reference
) -> None:
    expected = mpmath_reference(n, z)
    assert polylog(float(n), z) == pytest.approx(expected, rel=1e-9, abs=1e-9)


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
