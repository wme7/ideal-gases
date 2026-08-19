# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""In-tree C++ Fukushima polylog backend. Always runs (no gfortran)."""

from __future__ import annotations

import numpy as np
import pytest

from ideal_gases.equilibrium import G, set_polylog_backend
from ideal_gases.polylog import polylog_fukushima

ORDERS = (-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
Z_FD = np.array([0.05, 0.5, 0.99, 50.0, 901.284])
Z_BE = np.array([1e-4, 0.2, 0.55, 0.9, 0.99, 0.999999])


@pytest.fixture
def _restore_backend():
    from ideal_gases.equilibrium import POLYLOG_BACKEND

    saved = POLYLOG_BACKEND
    yield
    set_polylog_backend(saved)


def test_fukushima_matches_mpmath_on_fermi_arguments(mpmath_reference) -> None:
    for n in ORDERS:
        w = -Z_FD
        got = np.asarray(polylog_fukushima(float(n), w), dtype=float)
        expected = np.asarray(mpmath_reference(float(n), w), dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_fukushima_matches_mpmath_on_bose_arguments(mpmath_reference) -> None:
    for n in ORDERS:
        got = np.asarray(polylog_fukushima(float(n), Z_BE), dtype=float)
        expected = np.asarray(mpmath_reference(float(n), Z_BE), dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_fukushima_backend_g_fd(_restore_backend, mpmath_reference) -> None:
    z = np.array([0.05, 1.0, 50.0, 901.284])
    set_polylog_backend("fukushima")
    for n in (0.5, 1.5, 2.5):
        got = np.asarray(G(n, z, eta=-1), dtype=float)
        expected = -np.asarray(mpmath_reference(float(n), -z), dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_fukushima_backend_g_be(_restore_backend, mpmath_reference) -> None:
    z = np.array([0.05, 0.55, 0.99])
    set_polylog_backend("fukushima")
    for n in (0.5, 1.5, 2.5):
        got = np.asarray(G(n, z, eta=1), dtype=float)
        expected = np.asarray(mpmath_reference(float(n), z), dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_fukushima_li0_fallback_matches_closed_form() -> None:
    assert polylog_fukushima(0.0, 0.5) == pytest.approx(1.0, rel=1e-14)
