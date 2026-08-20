# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Tests for :mod:`ideal_gases.equilibrium`."""

from __future__ import annotations

import numpy as np
import pytest

from ideal_gases.equilibrium import (
    G,
    Z_BOSE_MAX,
    bose_ceiling,
    bose_density_parameter,
    equilibrium_moments,
    find_fugacity,
    find_moments,
    is_bose_ceiling_z,
    is_bose_condensed,
)


# -- Classical closed-form --------------------------------------------------


def test_classical_closed_form() -> None:
    dim = 3
    rho, e = 1.0, 1.5
    z, T, p = find_moments(rho, e, dim=dim, h=1.0, eta=0)
    T_cl = 2.0 * e / dim
    z_cl = rho / (2.0 * np.pi * T_cl) ** (dim / 2)
    assert T == pytest.approx(T_cl, rel=1e-12)
    assert z == pytest.approx(z_cl, rel=1e-12)
    assert p == pytest.approx((2.0 / dim) * rho * e, rel=1e-12)


# -- Round-trip (z, T) → (ρ, e) → (z, T) -----------------------------------

ROUND_TRIP_CASES = [
    (0, 0.05, 1.0, 1.0, 1e-6),
    (1, 0.0621, 1.0, 1.0, 1e-6),
    (1, 0.9906, 1.0, 3.3, 5e-5),
    (-1, 0.0649, 1.0, 1.0, 1e-6),
    (-1, 901.2840, 1.0, 6.0, 1e-6),
]


@pytest.mark.parametrize("eta, z_true, T_true, h, rtol", ROUND_TRIP_CASES)
def test_round_trip_find_moments(eta, z_true, T_true, h, rtol) -> None:
    dim = 3
    rho_f, e_f = equilibrium_moments(z_true, T_true, dim=dim, h=h, eta=eta)
    z_b, T_b, _ = find_moments(rho_f, e_f, dim=dim, h=h, eta=eta)
    assert z_b == pytest.approx(z_true, rel=rtol, abs=1e-8)
    assert T_b == pytest.approx(T_true, rel=rtol, abs=1e-8)


# -- Round-trip via find_fugacity --------------------------------------------


@pytest.mark.parametrize("eta, z_true, T_true, h, rtol", ROUND_TRIP_CASES)
def test_round_trip_find_fugacity(eta, z_true, T_true, h, rtol) -> None:
    dim = 3
    rho_f, _ = equilibrium_moments(z_true, T_true, dim=dim, h=h, eta=eta)
    z_b = find_fugacity(rho_f, T_true, dim=dim, h=h, eta=eta)
    assert z_b == pytest.approx(z_true, rel=rtol, abs=1e-8)


# -- Hu & Jin §4 published left states --------------------------------------


@pytest.mark.parametrize(
    "eta, h, z_pub",
    [
        (1, 1.0, 0.0621),
        (-1, 1.0, 0.0649),
        (1, 3.3, 0.9906),
        (-1, 6.0, 901.2840),
    ],
)
def test_hu_jin_published_states(eta, h, z_pub) -> None:
    dim = 3
    rho_f, e_f = equilibrium_moments(z_pub, 1.0, dim=dim, h=h, eta=eta)
    z_b, T_b, _ = find_moments(rho_f, e_f, dim=dim, h=h, eta=eta)
    assert z_b == pytest.approx(z_pub, rel=2e-3, abs=1e-4)
    assert T_b == pytest.approx(1.0, rel=2e-3, abs=1e-4)


# -- Broadcasting -----------------------------------------------------------


def test_broadcasting() -> None:
    z_v, T_v, p_v = find_moments(np.array([1.0, 0.4]), np.array([1.5, 0.9]), eta=0)
    assert z_v.shape == (2,)
    assert T_v.shape == (2,)
    assert p_v.shape == (2,)


# -- Bose condensation ceiling ----------------------------------------------


def test_hu_jin_bose_not_condensed_at_h_3_3() -> None:
    assert is_bose_condensed(1.0, 1.0, dim=3, h=3.3) is False


def test_bose_condensed_at_h_6() -> None:
    assert is_bose_condensed(1.0, 1.0, dim=3, h=6.0) is True


def test_is_bose_ceiling_z() -> None:
    assert is_bose_ceiling_z(Z_BOSE_MAX) is True
    assert is_bose_ceiling_z(0.99) is False
    mask = is_bose_ceiling_z(np.array([0.99, Z_BOSE_MAX]))
    np.testing.assert_array_equal(mask, np.array([False, True]))


def test_bose_density_parameter_matches_newton_data() -> None:
    rho, T, dim, h = 1.0, 1.0, 3.0, 3.3
    data = bose_density_parameter(rho, T, dim, h)
    expected = rho * h**dim / (2.0 * np.pi * T) ** (dim / 2.0)
    assert data == pytest.approx(expected)
    assert data < bose_ceiling(dim)


# -- G_n versus mpmath.polylog -----------------------------------------------


@pytest.mark.parametrize("n", [1.5, 2.5, 3.5])
@pytest.mark.parametrize("eta", [-1, 1])
def test_G_matches_mpmath_polylog(n, eta, mpmath_reference) -> None:
    z_pts = np.array([0.05, 0.5, 0.99])
    g_ig = np.asarray(G(n, z_pts, eta=eta), dtype=float)
    g_mp = eta * np.asarray(mpmath_reference(n, eta * z_pts), dtype=float)
    np.testing.assert_allclose(g_ig, g_mp, rtol=1e-5, atol=1e-8)
