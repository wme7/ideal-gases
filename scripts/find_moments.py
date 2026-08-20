#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Smoke-test driver for :mod:`ideal_gases.equilibrium`.

Runs round-trip inversions for all three statistics and compares ``G``
against ``mpmath.polylog``.
"""

from __future__ import annotations

import numpy as np

from ideal_gases.equilibrium import (
    G,
    equilibrium_moments,
    find_fugacity,
    find_moments,
)


def _assert_close(name: str, got, expected, rtol: float, atol: float = 0.0) -> None:
    if not np.allclose(got, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: got {got}, expected {expected}")
    print(f"  {name}: {got}  (ok)")


def _smoke() -> None:
    dim = 3
    print("Classical closed form")
    rho, e = 1.0, 1.5
    z, T, p = find_moments(rho, e, dim=dim, h=1.0, eta=0)
    T_cl = 2.0 * e / dim
    z_cl = rho / (2.0 * np.pi * T_cl) ** (dim / 2)
    _assert_close("T", T, T_cl, rtol=1e-12)
    _assert_close("z", z, z_cl, rtol=1e-12)
    _assert_close("p", p, (2.0 / dim) * rho * e, rtol=1e-12)

    print("Round-trip (z, T) → (ρ, e) → (z, T)")
    for eta, z_true, T_true, h in (
        (0, 0.05, 1.0, 1.0),
        (1, 0.0621, 1.0, 1.0),
        (1, 0.9906, 1.0, 3.3),
        (-1, 0.0649, 1.0, 1.0),
        (-1, 901.2840, 1.0, 6.0),
    ):
        rho_f, e_f = equilibrium_moments(z_true, T_true, dim=dim, h=h, eta=eta)
        z_b, T_b, _ = find_moments(rho_f, e_f, dim=dim, h=h, eta=eta)
        rtol = 5e-5 if eta == 1 and z_true > 0.9 else 1e-6
        _assert_close(f"eta={eta} h={h} z", z_b, z_true, rtol=rtol, atol=1e-8)
        _assert_close(f"eta={eta} h={h} T", T_b, T_true, rtol=rtol, atol=1e-8)

    print("Round-trip (z, T) → ρ → z via find_fugacity")
    for eta, z_true, T_true, h in (
        (0, 0.05, 1.0, 1.0),
        (1, 0.0621, 1.0, 1.0),
        (1, 0.9906, 1.0, 3.3),
        (-1, 0.0649, 1.0, 1.0),
        (-1, 901.2840, 1.0, 6.0),
    ):
        rho_f, _e_f = equilibrium_moments(z_true, T_true, dim=dim, h=h, eta=eta)
        z_b = find_fugacity(rho_f, T_true, dim=dim, h=h, eta=eta)
        rtol = 5e-5 if eta == 1 and z_true > 0.9 else 1e-6
        _assert_close(f"fugacity eta={eta} h={h} z", z_b, z_true, rtol=rtol, atol=1e-8)

    print("Hu & Jin §4 left states (ρ, T) = (1, 1)")
    for eta, h, z_pub in (
        (1, 1.0, 0.0621),
        (-1, 1.0, 0.0649),
        (1, 3.3, 0.9906),
        (-1, 6.0, 901.2840),
    ):
        rho_f, e_f = equilibrium_moments(z_pub, 1.0, dim=dim, h=h, eta=eta)
        z_b, T_b, _ = find_moments(rho_f, e_f, dim=dim, h=h, eta=eta)
        _assert_close(
            f"published z (eta={eta}, h={h})", z_b, z_pub, rtol=2e-3, atol=1e-4
        )
        _assert_close(f"published T (eta={eta}, h={h})", T_b, 1.0, rtol=2e-3, atol=1e-4)

    print("Broadcasting")
    z_v, T_v, p_v = find_moments(np.array([1.0, 0.4]), np.array([1.5, 0.9]), eta=0)
    assert z_v.shape == (2,) and T_v.shape == (2,) and p_v.shape == (2,)
    print("  array shape (2,): ok")

    print("\nAll find_moments smoke checks passed.")


def _g_mpmath(n: float, z, eta: int) -> np.ndarray:
    import mpmath as mp

    z_arr = np.asarray(z, dtype=float)

    def _one(zi: float) -> float:
        return eta * float(mp.re(mp.polylog(n, eta * zi)))

    return np.vectorize(_one, otypes=[float])(z_arr)


def _compare_mpmath() -> None:
    """G_n samples: C++ kernel vs mpmath.polylog."""
    z_pts = np.array([0.05, 0.5, 0.99])
    g_ig = {n: np.asarray(G(n, z_pts, eta=-1), dtype=float) for n in (1.5, 2.5, 3.5)}
    g_mp = {n: _g_mpmath(n, z_pts, eta=-1) for n in (1.5, 2.5, 3.5)}
    print("G_n versus mpmath.polylog (Fermi, z = 0.05, 0.5, 0.99)")
    for n in (1.5, 2.5, 3.5):
        diff = np.max(np.abs(g_ig[n] - g_mp[n]))
        print(f"  max |G_{n}(ideal_gases) - G_{n}(mpmath)| = {diff:.3e}")
        if not np.allclose(g_ig[n], g_mp[n], rtol=1e-5, atol=1e-8):
            raise AssertionError(f"G_{n} disagrees with mpmath: {g_ig[n]} vs {g_mp[n]}")
    print("  G agrees with mpmath (ok)")


if __name__ == "__main__":
    _smoke()
    print()
    _compare_mpmath()
