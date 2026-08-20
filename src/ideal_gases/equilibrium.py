# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Invert equilibrium density and internal energy into fugacity and temperature.

Given ρ and e, recover z and T for: Fermi (η = -1) and Bose (η = +1) statistics.

Planck's constant h may be treated as a free parameter that sets the degeneracy.

Solvers use the in-tree C++ ``polylog`` kernel (Fukushima minimax
integrals with Bhagat/integer fallback).
"""

from __future__ import annotations

import numpy as np

from ideal_gases.polylog import polylog

__all__ = [
    "G",
    "bose_ceiling",
    "bose_density_parameter",
    "equilibrium_moments",
    "find_fugacity",
    "find_moments",
    "is_bose_ceiling_z",
    "is_bose_condensed",
]

VALID_ETA = frozenset({-1, 0, 1})
Z_INIT = 1.0e-3
Z_MIN = 1.0e-15
Z_BOSE_MAX = 0.999_999
NEWTON_TOL = 1.0e-6
NEWTON_MAXITER = 200


def _polylog(n, z):
    """Real polylogarithm Li_n(z) via the C++ kernel."""
    return polylog(float(n), z)


def _validate_eta(eta) -> int:
    eta_i = int(eta)
    if eta_i not in VALID_ETA:
        raise ValueError(f"eta = {eta} invalid; expected eta in {{-1, 0, +1}}")
    return eta_i


def G(n, z, eta: int = 1):
    """Bose / Fermi / classical function G_n(z).

    G_n(z) = z                 (η = 0),
    G_n(z) = η * Li_n(η z)     (η = ±1).
    """
    eta = _validate_eta(eta)
    z = np.asarray(z, dtype=float)
    if eta == 0:
        return z if z.ndim else float(z)
    value = _polylog(n, eta * z) * eta
    return value


def bose_ceiling(dim: float) -> float:
    """Excited-gas Bose ceiling G_{D/2}(z_max) used by Newton clipping."""
    return float(G(dim / 2.0, Z_BOSE_MAX, eta=1))


def bose_density_parameter(
    rho, T, dim: float, h: float, m: float = 1.0, k_B: float = 1.0
):
    """Excited-gas density parameter data = ρ h^D / (m (2π m k_B T)^{D/2})."""
    rho_arr, t_arr = np.broadcast_arrays(
        np.asarray(rho, dtype=float),
        np.asarray(T, dtype=float),
    )
    data = rho_arr * h**dim / (m * (2.0 * np.pi * m * k_B * t_arr) ** (dim / 2.0))
    if rho_arr.ndim == 0:
        return float(data)
    return data


def is_bose_condensed(rho, T, dim: float, h: float, m: float = 1.0, k_B: float = 1.0):
    """True where data exceeds G_{D/2}(z_max) (independent T, not reconstructed)."""
    data = bose_density_parameter(rho, T, dim, h, m=m, k_B=k_B)
    ceiling = bose_ceiling(dim)
    if np.isscalar(data):
        return bool(float(data) > ceiling)
    return np.asarray(data, dtype=float) > ceiling


def is_bose_ceiling_z(z):
    """True where fugacity is at the Newton Bose clip Z_BOSE_MAX."""
    z_arr = np.asarray(z, dtype=float)
    mask = z_arr >= Z_BOSE_MAX
    if z_arr.ndim == 0:
        return bool(mask)
    return mask


def equilibrium_moments(
    z, T, dim: int = 3, h: float = 1.0, eta: int = 0, m: float = 1.0, k_B: float = 1.0
):
    """Forward map (z, T) → (ρ, e) in the same units as ``find_moments``."""
    eta = _validate_eta(eta)
    g0 = G(dim / 2, z, eta)
    g1 = G(dim / 2 + 1, z, eta)
    rho = m * (2.0 * np.pi * m * k_B * T / h**2) ** (dim / 2) * g0
    energy = 0.5 * dim * (k_B * T / m) * (g1 / g0)
    return rho, energy


def _ratio_and_derivative(z: float, dim: int, eta: int) -> tuple[float, float]:
    """Return (G_{D/2}^{γ} / G_{D/2+1}, d/dz of that ratio), γ = (D+2)/D."""
    gamma = (dim + 2) / dim
    g_m = float(G(dim / 2 - 1, z, eta))
    g_0 = float(G(dim / 2, z, eta))
    g_p = float(G(dim / 2 + 1, z, eta))
    ratio = g_0**gamma / g_p
    d_ratio = gamma * (g_0 ** (gamma - 1) * g_m / (z * g_p)) - g_0 ** (
        2.0 * (1 + dim) / dim
    ) / (z * g_p**2)
    return ratio, d_ratio


def _clip_z(z: float, eta: int) -> float:
    z = max(z, Z_MIN)
    if eta == 1:
        z = min(z, Z_BOSE_MAX)
    return z


def _newton_z(rho: float, e: float, dim: int, h: float, eta: int, m: float) -> float:
    data = (rho / m) ** (2.0 / dim) * dim * h**2 / (4.0 * np.pi * m**2 * e)
    z = Z_INIT
    for _ in range(NEWTON_MAXITER):
        ratio, d_ratio = _ratio_and_derivative(z, dim, eta)
        psi = ratio - data
        if d_ratio == 0.0 or not np.isfinite(d_ratio) or not np.isfinite(psi):
            raise RuntimeError(f"Newton residual is not finite at z = {z}")
        z_new = _clip_z(z - psi / d_ratio, eta)
        if abs(z_new - z) <= NEWTON_TOL * max(1.0, abs(z_new)):
            return z_new
        z = z_new
    raise RuntimeError(
        f"Newton inversion failed to converge in {NEWTON_MAXITER} iterations "
        f"(rho={rho}, e={e}, eta={eta}, last z={z})"
    )


def _newton_z_density(
    rho: float, T: float, dim: int, h: float, eta: int, m: float, k_B: float
) -> float:
    """Newton on G_{D/2}(z) = ρ h^D / (m (2π m k_B T)^{D/2})."""
    data = rho * h**dim / (m * (2.0 * np.pi * m * k_B * T) ** (dim / 2.0))
    z_cl = _clip_z(data, eta)
    z = z_cl if np.isfinite(z_cl) and z_cl > 0.0 else Z_INIT
    n0 = dim / 2.0
    n_m = n0 - 1.0
    for _ in range(NEWTON_MAXITER):
        g0 = float(G(n0, z, eta))
        gm = float(G(n_m, z, eta))
        psi = g0 - data
        d_psi = gm / z
        if d_psi == 0.0 or not np.isfinite(d_psi) or not np.isfinite(psi):
            raise RuntimeError(f"Newton residual is not finite at z = {z}")
        z_new = _clip_z(z - psi / d_psi, eta)
        if abs(z_new - z) <= NEWTON_TOL * max(1.0, abs(z_new)):
            return z_new
        z = z_new
    raise RuntimeError(
        f"Newton fugacity inversion failed in {NEWTON_MAXITER} iterations "
        f"(rho={rho}, T={T}, eta={eta}, last z={z})"
    )


def find_fugacity(
    rho, T, dim: int = 3, h: float = 1.0, eta: int = 0, m: float = 1.0, k_B: float = 1.0
):
    """Invert (ρ, T) → z from the equilibrium density relation.

    ρ = m (2π m k_B T / h²)^{D/2} G_{D/2}(z).
    """
    eta = _validate_eta(eta)
    rho_arr, t_arr = np.broadcast_arrays(
        np.asarray(rho, dtype=float),
        np.asarray(T, dtype=float),
    )
    scalar_input = rho_arr.ndim == 0
    rho_flat = np.atleast_1d(rho_arr).ravel()
    t_flat = np.atleast_1d(t_arr).ravel()
    z_flat = np.empty_like(rho_flat)

    if eta == 0:
        z_flat = (
            rho_flat * h**dim / (m * (2.0 * np.pi * m * k_B * t_flat) ** (dim / 2.0))
        )
    else:
        for i, (rho_i, t_i) in enumerate(zip(rho_flat, t_flat, strict=True)):
            z_flat[i] = _newton_z_density(float(rho_i), float(t_i), dim, h, eta, m, k_B)

    z_out = z_flat.reshape(rho_arr.shape)
    if scalar_input:
        return float(z_out)
    return z_out


def find_moments(
    rho, e, dim: int = 3, h: float = 1.0, eta: int = 0, m: float = 1.0, k_B: float = 1.0
):
    """Invert (ρ, e) → (z, T, p).

    Parameters
    ----------
    rho, e :
        Density and specific internal energy.  Scalars or arrays; arrays are
        broadcast against each other.
    dim :
        Velocity-space dimension D.
    h :
        Planck constant (may be used as a degeneracy parameter).
    eta :
        Statistics: −1 Fermi, 0 classical, +1 Bose.
    m, k_B :
        Particle mass and Boltzmann constant.  Default 1.

    Returns
    -------
    z, T, p :
        Fugacity, temperature and pressure p = (2/D) ρ e.  Scalars if both
        ``rho`` and ``e`` are scalars, otherwise arrays of the broadcast shape.
    """
    eta = _validate_eta(eta)
    rho_arr, e_arr = np.broadcast_arrays(
        np.asarray(rho, dtype=float),
        np.asarray(e, dtype=float),
    )
    scalar_input = rho_arr.ndim == 0
    rho_flat = np.atleast_1d(rho_arr).ravel()
    e_flat = np.atleast_1d(e_arr).ravel()
    z_flat = np.empty_like(rho_flat)

    if eta == 0:
        T_flat = 2.0 * m * e_flat / (dim * k_B)
        z_flat = rho_flat * h**dim / (m * (2.0 * np.pi * m * k_B * T_flat) ** (dim / 2))
    else:
        for i, (rho_i, e_i) in enumerate(zip(rho_flat, e_flat, strict=True)):
            z_flat[i] = _newton_z(float(rho_i), float(e_i), dim, h, eta, m)
        g0 = G(dim / 2, z_flat, eta)
        T_flat = (h**2 / (2.0 * np.pi * m * k_B)) * (rho_flat / (m * g0)) ** (2.0 / dim)

    p_flat = (2.0 / dim) * e_flat * rho_flat
    z_out = z_flat.reshape(rho_arr.shape)
    T_out = T_flat.reshape(rho_arr.shape)
    p_out = p_flat.reshape(rho_arr.shape)
    if scalar_input:
        return float(z_out), float(T_out), float(p_out)
    return z_out, T_out, p_out
