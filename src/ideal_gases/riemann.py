# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Exact Riemann solvers for classical and quantum Euler gases.

This module ports ``matlab/QEulerExactToro.m`` (Toro, 1999) with kinetic
pre-processing and quantum fugacity corrections for FD/BE/MB statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ideal_gases.equilibrium import G, find_fugacity, find_moments as _eq_find_moments

Statistic = Literal["FD", "BE", "MB"]

P_FLOOR = 1e-300
RHO_FLOOR = 1e-10
DEFAULT_DX = 0.002
DEFAULT_X0 = 0.5

__all__ = [
    "RiemannResult",
    "adiabatic_index",
    "classical_gas",
    "quantum_gas",
]


@dataclass(frozen=True)
class RiemannResult:
    x: NDArray[np.float64]
    rho: NDArray[np.float64]
    ux: NDArray[np.float64]
    p: NDArray[np.float64]
    e: NDArray[np.float64]
    z: NDArray[np.float64]
    t: NDArray[np.float64]
    mach: NDArray[np.float64]
    entropy: NDArray[np.float64]


@dataclass
class _RiemannSolution:
    case_id: str
    gamma: float
    rho_l: float
    u_l: float
    p_l: float
    c_l: float
    rho_r: float
    u_r: float
    p_r: float
    c_r: float
    p_star: float = 0.0
    u_star: float = 0.0
    rho_l_star: float = 0.0
    rho_r_star: float = 0.0
    wave_l: str = "rarefaction"
    wave_r: str = "rarefaction"
    dry_vel_l: float = 0.0
    dry_vel_r: float = 0.0


def adiabatic_index(n_gas_dofs: float) -> float:
    """Return ``gamma = (n + 2) / n`` for ``n`` spatial degrees of freedom."""
    return (n_gas_dofs + 2.0) / n_gas_dofs


def classical_gas(
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    t_end: float,
    gamma: float,
    *,
    x: ArrayLike | None = None,
    x0: float = DEFAULT_X0,
    dx: float = DEFAULT_DX,
) -> RiemannResult:
    """Exact Toro Riemann solver for a classical ideal gas."""
    return _solve_profile(
        rho_l=rho_l,
        u_l=u_l,
        p_l=p_l,
        rho_r=rho_r,
        u_r=u_r,
        p_r=p_r,
        t_end=t_end,
        gamma=gamma,
        statistic="MB",
        n=2.0 / max(gamma - 1.0, 1e-12),
        h=1.0,
        x=x,
        x0=x0,
        dx=dx,
        skip_preprocess=True,
    )


def quantum_gas(
    rho_l: float,
    u_l: float,
    t_l: float,
    rho_r: float,
    u_r: float,
    t_r: float,
    t_end: float,
    n: float,
    h: float,
    statistic: Statistic = "FD",
    *,
    x: ArrayLike | None = None,
    x0: float = DEFAULT_X0,
    dx: float = DEFAULT_DX,
) -> RiemannResult:
    """Exact Toro Riemann solver with quantum FD/BE/MB kinetic inputs."""
    gamma = adiabatic_index(n)
    p_l, p_r = _effective_pressures(rho_l, t_l, rho_r, t_r, n, h, statistic)
    return _solve_profile(
        rho_l=rho_l,
        u_l=u_l,
        p_l=p_l,
        rho_r=rho_r,
        u_r=u_r,
        p_r=p_r,
        t_end=t_end,
        gamma=gamma,
        statistic=statistic,
        n=n,
        h=h,
        x=x,
        x0=x0,
        dx=dx,
        skip_preprocess=True,
    )


def _solve_profile(
    *,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    t_end: float,
    gamma: float,
    statistic: Statistic,
    n: float,
    h: float,
    x: ArrayLike | None,
    x0: float,
    dx: float,
    skip_preprocess: bool,
) -> RiemannResult:
    del skip_preprocess  # pre-processing is done by the public entry points.

    if x is None:
        x_arr = np.arange(0.0, 1.0 + 0.5 * dx, dx, dtype=np.float64)
    else:
        x_arr = np.asarray(x, dtype=np.float64)

    sol = _riemann_exact_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    rho, ux, p = _sample_riemann_profile(sol, x_arr, x0, t_end, gamma)

    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    c = np.sqrt(np.maximum(gamma * p / rho, 0.0))
    mach = ux / np.maximum(c, 1e-12)
    entropy = np.log(p / rho**gamma)
    e = p / ((gamma - 1.0) * rho)

    z, t = _postprocess_quantum(statistic, n, h, rho, e)
    return RiemannResult(x_arr, rho, ux, p, e, z, t, mach, entropy)


_STATISTIC_TO_ETA = {"FD": -1, "BE": 1, "MB": 0}


def _effective_pressures(
    rho_l: float,
    t_l: float,
    rho_r: float,
    t_r: float,
    n: float,
    h: float,
    statistic: Statistic,
) -> tuple[float, float]:
    if statistic == "MB":
        return rho_l * t_l, rho_r * t_r

    eta = _STATISTIC_TO_ETA[statistic]

    if rho_l > RHO_FLOOR:
        z_l = find_fugacity(rho_l, t_l, dim=n, h=h, eta=eta, m=1.0, k_B=1.0)
        g0_l = G(n / 2.0, z_l, eta)
        g1_l = G(n / 2.0 + 1.0, z_l, eta)
        p_l = float(rho_l * t_l * g1_l / g0_l)
    else:
        p_l = 0.0

    if rho_r > RHO_FLOOR:
        z_r = find_fugacity(rho_r, t_r, dim=n, h=h, eta=eta, m=1.0, k_B=1.0)
        g0_r = G(n / 2.0, z_r, eta)
        g1_r = G(n / 2.0 + 1.0, z_r, eta)
        p_r = float(rho_r * t_r * g1_r / g0_r)
    else:
        p_r = 0.0

    return p_l, p_r


def _postprocess_quantum(
    statistic: Statistic,
    n: float,
    h: float,
    rho: NDArray[np.float64],
    e: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    eta = _STATISTIC_TO_ETA[statistic]
    z, t, _p = _eq_find_moments(rho, e, dim=n, h=h, eta=eta, m=1.0, k_B=1.0)
    return np.asarray(z, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _riemann_exact_state(
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    gamma: float,
) -> _RiemannSolution:
    c_l = np.sqrt(gamma * p_l / max(rho_l, 1e-300))
    c_r = np.sqrt(gamma * p_r / max(rho_r, 1e-300))
    dry_vel_l = u_l + 2.0 * c_l / (gamma - 1.0)
    dry_vel_r = u_r - 2.0 * c_r / (gamma - 1.0)

    sol = _RiemannSolution(
        case_id="standard",
        gamma=gamma,
        rho_l=rho_l,
        u_l=u_l,
        p_l=p_l,
        c_l=c_l,
        rho_r=rho_r,
        u_r=u_r,
        p_r=p_r,
        c_r=c_r,
    )

    if p_l < 1e-10 and rho_l < 1e-10:
        sol.case_id = "vacuum_left"
        return sol
    if p_r < 1e-10 and rho_r < 1e-10:
        sol.case_id = "vacuum_right"
        return sol
    if dry_vel_l <= dry_vel_r:
        sol.case_id = "vacuum_middle"
        sol.dry_vel_l = dry_vel_l
        sol.dry_vel_r = dry_vel_r
        return sol

    p_star = max(
        1e-8,
        0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (c_l + c_r),
    )
    change = 1.0
    for _ in range(50):
        f_l, df_l, wave_l = _pressure_wave(p_star, p_l, rho_l, gamma)
        f_r, df_r, wave_r = _pressure_wave(p_star, p_r, rho_r, gamma)
        f = f_l + f_r + u_r - u_l
        change = abs(f)
        if change < 1e-6:
            break
        p_star = max(1e-10, p_star - f / (df_l + df_r))

    if change > 1e-3:
        msg = f"Newton iteration did not converge (|f| = {change:.3e})."
        raise RuntimeError(msg)

    _, _, wave_l = _pressure_wave(p_star, p_l, rho_l, gamma)
    _, _, wave_r = _pressure_wave(p_star, p_r, rho_r, gamma)
    u_star = u_l - _pressure_wave_value(p_star, p_l, rho_l, gamma)

    sol.p_star = p_star
    sol.u_star = u_star
    sol.rho_l_star = _star_density(p_star, p_l, rho_l, gamma, wave_l)
    sol.rho_r_star = _star_density(p_star, p_r, rho_r, gamma, wave_r)
    sol.wave_l = wave_l
    sol.wave_r = wave_r
    return sol


def _sample_riemann_profile(
    sol: _RiemannSolution,
    x: NDArray[np.float64],
    x0: float,
    t: float,
    gamma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rho = np.zeros_like(x)
    ux = np.zeros_like(x)
    p = np.zeros_like(x)

    if t <= 0.0:
        left = x < x0
        rho[left] = sol.rho_l
        ux[left] = sol.u_l
        p[left] = sol.p_l
        rho[~left] = sol.rho_r
        ux[~left] = sol.u_r
        p[~left] = sol.p_r
        return rho, ux, p

    for i, xi in enumerate(x):
        rho[i], ux[i], p[i] = _sample_at_point(sol, xi, x0, t, gamma)
    return rho, ux, p


def _sample_at_point(
    sol: _RiemannSolution,
    xi: float,
    x0: float,
    t: float,
    gamma: float,
) -> tuple[float, float, float]:
    if sol.case_id == "vacuum_left":
        if xi < x0:
            return 0.0, 0.0, 0.0
        return sol.rho_r, sol.u_r, sol.p_r
    if sol.case_id == "vacuum_right":
        if xi < x0:
            return sol.rho_l, sol.u_l, sol.p_l
        return 0.0, 0.0, 0.0
    if sol.case_id == "vacuum_middle":
        return _sample_vacuum_middle_point(sol, xi, x0, t, gamma)
    return _sample_standard_point(sol, xi, x0, t, gamma)


def _sample_standard_point(
    sol: _RiemannSolution,
    xi: float,
    x0: float,
    t: float,
    gamma: float,
) -> tuple[float, float, float]:
    s = (xi - x0) / t
    rho_l, u_l, p_l, c_l = sol.rho_l, sol.u_l, sol.p_l, sol.c_l
    rho_r, u_r, p_r, c_r = sol.rho_r, sol.u_r, sol.p_r, sol.c_r
    p_star, u_star = sol.p_star, sol.u_star
    rho_l_star, rho_r_star = sol.rho_l_star, sol.rho_r_star

    left_active = (
        abs(rho_l_star - rho_l) > 1e-8 * max(rho_l, 1.0) or abs(u_star - u_l) > 1e-8
    )
    right_active = (
        abs(rho_r_star - rho_r) > 1e-8 * max(rho_r, 1.0) or abs(u_star - u_r) > 1e-8
    )

    left_outer = left_inner = right_inner = right_outer = 0.0

    if left_active:
        if sol.wave_l == "shock":
            left_outer = _shock_speed(rho_l, u_l, rho_l_star, u_star)
        else:
            c_l_star = np.sqrt(gamma * p_star / rho_l_star)
            left_outer = u_l - c_l
            left_inner = u_star - c_l_star

    if right_active:
        if sol.wave_r == "shock":
            right_outer = _shock_speed(rho_r, u_r, rho_r_star, u_star)
        else:
            c_r_star = np.sqrt(gamma * p_star / rho_r_star)
            right_inner = u_star + c_r_star
            right_outer = u_r + c_r

    if left_active:
        if sol.wave_l == "shock":
            if s <= left_outer:
                return rho_l, u_l, p_l
        elif s <= left_outer:
            return rho_l, u_l, p_l
        elif s <= left_inner:
            return _rarefaction_state_left(s, gamma, rho_l, u_l, p_l, c_l)

    if s <= u_star:
        return rho_l_star, u_star, p_star

    if right_active:
        if sol.wave_r == "shock":
            if s <= right_outer:
                return rho_r_star, u_star, p_star
        elif s <= right_inner:
            return rho_r_star, u_star, p_star
        elif s <= right_outer:
            return _rarefaction_state_right(s, gamma, rho_r, u_r, p_r, c_r)

    return rho_r, u_r, p_r


def _sample_vacuum_middle_point(
    sol: _RiemannSolution,
    xi: float,
    x0: float,
    t: float,
    gamma: float,
) -> tuple[float, float, float]:
    s = (xi - x0) / t
    rho_l, u_l, p_l, c_l = sol.rho_l, sol.u_l, sol.p_l, sol.c_l
    rho_r, u_r, p_r, c_r = sol.rho_r, sol.u_r, sol.p_r, sol.c_r

    if s <= u_l - c_l:
        return rho_l, u_l, p_l
    if s <= sol.dry_vel_l:
        return _rarefaction_state_left(s, gamma, rho_l, u_l, p_l, c_l)
    if s <= sol.dry_vel_r:
        return 0.0, 0.0, 0.0
    if s <= u_r + c_r:
        return _rarefaction_state_right(s, gamma, rho_r, u_r, p_r, c_r)
    return rho_r, u_r, p_r


def _rarefaction_state_left(
    s: float,
    gamma: float,
    rho_k: float,
    u_k: float,
    p_k: float,
    c_k: float,
) -> tuple[float, float, float]:
    u = ((gamma - 1.0) * u_k + 2.0 * c_k) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) * s
    c = (u_k + 2.0 * c_k / (gamma - 1.0) - u) * (gamma - 1.0) / 2.0
    c_ratio = max(c / c_k, 0.0)
    p_out = p_k * c_ratio ** (2.0 * gamma / (gamma - 1.0))
    rho = rho_k * c_ratio ** (2.0 / (gamma - 1.0))
    return rho, u, p_out


def _rarefaction_state_right(
    s: float,
    gamma: float,
    rho_k: float,
    u_k: float,
    p_k: float,
    c_k: float,
) -> tuple[float, float, float]:
    u = ((gamma - 1.0) * u_k - 2.0 * c_k) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) * s
    c = (u + 2.0 * c_k / (gamma - 1.0) - u_k) * (gamma - 1.0) / 2.0
    c_ratio = max(c / c_k, 0.0)
    p_out = p_k * c_ratio ** (2.0 * gamma / (gamma - 1.0))
    rho = rho_k * c_ratio ** (2.0 / (gamma - 1.0))
    return rho, u, p_out


def _pressure_wave(
    p: float, p_k: float, rho_k: float, gamma: float
) -> tuple[float, float, str]:
    c_k = np.sqrt(gamma * p_k / rho_k)
    if p >= p_k:
        wave = "shock"
        a_coef = 2.0 / ((gamma + 1.0) * rho_k)
        b_coef = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * np.sqrt(a_coef / (p + b_coef))
        df = np.sqrt(a_coef / (p + b_coef)) * (1.0 - 0.5 * (p - p_k) / (p + b_coef))
        return f, df, wave

    wave = "rarefaction"
    pr = p / p_k
    f = 2.0 * c_k / (gamma - 1.0) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    df = (1.0 / (rho_k * c_k)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df, wave


def _pressure_wave_value(p: float, p_k: float, rho_k: float, gamma: float) -> float:
    f, _, _ = _pressure_wave(p, p_k, rho_k, gamma)
    return f


def _star_density(
    p_star: float, p_k: float, rho_k: float, gamma: float, wave: str
) -> float:
    if wave == "shock":
        return (
            rho_k
            * ((gamma + 1.0) * p_star + (gamma - 1.0) * p_k)
            / ((gamma - 1.0) * p_star + (gamma + 1.0) * p_k)
        )
    return rho_k * (p_star / p_k) ** (1.0 / gamma)


def _shock_speed(rho_k: float, u_k: float, rho_star: float, u_star: float) -> float:
    return (rho_k * u_k - rho_star * u_star) / (rho_k - rho_star)
