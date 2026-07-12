# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Exact Riemann solvers for classical and quantum Euler gases.

This module ports ``matlab/QEulerExactToro.m`` (Toro, 1999) with kinetic
pre-processing and quantum fugacity corrections for FD/BE/MB statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ideal_gases.polylog import polylog

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
    p_l, p_r = _effective_pressures(
        rho_l, t_l, rho_r, t_r, n, h, statistic
    )
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


def _be(n: float, z: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(polylog(n, z), dtype=np.float64)


def _fd(n: float, z: ArrayLike) -> NDArray[np.float64]:
    return -np.asarray(polylog(n, -z), dtype=np.float64)


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

    q_func = _fd if statistic == "FD" else _be
    clip_fn = (lambda z: min(float(z), 0.999999)) if statistic == "BE" else None

    if rho_l > RHO_FLOOR:
        z_l = _newton_scalar(
            lambda z: float(_frhot(q_func, z, n, rho_l, t_l, h)),
            lambda z: float(_dfrhot(q_func, z, n)),
            0.001,
            clip=clip_fn,
        )
        p_l = rho_l * t_l * _safe_ratio(
            q_func(n / 2.0 + 1.0, z_l), q_func(n / 2.0, z_l)
        )
    else:
        p_l = 0.0

    if rho_r > RHO_FLOOR:
        z_r = _newton_scalar(
            lambda z: float(_frhot(q_func, z, n, rho_r, t_r, h)),
            lambda z: float(_dfrhot(q_func, z, n)),
            0.001,
            clip=clip_fn,
        )
        p_r = rho_r * t_r * _safe_ratio(
            q_func(n / 2.0 + 1.0, z_r), q_func(n / 2.0, z_r)
        )
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
    if statistic == "MB":
        t = (2.0 / n) * e
        z = rho * h**n / (2.0 * np.pi * t) ** (n / 2.0)
        return z, t

    q_func = _fd if statistic == "FD" else _be
    clip = _clip_be_fugacity if statistic == "BE" else None
    z = _newton_array(
        lambda zv: _frhoe(q_func, zv, n, rho, e, h),
        lambda zv: _dfrhoe(q_func, zv, n),
        np.full_like(rho, 0.001),
        clip=clip,
    )

    if statistic == "FD":
        t = (rho / _fd(n / 2.0, z)) ** (2.0 / n) * h**2 / (2.0 * np.pi)
    else:
        t = (rho / _be(n / 2.0, z)) ** (2.0 / n) * h**2 / (2.0 * np.pi)
    return z, t


def _frhot(
    q_func: Callable[[float, ArrayLike], NDArray[np.float64]],
    z: ArrayLike,
    n: float,
    rho: ArrayLike,
    t: ArrayLike,
    h: float,
) -> NDArray[np.float64]:
    rho_arr = np.asarray(rho, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    return rho_arr * h**n / (2.0 * np.pi * t_arr) ** (n / 2.0) - q_func(
        n / 2.0, z
    )


def _dfrhot(
    q_func: Callable[[float, ArrayLike], NDArray[np.float64]],
    z: ArrayLike,
    n: float,
) -> NDArray[np.float64]:
    z_arr = np.asarray(z, dtype=np.float64)
    return -q_func(n / 2.0 - 1.0, z_arr) / z_arr


def _frhoe(
    q_func: Callable[[float, ArrayLike], NDArray[np.float64]],
    z: ArrayLike,
    n: float,
    rho: ArrayLike,
    e: ArrayLike,
    h: float,
) -> NDArray[np.float64]:
    rho_arr = np.asarray(rho, dtype=np.float64)
    e_arr = np.asarray(e, dtype=np.float64)
    q_mid = q_func(n / 2.0, z)
    q_high = q_func((n + 2.0) / 2.0, z)
    return q_mid ** ((n + 2.0) / n) / q_high - h**2 * n * rho_arr ** (
        2.0 / n
    ) / (4.0 * np.pi * e_arr)


def _dfrhoe(
    q_func: Callable[[float, ArrayLike], NDArray[np.float64]],
    z: ArrayLike,
    n: float,
) -> NDArray[np.float64]:
    z_arr = np.asarray(z, dtype=np.float64)
    q_minus = q_func(n / 2.0 - 1.0, z_arr)
    q_mid = q_func(n / 2.0, z_arr)
    q_plus = q_func(n / 2.0 + 1.0, z_arr)
    return ((2.0 + n) / n) * (
        q_mid ** ((2.0 + n) / n - 1.0) * q_minus / (z_arr * q_plus)
    ) - q_mid ** (2.0 * (1.0 + n) / n) / (z_arr * q_plus**2)


def _clip_be_fugacity(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.minimum(z, 0.999999)


def _newton_scalar(
    func: Callable[[float], float],
    dfunc: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
    clip: Callable[[float], float] | None = None,
) -> float:
    x = x0
    for _ in range(max_iter):
        delta = func(x) / dfunc(x)
        x -= delta
        if clip is not None:
            x = clip(x)
        if abs(delta) < tol:
            break
    return x


def _newton_array(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dfunc: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
    clip: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    x = x0.astype(np.float64, copy=True)
    for _ in range(max_iter):
        delta = func(x) / dfunc(x)
        x -= delta
        if clip is not None:
            x = clip(x)
        if np.max(np.abs(delta)) < tol:
            break
    return x


def _safe_ratio(
    numerator: ArrayLike, denominator: ArrayLike
) -> NDArray[np.float64] | float:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.ndim == 0 and den.ndim == 0:
        return float(num / den)
    return num / den


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

    left_active = abs(rho_l_star - rho_l) > 1e-8 * max(rho_l, 1.0) or abs(
        u_star - u_l
    ) > 1e-8
    right_active = abs(rho_r_star - rho_r) > 1e-8 * max(rho_r, 1.0) or abs(
        u_star - u_r
    ) > 1e-8

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
        return rho_k * ((gamma + 1.0) * p_star + (gamma - 1.0) * p_k) / (
            (gamma - 1.0) * p_star + (gamma + 1.0) * p_k
        )
    return rho_k * (p_star / p_k) ** (1.0 / gamma)


def _shock_speed(
    rho_k: float, u_k: float, rho_star: float, u_star: float
) -> float:
    return (rho_k * u_k - rho_star * u_star) / (rho_k - rho_star)
