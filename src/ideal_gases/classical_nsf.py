# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""1-D classical Navier-Stokes-Fourier solver (ideal-gas EOS).

Cell states use ``T = (γ-1) e`` and ``p = ρ T`` with ``γ = (dim+2)/dim``.
There is no fugacity and no polylog.  The discrete scheme (MUSCL-HLLC
or Rusanov, Stokes stress, Heun, CFL) is duplicated here on purpose so
changes to the quantum NSF cannot silently alter this classical
reference.

For ``dim = 1`` the Stokes factor ``2(dim-1)/dim`` vanishes, so there is
no viscous momentum flux; heat conduction remains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ideal_gases._progress import (
    DEFAULT_PROGRESS_EVERY,
    _validate_progress_every,
    nsf_pbar,
)
from ideal_gases.riemann import DEFAULT_DX, DEFAULT_X0

__all__ = [
    "ClassicalNSFResult",
    "classical_nsf",
]

VALID_DIM = frozenset({1, 2, 3})
PR_CLASSICAL = 2.0 / 3.0
RHO_FLOOR = 1e-14
ConvFlux = Literal["hllc", "rusanov"]
TransportFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class ClassicalNSFResult(NamedTuple):
    """Cell-centered NSF fields. Unpack as ``rho, u, t, p, q``."""

    rho: NDArray[np.float64]
    u: NDArray[np.float64]
    t: NDArray[np.float64]
    p: NDArray[np.float64]
    q: NDArray[np.float64]


def _validate_dim(dim) -> int:
    dim_i = int(dim)
    if dim_i not in VALID_DIM or float(dim) != float(dim_i):
        raise ValueError(f"dim = {dim} invalid; expected dim in {{1, 2, 3}}")
    return dim_i


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = np.sign(a)
    return s * np.maximum(0.0, np.minimum(np.abs(a), s * b))


@dataclass
class _NSFProblem:
    dim: int
    kn: float
    pr: float
    gamma: float
    cp: float
    stokes_xx: float
    conv_flux: str
    viscosity: TransportFn
    conductivity: TransportFn


def _primitives(U: np.ndarray, prob: _NSFProblem):
    rho = np.maximum(U[0], RHO_FLOOR)
    u = U[1] / rho
    e = np.maximum(U[2] / rho - 0.5 * u * u, RHO_FLOOR)
    temp = (prob.gamma - 1.0) * e
    pressure = rho * temp
    return rho, u, temp, pressure


def _inviscid_flux(rho, u, pressure, rho_e):
    return np.vstack([rho * u, rho * u * u + pressure, u * (rho_e + pressure)])


def _conserved_from_primitives(rho, u, pressure, gamma: float):
    rho_e = 0.5 * rho * u * u + pressure / (gamma - 1.0)
    return np.vstack([rho, rho * u, rho_e])


def _rusanov_interface(U: np.ndarray, prob: _NSFProblem) -> np.ndarray:
    rho, u, _temp, pressure = _primitives(U, prob)
    c = np.sqrt(prob.gamma * np.maximum(pressure / np.maximum(rho, RHO_FLOOR), 0.0))
    flux = _inviscid_flux(rho, u, pressure, U[2])
    s = np.maximum(np.abs(u[:-1]) + c[:-1], np.abs(u[1:]) + c[1:])
    f_int = 0.5 * (flux[:, :-1] + flux[:, 1:]) - 0.5 * s[None, :] * (
        U[:, 1:] - U[:, :-1]
    )
    return np.concatenate([flux[:, 0:1], f_int, flux[:, -1:]], axis=1)


def _muscl_interface_primitives(rho, u, pressure, dx: float):
    qs = np.vstack([rho, u, pressure])
    nx = qs.shape[1]
    pad = np.empty((3, nx + 2), dtype=qs.dtype)
    pad[:, 0] = qs[:, 0]
    pad[:, 1:-1] = qs
    pad[:, -1] = qs[:, -1]

    sigma = np.zeros_like(pad)
    left = (pad[:, 1:-1] - pad[:, :-2]) / dx
    right = (pad[:, 2:] - pad[:, 1:-1]) / dx
    sigma[:, 1:-1] = _minmod(left, right)

    q_l = qs[:, :-1] + 0.5 * dx * sigma[:, 1:-2]
    q_r = qs[:, 1:] - 0.5 * dx * sigma[:, 2:-1]
    q_l[0] = np.maximum(q_l[0], RHO_FLOOR)
    q_r[0] = np.maximum(q_r[0], RHO_FLOOR)
    q_l[2] = np.maximum(q_l[2], RHO_FLOOR)
    q_r[2] = np.maximum(q_r[2], RHO_FLOOR)
    return q_l[0], q_l[1], q_l[2], q_r[0], q_r[1], q_r[2]


def _hllc_flux(ul: np.ndarray, ur: np.ndarray, gamma: float) -> np.ndarray:
    gm1 = gamma - 1.0
    rho_l = np.maximum(ul[0], RHO_FLOOR)
    rho_r = np.maximum(ur[0], RHO_FLOOR)
    u_l = ul[1] / rho_l
    u_r = ur[1] / rho_r
    p_l = np.maximum(gm1 * (ul[2] - 0.5 * rho_l * u_l * u_l), RHO_FLOOR)
    p_r = np.maximum(gm1 * (ur[2] - 0.5 * rho_r * u_r * u_r), RHO_FLOOR)
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    s_l = np.minimum(u_l - c_l, u_r - c_r)
    s_r = np.maximum(u_l + c_l, u_r + c_r)
    denom = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    s_star = np.where(
        np.abs(denom) > 1e-14,
        (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / denom,
        0.5 * (u_l + u_r),
    )

    def star_state(u_state, rho, vel, p, s):
        ds = np.where(np.abs(s - s_star) < 1e-14, 1e-14, s - s_star)
        su = np.where(np.abs(s - vel) < 1e-14, 1e-14, s - vel)
        factor = rho * su / ds
        e_star = u_state[2] / rho + (s_star - vel) * (s_star + p / (rho * su))
        return np.vstack([factor, factor * s_star, factor * e_star])

    u_star_l = star_state(ul, rho_l, u_l, p_l, s_l)
    u_star_r = star_state(ur, rho_r, u_r, p_r, s_r)
    f_l = _inviscid_flux(rho_l, u_l, p_l, ul[2])
    f_r = _inviscid_flux(rho_r, u_r, p_r, ur[2])

    flux = np.empty_like(ul)
    m1 = s_l >= 0.0
    m2 = (s_l < 0.0) & (s_star >= 0.0)
    m3 = (s_star < 0.0) & (s_r >= 0.0)
    m4 = s_r < 0.0
    flux[:, m1] = f_l[:, m1]
    flux[:, m2] = f_l[:, m2] + s_l[m2] * (u_star_l[:, m2] - ul[:, m2])
    flux[:, m3] = f_r[:, m3] + s_r[m3] * (u_star_r[:, m3] - ur[:, m3])
    flux[:, m4] = f_r[:, m4]
    leftover = ~(m1 | m2 | m3 | m4)
    if np.any(leftover):
        flux[:, leftover] = 0.5 * (f_l[:, leftover] + f_r[:, leftover])
    return flux


def _hllc_muscl_interface(U: np.ndarray, dx: float, prob: _NSFProblem) -> np.ndarray:
    rho, u, _temp, pressure = _primitives(U, prob)
    rho_l, u_l, p_l, rho_r, u_r, p_r = _muscl_interface_primitives(rho, u, pressure, dx)
    f_int = _hllc_flux(
        _conserved_from_primitives(rho_l, u_l, p_l, prob.gamma),
        _conserved_from_primitives(rho_r, u_r, p_r, prob.gamma),
        prob.gamma,
    )
    f_cell = _inviscid_flux(rho, u, pressure, U[2])
    return np.concatenate([f_cell[:, 0:1], f_int, f_cell[:, -1:]], axis=1)


def _viscous_interface(U: np.ndarray, dx: float, prob: _NSFProblem) -> np.ndarray:
    rho, u, temp, _p = _primitives(U, prob)
    mu = np.asarray(prob.viscosity(rho, temp), dtype=float)
    kappa = np.asarray(prob.conductivity(rho, temp), dtype=float)
    nx = U.shape[1]
    fv = np.zeros((3, nx + 1))
    ux = (u[1:] - u[:-1]) / dx
    tx = (temp[1:] - temp[:-1]) / dx
    mu_f = 0.5 * (mu[1:] + mu[:-1])
    kappa_f = 0.5 * (kappa[1:] + kappa[:-1])
    u_f = 0.5 * (u[1:] + u[:-1])
    tau_xx = prob.stokes_xx * mu_f * ux
    fv[1, 1:-1] = -tau_xx
    fv[2, 1:-1] = -kappa_f * tx - tau_xx * u_f
    return fv


def _rhs(U: np.ndarray, dx: float, prob: _NSFProblem) -> np.ndarray:
    if prob.conv_flux == "rusanov":
        fi = _rusanov_interface(U, prob)
    elif prob.conv_flux == "hllc":
        fi = _hllc_muscl_interface(U, dx, prob)
    else:
        raise ValueError(f"conv_flux={prob.conv_flux!r}; expected 'hllc' or 'rusanov'")
    fv = _viscous_interface(U, dx, prob)
    flux = fi + fv
    return -(flux[:, 1:] - flux[:, :-1]) / dx


def _timestep(U: np.ndarray, dx: float, cfl: float, prob: _NSFProblem) -> float:
    rho, u, temp, pressure = _primitives(U, prob)
    c = np.sqrt(prob.gamma * np.maximum(pressure / np.maximum(rho, RHO_FLOOR), 0.0))
    dt_conv = cfl * dx / np.max(np.abs(u) + c)
    mu = np.asarray(prob.viscosity(rho, temp), dtype=float)
    kappa = np.asarray(prob.conductivity(rho, temp), dtype=float)
    nu = np.max(prob.stokes_xx * mu / np.maximum(rho, RHO_FLOOR))
    alpha = np.max(kappa / np.maximum(rho * prob.cp, RHO_FLOOR))
    dt_diff = cfl * 0.5 * dx**2 / max(float(nu), float(alpha), 1e-14)
    return min(dt_conv, dt_diff)


def _initialize(
    x: np.ndarray,
    x0: float,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    gamma: float,
) -> np.ndarray:
    rho = np.where(x <= x0, rho_l, rho_r).astype(float)
    u = np.where(x <= x0, u_l, u_r).astype(float)
    pressure = np.where(x <= x0, p_l, p_r).astype(float)
    rho_e = 0.5 * rho * u * u + pressure / (gamma - 1.0)
    return np.vstack([rho, rho * u, rho_e])


def _fourier_heat_flux(temp: np.ndarray, kappa: np.ndarray, dx: float) -> np.ndarray:
    q = np.empty_like(temp)
    q[1:-1] = -kappa[1:-1] * (temp[2:] - temp[:-2]) / (2.0 * dx)
    q[0] = -kappa[0] * (temp[1] - temp[0]) / dx
    q[-1] = -kappa[-1] * (temp[-1] - temp[-2]) / dx
    return q


def classical_nsf(
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    t_end: float,
    dim: int,
    kn: float,
    pr: float = PR_CLASSICAL,
    *,
    x: ArrayLike | None = None,
    x0: float = DEFAULT_X0,
    dx: float = DEFAULT_DX,
    cfl: float = 0.4,
    conv_flux: ConvFlux = "hllc",
    viscosity: TransportFn | None = None,
    conductivity: TransportFn | None = None,
    progress: bool = False,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> ClassicalNSFResult:
    """Advance 1-D classical NSF from a Riemann initial condition.

    Parameters
    ----------
    dim:
        Gas dimension in ``{1, 2, 3}``.
    kn:
        Knudsen number; default CE closure is ``μ = kn ρ T``.
    pr:
        Prandtl number used by the default conductivity
        ``κ = Cp μ / Pr`` with ``Cp = (dim+2)/2``.
    viscosity, conductivity:
        Optional ``(rho, temp) -> array`` replacements for the CE
        closures.
    progress:
        If True, show a tqdm bar of physical time. Requires
        ``pip install ideal-gases[progress]``.
    progress_every:
        Refresh the bar every this many time steps (ignored when
        ``progress`` is False).
    """
    dim_i = _validate_dim(dim)
    every = _validate_progress_every(progress_every)
    if conv_flux not in ("hllc", "rusanov"):
        raise ValueError(f"conv_flux={conv_flux!r}; expected 'hllc' or 'rusanov'")

    gamma = (dim_i + 2.0) / dim_i
    cp = (dim_i + 2.0) / 2.0

    def mu_fn(rho, temp):
        return kn * np.asarray(rho, dtype=float) * np.asarray(temp, dtype=float)

    def kappa_fn(rho, temp):
        mu = (viscosity or mu_fn)(rho, temp)
        return cp * np.asarray(mu, dtype=float) / pr

    prob = _NSFProblem(
        dim=dim_i,
        kn=kn,
        pr=pr,
        gamma=gamma,
        cp=cp,
        stokes_xx=2.0 * (dim_i - 1) / dim_i,
        conv_flux=conv_flux,
        viscosity=viscosity or mu_fn,
        conductivity=conductivity or kappa_fn,
    )

    if x is None:
        x_arr = np.arange(0.0, 1.0 + 0.5 * dx, dx, dtype=np.float64)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size < 2:
        raise ValueError("x must be a 1-D grid with at least two cells")
    dx_cell = float(x_arr[1] - x_arr[0])

    u_cons = _initialize(x_arr, x0, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    t = 0.0
    n_step = 0
    pending = 0.0
    with nsf_pbar(progress, total=t_end, desc="classical NSF") as pbar:
        while t < t_end - 1e-15:
            dt = min(_timestep(u_cons, dx_cell, cfl, prob), t_end - t)
            k1 = _rhs(u_cons, dx_cell, prob)
            u_star = u_cons + dt * k1
            k2 = _rhs(u_star, dx_cell, prob)
            u_cons = u_cons + 0.5 * dt * (k1 + k2)
            t += dt
            pending += dt
            n_step += 1
            if n_step % every == 0:
                pbar.update(min(pending, pbar.total - pbar.n))
                pending = 0.0
        if pending > 0.0:
            pbar.update(min(pending, pbar.total - pbar.n))

    rho, u, temp, pressure = _primitives(u_cons, prob)
    kappa = np.asarray(prob.conductivity(rho, temp), dtype=float)
    q = _fourier_heat_flux(temp, kappa, dx_cell)
    return ClassicalNSFResult(rho, u, temp, pressure, q)
