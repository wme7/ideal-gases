# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Toro (1999) classical shock-tube test problems."""

from __future__ import annotations

from dataclasses import dataclass

from ideal_gases.riemann import DEFAULT_DX, DEFAULT_X0


@dataclass(frozen=True)
class ToroTest:
    number: int
    name: str
    rho_l: float
    u_l: float
    p_l: float
    rho_r: float
    u_r: float
    p_r: float
    t_end: float
    gamma: float = 1.4
    x0: float = DEFAULT_X0
    x_min: float = 0.0
    x_max: float = 1.0
    dx: float = DEFAULT_DX
    reference: str = "Toro (1999) Riemann Solvers and Numerical Methods for Fluid Dynamics"


TORO_TESTS: dict[int, ToroTest] = {
    1: ToroTest(
        number=1,
        name="Modified Sod",
        rho_l=1.0,
        u_l=0.75,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=0.2,
        x0=0.3,
    ),
    2: ToroTest(
        number=2,
        name="123 / double rarefaction",
        rho_l=1.0,
        u_l=-2.0,
        p_l=0.4,
        rho_r=1.0,
        u_r=2.0,
        p_r=0.4,
        t_end=0.15,
    ),
    3: ToroTest(
        number=3,
        name="Strong shock",
        rho_l=1.0,
        u_l=0.0,
        p_l=1000.0,
        rho_r=1.0,
        u_r=0.0,
        p_r=0.01,
        t_end=0.012,
    ),
    4: ToroTest(
        number=4,
        name="Shock–contact–rarefaction",
        rho_l=5.99924,
        u_l=19.5975,
        p_l=460.894,
        rho_r=5.99242,
        u_r=-6.19633,
        p_r=46.0950,
        t_end=0.035,
        x0=0.4,
    ),
    5: ToroTest(
        number=5,
        name="Near-vacuum",
        rho_l=1.0,
        u_l=-19.59745,
        p_l=1000.0,
        rho_r=1.0,
        u_r=-19.59745,
        p_r=0.01,
        t_end=0.012,
        x0=0.8,
    ),
    6: ToroTest(
        number=6,
        name="Stationary contact",
        rho_l=1.4,
        u_l=0.0,
        p_l=1.0,
        rho_r=1.0,
        u_r=0.0,
        p_r=1.0,
        t_end=2.0,
    ),
}
