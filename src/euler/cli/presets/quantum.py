# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Published quantum Euler benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

from euler.riemann import DEFAULT_DX, DEFAULT_X0


@dataclass(frozen=True)
class QuantumExample:
    number: int
    name: str
    rho_l: float
    u_l: float
    t_l: float
    rho_r: float
    u_r: float
    t_r: float
    t_end: float
    n: float
    h: float
    h_fd: float | None = None
    x0: float = DEFAULT_X0
    x_min: float = 0.0
    x_max: float = 1.0
    dx: float = DEFAULT_DX


QUANTUM_EXAMPLES: dict[int, QuantumExample] = {
    1: QuantumExample(
        number=1,
        name="Sod shock tube (classical regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=1.0,
    ),
    2: QuantumExample(
        number=2,
        name="Sod shock tube (quantum h = 2)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=2.0,
    ),
    3: QuantumExample(
        number=3,
        name="Sod shock tube (degenerate FD, h_FD = 6)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,
        h=2.65,
        h_fd=6.0,
    ),
    4: QuantumExample(
        number=4,
        name="Hu and Jin (classical regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,
        h=1.0,
    ),
    5: QuantumExample(
        number=5,
        name="Hu and Jin (degenerate regime)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,
        h=3.3,
        h_fd=6.0,
    ),
    6: QuantumExample(
        number=6,
        name="Yang and Shi (degenerate regime)",
        rho_l=3.086455,
        u_l=0.0,
        t_l=8.053324,
        rho_r=3.084272,
        u_r=0.0,
        t_r=8.067390,
        t_end=0.10,
        n=3.0,
        h=6.0,
    ),
    7: QuantumExample(
        number=7,
        name="Filbet and Jing (2010), classical",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,
        h=0.1,
    ),
    8: QuantumExample(
        number=8,
        name="Filbet and Jing (2010), quantum",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,
        h=3.0,
    ),
}
