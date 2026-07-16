# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Published quantum Euler benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

from ideal_gases.riemann import DEFAULT_DX, DEFAULT_X0


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
    h_be: float | None = None
    x0: float = DEFAULT_X0
    x_min: float = 0.0
    x_max: float = 1.0
    dx: float = DEFAULT_DX


QUANTUM_EXAMPLES: dict[int, QuantumExample] = {
    1: QuantumExample(
        number=1,
        name="Sod shock tube (n = 5, h = 1)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,  # Polytropic gas
        h=1.0,  # classical regime
    ),
    2: QuantumExample(
        number=2,
        name="Sod shock tube (n = 5, h = 2)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,  # Polytropic gas
        h=2.0,  # away from classical regime
    ),
    3: QuantumExample(
        number=3,
        name="Sod shock tube (n = 5, h_be = 2.65, h_fd = 6)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1 / 0.125,
        t_end=0.1,
        n=5.0,  # Polytropic gas
        h=1.0,  # classical regime
        h_be=2.65,  # degenerate regime
        h_fd=6.0,  # degenerate regime
    ),
    4: QuantumExample(
        number=4,
        name="Hu and Jin (classical gas vs bose degenerate)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,  # Monoatomic gas
        h=0.2,  # MB and FD correspond to classical regime
        h_be=3.3,  # corresponds to zl = 0.9906, zr = 0.9611
    ),
    5: QuantumExample(
        number=5,
        name="Hu and Jin (classical gas vs fermi degenerate)",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=0.18,
        n=3.0,  # Monoatomic gas
        h=0.2,  # MB and BE correspond to classical regime
        h_fd=6.0,  # corresponds to zl = 901.2840, zr = 459.5218
    ),
    6: QuantumExample(
        number=6,
        name="Yang and Shi (degenerate regime)",
        rho_l=3.086455,
        u_l=0.0,
        t_l=1.0,
        rho_r=3.084272,
        u_r=0.0,
        t_r=1.039,
        t_end=0.10,
        n=2.0,  # 2-D gas
        h=3.935,  # This corresponds to zl = 2000, zr = 1500
    ),
    7: QuantumExample(
        number=7,
        name="Filbet, Hu and Jing (2010), classical",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,  # 2-D gas
        h=0.1,  # Theta_0 = h^2 = 0.01 in the paper
    ),
    8: QuantumExample(
        number=8,
        name="Filbet, Hu and Jing (2010), quantum",
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.20,
        n=2.0,  # 2-D gas
        h=3.0,  # Theta_0 = h^2 = 9.0 in the paper
    ),
}
