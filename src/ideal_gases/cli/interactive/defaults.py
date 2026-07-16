# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Default problem settings for interactive Riemann explorers."""

from __future__ import annotations

from pathlib import Path

from ideal_gases.cli.config import ClassicalState, DomainConfig, QuantumState

INTERACTIVE_DOMAIN = DomainConfig(x_min=-10.0, x_max=10.0, x0=0.0, nx=1024)
INTERACTIVE_CLASSICAL_LEFT = ClassicalState(rho=1.0, u=0.0, p=1.0)
INTERACTIVE_CLASSICAL_RIGHT = ClassicalState(rho=0.125, u=0.0, p=0.1)
INTERACTIVE_QUANTUM_LEFT = QuantumState(rho=1.0, u=0.0, theta=1.0)
INTERACTIVE_QUANTUM_RIGHT = QuantumState(rho=0.125, u=0.0, theta=0.1)
INTERACTIVE_T_END = 1.0
INTERACTIVE_GAMMA = 5.0 / 3.0
INTERACTIVE_N = 2.0
INTERACTIVE_H = 1.0
DEFAULT_CLASSICAL_FIGURE_PATH = Path("classical-riemann-solver.png")
DEFAULT_QUANTUM_FIGURE_PATH = Path("riemann-solver.png")
