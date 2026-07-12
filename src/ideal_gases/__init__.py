# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Classical and quantum Euler solvers."""

from ideal_gases.polylog import polylog
from ideal_gases.riemann import (
    RiemannResult,
    adiabatic_index,
    classical_gas,
    quantum_gas,
)

__all__ = [
    "RiemannResult",
    "adiabatic_index",
    "polylog",
    "classical_gas",
    "quantum_gas",
]
