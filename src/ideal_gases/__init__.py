# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

from importlib.metadata import requires, version

__version__ = version("ideal_gases")
__requires__ = requires("ideal_gases")

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
