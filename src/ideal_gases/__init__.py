# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Classical and quantum Euler solvers."""

from importlib.metadata import requires, version

from ideal_gases.classical_nsf import ClassicalNSFResult, classical_nsf
from ideal_gases.equilibrium import (
    G,
    equilibrium_moments,
    find_fugacity,
    find_moments,
    set_polylog_backend,
)
from ideal_gases.polylog import polylog
from ideal_gases.quantum_nsf import NSFResult, QuantumNSFResult, quantum_nsf
from ideal_gases.riemann import (
    RiemannResult,
    adiabatic_index,
    classical_gas,
    quantum_gas,
)

__version__ = version("ideal_gases")
__requires__ = requires("ideal_gases")

__all__ = [
    "G",
    "ClassicalNSFResult",
    "NSFResult",
    "QuantumNSFResult",
    "RiemannResult",
    "adiabatic_index",
    "classical_gas",
    "classical_nsf",
    "equilibrium_moments",
    "find_fugacity",
    "find_moments",
    "polylog",
    "quantum_gas",
    "quantum_nsf",
    "set_polylog_backend",
]
