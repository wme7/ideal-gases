"""Classical and quantum Euler solvers."""

from euler.polylog import polylog
from euler.riemann import (
    RiemannResult,
    adiabatic_index,
    euler_gas,
    quantum_euler_gas,
)

__all__ = [
    "RiemannResult",
    "adiabatic_index",
    "euler_gas",
    "polylog",
    "quantum_euler_gas",
]
