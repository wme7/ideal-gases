# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Preset Riemann problem definitions."""

from ideal_gases.cli.presets.quantum import QUANTUM_EXAMPLES, QuantumExample
from ideal_gases.cli.presets.toro import TORO_TESTS, ToroTest

__all__ = [
    "QUANTUM_EXAMPLES",
    "QuantumExample",
    "TORO_TESTS",
    "ToroTest",
]
