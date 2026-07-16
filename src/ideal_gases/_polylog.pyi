# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Type stubs for the C++ polylogarithm extension module."""

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np

def polylog(n: float, z: float) -> float: ...
def polylog_1d(n: float, z: NDArray[np.float64]) -> NDArray[np.float64]: ...
