# QEuler

Exact Riemann solvers for classical and quantum Euler gases, with a fast polylogarithm kernel used in the quantum equation of state.

This repository ports the MATLAB reference implementation (Diaz, 2014) to Python. The C++ polylog core and Toro exact Riemann solver support Fermi–Dirac (FD), Bose–Einstein (BE), and Maxwell–Boltzmann (MB) statistics.

## Requirements

- Python 3.14+
- A C++17 compiler (for the pybind11 extension)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone <repo-url>
cd QEuler
uv sync
```

This installs the `euler` package in editable mode and builds the `_polylog` C++ extension via scikit-build-core.

To include dev tools (pytest, matplotlib, mpmath, ruff):

```bash
uv sync --group dev
```

## Project layout

```
cpp/          C++ polylog library and pybind11 bindings
matlab/       Original MATLAB reference solvers and plotting scripts
scripts/      Python utilities (e.g. solution plots)
src/euler/    Python package
tests/        pytest suite
```

## Quick start

### Classical Sod shock tube

```python
import numpy as np
from euler import adiabatic_index, euler_gas

x = np.linspace(0.0, 1.0, 101)
result = euler_gas(
    rho_l=1.0,
    u_l=0.0,
    p_l=1.0,
    rho_r=0.125,
    u_r=0.0,
    p_r=0.1,
    t_end=0.25,
    gamma=1.4,
    x=x,
    x0=0.5,
)

print(result.rho.min(), result.rho.max())
```

### Quantum Euler (FD / BE / MB)

Left and right states are given in terms of density `rho`, velocity `u`, and temperature `theta` (written `t` in the API). The solver converts these to effective pressures via the quantum EOS, then applies the Toro exact Riemann solver.

```python
import numpy as np
from euler import quantum_euler_gas

x = np.linspace(0.0, 1.0, 101)
result = quantum_euler_gas(
    rho_l=1.0,
    u_l=0.0,
    t_l=1.0,
    rho_r=0.125,
    u_r=0.0,
    t_r=0.25,
    t_end=0.20,
    n=2.0,          # spatial degrees of freedom; gamma = (n + 2) / n
    h=0.1,          # Planck constant (dimensionless parameter in the model)
    statistic="FD", # "FD", "BE", or "MB"
    x=x,
    x0=0.5,
)
```

`RiemannResult` fields: `x`, `rho`, `ux`, `p`, `e`, `z` (fugacity), `t` (temperature), `mach`, `entropy`.

In the classical limit, MB statistics with `h → 0` recover the ideal-gas behaviour (pressures `p = rho * theta`).

### Polylogarithm

```python
import numpy as np
from euler import polylog

polylog(2, 0.5)                         # scalar
polylog(1.5, np.linspace(0.2, 0.9, 50)) # array
```

Integer orders use an analytic branch; non-integer orders use the Bhagat/Kuhnert approximations from `matlab/PolyLog.m`.

## Plotting benchmark solutions

`scripts/plot_quantum_euler_solutions.py` reproduces the workflow of `matlab/PlotQEuler.m`. It solves eight published test cases and saves comparison figures.

```bash
uv run python scripts/plot_quantum_euler_solutions.py --example 7
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--example` | 7 | Benchmark case (1–8) |
| `--output-dir` | repo root | Where PNG files are saved |
| `--dx` | 0.002 | Grid spacing |
| `--x0` | 0.5 | Discontinuity location |
| `--show` | off | Open interactive plot windows |

Outputs per example:

- `QEuler_Eg{N}_AllPlots.png` — FD/MB/BE panels for ρ, u, p, e, θ, z
- `QEuler_Eg{N}_TogetherPlots.png` — overlay of all three statistics

## Public API

```python
from euler import (
    RiemannResult,
    adiabatic_index,
    euler_gas,
    polylog,
    quantum_euler_gas,
)
```

| Symbol | Role |
|--------|------|
| `polylog(n, z)` | Fast polylogarithm (scalar or NumPy array) |
| `adiabatic_index(n)` | Returns γ = (n + 2) / n |
| `euler_gas(...)` | Classical ideal-gas exact Riemann solver |
| `quantum_euler_gas(...)` | Quantum EOS + Toro exact Riemann solver |
| `RiemannResult` | Solution profiles on the spatial grid |

## Tests

```bash
uv run pytest
```

Polylog tests compare against `mpmath`; Riemann tests cover a Sod tube, MB/classical agreement, and FD finiteness.

### C++ unit tests (optional)

```bash
cmake -S cpp -B build/cpp -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_CPP_TESTS=ON
cmake --build build/cpp
ctest --test-dir build/cpp
```

## MATLAB reference

The `matlab/` directory keeps the original solvers (`QEulerExact.m`, `QEulerExactToro.m`, `PolyLog.m`) and plotting scripts. Python results can be checked against these cases via the plotting script above.

## License

MIT License. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2014 Manuel A. Diaz
