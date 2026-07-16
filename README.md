# Classical and Quantum Ideal Gases

Exact Riemann solvers for classical and quantum Euler gases, with a fast polylogarithm kernel used in the quantum equation of state.

This repository ports the MATLAB reference implementation (Diaz, 2014) to Python. The C++ polylog core and Toro exact Riemann solver support Fermi–Dirac (FD), Bose–Einstein (BE), and Maxwell–Boltzmann (MB) statistics.

## Requirements

- Python 3.11+
- A C++17 compiler (for the pybind11 extension)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone <repo-url>
cd QEuler
uv sync
```

This installs the `ideal-gases` distribution (import name `ideal_gases`) in editable mode and builds the `_polylog` C++ extension via scikit-build-core.

To include dev tools (pytest, matplotlib, mpmath, ruff):

```bash
uv sync --group dev
```

## Project layout

```
cpp/          C++ polylog library and pybind11 bindings
matlab/       Original MATLAB reference solvers and plotting scripts
scripts/      Python utilities (e.g. solution plots)
src/ideal_gases/    Python package
tests/        pytest suite
```

## Quick start

### Classical Sod shock tube

```python
import numpy as np
from ideal_gases import classical_gas

x = np.linspace(0.0, 1.0, 101)
result = classical_gas(
    rho_l=1.0,
    u_l=0.0,
    p_l=1.0,
    rho_r=0.125,
    u_r=0.0,
    p_r=0.1,
    t_end=0.2,
    gamma=1.4,
    x=x,
    x0=0.5,
)
```

### Quantum Euler (FD / BE / MB)

Left and right states are given in terms of density `rho`, velocity `u`, and temperature `theta` (written `t` in the API). The solver converts these to effective pressures via the quantum EOS, then applies the Toro exact Riemann solver.

```python
import numpy as np
from ideal_gases import quantum_gas

x = np.linspace(0.0, 1.0, 101)
result = quantum_gas(
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

This results in a `RiemannResult` object with the following fields: `x`, `rho`, `ux`, `p`, `e`, `z` (fugacity), `t` (temperature), `mach`, `entropy`.

Graphically, the results for the three statistics can be plotted as follows:

![Quantum Euler](./figures/quantum_sod_shock_tube_gamma2.png)

In the classical limit, MB statistics with `h → 0` recover the ideal-gas behaviour (pressures `p = rho * theta`).

### Polylogarithm

```python
import numpy as np
from ideal_gases import polylog

polylog(2, 0.5)                         # scalar
polylog(1.5, np.linspace(0.2, 0.9, 50)) # array
```

Integer orders (including negative integers) use an analytic branch; non-integer orders use the Bhagat/Kuhnert approximations from `matlab/PolyLog.m`.

The plot below reproduces `matlab/PlotPolyLog.m` (orders from \(-7/2\) to \(7/2\)):

![Polylogarithms](./figures/polylogarithms.png)

Generate it with:

```bash
uv run python scripts/plot_polylogarithm.py --output figures/polylogarithms.png
```

## Command-line interface

After `uv sync`, the `euler` command exports exact solution profiles to CSV or JSON for use in other languages and tools.

### Classical Sod shock tube

```bash
euler solve classical \
  --rho-l 1 --u-l 0 --p-l 1 \
  --rho-r 0.125 --u-r 0 --p-r 0.1 \
  --t-end 0.25 --gamma 1.4 \
  --nx 101 -o sod.csv
```

### Quantum Euler

```bash
euler solve quantum \
  --rho-l 1 --u-l 0 --t-l 1 \
  --rho-r 0.125 --u-r 0 --t-r 0.25 \
  --t-end 0.20 --n 2 --h 0.1 --statistic FD \
  -o euler_fd.csv
```

Write separate files for FD, MB, and BE with `--all-statistics` (e.g. `euler_case7_FD.csv`, `euler_case7_MB.csv`, `euler_case7_BE.csv`):

```bash
euler solve quantum ... --all-statistics -o euler_case7
```

### Toro classical tests (1–6)

```bash
euler toro 1 -o toro_test1.csv
euler list --toro
```

### Quantum benchmark cases (1–8)

```bash
euler quantum-example 7 --all-statistics -o euler_eg7
euler list --quantum
```

### JSON config files

Define a problem in JSON and run it with `euler run` or pass `--config` to `euler solve`:

```bash
euler run --config case.json
euler solve classical --config case.json -o override.csv
```

Example `case.json`:

```json
{
  "mode": "quantum",
  "left": {"rho": 1.0, "u": 0.0, "theta": 1.0},
  "right": {"rho": 0.125, "u": 0.0, "theta": 0.25},
  "t_end": 0.20,
  "n": 2.0,
  "h": 0.1,
  "statistic": "FD",
  "all_statistics": true,
  "format": "json",
  "output": "euler_case7",
  "domain": {"x_min": 0.0, "x_max": 1.0, "x0": 0.5, "nx": 101}
}
```

Use `--format json` (or a `.json` output path) for JSON instead of CSV. CLI flags override values from the config file.

### Visualization

Plotting requires matplotlib:

```bash
pip install ideal-gases[plot]
# or, for local development:
uv sync --group dev
```

Save a classical Sod shock tube figure:

```bash
euler plot classical \
  --rho-l 1 --u-l 0 --p-l 1 \
  --rho-r 0.125 --u-r 0 --p-r 0.1 \
  --t-end 0.2 --gamma 1.4 --nx 101 \
  -f sod.png
```

Plot a single quantum statistic or compare FD/MB/BE:

```bash
euler plot quantum \
  --rho-l 1 --u-l 0 --t-l 1 \
  --rho-r 0.125 --u-r 0 --t-r 0.25 \
  --t-end 0.20 --n 2 --h 0.1 --statistic FD \
  -f qfd.png

euler plot quantum-example 7 --all-statistics -f eg7
```

With `--all-statistics`, `-f eg7` writes `eg7_panels.png` (3×6 grid) and `eg7_comparison.png` (overlay). Use `--layout panels|comparison|both` to select one or both (default: `both`). Add `--show` for an interactive window, or `-o` to export CSV/JSON in the same run.

### Interactive exploration

Launch matplotlib widget explorers to build custom Riemann problems with sliders, statistic toggles (quantum), and Save/Reset controls. Y-axis limits autoscale automatically on each update.

```bash
euler interactive classical
euler interactive quantum
```

Seed the initial state from CLI flags or a JSON config (same fields as `euler solve`):

```bash
euler interactive classical --gamma 1.4 --t-end 0.5 --nx 101
euler interactive quantum --rho-l 2 --t-l 1.5 --n 3 --h 0.5
euler interactive classical --config case.json
```

Optional domain flags (`--x-min`, `--x-max`, `--x0`, `--nx`) default to an interactive Sod-tube layout (`x` in `[-10, 10]`, discontinuity at `x0=0`, `nx=1024`). Use `-f path.png` to set the **Save** button target; nothing is written until you click Save.

Equivalent script entry points:

```bash
uv run python scripts/interactive_classical_riemann.py
uv run python scripts/interactive_quantum_comparison.py
```

## Plotting benchmark solutions

`scripts/plot_quantum_euler_solutions.py` is a thin wrapper around `euler plot quantum-example`. It reproduces the workflow of `matlab/PlotQEuler.m`.

```bash
uv run python scripts/plot_quantum_euler_solutions.py --example 7
# equivalent:
euler plot quantum-example 7 --all-statistics -f QEuler_Eg7
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--example` | 7 | Benchmark case (1–8) |
| `--output-dir` | repo root | Directory for saved PNG figures |
| `--dx` | 0.002 | Grid spacing |
| `--x0` | 0.5 | Discontinuity location |
| `--show` | off | Open interactive plot windows |

Outputs per example (when using `-f QEuler_Eg{N}`):

- `QEuler_Eg{N}_panels.png` — FD/MB/BE panels for ρ, u, p, e, θ, z
- `QEuler_Eg{N}_comparison.png` — overlay of all three statistics

## Public API

```python
from ideal_gases import (
    RiemannResult,
    adiabatic_index,
    classical_gas,
    polylog,
    quantum_gas,
)
```

| Symbol | Role |
|--------|------|
| `polylog(n, z)` | Fast polylogarithm (scalar or NumPy array) |
| `adiabatic_index(n)` | Returns γ = (n + 2) / n |
| `classical_gas(...)` | Classical ideal-gas exact Riemann solver |
| `quantum_gas(...)` | Quantum EOS + Toro exact Riemann solver |
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

Copyright (c) 2026 Manuel A. Diaz
