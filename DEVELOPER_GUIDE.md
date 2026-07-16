# Developer guide

Contributor documentation for building, testing, linting, and releasing `ideal-gases`. End-user usage is in [README.md](README.md).

## Project layout

```
cpp/          C++ polylog library and pybind11 bindings
matlab/       Original MATLAB reference solvers and plotting scripts
scripts/      Python utilities (e.g. solution plots)
src/ideal_gases/    Python package
tests/        pytest suite
```

## Developer install

Requirements:

- Python 3.11+
- A C++17 compiler (for the pybind11 extension)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
git clone https://github.com/wme7/ideal-gases.git
cd ideal-gases
uv sync --group dev
```

This installs the `ideal-gases` distribution (import name `ideal_gases`) in editable mode and builds the `_polylog` C++ extension via scikit-build-core. The `dev` group includes pytest, matplotlib, mpmath, ruff, ty, and pre-commit.

### Pre-commit hooks

Install the git hooks once after syncing the `dev` group:

```bash
uv run pre-commit install
```

Hooks then run on each `git commit` (trailing whitespace, YAML checks, Ruff lint/format).

## Quality tooling

### Tests

```bash
uv run pytest
```

With coverage (as in CI):

```bash
uv run pytest --cov --cov-report=xml --cov-report=term-missing
```

Polylog tests compare against `mpmath`; Riemann tests cover a Sod tube, MB/classical agreement, and FD finiteness.

### Ruff (lint and format)

```bash
uv run ruff check .
uv run ruff format .
```

### Type checking

```bash
uv run ty check
```

### Pre-commit (all files)

```bash
uv run pre-commit run --all-files
```

### C++ unit tests (optional)

```bash
cmake -S cpp -B build/cpp -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_CPP_TESTS=ON
cmake --build build/cpp
ctest --test-dir build/cpp
```

## CI and releasing

GitHub Actions runs lint (pre-commit) and tests with coverage on Linux and macOS for every push/PR to `master` / `main`.

Releases build wheels (Linux + macOS) and an sdist, then upload them to **TestPyPI** and **PyPI** when you push a version tag:

1. One-time setup on [TestPyPI](https://test.pypi.org) and [PyPI](https://pypi.org): create accounts, then add a **Trusted Publisher** for this repo on each — owner `wme7`, repository `ideal-gases`, workflow `publish.yml`, environments `testpypi` and `pypi`.
2. In GitHub → Settings → Environments, create `testpypi` and `pypi` (optional: required reviewers on `pypi`).
3. Bump `version` in `pyproject.toml` so it matches the tag (e.g. `0.1.2` ↔ `v0.1.2`).
4. Tag and push:
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```
5. Approve the `pypi` environment in Actions if reviewers are required.
6. Install from PyPI:
   ```bash
   pip install ideal-gases
   ```
   Or from TestPyPI (dependencies still resolve from real PyPI):
   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ideal-gases
   ```

`workflow_dispatch` on the Publish workflow builds artifacts without uploading (useful dry run).

## MATLAB reference and figure scripts

The `matlab/` directory keeps the original solvers (`QEulerExact.m`, `QEulerExactToro.m`, `PolyLog.m`) and plotting scripts. Python results can be checked against these cases via the plotting scripts below.

### Quantum Euler benchmarks

`scripts/plot_quantum_euler_solutions.py` is a thin wrapper around `euler plot quantum-example`. It reproduces the workflow of `matlab/PlotQEuler.m`.

```bash
uv run python scripts/plot_quantum_euler_solutions.py --example 7
# equivalent:
euler plot quantum-example 7 --all-statistics -f QEuler_Eg7
```

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

### Polylogarithm figure

The plot below reproduces `matlab/PlotPolyLog.m` (orders from \(-7/2\) to \(7/2\)):

![Polylogarithms](./figures/polylogarithms.png)

Generate it with:

```bash
uv run python scripts/plot_polylogarithms.py --output figures/polylogarithms.png
```

### Classical Sod figure

```bash
uv run python scripts/plot_classical_sod_shock_tube.py
```
