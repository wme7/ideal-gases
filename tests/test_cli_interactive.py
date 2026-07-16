# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import pytest

from ideal_gases.cli import main
from ideal_gases.cli.interactive.defaults import INTERACTIVE_GAMMA


@pytest.fixture(autouse=True)
def _mock_show() -> Iterator[None]:
    with patch("matplotlib.pyplot.show"):
        yield


def test_interactive_classical_runs_with_defaults() -> None:
    exit_code = main(["interactive", "classical", "--nx", "21"])
    assert exit_code == 0


def test_interactive_quantum_runs_with_defaults() -> None:
    exit_code = main(["interactive", "quantum", "--nx", "21"])
    assert exit_code == 0


def test_interactive_classical_seeds_gamma_from_cli() -> None:
    with patch(
        "ideal_gases.cli.interactive.classical.classical_gas",
        wraps=__import__(
            "ideal_gases.riemann", fromlist=["classical_gas"]
        ).classical_gas,
    ) as solve:
        exit_code = main(
            [
                "interactive",
                "classical",
                "--gamma",
                "1.4",
                "--nx",
                "21",
            ]
        )
    assert exit_code == 0
    assert solve.call_args.args[7] == pytest.approx(1.4)


def test_interactive_classical_uses_default_gamma_without_flags() -> None:
    with patch(
        "ideal_gases.cli.interactive.classical.classical_gas",
        wraps=__import__(
            "ideal_gases.riemann", fromlist=["classical_gas"]
        ).classical_gas,
    ) as solve:
        exit_code = main(["interactive", "classical", "--nx", "21"])
    assert exit_code == 0
    assert solve.call_args.args[7] == pytest.approx(INTERACTIVE_GAMMA)


def test_interactive_classical_loads_config(tmp_path: Path) -> None:
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "classical",
                "left": {"rho": 2.0, "u": 0.0, "p": 2.0},
                "right": {"rho": 0.25, "u": 0.0, "p": 0.2},
                "t_end": 0.5,
                "gamma": 1.4,
                "domain": {"x_min": -5.0, "x_max": 5.0, "x0": 0.0, "nx": 21},
            }
        ),
        encoding="utf-8",
    )
    with patch(
        "ideal_gases.cli.interactive.classical.classical_gas",
        wraps=__import__(
            "ideal_gases.riemann", fromlist=["classical_gas"]
        ).classical_gas,
    ) as solve:
        exit_code = main(
            [
                "interactive",
                "classical",
                "--config",
                str(config_path),
            ]
        )
    assert exit_code == 0
    assert solve.call_args.args[0] == pytest.approx(2.0)
    assert solve.call_args.kwargs["x0"] == pytest.approx(0.0)


def test_interactive_quantum_loads_config(tmp_path: Path) -> None:
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "quantum",
                "left": {"rho": 1.0, "u": 0.0, "theta": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "theta": 0.25},
                "t_end": 0.2,
                "n": 3.0,
                "h": 0.5,
                "domain": {"x_min": -5.0, "x_max": 5.0, "x0": 0.0, "nx": 21},
            }
        ),
        encoding="utf-8",
    )
    with patch(
        "ideal_gases.cli.interactive.quantum.quantum_gas",
        wraps=__import__("ideal_gases.riemann", fromlist=["quantum_gas"]).quantum_gas,
    ) as solve:
        exit_code = main(
            [
                "interactive",
                "quantum",
                "--config",
                str(config_path),
            ]
        )
    assert exit_code == 0
    assert solve.call_args.args[7] == pytest.approx(3.0)
    assert solve.call_args.args[8] == pytest.approx(0.5)


def test_interactive_missing_matplotlib_reports_install_hint() -> None:
    with patch(
        "ideal_gases.cli.interactive.classical._require_matplotlib",
        side_effect=RuntimeError(
            "Plotting requires matplotlib. Install with: pip install ideal-gases[plot]"
        ),
    ):
        exit_code = main(["interactive", "classical", "--nx", "21"])
    assert exit_code == 1
