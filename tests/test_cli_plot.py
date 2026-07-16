# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import pytest

from ideal_gases.cli import main


def test_plot_classical_creates_png(tmp_path: Path) -> None:
    output = tmp_path / "sod.png"
    exit_code = main(
        [
            "plot",
            "classical",
            "--rho-l",
            "1",
            "--u-l",
            "0",
            "--p-l",
            "1",
            "--rho-r",
            "0.125",
            "--u-r",
            "0",
            "--p-r",
            "0.1",
            "--t-end",
            "0.25",
            "--gamma",
            "1.4",
            "--nx",
            "21",
            "-f",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_quantum_all_statistics_creates_figures(tmp_path: Path) -> None:
    stem = tmp_path / "eg7"
    exit_code = main(
        [
            "plot",
            "quantum",
            "--rho-l",
            "1",
            "--u-l",
            "0",
            "--t-l",
            "1",
            "--rho-r",
            "0.125",
            "--u-r",
            "0",
            "--t-r",
            "0.25",
            "--t-end",
            "0.20",
            "--n",
            "2",
            "--h",
            "0.1",
            "--all-statistics",
            "--nx",
            "21",
            "-f",
            str(stem),
        ]
    )
    assert exit_code == 0
    panels = tmp_path / "eg7_panels.png"
    comparison = tmp_path / "eg7_comparison.png"
    assert panels.is_file()
    assert comparison.is_file()
    assert panels.stat().st_size > 0
    assert comparison.stat().st_size > 0


def test_plot_quantum_example_layout_panels_only(tmp_path: Path) -> None:
    stem = tmp_path / "eg1"
    exit_code = main(
        [
            "plot",
            "quantum-example",
            "1",
            "--all-statistics",
            "--layout",
            "panels",
            "--nx",
            "21",
            "-f",
            str(stem),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "eg1_panels.png").is_file()
    assert not (tmp_path / "eg1_comparison.png").exists()


def test_plot_requires_figure_or_show() -> None:
    exit_code = main(
        [
            "plot",
            "classical",
            "--rho-l",
            "1",
            "--u-l",
            "0",
            "--p-l",
            "1",
            "--rho-r",
            "0.125",
            "--u-r",
            "0",
            "--p-r",
            "0.1",
            "--t-end",
            "0.25",
            "--gamma",
            "1.4",
            "--nx",
            "21",
        ]
    )
    assert exit_code == 1


def test_plot_missing_matplotlib_reports_install_hint() -> None:
    with patch(
        "ideal_gases.cli.plot._require_matplotlib",
        side_effect=RuntimeError(
            "Plotting requires matplotlib. Install with: pip install ideal-gases[plot]"
        ),
    ):
        exit_code = main(
            [
                "plot",
                "toro",
                "1",
                "-f",
                "out.png",
                "--nx",
                "21",
            ]
        )
    assert exit_code == 1
