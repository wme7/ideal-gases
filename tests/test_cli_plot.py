# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

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


def test_plot_run_classical_config(tmp_path: Path) -> None:
    config_path = tmp_path / "classical.json"
    figure = tmp_path / "classical.png"
    output = tmp_path / "classical.csv"
    config_path.write_text(
        json.dumps(
            {
                "mode": "classical",
                "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
                "t_end": 0.25,
                "gamma": 1.4,
                "domain": {"nx": 21},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "plot",
            "run",
            "--config",
            str(config_path),
            "-f",
            str(figure),
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert figure.is_file()
    assert figure.stat().st_size > 0
    assert "# solver=classical" in output.read_text(encoding="utf-8")


def test_plot_run_quantum_config(tmp_path: Path) -> None:
    config_path = tmp_path / "quantum.json"
    figure = tmp_path / "quantum.png"
    output = tmp_path / "quantum.csv"
    config_path.write_text(
        json.dumps(
            {
                "mode": "quantum",
                "left": {"rho": 1.0, "u": 0.0, "theta": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "theta": 0.25},
                "t_end": 0.2,
                "n": 2.0,
                "h": 0.1,
                "statistic": "MB",
                "domain": {"nx": 21},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "plot",
            "run",
            "--config",
            str(config_path),
            "-f",
            str(figure),
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert figure.is_file()
    text = output.read_text(encoding="utf-8")
    assert "# solver=quantum" in text
    assert "# statistic=MB" in text


def test_plot_run_quantum_all_statistics_config(tmp_path: Path) -> None:
    config_path = tmp_path / "quantum_all.json"
    figure = tmp_path / "qeuler"
    output = tmp_path / "qeuler"
    config_path.write_text(
        json.dumps(
            {
                "mode": "quantum",
                "left": {"rho": 1.0, "u": 0.0, "theta": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "theta": 0.25},
                "t_end": 0.2,
                "n": 2.0,
                "h": 0.1,
                "all_statistics": True,
                "domain": {"nx": 21},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "plot",
            "run",
            "--config",
            str(config_path),
            "-f",
            str(figure),
            "--layout",
            "both",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "qeuler_panels.png").is_file()
    assert (tmp_path / "qeuler_comparison.png").is_file()
    for statistic in ("FD", "MB", "BE"):
        assert (tmp_path / f"qeuler_{statistic}.csv").exists()


def test_plot_quantum_single_statistic(tmp_path: Path) -> None:
    figure = tmp_path / "fd.png"
    output = tmp_path / "fd.csv"
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
            "--statistic",
            "FD",
            "--nx",
            "21",
            "-f",
            str(figure),
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert figure.is_file()
    assert "# statistic=FD" in output.read_text(encoding="utf-8")


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
