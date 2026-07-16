# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

import json
from pathlib import Path

import pytest

from ideal_gases import __version__
from ideal_gases.cli import main
from ideal_gases.cli.config import load_config
from ideal_gases.cli.export import result_to_csv, result_to_json
from ideal_gases.cli.grid import build_grid, validate_x0
from ideal_gases.riemann import classical_gas


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"euler {__version__}"


def test_build_grid_with_dx() -> None:
    x = build_grid(x_min=0.0, x_max=1.0, dx=0.5)
    assert x.tolist() == [0.0, 0.5, 1.0]


def test_build_grid_with_nx() -> None:
    x = build_grid(x_min=0.0, x_max=1.0, nx=5)
    assert len(x) == 5
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


def test_validate_x0_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="x0"):
        validate_x0(x0=1.5, x_min=0.0, x_max=1.0)


def test_load_classical_config(tmp_path: Path) -> None:
    config_path = tmp_path / "case.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "classical",
                "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
                "t_end": 0.25,
                "gamma": 1.4,
                "output": "out.csv",
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.mode == "classical"
    assert config.left.rho == 1.0
    assert config.output == "out.csv"


def test_cli_classical_csv(tmp_path: Path) -> None:
    output = tmp_path / "sod.csv"
    exit_code = main(
        [
            "solve",
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
            "--nx",
            "21",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# solver=classical" in text
    assert "x,rho,ux,p,e,mach,entropy" in text
    assert text.count("\n") > 22


def test_cli_toro_preset(tmp_path: Path) -> None:
    output = tmp_path / "toro1.csv"
    exit_code = main(["toro", "1", "--nx", "11", "-o", str(output)])
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# preset=toro:1" in text
    assert "Modified Sod" in text or "# name=Modified Sod" in text


def test_cli_quantum_all_statistics(tmp_path: Path) -> None:
    output = tmp_path / "qeuler_case7"
    exit_code = main(
        [
            "solve",
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
            "11",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    for statistic in ("FD", "MB", "BE"):
        path = tmp_path / f"qeuler_case7_{statistic}.csv"
        assert path.exists()
        assert f"# statistic={statistic}" in path.read_text(encoding="utf-8")


def test_cli_run_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "case.json"
    output = tmp_path / "out.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "classical",
                "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "p": 0.1},
                "t_end": 0.25,
                "gamma": 1.4,
                "format": "json",
                "output": str(output),
                "domain": {"nx": 11},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["solver"] == "classical"
    assert payload["columns"] == ["x", "rho", "ux", "p", "e", "mach", "entropy"]
    assert len(payload["data"]["x"]) == 11


def test_cli_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list", "--toro"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Toro classical tests:" in captured.out
    assert "Modified Sod" in captured.out


def test_export_helpers_round_trip(tmp_path: Path) -> None:
    result = classical_gas(
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=0.25,
        gamma=1.4,
        x=build_grid(x_min=0.0, x_max=1.0, nx=5),
    )
    metadata = {"solver": "classical"}
    columns = ("x", "rho", "ux", "p")

    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    result_to_csv(result, csv_path, metadata=metadata, columns=columns)
    result_to_json(result, json_path, metadata=metadata, columns=columns)

    assert "x,rho,ux,p" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["data"]["rho"] == result.rho.tolist()
