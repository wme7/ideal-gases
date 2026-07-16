# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

import json
from pathlib import Path

import pytest

from ideal_gases import __version__
from ideal_gases.cli import main
from ideal_gases.cli.config import load_config
from ideal_gases.cli.export import resolve_columns, result_to_csv, result_to_json
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


def test_build_grid_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="x_max"):
        build_grid(x_min=1.0, x_max=0.0)


def test_build_grid_rejects_small_nx() -> None:
    with pytest.raises(ValueError, match="nx"):
        build_grid(x_min=0.0, x_max=1.0, nx=1)


def test_build_grid_rejects_nonpositive_dx() -> None:
    with pytest.raises(ValueError, match="dx"):
        build_grid(x_min=0.0, x_max=1.0, dx=0.0)


def test_build_grid_appends_x_max_when_dx_does_not_land() -> None:
    x = build_grid(x_min=0.0, x_max=1.0, dx=0.3)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)
    assert len(x) == 5


def test_resolve_columns_custom_and_unknown() -> None:
    assert resolve_columns(quantum=False, columns="x, rho, p") == ("x", "rho", "p")
    with pytest.raises(ValueError, match="Unknown column"):
        resolve_columns(quantum=True, columns="x,foo")


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


def test_cli_quantum_single_statistic(tmp_path: Path) -> None:
    output = tmp_path / "qeuler_fd.csv"
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
            "--statistic",
            "FD",
            "--columns",
            "x,rho,theta,z",
            "--nx",
            "11",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# statistic=FD" in text
    assert "x,rho,theta,z" in text


def test_cli_classical_uses_n_for_gamma(tmp_path: Path) -> None:
    output = tmp_path / "mono.csv"
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
            "--n",
            "3",
            "--nx",
            "11",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    assert "# gamma=1.66666666667" in output.read_text(encoding="utf-8")


def test_cli_quantum_example(tmp_path: Path) -> None:
    output = tmp_path / "eg1.csv"
    exit_code = main(
        ["quantum-example", "1", "--statistic", "MB", "--nx", "11", "-o", str(output)]
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# preset=quantum:1" in text
    assert "# statistic=MB" in text
    assert "Sod shock tube" in text


def test_cli_quantum_example_all_statistics(tmp_path: Path) -> None:
    output = tmp_path / "eg1"
    exit_code = main(
        [
            "quantum-example",
            "1",
            "--all-statistics",
            "--nx",
            "11",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    for statistic in ("FD", "MB", "BE"):
        path = tmp_path / f"eg1_{statistic}.csv"
        assert path.exists()
        assert f"# statistic={statistic}" in path.read_text(encoding="utf-8")


def test_resolve_stat_h_overrides() -> None:
    from ideal_gases.cli.commands import _resolve_stat_h

    assert _resolve_stat_h(0.2, "MB", h_fd=6.0, h_be=3.3) == 0.2
    assert _resolve_stat_h(0.2, "FD", h_fd=6.0, h_be=3.3) == 6.0
    assert _resolve_stat_h(0.2, "BE", h_fd=6.0, h_be=3.3) == 3.3
    assert _resolve_stat_h(0.2, "FD", h_fd=None, h_be=3.3) == 0.2
    assert _resolve_stat_h(0.2, "BE", h_fd=6.0, h_be=None) == 0.2


def test_cli_quantum_example_4_uses_h_be(tmp_path: Path) -> None:
    output = tmp_path / "eg4_be.csv"
    exit_code = main(
        [
            "quantum-example",
            "4",
            "--statistic",
            "BE",
            "--nx",
            "21",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# h=3.3" in text
    # Degenerate BE left state: z ≈ 0.9906 (preset comment)
    data_lines = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    header = data_lines[0].split(",")
    z_idx = header.index("z")
    z_left = float(data_lines[1].split(",")[z_idx])
    assert z_left == pytest.approx(0.9906, rel=1e-2)


def test_cli_quantum_example_5_uses_h_fd(tmp_path: Path) -> None:
    output = tmp_path / "eg5_fd.csv"
    exit_code = main(
        [
            "quantum-example",
            "5",
            "--statistic",
            "FD",
            "--nx",
            "21",
            "-o",
            str(output),
        ]
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# h=6" in text
    # Degenerate FD left state: z ≈ 901 (preset comment)
    data_lines = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    header = data_lines[0].split(",")
    z_idx = header.index("z")
    z_left = float(data_lines[1].split(",")[z_idx])
    assert z_left == pytest.approx(901.2840, rel=5e-2)


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


def test_cli_run_quantum_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "quantum.json"
    output = tmp_path / "out.csv"
    config_path.write_text(
        json.dumps(
            {
                "mode": "quantum",
                "left": {"rho": 1.0, "u": 0.0, "theta": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "theta": 0.25},
                "t_end": 0.2,
                "n": 2.0,
                "h": 0.1,
                "statistic": "FD",
                "output": str(output),
                "domain": {"nx": 11},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# solver=quantum" in text
    assert "# statistic=FD" in text


def test_cli_run_quantum_all_statistics_config(tmp_path: Path) -> None:
    config_path = tmp_path / "quantum_all.json"
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
                "output": str(output),
                "domain": {"nx": 11},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    for statistic in ("FD", "MB", "BE"):
        assert (tmp_path / f"qeuler_{statistic}.csv").exists()


def test_cli_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list", "--toro"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Toro classical tests:" in captured.out
    assert "Modified Sod" in captured.out


def test_cli_list_quantum_presets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list", "--quantum"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Quantum Euler benchmarks:" in captured.out
    assert "Sod shock tube" in captured.out
    assert "Toro classical tests:" not in captured.out


def test_load_config_validation_errors(tmp_path: Path) -> None:
    cases = [
        ("[]", "JSON object"),
        ('{"mode": "other"}', '"mode"'),
        (
            json.dumps(
                {
                    "mode": "classical",
                    "left": "bad",
                    "right": {"rho": 1, "u": 0, "p": 1},
                    "t_end": 0.1,
                }
            ),
            '"left"',
        ),
        (
            json.dumps(
                {
                    "mode": "classical",
                    "left": {"rho": "x", "u": 0, "p": 1},
                    "right": {"rho": 1, "u": 0, "p": 1},
                    "t_end": 0.1,
                }
            ),
            "number",
        ),
        (
            json.dumps(
                {
                    "mode": "classical",
                    "left": {"rho": 1, "u": 0, "p": 1},
                    "right": {"rho": 1, "u": 0, "p": 1},
                    "t_end": 0.1,
                    "format": "xml",
                }
            ),
            '"format"',
        ),
        (
            json.dumps(
                {
                    "mode": "classical",
                    "left": {"rho": 1, "u": 0, "p": 1},
                    "right": {"rho": 1, "u": 0, "p": 1},
                    "t_end": 0.1,
                    "gamma": 1.4,
                    "n": 3,
                }
            ),
            "either",
        ),
        (
            json.dumps(
                {
                    "mode": "quantum",
                    "left": {"rho": 1, "u": 0, "theta": 1},
                    "right": {"rho": 1, "u": 0, "theta": 1},
                    "t_end": 0.1,
                    "n": 2,
                    "h": 1,
                    "statistic": "XX",
                }
            ),
            '"statistic"',
        ),
        (
            json.dumps(
                {
                    "mode": "quantum",
                    "left": {"rho": 1, "u": 0, "theta": 1},
                    "right": {"rho": 1, "u": 0, "theta": 1},
                    "t_end": 0.1,
                    "n": 2,
                    "h": 1,
                    "all_statistics": "yes",
                }
            ),
            '"all_statistics"',
        ),
        (
            json.dumps(
                {
                    "mode": "classical",
                    "left": {"rho": 1, "u": 0, "p": 1},
                    "right": {"rho": 1, "u": 0, "p": 1},
                    "t_end": 0.1,
                    "domain": {"nx": 11.5},
                }
            ),
            "domain.nx",
        ),
    ]
    for payload, match in cases:
        path = tmp_path / "bad.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(path)


def test_cli_solve_missing_required_args() -> None:
    assert main(["solve", "classical", "--nx", "5", "-o", "out.csv"]) == 1
    assert main(["solve", "quantum", "--nx", "5", "-o", "out.csv"]) == 1


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
