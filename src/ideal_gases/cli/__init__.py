# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Command-line interface for QEuler solvers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ideal_gases.cli.commands import (
    cmd_list,
    cmd_quantum_example,
    cmd_run_config,
    cmd_solve_classical,
    cmd_solve_quantum,
    cmd_toro,
)


def _add_domain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x-min", type=float, default=None, help="Domain start.")
    parser.add_argument("--x-max", type=float, default=None, help="Domain end.")
    parser.add_argument("--x0", type=float, default=None, help="Discontinuity location.")
    parser.add_argument("--dx", type=float, default=None, help="Grid spacing.")
    parser.add_argument("--nx", type=int, default=None, help="Number of grid points.")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (.csv or .json).",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default=None,
        help="Output format (default: csv, or inferred from file extension).",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated output columns.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler",
        description="Exact classical and quantum Euler Riemann solvers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List preset test problems.")
    list_parser.add_argument(
        "--toro",
        action="store_true",
        help="List Toro classical shock-tube tests.",
    )
    list_parser.add_argument(
        "--quantum",
        action="store_true",
        help="List published quantum Euler benchmark cases.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Solve a problem defined in a JSON config file.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON configuration file.",
    )
    _add_output_args(run_parser)
    _add_domain_args(run_parser)

    solve_parser = subparsers.add_parser("solve", help="Solve a custom Riemann problem.")
    solve_sub = solve_parser.add_subparsers(dest="solver", required=True)

    classical = solve_sub.add_parser("classical", help="Classical ideal-gas solver.")
    classical.add_argument("--config", type=Path, default=None, help="JSON config file.")
    classical.add_argument("--rho-l", type=float, default=None)
    classical.add_argument("--u-l", type=float, default=None)
    classical.add_argument("--p-l", type=float, default=None)
    classical.add_argument("--rho-r", type=float, default=None)
    classical.add_argument("--u-r", type=float, default=None)
    classical.add_argument("--p-r", type=float, default=None)
    classical.add_argument("--t-end", type=float, default=None)
    classical.add_argument("--gamma", type=float, default=None)
    classical.add_argument("--n", type=float, default=None, help="DoF; gamma = (n+2)/n.")
    _add_domain_args(classical)
    _add_output_args(classical)

    quantum = solve_sub.add_parser("quantum", help="Quantum FD/BE/MB solver.")
    quantum.add_argument("--config", type=Path, default=None, help="JSON config file.")
    quantum.add_argument("--rho-l", type=float, default=None)
    quantum.add_argument("--u-l", type=float, default=None)
    quantum.add_argument("--t-l", type=float, default=None, help="Left temperature theta.")
    quantum.add_argument("--rho-r", type=float, default=None)
    quantum.add_argument("--u-r", type=float, default=None)
    quantum.add_argument("--t-r", type=float, default=None, help="Right temperature theta.")
    quantum.add_argument("--t-end", type=float, default=None)
    quantum.add_argument("--n", type=float, default=None, help="Spatial degrees of freedom.")
    quantum.add_argument("--h", type=float, default=None, help="Planck constant (model parameter).")
    quantum.add_argument(
        "--statistic",
        choices=("FD", "BE", "MB"),
        default=None,
        help="Quantum statistic (default: FD).",
    )
    quantum.add_argument(
        "--all-statistics",
        action="store_true",
        help="Write separate output files for FD, MB, and BE.",
    )
    _add_domain_args(quantum)
    _add_output_args(quantum)

    toro = subparsers.add_parser("toro", help="Run a Toro (1999) classical test.")
    toro.add_argument("number", type=int, choices=range(1, 7))
    _add_domain_args(toro)
    _add_output_args(toro)

    quantum_example = subparsers.add_parser(
        "quantum-example",
        help="Run a published quantum Euler benchmark case.",
    )
    quantum_example.add_argument("number", type=int, choices=range(1, 9))
    quantum_example.add_argument(
        "--statistic",
        choices=("FD", "BE", "MB"),
        default=None,
        help="Quantum statistic (default: FD).",
    )
    quantum_example.add_argument(
        "--all-statistics",
        action="store_true",
        help="Write separate output files for FD, MB, and BE.",
    )
    _add_domain_args(quantum_example)
    _add_output_args(quantum_example)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list(args)

    if args.command == "run":
        return cmd_run_config(args)

    if args.command == "solve":
        if args.solver == "classical":
            return cmd_solve_classical(args)
        return cmd_solve_quantum(args)

    if args.command == "toro":
        return cmd_toro(args)

    if args.command == "quantum-example":
        return cmd_quantum_example(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
