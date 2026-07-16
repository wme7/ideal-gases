# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Command-line interface for QEuler solvers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ideal_gases import __version__
from ideal_gases.cli.commands import (
    cmd_interactive_classical,
    cmd_interactive_quantum,
    cmd_list,
    cmd_plot_classical,
    cmd_plot_quantum,
    cmd_plot_quantum_example,
    cmd_plot_run,
    cmd_plot_toro,
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


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--figure",
        type=Path,
        default=None,
        help="Output PNG path or stem for figure(s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--layout",
        choices=("panels", "comparison", "both"),
        default="both",
        help="All-statistics layout (default: both).",
    )


def _add_classical_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="JSON config file.")
    parser.add_argument("--rho-l", type=float, default=None)
    parser.add_argument("--u-l", type=float, default=None)
    parser.add_argument("--p-l", type=float, default=None)
    parser.add_argument("--rho-r", type=float, default=None)
    parser.add_argument("--u-r", type=float, default=None)
    parser.add_argument("--p-r", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n", type=float, default=None, help="DoF; gamma = (n+2)/n.")


def _add_quantum_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="JSON config file.")
    parser.add_argument("--rho-l", type=float, default=None)
    parser.add_argument("--u-l", type=float, default=None)
    parser.add_argument("--t-l", type=float, default=None, help="Left temperature theta.")
    parser.add_argument("--rho-r", type=float, default=None)
    parser.add_argument("--u-r", type=float, default=None)
    parser.add_argument("--t-r", type=float, default=None, help="Right temperature theta.")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--n", type=float, default=None, help="Spatial degrees of freedom.")
    parser.add_argument("--h", type=float, default=None, help="Planck constant (model parameter).")
    parser.add_argument(
        "--statistic",
        choices=("FD", "BE", "MB"),
        default=None,
        help="Quantum statistic (default: FD).",
    )
    parser.add_argument(
        "--all-statistics",
        action="store_true",
        help="Plot or export FD, MB, and BE together.",
    )


def _add_interactive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--figure",
        type=Path,
        default=None,
        help="Default PNG path for the Save button.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler",
        description="Exact classical and quantum Euler Riemann solvers.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
    _add_classical_solver_args(classical)
    _add_domain_args(classical)
    _add_output_args(classical)

    quantum = solve_sub.add_parser("quantum", help="Quantum FD/BE/MB solver.")
    _add_quantum_solver_args(quantum)
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

    plot_parser = subparsers.add_parser("plot", help="Solve and visualize results.")
    plot_sub = plot_parser.add_subparsers(dest="plot_target", required=True)

    plot_classical = plot_sub.add_parser("classical", help="Plot a classical solution.")
    _add_classical_solver_args(plot_classical)
    _add_domain_args(plot_classical)
    _add_plot_args(plot_classical)
    _add_output_args(plot_classical)

    plot_quantum = plot_sub.add_parser("quantum", help="Plot a quantum solution.")
    _add_quantum_solver_args(plot_quantum)
    _add_domain_args(plot_quantum)
    _add_plot_args(plot_quantum)
    _add_output_args(plot_quantum)

    plot_run = plot_sub.add_parser("run", help="Plot from a JSON config file.")
    plot_run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON configuration file.",
    )
    _add_domain_args(plot_run)
    _add_plot_args(plot_run)
    _add_output_args(plot_run)

    plot_toro = plot_sub.add_parser("toro", help="Plot a Toro classical test.")
    plot_toro.add_argument("number", type=int, choices=range(1, 7))
    _add_domain_args(plot_toro)
    _add_plot_args(plot_toro)
    _add_output_args(plot_toro)

    plot_quantum_example = plot_sub.add_parser(
        "quantum-example",
        help="Plot a published quantum Euler benchmark case.",
    )
    plot_quantum_example.add_argument("number", type=int, choices=range(1, 9))
    plot_quantum_example.add_argument(
        "--statistic",
        choices=("FD", "BE", "MB"),
        default=None,
        help="Quantum statistic (default: FD).",
    )
    plot_quantum_example.add_argument(
        "--all-statistics",
        action="store_true",
        help="Plot FD, MB, and BE comparison figures.",
    )
    _add_domain_args(plot_quantum_example)
    _add_plot_args(plot_quantum_example)
    _add_output_args(plot_quantum_example)

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Launch an interactive matplotlib Riemann explorer.",
    )
    interactive_sub = interactive_parser.add_subparsers(dest="interactive_target", required=True)

    interactive_classical = interactive_sub.add_parser(
        "classical",
        help="Interactive classical ideal-gas explorer.",
    )
    _add_classical_solver_args(interactive_classical)
    _add_domain_args(interactive_classical)
    _add_interactive_args(interactive_classical)

    interactive_quantum = interactive_sub.add_parser(
        "quantum",
        help="Interactive quantum FD/BE/MB comparison explorer.",
    )
    _add_quantum_solver_args(interactive_quantum)
    _add_domain_args(interactive_quantum)
    _add_interactive_args(interactive_quantum)

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

    if args.command == "interactive":
        if args.interactive_target == "classical":
            return cmd_interactive_classical(args)
        return cmd_interactive_quantum(args)

    if args.command == "plot":
        if args.plot_target == "classical":
            return cmd_plot_classical(args)
        if args.plot_target == "quantum":
            return cmd_plot_quantum(args)
        if args.plot_target == "run":
            return cmd_plot_run(args)
        if args.plot_target == "toro":
            return cmd_plot_toro(args)
        if args.plot_target == "quantum-example":
            return cmd_plot_quantum_example(args)
        parser.error(f"Unknown plot target: {args.plot_target}")
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
