#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Plot quantum Euler Riemann solutions for visual verification.

Thin wrapper around ``euler plot quantum-example``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ideal_gases.cli import main

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        type=int,
        default=7,
        choices=range(1, 9),
        help="Example case number (default: 7).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for saved PNG figures.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.002,
        help="Spatial grid spacing.",
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=0.5,
        help="Initial discontinuity location.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    return parser.parse_args()


def main_script() -> None:
    args = parse_args()
    figure_stem = args.output_dir / f"QEuler_Eg{args.example}"
    argv = [
        "plot",
        "quantum-example",
        str(args.example),
        "--all-statistics",
        "-f",
        str(figure_stem),
        "--dx",
        str(args.dx),
        "--x0",
        str(args.x0),
    ]
    if args.show:
        argv.append("--show")
    raise SystemExit(main(argv))


if __name__ == "__main__":
    main_script()
