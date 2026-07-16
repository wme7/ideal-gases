# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Export Riemann solution profiles to CSV or JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ideal_gases.riemann import RiemannResult

OutputFormat = Literal["csv", "json"]

CLASSICAL_COLUMNS = ("x", "rho", "ux", "p", "e", "mach", "entropy")
QUANTUM_COLUMNS = ("x", "rho", "ux", "p", "e", "theta", "z", "mach", "entropy")


def resolve_output_format(
    explicit: OutputFormat | None,
    output: Path,
    *,
    config_format: OutputFormat | None = None,
) -> OutputFormat:
    if explicit is not None:
        return explicit
    if output.suffix.lower() == ".json":
        return "json"
    if config_format is not None:
        return config_format
    return "csv"


_COLUMN_ATTRS = {
    "x": "x",
    "rho": "rho",
    "ux": "ux",
    "p": "p",
    "e": "e",
    "theta": "t",
    "z": "z",
    "mach": "mach",
    "entropy": "entropy",
}


def resolve_columns(*, quantum: bool, columns: str | None) -> tuple[str, ...]:
    default = QUANTUM_COLUMNS if quantum else CLASSICAL_COLUMNS
    if columns is None:
        return default
    requested = tuple(col.strip() for col in columns.split(",") if col.strip())
    valid = set(_COLUMN_ATTRS)
    unknown = [col for col in requested if col not in valid]
    if unknown:
        msg = f"Unknown column(s): {', '.join(unknown)}. Valid: {', '.join(valid)}."
        raise ValueError(msg)
    return requested


def _column_arrays(
    result: RiemannResult, columns: tuple[str, ...]
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    for col in columns:
        data[col] = getattr(result, _COLUMN_ATTRS[col])
    return data


def _format_metadata(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, float):
            lines.append(f"# {key}={value:.12g}")
        else:
            lines.append(f"# {key}={value}")
    return lines


def result_to_csv(
    result: RiemannResult,
    path: Path,
    *,
    metadata: dict[str, Any],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _column_arrays(result, columns)
    n_points = len(result.x)

    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in _format_metadata(metadata):
            handle.write(f"{line}\n")
        handle.write(f"# n_points={n_points}\n")

        writer = csv.writer(handle)
        writer.writerow(columns)
        for row_idx in range(n_points):
            writer.writerow(f"{arrays[col][row_idx]:.12g}" for col in columns)


def result_to_json(
    result: RiemannResult,
    path: Path,
    *,
    metadata: dict[str, Any],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _column_arrays(result, columns)
    payload: dict[str, Any] = {
        "metadata": metadata,
        "columns": list(columns),
        "data": {col: arrays[col].tolist() for col in columns},
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_result(
    result: RiemannResult,
    path: Path,
    *,
    metadata: dict[str, Any],
    columns: tuple[str, ...],
    output_format: OutputFormat,
) -> None:
    if output_format == "csv":
        result_to_csv(result, path, metadata=metadata, columns=columns)
        return
    result_to_json(result, path, metadata=metadata, columns=columns)


def output_path_for_statistic(
    base: Path, statistic: str, output_format: OutputFormat
) -> Path:
    suffix = ".json" if output_format == "json" else ".csv"
    return base.parent / f"{base.stem}_{statistic}{suffix}"
