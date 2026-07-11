# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Load solver configuration from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from euler.riemann import DEFAULT_DX, DEFAULT_X0, Statistic

SolverMode = Literal["classical", "quantum"]


@dataclass(frozen=True)
class DomainConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    x0: float = DEFAULT_X0
    dx: float = DEFAULT_DX
    nx: int | None = None


@dataclass(frozen=True)
class ClassicalState:
    rho: float
    u: float
    p: float


@dataclass(frozen=True)
class QuantumState:
    rho: float
    u: float
    theta: float


@dataclass(frozen=True)
class ClassicalConfig:
    mode: Literal["classical"]
    left: ClassicalState
    right: ClassicalState
    t_end: float
    domain: DomainConfig
    gamma: float | None = None
    n: float | None = None
    output: str | None = None
    format: Literal["csv", "json"] = "csv"
    columns: str | None = None


@dataclass(frozen=True)
class QuantumConfig:
    mode: Literal["quantum"]
    left: QuantumState
    right: QuantumState
    t_end: float
    domain: DomainConfig
    n: float
    h: float
    statistic: Statistic = "FD"
    all_statistics: bool = False
    output: str | None = None
    format: Literal["csv", "json"] = "csv"
    columns: str | None = None


SolverConfig = ClassicalConfig | QuantumConfig


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        msg = f'"{key}" must be an object.'
        raise ValueError(msg)
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if not isinstance(value, (int, float)):
        msg = f'"{key}" must be a number.'
        raise ValueError(msg)
    return float(value)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    return _require_float(data, key)


def _parse_domain(data: dict[str, Any]) -> DomainConfig:
    if "domain" not in data:
        return DomainConfig()
    domain = _require_mapping(data, "domain")
    nx = domain.get("nx")
    if nx is not None and not isinstance(nx, int):
        msg = '"domain.nx" must be an integer.'
        raise ValueError(msg)
    return DomainConfig(
        x_min=_require_float(domain, "x_min") if "x_min" in domain else 0.0,
        x_max=_require_float(domain, "x_max") if "x_max" in domain else 1.0,
        x0=_require_float(domain, "x0") if "x0" in domain else DEFAULT_X0,
        dx=_require_float(domain, "dx") if "dx" in domain else DEFAULT_DX,
        nx=nx,
    )


def load_config(path: Path) -> SolverConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "Config root must be a JSON object."
        raise ValueError(msg)

    mode = raw.get("mode")
    if mode not in {"classical", "quantum"}:
        msg = '"mode" must be "classical" or "quantum".'
        raise ValueError(msg)

    left = _require_mapping(raw, "left")
    right = _require_mapping(raw, "right")
    t_end = _require_float(raw, "t_end")
    domain = _parse_domain(raw)

    output_format = raw.get("format", "csv")
    if output_format not in {"csv", "json"}:
        msg = '"format" must be "csv" or "json".'
        raise ValueError(msg)

    output = raw.get("output")
    if output is not None and not isinstance(output, str):
        msg = '"output" must be a string path.'
        raise ValueError(msg)

    columns = raw.get("columns")
    if columns is not None and not isinstance(columns, str):
        msg = '"columns" must be a comma-separated string.'
        raise ValueError(msg)

    if mode == "classical":
        gamma = _optional_float(raw, "gamma")
        n = _optional_float(raw, "n")
        if gamma is not None and n is not None:
            msg = 'Provide either "gamma" or "n", not both.'
            raise ValueError(msg)
        return ClassicalConfig(
            mode="classical",
            left=ClassicalState(
                rho=_require_float(left, "rho"),
                u=_require_float(left, "u"),
                p=_require_float(left, "p"),
            ),
            right=ClassicalState(
                rho=_require_float(right, "rho"),
                u=_require_float(right, "u"),
                p=_require_float(right, "p"),
            ),
            t_end=t_end,
            domain=domain,
            gamma=gamma,
            n=n,
            output=output,
            format=output_format,
            columns=columns,
        )

    statistic = raw.get("statistic", "FD")
    if statistic not in {"FD", "BE", "MB"}:
        msg = '"statistic" must be "FD", "BE", or "MB".'
        raise ValueError(msg)

    all_statistics = raw.get("all_statistics", False)
    if not isinstance(all_statistics, bool):
        msg = '"all_statistics" must be a boolean.'
        raise ValueError(msg)

    return QuantumConfig(
        mode="quantum",
        left=QuantumState(
            rho=_require_float(left, "rho"),
            u=_require_float(left, "u"),
            theta=_require_float(left, "theta"),
        ),
        right=QuantumState(
            rho=_require_float(right, "rho"),
            u=_require_float(right, "u"),
            theta=_require_float(right, "theta"),
        ),
        t_end=t_end,
        domain=domain,
        n=_require_float(raw, "n"),
        h=_require_float(raw, "h"),
        statistic=statistic,
        all_statistics=all_statistics,
        output=output,
        format=output_format,
        columns=columns,
    )
