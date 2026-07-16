# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""CLI command implementations."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

from ideal_gases.cli.config import (
    ClassicalConfig,
    ClassicalState,
    DomainConfig,
    QuantumConfig,
    QuantumState,
    load_config,
)
from ideal_gases.cli.export import (
    OutputFormat,
    output_path_for_statistic,
    resolve_columns,
    resolve_output_format,
    write_result,
)
from ideal_gases.cli.grid import build_grid, validate_x0
from ideal_gases.cli.interactive.classical import run_classical_interactive
from ideal_gases.cli.interactive.defaults import (
    DEFAULT_CLASSICAL_FIGURE_PATH,
    DEFAULT_QUANTUM_FIGURE_PATH,
    INTERACTIVE_CLASSICAL_LEFT,
    INTERACTIVE_CLASSICAL_RIGHT,
    INTERACTIVE_DOMAIN,
    INTERACTIVE_GAMMA,
    INTERACTIVE_H,
    INTERACTIVE_N,
    INTERACTIVE_QUANTUM_LEFT,
    INTERACTIVE_QUANTUM_RIGHT,
    INTERACTIVE_T_END,
)
from ideal_gases.cli.interactive.quantum import run_quantum_interactive
from ideal_gases.cli.plot import PlotLayout, plot_all_statistics, plot_single
from ideal_gases.cli.presets.quantum import QUANTUM_EXAMPLES
from ideal_gases.cli.presets.toro import TORO_TESTS
from ideal_gases.riemann import (
    RiemannResult,
    Statistic,
    adiabatic_index,
    classical_gas,
    quantum_gas,
)

STATISTICS: tuple[Statistic, ...] = ("FD", "MB", "BE")


def _resolve_gamma(
    *, gamma: float | None, n: float | None, default: float = 1.4
) -> float:
    if gamma is not None:
        return gamma
    if n is not None:
        return adiabatic_index(n)
    return default


def _resolve_domain(
    *,
    domain: DomainConfig,
    x_min: float | None,
    x_max: float | None,
    x0: float | None,
    dx: float | None,
    nx: int | None,
) -> DomainConfig:
    return DomainConfig(
        x_min=x_min if x_min is not None else domain.x_min,
        x_max=x_max if x_max is not None else domain.x_max,
        x0=x0 if x0 is not None else domain.x0,
        dx=dx if dx is not None else domain.dx,
        nx=nx if nx is not None else domain.nx,
    )


def _grid_from_domain(domain: DomainConfig):
    validate_x0(x0=domain.x0, x_min=domain.x_min, x_max=domain.x_max)
    return build_grid(
        x_min=domain.x_min,
        x_max=domain.x_max,
        x0=domain.x0,
        dx=domain.dx,
        nx=domain.nx,
    )


def _write_outputs(
    *,
    result: RiemannResult,
    output: Path,
    output_format: OutputFormat,
    metadata: dict[str, Any],
    columns: tuple[str, ...],
) -> None:
    write_result(
        result,
        output,
        metadata=metadata,
        columns=columns,
        output_format=output_format,
    )
    print(f"Wrote {len(result.x)} rows -> {output}", file=__import__("sys").stderr)


def _write_quantum_outputs(
    *,
    results: dict[str, RiemannResult],
    output: Path,
    output_format: OutputFormat,
    base_metadata: dict[str, Any],
    columns: tuple[str, ...],
) -> None:
    for statistic, result in results.items():
        path = output_path_for_statistic(output, statistic, output_format)
        metadata = {**base_metadata, "statistic": statistic}
        _write_outputs(
            result=result,
            output=path,
            output_format=output_format,
            metadata=metadata,
            columns=columns,
        )


def _solve_classical(
    *,
    left: ClassicalState,
    right: ClassicalState,
    t_end: float,
    domain: DomainConfig,
    gamma: float,
) -> RiemannResult:
    x = _grid_from_domain(domain)
    return classical_gas(
        rho_l=left.rho,
        u_l=left.u,
        p_l=left.p,
        rho_r=right.rho,
        u_r=right.u,
        p_r=right.p,
        t_end=t_end,
        gamma=gamma,
        x=x,
        x0=domain.x0,
        dx=domain.dx,
    )


def _resolve_stat_h(
    h: float,
    statistic: Statistic,
    *,
    h_fd: float | None = None,
    h_be: float | None = None,
) -> float:
    """Return per-statistic thermal scale, honoring FD/BE overrides when set."""
    if statistic == "FD" and h_fd is not None:
        return h_fd
    if statistic == "BE" and h_be is not None:
        return h_be
    return h


def _solve_quantum(
    *,
    left: QuantumState,
    right: QuantumState,
    t_end: float,
    domain: DomainConfig,
    n: float,
    h: float,
    statistic: Statistic,
) -> RiemannResult:
    x = _grid_from_domain(domain)
    return quantum_gas(
        rho_l=left.rho,
        u_l=left.u,
        t_l=left.theta,
        rho_r=right.rho,
        u_r=right.u,
        t_r=right.theta,
        t_end=t_end,
        n=n,
        h=h,
        statistic=statistic,
        x=x,
        x0=domain.x0,
        dx=domain.dx,
    )


def _solve_quantum_all_statistics(
    *,
    left: QuantumState,
    right: QuantumState,
    t_end: float,
    domain: DomainConfig,
    n: float,
    h: float,
    h_fd: float | None = None,
    h_be: float | None = None,
) -> dict[str, RiemannResult]:
    results: dict[str, RiemannResult] = {}
    for stat in STATISTICS:
        results[stat] = _solve_quantum(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            n=n,
            h=_resolve_stat_h(h, stat, h_fd=h_fd, h_be=h_be),
            statistic=stat,
        )
    return results


def _require_output(path: Path | None, config_output: str | None) -> Path:
    if path is not None:
        return path
    if config_output is not None:
        return Path(config_output)
    msg = 'Output path is required (-o/--output or config "output").'
    raise ValueError(msg)


def _optional_output(path: Path | None, config_output: str | None) -> Path | None:
    if path is not None:
        return path
    if config_output is not None:
        return Path(config_output)
    return None


def _require_figure(figure: Path | None, show: bool) -> Path | None:
    if figure is not None:
        return figure
    if show:
        return None
    msg = "Figure path (-f/--figure) or --show is required."
    raise ValueError(msg)


def _single_figure_path(figure: Path) -> Path:
    if figure.suffix.lower() == ".png":
        return figure
    return figure.with_suffix(".png")


def _print_saved(paths: Sequence[Path | None]) -> None:
    for path in paths:
        if path is not None:
            print(f"Saved {path}", file=sys.stderr)


def _classical_metadata(
    *,
    left: ClassicalState,
    right: ClassicalState,
    t_end: float,
    domain: DomainConfig,
    gamma: float,
    preset: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "solver": "classical",
        "gamma": gamma,
        "rho_l": left.rho,
        "u_l": left.u,
        "p_l": left.p,
        "rho_r": right.rho,
        "u_r": right.u,
        "p_r": right.p,
        "t_end": t_end,
        "x_min": domain.x_min,
        "x_max": domain.x_max,
        "x0": domain.x0,
        "dx": domain.dx,
    }
    if domain.nx is not None:
        metadata["nx"] = domain.nx
    if preset is not None:
        metadata["preset"] = preset
    return metadata


def _quantum_metadata(
    *,
    left: QuantumState,
    right: QuantumState,
    t_end: float,
    domain: DomainConfig,
    n: float,
    h: float,
    statistic: Statistic | None,
    preset: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "solver": "quantum",
        "n": n,
        "h": h,
        "rho_l": left.rho,
        "u_l": left.u,
        "theta_l": left.theta,
        "rho_r": right.rho,
        "u_r": right.u,
        "theta_r": right.theta,
        "t_end": t_end,
        "x_min": domain.x_min,
        "x_max": domain.x_max,
        "x0": domain.x0,
        "dx": domain.dx,
    }
    if statistic is not None:
        metadata["statistic"] = statistic
    if domain.nx is not None:
        metadata["nx"] = domain.nx
    if preset is not None:
        metadata["preset"] = preset
    return metadata


@overload
def _merge_classical_from_args(
    args: argparse.Namespace,
    config: ClassicalConfig | None,
    *,
    require_output: Literal[True] = True,
) -> tuple[
    ClassicalState, ClassicalState, float, DomainConfig, float, str | None, Path
]: ...


@overload
def _merge_classical_from_args(
    args: argparse.Namespace,
    config: ClassicalConfig | None,
    *,
    require_output: Literal[False],
) -> tuple[
    ClassicalState, ClassicalState, float, DomainConfig, float, str | None, Path | None
]: ...


def _merge_classical_from_args(
    args: argparse.Namespace,
    config: ClassicalConfig | None,
    *,
    require_output: bool = True,
) -> tuple[
    ClassicalState, ClassicalState, float, DomainConfig, float, str | None, Path | None
]:
    if config is None and args.t_end is None:
        msg = "--t-end is required."
        raise ValueError(msg)
    if config is None and any(
        v is None
        for v in (
            args.rho_l,
            args.u_l,
            args.p_l,
            args.rho_r,
            args.u_r,
            args.p_r,
        )
    ):
        msg = "Left and right states (--rho-l, --u-l, --p-l, --rho-r, --u-r, --p-r) are required."
        raise ValueError(msg)

    if config is None:
        left = ClassicalState(rho=args.rho_l, u=args.u_l, p=args.p_l)
        right = ClassicalState(rho=args.rho_r, u=args.u_r, p=args.p_r)
        t_end = args.t_end
    else:
        left = ClassicalState(
            rho=args.rho_l if args.rho_l is not None else config.left.rho,
            u=args.u_l if args.u_l is not None else config.left.u,
            p=args.p_l if args.p_l is not None else config.left.p,
        )
        right = ClassicalState(
            rho=args.rho_r if args.rho_r is not None else config.right.rho,
            u=args.u_r if args.u_r is not None else config.right.u,
            p=args.p_r if args.p_r is not None else config.right.p,
        )
        t_end = args.t_end if args.t_end is not None else config.t_end
    domain = _resolve_domain(
        domain=config.domain if config else DomainConfig(),
        x_min=args.x_min,
        x_max=args.x_max,
        x0=args.x0,
        dx=args.dx,
        nx=args.nx,
    )
    gamma = _resolve_gamma(
        gamma=args.gamma
        if args.gamma is not None
        else (config.gamma if config else None),
        n=args.n if args.n is not None else (config.n if config else None),
    )
    columns = (
        args.columns
        if args.columns is not None
        else (config.columns if config else None)
    )
    if require_output:
        output = _require_output(args.output, config.output if config else None)
    else:
        output = _optional_output(args.output, config.output if config else None)
    return left, right, t_end, domain, gamma, columns, output


@overload
def _merge_quantum_from_args(
    args: argparse.Namespace,
    config: QuantumConfig | None,
    *,
    require_output: Literal[True] = True,
) -> tuple[
    QuantumState,
    QuantumState,
    float,
    DomainConfig,
    float,
    float,
    Statistic,
    bool,
    str | None,
    Path,
]: ...


@overload
def _merge_quantum_from_args(
    args: argparse.Namespace,
    config: QuantumConfig | None,
    *,
    require_output: Literal[False],
) -> tuple[
    QuantumState,
    QuantumState,
    float,
    DomainConfig,
    float,
    float,
    Statistic,
    bool,
    str | None,
    Path | None,
]: ...


def _merge_quantum_from_args(
    args: argparse.Namespace,
    config: QuantumConfig | None,
    *,
    require_output: bool = True,
) -> tuple[
    QuantumState,
    QuantumState,
    float,
    DomainConfig,
    float,
    float,
    Statistic,
    bool,
    str | None,
    Path | None,
]:
    if config is None and args.t_end is None:
        msg = "--t-end is required."
        raise ValueError(msg)
    if config is None and any(
        v is None
        for v in (
            args.rho_l,
            args.u_l,
            args.t_l,
            args.rho_r,
            args.u_r,
            args.t_r,
            args.n,
            args.h,
        )
    ):
        msg = (
            "Quantum states and model parameters are required "
            "(--rho-l, --u-l, --t-l, --rho-r, --u-r, --t-r, --n, --h)."
        )
        raise ValueError(msg)

    if config is None:
        left = QuantumState(rho=args.rho_l, u=args.u_l, theta=args.t_l)
        right = QuantumState(rho=args.rho_r, u=args.u_r, theta=args.t_r)
        t_end = args.t_end
        n = args.n
        h = args.h
    else:
        left = QuantumState(
            rho=args.rho_l if args.rho_l is not None else config.left.rho,
            u=args.u_l if args.u_l is not None else config.left.u,
            theta=args.t_l if args.t_l is not None else config.left.theta,
        )
        right = QuantumState(
            rho=args.rho_r if args.rho_r is not None else config.right.rho,
            u=args.u_r if args.u_r is not None else config.right.u,
            theta=args.t_r if args.t_r is not None else config.right.theta,
        )
        t_end = args.t_end if args.t_end is not None else config.t_end
        n = args.n if args.n is not None else config.n
        h = args.h if args.h is not None else config.h
    domain = _resolve_domain(
        domain=config.domain if config else DomainConfig(),
        x_min=args.x_min,
        x_max=args.x_max,
        x0=args.x0,
        dx=args.dx,
        nx=args.nx,
    )
    statistic = (
        args.statistic
        if args.statistic is not None
        else (config.statistic if config else "FD")
    )
    all_statistics = args.all_statistics or (config.all_statistics if config else False)
    columns = (
        args.columns
        if args.columns is not None
        else (config.columns if config else None)
    )
    if require_output:
        output = _require_output(args.output, config.output if config else None)
    else:
        output = _optional_output(args.output, config.output if config else None)
    return left, right, t_end, domain, n, h, statistic, all_statistics, columns, output


def _interactive_domain_from_args(
    args: argparse.Namespace,
    config_domain: DomainConfig | None,
) -> DomainConfig:
    base = config_domain if config_domain is not None else INTERACTIVE_DOMAIN
    return _resolve_domain(
        domain=base,
        x_min=args.x_min,
        x_max=args.x_max,
        x0=args.x0,
        dx=args.dx,
        nx=args.nx,
    )


def _merge_classical_interactive_from_args(
    args: argparse.Namespace,
    config: ClassicalConfig | None,
) -> tuple[ClassicalState, ClassicalState, float, DomainConfig, float, Path]:
    left = ClassicalState(
        rho=args.rho_l
        if args.rho_l is not None
        else (config.left.rho if config else INTERACTIVE_CLASSICAL_LEFT.rho),
        u=args.u_l
        if args.u_l is not None
        else (config.left.u if config else INTERACTIVE_CLASSICAL_LEFT.u),
        p=args.p_l
        if args.p_l is not None
        else (config.left.p if config else INTERACTIVE_CLASSICAL_LEFT.p),
    )
    right = ClassicalState(
        rho=args.rho_r
        if args.rho_r is not None
        else (config.right.rho if config else INTERACTIVE_CLASSICAL_RIGHT.rho),
        u=args.u_r
        if args.u_r is not None
        else (config.right.u if config else INTERACTIVE_CLASSICAL_RIGHT.u),
        p=args.p_r
        if args.p_r is not None
        else (config.right.p if config else INTERACTIVE_CLASSICAL_RIGHT.p),
    )
    t_end = (
        args.t_end
        if args.t_end is not None
        else (config.t_end if config else INTERACTIVE_T_END)
    )
    domain = _interactive_domain_from_args(
        args,
        config.domain if config else None,
    )
    gamma = _resolve_gamma(
        gamma=args.gamma
        if args.gamma is not None
        else (config.gamma if config else INTERACTIVE_GAMMA),
        n=args.n if args.n is not None else (config.n if config else None),
        default=INTERACTIVE_GAMMA,
    )
    figure = args.figure if args.figure is not None else DEFAULT_CLASSICAL_FIGURE_PATH
    return left, right, t_end, domain, gamma, figure


def _merge_quantum_interactive_from_args(
    args: argparse.Namespace,
    config: QuantumConfig | None,
) -> tuple[QuantumState, QuantumState, float, DomainConfig, float, float, Path]:
    left = QuantumState(
        rho=args.rho_l
        if args.rho_l is not None
        else (config.left.rho if config else INTERACTIVE_QUANTUM_LEFT.rho),
        u=args.u_l
        if args.u_l is not None
        else (config.left.u if config else INTERACTIVE_QUANTUM_LEFT.u),
        theta=args.t_l
        if args.t_l is not None
        else (config.left.theta if config else INTERACTIVE_QUANTUM_LEFT.theta),
    )
    right = QuantumState(
        rho=args.rho_r
        if args.rho_r is not None
        else (config.right.rho if config else INTERACTIVE_QUANTUM_RIGHT.rho),
        u=args.u_r
        if args.u_r is not None
        else (config.right.u if config else INTERACTIVE_QUANTUM_RIGHT.u),
        theta=args.t_r
        if args.t_r is not None
        else (config.right.theta if config else INTERACTIVE_QUANTUM_RIGHT.theta),
    )
    t_end = (
        args.t_end
        if args.t_end is not None
        else (config.t_end if config else INTERACTIVE_T_END)
    )
    domain = _interactive_domain_from_args(
        args,
        config.domain if config else None,
    )
    n = args.n if args.n is not None else (config.n if config else INTERACTIVE_N)
    h = args.h if args.h is not None else (config.h if config else INTERACTIVE_H)
    figure = args.figure if args.figure is not None else DEFAULT_QUANTUM_FIGURE_PATH
    return left, right, t_end, domain, n, h, figure


def cmd_interactive_classical(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "classical":
            raise ValueError('Config mode must be "classical" for this command.')

        left, right, t_end, domain, gamma, figure = (
            _merge_classical_interactive_from_args(
                args,
                config if isinstance(config, ClassicalConfig) else None,
            )
        )
        x = _grid_from_domain(domain)
        return run_classical_interactive(
            left=left,
            right=right,
            t_end=t_end,
            gamma=gamma,
            x=x,
            x0=domain.x0,
            x_min=domain.x_min,
            x_max=domain.x_max,
            figure_path=figure,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_interactive_quantum(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "quantum":
            raise ValueError('Config mode must be "quantum" for this command.')

        left, right, t_end, domain, n, h, figure = _merge_quantum_interactive_from_args(
            args,
            config if isinstance(config, QuantumConfig) else None,
        )
        x = _grid_from_domain(domain)
        return run_quantum_interactive(
            left=left,
            right=right,
            t_end=t_end,
            n=n,
            h=h,
            x=x,
            x0=domain.x0,
            x_min=domain.x_min,
            x_max=domain.x_max,
            figure_path=figure,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_solve_classical(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "classical":
            raise ValueError('Config mode must be "classical" for this command.')

        left, right, t_end, domain, gamma, columns_raw, output = (
            _merge_classical_from_args(
                args, config if isinstance(config, ClassicalConfig) else None
            )
        )
        columns = resolve_columns(quantum=False, columns=columns_raw)
        output_format = resolve_output_format(
            args.format,
            output,
            config_format=config.format if config else None,
        )
        result = _solve_classical(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            gamma=gamma,
        )
        metadata = _classical_metadata(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            gamma=gamma,
        )
        _write_outputs(
            result=result,
            output=output,
            output_format=output_format,
            metadata=metadata,
            columns=columns,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


def cmd_solve_quantum(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "quantum":
            raise ValueError('Config mode must be "quantum" for this command.')

        (
            left,
            right,
            t_end,
            domain,
            n,
            h,
            statistic,
            all_statistics,
            columns_raw,
            output,
        ) = _merge_quantum_from_args(
            args, config if isinstance(config, QuantumConfig) else None
        )
        columns = resolve_columns(quantum=True, columns=columns_raw)
        output_format = resolve_output_format(
            args.format,
            output,
            config_format=config.format if config else None,
        )
        base_metadata = _quantum_metadata(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            n=n,
            h=h,
            statistic=None if all_statistics else statistic,
        )

        if all_statistics:
            results = _solve_quantum_all_statistics(
                left=left,
                right=right,
                t_end=t_end,
                domain=domain,
                n=n,
                h=h,
            )
            _write_quantum_outputs(
                results=results,
                output=output,
                output_format=output_format,
                base_metadata=base_metadata,
                columns=columns,
            )
        else:
            result = _solve_quantum(
                left=left,
                right=right,
                t_end=t_end,
                domain=domain,
                n=n,
                h=h,
                statistic=statistic,
            )
            metadata = {**base_metadata, "statistic": statistic}
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


def cmd_run_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        if isinstance(config, ClassicalConfig):
            domain = _resolve_domain(
                domain=config.domain,
                x_min=args.x_min,
                x_max=args.x_max,
                x0=args.x0,
                dx=args.dx,
                nx=args.nx,
            )
            gamma = _resolve_gamma(gamma=config.gamma, n=config.n)
            output = _require_output(args.output, config.output)
            columns = resolve_columns(
                quantum=False,
                columns=args.columns if args.columns is not None else config.columns,
            )
            output_format = resolve_output_format(
                args.format,
                output,
                config_format=config.format,
            )
            result = _solve_classical(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                gamma=gamma,
            )
            metadata = _classical_metadata(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                gamma=gamma,
            )
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
            return 0

        domain = _resolve_domain(
            domain=config.domain,
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        output = _require_output(args.output, config.output)
        columns = resolve_columns(
            quantum=True,
            columns=args.columns if args.columns is not None else config.columns,
        )
        output_format = resolve_output_format(
            args.format,
            output,
            config_format=config.format,
        )
        base_metadata = _quantum_metadata(
            left=config.left,
            right=config.right,
            t_end=config.t_end,
            domain=domain,
            n=config.n,
            h=config.h,
            statistic=None if config.all_statistics else config.statistic,
        )

        if config.all_statistics:
            results = _solve_quantum_all_statistics(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                n=config.n,
                h=config.h,
            )
            _write_quantum_outputs(
                results=results,
                output=output,
                output_format=output_format,
                base_metadata=base_metadata,
                columns=columns,
            )
        else:
            result = _solve_quantum(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                n=config.n,
                h=config.h,
                statistic=config.statistic,
            )
            metadata = {**base_metadata, "statistic": config.statistic}
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


def cmd_toro(args: argparse.Namespace) -> int:
    try:
        preset = TORO_TESTS[args.number]
        if args.output is None:
            output = Path(f"toro{preset.number}.csv")
        else:
            output = args.output
        output_format = resolve_output_format(args.format, output)

        domain = _resolve_domain(
            domain=DomainConfig(
                x_min=preset.x_min,
                x_max=preset.x_max,
                x0=preset.x0,
                dx=preset.dx,
            ),
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        left = ClassicalState(rho=preset.rho_l, u=preset.u_l, p=preset.p_l)
        right = ClassicalState(rho=preset.rho_r, u=preset.u_r, p=preset.p_r)
        columns = resolve_columns(quantum=False, columns=args.columns)
        result = _solve_classical(
            left=left,
            right=right,
            t_end=preset.t_end,
            domain=domain,
            gamma=preset.gamma,
        )
        metadata = _classical_metadata(
            left=left,
            right=right,
            t_end=preset.t_end,
            domain=domain,
            gamma=preset.gamma,
            preset=f"toro:{preset.number}",
        )
        metadata["reference"] = preset.reference
        metadata["name"] = preset.name
        _write_outputs(
            result=result,
            output=output,
            output_format=output_format,
            metadata=metadata,
            columns=columns,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


def cmd_quantum_example(args: argparse.Namespace) -> int:
    try:
        preset = QUANTUM_EXAMPLES[args.number]
        if args.output is None:
            output = Path(f"qeuler_eg{preset.number}.csv")
        else:
            output = args.output
        output_format = resolve_output_format(args.format, output)

        domain = _resolve_domain(
            domain=DomainConfig(
                x_min=preset.x_min,
                x_max=preset.x_max,
                x0=preset.x0,
                dx=preset.dx,
            ),
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        left = QuantumState(rho=preset.rho_l, u=preset.u_l, theta=preset.t_l)
        right = QuantumState(rho=preset.rho_r, u=preset.u_r, theta=preset.t_r)
        statistic = args.statistic if args.statistic is not None else "FD"
        all_statistics = args.all_statistics
        columns = resolve_columns(quantum=True, columns=args.columns)
        base_metadata = _quantum_metadata(
            left=left,
            right=right,
            t_end=preset.t_end,
            domain=domain,
            n=preset.n,
            h=preset.h,
            statistic=None if all_statistics else statistic,
            preset=f"quantum:{preset.number}",
        )
        base_metadata["name"] = preset.name

        if all_statistics:
            results = _solve_quantum_all_statistics(
                left=left,
                right=right,
                t_end=preset.t_end,
                domain=domain,
                n=preset.n,
                h=preset.h,
                h_be=preset.h_be,
                h_fd=preset.h_fd,
            )
            _write_quantum_outputs(
                results=results,
                output=output,
                output_format=output_format,
                base_metadata=base_metadata,
                columns=columns,
            )
        else:
            h = _resolve_stat_h(preset.h, statistic, h_fd=preset.h_fd, h_be=preset.h_be)
            result = _solve_quantum(
                left=left,
                right=right,
                t_end=preset.t_end,
                domain=domain,
                n=preset.n,
                h=h,
                statistic=statistic,
            )
            metadata = {**base_metadata, "statistic": statistic, "h": h}
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


def cmd_plot_classical(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "classical":
            raise ValueError('Config mode must be "classical" for this command.')

        left, right, t_end, domain, gamma, columns_raw, output = (
            _merge_classical_from_args(
                args,
                config if isinstance(config, ClassicalConfig) else None,
                require_output=False,
            )
        )
        figure = _require_figure(args.figure, args.show)
        result = _solve_classical(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            gamma=gamma,
        )
        plot_path = _single_figure_path(figure) if figure is not None else None
        saved = plot_single(
            result,
            title="Classical Euler Riemann solution",
            output=plot_path,
            show=args.show,
            kind="classical",
        )
        _print_saved([saved])

        if output is not None:
            columns = resolve_columns(quantum=False, columns=columns_raw)
            output_format = resolve_output_format(
                args.format,
                output,
                config_format=config.format if config else None,
            )
            metadata = _classical_metadata(
                left=left,
                right=right,
                t_end=t_end,
                domain=domain,
                gamma=gamma,
            )
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_plot_quantum(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "quantum":
            raise ValueError('Config mode must be "quantum" for this command.')

        (
            left,
            right,
            t_end,
            domain,
            n,
            h,
            statistic,
            all_statistics,
            columns_raw,
            output,
        ) = _merge_quantum_from_args(
            args,
            config if isinstance(config, QuantumConfig) else None,
            require_output=False,
        )
        figure = _require_figure(args.figure, args.show)
        layout: PlotLayout = args.layout
        result: RiemannResult | None = None
        results: dict[str, RiemannResult] | None = None
        base_metadata = _quantum_metadata(
            left=left,
            right=right,
            t_end=t_end,
            domain=domain,
            n=n,
            h=h,
            statistic=None if all_statistics else statistic,
        )

        if all_statistics:
            if figure is None:
                msg = (
                    "Figure path (-f/--figure) is required for --all-statistics plots."
                )
                raise ValueError(msg)
            results = _solve_quantum_all_statistics(
                left=left,
                right=right,
                t_end=t_end,
                domain=domain,
                n=n,
                h=h,
            )
            saved = plot_all_statistics(
                results,
                title="Quantum Euler Riemann solution",
                figure=figure,
                layout=layout,
                show=args.show,
            )
            _print_saved(saved)
        else:
            result = _solve_quantum(
                left=left,
                right=right,
                t_end=t_end,
                domain=domain,
                n=n,
                h=h,
                statistic=statistic,
            )
            plot_path = _single_figure_path(figure) if figure is not None else None
            saved = plot_single(
                result,
                title=f"Quantum Euler Riemann solution ({statistic})",
                output=plot_path,
                show=args.show,
                kind="quantum",
            )
            _print_saved([saved])

        if output is not None:
            columns = resolve_columns(quantum=True, columns=columns_raw)
            output_format = resolve_output_format(
                args.format,
                output,
                config_format=config.format if config else None,
            )
            if all_statistics:
                assert results is not None
                _write_quantum_outputs(
                    results=results,
                    output=output,
                    output_format=output_format,
                    base_metadata=base_metadata,
                    columns=columns,
                )
            else:
                assert result is not None
                metadata = {**base_metadata, "statistic": statistic}
                _write_outputs(
                    result=result,
                    output=output,
                    output_format=output_format,
                    metadata=metadata,
                    columns=columns,
                )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_plot_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        figure = _require_figure(args.figure, args.show)
        layout: PlotLayout = args.layout

        if isinstance(config, ClassicalConfig):
            domain = _resolve_domain(
                domain=config.domain,
                x_min=args.x_min,
                x_max=args.x_max,
                x0=args.x0,
                dx=args.dx,
                nx=args.nx,
            )
            gamma = _resolve_gamma(gamma=config.gamma, n=config.n)
            result = _solve_classical(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                gamma=gamma,
            )
            plot_path = _single_figure_path(figure) if figure is not None else None
            saved = plot_single(
                result,
                title="Classical Euler Riemann solution",
                output=plot_path,
                show=args.show,
                kind="classical",
            )
            _print_saved([saved])

            output = _optional_output(args.output, config.output)
            if output is not None:
                columns = resolve_columns(
                    quantum=False,
                    columns=args.columns
                    if args.columns is not None
                    else config.columns,
                )
                output_format = resolve_output_format(
                    args.format,
                    output,
                    config_format=config.format,
                )
                metadata = _classical_metadata(
                    left=config.left,
                    right=config.right,
                    t_end=config.t_end,
                    domain=domain,
                    gamma=gamma,
                )
                _write_outputs(
                    result=result,
                    output=output,
                    output_format=output_format,
                    metadata=metadata,
                    columns=columns,
                )
            return 0

        domain = _resolve_domain(
            domain=config.domain,
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        base_metadata = _quantum_metadata(
            left=config.left,
            right=config.right,
            t_end=config.t_end,
            domain=domain,
            n=config.n,
            h=config.h,
            statistic=None if config.all_statistics else config.statistic,
        )

        if config.all_statistics:
            if figure is None:
                msg = (
                    "Figure path (-f/--figure) is required for --all-statistics plots."
                )
                raise ValueError(msg)
            results = _solve_quantum_all_statistics(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                n=config.n,
                h=config.h,
            )
            saved = plot_all_statistics(
                results,
                title="Quantum Euler Riemann solution",
                figure=figure,
                layout=layout,
                show=args.show,
            )
            _print_saved(saved)
        else:
            result = _solve_quantum(
                left=config.left,
                right=config.right,
                t_end=config.t_end,
                domain=domain,
                n=config.n,
                h=config.h,
                statistic=config.statistic,
            )
            plot_path = _single_figure_path(figure) if figure is not None else None
            saved = plot_single(
                result,
                title=f"Quantum Euler Riemann solution ({config.statistic})",
                output=plot_path,
                show=args.show,
                kind="quantum",
            )
            _print_saved([saved])

        output = _optional_output(args.output, config.output)
        if output is not None:
            columns = resolve_columns(
                quantum=True,
                columns=args.columns if args.columns is not None else config.columns,
            )
            output_format = resolve_output_format(
                args.format,
                output,
                config_format=config.format,
            )
            if config.all_statistics:
                assert results is not None
                _write_quantum_outputs(
                    results=results,
                    output=output,
                    output_format=output_format,
                    base_metadata=base_metadata,
                    columns=columns,
                )
            else:
                assert result is not None
                metadata = {**base_metadata, "statistic": config.statistic}
                _write_outputs(
                    result=result,
                    output=output,
                    output_format=output_format,
                    metadata=metadata,
                    columns=columns,
                )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_plot_toro(args: argparse.Namespace) -> int:
    try:
        preset = TORO_TESTS[args.number]
        figure = _require_figure(args.figure, args.show)
        domain = _resolve_domain(
            domain=DomainConfig(
                x_min=preset.x_min,
                x_max=preset.x_max,
                x0=preset.x0,
                dx=preset.dx,
            ),
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        left = ClassicalState(rho=preset.rho_l, u=preset.u_l, p=preset.p_l)
        right = ClassicalState(rho=preset.rho_r, u=preset.u_r, p=preset.p_r)
        result = _solve_classical(
            left=left,
            right=right,
            t_end=preset.t_end,
            domain=domain,
            gamma=preset.gamma,
        )
        plot_path = _single_figure_path(figure) if figure is not None else None
        saved = plot_single(
            result,
            title=f"Toro test {preset.number}: {preset.name}",
            output=plot_path,
            show=args.show,
            kind="classical",
        )
        _print_saved([saved])

        if args.output is not None:
            output = args.output
            output_format = resolve_output_format(args.format, output)
            columns = resolve_columns(quantum=False, columns=args.columns)
            metadata = _classical_metadata(
                left=left,
                right=right,
                t_end=preset.t_end,
                domain=domain,
                gamma=preset.gamma,
                preset=f"toro:{preset.number}",
            )
            metadata["reference"] = preset.reference
            metadata["name"] = preset.name
            _write_outputs(
                result=result,
                output=output,
                output_format=output_format,
                metadata=metadata,
                columns=columns,
            )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_plot_quantum_example(args: argparse.Namespace) -> int:
    try:
        preset = QUANTUM_EXAMPLES[args.number]
        figure = _require_figure(args.figure, args.show)
        layout: PlotLayout = args.layout
        domain = _resolve_domain(
            domain=DomainConfig(
                x_min=preset.x_min,
                x_max=preset.x_max,
                x0=preset.x0,
                dx=preset.dx,
            ),
            x_min=args.x_min,
            x_max=args.x_max,
            x0=args.x0,
            dx=args.dx,
            nx=args.nx,
        )
        left = QuantumState(rho=preset.rho_l, u=preset.u_l, theta=preset.t_l)
        right = QuantumState(rho=preset.rho_r, u=preset.u_r, theta=preset.t_r)
        statistic = args.statistic if args.statistic is not None else "FD"
        all_statistics = args.all_statistics
        base_metadata = _quantum_metadata(
            left=left,
            right=right,
            t_end=preset.t_end,
            domain=domain,
            n=preset.n,
            h=preset.h,
            statistic=None if all_statistics else statistic,
            preset=f"quantum:{preset.number}",
        )
        base_metadata["name"] = preset.name
        title = f"Quantum Euler example {preset.number}: {preset.name}"
        result: RiemannResult | None = None
        results: dict[str, RiemannResult] | None = None
        h = _resolve_stat_h(preset.h, statistic, h_fd=preset.h_fd, h_be=preset.h_be)

        if all_statistics:
            if figure is None:
                msg = (
                    "Figure path (-f/--figure) is required for --all-statistics plots."
                )
                raise ValueError(msg)
            results = _solve_quantum_all_statistics(
                left=left,
                right=right,
                t_end=preset.t_end,
                domain=domain,
                n=preset.n,
                h=preset.h,
                h_fd=preset.h_fd,
                h_be=preset.h_be,
            )
            saved = plot_all_statistics(
                results,
                title=title,
                figure=figure,
                layout=layout,
                show=args.show,
            )
            _print_saved(saved)
        else:
            result = _solve_quantum(
                left=left,
                right=right,
                t_end=preset.t_end,
                domain=domain,
                n=preset.n,
                h=h,
                statistic=statistic,
            )
            plot_path = _single_figure_path(figure) if figure is not None else None
            saved = plot_single(
                result,
                title=f"{title} ({statistic})",
                output=plot_path,
                show=args.show,
                kind="quantum",
            )
            _print_saved([saved])

        if args.output is not None:
            output = args.output
            output_format = resolve_output_format(args.format, output)
            columns = resolve_columns(quantum=True, columns=args.columns)
            if all_statistics:
                assert results is not None
                _write_quantum_outputs(
                    results=results,
                    output=output,
                    output_format=output_format,
                    base_metadata=base_metadata,
                    columns=columns,
                )
            else:
                assert result is not None
                metadata = {**base_metadata, "statistic": statistic, "h": h}
                _write_outputs(
                    result=result,
                    output=output,
                    output_format=output_format,
                    metadata=metadata,
                    columns=columns,
                )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    show_toro = args.toro or (not args.toro and not args.quantum)
    show_quantum = args.quantum or (not args.toro and not args.quantum)

    if show_toro:
        print("Toro classical tests:")
        for number in sorted(TORO_TESTS):
            test = TORO_TESTS[number]
            print(
                f"  {number}: {test.name} | "
                f"L=({test.rho_l}, {test.u_l}, {test.p_l}) "
                f"R=({test.rho_r}, {test.u_r}, {test.p_r}) | "
                f"t_end={test.t_end}, x0={test.x0}, gamma={test.gamma}"
            )

    if show_quantum:
        if show_toro:
            print()
        print("Quantum Euler benchmarks:")
        for number in sorted(QUANTUM_EXAMPLES):
            example = QUANTUM_EXAMPLES[number]
            print(
                f"  {number}: {example.name} | "
                f"L=({example.rho_l}, {example.u_l}, {example.t_l}) "
                f"R=({example.rho_r}, {example.u_r}, {example.t_r}) | "
                f"t_end={example.t_end}, n={example.n}, h={example.h}"
            )

    return 0
