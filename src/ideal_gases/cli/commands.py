# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""CLI command implementations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ideal_gases.cli.config import (
    ClassicalConfig,
    ClassicalState,
    DomainConfig,
    QuantumConfig,
    QuantumState,
    SolverConfig,
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


def _resolve_gamma(*, gamma: float | None, n: float | None, default: float = 1.4) -> float:
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


def _require_output(path: Path | None, config_output: str | None) -> Path:
    if path is not None:
        return path
    if config_output is not None:
        return Path(config_output)
    msg = "Output path is required (-o/--output or config \"output\")."
    raise ValueError(msg)


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


def _merge_classical_from_args(
    args: argparse.Namespace,
    config: ClassicalConfig | None,
) -> tuple[ClassicalState, ClassicalState, float, DomainConfig, float, str | None, Path]:
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

    left = ClassicalState(
        rho=args.rho_l if args.rho_l is not None else config.left.rho,  # type: ignore[union-attr]
        u=args.u_l if args.u_l is not None else config.left.u,  # type: ignore[union-attr]
        p=args.p_l if args.p_l is not None else config.left.p,  # type: ignore[union-attr]
    )
    right = ClassicalState(
        rho=args.rho_r if args.rho_r is not None else config.right.rho,  # type: ignore[union-attr]
        u=args.u_r if args.u_r is not None else config.right.u,  # type: ignore[union-attr]
        p=args.p_r if args.p_r is not None else config.right.p,  # type: ignore[union-attr]
    )
    t_end = args.t_end if args.t_end is not None else config.t_end  # type: ignore[union-attr]
    domain = _resolve_domain(
        domain=config.domain if config else DomainConfig(),
        x_min=args.x_min,
        x_max=args.x_max,
        x0=args.x0,
        dx=args.dx,
        nx=args.nx,
    )
    gamma = _resolve_gamma(
        gamma=args.gamma if args.gamma is not None else (config.gamma if config else None),
        n=args.n if args.n is not None else (config.n if config else None),
    )
    columns = args.columns if args.columns is not None else (config.columns if config else None)
    output = _require_output(args.output, config.output if config else None)
    return left, right, t_end, domain, gamma, columns, output


def _merge_quantum_from_args(
    args: argparse.Namespace,
    config: QuantumConfig | None,
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

    left = QuantumState(
        rho=args.rho_l if args.rho_l is not None else config.left.rho,  # type: ignore[union-attr]
        u=args.u_l if args.u_l is not None else config.left.u,  # type: ignore[union-attr]
        theta=args.t_l if args.t_l is not None else config.left.theta,  # type: ignore[union-attr]
    )
    right = QuantumState(
        rho=args.rho_r if args.rho_r is not None else config.right.rho,  # type: ignore[union-attr]
        u=args.u_r if args.u_r is not None else config.right.u,  # type: ignore[union-attr]
        theta=args.t_r if args.t_r is not None else config.right.theta,  # type: ignore[union-attr]
    )
    t_end = args.t_end if args.t_end is not None else config.t_end  # type: ignore[union-attr]
    domain = _resolve_domain(
        domain=config.domain if config else DomainConfig(),
        x_min=args.x_min,
        x_max=args.x_max,
        x0=args.x0,
        dx=args.dx,
        nx=args.nx,
    )
    n = args.n if args.n is not None else config.n  # type: ignore[union-attr]
    h = args.h if args.h is not None else config.h  # type: ignore[union-attr]
    statistic = (
        args.statistic
        if args.statistic is not None
        else (config.statistic if config else "FD")
    )
    all_statistics = args.all_statistics or (config.all_statistics if config else False)
    columns = args.columns if args.columns is not None else (config.columns if config else None)
    output = _require_output(args.output, config.output if config else None)
    return left, right, t_end, domain, n, h, statistic, all_statistics, columns, output


def cmd_solve_classical(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else None
        if config is not None and config.mode != "classical":
            raise ValueError('Config mode must be "classical" for this command.')

        left, right, t_end, domain, gamma, columns_raw, output = _merge_classical_from_args(
            args, config if isinstance(config, ClassicalConfig) else None
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
        ) = _merge_quantum_from_args(args, config if isinstance(config, QuantumConfig) else None)
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
            results = {
                stat: _solve_quantum(
                    left=left,
                    right=right,
                    t_end=t_end,
                    domain=domain,
                    n=n,
                    h=h,
                    statistic=stat,
                )
                for stat in STATISTICS
            }
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
            results = {
                stat: _solve_quantum(
                    left=config.left,
                    right=config.right,
                    t_end=config.t_end,
                    domain=domain,
                    n=config.n,
                    h=config.h,
                    statistic=stat,
                )
                for stat in STATISTICS
            }
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
            results: dict[str, RiemannResult] = {}
            for stat in STATISTICS:
                h = preset.h_fd if stat == "FD" and preset.h_fd is not None else preset.h
                results[stat] = _solve_quantum(
                    left=left,
                    right=right,
                    t_end=preset.t_end,
                    domain=domain,
                    n=preset.n,
                    h=h,
                    statistic=stat,
                )
            _write_quantum_outputs(
                results=results,
                output=output,
                output_format=output_format,
                base_metadata=base_metadata,
                columns=columns,
            )
        else:
            h = preset.h_fd if statistic == "FD" and preset.h_fd is not None else preset.h
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
