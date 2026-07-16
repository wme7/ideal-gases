# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

import numpy as np
import pytest

from ideal_gases.riemann import (
    RHO_FLOOR,
    adiabatic_index,
    classical_gas,
    quantum_gas,
)


def test_adiabatic_index_monatomic() -> None:
    assert adiabatic_index(3) == pytest.approx(5.0 / 3.0)


def test_classical_sod_shock_tube() -> None:
    result = classical_gas(
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=0.25,
        gamma=1.4,
        x=np.linspace(0.0, 1.0, 51),
    )

    assert result.rho.shape == (51,)
    assert np.min(result.rho) > 0.0
    assert np.min(result.p) > 0.0
    assert np.any(result.ux != 0.0)
    assert np.all(result.mach.shape == result.rho.shape)
    assert np.all(result.entropy.shape == result.rho.shape)


def test_classical_default_grid_from_dx() -> None:
    result = classical_gas(
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=0.1,
        gamma=1.4,
        dx=0.05,
    )
    assert result.x[0] == pytest.approx(0.0)
    assert result.x[-1] == pytest.approx(1.0)
    assert len(result.x) == 21
    assert np.all(np.isfinite(result.rho))


def test_classical_t_end_zero_keeps_initial_states() -> None:
    x = np.linspace(0.0, 1.0, 21)
    x0 = 0.5
    result = classical_gas(
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=0.0,
        gamma=1.4,
        x=x,
        x0=x0,
    )
    left = x < x0
    np.testing.assert_allclose(result.rho[left], 1.0)
    np.testing.assert_allclose(result.p[left], 1.0)
    np.testing.assert_allclose(result.rho[~left], 0.125)
    np.testing.assert_allclose(result.p[~left], 0.1)
    np.testing.assert_allclose(result.ux, 0.0)


def test_classical_vacuum_left() -> None:
    x = np.linspace(0.0, 1.0, 51)
    x0 = 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        result = classical_gas(
            rho_l=0.0,
            u_l=0.0,
            p_l=0.0,
            rho_r=1.0,
            u_r=0.0,
            p_r=1.0,
            t_end=0.15,
            gamma=1.4,
            x=x,
            x0=x0,
        )
    left = x < x0
    assert np.all(result.rho[left] == RHO_FLOOR)
    np.testing.assert_allclose(result.rho[~left], 1.0)
    np.testing.assert_allclose(result.p[~left], 1.0)


def test_classical_vacuum_right() -> None:
    x = np.linspace(0.0, 1.0, 51)
    x0 = 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        result = classical_gas(
            rho_l=1.0,
            u_l=0.0,
            p_l=1.0,
            rho_r=0.0,
            u_r=0.0,
            p_r=0.0,
            t_end=0.15,
            gamma=1.4,
            x=x,
            x0=x0,
        )
    left = x < x0
    np.testing.assert_allclose(result.rho[left], 1.0)
    np.testing.assert_allclose(result.p[left], 1.0)
    assert np.all(result.rho[~left] == RHO_FLOOR)


def test_classical_vacuum_middle_from_strong_rarefaction() -> None:
    x = np.linspace(0.0, 1.0, 201)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = classical_gas(
            rho_l=1.0,
            u_l=-5.0,
            p_l=0.4,
            rho_r=1.0,
            u_r=5.0,
            p_r=0.4,
            t_end=0.05,
            gamma=1.4,
            x=x,
        )
    assert np.all(np.isfinite(result.rho))
    assert np.min(result.rho) == RHO_FLOOR
    assert np.max(result.rho) == pytest.approx(1.0)


def test_classical_left_shock_right_rarefaction() -> None:
    """Reverse Sod: high-pressure right state drives a left-going shock."""
    x = np.linspace(0.0, 1.0, 201)
    result = classical_gas(
        rho_l=0.125,
        u_l=0.0,
        p_l=0.1,
        rho_r=1.0,
        u_r=0.0,
        p_r=1.0,
        t_end=0.2,
        gamma=1.4,
        x=x,
    )
    assert np.all(np.isfinite(result.rho))
    assert np.min(result.rho) == pytest.approx(0.125)
    assert np.max(result.rho) == pytest.approx(1.0)
    assert np.min(result.ux) < 0.0
    assert np.max(result.ux) == pytest.approx(0.0)


def test_classical_toro4_shock_contact_rarefaction() -> None:
    x = np.linspace(0.0, 1.0, 201)
    result = classical_gas(
        rho_l=5.99924,
        u_l=19.5975,
        p_l=460.894,
        rho_r=5.99242,
        u_r=-6.19633,
        p_r=46.0950,
        t_end=0.035,
        gamma=1.4,
        x=x,
        x0=0.4,
    )
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.p))
    assert np.min(result.rho) > 0.0


def test_quantum_mb_matches_classical_pressures() -> None:
    n = 3.0
    h = 1.0
    t_l, t_r = 1.0, 0.8
    rho_l, rho_r = 1.0, 0.5
    x = np.linspace(0.0, 1.0, 41)

    classical = classical_gas(
        rho_l=rho_l,
        u_l=0.0,
        p_l=rho_l * t_l,
        rho_r=rho_r,
        u_r=0.0,
        p_r=rho_r * t_r,
        t_end=0.2,
        gamma=adiabatic_index(n),
        x=x,
    )
    quantum = quantum_gas(
        rho_l=rho_l,
        u_l=0.0,
        t_l=t_l,
        rho_r=rho_r,
        u_r=0.0,
        t_r=t_r,
        t_end=0.2,
        n=n,
        h=h,
        statistic="MB",
        x=x,
    )

    np.testing.assert_allclose(quantum.rho, classical.rho, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(quantum.ux, classical.ux, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(quantum.p, classical.p, rtol=0.0, atol=1e-12)


def test_quantum_fd_runs() -> None:
    result = quantum_gas(
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.1,
        t_end=0.25,
        n=3.0,
        h=1.0,
        statistic="FD",
        x=np.linspace(0.0, 1.0, 21),
    )

    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.p))
    assert np.all(np.isfinite(result.z))
    assert np.all(np.isfinite(result.t))


def test_quantum_be_runs() -> None:
    result = quantum_gas(
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        t_r=0.25,
        t_end=0.15,
        n=3.0,
        h=1.0,
        statistic="BE",
        x=np.linspace(0.0, 1.0, 21),
    )
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.p))
    assert np.all(np.isfinite(result.z))
    assert np.all(result.z > 0.0)
    assert np.all(result.z < 1.0)


def test_quantum_fd_near_zero_left_density() -> None:
    x = np.linspace(0.0, 1.0, 21)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = quantum_gas(
            rho_l=0.0,
            u_l=0.0,
            t_l=1.0,
            rho_r=1.0,
            u_r=0.0,
            t_r=1.0,
            t_end=0.1,
            n=3.0,
            h=1.0,
            statistic="FD",
            x=x,
            x0=0.5,
        )
    assert np.all(np.isfinite(result.rho))
    assert np.all(result.rho[x < 0.5] == RHO_FLOOR)
    assert np.all(np.isfinite(result.p[x >= 0.5]))


def test_quantum_fd_near_zero_right_density() -> None:
    x = np.linspace(0.0, 1.0, 21)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = quantum_gas(
            rho_l=1.0,
            u_l=0.0,
            t_l=1.0,
            rho_r=0.0,
            u_r=0.0,
            t_r=1.0,
            t_end=0.1,
            n=3.0,
            h=1.0,
            statistic="FD",
            x=x,
            x0=0.5,
        )
    assert np.all(np.isfinite(result.rho))
    assert np.all(result.rho[x >= 0.5] == RHO_FLOOR)
    assert np.all(np.isfinite(result.p[x < 0.5]))
