# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

import numpy as np
import pytest

from ideal_gases.riemann import adiabatic_index, classical_gas, quantum_gas


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
