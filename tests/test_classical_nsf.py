# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Tests for :mod:`ideal_gases.classical_nsf`."""

from __future__ import annotations

import numpy as np
import pytest

from ideal_gases.classical_nsf import ClassicalNSFResult, classical_nsf
from ideal_gases.quantum_nsf import quantum_nsf


def _riemann_kwargs(dim: int, t_end: float = 0.0):
    x = np.linspace(0.0, 1.0, 11)
    return dict(
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        t_end=t_end,
        dim=dim,
        kn=0.01,
        x=x,
        x0=0.5,
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_t_end_zero_recovers_riemann_states(dim: int) -> None:
    x = np.linspace(0.0, 1.0, 11)
    result = classical_nsf(**_riemann_kwargs(dim, t_end=0.0))
    left = x <= 0.5
    np.testing.assert_allclose(result.rho[left], 1.0)
    np.testing.assert_allclose(result.rho[~left], 0.125)
    np.testing.assert_allclose(result.p[left], 1.0)
    np.testing.assert_allclose(result.p[~left], 0.1)
    np.testing.assert_allclose(result.t[left], 1.0)
    np.testing.assert_allclose(result.t[~left], 0.8)
    np.testing.assert_allclose(result.u, 0.0, atol=1e-12)
    assert np.all(np.isfinite(result.q))


def test_dim_one_smoke() -> None:
    result = classical_nsf(**_riemann_kwargs(1, t_end=0.0))
    assert result.rho.shape == (11,)
    assert np.all(np.isfinite(result.q))


def test_unpack_namedtuple() -> None:
    result = classical_nsf(**_riemann_kwargs(3, t_end=0.0))
    rho, u, temp, p, q = result
    assert isinstance(result, ClassicalNSFResult)
    assert rho.shape == u.shape == temp.shape == p.shape == q.shape
    assert len(result) == 5


def test_custom_transport_callables_are_invoked() -> None:
    calls = {"mu": 0, "kappa": 0}

    def viscosity(rho, temp):
        calls["mu"] += 1
        return np.full_like(np.asarray(rho, dtype=float), 1e-4)

    def conductivity(rho, temp):
        calls["kappa"] += 1
        return np.full_like(np.asarray(rho, dtype=float), 1e-4)

    classical_nsf(
        **_riemann_kwargs(3, t_end=1e-8),
        viscosity=viscosity,
        conductivity=conductivity,
    )
    assert calls["mu"] >= 1
    assert calls["kappa"] >= 1


def test_reject_invalid_dim() -> None:
    kwargs = _riemann_kwargs(3, t_end=0.0)
    kwargs["dim"] = 4
    with pytest.raises(ValueError, match="dim"):
        classical_nsf(**kwargs)
    kwargs["dim"] = 1.5
    with pytest.raises(ValueError, match="dim"):
        classical_nsf(**kwargs)


def test_mb_quantum_matches_classical_pressure_at_t_end_zero() -> None:
    x = np.linspace(0.0, 1.0, 11)
    rho_l, t_l = 1.0, 1.0
    rho_r, t_r = 0.4, 0.6
    p_l, p_r = rho_l * t_l, rho_r * t_r
    quantum = quantum_nsf(
        rho_l,
        0.0,
        t_l,
        rho_r,
        0.0,
        t_r,
        t_end=0.0,
        dim=3,
        h=1.0,
        kn=0.01,
        statistic="MB",
        x=x,
        x0=0.5,
    )
    classical = classical_nsf(
        rho_l,
        0.0,
        p_l,
        rho_r,
        0.0,
        p_r,
        t_end=0.0,
        dim=3,
        kn=0.01,
        x=x,
        x0=0.5,
    )
    np.testing.assert_allclose(quantum.rho, classical.rho)
    np.testing.assert_allclose(quantum.t, classical.t)
    np.testing.assert_allclose(quantum.p, classical.p)
    np.testing.assert_allclose(classical.p, classical.rho * classical.t)
