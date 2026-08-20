# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Tests for :mod:`ideal_gases.quantum_nsf`."""

from __future__ import annotations

import numpy as np
import pytest

from ideal_gases.quantum_nsf import NSFResult, quantum_nsf


def _riemann_kwargs(dim: int, statistic: str, t_end: float = 0.0):
    x = np.linspace(0.0, 1.0, 11)
    return dict(
        rho_l=1.0,
        u_l=0.0,
        t_l=1.0,
        rho_r=0.4,
        u_r=0.0,
        t_r=0.6,
        t_end=t_end,
        dim=dim,
        h=1.0,
        kn=0.01,
        statistic=statistic,
        x=x,
        x0=0.5,
    )


@pytest.mark.parametrize("statistic", ["FD", "BE", "MB"])
@pytest.mark.parametrize("dim", [2, 3])
def test_t_end_zero_recovers_riemann_states(dim: int, statistic: str) -> None:
    x = np.linspace(0.0, 1.0, 11)
    result = quantum_nsf(**_riemann_kwargs(dim, statistic, t_end=0.0))
    left = x <= 0.5
    np.testing.assert_allclose(result.rho[left], 1.0)
    np.testing.assert_allclose(result.rho[~left], 0.4)
    np.testing.assert_allclose(result.t[left], 1.0, rtol=1e-6)
    np.testing.assert_allclose(result.t[~left], 0.6, rtol=1e-6)
    np.testing.assert_allclose(result.u, 0.0, atol=1e-12)
    assert np.all(np.isfinite(result.q))


def test_dim_one_smoke() -> None:
    result = quantum_nsf(**_riemann_kwargs(1, "MB", t_end=0.0))
    assert result.rho.shape == (11,)
    assert np.all(np.isfinite(result.q))


def test_unpack_namedtuple() -> None:
    result = quantum_nsf(**_riemann_kwargs(3, "MB", t_end=0.0))
    rho, u, temp, p, z, q = result
    assert isinstance(result, NSFResult)
    assert rho.shape == u.shape == temp.shape == p.shape == z.shape == q.shape
    assert len(result) == 6


def test_custom_transport_callables_are_invoked() -> None:
    calls = {"mu": 0, "kappa": 0}

    def viscosity(rho, temp, z):
        calls["mu"] += 1
        return np.full_like(np.asarray(rho, dtype=float), 1e-4)

    def conductivity(rho, temp, z):
        calls["kappa"] += 1
        return np.full_like(np.asarray(rho, dtype=float), 1e-4)

    quantum_nsf(
        **_riemann_kwargs(3, "MB", t_end=1e-8),
        viscosity=viscosity,
        conductivity=conductivity,
    )
    assert calls["mu"] >= 1
    assert calls["kappa"] >= 1


def test_reject_invalid_dim() -> None:
    kwargs = _riemann_kwargs(3, "MB", t_end=0.0)
    kwargs["dim"] = 4
    with pytest.raises(ValueError, match="dim"):
        quantum_nsf(**kwargs)
    kwargs["dim"] = 1.5
    with pytest.raises(ValueError, match="dim"):
        quantum_nsf(**kwargs)
