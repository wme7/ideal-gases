# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Interactive classical ideal-gas Riemann explorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ideal_gases.cli.config import ClassicalState
from ideal_gases.cli.interactive._widgets import (
    SAVE_DPI,
    autoscale_axes,
    piecewise_ic,
)
from ideal_gases.cli.plot import _require_matplotlib
from ideal_gases.riemann import classical_gas


def run_classical_interactive(
    *,
    left: ClassicalState,
    right: ClassicalState,
    t_end: float,
    gamma: float,
    x: NDArray[np.float64],
    x0: float,
    x_min: float,
    x_max: float,
    figure_path: Path,
) -> int:
    plt = _require_matplotlib(show=True)
    from matplotlib.widgets import Button, Slider

    result = classical_gas(
        left.rho,
        left.u,
        left.p,
        right.rho,
        right.u,
        right.p,
        t_end,
        gamma,
        x=x,
        x0=x0,
    )

    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    fig.subplots_adjust(bottom=0.30, top=0.86)
    fig.suptitle("Interactive classical Riemann solver")

    ax_rho = fig.add_subplot(311)
    ax_vx = fig.add_subplot(312)
    ax_p = fig.add_subplot(313)

    ic_rho = piecewise_ic(x, x0, left.rho, right.rho)
    ic_vx = piecewise_ic(x, x0, left.u, right.u)
    ic_p = piecewise_ic(x, x0, left.p, right.p)

    (ic_line_rho,) = ax_rho.plot(x, ic_rho, "--", color="0.5", label="IC")
    (sol_line_rho,) = ax_rho.plot(x, result.rho, linewidth=2, label="solution")
    ax_rho.set_xticks([])
    ax_rho.set_xlim(x_min, x_max)
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.legend(loc="upper right", fontsize=8)

    (ic_line_vx,) = ax_vx.plot(x, ic_vx, "--", color="0.5")
    (sol_line_vx,) = ax_vx.plot(x, result.ux, linewidth=2)
    ax_vx.set_xticks([])
    ax_vx.set_xlim(x_min, x_max)
    ax_vx.set_ylabel(r"$u_x$")

    (ic_line_p,) = ax_p.plot(x, ic_p, "--", color="0.5")
    (sol_line_p,) = ax_p.plot(x, result.p, linewidth=2)
    ax_p.set_xlim(x_min, x_max)
    ax_p.set_ylabel(r"$p$")
    ax_p.set_xlabel(r"$x$")

    ax_reset = fig.add_axes([0.15, 0.9, 0.09, 0.035])
    ax_save = fig.add_axes([0.25, 0.9, 0.09, 0.035])
    btn_reset = Button(ax_reset, "Reset")
    btn_save = Button(ax_save, "Save")

    ax_t = fig.add_axes([0.10, 0.22, 0.80, 0.03])
    ax_rho_l = fig.add_axes([0.10, 0.17, 0.35, 0.03])
    ax_vx_l = fig.add_axes([0.10, 0.12, 0.35, 0.03])
    ax_p_l = fig.add_axes([0.10, 0.07, 0.35, 0.03])
    ax_rho_r = fig.add_axes([0.55, 0.17, 0.35, 0.03])
    ax_vx_r = fig.add_axes([0.55, 0.12, 0.35, 0.03])
    ax_p_r = fig.add_axes([0.55, 0.07, 0.35, 0.03])
    ax_gamma = fig.add_axes([0.10, 0.02, 0.80, 0.03])

    sl_t = Slider(ax_t, "time", 0.0, 10.0, valinit=t_end)
    sl_rho_l = Slider(ax_rho_l, r"$\rho_L$", 0.1, 4.0, valinit=left.rho)
    sl_vx_l = Slider(ax_vx_l, r"$u_L$", -2.0, 2.0, valinit=left.u)
    sl_p_l = Slider(ax_p_l, r"$p_L$", 0.1, 4.0, valinit=left.p)
    sl_rho_r = Slider(ax_rho_r, r"$\rho_R$", 0.1, 4.0, valinit=right.rho)
    sl_vx_r = Slider(ax_vx_r, r"$u_R$", -2.0, 2.0, valinit=right.u)
    sl_p_r = Slider(ax_p_r, r"$p_R$", 0.1, 4.0, valinit=right.p)
    sl_gamma = Slider(ax_gamma, r"$\gamma$", 1.1, 2.0, valinit=gamma)

    default_slider_values = {
        sl_t: t_end,
        sl_rho_l: left.rho,
        sl_vx_l: left.u,
        sl_p_l: left.p,
        sl_rho_r: right.rho,
        sl_vx_r: right.u,
        sl_p_r: right.p,
        sl_gamma: gamma,
    }

    def scale_axes(_=None) -> None:
        autoscale_axes(
            fig,
            [
                (ax_rho, (ic_line_rho, sol_line_rho)),
                (ax_vx, (ic_line_vx, sol_line_vx)),
                (ax_p, (ic_line_p, sol_line_p)),
            ],
        )

    def reset_controls(_=None) -> None:
        for slider, value in default_slider_values.items():
            slider.set_val(value)
        update()

    def save_figure(_=None) -> None:
        fig.savefig(figure_path, dpi=SAVE_DPI)
        print(f"Saved figure to {figure_path}")

    def update(_=None) -> None:
        t = sl_t.val
        rho_l_val = sl_rho_l.val
        vx_l_val = sl_vx_l.val
        p_l_val = sl_p_l.val
        rho_r_val = sl_rho_r.val
        vx_r_val = sl_vx_r.val
        p_r_val = sl_p_r.val
        gamma_val = sl_gamma.val

        ic_line_rho.set_ydata(piecewise_ic(x, x0, rho_l_val, rho_r_val))
        ic_line_vx.set_ydata(piecewise_ic(x, x0, vx_l_val, vx_r_val))
        ic_line_p.set_ydata(piecewise_ic(x, x0, p_l_val, p_r_val))

        gas = classical_gas(
            rho_l_val,
            vx_l_val,
            p_l_val,
            rho_r_val,
            vx_r_val,
            p_r_val,
            t,
            gamma_val,
            x=x,
            x0=x0,
        )
        sol_line_rho.set_ydata(gas.rho)
        sol_line_vx.set_ydata(gas.ux)
        sol_line_p.set_ydata(gas.p)
        scale_axes()

    btn_reset.on_clicked(reset_controls)
    btn_save.on_clicked(save_figure)
    for slider in (
        sl_t,
        sl_rho_l,
        sl_vx_l,
        sl_p_l,
        sl_rho_r,
        sl_vx_r,
        sl_p_r,
        sl_gamma,
    ):
        slider.on_changed(update)

    update()
    plt.show()
    return 0
