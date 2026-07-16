# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Interactive quantum Riemann explorer with FD / MB / BE overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ideal_gases.cli.config import QuantumState
from ideal_gases.cli.interactive._widgets import (
    SAVE_DPI,
    autoscale_axes,
    piecewise_ic,
)
from ideal_gases.cli.plot import STATISTIC_COLORS, STATISTICS, _require_matplotlib
from ideal_gases.riemann import quantum_gas


def run_quantum_interactive(
    *,
    left: QuantumState,
    right: QuantumState,
    t_end: float,
    n: float,
    h: float,
    x: NDArray[np.float64],
    x0: float,
    x_min: float,
    x_max: float,
    figure_path: Path,
) -> int:
    plt = _require_matplotlib(show=True)
    from matplotlib.lines import Line2D
    from matplotlib.widgets import Button, CheckButtons, Slider

    ic_rho = piecewise_ic(x, x0, left.rho, right.rho)
    ic_vx = piecewise_ic(x, x0, left.u, right.u)
    ic_theta = piecewise_ic(x, x0, left.theta, right.theta)

    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    fig.subplots_adjust(left=0.18, bottom=0.30)
    fig.suptitle("Interactive quantum Riemann solver (FD / MB / BE)")

    ax_rho = fig.add_subplot(311)
    ax_vx = fig.add_subplot(312)
    ax_theta = fig.add_subplot(313)

    (ic_line_rho,) = ax_rho.plot(x, ic_rho, "--", color="0.5", label="IC")
    (ic_line_vx,) = ax_vx.plot(x, ic_vx, "--", color="0.5")
    (ic_line_theta,) = ax_theta.plot(x, ic_theta, "--", color="0.5")

    solution_lines: dict[str, dict[str, Line2D]] = {}
    for stat in STATISTICS:
        gas = quantum_gas(
            left.rho,
            left.u,
            left.theta,
            right.rho,
            right.u,
            right.theta,
            t_end,
            n,
            h,
            stat,
            x=x,
            x0=x0,
        )
        solution_lines[stat] = {
            "rho": ax_rho.plot(
                x, gas.rho, color=STATISTIC_COLORS[stat], linewidth=2, label=stat
            )[0],
            "ux": ax_vx.plot(x, gas.ux, color=STATISTIC_COLORS[stat], linewidth=2)[0],
            "theta": ax_theta.plot(x, gas.t, color=STATISTIC_COLORS[stat], linewidth=2)[
                0
            ],
        }

    ax_rho.set_xticks([])
    ax_rho.set_xlim(x_min, x_max)
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.legend(loc="upper right", fontsize=8)

    ax_vx.set_xticks([])
    ax_vx.set_xlim(x_min, x_max)
    ax_vx.set_ylabel(r"$u_x$")

    ax_theta.set_xlim(x_min, x_max)
    ax_theta.set_ylabel(r"$\theta$")
    ax_theta.set_xlabel(r"$x$")

    ax_check = fig.add_axes([0.19, 0.9, 0.2, 0.035])
    checks = CheckButtons(
        ax_check,
        STATISTICS,
        actives=[True, True, True],
        layout="horizontal",
    )

    ax_reset = fig.add_axes([0.40, 0.9, 0.09, 0.035])
    ax_save = fig.add_axes([0.50, 0.9, 0.09, 0.035])
    btn_reset = Button(ax_reset, "Reset")
    btn_save = Button(ax_save, "Save")

    ax_t = fig.add_axes([0.10, 0.22, 0.80, 0.03])
    ax_rho_l = fig.add_axes([0.10, 0.17, 0.35, 0.03])
    ax_vx_l = fig.add_axes([0.10, 0.12, 0.35, 0.03])
    ax_theta_l = fig.add_axes([0.10, 0.07, 0.35, 0.03])
    ax_rho_r = fig.add_axes([0.55, 0.17, 0.35, 0.03])
    ax_vx_r = fig.add_axes([0.55, 0.12, 0.35, 0.03])
    ax_theta_r = fig.add_axes([0.55, 0.07, 0.35, 0.03])
    ax_n = fig.add_axes([0.10, 0.02, 0.35, 0.03])
    ax_h = fig.add_axes([0.55, 0.02, 0.35, 0.03])

    sl_t = Slider(ax_t, "time", 0.0, 10.0, valinit=t_end)
    sl_rho_l = Slider(ax_rho_l, r"$\rho_L$", 0.1, 4.0, valinit=left.rho)
    sl_vx_l = Slider(ax_vx_l, r"$u_L$", -2.0, 2.0, valinit=left.u)
    sl_theta_l = Slider(ax_theta_l, r"$\theta_L$", 0.1, 4.0, valinit=left.theta)
    sl_rho_r = Slider(ax_rho_r, r"$\rho_R$", 0.1, 4.0, valinit=right.rho)
    sl_vx_r = Slider(ax_vx_r, r"$u_R$", -2.0, 2.0, valinit=right.u)
    sl_theta_r = Slider(ax_theta_r, r"$\theta_R$", 0.1, 4.0, valinit=right.theta)
    sl_n = Slider(ax_n, "n", 1, 5, valinit=n, valstep=1, valfmt="%d")
    sl_h = Slider(ax_h, "h", 0.1, 10.0, valinit=h)

    default_slider_values = {
        sl_t: t_end,
        sl_rho_l: left.rho,
        sl_vx_l: left.u,
        sl_theta_l: left.theta,
        sl_rho_r: right.rho,
        sl_vx_r: right.u,
        sl_theta_r: right.theta,
        sl_n: n,
        sl_h: h,
    }

    def active_statistics() -> list[str]:
        return [stat for stat, on in zip(STATISTICS, checks.get_status()) if on]

    def scale_axes(_=None) -> None:
        autoscale_axes(
            fig,
            [
                (
                    ax_rho,
                    (
                        ic_line_rho,
                        *(solution_lines[stat]["rho"] for stat in STATISTICS),
                    ),
                ),
                (
                    ax_vx,
                    (
                        ic_line_vx,
                        *(solution_lines[stat]["ux"] for stat in STATISTICS),
                    ),
                ),
                (
                    ax_theta,
                    (
                        ic_line_theta,
                        *(solution_lines[stat]["theta"] for stat in STATISTICS),
                    ),
                ),
            ],
        )

    def reset_controls(_=None) -> None:
        for slider, value in default_slider_values.items():
            slider.set_val(value)
        for index, active in enumerate((True, True, True)):
            if checks.get_status()[index] != active:
                checks.set_active(index)
        update()

    def save_figure(_=None) -> None:
        fig.savefig(figure_path, dpi=SAVE_DPI)
        print(f"Saved figure to {figure_path}")

    def on_check(label: str | None) -> None:
        if label is None:
            return
        if not any(checks.get_status()):
            checks.set_active(STATISTICS.index(label))
        update()

    def update(_=None) -> None:
        t = sl_t.val
        rho_l_val = sl_rho_l.val
        vx_l_val = sl_vx_l.val
        theta_l_val = sl_theta_l.val
        rho_r_val = sl_rho_r.val
        vx_r_val = sl_vx_r.val
        theta_r_val = sl_theta_r.val
        n_val = int(sl_n.val)
        h_val = sl_h.val

        ic_line_rho.set_ydata(piecewise_ic(x, x0, rho_l_val, rho_r_val))
        ic_line_vx.set_ydata(piecewise_ic(x, x0, vx_l_val, vx_r_val))
        ic_line_theta.set_ydata(piecewise_ic(x, x0, theta_l_val, theta_r_val))

        active = active_statistics()
        for stat in STATISTICS:
            visible = stat in active
            for line in solution_lines[stat].values():
                line.set_visible(visible)
            if not visible:
                continue

            gas = quantum_gas(
                rho_l_val,
                vx_l_val,
                theta_l_val,
                rho_r_val,
                vx_r_val,
                theta_r_val,
                t,
                n_val,
                h_val,
                stat,
                x=x,
                x0=x0,
            )
            solution_lines[stat]["rho"].set_ydata(gas.rho)
            solution_lines[stat]["ux"].set_ydata(gas.ux)
            solution_lines[stat]["theta"].set_ydata(gas.t)

        scale_axes()

    btn_reset.on_clicked(reset_controls)
    btn_save.on_clicked(save_figure)
    checks.on_clicked(on_check)
    for slider in (
        sl_t,
        sl_rho_l,
        sl_vx_l,
        sl_theta_l,
        sl_rho_r,
        sl_vx_r,
        sl_theta_r,
        sl_n,
        sl_h,
    ):
        slider.on_changed(update)

    update()
    plt.show()
    return 0
