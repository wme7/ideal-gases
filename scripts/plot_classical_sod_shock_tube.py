import numpy as np
from ideal_gases import classical_gas
from matplotlib import pyplot as plt

x = np.linspace(0.0, 1.0, 101)
result = classical_gas(
    rho_l=1.0,
    u_l=0.0,
    p_l=1.0,
    rho_r=0.125,
    u_r=0.0,
    p_r=0.1,
    t_end=0.2,
    gamma=1.4,
    x=x,
    x0=0.5,
)

# Plot results in subplots
fig, axes = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
axes[0, 0].plot(x, result.rho, '-b', label="$\\rho$")
axes[0, 0].legend(loc="best")
axes[0, 0].set_ylabel("$\\rho$")
axes[0, 1].plot(x, result.ux, '-b', label="$u_x$")
axes[0, 1].legend(loc="best")
axes[0, 1].set_ylabel("$u_x$")
axes[0, 2].plot(x, result.p, '-b', label="$p$")
axes[0, 2].legend(loc="best")
axes[0, 2].set_ylabel("$p$")
axes[1, 0].plot(x, result.e, '-b', label="$e$")
axes[1, 0].legend(loc="best")
axes[1, 0].set_ylabel("$e$")
axes[1, 1].plot(x, result.z, '-b', label="$z$")
axes[1, 1].legend(loc="best")
axes[1, 1].set_ylabel("$z$")
axes[1, 2].plot(x, result.t, '-b', label="$\\theta$")
axes[1, 2].legend(loc="best")
axes[1, 2].set_ylabel("$\\theta$")
axes[1, 0].set_xlabel("x")
axes[1, 1].set_xlabel("x")
axes[1, 2].set_xlabel("x")
plt.show()