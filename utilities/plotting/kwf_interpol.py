import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import sys

# -----------------------------------------------------------
# Configure plot
# -----------------------------------------------------------

fontsize = 15
latex = True  # Set to True if LaTeX is installed
if latex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["font.size"] = fontsize

# -----------------------------------------------------------
# Read states from file
# -----------------------------------------------------------
filename = sys.argv[1]
file = open(filename, "r")
lines = file.readlines()
kpoints = []
ks = []
coefs = []
states = []
for line in lines:
    if line[0] == "k":
        continue
    if line[0] == "#":
        states.append(np.array(coefs))
        ks.append(kpoints)
        kpoints = []
        coefs = []
        continue
    line = line.split()
    kpoint = [float(num) for num in line[0:3]]
    kpoints.append(kpoint)
    coefs.append(float(line[-1]))

kpoints = np.array(ks[0])
file.close()

# -----------------------------------------------------------
# Sum densities over degeneracies
# -----------------------------------------------------------

degeneracies = [1, 1, 1, 1, 1, 1, 1, 1]
wfs = []
counter = 0
for deg in degeneracies:
    state = np.zeros(states[0].shape)
    for i in range(deg):
        state += states[counter + i]
    counter += deg
    wfs.append(state)

# -----------------------------------------------------------
# Create interpolation grid
# -----------------------------------------------------------

# Define fine grid resolution
grid_res = 500  # Increase for finer interpolation
k_x = np.linspace(min(kpoints[:, 0]), max(kpoints[:, 0]), grid_res)
k_y = np.linspace(min(kpoints[:, 1]), max(kpoints[:, 1]), grid_res)
k_x_grid, k_y_grid = np.meshgrid(k_x, k_y)

# Interpolated states
interpolated_wfs = []
for wf in wfs:
    interpolated_wf = griddata((kpoints[:, 0], kpoints[:, 1]), wf, (k_x_grid, k_y_grid), method="cubic")
    interpolated_wfs.append(interpolated_wf)

# -----------------------------------------------------------
# Plot states
# -----------------------------------------------------------

fig, ax = plt.subplots(4, 2, figsize=(8, 10), sharex=True, sharey=True)

# Use smoother interpolation instead of tripcolor
ax[0, 0].contourf(k_x_grid, k_y_grid, interpolated_wfs[0], levels=100, cmap="viridis")
ax[0, 1].contourf(k_x_grid, k_y_grid, interpolated_wfs[1], levels=100, cmap="viridis")
ax[1, 0].contourf(k_x_grid, k_y_grid, interpolated_wfs[2], levels=100, cmap="viridis")
ax[1, 1].contourf(k_x_grid, k_y_grid, interpolated_wfs[3], levels=100, cmap="viridis")
ax[2, 0].contourf(k_x_grid, k_y_grid, interpolated_wfs[4], levels=100, cmap="viridis")
ax[2, 1].contourf(k_x_grid, k_y_grid, interpolated_wfs[5], levels=100, cmap="viridis")
ax[3, 0].contourf(k_x_grid, k_y_grid, interpolated_wfs[6], levels=100, cmap="viridis")
ax[3, 1].contourf(k_x_grid, k_y_grid, interpolated_wfs[7], levels=100, cmap="viridis")

for axis_row in ax:
    for axis in axis_row:
        axis.axis("square")
        for side in ['top', 'bottom', 'left', 'right']:
            axis.spines[side].set_linewidth(2)

xlim = max(kpoints[:, 0])
ylim = max(kpoints[:, 1])
# xlim = 0.3
# ylim = 0.3
ax[0, 0].set_xlim([-xlim, xlim])
ax[0, 0].set_ylim([-ylim, ylim])

ax[0, 0].set_ylabel(r"$k_y$ (\AA$^{-1}$)")
ax[1, 0].set_ylabel(r"$k_y$ (\AA$^{-1}$)")
ax[2, 0].set_ylabel(r"$k_y$ (\AA$^{-1}$)")

ax[1, 1].set_xlabel(r"$k_x$ (\AA$^{-1}$)")
ax[1, 1].set_xticks([-xlim, 0, xlim])
ax[3, 0].set_xlabel(r"$k_x$ (\AA$^{-1}$)")
ax[3, 0].set_xticks([-xlim, 0, xlim])

ax[2, 1].axis("off")
# ax[2, 1].axis("off")

fig.tight_layout()
plt.savefig("kwf_interpolated.png", dpi=300)
plt.show()
