#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
# -----------------------------------------------------------
# Configure plot; Note that is requires latex to be installed
# -----------------------------------------------------------

fontsize   = 15
markersize = 50  # Adjust size of points in plot
latex      = True # Set to True if latex is installed
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
# First extract states from file
# -----------------------------------------------------------
filename = sys.argv[1]
file = open(filename, "r")

lines = file.readlines()
xArray = []
yArray = []
eDensity = []
states = []
x, y = [], []

for line in lines:
    if line[0] == "#":
        states.append(eDensity[1:])       
        holePosition = [x[0], y[0]]
        xArray = np.array(x[1:])
        yArray = np.array(y[1:])     
        x, y, eDensity = [], [], []
        continue
    
    lineData = line.split()
    x.append(float(lineData[0]))
    y.append(float(lineData[1]))
    eDensity.append(float(lineData[2]))

xArray = np.array(xArray)
yArray = np.array(yArray)

# -----------------------------------------------------------
# Sum densities over degeneracies
# -----------------------------------------------------------

degeneracies = [1, 1, 1, 1, 1, 1, 1, 1]
wfs = []
counter = 0
for degeneracy in degeneracies:
    wf = np.array(states[counter])
    for stateIdx in range(counter + 1, degeneracy + counter):
        wf += np.array(states[stateIdx])
    wfs.append(wf)
    counter += degeneracy

# -----------------------------------------------------------
# Plot states
# -----------------------------------------------------------

fig, ax = plt.subplots(4, 2, figsize=(6, 8), sharex=True, sharey=True)

state = wfs[0] / np.max(wfs[0])
ax[0, 0].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[0, 0].scatter(0, 0, c="r", label="Hole", s=markersize)
ax[0, 0].legend(loc="upper left")

state = wfs[1] / np.max(wfs[1])
ax[0, 1].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[0, 1].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[2] / np.max(wfs[2])
ax[1, 0].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[1, 0].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[3] / np.max(wfs[3])
ax[1, 1].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[1, 1].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[4] / np.max(wfs[4])
ax[2, 0].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[2, 0].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[5] / np.max(wfs[5])
ax[2, 1].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[2, 1].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[6] / np.max(wfs[6])
ax[3, 0].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[3, 0].scatter(0, 0, c="r", label="Hole", s=markersize)

state = wfs[7] / np.max(wfs[7])
ax[3, 1].scatter(xArray - holePosition[0], yArray - holePosition[1], c=state, cmap="Greens", s=markersize - markersize*(state - 1)**10)
ax[3, 1].scatter(0, 0, c="r", label="Hole", s=markersize)

for axis_row in ax:
    for axis in axis_row:
        axis.axis("square")
        for side in ['top','bottom','left','right']:
            axis.spines[side].set_linewidth(2)
# ax[2, 1].axis("off")

ylim = max(yArray)    
ax[0, 0].set_ylim([-ylim, ylim])
xlim = max(xArray)
ax[0, 0].set_xlim([-xlim, xlim])
# ax[0].set_xlabel(r"$k_x$ (\AA$^{-1}$)")

ax[0, 0].set_ylabel(r"$y$ (\AA)")
ax[1, 0].set_ylabel(r"$y$ (\AA)")
ax[2, 0].set_ylabel(r"$y$ (\AA)")
ax[3, 0].set_ylabel(r"$y$ (\AA)")
ax[3, 0].set_xlabel(r"$x$ (\AA)")
ax[3, 1].set_xlabel(r"$x$ (\AA)")
ax[1, 1].set_xticks([-xlim, 0, xlim])

ax[1,1].tick_params(labelbottom=True)
plt.tight_layout()
plt.savefig("rswf.png", dpi=300)