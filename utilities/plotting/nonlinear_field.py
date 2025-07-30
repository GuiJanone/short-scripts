#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorcet as cc
import sys

# -----------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------
fontsize   = 15
markersize = 110
latex      = True
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
# Setup
# -----------------------------------------------------------
def compare_all(filename):
    sp_file = filename
    y_column = 1  # Index of the column to plot

    # 1) Collect folders that are field values
    folders = []
    for f in os.listdir():
        if os.path.isdir(f):
            try:
                float(f)
                folders.append(f)
            except ValueError:
                continue
    folders = sorted(folders, key=lambda x: float(x))
    # Apply mask: keep only certain field ranges or values
    min_field = -0.010
    max_field =  0.010
    folders = [f for f in folders if min_field <= float(f) <= max_field]


    # 2) Prepare figure and colormap
    fig, ax = plt.subplots(figsize=(8, 5))
    field_vals = np.array([float(f) for f in folders])
    norm = colors.Normalize(vmin=field_vals.min(), vmax=field_vals.max())
    cmap = cm.get_cmap("cet_bkr")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    # 3) Plot each curve with its corresponding color
    for folder in folders:
        filepath = os.path.join(folder, sp_file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        data = np.loadtxt(filepath)
        energy = data[:, 0]
        sigma = data[:, y_column]
        E_field = float(folder)

        mask = energy <= 2.5
        energy = energy[mask]
        sigma = sigma[mask]

        ax.plot(energy, sigma, color=sm.to_rgba(E_field), alpha=0.7)

    # 4) Configure axes and colorbar
    ax.set_xlabel(r"$E_{\mathrm{photon}}$ (eV)", fontsize=18)
    ax.set_ylabel(r"$\sigma^{(2)}_{xxx}$ ($\mu$A/V$^2 \cdot$nm)", fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', length=5)

    ax.set_xlim(1.5, 2.5)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"Field Intensity (eV/\AA)", fontsize=14)

    plt.tight_layout()
    plt.savefig("field_dependence_masked.png", dpi=400)
    plt.show()

#===============================================================================#
# zero vs field 
#===============================================================================#
def compare_two(filename, folder1, folder2):

    path = str.join("./", folder1, "/", filename)
    energy, y_data1 = np.loadtxt(path, unpack=True, usecols=(0,1))

    path = str.join("./", folder2, "/", filename)
    energy, y_data2 = np.loadtxt(path, unpack=True, usecols=(0,1))

    fig, ax = plt.subplots(figsize=(8, 5))


    ax.plot(energy, y_data1, label=f"0.000", color='r')
    ax.plot(energy2, y_data2, label=f"0.002", color='g')

    ax.set_xlabel(r"$E_{photon}$ (eV)", fontsize=18)
    ax.set_ylabel(r"$\sigma^{(2)}_{xxx}\ (\mu$A/V$^2 \cdot nm)$", fontsize=18)
    ax.set_xlim(1.5, 4.0)
    ax.set_ylim(-25, 25)
    ax.set_xticks(np.arange(1.5, 4.1, 0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', length=7)
    # ax.set_xticks(np.arange(1.5, 3.1, 0.25))

    ax.legend(title=r"Field (eV/\AA)", fontsize=12)
    plt.tight_layout()
    plt.savefig("field_comapare.png", dpi=400)

    return

#===============================================================================#
# 2D heatmap: omega x E_dc x sigma (INTERPOALTED)
#===============================================================================#
from scipy.interpolate import griddata

import os
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import TwoSlopeNorm

def contour_fields_interpolated(filename):
    sp_file = filename
    y_column = 2  # column for conductivity

    # 1) Pick up folder names that parse as floats (including negatives)
    folders = []
    for f in os.listdir():
        if os.path.isdir(f):
            try:
                _ = float(f)
                folders.append(f)
            except ValueError:
                continue
    folders = sorted(folders, key=lambda x: float(x))

    field_intensities = []
    all_energy = []
    all_sigma = []

    for folder in folders:
        filepath = os.path.join(folder, sp_file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        data = np.loadtxt(filepath)
        energy = data[:, 0]
        sigma  = data[:, y_column]

        E_field = float(folder)
        field_intensities.extend([E_field] * len(energy))
        all_energy.extend(energy)
        all_sigma.extend(sigma)

    # Filter out energy > 2.0 eV to cap the plot
    field_intensities = np.array(field_intensities)
    all_energy        = np.array(all_energy)
    all_sigma         = np.array(all_sigma)

    # masking to get initial response near gap
    # mask = all_energy <= 1.73                
    # all_energy = all_energy[mask]
    # all_sigma = all_sigma[mask]
    # field_intensities = field_intensities[mask]

    # 2) INTERPOLATED contour
    xi = np.linspace(all_energy.min(), all_energy.max(), 1000)
    yi = np.linspace(field_intensities.min(), field_intensities.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((all_energy, field_intensities), all_sigma,
                  (Xi, Yi), method="linear")
    # Zi = np.log(np.abs(Zi)+1e-5)
    # Zi = np.sign(Zi) * np.sqrt(np.abs(Zi))  # sqrt scaling to boost low values

    # Normalizing colormap
    vmin = Zi.min()/5     # to enhance smaller values...?
    vmax = Zi.max()/5     # to enhance smaller values...?
    vcenter = 0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = get_cmap("cet_bkr")

    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(Xi, Yi*1000, Zi, levels=800, cmap=cmap, norm=norm)  # filled contours
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r"$\sigma^{(2)}_{xxx}$ ($\mu\,$A/V$^2$)", rotation=270, labelpad=20)

    # cbar.set_ticks([vmin, 0, np.floor(vmax)])
    # cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])  # Optional formatting

    levels_contour = np.linspace(vmin, vmax, 10)  # More lines, guaranteed to cover low values
    ax.contour(Xi, Yi*1000, Zi, levels=levels_contour, colors='white', norm=norm)


    ax.set_xlim(1,3.5)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"Field Intensity (m\,eV/\AA)")
    plt.tight_layout()
    plt.savefig("colormap_xyy_E-Field-Sigma.png", dpi=600)
    plt.show()
    return

#===============================================================================#
# 2D heatmap: omega x E_dc x sigma (RAW)
#===============================================================================#
def contour_fields(filename):
    sp_file = filename
    y_column = 1  # column for conductivity

    # 1) Pick up folder names that parse as floats (including negatives)
    folders = []
    for f in os.listdir():
        if os.path.isdir(f):
            try:
                _ = float(f)
                folders.append(f)
            except ValueError:
                continue
    folders = sorted(folders, key=lambda x: float(x))

    field_intensities = []
    all_energy = []
    all_sigma = []

    for folder in folders:
        filepath = os.path.join(folder, sp_file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        data = np.loadtxt(filepath)
        energy = data[:, 0]
        sigma  = data[:, y_column]

        E_field = float(folder)
        field_intensities.extend([E_field] * len(energy))
        all_energy.extend(energy)
        all_sigma.extend(sigma)

    field_intensities = np.array(field_intensities)
    all_energy        = np.array(all_energy)
    all_sigma         = np.array(all_sigma)

    # 3) RAW scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(all_energy, field_intensities*1000, c=all_sigma,
                    cmap="seismic", s=5**2, alpha=0.7,
                    norm=TwoSlopeNorm(vmin=all_sigma.min(),
                                      vcenter=0,
                                      vmax=all_sigma.max()))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\sigma^{(2)}_{xxx}$ ($\mu\,$A/V$^2$)", rotation=270)

    ax.set_xlim(2,3.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"Field Intensity (m\,eV/\AA)")
    plt.tight_layout()
    plt.savefig('raw_colormap.png',dpi=600)
    # plt.show()
#################################################################################
option = int(sys.argv[1])
if option == 1:
    filename = sys.argv[2]
    compare_all(filename)
elif option == 2:
    file1 = sys.argv[2]
    file2 = sys.argv[3]
    compare_two(file1, file2)
elif option == 3:
    filename = sys.argv[2]
    contour_fields_interpolated(filename)
elif option == 4:
    filename = sys.argv[2]
    contour_fields(filename)
else:
    print("xxxxx Plot option not found! xxxxx")
    print("---------------------------------------------------------------")
    print("Choose from: ")
    print("1 -> Plot sigma vs. energy for every field Intensity")
    print("2 -> Plot sigma vs. energy for two field Intensities")
    print("3 -> Plot Interpolated Energy vs. Field Intensity with sigma as the colormap")
    print("4 -> Plot Energy vs. Field Intensity with sigma as the colormap")
    print("---------------------------------------------------------------")