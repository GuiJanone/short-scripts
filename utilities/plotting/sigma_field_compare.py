#!/usr/bin/env python3
"""
sigma_vs_field_two_systems.py

Plot sigma vs DC field at a fixed photon energy (omega) for TWO systems.
Each system lives in a directory that contains subfolders named by field
values (floats), e.g., -0.010, 0.000, 0.020, ... Each subfolder contains
the same data file (e.g., 'spectra.dat') with at least two columns:
    col 0: energy (eV)
    col ycol: the sigma component to plot (1-based index)

For every numeric field folder:
  - read the file
  - find the row where energy is closest to --omega
  - take sigma from column --ycol
Then plot field (meV/Ang) vs that sigma, for System A and System B.

Paper-ready: no title; labeled axes and legend only.

Usage examples:
  python sigma_vs_field_two_systems.py --file spectra.dat --ycol 1 --omega 3.10
  python sigma_vs_field_two_systems.py --file data.dat --ycol 5 --omega 2.25 \
      --sysA "/path/to/systemA" --sysB "/path/to/systemB" \
      --labelA "Monolayer" --labelB "Bilayer" --fmax 0.03
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------- Hardcoded defaults (you can change these) ----------
SYSTEM_A_DIR = "/home/visitor/Projects/MoS2/monolayer/CWF/FIELD"   # <-- set your default base path for system A
SYSTEM_B_DIR = "/home/visitor/Projects/MoS2/bilayer/AA/CWF/FIELD"   # <-- set your default base path for system B
# ---------------------------------------------------------------

def find_numeric_folders(base_dir):
    """Return sorted list of (field_value: float, subdir_path) inside base_dir."""
    out = []
    if not os.path.isdir(base_dir):
        print(f"Error: base directory not found: {base_dir}")
        return out
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            try:
                val = float(name)
                out.append((val, p))
            except ValueError:
                continue
    out.sort(key=lambda t: t[0])
    return out

def read_sigma_vs_field(base_dir, filename, ycol, omega, fmax=None):
    """
    For each numeric-named field folder in base_dir, read 'filename',
    pick the row with energy closest to 'omega', and collect sigma from column ycol.
    Returns (fields, sigmas) sorted by field.
    """
    folders = find_numeric_folders(base_dir)
    if not folders:
        raise RuntimeError(f"No numeric-named folders found in {base_dir}")

    fields = []
    sigmas = []

    for fval, fpath in folders:
        if fmax is not None and abs(fval) > float(fmax):
            continue

        ffile = os.path.join(fpath, filename)
        if not os.path.exists(ffile):
            print(f"Warning: missing {ffile}; skipping.")
            continue

        try:
            data = np.loadtxt(ffile)
        except Exception as e:
            print(f"Warning: could not read {ffile}: {e}; skipping.")
            continue

        if data.ndim != 2 or data.shape[1] <= max(0, ycol):
            print(f"Warning: {ffile} shape unexpected or missing column {ycol}; skipping.")
            continue

        E = data[:, 0]
        # nearest energy row
        idx = int(np.argmin(np.abs(E - omega)))
        sigma_val = float(data[idx, ycol])

        fields.append(fval)
        sigmas.append(sigma_val)

    if not fields:
        raise RuntimeError(f"No valid data points found in {base_dir} after filtering.")

    # sort by field
    fields = np.asarray(fields, float)
    sigmas = np.asarray(sigmas, float)
    order = np.argsort(fields)
    return fields[order], sigmas[order]

def nice_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)

def plot_two_systems(fieldsA, sigmaA, fieldsB, sigmaB, labelA, labelB,
                     ylabel, out="sigma_vs_field.png"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))  # good aspect for papers

    # convert to meV/Ang for x-axis
    ax.plot(fieldsA * 1000.0, sigmaA, "-o", color='black', linewidth=1.6, label=labelA, alpha=0.50,
        markerfacecolor='none', markeredgewidth=2, markersize=4)
    ax.plot(fieldsB * 1000.0, sigmaB, "-s", color='red', linewidth=1.6, label=labelB, alpha=0.50,
        markerfacecolor='none', markeredgewidth=2, markersize=4)

    ax.set_xlabel(r"Field $E_{\mathrm{DC}}$ (meV/$\AA$)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    nice_axes(ax)

    # tight legend, no box
    leg = ax.legend(loc="best", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=800)
    print(f"Saved figure: {out}")
    plt.show()

def parse_args():
    p = argparse.ArgumentParser(description="Plot sigma vs field at fixed omega for two systems.")
    p.add_argument("--file", required=True, help="filename to read inside each numeric field folder")
    p.add_argument("--ycol", type=int, default=1, help="1-based sigma column index (col 0 is energy)")
    p.add_argument("--omega", type=float, required=True, help="target photon energy in eV")
    p.add_argument("--sysA", default=SYSTEM_A_DIR, help="base directory for System A (contains numeric field folders)")
    p.add_argument("--sysB", default=SYSTEM_B_DIR, help="base directory for System B (contains numeric field folders)")
    p.add_argument("--labelA", default="System A", help="legend label for System A")
    p.add_argument("--labelB", default="System B", help="legend label for System B")
    p.add_argument("--comp", default="xxx", help="component label (ijk) to show in y-axis, e.g., 'xxx'")
    p.add_argument("--fmax", type=float, default=None, help="max |field| (eV/Ang) to include; default: no limit")
    p.add_argument("--out", default=None, help="output image filename (default auto)")
    return p.parse_args()

def main():
    args = parse_args()

    # validate ycol (1-based)
    # Peek one file to estimate width; if not possible, continue and let loader warn.
    ycol = int(args.ycol)
    if ycol < 1:
        print("Error: --ycol must be >= 1 (col 0 is energy).")
        sys.exit(1)

    # read both systems
    fieldsA, sigmaA = read_sigma_vs_field(
        base_dir=args.sysA, filename=args.file, ycol=ycol, omega=args.omega, fmax=args.fmax
    )
    fieldsB, sigmaB = read_sigma_vs_field(
        base_dir=args.sysB, filename=args.file, ycol=ycol, omega=args.omega, fmax=args.fmax
    )

    # ylabel like sigma^{(2)}_{ijk}
    comp_label = str(args.comp)
    ylabel = rf"$\sigma^{{(2)}}_{{{comp_label}}}$ (nm$\cdot\mu$A/V$^2$)"

    out = args.out if args.out else f"sigma_vs_field_{comp_label}_omega{args.omega:.3f}.png"

    plot_two_systems(fieldsA, sigmaA, fieldsB, sigmaB,
                     labelA=args.labelA, labelB=args.labelB,
                     ylabel=ylabel, out=out)

if __name__ == "__main__":
    main()
