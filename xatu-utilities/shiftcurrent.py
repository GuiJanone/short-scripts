#!/usr/bin/env python3
"""
Shift-current plotter.

Schemas, chosen by filename:
1) Regular filenames: columns E(omega), xxx, xyy, yyy, yxx
2) Filenames ending in '_z' before the extension (e.g. 'foo_z.dat'):
   columns E(omega), xxz, yyz, zzz, zxx, zyy

Examples:
  python shiftcurrent.py sigma.dat
  python shiftcurrent.py sigma.dat --components xxx,yyy --emin 1.2 --emax 4.8
  python shiftcurrent.py A.dat B.dat                  # overlay files
  python shiftcurrent.py A.dat B.dat --sum --offset 0.5 1.0
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from xatu_common import configure_matplotlib

REGULAR_KEYS = ["xxx", "xyy", "yyy", "yxx"]
Z_KEYS = ["xxz", "yyz", "zzz", "zxx", "zyy"]


def keys_for(path, ncols):
    base = os.path.splitext(os.path.basename(path))[0]
    keys = Z_KEYS if base.endswith("_z") else REGULAR_KEYS
    return keys[:ncols - 1]


def load_data(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        sys.exit(f"Error: '{path}' does not look like a table with at least 2 columns.")
    return data


def build_parser():
    p = argparse.ArgumentParser(description="Plot shift-current components.")
    p.add_argument("files", nargs="+", help="Shift-current data file(s).")
    p.add_argument("--sum", action="store_true", help="Sum the component columns of all files (same grid required).")
    p.add_argument("--offset", nargs="+", type=float, default=None,
                   help="Energy offset in eV per file, in the same order as the files.")
    p.add_argument("--components", default=None, help="Comma-separated components, e.g. xxx,yyy. Default: all.")
    p.add_argument("--labels", default=None, help="Comma-separated legend labels, one per plotted component.")
    p.add_argument("--emin", type=float, default=None, help="Minimum energy in eV.")
    p.add_argument("--emax", type=float, default=None, help="Maximum energy in eV.")
    p.add_argument("-o", "--out", default="shift_current.png", help="Output image.")
    p.add_argument("--title", default=None, help="Plot title.")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    p.add_argument("--no-latex", action="store_true", help="Disable LaTeX text rendering.")
    p.add_argument("--show", action="store_true", help="Show the plot after saving.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_matplotlib(latex=not args.no_latex)

    offsets = args.offset or [0.0] * len(args.files)
    if len(offsets) != len(args.files):
        sys.exit("Error: give one --offset value per file.")

    datasets = []
    for path, offset in zip(args.files, offsets):
        data = load_data(path)
        data[:, 0] += offset
        datasets.append((path, data))

    if args.sum and len(datasets) > 1:
        shape = datasets[0][1].shape
        if any(data.shape != shape for _, data in datasets):
            sys.exit("Error: --sum needs files with the same number of rows and columns.")
        total = datasets[0][1].copy()
        for _, data in datasets[1:]:
            total[:, 1:] += data[:, 1:]
        datasets = [(datasets[0][0], total)]

    fig, ax = plt.subplots(figsize=(8, 5))
    emin, emax = np.inf, -np.inf
    for path, data in datasets:
        keys = keys_for(path, data.shape[1])
        selection = args.components.split(",") if args.components else keys
        bad = [k for k in selection if k not in keys]
        if bad:
            sys.exit(f"Error: components {bad} not in '{path}'. Available: {keys}")
        labels = args.labels.split(",") if args.labels else selection
        if len(labels) != len(selection):
            sys.exit("Error: --labels needs one label per plotted component.")

        energy = data[:, 0]
        mask = np.ones_like(energy, dtype=bool)
        if args.emin is not None:
            mask &= energy >= args.emin
        if args.emax is not None:
            mask &= energy <= args.emax
        if not np.any(mask):
            sys.exit("Error: --emin/--emax removed all points.")

        prefix = "" if len(datasets) == 1 else f"{os.path.basename(path)}: "
        for key, label in zip(selection, labels):
            ax.plot(energy[mask], data[mask, keys.index(key) + 1], label=prefix + label, alpha=0.9, lw=2.0)
        emin, emax = min(emin, energy[mask][0]), max(emax, energy[mask][-1])

    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(emin, emax)
    ax.set_xlabel(r"Photon Energy $\omega$ (eV)")
    ax.set_ylabel(r"$\sigma^{2}$ (nm $\cdot \mu$A / V$^{2}$)")
    if args.title:
        ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved figure: {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
