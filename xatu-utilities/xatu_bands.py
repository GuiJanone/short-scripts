#!/usr/bin/env python3
"""
Quick plot of a XATU band structure.

`xatu -b KPOINTS` writes KPOINTS.bands with one row of band energies per
k-point listed in KPOINTS. The x axis is the accumulated |dk| along the list.
With --nocc the energies are referred to the valence band maximum and the
fundamental and direct gaps are printed.

Examples:
  python xatu_bands.py kpoints.txt
  python xatu_bands.py kpoints.txt --nocc 26 --emin -3 --emax 3
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from xatu_common import configure_matplotlib


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot a XATU .bands file.")
    p.add_argument("kpoints", help="k-point file given to xatu -b (kx ky kz per line).")
    p.add_argument("--bands", default=None, help="Bands file. Default: KPOINTS.bands.")
    p.add_argument("--nocc", type=int, default=None,
                   help="Number of occupied bands: shifts energies to the VBM and prints the gaps.")
    p.add_argument("--emin", type=float, default=None, help="Lower energy limit.")
    p.add_argument("--emax", type=float, default=None, help="Upper energy limit.")
    p.add_argument("-o", "--out", default=None, help="Output image. Default: KPOINTS_bands.png.")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    p.add_argument("--no-latex", action="store_true", help="Disable LaTeX text rendering.")
    args = p.parse_args(argv)

    kpoints = np.loadtxt(args.kpoints, usecols=(0, 1, 2), ndmin=2)
    bands = np.loadtxt(args.bands or args.kpoints + ".bands", ndmin=2)
    if len(kpoints) != len(bands):
        print(f"{len(kpoints)} k-points but {len(bands)} rows of bands.", file=sys.stderr)
        return 2

    kpath = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(kpoints, axis=0), axis=1))))
    nbands = bands.shape[1]
    print(f"{len(kpoints)} k-points, {nbands} bands, E in [{bands.min():.4f}, {bands.max():.4f}] eV")

    ylabel = "E (eV)"
    if args.nocc is not None:
        if not 0 < args.nocc < nbands:
            print(f"--nocc must be between 1 and {nbands - 1}.", file=sys.stderr)
            return 2
        vb, cb = bands[:, args.nocc - 1], bands[:, args.nocc]
        ivbm, icbm, idir = np.argmax(vb), np.argmin(cb), np.argmin(cb - vb)
        vbm = vb[ivbm]
        print(f"VBM {vbm:.4f} eV at k = {kpoints[ivbm]}")
        print(f"CBM {cb[icbm]:.4f} eV at k = {kpoints[icbm]}")
        print(f"gap {cb[icbm] - vbm:.4f} eV ({'direct' if ivbm == icbm else 'indirect'}), "
              f"smallest direct gap {cb[idir] - vb[idir]:.4f} eV at k = {kpoints[idir]}")
        bands = bands - vbm
        ylabel = r"E - E$_\mathrm{VBM}$ (eV)"

    configure_matplotlib(latex=not args.no_latex)
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in range(nbands):
        occupied = args.nocc is not None and n < args.nocc
        ax.plot(kpath, bands[:, n], color="tab:blue" if occupied else "k", lw=1.2)
    if args.nocc is not None:
        ax.axhline(0.0, color="gray", linestyle="--", lw=0.8)
    ax.set_xlim(kpath[0], kpath[-1])
    ax.set_ylim(args.emin, args.emax)
    ax.set_xlabel("k path")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    out = args.out or f"{args.kpoints.rsplit('.', 1)[0]}_bands.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
