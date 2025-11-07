#!/usr/bin/env python3
"""
compare_jdos_fields_simple.py

Scan numeric-named folders (field values), read 'jdos.dat' in each (two columns:
omega, JDOS), and plot:
  1) N curves of JDOS vs energy (colored by field), or
  2) optionally, the difference baseline(E) - JDOS(E, F), where baseline is the
     F=0 curve if present, otherwise the least-squares intercept at F=0.

Usage examples:
  python compare_jdos_fields_simple.py
  python compare_jdos_fields_simple.py --file jdos.dat --emin 0 --emax 4 --fmax 0.02
  python compare_jdos_fields_simple.py --diff

Args:
  --file : filename inside each numeric folder (default: jdos.dat)
  --emin : min energy (eV)
  --emax : max energy (eV)
  --fmax : include only folders with |field| <= fmax (same units as folder names)
  --diff : plot baseline(E) - JDOS(E,F) and append '_diff' to output name
  --out  : output image filename (default: compare_jdos.png or *_diff.png)
  --title: optional plot title
  --latex: True/False to use LaTeX text rendering (default True)

Notes:
- Folder names must be numeric (e.g., '0.000', '0.010', '-0.005') and are interpreted
  as the applied field (units up to you; colorbar shows mV/Ang if your folders are eV/Ang).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import colorcet as cc  # noqa: F401
CMAP_NAME = "cet_bkr"
CBAR_LABEL = r"Field $E_{\mathrm{DC}}$ (mV/$\mathrm{\AA}$)"


def configure_matplotlib(use_latex=True, fontsize=14):
    plt.rcParams.update({"text.usetex": bool(use_latex), "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize + 2
    plt.rcParams["axes.titlesize"] = fontsize + 4
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize


def find_numeric_folders():
    out = []
    for name in os.listdir():
        if os.path.isdir(name):
            try:
                val = float(name)
                out.append((val, name))
            except ValueError:
                continue
    out.sort(key=lambda t: t[0])
    return out


def collect_series(filename, emin=None, emax=None, fmax=None):
    folders = find_numeric_folders()
    if not folders:
        raise RuntimeError("No numeric-named folders found.")

    energy_ref = None
    series = []
    fields = []

    for fval, folder in folders:
        if fmax is not None and abs(fval) > float(fmax):
            continue

        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            print(f"Warning: missing {path}; skipping.")
            continue

        try:
            data = np.loadtxt(path)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}; skipping.")
            continue

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Warning: {path} has unexpected shape; skipping.")
            continue

        energy = data[:, 0]
        jdos = data[:, 1]

        if energy_ref is None:
            energy_ref = energy
        else:
            if energy.shape[0] != energy_ref.shape[0]:
                print(f"Warning: {path} energy grid length mismatch; skipping.")
                continue
            if not np.allclose(energy, energy_ref, rtol=0, atol=1e-12):
                print(f"Warning: {path} energy values differ; skipping.")
                continue

        m = np.ones_like(energy, dtype=bool)
        if emin is not None:
            m &= (energy >= float(emin))
        if emax is not None:
            m &= (energy <= float(emax))
        if not np.any(m):
            continue

        series.append((fval, jdos[m]))
        fields.append(fval)
        energy_masked = energy[m]

    if not series:
        raise RuntimeError("No valid data found after filtering.")

    return energy_masked, series, np.asarray(fields, dtype=float)


def apply_diff_baseline_minus(energy, series):
    """
    series: list of (field, y(E)).
    Returns new series where y := baseline(E) - y(E).
    Baseline is mean of curves with field ~ 0; if none, LS intercept at F=0.
    """
    fields = np.array([f for f, _ in series], dtype=float)
    Y = np.vstack([y for _, y in series])  # shape (nF, nE)

    tol = max(1e-12, 1e-3 * np.max(np.abs(fields))) if fields.size else 1e-6
    near0 = np.isclose(fields, 0.0, atol=tol, rtol=0.0)

    if np.any(near0):
        baseline = np.mean(Y[near0, :], axis=0)
    else:
        F = fields.reshape(-1, 1)
        A = np.hstack([F, np.ones_like(F)])  # [F, 1]
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        baseline = coef[1, :]  # intercept at F=0

    Ydiff = baseline[np.newaxis, :] - Y
    out = [(f, y) for (f, _), y in zip(series, Ydiff)]
    return out


def insert_diff_suffix(path):
    base, ext = os.path.splitext(path)
    return f"{base}_diff{ext or '.png'}"


def plot_compare(energy, series, fields, out="compare_jdos.png", title=None, diff=False, dpi=600):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Color by field (convert to mV/Ang if your folder values are eV/Ang)
    fields_mV = fields * 1000.0
    vmin, vmax = float(np.min(fields_mV)), float(np.max(fields_mV))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    norm = SymLogNorm(linthresh=2.0, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
    cmap = plt.get_cmap(CMAP_NAME)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for fval, y in series:
        ax.plot(energy, y, color=cmap(norm(fval * 1000.0)), lw=1.6, alpha=0.95)

    ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Delta JDOS (baseline - curve)" if diff else "JDOS (arb. units)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.30, linestyle=":", linewidth=0.8)

    cbar = plt.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label(CBAR_LABEL)
    cbar.locator = MaxNLocator(nbins=8, symmetric=True, prune=None)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.show()
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot JDOS(E) across fields; optional baseline minus curve.")
    ap.add_argument("--file", default="jdos.dat", help="file to read inside each numeric folder (default: jdos.dat)")
    ap.add_argument("--emin", type=float, default=None, help="min energy (eV)")
    ap.add_argument("--emax", type=float, default=None, help="max energy (eV)")
    ap.add_argument("--fmax", type=float, default=None, help="max |field| to include (same units as folder names)")
    ap.add_argument("--diff", action="store_true", help="plot baseline(E) - JDOS(E,F)")
    ap.add_argument("--out", default=None, help="output image filename")
    ap.add_argument("--title", default=None, help="optional plot title")
    ap.add_argument("--latex", default="True", help="use LaTeX text rendering (True/False)")
    args = ap.parse_args()

    use_latex = str(args.latex).lower() == "true"
    configure_matplotlib(use_latex=use_latex)

    energy, series, fields = collect_series(
        filename=args.file, emin=args.emin, emax=args.emax, fmax=args.fmax
    )

    if args.diff:
        series = apply_diff_baseline_minus(energy, series)

    out = args.out or ("compare_jdos_diff.png" if args.diff else "compare_jdos.png")
    plot_compare(energy, series, fields, out=out, title=args.title, diff=args.diff)


if __name__ == "__main__":
    main()
