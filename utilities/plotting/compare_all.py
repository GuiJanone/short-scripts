#!/usr/bin/env python3
"""
compare_all.py

Sweep numeric-named folders (field values), read a file in each,
and plot a single selected column vs photon energy.

Args:
  --file   : filename inside each numeric folder
  --ycol   : 1-based column index to plot (col 0 is energy)
  --label  : optional label for that column (used in ylabel and optional legend)
  --emin   : min energy (eV)
  --emax   : max energy (eV)
  --fmax   : max |field| (eV/Ang) from folder names
  --diff   : plot Δsigma(E,F) = sigma(E,F) - sigma(E,0)
  --out    : output image filename (we append _diff if --diff)
  --title  : optional plot title

Curves are colored by field; colorbar shows field in mV/Ang.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, PowerNorm, SymLogNorm
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# keep your colormap
import colorcet as cc  # noqa: F401
CMAP_NAME = "cet_bkr"
CBAR_LABEL = r"Field $E_{\mathrm{DC}}$ (mV/$\mathrm{\AA}$)"


def configure_matplotlib(fontsize=15):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize+2
    plt.rcParams["axes.titlesize"] = fontsize+4
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    # plt.rcParams["font.size"] = fontsize


def find_numeric_folders():
    """Return sorted list of (field_value: float, folder_name: str) for all numeric-named folders."""
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


def validate_ycol(ycol_str, ncols_data):
    """Parse a single 1-based ycol and validate against the data width."""
    if ycol_str is None:
        raise ValueError("--ycol is required (single 1-based column index; col 0 is energy).")
    try:
        idx = int(str(ycol_str).strip())
    except ValueError:
        raise ValueError(f"Invalid --ycol '{ycol_str}': must be an integer (1-based).")
    if idx < 1 or idx >= ncols_data:
        raise ValueError(
            f"--ycol {idx} out of range. Data file has columns 0..{ncols_data-1} "
            f"(col 0 is energy; valid ycol are 1..{ncols_data-1})."
        )
    return idx


def collect_field_data(filename, ycol, label, emin=None, emax=None, fmax=None):
    """
    Scan numeric folders, load `filename` in each, and collect series.

    Returns:
        energy_m : 1D array after masking
        series   : list of (field_value, y_at_masked_energy) for the chosen column
        fields   : 1D array of field values that contributed data
    """
    folders = find_numeric_folders()
    if not folders:
        raise RuntimeError("No numeric-named folders found in the current directory.")

    energy_ref = None
    series = []
    fields_used = []

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
        if energy_ref is None:
            energy_ref = energy
        else:
            if energy.shape[0] != energy_ref.shape[0]:
                print(f"Warning: {path} energy grid length mismatch; skipping.")
                continue

        # energy mask
        m = np.ones_like(energy, dtype=bool)
        if emin is not None:
            m &= (energy >= float(emin))
        if emax is not None:
            m &= (energy <= float(emax))
        if not np.any(m):
            continue

        if ycol >= data.shape[1]:
            print(f"Warning: {path} does not contain column {ycol} ('{label}'); skipping this folder.")
            continue

        energy_m = energy[m]
        series.append((fval, data[:, ycol][m]))
        fields_used.append(fval)

    if energy_ref is None or not fields_used:
        raise RuntimeError("No valid data found. Check filenames, masks, and folder structure.")

    return energy_m, series, np.asarray(fields_used, dtype=float)


def apply_diff(energy, series):
    """
    Given energy (nE,) and series = list[(field, y(nE,))],
    compute baseline sigma(E,0) and subtract from each curve.
    If an exact F=0 curve exists, use it. Else estimate via linear LS fit y ~ a*F + b at each E.
    Returns a new list like series with baseline subtracted.
    """
    fields = np.array([f for f, _ in series], dtype=float)
    Y = np.vstack([y for _, y in series])  # shape (nF, nE)

    # look for exact/near zero field
    tol = max(1e-12, 1e-3 * np.max(np.abs(fields))) if fields.size else 1e-6
    near0 = np.isclose(fields, 0.0, atol=tol, rtol=0.0)

    if np.any(near0):
        baseline = np.mean(Y[near0, :], axis=0)
    else:
        # linear least squares across fields at each energy:
        # Solve [F, 1] * [a; b] ~= Y for all columns at once
        F = fields.reshape(-1, 1)
        A = np.hstack([F, np.ones_like(F)])
        # coefficients shape (2, nE)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        baseline = coef[1, :]  # intercept at F=0

    Ydiff = Y - baseline[np.newaxis, :]
    # rebuild list
    out = [(f, y) for (f, _), y in zip(series, Ydiff)]
    return out


def plot_compare_all(energy, series, fields, label, out="compare_all.png", title=None, diff=False):
    """
    Plot one curve per field for the selected data column.
    Color encodes field value; colorbar shows field in mV/Ang.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # use mV/Ang for the color scale to match the colorbar label
    fields_mV = fields * 1000.0
    vmin, vmax = float(np.min(fields_mV)), float(np.max(fields_mV))
    # norm = Normalize(vmin=vmin, vmax=vmax)

    norm = SymLogNorm(linthresh=2e-0, linscale=1.0,
                  vmin=vmin, vmax=vmax, base=10)
    cmap = plt.get_cmap(CMAP_NAME)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # draw curves
    plotted_any = False
    for fval, y in series:
        ax.plot(energy, y, color=cmap(norm(fval * 1000.0)), alpha=0.95, lw=1.8)
        plotted_any = True

    ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
    n_div = 6
    xticks = np.linspace(energy.min(), energy.max(), n_div)
    ax.set_xticks(xticks)

    if not plotted_any:
        print("Nothing to plot after filtering. Exiting.")
        return

    # colorbar for field value (mV/Ang)
    cbar = plt.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label(CBAR_LABEL)

    # for logarithmic colors
    cbar.locator = MaxNLocator(nbins=8, symmetric=True, prune=None)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # axes labels and cosmetics
    ax.set_xlabel("Photon energy (eV)")
    safe_label = str(label)
    if diff:
        ax.set_ylabel(rf"$\Delta\sigma^{{(2)}}_{{{safe_label}}}$ (nm$\cdot\mu$A/V$^2$)")
    else:
        ax.set_ylabel(rf"$\sigma^{{(2)}}_{{{safe_label}}}$ (nm$\cdot\mu$A/V$^2$)")

    # if title:
    #     ax.set_title(title)
    ax.grid(alpha=0.30, linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved: {out}")
    # plt.show()


def insert_diff_suffix(path):
    """
    Insert '_diff' before the extension of 'path', or append if no extension.
    """
    base, ext = os.path.splitext(path)
    if ext:
        return f"{base}_diff{ext}"
    return f"{path}_diff"


def main():
    ap = argparse.ArgumentParser(description="Compare spectra across fields with field-colored curves (single series).")
    ap.add_argument("--file", required=True, help="file to read inside each numeric folder")
    ap.add_argument("--ycol", required=True,
                    help="1-based column index to plot (col 0 is energy). Example: '1'")
    ap.add_argument("--label", default=None,
                    help="label for the selected column (used in ylabel); default 'col{ycol}'")
    ap.add_argument("--emin", type=float, default=None, help="min energy (eV); default: no lower bound")
    ap.add_argument("--emax", type=float, default=None, help="max energy (eV); default: no upper bound")
    ap.add_argument("--fmax", type=float, default=None, help="max |field| (eV/Ang); default: no field filter")
    ap.add_argument("--diff", action="store_true",
                    help="plot Δsigma(E,F) = sigma(E,F) - sigma(E,0) and append '_diff' to output file")
    ap.add_argument("--out", default=None, help="output image filename")
    ap.add_argument("--title", default=None, help="optional plot title")
    ap.add_argument("--latex", action="store_false", default=True, help="disable LaTeX rendering")
    args = ap.parse_args()


    configure_matplotlib()

    # Peek at one file to validate ycol range
    ncols = None
    for _, folder in find_numeric_folders():
        test_path = os.path.join(folder, args.file)
        if os.path.exists(test_path):
            try:
                tmp = np.loadtxt(test_path)
                if tmp.ndim == 2 and tmp.shape[1] >= 2:
                    ncols = tmp.shape[1]
                    break
            except Exception:
                continue
    if ncols is None:
        print("Error: could not find a readable data file to validate --ycol.")
        sys.exit(1)

    ycol = validate_ycol(args.ycol, ncols)
    label = args.label if args.label is not None else f"col{ycol}"

    # Default output filename if not provided
    out = args.out if args.out is not None else f"compare_fields_{label}.png"
    if args.diff:
        out = insert_diff_suffix(out)

    energy, series, fields = collect_field_data(
        filename=args.file,
        ycol=ycol,
        label=label,
        emin=args.emin,
        emax=args.emax,
        fmax=args.fmax
    )

    # Apply Δ if requested
    if args.diff:
        series = apply_diff(energy, series)

    title = args.title if args.title is not None else os.path.basename(args.file)
    plot_compare_all(energy, series, fields, label=label, out=out, title=title, diff=args.diff)


if __name__ == "__main__":
    main()
