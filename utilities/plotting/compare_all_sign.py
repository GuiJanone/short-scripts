#!/usr/bin/env python3
"""
compare_all.py

Sweep numeric-named folders (field values), read a file in each,
and plot a single selected column vs photon energy.

Hardcoded sign filter:
  Set KEEP_ONLY_SIGN = "positive" or "negative" to keep only that side (zero is excluded).
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
from matplotlib.colors import LinearSegmentedColormap

# custom colormap (kept as in your script)
CMAP_NAME = LinearSegmentedColormap.from_list(
    "black_red", [(0.0, "black"), (0.5, "#8b0000"), (1.0, "red")]
)
CBAR_LABEL = r"Field $E_{\mathrm{DC}}$ (meV/$\mathrm{\AA}$)"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Hardcoded sign filter: choose "positive" or "negative"
KEEP_ONLY_SIGN = "positive"   # change to "negative" if you want only negative fields
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def configure_matplotlib(fontsize=15):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize+2
    plt.rcParams["axes.titlesize"] = fontsize+4
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

def validate_ycol(ycol_str, ncols_data):
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

def _keep_sign(fval: float) -> bool:
    """Return True if this field value passes the hardcoded sign filter."""
    if KEEP_ONLY_SIGN.lower() == "positive":
        return fval > 0.0   # strictly positive; excludes 0
    if KEEP_ONLY_SIGN.lower() == "negative":
        return fval < 0.0   # strictly negative; excludes 0
    # fallback: no filtering if misconfigured
    return True

def collect_field_data(filename, ycol, label, emin=None, emax=None, fmax=None):
    folders = find_numeric_folders()
    if not folders:
        raise RuntimeError("No numeric-named folders found in the current directory.")

    energy_ref = None
    series = []
    fields_used = []

    for fval, folder in folders:
        # hardcoded sign filter
        if not _keep_sign(fval):
            continue

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

        series.append((fval, data[:, ycol][m]))
        fields_used.append(fval)

    if energy_ref is None or not fields_used:
        raise RuntimeError("No valid data found. Check filenames, masks, and folder structure.")

    # Use the masked energy grid of the last processed file (all are identical by construction)
    energy_m = energy[m]
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
    tol = max(1e-12, 1e-3 * np.max(np.abs(fields))) if fields.size else 1e-6
    near0 = np.isclose(fields, 0.0, atol=tol, rtol=0.0)

    if np.any(near0):
        baseline = np.mean(Y[near0, :], axis=0)
    else:
        F = fields.reshape(-1, 1)
        A = np.hstack([F, np.ones_like(F)])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        baseline = coef[1, :]  # intercept at F=0

    Ydiff = Y - baseline[np.newaxis, :]
    out = [(f, y) for (f, _), y in zip(series, Ydiff)]
    return out

def _build_norm_and_ticks(fields):
    """Build the color norm and ticks based on sign of fields (meV/Å)."""
    fields_meV = fields * 1000.0
    fmin, fmax = float(np.nanmin(fields_meV)), float(np.nanmax(fields_meV))

    if fmin < 0.0 and fmax > 0.0:
        norm = TwoSlopeNorm(vmin=min(fmin, -fmax), vcenter=0.0, vmax=max(fmax, -fmin))
        ticks = np.linspace(-max(abs(fmin), abs(fmax)), max(abs(fmin), abs(fmax)), 9)
    elif fmax <= 0.0:
        norm = Normalize(vmin=fmin, vmax=0.0)
        ticks = np.linspace(fmin, 0.0, 8)
    else:
        norm = Normalize(vmin=0.0, vmax=fmax)
        ticks = np.linspace(0.0, fmax, 8)
    return norm, ticks

def plot_compare_all(energy, series, fields, label,
                     out="compare_all.png", title=None, diff=False, series_diff=None):
    """
    If diff==False -> single axes with 'series'.
    If diff==True  -> two subplots:
        top    : original 'series'
        bottom : 'series_diff' (Δsigma)
    Both panels share the SAME colormap/norm and a single colorbar.
    """
    # shared color mapping
    norm, ticks = _build_norm_and_ticks(fields)
    cmap = CMAP_NAME if isinstance(CMAP_NAME, LinearSegmentedColormap) else plt.get_cmap(CMAP_NAME)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # layout
    if not diff:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        axes = [ax]
        panels = [(ax, series, rf"$\sigma^{{(2)}}_{{{label}}}$ (nm$\cdot\mu$A/V$^2$)")]
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9, 7.0), sharex=True,
            gridspec_kw=dict(hspace=0.10, height_ratios=[1.0, 0.85])
        )
        axes = [ax_top, ax_bot]
        panels = [
            (ax_top, series, rf"$\sigma^{{(2)}}_{{{label}}}$ (nm$\cdot\mu$A/V$^2$)"),
            (ax_bot, series_diff if series_diff is not None else [],
             rf"$\Delta\sigma^{{(2)}}_{{{label}}}$ (nm$\cdot\mu$A/V$^2$)")
        ]

    # plot panels
    for ax, panel_series, ylabel in panels:
        for fval, y in panel_series:
            ax.plot(energy, y, color=cmap(norm(fval * 1000.0)), alpha=0.95, lw=1.8)
        ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
        xticks = np.linspace(energy.min(), energy.max(), 10)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks(xticks)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.30, linestyle=":", linewidth=0.8)

    axes[-1].set_xlabel("Photon energy (eV)")

    # shared colorbar
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label(CBAR_LABEL)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved: {out}")
    # plt.show()

def insert_diff_suffix(path):
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

    # keep BOTH: original and diff
    series_diff = apply_diff(energy, series) if args.diff else None

    title = args.title if args.title is not None else os.path.basename(args.file)
    plot_compare_all(
        energy, series, fields, label=label, out=out, title=title,
        diff=args.diff, series_diff=series_diff
    )

if __name__ == "__main__":
    main()
