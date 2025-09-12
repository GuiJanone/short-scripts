#!/usr/bin/env python3
"""
2D heatmap: Energy (eV) vs DC field (eV/Ang) of a conductivity component.

- Expects subfolders named by the field value (e.g., "-0.005", "0.000", "0.010").
- In each folder, reads a file with columns: E, sigma_xx, sigma_xy, sigma_xz, sigma_yx, sigma_yy, ...
  Pick which sigma column to plot with --ycol (default 1 for xx).

Defaults:
- No masking by energy gap.
- No field zoom.
- All points are used.

Optional masks:
- Numeric energy cutoff: --emask <float> keeps E <= cutoff.
- "gap" masking: --emask gap --gapfile bandgap_vs_field.dat (2 columns: field, Egap).
  You can adjust with --gap-scale-field and --gap-margin.
- Field zoom: --fzoom <float> keeps |field| <= fzoom.

Optional overlay (no masking):
- --overlay-gap --gapfile bandgap_vs_field.dat
  Plots a dashed line of Egap(field) on top of the heatmap as reference.
  Honors --gap-scale-field and --gap-margin for the overlay curve only.

Examples:
  python surface.py --file shift_sp_shiftvector_field.dat
  python surface.py --file shift_sp_shiftvector_field.dat --overlay-gap --gapfile bandgap_vs_field.dat
  python surface.py --file shift_sp_shiftvector_field.dat --emask 4.8 --fzoom 0.008
  python surface.py --file shift_sp_shiftvector_field.dat --emask gap --gapfile bandgap_vs_field.dat --gap-margin 0.4
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata, interp1d

# Try colorcet if available; else fall back to matplotlib
try:
    import colorcet as cc  # noqa: F401
    DEFAULT_CMAP = "cet_bkr"
except Exception:
    DEFAULT_CMAP = "seismic"


# -----------------------------
# Plot configuration helper
# -----------------------------
def configure_matplotlib(fontsize=15, latex=False):
    if latex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    plt.rcParams["font.size"] = fontsize


# -----------------------------
# Data collection
# -----------------------------
def collect_field_spectra(filename, ycol):
    """
    Scan subfolders whose names parse as floats (field values).
    Read 'filename' in each folder and return stacked arrays of
    (fields, energies, sigma).
    """
    candidates = []
    for f in os.listdir():
        if os.path.isdir(f):
            try:
                val = float(f)
                candidates.append((val, f))
            except ValueError:
                continue
    if not candidates:
        raise RuntimeError("No numeric-named folders (field values) found.")

    # sort by field value
    candidates.sort(key=lambda t: t[0])

    fields, energies, sigma = [], [], []
    for fval, folder in candidates:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            print(f"Warning: missing {path}, skipping.")
            continue
        try:
            data = np.loadtxt(path)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}, skipping.")
            continue
        if data.ndim != 2 or data.shape[1] <= ycol:
            print(f"Warning: {path} does not have column {ycol}, skipping.")
            continue

        E = data[:, 0]
        Y = data[:, ycol]

        # extend with one point per energy at this field
        fields.extend([fval] * len(E))
        energies.extend(E)
        sigma.extend(Y)

    if not energies:
        raise RuntimeError("No valid spectra collected after scanning folders.")

    return np.asarray(fields, float), np.asarray(energies, float), np.asarray(sigma, float)


# -----------------------------
# Helpers
# -----------------------------
def load_two_col_float(path, colx=0, coly=1):
    """
    Robust reader for a 2-column float table.
    - Accepts commas or whitespace.
    - Skips header lines, comments (#), and malformed rows.
    Returns (x, y) as 1D float arrays sorted by x.
    """
    xs, ys = [], []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # strip inline comments
            if "#" in s:
                s = s.split("#", 1)[0].strip()
                if not s:
                    continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) <= max(colx, coly):
                continue
            try:
                x = float(parts[colx])
                y = float(parts[coly])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"No numeric data found in {path}")
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    idx = np.argsort(x)
    return x[idx], y[idx]


# -----------------------------
# Mask logic (all optional)
# -----------------------------
def apply_masks(fields, energies, sigma, emask=None, fzoom=None,
                gapfile=None, gap_scale_field=1.0, gap_margin=0.0):
    """
    Apply optional masks. By default, does nothing.

    - If emask is a float, keep E <= emask.
    - If emask == 'gap' and gapfile provided, keep E <= (1+gap_margin)*Egap(field*gap_scale_field).
    - If fzoom is a float > 0, keep |field| <= fzoom.
    """
    f = np.asarray(fields, float)
    E = np.asarray(energies, float)
    S = np.asarray(sigma, float)

    keep = np.ones_like(E, dtype=bool)

    # energy mask
    if emask is None:
        pass
    elif isinstance(emask, str) and emask.lower() == "gap":
        if gapfile is None or not os.path.exists(gapfile):
            raise FileNotFoundError("gap mask requested but --gapfile not found.")
        g_f, g_E = load_two_col_float(gapfile, colx=0, coly=1)
        Egap_of_field = interp1d(
            g_f * float(gap_scale_field),
            (1.0 + float(gap_margin)) * g_E,
            bounds_error=False, fill_value="extrapolate",
        )
        Eg = Egap_of_field(f)
        keep &= np.isfinite(Eg) & (E <= Eg)
    else:
        cutoff = float(emask)
        keep &= (E <= cutoff)

    # field zoom
    if fzoom is not None and float(fzoom) > 0.0:
        keep &= (np.abs(f) <= float(fzoom))

    f = f[keep]
    E = E[keep]
    S = S[keep]

    # diagnostics
    n_fields = np.unique(np.round(f, 12)).size
    n_energies = np.unique(np.round(E, 12)).size
    print(f"[mask] kept {S.size} points | unique fields: {n_fields} | unique energies: {n_energies}")
    if S.size > 0:
        print(f"[mask] F range: [{f.min():.6f}, {f.max():.6f}] eV/Ang | "
              f"E range: [{E.min():.4f}, {E.max():.4f}] eV")

    return f, E, S


# -----------------------------
# Grid + plot
# -----------------------------
def plot_heatmap(fields, energies, sigma,
                 nx=600, ny=600, method="linear",
                 cmap_name=DEFAULT_CMAP, out="colormap.png",
                 xlabel="Energy (eV)", ylabel="Field Intensity (m eV/Ang)",
                 overlay=None):
    """
    overlay: dict or None. If provided, expects keys:
        {'Egap': 1D array in eV, 'F': 1D array in eV/Ang,
         'color': str, 'lw': float, 'ls': str, 'alpha': float, 'label': str}
    Drawn as a dashed line on top of the heatmap (x=Egap, y=F*1000).
    """
    # 0) basic cleaning and de-duplication
    f = np.asarray(fields, float)
    E = np.asarray(energies, float)
    S = np.asarray(sigma, float)
    m = np.isfinite(f) & np.isfinite(E) & np.isfinite(S)
    f, E, S = f[m], E[m], S[m]
    if f.size == 0:
        raise RuntimeError("No points after cleaning (NaN/inf).")

    # merge duplicate (E, f) by averaging S
    pairs = np.round(np.column_stack([E, f]), 12)  # robust key
    acc_sum = defaultdict(float)
    acc_cnt = defaultdict(int)
    for (e, ff), s in zip(pairs, S):
        key = (e, ff)
        acc_sum[key] += float(s)
        acc_cnt[key] += 1
    uniq = np.array(list(acc_sum.keys()), float)
    Eu = uniq[:, 0]
    fu = uniq[:, 1]
    Su = np.array([acc_sum[k] / acc_cnt[k] for k in acc_sum.keys()], float)

    # 1) grid
    xi = np.linspace(Eu.min(), Eu.max(), int(nx))
    yi = np.linspace(fu.min(), fu.max(), int(ny))
    Xi, Yi = np.meshgrid(xi, yi)

    # 2) colormap
    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:
        print(f"Warning: cmap '{cmap_name}' not found. Falling back to '{DEFAULT_CMAP}'.")
        cmap = plt.get_cmap(DEFAULT_CMAP)

    # 3) interpolate with fallback
    Zi = None
    can_interp = (np.unique(Eu).size >= 2) and (np.unique(fu).size >= 2) and (Eu.size >= 3)
    if can_interp:
        try:
            Zi = griddata((Eu, fu), Su, (Xi, Yi), method=method)
        except Exception as e:
            print(f"[surface] griddata('{method}') failed: {e}")
            Zi = None
        if Zi is None or np.all(np.isnan(Zi)):
            print("[surface] falling back to method='nearest'")
            try:
                Zi = griddata((Eu, fu), Su, (Xi, Yi), method="nearest")
            except Exception as e:
                print(f"[surface] griddata('nearest') also failed: {e}")
                Zi = None

    # 4) plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.15)

    if Zi is not None and not np.all(np.isnan(Zi)):
        valid = Zi[~np.isnan(Zi)]
        vmax = float(np.nanmax(np.abs(valid)))
        if vmax == 0.0:
            vmax = 1.0  # avoid zero range
        levels = np.linspace(-vmax, vmax, 256)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        cf = ax.contourf(Xi, Yi * 1000.0, Zi, levels=levels, cmap=cmap, norm=norm, extend="both")
        cf.set_edgecolor("face")
        cbar = plt.colorbar(cf, ax=ax, pad=0.01)
        cbar.set_label(r"$\sigma^{(2)}_{xxx}$ (nm$\cdot\mu$A/V$^2$)", fontsize=16, rotation=270, labelpad=18)
        ax.contour(Xi, Yi * 1000.0, Zi, levels=np.linspace(-vmax, vmax, 12), colors="white", linewidths=0.6)
    else:
        # final fallback: scatter so you see data instead of a blank figure
        print("[surface] interpolation not possible; drawing scatter fallback.")
        sc = ax.scatter(Eu, fu * 1000.0, c=Su, s=8, cmap=cmap)
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(r"$\sigma^{(2)}_{xxx}$ (nm$\cdot\mu$A/V$^2$)", fontsize=16, rotation=270, labelpad=18)

    # 5) optional Egap overlay (no masking)
    if overlay is not None:
        Eg = np.asarray(overlay.get("Egap", []), float)
        Fg = np.asarray(overlay.get("F", []), float)
        if Eg.size and Fg.size and Eg.size == Fg.size:
            # sort by energy for a clean line
            idx = np.argsort(Fg)
            Eg = Eg[idx]
            Fg = Fg[idx]
            ax.plot(
                Eg, Fg * 1000.0,
                linestyle=overlay.get("ls", "-.-"),
                linewidth=float(overlay.get("lw", 1.6)),
                color=overlay.get("color", "magenta"),
                alpha=float(overlay.get("alpha", 0.9)),
                label=overlay.get("label", "Egap(field)")
            )
            # optional legend only if we actually draw the line
            # ax.legend(loc="best", frameon=False)

    # axes cosmetics
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min() * 1000.0, yi.max() * 1000.0)

    plt.tight_layout()
    plt.savefig(out, dpi=800)
    print(f"--> Saved figure as {out}")
    plt.show()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Energy vs Field heatmap of conductivity.")
    p.add_argument("--file", required=True, help="file name inside each field folder to read")
    p.add_argument("--ycol", type=int, default=1, help="column index for sigma (default 1 = xx)")
    # masks off by default:
    p.add_argument("--emask", default=None,
                   help="energy mask: float cutoff (e.g. 4.8) or 'gap' to use --gapfile; default: no mask")
    p.add_argument("--fzoom", type=float, default=None, help="limit |field| <= fzoom (eV/Ang); default: no zoom")
    p.add_argument("--gapfile", default=None, help="path to bandgap vs field table (2 columns: field, Egap)")
    p.add_argument("--gap-scale-field", type=float, default=1.0,
                   help="scale factor applied to field before looking up Egap(field) or drawing overlay")
    p.add_argument("--gap-margin", type=float, default=0.0,
                   help="relative margin on Egap for masking or overlay (Egap -> Egap*(1+margin))")
    # overlay controls
    p.add_argument("--overlay-gap", action="store_true", help="draw dashed Egap(field) reference line (no masking)")
    p.add_argument("--gap-color", default="magenta", help="overlay line color (default 'k')")
    p.add_argument("--gap-lw", type=float, default=1.6, help="overlay line width")
    p.add_argument("--gap-ls", default="--", help="overlay line style (default '--')")
    p.add_argument("--gap-alpha", type=float, default=0.9, help="overlay line alpha")
    # grid/plot
    p.add_argument("--nx", type=int, default=600, help="grid points along energy")
    p.add_argument("--ny", type=int, default=600, help="grid points along field")
    p.add_argument("--interp", choices=["linear", "nearest", "cubic"], default="linear", help="griddata method")
    p.add_argument("--cmap", default=DEFAULT_CMAP, help="matplotlib or colorcet colormap name")
    p.add_argument("--latex", action="store_true", help="enable LaTeX rendering")
    p.add_argument("--out", default="surface_sig-omega-field.png", help="output image filename (default auto)")
    return p.parse_args()


def main():
    args = parse_args()
    configure_matplotlib(latex=args.latex)

    fields, energies, sigma = collect_field_spectra(args.file, args.ycol)

    # Apply masks (all optional; defaults do nothing)
    fields_m, energies_m, sigma_m = apply_masks(
        fields, energies, sigma,
        emask=args.emask,
        fzoom=args.fzoom,
        gapfile=args.gapfile,
        gap_scale_field=args.gap_scale_field,
        gap_margin=args.gap_margin,
    )

    # Build optional overlay dict
    overlay = None
    if args.overlay_gap:
        if args.gapfile is None or not os.path.exists(args.gapfile):
            print("Warning: --overlay-gap requested but --gapfile not found; skipping overlay.")
        else:
            g_f, g_E = load_two_col_float(args.gapfile, colx=0, coly=1)
            # scale field and apply margin for the overlay only
            g_f = g_f * float(args.gap_scale_field)
            g_E = g_E * (1.0 + float(args.gap_margin))
            overlay = {
                "Egap": g_E, "F": g_f,
                "color": args.gap_color, "lw": args.gap_lw,
                "ls": args.gap_ls, "alpha": args.gap_alpha,
                "label": "Egap(field)"
            }

    # Auto output name if not provided
    out = args.out
    if out is None:
        tag = os.path.splitext(os.path.basename(args.file))[0]
        extra = []
        if args.emask is not None:
            if isinstance(args.emask, str) and args.emask.lower() == "gap":
                extra.append("gapmask")
            else:
                extra.append(f"E<={args.emask}")
        if args.fzoom is not None:
            extra.append(f"|F|<={args.fzoom}")
        if args.overlay_gap:
            extra.append("overlay_gap")
        suffix = "_" + "_".join(extra) if extra else ""
        out = f"colormap_{tag}{suffix}.png".replace(" ", "")


    # Plot
    plot_heatmap(
        fields_m, energies_m, sigma_m,
        nx=args.nx, ny=args.ny, method=args.interp,
        cmap_name=args.cmap, out=out,
        overlay=overlay
    )


if __name__ == "__main__":
    main()
