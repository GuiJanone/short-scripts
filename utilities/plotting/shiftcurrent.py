#!/usr/bin/env python3
"""
Generic shift-current plotter with optional energy masking.

Schemas:
1) Regular filenames (no '_z' before extension):
   Columns: E(omega), xxx, xyy, yyy, yxx

2) Filenames ending in '_z' before extension (e.g. 'foo_z.dat'):
   Columns: E(omega), xxz, yyz, zzz, zxx, zyy

Selection:
- --components xxz,zzz     choose components explicitly
- --labels labels.txt      either list known keys to select order, or provide custom labels

Energy mask (optional):
- --emax 4.8               keep E <= 4.8 eV
- --emin 1.2               keep E >= 1.2 eV
(omit both for full range)

Examples:
  python shiftcurrent.py data.dat
  python shiftcurrent.py data_z.dat --components xxz,zzz --emax 3.5
  python shiftcurrent.py data.dat --labels labels.txt --emin 1.5 --emax 4.8
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Schemas
# --------------------------
REGULAR_KEYS = ["xxx", "xyy", "yyy", "yxx"]
Z_KEYS       = ["xxz", "yyz", "zzz", "zxx", "zyy"]

def is_z_schema(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.endswith("_z")

def default_keys_for(path):
    return Z_KEYS if is_z_schema(path) else REGULAR_KEYS

# --------------------------
# Labels / selection helpers
# --------------------------
def read_label_lines(path):
    lines = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines

def interpret_labels_or_selection(lines, known_keys):
    """
    If all lines are known component names -> treat as selection order, labels=those names.
    Else -> treat the lines as custom labels for the default order (no selection).
    Returns (selection_keys_or_None, custom_labels_or_None).
    """
    kset = set(known_keys)
    all_known = all((ln in kset) for ln in lines)
    if all_known:
        return list(lines), None
    else:
        return None, list(lines)

# --------------------------
# Data loader
# --------------------------
def load_data(path):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"Error: could not read '{path}': {e}")
        sys.exit(1)
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"Error: '{path}' does not look like a 2D table with at least 2 columns.")
        sys.exit(1)
    return data

# --------------------------
# Plot
# --------------------------
def plot_components(energy, series, out="shift_current.png", title=None, xlim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, y in series:
        ax.plot(energy, y, label=label, alpha=0.85, lw=2.0)
    ax.set_xlabel("E (eV)", fontsize=14)
    ax.set_ylabel("sigma^(2) (uA/V^2 * nm)", fontsize=14)
    if xlim is None:
        ax.set_xlim(float(energy[0]), float(energy[-1]))
    else:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    print(f"Saved figure: {out}")
    plt.show()

# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser(description="Generic shift-current plotter.")
    p.add_argument("file", help="data file to plot")
    p.add_argument("--components", default=None,
                   help="comma-separated list of components to plot (e.g. 'xxx,yyy' or 'xxz,zzz')")
    p.add_argument("--labels", default=None,
                   help="path to a labels file. If lines match known keys, they select order; otherwise used as custom labels.")
    # energy mask
    p.add_argument("--emin", type=float, default=None, help="min energy in eV (keep E >= emin)")
    p.add_argument("--emax", type=float, default=None, help="max energy in eV (keep E <= emax)")
    p.add_argument("--out", default="shift_current.png", help="output image filename")
    p.add_argument("--title", default=None, help="optional plot title")
    args = p.parse_args()

    data = load_data(args.file)
    energy = data[:, 0]
    keys = default_keys_for(args.file)

    # how many component columns exist in the file
    n_comp_cols = min(len(keys), data.shape[1] - 1)
    keys = keys[:n_comp_cols]

    # energy mask (apply before slicing columns so series align)
    mask = np.ones_like(energy, dtype=bool)
    if args.emin is not None:
        mask &= (energy >= float(args.emin))
    if args.emax is not None:
        mask &= (energy <= float(args.emax))
    if not np.any(mask):
        print("Error: energy mask removed all points. Loosen --emin/--emax.")
        sys.exit(1)
    energy_m = energy[mask]
    data_m = data[mask, :]

    # selection via --components (priority)
    selection = None
    if args.components:
        selection = [s.strip() for s in args.components.split(",") if s.strip()]
        bad = [k for k in selection if k not in keys]
        if bad:
            print(f"Error: components not recognized for this file schema: {bad}")
            print(f"Allowed: {keys}")
            sys.exit(1)

    # labels file behavior
    custom_labels = None
    if args.labels:
        lines = read_label_lines(args.labels)
        if lines:
            sel_from_labels, custom_labels = interpret_labels_or_selection(lines, keys)
            if sel_from_labels is not None:
                selection = sel_from_labels
        else:
            print(f"Warning: labels file '{args.labels}' is empty. Ignoring.")

    # fallback selection = all available in default order
    if selection is None:
        selection = list(keys)

    # build series (label, y-array)
    name_to_col = {k: i+1 for i, k in enumerate(keys)}  # +1 to skip energy column
    series = []
    for i, comp in enumerate(selection):
        col_idx = name_to_col[comp]
        if col_idx >= data_m.shape[1]:
            print(f"Warning: column for '{comp}' not present in file. Skipping.")
            continue
        label = custom_labels[i] if (custom_labels is not None and i < len(custom_labels)) else comp
        series.append((label, data_m[:, col_idx]))

    if not series:
        print("Nothing to plot (no valid components).")
        sys.exit(0)

    # auto title if not provided
    title = args.title if args.title is not None else os.path.basename(args.file)

    # xlim matches masked energy span; ylim stays automatic
    xlim = (float(energy_m.min()), float(energy_m.max()))
    plot_components(energy_m, series, out=args.out, title=title, xlim=xlim)

if __name__ == "__main__":
    main()
