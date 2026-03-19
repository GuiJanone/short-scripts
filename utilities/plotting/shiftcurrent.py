#!/usr/bin/env python3
"""
Generic shift-current plotter with optional energy masking.

Schemas:
1) Regular filenames (no '_z' before extension):
   Columns: E(omega), xxx, xyy, yyy, yxx

2) Filenames ending in '_z' before extension (e.g. 'foo_z.dat'):
   Columns: E(omega), xxz, yyz, zzz, zxx, zyy

JSON Configuration:
The script reads from a JSON file with the following structure:
{
  "file": "path/to/data.dat",  (or "files": ["fileA.dat", "fileB.dat"] for multiple files)
  "components": ["xxx", "yyy"],  (optional: explicit component selection)
  "labels_file": "path/to/labels.txt",  (optional: custom labels or component selection)
  "sum_files": true,  (optional: sum corresponding columns from multiple files, default: false)
  "x_offset1": 0.5,  (optional: x-axis offset for first file in eV)
  "x_offset2": 1.0,  (optional: x-axis offset for second file in eV)
  "emin": 1.2,  (optional: minimum energy in eV)
  "emax": 4.8,  (optional: maximum energy in eV)
  "output": "shift_current.png",  (optional: output filename, default: shift_current.png)
  "title": "My Plot"  (optional: plot title)
}

To generate an example JSON file, run:
  python shiftcurrent.py --generate-example output_config.json

Examples:
  python shiftcurrent.py config.json
  python shiftcurrent.py --generate-example config.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Schemas
# --------------------------
REGULAR_KEYS = ["xxx", "xyy", "yyy", "yxx"]
Z_KEYS       = ["xxz", "yyz", "zzz", "zxx", "zyy"]

def generate_example_json(output_path):
    """Generate an example JSON configuration file with all possible options."""
    example_config = {
        "file": "data.dat",
        "components": ["xxx", "yyy"],
        "labels_file": None,
        "emin": 1.2,
        "emax": 4.8,
        "output": "shift_current.png",
        "title": "Shift Current"
    }
    example_config_multi = {
        "files": ["fileA.dat", "fileB.dat"],
        "sum_files": True,
        "x_offset1": 0.5,
        "x_offset2": 1.0,
        "components": ["xxx", "yyy"],
        "labels_file": None,
        "emin": 1.2,
        "emax": 4.8,
        "output": "shift_current_summed.png",
        "title": "Summed Shift Current"
    }
    with open(output_path, "w") as f:
        json.dump(example_config, f, indent=2)
    print(f"Generated example JSON configuration: {output_path}")
    print("Note: To use multiple files with summing, use 'files' (list) instead of 'file' (string) and set 'sum_files': true")
    print("Note: To apply x-axis offsets, use 'x_offset1', 'x_offset2', etc. in eV units")

def load_config_json(json_path):
    """Load configuration from JSON file."""
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error: could not read JSON config '{json_path}': {e}")
        sys.exit(1)
    
    if "file" not in config and "files" not in config:
        print("Error: JSON config must have either 'file' or 'files' field.")
        sys.exit(1)
    
    return config

def configure_matplotlib(fontsize=15):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    plt.rcParams["font.size"] = fontsize

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

def sum_file_columns(data_list):
    """
    Sum corresponding columns from multiple data arrays.
    All files must have the same shape. Energy column (col 0) is taken from first file.
    """
    if not data_list:
        return None
    if len(data_list) == 1:
        return data_list[0]
    
    # Check all files have same shape
    shape0 = data_list[0].shape
    for i, data in enumerate(data_list[1:], 1):
        if data.shape != shape0:
            print(f"Error: all input files must have the same dimensions.")
            print(f"File 0 shape: {shape0}, File {i} shape: {data.shape}")
            sys.exit(1)
    
    # Sum all columns except energy (col 0 stays from first file)
    result = data_list[0].copy()
    for data in data_list[1:]:
        result[:, 1:] += data[:, 1:]
    
    return result

def apply_x_offset(data, offset):
    """
    Apply an offset to the x-axis (energy/omega column, column 0).
    Returns a copy of the data with the offset applied.
    """
    if offset is None or offset == 0:
        return data
    result = data.copy()
    result[:, 0] += float(offset)
    return result

# --------------------------
# Plot
# --------------------------
def plot_components(energy, series, out="shift_current.png", title=None, xlim=None, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, y in series:
        ax.plot(energy, y, label=label, alpha=0.9, lw=2.0)
    ax.hlines(0.0, energy[0], energy[-1], colors='k', linestyles='dashed', alpha=0.5)
    ax.set_xlabel(r"Photon Energy $\omega$ (eV)", fontsize=14)
    ax.set_ylabel(r"$\sigma^{2}$ (nm $\cdot \mu$A / V$^{2}$)", fontsize=14)
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
    if show:
        plt.show()

# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser(description="Generic shift-current plotter.")
    p.add_argument("config", nargs='?', default=None, help="JSON configuration file")
    p.add_argument("--generate-example", default=None,
                   help="generate an example JSON config file and exit")
    args = p.parse_args()

    # Handle example generation
    if args.generate_example:
        generate_example_json(args.generate_example)
        sys.exit(0)

    # Config is required if not generating example
    if args.config is None:
        print("Error: config file is required (or use --generate-example)")
        p.print_help()
        sys.exit(1)

    # Load JSON configuration
    config = load_config_json(args.config)
    
    # Handle file(s) - support both single "file" and multiple "files"
    if "files" in config:
        file_list = config["files"]
        if not isinstance(file_list, list):
            print("Error: 'files' field must be a list of file paths.")
            sys.exit(1)
        data_list = [load_data(f) for f in file_list]
        
        # Apply x-offsets if provided (x_offset1, x_offset2, etc.)
        for i, file_data in enumerate(data_list):
            offset_key = f"x_offset{i+1}"
            if offset_key in config:
                offset = config[offset_key]
                data_list[i] = apply_x_offset(file_data, offset)
                print(f"Applied x-offset {offset} to file {i+1}: {file_list[i]}")
        
        sum_files = config.get("sum_files", False)
        if sum_files:
            data = sum_file_columns(data_list)
            schema_file = file_list[0]  # use first file for schema detection
            data_file_display = f"{len(file_list)} files (summed)"
        else:
            data = data_list[0]
            schema_file = file_list[0]
            data_file_display = file_list[0]
    else:
        schema_file = config["file"]
        data_file_display = schema_file
        data = load_data(schema_file)
        
        # Apply x-offset if provided for single file (x_offset1)
        if "x_offset1" in config:
            offset = config["x_offset1"]
            data = apply_x_offset(data, offset)
            print(f"Applied x-offset {offset} to file: {schema_file}")
    
    components = config.get("components", None)
    labels_file = config.get("labels_file", None)
    emin = config.get("emin", None)
    emax = config.get("emax", None)
    output_file = config.get("output", "shift_current.png")
    title_arg = config.get("title", None)
    show_plot = config.get("show", False)

    energy = data[:, 0]
    keys = default_keys_for(schema_file)

    # how many component columns exist in the file
    n_comp_cols = min(len(keys), data.shape[1] - 1)
    keys = keys[:n_comp_cols]

    # energy mask (apply before slicing columns so series align)
    mask = np.ones_like(energy, dtype=bool)
    if emin is not None:
        mask &= (energy >= float(emin))
    if emax is not None:
        mask &= (energy <= float(emax))
    if not np.any(mask):
        print("Error: energy mask removed all points. Loosen emin/emax.")
        sys.exit(1)
    energy_m = energy[mask]
    data_m = data[mask, :]

    # selection via components (priority)
    selection = None
    if components:
        selection = [s.strip() for s in components if isinstance(components, list)]
        bad = [k for k in selection if k not in keys]
        if bad:
            print(f"Error: components not recognized for this file schema: {bad}")
            print(f"Allowed: {keys}")
            sys.exit(1)

    # labels file behavior
    custom_labels = None
    if labels_file:
        lines = read_label_lines(labels_file)
        if lines:
            sel_from_labels, custom_labels = interpret_labels_or_selection(lines, keys)
            if sel_from_labels is not None:
                selection = sel_from_labels
        else:
            print(f"Warning: labels file '{labels_file}' is empty. Ignoring.")

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
    title = title_arg if title_arg is not None else None

    # xlim matches masked energy span; ylim stays automatic
    xlim = (float(energy_m.min()), float(energy_m.max()))
    plot_components(energy_m, series, out=output_file, title=title, xlim=xlim, show=show_plot)

if __name__ == "__main__":
    main()
