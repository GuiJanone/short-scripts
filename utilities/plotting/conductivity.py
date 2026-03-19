#!/usr/bin/env python3
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def configure_matplotlib(fontsize=15, markersize=110, use_latex=True):
    """Set global matplotlib rcParams."""
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
    plt.rcParams["axes.labelsize"]   = fontsize
    plt.rcParams["axes.titlesize"]   = fontsize
    plt.rcParams["xtick.labelsize"]  = fontsize
    plt.rcParams["ytick.labelsize"]  = fontsize
    plt.rcParams["legend.fontsize"]  = fontsize
    plt.rcParams["font.size"]        = fontsize


def read_tensor_file(path):
    """
    Read a conductivity/tensor file where each row contains
    omega xx xy xz yx yy yz zx zy zz
    Returns a tuple ``(energy, data_dict)`` where ``data_dict``
    maps component names (e.g. 'xx','xy',...) to numpy arrays.
    """
    energy = []
    comps = {k: [] for k in ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]}

    with open(path, 'r') as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            vals = [float(v) for v in parts]
            energy.append(vals[0])
            comps["xx"].append(vals[1])
            comps["xy"].append(vals[2])
            comps["xz"].append(vals[3])
            comps["yx"].append(vals[4])
            comps["yy"].append(vals[5])
            comps["yz"].append(vals[6])
            comps["zx"].append(vals[7])
            comps["zy"].append(vals[8])
            comps["zz"].append(vals[9])

    return np.array(energy), {k: np.array(v) for k, v in comps.items()}


# keep the old helper names for backward compatibility,
# they simply pull out the xx/yy/zz entries from the full tensor.

def read_exciton_file(path):
    """
    Read exciton file: energy and BSE conductivities only.
    Returns:
      energy, sigma_xx, sigma_yy, sigma_zz
    """
    energy, data = read_tensor_file(path)
    return energy, data["xx"], data["yy"], data["zz"]


def read_sp_file(path):
    """
    Read single-particle conductivity file.
    Returns:
      sigma_sp_xx, sigma_sp_yy, sigma_sp_zz
    """
    _, data = read_tensor_file(path)
    return data["xx"], data["yy"], data["zz"]


def read_oscillator_file(path):
    """
    Read oscillator strengths from third file.
    Returns:
      E_exc, Vx, Vy, Vz
    """
    E_exc = []
    Vx = []
    Vy = []
    Vz = []
    with open(path, 'r') as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            vals = [float(v) for v in parts]
            E_exc.append(vals[0])
            Vx.append(abs(vals[1] + 1j*vals[2])**2)
            Vy.append(abs(vals[3] + 1j*vals[4])**2)
            Vz.append(abs(vals[5] + 1j*vals[6])**2)
    return np.array(E_exc), np.array(Vx), np.array(Vy), np.array(Vz)


def print_header(plot_type, output_file):
    """Print a standardized header with current plot info.

    ``plot_type`` is an arbitrary string describing the mode ("json",
    "1", "2", etc.).
    """
    print("===================================")
    print("       PLOTTING CONDUCTIVITY       ")
    print("===================================")
    print(f"Mode: {plot_type}")
    print(f"Output file: {output_file}")
    print()


def print_usage():
    print("Usage:")
    print("  script.py <exciton_file> <sp_file> [oscillator_file]")
    print("  script.py <config.json>               # use JSON configuration")
    print("  script.py --example                   # write example.json and exit")
    print()
    print("Running with two files does plot type 1.")
    print("Add a third file to get plot type 2.")
    print("JSON input allows arbitrary datasets, tensor components, axis")
    print("limits and offsets. See generated example.json for format.")


def plot_type1(energy, sigma_xx, sigma_yy, sigma_zz,
               sigma_sp_xx, sigma_sp_yy, sigma_sp_zz,
               output="conductivity.png"):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(energy, sigma_xx,    "r-",  label="BSE_xx", alpha=0.6)
    ax.plot(energy, sigma_yy,    "orange", label="BSE_yy", alpha=0.6)
    # ax.plot(energy, sigma_zz,    "g-",  label="BSE_zz", alpha=0.6)
    ax.plot(energy, sigma_sp_xx, "b--", label="IPA_xx")
    ax.plot(energy, sigma_sp_yy, "c--", label="IPA_yy")
    # ax.plot(energy, sigma_sp_zz, ls="--", c="navy", label="IPA_zz")

    ax.set_xlabel("E (eV)", fontsize=18)
    ax.set_ylabel(r"$\sigma$ ($e^2/\hbar$)", fontsize=18)
    ax.legend()
    ax.set_xlim(energy[0], energy[-1])
    plt.tight_layout()
    plt.savefig(output, dpi=600)


def plot_type2(energy, sigma_xx, sigma_yy, sigma_zz,
               sigma_sp_xx, sigma_sp_yy, sigma_sp_zz,
               E_exc, Vx, Vy, Vz,
               output="conductivity_oscillators.png"):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8))
    ax1 = axes[0]
    ax1.plot(energy, sigma_xx,    "r-",     label="BSE_xx")
    ax1.plot(energy, sigma_yy,    "orange", label="BSE_yy")
    # ax1.plot(energy, sigma_zz,    "g-",     label="BSE_zz")
    ax1.plot(energy, sigma_sp_xx, "b--",    label="IPA_xx")
    ax1.plot(energy, sigma_sp_yy, "c--",    label="IPA_yy")
    # ax1.plot(energy, sigma_sp_zz, ls="--",  c="navy", label="IPA_zz")
    ax1.set_ylabel(r"$\sigma$ ($e^2/\hbar$)", fontsize=18)

    ax2 = axes[1]

    # mask weak oscilaltors (plotting everything is costly and messy)
    mask_vx = Vx > 0.1
    mask_vy = Vy > 0.1
    mask_vz = Vz > 0.5

    if mask_vx.any():
        ax2.bar(E_exc[mask_vx], E_exc[mask_vx]*Vx[mask_vx], width=0.03, alpha=0.3, label="xx")
    if mask_vy.any():
        ax2.bar(E_exc[mask_vy], E_exc[mask_vy]*Vy[mask_vy], width=0.03, alpha=0.3, label="yy")
    # if mask_vz.any():
    #     ax2.bar(E_exc[mask_vz], E_exc[mask_vz]*Vz[mask_vz], width=0.03, alpha=0.3, label="zz")

    ax2.set_ylabel(r"$|V^\alpha|^2$", fontsize=18)
    ax2.set_xlabel("E (eV)", fontsize=18)

    for ax in axes:
        ax.set_xlim(energy[0], energy[-1])
        ax.tick_params(labelsize=12)

    ax1.legend()
    ax2.legend()
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.savefig(output, dpi=600)


def plot_multi(energies, datas, labels, components,
               output="conductivity.png", xlim=None, ylim=None):
    """Plot a collection of datasets and return the figure.

    *energies* is a list of 1‑D numpy arrays (one per dataset).
    *datas* is a list of dictionaries mapping tensor components to arrays.
    *labels* provides a human‑readable name for each dataset.
    *components* is the list of tensor components to draw (e.g. ['xx','xy']).
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for energy, data, lab in zip(energies, datas, labels):
        for comp in components:
            arr = data.get(comp)
            if arr is None:
                continue
            ax.plot(energy, arr, label=f"{lab}_{comp}")
    ax.set_xlabel("E (eV)", fontsize=18)
    ax.set_ylabel(r"$\sigma$ ($e^2/\hbar$)", fontsize=18)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    return fig


def generate_example_json(path="example.json"):
    """Write a skeleton configuration file to *path*."""
    example = {
        "datasets": [
            {"path": "exciton.dat", "label": "BSE", "offset": 0.0},
            {"path": "sp.dat", "label": "IPA", "offset": 0.0}
        ],
        "components": ["xx", "yy"],
        "xlim": [0, 10],
        "ylim": [0, 5],
        "output": "conductivity.png"
    }
    with open(path, 'w') as f:
        json.dump(example, f, indent=2)
    print(f"Example JSON written to {path}")


def run_from_json(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    datasets = cfg.get('datasets', [])
    if not datasets:
        raise ValueError('configuration must contain a non‑empty "datasets" list')
    components = cfg.get('components', [])
    if not components:
        raise ValueError('must specify at least one component in "components"')
    xlim = cfg.get('xlim', None)
    ylim = cfg.get('ylim', None)
    output = cfg.get('output', 'conductivity.png')

    energies = []
    datas = []
    labels = []
    for ds in datasets:
        path = ds['path']
        lab = ds.get('label', path)
        offset = ds.get('offset', 0.0)
        energy, data = read_tensor_file(path)
        energies.append(energy + offset)
        datas.append(data)
        labels.append(lab)

    print_header('json', output)
    configure_matplotlib()
    plot_multi(energies, datas, labels, components,
               output=output, xlim=xlim, ylim=ylim)
    plt.show()


def main():
    # handle simple command-line flags first
    if len(sys.argv) == 2 and sys.argv[1] in ('--example', '--generate-json'):
        generate_example_json('example.json')
        sys.exit(0)

    # JSON configuration mode
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith('.json'):
        run_from_json(sys.argv[1])
        sys.exit(0)

    # legacy behaviour for two or three files
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    exciton_path = sys.argv[1]
    sp_path      = sys.argv[2]
    has_osc      = len(sys.argv) == 4
    plot_type    = 2 if has_osc else 1
    output_file  = "conductivity_oscillators.png" if has_osc else "conductivity.png"

    print_header(plot_type, output_file)
    configure_matplotlib()

    energy, sigma_xx, sigma_yy, sigma_zz = read_exciton_file(exciton_path)
    sigma_sp_xx, sigma_sp_yy, sigma_sp_zz = read_sp_file(sp_path)

    if plot_type == 1:
        plot_type1(energy, sigma_xx, sigma_yy, sigma_zz,
                   sigma_sp_xx, sigma_sp_yy, sigma_sp_zz,
                   output_file)
    else:
        E_exc, Vx, Vy, Vz = read_oscillator_file(sys.argv[3])
        plot_type2(energy, sigma_xx, sigma_yy, sigma_zz,
                   sigma_sp_xx, sigma_sp_yy, sigma_sp_zz,
                   E_exc, Vx, Vy, Vz,
                   output_file)

    plt.show()

if __name__ == "__main__":
    main()
