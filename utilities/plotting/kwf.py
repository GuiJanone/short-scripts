#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def configure_matplotlib(fontsize=15, latex=True):
    """Set global matplotlib rcParams."""
    if latex:
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


def read_kwf_file(path):
    """
    Read excitonic wavefunction file.
    Returns:
      kpoints (N x 3 array), states (list of length M arrays of length N)
    """
    ks = []
    states = []
    kpoints = []
    coefs = []

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            if line[0] == 'k':
                continue
            if line[0] == '#':
                states.append(np.array(coefs))
                ks.append(kpoints)
                kpoints = []
                coefs = []
                continue
            parts = line.split()
            # first three columns are k-point coordinates
            kp = [float(parts[i]) for i in range(3)]
            kpoints.append(kp)
            # last column is coefficient
            coefs.append(float(parts[-1]))
    # append last state if missing trailing '#'
    if coefs:
        states.append(np.array(coefs))
        ks.append(kpoints)

    if not ks:
        raise ValueError(f"No states found in file {path}")

    kpoints_arr = np.array(ks[0])
    return kpoints_arr, states


def plot_and_save_pdf(kpoints, states, n, xlim, ylim, output_pdf):
    """Plot first n states into a multi-page PDF."""
    with PdfPages(output_pdf) as pdf:
        for idx in range(n):
            fig, ax = plt.subplots(figsize=(6, 8))
            wf = states[idx]
            ax.tripcolor(
                kpoints[:, 0], kpoints[:, 1], wf,
                cmap='viridis'
            )
            ax.axis('square')
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
            ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
            ax.set_title(f'Exciton $\psi$ \#{idx+1}')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_and_save_png(kpoints, states, n, xlim, ylim, prefix):
    """Plot first n states into separate PNG files."""
    for idx in range(n):
        fig, ax = plt.subplots(figsize=(6, 8))
        wf = states[idx]
        ax.tripcolor(
            kpoints[:, 0], kpoints[:, 1], wf,
            cmap='viridis'
        )
        ax.axis('square')
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
        ax.set_title(f'Exciton $\psi$ \#{idx+1}')
        plt.tight_layout()
        fname = f"{prefix}_{idx+1}.png"
        plt.savefig(fname, dpi=600)
        plt.close(fig)


def print_usage():
    print("Usage: kwf_refactored.py <input_file> <n_states> [pdf|png]")
    print("  <input_file>  : file containing exciton wavefunctions")
    print("  <n_states>    : number of wavefunctions to plot")
    print("  pdf (default) : output as multi-page PDF")
    print("  png           : output as separate PNG files")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    infile = sys.argv[1]
    try:
        n = int(sys.argv[2])
    except ValueError:
        print("Error: <n_states> must be an integer.")
        print_usage()
        sys.exit(1)

    mode = 'pdf'
    if len(sys.argv) >= 4:
        mode_arg = sys.argv[3].lower()
        if mode_arg in ('pdf', 'png'):
            mode = mode_arg
        else:
            print(f"Unknown mode '{mode_arg}'")
            print_usage()
            sys.exit(1)

    configure_matplotlib()

    # Read data
    kpoints, states = read_kwf_file(infile)
    total = len(states)
    if n > total:
        print(f"Requested {n} states, but file contains only {total}.")
        sys.exit(1)

    # Determine plot limits
    xlim = np.abs(kpoints[:, 0]).max()
    ylim = np.abs(kpoints[:, 1]).max()

    base = os.path.splitext(os.path.basename(infile))[0]
    if mode == 'pdf':
        out_pdf = f"{base}_wfs.pdf"
        plot_and_save_pdf(kpoints, states, n, xlim, ylim, out_pdf)
    else:
        prefix = f"{base}_wf"
        plot_and_save_png(kpoints, states, n, xlim, ylim, prefix)


if __name__ == '__main__':
    main()
