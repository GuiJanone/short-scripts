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


def read_eigenvalues(path, n):
    """
    Read eigenvalues file: skip first three lines, then read n values.
    """
    vals = []
    with open(path, 'r') as f:
        for _ in range(3):
            next(f, None)
        for i in range(n):
            line = next(f, None)
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    if len(vals) < n:
        raise ValueError(f"Expected {n} eigenvalues but got {len(vals)} in {path}")
    return np.array(vals)


def combine_degenerate_states(eigvals, states, threshold=1e-3):
    """Combine at most pairs of wavefunctions whose eigenvalues are degenerate within threshold and include eigenvalues in titles."""
    combined_states = []
    titles = []
    i = 0
    while i < len(eigvals):
        if i + 1 < len(eigvals) and abs(eigvals[i] - eigvals[i+1]) <= threshold:
            # combine pair
            new_wf = states[i] + states[i+1]
            combined_states.append(new_wf)
            titles.append(f"Exciton $\psi$ \#{i+1}+{i+2} E={eigvals[i]:.5f}, {eigvals[i+1]:.5f}")
            i += 2
        else:
            combined_states.append(states[i])
            titles.append(f"Exciton $\psi$ \#{i+1} E={eigvals[i]:.5f}")
            i += 1
    return combined_states, titles


def plot_and_save_pdf(kpoints, states, titles, xlim, ylim, output_pdf):
    """Plot given states with titles into a multi-page PDF."""
    with PdfPages(output_pdf) as pdf:
        for idx, wf in enumerate(states):
            fig, ax = plt.subplots(figsize=(6, 8))
            cf = ax.tripcolor(
                    kpoints[:, 0], kpoints[:, 1], wf,
                    cmap='viridis'
                )
            cf.set_edgecolor("face") 
            ax.axis('square')
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
            ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
            ax.set_title(titles[idx])
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_and_save_png(kpoints, states, titles, xlim, ylim, prefix):
    """Plot given states with titles into separate PNG files."""
    for idx, wf in enumerate(states):
        fig, ax = plt.subplots(figsize=(6, 8))
        cf = ax.tripcolor(
                kpoints[:, 0], kpoints[:, 1], wf,
                cmap='viridis'
            )
        cf.set_edgecolor("face") 
        ax.axis('square')
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
        ax.set_title(titles[idx])
        plt.tight_layout()
        fname = f"{prefix}_{idx+1}.png"
        plt.savefig(fname, dpi=600)
        plt.close(fig)


def print_usage():
    print("Usage: kwf_refactored.py <kwf_file> <n_states> [eigval_file] [pdf|png]")
    print("  <kwf_file>     : file containing exciton wavefunctions")
    print("  <n_states>    : number of wavefunctions to consider from both files")
    print("  [eigval_file] : optional file with eigenvalues (first three lines skipped)")
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

    eigfile = None
    mode = 'pdf'
    remaining = sys.argv[3:]
    if len(remaining) == 1:
        if remaining[0].lower() in ('pdf', 'png'):
            mode = remaining[0].lower()
        else:
            eigfile = remaining[0]
    elif len(remaining) >= 2:
        # first non-mode argument is eigenvalue file; last if mode
        if remaining[-1].lower() in ('pdf', 'png'):
            mode = remaining[-1].lower()
            eigfile = remaining[0]
        else:
            eigfile = remaining[0]

    configure_matplotlib()

    # Read data
    kpoints, states = read_kwf_file(infile)
    total = len(states)
    if n > total:
        print(f"Requested {n} states, but file contains only {total}.")
        sys.exit(1)
    states = states[:n]

    titles = [f"Exciton $\psi$ \#{i+1}" for i in range(n)]
    if eigfile:
        eigvals = read_eigenvalues(eigfile, n)
        eigvals = eigvals[:n]
        combined_states, titles = combine_degenerate_states(eigvals, states)
    else:
        combined_states = states

    # Determine plot limits from kpoints of first block
    xlim = np.abs(kpoints[:, 0]).max()
    ylim = np.abs(kpoints[:, 1]).max()

    base = os.path.splitext(os.path.basename(infile))[0]
    if mode == 'pdf':
        out_pdf = f"{base}_wfs.pdf"
        plot_and_save_pdf(kpoints, combined_states, titles, xlim, ylim, out_pdf)
    else:
        prefix = f"{base}_wf"
        plot_and_save_png(kpoints, combined_states, titles, xlim, ylim, prefix)


if __name__ == '__main__':
    main()
