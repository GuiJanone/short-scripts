#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def plot_bands(filename, Efermi, emin=None, emax=None):
    # Load data
    k_points, energies = np.loadtxt(filename, unpack=True)

    # Find indices where k decreases, indicating a new segment
    diffs = np.diff(k_points)
    split_indices = np.where(diffs < 0)[0]

    # Determine start and end indices for each segment
    start_indices = np.concatenate([[0], split_indices + 1])
    end_indices = np.concatenate([split_indices + 1, [len(k_points)]])

    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot each segment separately
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        plt.plot(k_points[start:end], energies[start:end] - Efermi,
                 color='black', linewidth=1.5)

    # Add labels
    plt.xlabel("k-space", fontsize=12)
    plt.ylabel("Energy (eV)", fontsize=12)
    # plt.title("Electronic Band Structure", fontsize=14)

    plt.xlim(0, k_points[end-1])

    # Apply energy limits if provided
    if emin is not None or emax is not None:
        plt.ylim(
            emin if emin is not None else plt.ylim()[0],
            emax if emax is not None else plt.ylim()[1]
        )

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save figure
    plt.tight_layout()
    plt.savefig("bands.png", dpi=600)
    print("Saved figure as bands.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot electronic band structure")
    parser.add_argument("filename", help="input file with k-points and energies")
    parser.add_argument("fermi", type=float, help="Fermi energy (eV)")
    parser.add_argument("--emin", type=float, default=None, help="minimum energy for y-axis (eV)")
    parser.add_argument("--emax", type=float, default=None, help="maximum energy for y-axis (eV)")
    args = parser.parse_args()

    plot_bands(args.filename, args.fermi, emin=args.emin, emax=args.emax)
