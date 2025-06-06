#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_bands(filename, Efermi):
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
        label = "Electronic Bands" if i == 0 else None
        plt.plot(k_points[start:end], energies[start:end]-Efermi, 
                 color='black', linewidth=1.5, label=label)
    
    # Add labels and title
    plt.xlabel("k-space", fontsize=12)
    plt.ylabel("Energy (eV)", fontsize=12)
    plt.title("Electronic Band Structure", fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("bands.png", dpi=400)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <filename> <fermi energy>")
        sys.exit(1)
    
    filename = sys.argv[1]
    Efermi = float(sys.argv[2])
    print(Efermi)
    plot_bands(filename, Efermi)