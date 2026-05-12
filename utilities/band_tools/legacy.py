#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import json

def configure_matplotlib(fontsize=18):
    plt.rcParams.update({"text.usetex": True, "font.family": "sans_serif"})
    plt.rcParams["axes.labelsize"] = fontsize+2
    plt.rcParams["axes.titlesize"] = fontsize+2
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    plt.rcParams["font.size"] = fontsize

def load_band_blocks(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]

    blocks = []
    current = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(np.loadtxt(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(np.loadtxt(current))

    if not blocks:
        raise ValueError(f"bands_file '{filename}' contains no data blocks.")

    for i, block in enumerate(blocks, start=1):
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(
                f"Each block in '{filename}' must have exactly two columns (k and energy). "
                f"Block {i} has shape {block.shape}."
            )

    return blocks


def parse_red_bands(raw_red_bands, n_bands):
    if raw_red_bands is None:
        return []
    if isinstance(raw_red_bands, (int, np.integer)):
        raw_red_bands = [raw_red_bands]
    if not isinstance(raw_red_bands, (list, tuple, np.ndarray)):
        raise ValueError("red_bands must be a list of band indices")

    indices = []
    for value in raw_red_bands:
        if isinstance(value, str):
            if not value.isdigit():
                raise ValueError(f"red_bands values must be integer strings or integers, got {value!r}")
            value = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"red_bands values must be integer-like, got {value}")
            value = int(value)
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        else:
            raise ValueError(f"red_bands values must be integers, got {type(value).__name__}")
        indices.append(value)

    if not indices:
        return []

    zero_based = all(0 <= i < n_bands for i in indices)
    one_based = all(1 <= i <= n_bands for i in indices)

    if zero_based and not one_based:
        return indices
    if one_based and not zero_based:
        return [i - 1 for i in indices]
    if zero_based and one_based:
        return indices

    raise ValueError(
        f"red_bands values must be within 0..{n_bands - 1} or 1..{n_bands} consistently. Got {indices}"
    )


def plot_bands(params):
    configure_matplotlib()
    
    blocks = load_band_blocks(params['bands_file'])
    n_bands = len(blocks)
    red_bands = parse_red_bands(params.get('red_bands', []), n_bands)


    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot each block as a separate band
    for band_index, block in enumerate(blocks):
        color = 'red' if band_index in red_bands else 'black'
        k_points = block[:, 0]
        energies = block[:, 1]

        diffs = np.diff(k_points)
        split_indices = np.where(diffs < 0)[0]
        start_indices = np.concatenate([[0], split_indices + 1])
        end_indices = np.concatenate([split_indices + 1, [len(k_points)]])

        for start, end in zip(start_indices, end_indices):
            plt.plot(k_points[start:end], energies[start:end] - params['Efermi'],
                     color=color, linewidth=1.5)

    # Add labels
    plt.xlabel("k-space")
    plt.ylabel("Energy (eV)")

    # Set limits
    plt.xlim(params['xlim'])
    plt.ylim(params['ylim'])

    # Set xticks
    plt.xticks(params['xticks']['locations'], params['xticks']['labels'])

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save figure
    plt.tight_layout()
    outfile = params.get('output_file', 'bands.png')
    plt.savefig(outfile, dpi=600)
    print(f"Saved figure as {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot electronic band structure from JSON config")
    parser.add_argument("--json", help="JSON config file")
    parser.add_argument("--generate-example", action="store_true", help="Generate an example JSON file")
    args = parser.parse_args()

    if args.generate_example:
        example = {
            "bands_file": "bands.dat",
            "Efermi": 0.0,
            "xlim": [0, 10],
            "ylim": [-5, 5],
            "xticks": {
                "locations": [0, 5, 10],
                "labels": ["$\\Gamma$", "X", "M"]
            },
            "red_bands": [9, 10]
        }
        with open('example_bands.json', 'w') as f:
            json.dump(example, f, indent=4)
        print("Generated example_bands.json")
    else:
        if not args.json:
            parser.error("--json is required unless --generate-example is used")
        with open(args.json) as f:
            params = json.load(f)
        plot_bands(params)
