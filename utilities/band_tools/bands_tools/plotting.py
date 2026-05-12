import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib(fontsize: int = 18) -> None:
    plt.rcParams.update({"text.usetex": True, "font.family": "sans_serif"})
    plt.rcParams["axes.labelsize"] = fontsize + 2
    plt.rcParams["axes.titlesize"] = fontsize + 2
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    plt.rcParams["font.size"] = fontsize


def plot_bands(blocks: list[np.ndarray], params: dict, red_bands: list[int] | None = None) -> None:
    configure_matplotlib()

    selected_red_bands = red_bands or []

    plt.figure(figsize=(8, 6))

    for band_index, block in enumerate(blocks):
        color = "red" if band_index in selected_red_bands else "black"
        k_points = block[:, 0]
        energies = block[:, 1]

        diffs = np.diff(k_points)
        split_indices = np.where(diffs < 0)[0]
        start_indices = np.concatenate([[0], split_indices + 1])
        end_indices = np.concatenate([split_indices + 1, [len(k_points)]])

        for start, end in zip(start_indices, end_indices):
            plt.plot(
                k_points[start:end],
                energies[start:end] - params["Efermi"],
                color=color,
                linewidth=1.5,
            )

    plt.xlabel("k-space")
    plt.ylabel("Energy (eV)")

    plt.xlim(params["xlim"])
    plt.ylim(params["ylim"])

    plt.xticks(params["xticks"]["locations"], params["xticks"]["labels"])

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    outfile = params.get("output_file", "bands.png")
    plt.savefig(outfile, dpi=600)
    print(f"Saved figure as {outfile}")
