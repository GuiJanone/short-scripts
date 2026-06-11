#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt


DIR_PATTERN = re.compile(
    r"Output_screening_qx_([-+0-9.eE]+)_qy_([-+0-9.eE]+)"
)


def read_eps_inv_00(filepath):
    """
    Read the G0G0 element from a Xatu inverse dielectric matrix file.

    Expected first numeric row:
        Re(G0G0) Im(G0G0) Re(G0G1) Im(G0G1) ...

    Returns:
        complex epsilon^{-1}_{00}
    """
    with open(filepath, "r", encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                real_part = float(parts[0])
                imag_part = float(parts[1])
            except ValueError:
                continue

            return real_part + 1j * imag_part

    raise ValueError(f"No numeric matrix row found in {filepath}")


def collect_data(base_dir, filename):
    rows = []

    base_path = Path(base_dir)

    for folder in sorted(base_path.glob("Output_screening_qx_*_qy_*")):
        match = DIR_PATTERN.fullmatch(folder.name)
        if match is None:
            continue

        filepath = folder / filename

        if not filepath.exists():
            print(f"Skipping missing file: {filepath}")
            continue

        qx = float(match.group(1))
        qy = float(match.group(2))
        q_abs = np.sqrt(qx * qx + qy * qy)

        eps_inv_00 = read_eps_inv_00(filepath)
        eps_00 = 1.0 / eps_inv_00

        rows.append(
            (
                qx,
                qy,
                q_abs,
                eps_inv_00.real,
                eps_inv_00.imag,
                eps_00.real,
                eps_00.imag,
            )
        )

    if not rows:
        raise RuntimeError(
            "No data found. Check --base-dir, --filename, and folder names."
        )

    data = np.array(rows, dtype=float)
    return data[np.argsort(data[:, 2])]


def fit_directional_screening_lengths(data, qmax=None, zero_tol=1e-10):
    qx = data[:, 0]
    qy = data[:, 1]
    eps_macro_real = data[:, 5]

    x_mask = np.abs(qy) < zero_tol
    y_mask = np.abs(qx) < zero_tol

    if qmax is not None:
        x_mask &= np.abs(qx) <= qmax
        y_mask &= np.abs(qy) <= qmax

    results = {}

    for direction, q_values, mask in [
        ("x", np.abs(qx), x_mask),
        ("y", np.abs(qy), y_mask),
    ]:
        q_fit = q_values[mask]
        eps_fit = eps_macro_real[mask]

        order = np.argsort(q_fit)
        q_fit = q_fit[order]
        eps_fit = eps_fit[order]

        if len(q_fit) < 2:
            print(f"Warning: not enough points for {direction}-direction fit.")
            results[direction] = None
            continue

        coeffs = np.polyfit(q_fit, eps_fit, 1)
        r0 = coeffs[0]
        eps0 = coeffs[1]

        results[direction] = {
            "r0": r0,
            "eps0": eps0,
            "q_fit": q_fit,
            "eps_fit": eps_fit,
        }

    return results


def plot_directional_results(data, fit_results, output_prefix):
    qx = data[:, 0]
    qy = data[:, 1]
    eps_macro_real = data[:, 5]

    plt.figure(figsize=(6, 4))

    for direction, marker, color, ls in [("x", "o", "black", "-"), ("y", "^", "red", "--")]:
        result = fit_results[direction]

        if direction == "x":
            mask = np.abs(qy) < 1e-10
            q_plot = np.abs(qx[mask])
        else:
            mask = np.abs(qx) < 1e-10
            q_plot = np.abs(qy[mask])

        eps_plot = eps_macro_real[mask]
        order = np.argsort(q_plot)

        plt.scatter(
            q_plot[order],
            eps_plot[order],
            marker=marker,
            label=fr"data ${direction}$",
            c=color,
            alpha=0.7            
        )

        if result is not None:
            q_line = np.linspace(0.0, result["q_fit"].max(), 300)
            eps_line = result["eps0"] + result["r0"] * q_line

            plt.plot(
                q_line,
                eps_line,
                label=fr"${direction}$ fit: $r_0 = {result['r0']:.3f}$",
                c=color,
                linestyle=ls
            )

    plt.xlabel(r"$|q|$")
    plt.ylabel(r"Re $\epsilon_M(q)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_directional_epsilon_fit.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description="Extract epsilon^{-1}_{00}(q), invert it, and fit r0."
    )

    parser.add_argument(
        "--base-dir",
        default="results-Q-inverse",
        help="Directory containing Output_screening_qx_*_qy_* folders.",
    )

    parser.add_argument(
        "--filename",
        default="GaAs_invpesilon.dat",
        help="Dielectric matrix filename inside each q folder.",
    )

    parser.add_argument(
        "--qmax",
        type=float,
        default=None,
        help="Maximum |q| used in the small-q fit.",
    )

    parser.add_argument(
        "--output",
        default="screening_data.dat",
        help="Output data file.",
    )

    parser.add_argument(
        "--plot-prefix",
        default="screening",
        help="Prefix for output plot files.",
    )

    args = parser.parse_args()

    data = collect_data(args.base_dir, args.filename)

    np.savetxt(
        args.output,
        data,
        header=(
            "qx qy q_abs "
            "Re_eps_inv_00 Im_eps_inv_00 "
            "Re_eps_00 Im_eps_00"
        ),
    )

    fit_results = fit_directional_screening_lengths(data, args.qmax)

    for direction in ["x", "y"]:
        result = fit_results[direction]

        if result is None:
            continue

        print(f"\n{direction}-direction fit:")
        print(f"r0_{direction} = {result['r0']:.12g}")
        print(f"eps0_{direction} = {result['eps0']:.12g}")

    plot_directional_results(data, fit_results, args.plot_prefix)

    print(f"Wrote data to: {args.output}")
    print(f"Number of q points: {len(data)}")
    print(f"Number of q points in fit: {len(fit_results['x']['q_fit'])} (x), {len(fit_results['y']['q_fit'])} (y)")


if __name__ == "__main__":
    main()