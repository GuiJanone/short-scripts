#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt


DIR_PATTERN = re.compile(
    r"Output_screening_q([xy])_([-+0-9.eE]+)"
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

    for folder in sorted(base_path.glob("Output_screening_q[xy]_*")):
        match = DIR_PATTERN.fullmatch(folder.name)
        if match is None:
            continue

        direction = match.group(1)
        q = float(match.group(2))

        if direction == "x":
            qx, qy = q, 0.0
        else:
            qx, qy = 0.0, q

        filepath = folder / filename
        if not filepath.exists():
            print(f"Skipping missing file: {filepath}")
            continue

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


def plot_directional_results(data, fit_results, output_prefix, include_fit=True):
    qx = data[:, 0]
    qy = data[:, 1]
    eps_real = data[:, 5]
    eps_imag = data[:, 6]

    for part_name, eps_data, ylabel, suffix in [
        ("real", eps_real, r"Re $\epsilon_M(q)$", "real"),
        ("imag", eps_imag, r"Im $\epsilon_M(q)$", "imag"),
    ]:
        plt.figure(figsize=(6, 4))

        for direction, marker, color in [
            ("x", "o", "black"),
            ("y", "^", "red"),
        ]:
            if direction == "x":
                mask = np.abs(qy) < 1e-10
                q_plot = np.abs(qx[mask])
            else:
                mask = np.abs(qx) < 1e-10
                q_plot = np.abs(qy[mask])

            eps_plot = eps_data[mask]
            order = np.argsort(q_plot)

            plt.scatter(
                q_plot[order],
                eps_plot[order],
                marker=marker,
                label=fr"${direction}$ direction",
                c=color,
                alpha=0.7,
            )

        plt.xlabel(r"$|q|$")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_epsilon_{suffix}.png", dpi=300)


def write_macro_screening(data, output="macro_screening.dat", zero_tol=1e-10):
    qx = data[:, 0]
    qy = data[:, 1]
    eps_macro_real = data[:, 5]

    has_x = np.any(np.abs(qx) >= zero_tol)
    has_y = np.any(np.abs(qy) >= zero_tol)

    if has_x and has_y:
        x_mask = np.abs(qy) < zero_tol
        y_mask = np.abs(qx) < zero_tol
        q_x = np.abs(qx[x_mask])
        q_y = np.abs(qy[y_mask])
        eps_x = eps_macro_real[x_mask]
        eps_y = eps_macro_real[y_mask]
        q_values = np.unique(np.concatenate((q_x, q_y)))
        macro_data = np.full((len(q_values), 3), np.nan)
        macro_data[:, 0] = q_values

        for row, q_value in enumerate(q_values):
            x_match = np.isclose(q_x, q_value, atol=zero_tol, rtol=0.0)
            y_match = np.isclose(q_y, q_value, atol=zero_tol, rtol=0.0)

            if np.any(x_match):
                macro_data[row, 1] = eps_x[x_match][0]
            if np.any(y_match):
                macro_data[row, 2] = eps_y[y_match][0]

        header = "q epsilon_M_qx epsilon_M_qy"
    else:
        direction_mask = np.abs(qy) < zero_tol if has_x else np.abs(qx) < zero_tol
        q_values = np.abs(qx[direction_mask] if has_x else qy[direction_mask])
        eps_values = eps_macro_real[direction_mask]
        order = np.argsort(q_values)
        macro_data = np.column_stack((q_values[order], eps_values[order]))
        header = "q epsilon_M"

    np.savetxt(output, macro_data, header=header)


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
        "--no-fit",
        action="store_true",
        help="Skip the small-q fit and plot only the extracted data.",
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

    macro_screening_output = "macro_screening.dat"
    write_macro_screening(data, macro_screening_output)

    fit_results = None

    if not args.no_fit:
        fit_results = fit_directional_screening_lengths(data, args.qmax)

        for direction in ["x", "y"]:
            result = fit_results[direction]

            if result is None:
                continue

            print(f"\n{direction}-direction fit:")
            print(f"r0_{direction} = {result['r0']:.12g}")
            print(f"eps0_{direction} = {result['eps0']:.12g}")

    plot_directional_results(data, fit_results, args.plot_prefix, include_fit=not args.no_fit)

    print(f"Wrote data to: {args.output}")
    print(f"Wrote plotted macro screening data to: {macro_screening_output}")
    print(f"Number of q points: {len(data)}")
    if fit_results is not None:
        fit_counts = {
            direction: len(result["q_fit"]) if result is not None else 0
            for direction, result in fit_results.items()
        }
        print(f"Number of q points in fit: {fit_counts['x']} (x), {fit_counts['y']} (y)")


if __name__ == "__main__":
    main()
