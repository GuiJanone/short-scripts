#!/usr/bin/env python3
"""
kwf_plot.py - Plot excitonic wavefunctions from a KWF file.

Changes in this version:
- Square plots by default (both figure shape and data extent).
- New flag --no-square to disable squaring (keeps backward compatibility).
- Use constrained_layout to avoid colorbar/labels distorting aspect.
- For tri interpolation, NaNs are masked to avoid artifacts.

Example usage:
  python kwf_plot.py data.kwf -n 6
  python kwf_plot.py data.kwf -n 10 --eigvals EIG.dat --deg-thresh 1e-3
  python kwf_plot.py data.kwf --select 1-4,7 --format png --out figs/run1 --interp gouraud --font-size 16 --no-latex
  python kwf_plot.py data.kwf -n 8 --interp tri --grid 300 --cmap plasma
"""
from __future__ import annotations
import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.tri import Triangulation, LinearTriInterpolator
from typing import List, Tuple, Optional

# ------------ IO & parsing ------------

def configure_matplotlib(fontsize: int = 15, latex: bool = True) -> None:
    """Set global matplotlib rcParams."""
    if latex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    else:
        plt.rcParams.update({"text.usetex": False})
    plt.rcParams["axes.labelsize"]  = fontsize
    plt.rcParams["axes.titlesize"]  = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"]= fontsize
    plt.rcParams["font.size"]       = fontsize


def read_kwf_file(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Read excitonic wavefunction file.

    Returns:
      kpoints: (N x 3) array of k-point coords (only first 2 used)
      states : list of length M, each a (N,) array of coefficients
    """
    ks_blocks: List[List[List[float]]] = []
    states: List[np.ndarray] = []
    kpoints: List[List[float]] = []
    coefs: List[float] = []

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            c0 = line[0]
            if c0 == 'k':
                continue
            if c0 == '#':
                if coefs:
                    states.append(np.array(coefs, dtype=float))
                    ks_blocks.append(kpoints)
                kpoints = []
                coefs = []
                continue
            parts = line.split()
            kp = [float(parts[i]) for i in range(3)]
            kpoints.append(kp)
            coefs.append(float(parts[-1]))

    if coefs:
        states.append(np.array(coefs, dtype=float))
        ks_blocks.append(kpoints)

    if not ks_blocks:
        raise ValueError(f"No states found in file: {path}")

    kpoints_arr = np.array(ks_blocks[0], dtype=float)  # assume same grid for all states
    return kpoints_arr, states


def read_eigenvalues(path: str, n: int) -> np.ndarray:
    """Read eigenvalues file: skip first three lines, then read n numeric values."""
    vals: List[float] = []
    with open(path, 'r') as f:
        for _ in range(3):
            next(f, None)
        while len(vals) < n:
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
    return np.array(vals, dtype=float)


# ------------ Processing ------------

def combine_degenerate_states(
    eigvals: np.ndarray,
    states: List[np.ndarray],
    threshold: float
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Combine (pairs of) wavefunctions whose eigenvalues are within 'threshold'.
    Titles include energies. Keeps order, merges i and i+1 when degenerate.
    """
    combined_states: List[np.ndarray] = []
    titles: List[str] = []
    i = 0
    while i < len(eigvals):
        if i + 1 < len(eigvals) and abs(eigvals[i] - eigvals[i + 1]) <= threshold:
            wf = states[i] + states[i + 1]
            combined_states.append(wf)
            titles.append(rf"Exciton $\\psi$ \#{i+1}+{i+2}  E={eigvals[i]:.6f}, {eigvals[i+1]:.6f}")
            i += 2
        else:
            combined_states.append(states[i])
            titles.append(rf"Exciton $\\psi$ \#{i+1}  E={eigvals[i]:.6f}")
            i += 1
    return combined_states, titles


def parse_selection(select_str: str, max_n: int) -> List[int]:
    """
    Parse selection like '1,2,5-7' into 0-based indices. Clips to [0, max_n).
    """
    out: List[int] = []
    parts = [p.strip() for p in select_str.split(',') if p.strip()]
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            start = max(1, int(a))
            end   = int(b)
            if end < start:
                start, end = end, start
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(p))
    idxs = sorted({i-1 for i in out if 1 <= i <= max_n})
    return idxs


# ------------ Plotting ------------

def regular_grid_from_bounds(xmin, xmax, ymin, ymax, n: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    return np.meshgrid(xs, ys)


def make_titles(n: int) -> List[str]:
    return [rf"Exciton $\\psi$ \#{i+1}" for i in range(n)]


def plot_state_tripcolor(ax, kx, ky, wf, shading: str, cmap: str):
    # shading in {'flat','gouraud'}
    t = ax.tripcolor(kx, ky, wf, shading=shading, cmap=cmap)
    t.set_edgecolor("face")
    return t


def plot_state_tri_interp(ax, kx, ky, wf, bounds, grid_n: int, cmap: str):
    triang = Triangulation(kx, ky)
    interp = LinearTriInterpolator(triang, wf)
    (xmin, xmax, ymin, ymax) = bounds
    Xi, Yi = regular_grid_from_bounds(xmin, xmax, ymin, ymax, grid_n)
    Zi = interp(Xi, Yi)
    # Mask out NaNs to avoid artifacts at the convex hull boundary
    Zi = np.ma.masked_invalid(Zi)
    t = ax.pcolormesh(Xi, Yi, Zi, shading='auto', cmap=cmap)
    t.set_edgecolor("face")
    return t


def set_axes(ax, xlim, ylim, xlabel, ylabel, title, square: bool):
    if square:
        ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def save_pdf(pages: List[plt.Figure], path: str) -> None:
    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig)  # keep figure sizing/aspect as-is
            plt.close(fig)


def save_pngs(figs: List[plt.Figure], prefix: str, dpi: int) -> None:
    for i, fig in enumerate(figs, start=1):
        fname = f"{prefix}_{i}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


# ------------ Main CLI ------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kwf_plot.py",
        description="Plot excitonic wavefunctions (KWF) to PDF or PNG with optional interpolation and eigenvalue-aware titles."
    )
    p.add_argument("kwf_file", help="KWF file containing exciton wavefunctions.")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("-n", "--n-states", type=int, help="Number of states to plot (from the beginning).")
    sel.add_argument("--select", type=str, help="Explicit state list/ranges (1-based), e.g. '1,2,5-7'.")

    p.add_argument("--eigvals", type=str, default=None, help="Eigenvalues file (skips first 3 lines).")
    p.add_argument("--deg-thresh", type=float, default=None,
                   help="Threshold for combining degenerate pairs (e.g. 1e-3). Disabled if omitted.")

    p.add_argument("--interp", choices=["flat", "gouraud", "tri"], default="flat",
                   help="Color interpolation: flat/gouraud shading on triangulation, or 'tri' regular-grid interpolation.")
    p.add_argument("--grid", type=int, default=250, help="Grid size (N x N) for --interp tri.")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name.")
    p.add_argument("--format", choices=["pdf", "png"], default="pdf", help="Output format.")
    p.add_argument("--out", type=str, default=None, help="Output base path (e.g. 'out/run1').")
    p.add_argument("--dpi", type=int, default=600, help="PNG DPI.")
    p.add_argument("--font-size", type=int, default=15, help="Base font size.")
    p.add_argument("--no-latex", action="store_true", help="Disable LaTeX text rendering.")
    p.add_argument("--xlabel", default=r"$k_x$ (Å$^{-1}$)", help="X axis label.")
    p.add_argument("--ylabel", default=r"$k_y$ (Å$^{-1}$)", help="Y axis label.")

    # Square is now the default; provide an opt-out.
    p.add_argument("--square", action="store_true", help="(Deprecated) Square aspect ratio (now default).")
    p.add_argument("--no-square", action="store_true", help="Disable square aspect/extent.")

    p.add_argument("--xlim", type=float, default=None, help="Override |kx| limit (default: auto from data).")
    p.add_argument("--ylim", type=float, default=None, help="Override |ky| limit (default: auto from data).")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--version", action="version", version="kwf_plot.py 2.1")

    # Make square the default behavior
    p.set_defaults(square=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve square vs no-square
    if getattr(args, "no_square", False):
        args.square = False  # explicit opt-out

    configure_matplotlib(fontsize=args.font_size, latex=not args.no_latex)

    # Load KWF
    kpoints, states = read_kwf_file(args.kwf_file)
    total_states = len(states)

    # Selection
    if args.select:
        idxs = parse_selection(args.select, total_states)
        if not idxs:
            print("No valid state indices selected.", file=sys.stderr)
            return 2
        states = [states[i] for i in idxs]
        titles = make_titles(len(states))
        eig_slice = idxs
    else:
        n = args.n_states
        if n is None or n <= 0:
            print("Please provide a positive --n-states.", file=sys.stderr)
            return 2
        if n > total_states:
            print(f"Requested {n} states, but file contains only {total_states}.", file=sys.stderr)
            return 2
        states = states[:n]
        titles = make_titles(n)
        eig_slice = list(range(n))

    # Eigenvalues (optional)
    if args.eigvals:
        eigvals_all = read_eigenvalues(args.eigvals, max(eig_slice) + 1)
        eigvals = eigvals_all[eig_slice]
        if args.deg_thresh is not None and args.deg_thresh > 0:
            states, titles = combine_degenerate_states(eigvals, states, threshold=args.deg_thresh)
        else:
            titles = [f"{t}  E={e:.6f}" for t, e in zip(titles, eigvals)]

    # Determine limits
    kx = kpoints[:, 0]
    ky = kpoints[:, 1]
    auto_xlim = np.max(np.abs(kx))
    auto_ylim = np.max(np.abs(ky))
    xlim = args.xlim if args.xlim is not None else auto_xlim
    ylim = args.ylim if args.ylim is not None else auto_ylim

    # If square is desired, enforce identical extents to guarantee a square field
    if args.square:
        L = max(xlim, ylim)
        xlim = L
        ylim = L
    bounds = (-xlim, xlim, -ylim, ylim)

    # Output naming
    base_in = os.path.splitext(os.path.basename(args.kwf_file))[0]
    outbase = args.out if args.out else f"{base_in}_wf"

    figs: List[plt.Figure] = []
    for idx, (wf, title) in enumerate(zip(states, titles), start=1):
        # Square figure and constrained layout -> prevents colorbar/labels from distorting aspect
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

        if args.interp in ("flat", "gouraud"):
            mappable = plot_state_tripcolor(ax, kx, ky, wf, shading=args.interp, cmap=args.cmap)
        else:  # 'tri' regular grid interpolation
            mappable = plot_state_tri_interp(ax, kx, ky, wf, (bounds[0], bounds[1], bounds[2], bounds[3]), args.grid, args.cmap)

        set_axes(ax, xlim, ylim, args.xlabel, args.ylabel, title, square=args.square)

        cbar = fig.colorbar(mappable, fraction=0.046, pad=0.04, ax=ax)
        cbar.set_label("Amplitude")

        figs.append(fig)

    if args.format == "pdf":
        pdf_path = f"{outbase}.pdf"
        if args.verbose:
            print(f"Writing PDF: {pdf_path}")
        save_pdf(figs, pdf_path)
    else:
        if args.verbose:
            print(f"Writing PNGs with prefix: {outbase}_<idx>.png (dpi={args.dpi})")
        save_pngs(figs, outbase, dpi=args.dpi)

    if args.verbose:
        if args.eigvals:
            print("Eigenvalues used:", "yes")
            if args.deg_thresh:
                print(f"Degenerate-combine threshold: {args.deg_thresh}")
        print(f"Interpolation mode: {args.interp}")
        print(f"Colormap: {args.cmap}")
        print(f"States plotted: {len(states)}")
        print(f"Square plots: {'on' if args.square else 'off'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
