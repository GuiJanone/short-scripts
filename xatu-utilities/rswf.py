#!/usr/bin/env python3
"""
rswf.py - Plot excitonic real-space wavefunction densities.

Accepted input formats per state block:
  rx ry prob
  rx ry rz prob

Blocks are separated by lines starting with '#'. By default, the first numeric
row of each block is treated as the fixed hole position and is not plotted as
part of the electron density. This preserves the convention used by the old
script.

Examples:
  python rswf.py exciton.rswf --select 1 --mode 3d --format pdf --out exciton_1
  python rswf.py exciton.rswf --select 1,2 --sum-selected --mode 3d --format pdf --out exciton_1_2
  python rswf.py exciton.rswf --select 1 --backend plotly --format html --threshold 0.01 --out exciton_1
  python rswf.py exciton.rswf --select 1,2 --sum-selected --backend plotly --format html --threshold 0.01 --out exciton_1_2

Notes:
  - The 3D plot is a scatter plot on the real-space sampling points contained
    in the file. It does not invent intermediate z points.
  - If the file only contains two distinct z layers, the 3D plot will show two
    sheets. Use --mode 2d --plane xz or --plane yz to inspect the z profile.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


@dataclass
class RealSpaceData:
    coords: np.ndarray
    states: List[np.ndarray]
    hole_position: np.ndarray


@dataclass
class PlotState:
    prob: np.ndarray
    title: str
    raw_indices: List[int]


# ------------------------- Matplotlib setup -------------------------


def configure_matplotlib(fontsize: int = 15, latex: bool = True) -> None:
    """Set global matplotlib options."""
    if latex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    else:
        plt.rcParams.update({"text.usetex": False})

    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize
    plt.rcParams["font.size"] = fontsize


# ------------------------- IO and parsing -------------------------


def read_rswf_file(path: str, hole_row: bool = True) -> RealSpaceData:
    """
    Read a real-space wavefunction file.

    Accepted numeric rows are:
      rx ry prob
      rx ry rz prob

    If more than four columns are present, the first three columns are treated
    as coordinates and the last column is treated as the probability.
    """
    coord_blocks: List[np.ndarray] = []
    state_blocks: List[np.ndarray] = []
    hole_positions: List[np.ndarray] = []

    current_coords: List[List[float]] = []
    current_prob: List[float] = []

    def finish_block() -> None:
        nonlocal current_coords, current_prob
        if not current_prob:
            current_coords = []
            current_prob = []
            return

        coords_arr = np.array(current_coords, dtype=float)
        prob_arr = np.array(current_prob, dtype=float)

        if hole_row:
            if coords_arr.shape[0] < 2:
                raise ValueError(
                    "A state block has fewer than two rows. Use --no-hole-row "
                    "if the file does not store the hole as the first row."
                )
            hole_positions.append(coords_arr[0].copy())
            coords_arr = coords_arr[1:]
            prob_arr = prob_arr[1:]
        else:
            hole_positions.append(np.zeros(3, dtype=float))

        coord_blocks.append(coords_arr)
        state_blocks.append(prob_arr)
        current_coords = []
        current_prob = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                finish_block()
                continue

            parts = stripped.split()
            try:
                values = [float(item) for item in parts]
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric row at line {line_number}: {line.rstrip()}"
                ) from exc

            if len(values) == 3:
                if hole_row and not current_coords:
                    rx, ry, rz = values
                    prob = 0.0
                else:
                    rx, ry, prob = values
                    rz = 0.0
            elif len(values) >= 4:
                rx, ry, rz = values[0], values[1], values[2]
                prob = values[-1]
            else:
                raise ValueError(
                    f"Expected 3 or at least 4 columns at line {line_number}, "
                    f"got {len(values)}."
                )

            current_coords.append([rx, ry, rz])
            current_prob.append(prob)

    finish_block()

    if not state_blocks:
        raise ValueError(f"No states found in file: {path}")

    reference_coords = coord_blocks[0]
    for state_index, coords_arr in enumerate(coord_blocks[1:], start=2):
        if coords_arr.shape != reference_coords.shape:
            raise ValueError(
                f"State {state_index} has a different number of grid points."
            )
        if not np.allclose(coords_arr, reference_coords, rtol=1e-10, atol=1e-12):
            raise ValueError(f"State {state_index} uses a different real-space grid.")

    hole_position = hole_positions[0]
    for state_index, candidate in enumerate(hole_positions[1:], start=2):
        if not np.allclose(candidate, hole_position, rtol=1e-10, atol=1e-12):
            print(
                f"Warning: hole position in state {state_index} differs from "
                "state 1. Using the state-1 hole position.",
                file=sys.stderr,
            )
            break

    return RealSpaceData(
        coords=reference_coords,
        states=state_blocks,
        hole_position=hole_position,
    )


def read_eigenvalues(path: str, n: int) -> np.ndarray:
    """Read eigenvalues file: skip first three lines, then read n numeric values."""
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for _ in range(3):
            next(handle, None)
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(float(stripped.split()[0]))
            except ValueError:
                continue
            if len(values) == n:
                break

    if len(values) < n:
        raise ValueError(f"Expected {n} eigenvalues but found {len(values)} in {path}")
    return np.array(values, dtype=float)


# ------------------------- State selection and combining -------------------------


def parse_selection(select_str: str, max_n: int) -> List[int]:
    """Parse a 1-based selection string like '1,2,5-7'."""
    selected: List[int] = []
    for item in select_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))

    return sorted({idx - 1 for idx in selected if 1 <= idx <= max_n})


def parse_degeneracies(text: str) -> List[int]:
    """Parse a comma-separated list such as '1,2,1,3'."""
    degeneracies = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not degeneracies or any(item <= 0 for item in degeneracies):
        raise ValueError("Degeneracies must be positive integers.")
    return degeneracies


def make_plot_states(states: Sequence[np.ndarray], raw_indices: Sequence[int]) -> List[PlotState]:
    """Wrap raw states in PlotState objects."""
    out: List[PlotState] = []
    for prob, raw_index in zip(states, raw_indices):
        out.append(
            PlotState(
                prob=np.array(prob, dtype=float),
                title=f"Exciton psi #{raw_index + 1}",
                raw_indices=[raw_index],
            )
        )
    return out


def combine_selected_states(plot_states: Sequence[PlotState]) -> List[PlotState]:
    """Sum all selected states into one PlotState."""
    if not plot_states:
        return []
    prob = np.sum([item.prob for item in plot_states], axis=0)
    raw_indices = [idx for item in plot_states for idx in item.raw_indices]
    label = "+".join(str(idx + 1) for idx in raw_indices)
    return [
        PlotState(
            prob=prob,
            title=f"Exciton psi #{label}",
            raw_indices=raw_indices,
        )
    ]


def combine_with_manual_degeneracies(
    plot_states: Sequence[PlotState],
    degeneracies: Sequence[int],
    eigvals: Optional[np.ndarray] = None,
) -> List[PlotState]:
    """Sum consecutive states according to explicit degeneracy groups and attach energies if available."""
    if sum(degeneracies) > len(plot_states):
        raise ValueError("The degeneracy groups contain more states than selected.")

    combined: List[PlotState] = []
    offset = 0
    for degeneracy in degeneracies:
        group = plot_states[offset:offset + degeneracy]
        prob = np.sum([item.prob for item in group], axis=0)
        raw_indices = [idx for item in group for idx in item.raw_indices]
        label = "+".join(str(idx + 1) for idx in raw_indices)
        
        # Build title with energies if the eigenvalue file is loaded
        if eigvals is not None:
            energies = ", ".join(f"{eigvals[k]:.6f}" for k in range(offset, offset + degeneracy))
            title = f"Exciton psi #{label}  E={energies}"
        else:
            title = f"Exciton psi #{label}"
            
        combined.append(
            PlotState(
                prob=prob,
                title=title,
                raw_indices=raw_indices,
            )
        )
        offset += degeneracy

    if offset < len(plot_states):
        # Handle any leftover states that weren't explicitly grouped
        for i in range(offset, len(plot_states)):
            item = plot_states[i]
            if eigvals is not None:
                title = f"{item.title}  E={eigvals[i]:.6f}"
            else:
                title = item.title
            combined.append(
                PlotState(
                    prob=item.prob,
                    title=title,
                    raw_indices=item.raw_indices,
                )
            )
    return combined


def combine_with_eigenvalue_threshold(
    plot_states: Sequence[PlotState],
    eigvals: np.ndarray,
    threshold: float,
) -> List[PlotState]:
    """Sum adjacent selected states whose eigenvalues differ by at most threshold."""
    combined: List[PlotState] = []
    i = 0
    while i < len(plot_states):
        group = [plot_states[i]]
        j = i + 1
        # Compare eigenvalues mapped directly via identical indices matching the plot_states list
        while j < len(plot_states) and abs(eigvals[j] - eigvals[j - 1]) <= threshold:
            group.append(plot_states[j])
            j += 1

        prob = np.sum([item.prob for item in group], axis=0)
        raw_indices = [idx for item in group for idx in item.raw_indices]
        label = "+".join(str(idx + 1) for idx in raw_indices)
        energies = ", ".join(f"{eigvals[k]:.6f}" for k in range(i, j))
        combined.append(
            PlotState(
                prob=prob,
                title=f"Exciton psi #{label}  E={energies}",
                raw_indices=raw_indices,
            )
        )
        i = j
    return combined


def attach_eigenvalue_titles(
    plot_states: Sequence[PlotState],
    eigvals: np.ndarray,
) -> List[PlotState]:
    """Append eigenvalues to titles without combining states."""
    out: List[PlotState] = []
    for item, eigval in zip(plot_states, eigvals):
        out.append(
            PlotState(
                prob=item.prob,
                title=f"{item.title}  E={eigval:.6f}",
                raw_indices=item.raw_indices,
            )
        )
    return out


# ------------------------- Data transforms -------------------------


def normalize_probability(prob: np.ndarray, mode: str) -> np.ndarray:
    """Normalize probability for plotting."""
    prob = np.array(prob, dtype=float)
    if mode == "none":
        return prob.copy()
    if mode == "max":
        max_value = np.max(prob)
        return prob / max_value if max_value > 0.0 else prob.copy()
    if mode == "sum":
        total = np.sum(prob)
        return prob / total if total > 0.0 else prob.copy()
    raise ValueError(f"Unknown normalization mode: {mode}")


def centered_coordinates(
    coords: np.ndarray,
    hole_position: np.ndarray,
    absolute_coords: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return plotting coordinates and hole position."""
    if absolute_coords:
        return coords.copy(), hole_position.copy()
    return coords - hole_position[None, :], np.zeros(3, dtype=float)


def project_points_to_plane(coords: np.ndarray, plane: str) -> np.ndarray:
    """Project coordinates onto a 2D plane without density reduction."""
    index_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    i, j = index_map[plane]
    return coords[:, [i, j]]


def project_to_plane_exact(
    coords: np.ndarray,
    prob: np.ndarray,
    plane: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project density by summing only exactly identical plane coordinates."""
    points = project_points_to_plane(coords, plane)

    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    projected_prob = np.zeros(unique_points.shape[0], dtype=float)
    np.add.at(projected_prob, inverse, prob)
    return unique_points[:, 0], unique_points[:, 1], projected_prob


def infer_projection_bin_size(points: np.ndarray) -> float:
    """Infer a projection bin size from median nearest spacing in plotted axes."""
    differences: List[np.ndarray] = []
    for axis in range(points.shape[1]):
        values = np.unique(np.round(points[:, axis], decimals=8))
        if values.size < 2:
            continue
        diffs = np.diff(np.sort(values))
        positive = diffs[diffs > 0.0]
        if positive.size:
            differences.append(positive)

    if not differences:
        return 0.1

    inferred = float(np.median(np.concatenate(differences)))
    if inferred <= 0.0 or not np.isfinite(inferred):
        return 0.1
    return inferred


def project_to_plane_binned(
    coords: np.ndarray,
    prob: np.ndarray,
    plane: str,
    bin_size: Optional[float],
    reducer: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project density onto a regular 2D grid and reduce probabilities per bin."""
    points = project_points_to_plane(coords, plane)
    if bin_size is None:
        bin_size = infer_projection_bin_size(points)

    origin = np.min(points, axis=0)
    bin_indices = np.rint((points - origin) / bin_size).astype(int)
    unique_bins, inverse = np.unique(bin_indices, axis=0, return_inverse=True)

    if reducer == "sum":
        projected_prob = np.zeros(unique_bins.shape[0], dtype=float)
        np.add.at(projected_prob, inverse, prob)
    elif reducer == "max":
        projected_prob = np.full(unique_bins.shape[0], -np.inf, dtype=float)
        np.maximum.at(projected_prob, inverse, prob)
        projected_prob[~np.isfinite(projected_prob)] = 0.0
    else:
        raise ValueError(f"Unknown projection reducer: {reducer}")

    projected_points = origin + unique_bins * bin_size
    return projected_points[:, 0], projected_points[:, 1], projected_prob


def project_density_to_plane(
    coords: np.ndarray,
    prob: np.ndarray,
    plane: str,
    projection: str,
    bin_size: Optional[float],
    reducer: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch density projection to exact-coordinate or binned reduction."""
    if projection == "exact":
        return project_to_plane_exact(coords, prob, plane)
    if projection == "bin":
        return project_to_plane_binned(coords, prob, plane, bin_size, reducer)
    raise ValueError(f"Unknown projection mode: {projection}")


def plane_labels(plane: str) -> Tuple[str, str]:
    """Return axis labels for a projection plane."""
    labels = {
        "x": r"$x$ (\AA)",
        "y": r"$y$ (\AA)",
        "z": r"$z$ (\AA)",
    }
    return labels[plane[0]], labels[plane[1]]


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters when `text.usetex` is enabled.

    If LaTeX rendering is disabled, the original text is returned.
    """
    try:
        use_latex = bool(plt.rcParams.get("text.usetex", False))
    except Exception:
        use_latex = False

    if not use_latex or not text:
        return text

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }

    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


# ------------------------- Plot scaling helpers -------------------------


def positive_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a percentile from positive finite values."""
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return 1.0
    return float(np.percentile(valid, percentile))


def color_norm(values: np.ndarray, args: argparse.Namespace) -> Optional[mcolors.Normalize]:
    """Build a matplotlib normalization object for better contrast."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    vmax = positive_percentile(finite, args.vmax_percentile)
    if args.vmin is not None:
        vmin = args.vmin
    elif args.color_scale == "log":
        vmin = positive_percentile(finite, args.vmin_percentile)
    else:
        vmin = 0.0

    if vmax <= vmin:
        vmax = float(np.max(finite)) if np.max(finite) > vmin else vmin + 1.0

    if args.color_scale == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    if args.color_scale == "power":
        return mcolors.PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax, clip=True)
    if args.color_scale == "log":
        safe_vmin = max(vmin, np.finfo(float).tiny)
        safe_vmax = max(vmax, safe_vmin * 10.0)
        return mcolors.LogNorm(vmin=safe_vmin, vmax=safe_vmax, clip=True)
    raise ValueError(f"Unknown color scale: {args.color_scale}")


def add_matplotlib_colorbar(fig, ax, scatter, args: argparse.Namespace, pad: float):
    """Add a colorbar with controlled tick density."""
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=pad)
    cbar.set_label(args.cbar_label)
    if not args.keep_cbar_top_tick:
        cbar.locator = MaxNLocator(nbins=args.cbar_ticks, prune="upper")
    else:
        cbar.locator = MaxNLocator(nbins=args.cbar_ticks)
    cbar.update_ticks()
    return cbar


def color_limits(values: np.ndarray, args: argparse.Namespace) -> Tuple[float, float]:
    """Return numeric color limits for non-matplotlib backends."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    vmax = positive_percentile(finite, args.vmax_percentile)
    if args.vmin is not None:
        vmin = args.vmin
    elif args.color_scale == "log":
        vmin = positive_percentile(finite, args.vmin_percentile)
    else:
        vmin = 0.0

    if vmax <= vmin:
        max_finite = float(np.max(finite))
        vmax = max_finite if max_finite > vmin else vmin + 1.0
    return float(vmin), float(vmax)


def plotly_color_values(values: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, float, float]:
    """Transform values and limits for Plotly color rendering."""
    vmin, vmax = color_limits(values, args)
    if args.color_scale == "power":
        clipped = np.clip(values, max(vmin, 0.0), vmax)
        return clipped ** args.gamma, max(vmin, 0.0) ** args.gamma, vmax ** args.gamma
    if args.color_scale == "log":
        safe_vmin = max(vmin, np.finfo(float).tiny)
        safe_vmax = max(vmax, safe_vmin * 10.0)
        clipped = np.clip(values, safe_vmin, safe_vmax)
        return np.log10(clipped), np.log10(safe_vmin), np.log10(safe_vmax)
    return values, vmin, vmax


def plotly_colorscale_name(cmap: str) -> str:
    """Map common matplotlib colormap names to Plotly colorscales."""
    mapping = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
    }
    key = cmap.lower()
    if key not in mapping:
        print(
            f"Warning: unknown Plotly colorscale for cmap '{cmap}', using Viridis.",
            file=sys.stderr,
        )
    return mapping.get(key, "Viridis")


def marker_sizes(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Return scatter marker sizes.

    The default behavior mimics the old script qualitatively: larger markers
    indicate larger probability, while tiny nonzero values remain visible.
    """
    values = np.array(values, dtype=float)
    if not args.scale_markers:
        return np.full(values.shape, args.marker_size, dtype=float)

    clipped = np.clip(values, 0.0, None)
    max_value = np.max(clipped) if clipped.size else 0.0
    if max_value > 0.0:
        scaled = (clipped / max_value) ** args.size_gamma
    else:
        scaled = np.zeros_like(clipped)

    min_size = args.marker_size * args.min_marker_fraction
    return min_size + (args.marker_size - min_size) * scaled


def axis_limits(values: np.ndarray, override: Optional[float], symmetric: bool = True) -> Tuple[float, float]:
    """Return axis limits. If symmetric=False, bounds to min/max with padding."""
    if override is not None:
        return -abs(override), abs(override)
    if values.size == 0:
        return -1.0, 1.0
        
    if symmetric:
        bound = np.max(np.abs(values))
        if bound == 0.0:
            bound = 1.0
        return -bound, bound
    else:
        # Non-symmetric: Use min and max with a 20% padding margin
        vmin, vmax = np.min(values), np.max(values)
        vdist = vmax - vmin
        if vdist == 0.0:
            vdist = 1.0  # Avoid zero-width dimensions
            
        return vmin - 0.2 * abs(vmin if vmin != 0 else vdist), vmax + 0.2 * abs(vmax if vmax != 0 else vdist)


def finite_aspect_limits(coords: np.ndarray, args: argparse.Namespace) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Return axis limits for 3D plots."""
    xlim = axis_limits(coords[:, 0], args.xlim)
    ylim = axis_limits(coords[:, 1], args.ylim)
    zlim = axis_limits(coords[:, 2], args.zlim)
    return xlim, ylim, zlim


def set_3d_box_aspect(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], zlim: Tuple[float, float]) -> None:
    """Set a physically meaningful 3D box aspect when supported by matplotlib."""
    if not hasattr(ax, "set_box_aspect"):
        return
    dx = max(xlim[1] - xlim[0], 1.0)
    dy = max(ylim[1] - ylim[0], 1.0)
    dz = max(zlim[1] - zlim[0], 1.0)
    ax.set_box_aspect((dx, dy, dz))


# ------------------------- Plotting -------------------------


def plot_state_2d(
    plot_state: PlotState,
    coords: np.ndarray,
    hole_position: np.ndarray,
    args: argparse.Namespace,
) -> plt.Figure:
    """Create one 2D projected density plot."""
    prob = normalize_probability(plot_state.prob, args.normalize)
    inferred_bin_size = None
    if args.projection == "bin" and args.projection_bin_size is None:
        inferred_bin_size = infer_projection_bin_size(
            project_points_to_plane(coords, args.plane)
        )
    x, y, projected_prob = project_density_to_plane(
        coords,
        prob,
        args.plane,
        args.projection,
        args.projection_bin_size,
        args.projection_reducer,
    )
    hole_xy = project_points_to_plane(
        hole_position.reshape(1, 3),
        args.plane,
    )

    if args.verbose:
        bin_size = args.projection_bin_size
        if bin_size is None:
            bin_size = inferred_bin_size
        bin_size_text = "n/a" if bin_size is None else f"{bin_size:.10g}"
        print(f"Projection mode: {args.projection}")
        print(f"Projection bin size: {bin_size_text}")
        print(f"Number of original 3D points: {coords.shape[0]}")
        print(f"Number of projected 2D points: {projected_prob.shape[0]}")

    norm = color_norm(projected_prob, args)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    sc = ax.scatter(
        x,
        y,
        c=projected_prob,
        s=marker_sizes(projected_prob, args),
        cmap=args.cmap,
        norm=norm,
        edgecolors="none",
    )
    ax.scatter(hole_xy[0, 0], hole_xy[0, 1], c="red", s=args.hole_marker_size, label="Hole")
    ax.legend(loc="upper right")

    xlabel, ylabel = plane_labels(args.plane)
    ax.set_xlabel(args.xlabel if args.xlabel is not None else xlabel)
    ax.set_ylabel(args.ylabel if args.ylabel is not None else ylabel)
    ax.set_title(latex_escape(plot_state.title))

    if args.square:
        ax.set_aspect("equal", adjustable="box")
        lim = max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0)
        xlim = (-lim, lim)
        ylim = (-lim, lim)
    else:
        # 1. Horizontal Axis (In-plane periodic dimension): Keep tight to the grid edges
        x_min, x_max = np.min(x), np.max(x)
        if x_min == x_max:  # Avoid zero-width dimensions
            x_min, x_max = x_min - 1.0, x_max + 1.0
        xlim = (x_min, x_max)
        
        # 2. Vertical Axis (Out-of-plane dimension): Apply a 10% padding margin
        y_min, y_max = np.min(y), np.max(y)
        y_dist = y_max - y_min
        if y_dist == 0.0:
            y_dist = 1.0
            
        # Pad symmetrically outwards by 10% of the span
        ylim = (y_min - 0.1 * y_dist, y_max + 0.1 * y_dist)

    if args.xlim is not None:
        xlim = (-abs(args.xlim), abs(args.xlim))
    if args.ylim is not None:
        ylim = (-abs(args.ylim), abs(args.ylim))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    add_matplotlib_colorbar(fig, ax, sc, args, pad=0.04)
    return fig


def plot_state_3d(
    plot_state: PlotState,
    coords: np.ndarray,
    hole_position: np.ndarray,
    args: argparse.Namespace,
) -> plt.Figure:
    """Create one 3D density scatter plot."""
    prob = normalize_probability(plot_state.prob, args.normalize)

    mask = np.ones(prob.shape, dtype=bool)
    if args.threshold > 0.0:
        max_value = np.max(prob)
        if max_value > 0.0:
            mask = prob >= args.threshold * max_value

    x = coords[mask, 0]
    y = coords[mask, 1]
    z = coords[mask, 2]
    values = prob[mask]

    norm = color_norm(values, args)

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        z,
        c=values,
        s=marker_sizes(values, args),
        cmap=args.cmap,
        norm=norm,
        depthshade=not args.no_depthshade,
        alpha=args.alpha,
        edgecolors="none",
    )
    ax.scatter(
        [hole_position[0]],
        [hole_position[1]],
        [hole_position[2]],
        c="red",
        s=args.hole_marker_size,
        label="Hole",
        depthshade=False,
    )
    ax.legend(loc="upper right")

    ax.set_xlabel(args.xlabel if args.xlabel is not None else r"$x$ (\AA)")
    ax.set_ylabel(args.ylabel if args.ylabel is not None else r"$y$ (\AA)")
    ax.set_zlabel(args.zlabel if args.zlabel is not None else r"$z$ (\AA)")
    ax.set_title(latex_escape(plot_state.title))

    xlim = axis_limits(x, args.xlim)
    ylim = axis_limits(y, args.ylim)
    zlim = axis_limits(z, args.zlim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    if args.equal_3d_aspect:
        set_3d_box_aspect(ax, xlim, ylim, zlim)

    if args.view is not None:
        elev, azim = args.view
        ax.view_init(elev=elev, azim=azim)

    add_matplotlib_colorbar(fig, ax, sc, args, pad=0.08)
    return fig


def plot_state_3d_plotly(
    plot_state: PlotState,
    coords: np.ndarray,
    hole_position: np.ndarray,
    args: argparse.Namespace,
):
    """Create one interactive Plotly 3D density scatter plot."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly output requires plotly. Install it with: pip install plotly"
        ) from exc

    prob = normalize_probability(plot_state.prob, args.normalize)

    mask = np.ones(prob.shape, dtype=bool)
    if args.threshold > 0.0:
        max_value = np.max(prob)
        if max_value > 0.0:
            mask = prob >= args.threshold * max_value

    plot_coords = coords[mask]
    values = prob[mask]
    if args.max_points is not None and values.size > args.max_points:
        keep = np.argsort(values)[-args.max_points:]
        plot_coords = plot_coords[keep]
        values = values[keep]

    color_values, cmin, cmax = plotly_color_values(values, args)
    sizes = marker_sizes(values, args)
    colorscale = plotly_colorscale_name(args.cmap)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=plot_coords[:, 0],
            y=plot_coords[:, 1],
            z=plot_coords[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color_values,
                colorscale=colorscale,
                opacity=args.alpha,
                line=dict(width=0.1, color="black"),
                colorbar=dict(title=args.cbar_label, nticks=args.cbar_ticks),
                cmin=cmin,
                cmax=cmax,
            ),
            customdata=values,
            hovertemplate=(
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                "P=%{customdata:.6e}<extra></extra>"
            ),
            name="Electron density",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[hole_position[0]],
            y=[hole_position[1]],
            z=[hole_position[2]],
            mode="markers",
            marker=dict(size=args.hole_marker_size_plotly, color="red"),
            name="Hole",
        )
    )

    fig.update_layout(
        title=plot_state.title,
        scene=dict(
            xaxis_title=args.xlabel if args.xlabel is not None else "x (Angstrom)",
            yaxis_title=args.ylabel if args.ylabel is not None else "y (Angstrom)",
            zaxis_title=args.zlabel if args.zlabel is not None else "z (Angstrom)",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def save_plotly_html(fig, path: str) -> None:
    """Save a Plotly figure as standalone HTML using CDN Plotly JS."""
    fig.write_html(path, include_plotlyjs="cdn")


def save_pdf(figs: Sequence[plt.Figure], path: str) -> None:
    """Save figures as a multi-page PDF."""
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)


def save_pngs(figs: Sequence[plt.Figure], prefix: str, dpi: int) -> None:
    """Save one PNG file per figure."""
    for index, fig in enumerate(figs, start=1):
        fig.savefig(f"{prefix}_{index}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ------------------------- CLI -------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rswf.py",
        description="Plot real-space exciton densities from rswf files.",
    )
    parser.add_argument("rswf_file", help="Real-space wavefunction file.")

    parser.add_argument(
        "-n",
        "--n-states",
        type=int,
        help="Number of states to plot from the beginning.",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Explicit 1-based state list/ranges, e.g. '1,2,5-7'. Default: 1.",
    )
    parser.add_argument(
        "--sum-selected",
        action="store_true",
        help="Sum all explicitly selected states into one plotted density.",
    )

    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="3d",
        help="Plot mode. 2d projects the density onto a plane. Default: 3d.",
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Plotting backend. Default: matplotlib.",
    )
    parser.add_argument(
        "--plane",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Projection plane used only for --mode 2d. Default: xy.",
    )
    parser.add_argument(
        "--projection",
        choices=["bin", "exact"],
        default="bin",
        help="2D projection method. 'exact' preserves old exact-coordinate grouping. Default: bin.",
    )
    parser.add_argument(
        "--projection-bin-size",
        type=float,
        default=None,
        help="2D binned projection bin size in Angstrom. Default: infer from coordinate spacing.",
    )
    parser.add_argument(
        "--projection-reducer",
        choices=["sum", "max"],
        default="sum",
        help="Reducer for points inside each 2D projection bin. Default: sum.",
    )

    parser.add_argument(
        "--eigvals",
        type=str,
        default=None,
        help="Optional eigenvalue file. Non-numeric/header lines are skipped.",
    )
    parser.add_argument(
        "--sum-degenerate",
        action="store_true",
        help="Sum adjacent states whose eigenvalues differ by --deg-thresh.",
    )
    parser.add_argument(
        "--deg-thresh",
        type=float,
        default=1e-3,
        help="Eigenvalue threshold for --sum-degenerate. Default: 1e-3.",
    )
    parser.add_argument(
        "--degeneracies",
        type=str,
        default=None,
        help="Manual degeneracy groups to sum, e.g. '2,1,2'. Applied after selection.",
    )

    parser.add_argument(
        "--no-hole-row",
        action="store_true",
        help="Do not treat the first row of each state as the fixed hole position.",
    )
    parser.add_argument(
        "--absolute-coords",
        action="store_true",
        help="Plot absolute coordinates instead of coordinates relative to the hole.",
    )

    parser.add_argument(
        "--normalize",
        choices=["max", "sum", "none"],
        default="max",
        help="Normalize each plotted density. Default: max.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="For 3D, hide points below threshold times max density. Default: 0.",
    )

    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap.")
    parser.add_argument(
        "--color-scale",
        choices=["linear", "power", "log"],
        default="power",
        help="Color normalization. 'power' improves weak-density contrast. Default: power.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.55,
        help="Gamma for --color-scale power. Values below 1 boost weak densities.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Explicit color minimum. Default: 0 for linear/power, percentile for log.",
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=1.0,
        help="Positive-value percentile used as log-scale vmin. Default: 1.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Color maximum percentile. Reduces domination by isolated peaks. Default: 99.5.",
    )

    parser.add_argument("--marker-size", type=float, default=50.0, help="Maximum marker size.")
    parser.add_argument(
        "--hole-marker-size",
        type=float,
        default=80.0,
        help="Hole marker size.",
    )
    parser.add_argument(
        "--hole-marker-size-plotly",
        type=float,
        default=7.0,
        help="Hole marker size for Plotly.",
    )
    parser.add_argument(
        "--no-scale-markers",
        dest="scale_markers",
        action="store_false",
        help="Disable marker-size scaling with probability.",
    )
    parser.add_argument(
        "--size-gamma",
        type=float,
        default=0.7,
        help="Gamma for marker-size scaling. Values below 1 boost weak densities.",
    )
    parser.add_argument(
        "--min-marker-fraction",
        type=float,
        default=0.08,
        help="Minimum marker size as a fraction of --marker-size. Default: 0.08.",
    )
    parser.set_defaults(scale_markers=True)

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Marker opacity for 3D plots. Default: 0.85.",
    )
    parser.add_argument(
        "--no-depthshade",
        action="store_true",
        help="Disable matplotlib 3D depth shading.",
    )
    parser.add_argument(
        "--equal-3d-aspect",
        action="store_true",
        help="Use physical x:y:z box aspect in 3D plots.",
    )

    parser.add_argument(
        "--format",
        choices=["pdf", "png", "html"],
        default="pdf",
        help="Output format. Matplotlib supports pdf/png; Plotly supports html.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output base path. Default: input basename + '_rswf'.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG DPI. Default: 600.")

    parser.add_argument("--font-size", type=int, default=15, help="Base font size.")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX rendering.")
    parser.add_argument("--xlabel", type=str, default=None, help="Override x-axis label.")
    parser.add_argument("--ylabel", type=str, default=None, help="Override y-axis label.")
    parser.add_argument("--zlabel", type=str, default=None, help="Override z-axis label.")
    parser.add_argument("--cbar-label", type=str, default="Probability", help="Colorbar label.")
    parser.add_argument(
        "--cbar-ticks",
        type=int,
        default=6,
        help="Approximate number of colorbar ticks. Default: 6.",
    )
    parser.add_argument(
        "--keep-cbar-top-tick",
        action="store_true",
        help="Keep the upper colorbar tick instead of pruning it.",
    )

    parser.add_argument("--xlim", type=float, default=None, help="Override symmetric x limit.")
    parser.add_argument("--ylim", type=float, default=None, help="Override symmetric y limit.")
    parser.add_argument("--zlim", type=float, default=None, help="Override symmetric z limit.")
    parser.add_argument("--no-square", dest="square", action="store_false", help="Disable square 2D plots.")
    parser.set_defaults(square=True)
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="For Plotly, keep only the N largest-probability points after thresholding.",
    )
    parser.add_argument(
        "--view",
        type=float,
        nargs=2,
        metavar=("ELEV", "AZIM"),
        default=None,
        help="3D camera view angles, e.g. --view 25 45.",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Print diagnostics.")
    parser.add_argument("--version", action="version", version="rswf.py 2.1")
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """Return an error string if the arguments are inconsistent."""
    if args.cbar_ticks < 2:
        return "--cbar-ticks must be at least 2."
    if args.max_points is not None and args.max_points <= 0:
        return "--max-points must be positive."
    if args.projection_bin_size is not None and args.projection_bin_size <= 0.0:
        return "--projection-bin-size must be positive."
    if args.backend == "plotly" and args.mode != "3d":
        return "--backend plotly requires --mode 3d."
    if args.backend == "plotly" and args.format != "html":
        return "--backend plotly requires --format html."
    if args.backend == "matplotlib" and args.format == "html":
        return "--backend matplotlib cannot use --format html."
    if args.threshold < 0.0:
        return "--threshold must be non-negative."
    if args.gamma <= 0.0:
        return "--gamma must be positive."
    if args.size_gamma <= 0.0:
        return "--size-gamma must be positive."
    if not (0.0 <= args.min_marker_fraction <= 1.0):
        return "--min-marker-fraction must be between 0 and 1."
    if not (0.0 < args.vmax_percentile <= 100.0):
        return "--vmax-percentile must be in (0, 100]."
    if not (0.0 < args.vmin_percentile <= 100.0):
        return "--vmin-percentile must be in (0, 100]."
    if args.vmin_percentile > args.vmax_percentile:
        return "--vmin-percentile cannot exceed --vmax-percentile."
    if not (0.0 < args.alpha <= 1.0):
        return "--alpha must be in (0, 1]."
    return None


def summarize_grid(coords: np.ndarray, states: Sequence[PlotState]) -> None:
    """Print simple diagnostics about the real-space grid and densities."""
    unique_z = np.unique(np.round(coords[:, 2], decimals=10))
    print(f"Distinct z values in plotted grid: {len(unique_z)}")
    if len(unique_z) <= 12:
        print("z values:", ", ".join(f"{value:g}" for value in unique_z))
    for item in states:
        prob = item.prob
        print(
            f"{item.title}: min={np.min(prob):.6g}, max={np.max(prob):.6g}, "
            f"sum={np.sum(prob):.6g}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    error = validate_args(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 2

    configure_matplotlib(fontsize=args.font_size, latex=not args.no_latex)

    data = read_rswf_file(args.rswf_file, hole_row=not args.no_hole_row)
    total_states = len(data.states)

    if args.select is not None:
        selection_text = args.select
    elif args.n_states is not None:
        if args.n_states > total_states:
            print(
                f"Requested {args.n_states} states, but file contains only {total_states}.",
                file=sys.stderr,
            )
            return 2
        if args.n_states <= 0:
            print("Please provide a positive --n-states.", file=sys.stderr)
            return 2
        raw_indices = list(range(args.n_states))
        selection_text = None
    else:
        selection_text = "1"

    if selection_text is not None:
        raw_indices = parse_selection(selection_text, total_states)
        if not raw_indices:
            print("No valid state indices selected.", file=sys.stderr)
            return 2

    selected_states = [data.states[index] for index in raw_indices]
    plot_states = make_plot_states(selected_states, raw_indices)

    if args.sum_selected:
        plot_states = combine_selected_states(plot_states)

    if args.degeneracies is not None and args.sum_degenerate:
        print("Use either --degeneracies or --sum-degenerate, not both.", file=sys.stderr)
        return 2

    # Load eigenvalues early if requested so both manual and auto modes can use them
    selected_eigvals = None
    if args.eigvals is not None:
        eigvals_all = read_eigenvalues(args.eigvals, max(raw_indices) + 1)
        selected_eigvals = eigvals_all[raw_indices]

    if args.degeneracies is not None:
        degeneracies = parse_degeneracies(args.degeneracies)
        # Pass the eigenvalues here so they get printed on the manually combined titles!
        plot_states = combine_with_manual_degeneracies(plot_states, degeneracies, eigvals=selected_eigvals)

    elif selected_eigvals is not None:
        if args.sum_degenerate:
            plot_states = combine_with_eigenvalue_threshold(
                plot_states,
                selected_eigvals,
                threshold=args.deg_thresh,
            )
        else:
            plot_states = attach_eigenvalue_titles(plot_states, selected_eigvals)

    if args.eigvals is not None:
        # Match kwf.py's logic: read up to the max index requested from selection array
        eigvals_all = read_eigenvalues(args.eigvals, max(raw_indices) + 1)
        selected_eigvals = eigvals_all[raw_indices]
        if args.sum_degenerate:
            plot_states = combine_with_eigenvalue_threshold(
                plot_states,
                selected_eigvals,
                threshold=args.deg_thresh,
            )
        elif args.degeneracies is None:
            plot_states = attach_eigenvalue_titles(plot_states, selected_eigvals)

    coords, hole_position = centered_coordinates(
        data.coords,
        data.hole_position,
        absolute_coords=args.absolute_coords,
    )

    base_name = os.path.splitext(os.path.basename(args.rswf_file))[0]
    outbase = args.out if args.out is not None else f"{base_name}_rswf"

    if args.backend == "plotly":
        output_paths: List[str] = []
        for index, plot_state in enumerate(plot_states, start=1):
            fig = plot_state_3d_plotly(plot_state, coords, hole_position, args)
            if len(plot_states) == 1:
                output_path = f"{outbase}.html"
            else:
                label = "_".join(str(raw_index + 1) for raw_index in plot_state.raw_indices)
                output_path = f"{outbase}_{label if label else index}.html"
            save_plotly_html(fig, output_path)
            output_paths.append(output_path)
        output_path = ", ".join(output_paths)
    else:
        figs: List[plt.Figure] = []
        for plot_state in plot_states:
            if args.mode == "2d":
                fig = plot_state_2d(plot_state, coords, hole_position, args)
            else:
                fig = plot_state_3d(plot_state, coords, hole_position, args)
            figs.append(fig)

        if args.format == "pdf":
            output_path = f"{outbase}.pdf"
            save_pdf(figs, output_path)
        else:
            output_path = f"{outbase}_<state>.png"
            save_pngs(figs, outbase, dpi=args.dpi)

    if args.verbose:
        print(f"Input states: {total_states}")
        print(f"Selected raw states: {[idx + 1 for idx in raw_indices]}")
        print(f"Plotted states: {len(plot_states)}")
        print(f"Grid points per state: {data.coords.shape[0]}")
        print(f"Hole position: {data.hole_position.tolist()}")
        print(f"Mode: {args.mode}")
        if args.mode == "2d":
            print(f"Plane: {args.plane}")
        print(f"Output: {output_path}")
        summarize_grid(coords, plot_states)

    return 0


if __name__ == "__main__":
    sys.exit(main())
