"""Helpers shared by the xatu-utilities scripts."""
from __future__ import annotations

import shutil
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib(fontsize: int = 15, latex: bool = True) -> None:
    """Set font sizes. LaTeX text is used only if requested and installed."""
    use_latex = latex and shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif" if use_latex else "sans-serif",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "font.size": fontsize,
    })


def read_eigval_file(path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """Read a XATU .eigval file: ncell, basis dimension and state count, then one energy per line."""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.split() for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a XATU .eigval file")
    header = {"ncell": int(lines[0][0]), "basisdim": int(lines[1][0]), "nstates": int(lines[2][0])}
    return header, np.array([float(parts[0]) for parts in lines[3:]], dtype=float)


def read_eigenvalues(path: str, n: int) -> np.ndarray:
    """First n exciton energies from a XATU .eigval file."""
    _, values = read_eigval_file(path)
    if len(values) < n:
        raise ValueError(f"Expected {n} eigenvalues but found {len(values)} in {path}")
    return values[:n]


def parse_selection(select_str: str, max_n: int) -> List[int]:
    """Parse a 1-based selection like '1,2,5-7' into sorted 0-based indices within [0, max_n)."""
    selected: List[int] = []
    for item in select_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = sorted((int(start_text), int(end_text)))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))
    return sorted({idx - 1 for idx in selected if 1 <= idx <= max_n})
