#!/usr/bin/env python3
"""
Print a quick text summary of a XATU exciton run.

Reads LABEL.eigval and, when present next to it, LABEL.spin and LABEL.layers.
For each state it prints the energy, the shift from the lowest state,
the degeneracy group, the binding energy (with --gap), the spin, and the
intralayer weight with the dominant electron/hole layer pair.

Examples:
  python xatu_summary.py MoS2
  python xatu_summary.py runs/MoS2.eigval -n 10 --gap 2.48
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from xatu_common import read_eigval_file


def read_spin_file(path: str) -> np.ndarray:
    """Rows of (St, Se, Sh) per state from a XATU .spin file."""
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return data[:, 1:4]


def read_layers_file(path: str) -> np.ndarray:
    """P(Le, Lh) per state from a XATU .layers file, shape (nstates, nlayers, nlayers)."""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.split() for line in handle if line.strip() and not line.startswith("#")]
    nlayers = None
    matrices: List[List[List[float]]] = []
    for parts in lines:
        if parts[0] == "layers":
            nlayers = int(parts[1])
        elif parts[0] == "state":
            matrices.append([])
        elif parts[0] not in ("split", "states") and matrices:
            matrices[-1].append([float(x) for x in parts])
    if nlayers is None or not matrices:
        raise ValueError(f"{path} does not look like a XATU .layers file")
    return np.array(matrices, dtype=float)


def degeneracy_groups(energies: np.ndarray, threshold: float) -> List[int]:
    groups = [1]
    for i in range(1, len(energies)):
        groups.append(groups[-1] + (abs(energies[i] - energies[i - 1]) > threshold))
    return groups


def label_from_path(path: str) -> str:
    return path[: -len(".eigval")] if path.endswith(".eigval") else path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Text summary of a XATU exciton run.")
    p.add_argument("label", help="Run label or path to LABEL.eigval.")
    p.add_argument("-n", "--n-states", type=int, default=20, help="Number of states to list. Default: 20.")
    p.add_argument("--gap", type=float, default=None, help="Quasiparticle gap in eV, to print binding energies.")
    p.add_argument("--deg-thresh", type=float, default=1e-3,
                   help="Energy difference in eV below which states count as degenerate. Default: 1e-3.")
    args = p.parse_args(argv)

    label = label_from_path(args.label)
    eigval_path = label + ".eigval"
    if not os.path.exists(eigval_path):
        print(f"No such file: {eigval_path}", file=sys.stderr)
        return 2

    header, energies = read_eigval_file(eigval_path)
    n = min(args.n_states, len(energies))
    energies = energies[:n]
    groups = degeneracy_groups(energies, args.deg_thresh)

    spin = read_spin_file(label + ".spin")[:n] if os.path.exists(label + ".spin") else None
    layers = read_layers_file(label + ".layers")[:n] if os.path.exists(label + ".layers") else None

    print(f"run: {label}")
    print(f"ncell = {header['ncell']}, BSE basis dimension = {header['basisdim']}, "
          f"states written = {header['nstates']}")
    if args.gap is not None:
        print(f"gap = {args.gap:.4f} eV, binding energy of state 1 = {1000 * (args.gap - energies[0]):.1f} meV")
    print(f"distinct levels in the first {n} states (threshold {args.deg_thresh:g} eV): {groups[-1]}")
    print()

    columns = ["n", "E (eV)", "E-E1 (meV)", "deg"]
    if args.gap is not None:
        columns.append("Eb (meV)")
    if spin is not None:
        columns += ["St", "Se", "Sh"]
    if layers is not None:
        columns += ["intra", "main (Le,Lh)"]
    print("  ".join(f"{c:>12}" for c in columns))

    for i in range(n):
        row = [f"{i + 1}", f"{energies[i]:.6f}", f"{1000 * (energies[i] - energies[0]):.2f}", f"{groups[i]}"]
        if args.gap is not None:
            row.append(f"{1000 * (args.gap - energies[i]):.1f}")
        if spin is not None and i < len(spin):
            row += [f"{value:.3f}" for value in spin[i]]
        elif spin is not None:
            row += ["-"] * 3
        if layers is not None and i < len(layers):
            P = layers[i]
            Le, Lh = np.unravel_index(np.argmax(P), P.shape)
            row += [f"{np.trace(P) / np.sum(P):.3f}", f"({Le},{Lh}) {P[Le, Lh]:.3f}"]
        elif layers is not None:
            row += ["-"] * 2
        print("  ".join(f"{c:>12}" for c in row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
