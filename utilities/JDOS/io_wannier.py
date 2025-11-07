# io_wannier.py
"""
I/O utilities for Wannier tight-binding (_tb.dat) files.

This module parses the Hamiltonian and position-like matrices
written by Wannier90 (or similar custom generators) in the
_tb.dat format used by Guilherme's codes.

It provides:
    - parse_tb_file(path) -> WannierTB object

usage:
    from io_wannier import parse_tb_file
    tb = parse_tb_file('my_system_tb.dat', verbose=True)

"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WannierTB:
    """Container for Wannier tight-binding data."""
    bravais_vectors: np.ndarray      # (3,3)
    nFock: int                       # number of real-space hoppings (R vectors)
    mSize: int                       # number of Wannier orbitals
    iRn: np.ndarray                  # (nFock,3) integer lattice indices
    H: np.ndarray                    # (nFock,mSize,mSize) complex hoppings
    WF: np.ndarray                   # (nFock,mSize,mSize,3) position-like matrices (x,y,z)
    diag_index: int                  # index of the on-site (R=0) term
    degen: np.ndarray | None = None  # optional degeneracy info


def parse_tb_file(filename: str | Path, verbose: bool = False) -> WannierTB:
    """
    Parse a Wannier *_tb.dat file containing hopping matrices and
    position matrix elements.

    Parameters
    ----------
    filename : str or Path
        Path to the _tb.dat file.
    verbose : bool
        If True, prints progress information.

    Returns
    -------
    WannierTB
        A dataclass containing all parsed arrays.
    """
    filename = Path(filename)
    if verbose:
        print(f"Reading {filename} ...")

    with open(filename, 'r') as file:
        # --- Header and lattice vectors ---
        next(file)  # skip header line
        bravais_vectors = np.array(
            [[float(x) for x in next(file).split()] for _ in range(3)],
            dtype=float
        )

        mSize = int(next(file).strip())
        nFock = int(next(file).strip())

        # --- Degeneracy list ---
        degen = np.zeros(nFock, dtype=int)
        per_line = 15
        full_lines, rem = divmod(nFock, per_line)

        for i in range(full_lines):
            vals = list(map(int, next(file).split()))
            degen[i * per_line:(i + 1) * per_line] = vals

        if rem > 0:
            vals = list(map(int, next(file).split()))
            degen[full_lines * per_line:full_lines * per_line + rem] = vals[:rem]

        next(file)  # blank line

        # --- First set of Fock matrices (Hamiltonians) ---
        H = np.zeros((nFock, mSize, mSize), dtype=np.complex128)
        iRn = np.zeros((nFock, 3), dtype=int)
        diag_index = None

        for n in range(nFock):
            iRn[n] = [int(x) for x in next(file).split()]
            if np.all(iRn[n] == 0):
                diag_index = n

            for _ in range(mSize * mSize):
                parts = next(file).split()
                if not parts:
                    continue
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                real, imag = float(parts[2]), float(parts[3])
                H[n, i, j] = real + 1j * imag

            next(file)  # blank line between matrices

        # --- Second set: position-like matrices (WF) ---
        WF = np.zeros((nFock, mSize, mSize, 3), dtype=np.complex128)
        for n in range(nFock):
            next(file)  # R indices (ignored, same order)
            for _ in range(mSize * mSize):
                parts = next(file).split()
                if not parts:
                    continue
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                if i != j:
                    continue  # keep only diagonal as per your parser
                WF[n, i, j, 0] = float(parts[2]) + 1j * float(parts[3])
                WF[n, i, j, 1] = float(parts[4]) + 1j * float(parts[5])
                WF[n, i, j, 2] = float(parts[6]) + 1j * float(parts[7])
            if n + 1 != nFock:
                next(file)  # blank line between matrices

    if diag_index is None:
        raise ValueError("Could not find the on-site (R=0) Hamiltonian block.")

    if verbose:
        print(f"Parsed {nFock} hopping matrices with size {mSize}x{mSize}")

    return WannierTB(
        bravais_vectors=bravais_vectors,
        nFock=nFock,
        mSize=mSize,
        iRn=iRn,
        H=H,
        WF=WF,
        diag_index=diag_index,
        degen=degen
    )
