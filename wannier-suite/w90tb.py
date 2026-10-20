"""
Reader/writer for Wannier90 seedname_tb.dat files, including the optional
overlap section appended by crystal2wannier (non-orthogonal basis).

File layout (blocks separated by blank lines):
    header, a1, a2, a3, mSize, nFock, degeneracies (15 per line)
    nFock blocks:  R1 R2 R3, then mSize^2 lines "i j Re Im"            -> H(R)
    optional, nFock blocks: R1 R2 R3, then mSize^2 lines "i j Rx Ix Ry Iy Rz Iz" -> r(R)
    optional (needs r), nFock blocks like H                                       -> S(R)

All matrices are stored raw (not divided by the degeneracy), as in the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TbData:
    header: str
    bravais: np.ndarray            # (3, 3)
    degen: np.ndarray              # (nFock,)
    i_rn: np.ndarray               # (nFock, 3)
    h_r: np.ndarray                # (nFock, m, m)
    wf: np.ndarray | None          # (nFock, m, m, 3) position matrix elements, or None if absent
    overlap: np.ndarray | None     # (nFock, m, m) or None if orthogonal basis

    @property
    def m_size(self) -> int:
        return self.h_r.shape[1]

    @property
    def n_fock(self) -> int:
        return self.h_r.shape[0]

    @property
    def diag_index(self) -> int | None:
        """Index of the R=(0,0,0) block, if present."""
        m = np.where(np.all(self.i_rn == 0, axis=1))[0]
        return int(m[0]) if len(m) else None

    def normalized(self) -> "TbData":
        """Copy with every matrix divided by its degeneracy (degen set to 1), as XATU does on load."""
        w = self.degen[:, None, None].astype(float)
        return TbData(
            self.header, self.bravais, np.ones_like(self.degen), self.i_rn,
            self.h_r / w,
            None if self.wf is None else self.wf / w[..., None],
            None if self.overlap is None else self.overlap / w,
        )


def _block(lines: list[str], start: int, m_size: int, ncols: int) -> np.ndarray:
    """Parse the m^2 data lines after a block's R header into an (m, m, ncols-2) array."""
    stop = start + 1 + m_size * m_size
    if stop > len(lines):
        raise ValueError("Unexpected end of file inside a matrix block.")
    data = np.fromstring("\n".join(lines[start + 1 : stop]), sep=" ")
    if data.size != m_size * m_size * ncols:
        raise ValueError(f"Malformed matrix block starting at data line {start}.")
    data = data.reshape(-1, ncols)
    i = data[:, 0].astype(int) - 1
    j = data[:, 1].astype(int) - 1
    if i.min() < 0 or j.min() < 0 or i.max() >= m_size or j.max() >= m_size:
        raise ValueError(f"Orbital index out of range in block at data line {start}.")
    out = np.zeros((m_size, m_size, (ncols - 2) // 2), dtype=complex)
    out[i, j] = data[:, 2::2] + 1j * data[:, 3::2]
    return out


def _read_section(lines, pos, n_fock, m_size, ncols, i_rn, name):
    """Read n_fock blocks, checking R ordering against i_rn. Returns (array, new pos)."""
    out = np.zeros((n_fock, m_size, m_size, (ncols - 2) // 2), dtype=complex)
    for n in range(n_fock):
        if pos >= len(lines):
            raise ValueError(f"Incomplete {name} section: block {n} missing.")
        r = [int(x) for x in lines[pos].split()[:3]]
        if r != list(i_rn[n]):
            raise ValueError(f"{name} block {n}: R={r} does not match Hamiltonian R={list(i_rn[n])}.")
        out[n] = _block(lines, pos, m_size, ncols)
        pos += 1 + m_size * m_size
    return out, pos


def read_tb(filename: str | Path) -> TbData:
    with open(filename, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        lines = [ln.strip() for ln in fh if ln.strip()]

    bravais = np.array([[float(x) for x in lines[k].split()[:3]] for k in range(3)])
    m_size = int(lines[3].split()[0])
    n_fock = int(lines[4].split()[0])

    pos = 5
    degen: list[int] = []
    while len(degen) < n_fock:
        degen.extend(int(x) for x in lines[pos].split())
        pos += 1
    degen_arr = np.array(degen[:n_fock], dtype=int)

    # H block R headers define the reference ordering for the other sections
    i_rn = np.zeros((n_fock, 3), dtype=int)
    stride = 1 + m_size * m_size
    for n in range(n_fock):
        i_rn[n] = [int(x) for x in lines[pos + n * stride].split()[:3]]

    h_r, pos = _read_section(lines, pos, n_fock, m_size, 4, i_rn, "Hamiltonian")
    wf = overlap = None
    if pos < len(lines):
        wf, pos = _read_section(lines, pos, n_fock, m_size, 8, i_rn, "Position")
    if pos < len(lines):
        overlap, pos = _read_section(lines, pos, n_fock, m_size, 4, i_rn, "Overlap")
        if pos < len(lines):
            raise ValueError("Unexpected content after the overlap section.")

    return TbData(header, bravais, degen_arr, i_rn, h_r[..., 0], wf, overlap[..., 0] if overlap is not None else None)


def _write_section(fh, i_rn, arr) -> None:
    """arr: (nFock, m, m, ncomp) complex."""
    m = arr.shape[1]
    ii, jj = np.meshgrid(np.arange(1, m + 1), np.arange(1, m + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    for n in range(arr.shape[0]):
        fh.write(f"{i_rn[n,0]:5d} {i_rn[n,1]:5d} {i_rn[n,2]:5d}\n")
        flat = arr[n].reshape(m * m, -1)
        cols = np.empty((m * m, 2 * flat.shape[1]))
        cols[:, 0::2], cols[:, 1::2] = flat.real, flat.imag
        fmt = "%5d %5d" + " %22.14e" * cols.shape[1]
        fh.write("\n".join(fmt % ((a, b) + tuple(row)) for a, b, row in zip(ii, jj, cols)))
        fh.write("\n\n")


def write_tb(filename: str | Path, tb: TbData) -> None:
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(tb.header + "\n")
        for v in tb.bravais:
            fh.write(f"{v[0]:22.14f} {v[1]:22.14f} {v[2]:22.14f}\n")
        fh.write(f"{tb.m_size:d}\n{tb.n_fock:d}\n")
        for s in range(0, tb.n_fock, 15):
            fh.write(" ".join(str(d) for d in tb.degen[s : s + 15]) + "\n")
        fh.write("\n")
        _write_section(fh, tb.i_rn, tb.h_r[..., None])
        if tb.wf is None and tb.overlap is not None:
            raise ValueError("Cannot write an overlap section without a position-matrix section.")
        if tb.wf is not None:
            _write_section(fh, tb.i_rn, tb.wf)
        if tb.overlap is not None:
            _write_section(fh, tb.i_rn, tb.overlap[..., None])
