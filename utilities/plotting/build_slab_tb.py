#!/usr/bin/env python3
"""
Build a finite-slab tight-binding Hamiltonian from a Wannier90 seedname_tb.dat file.

First implementation: slab finite along the third lattice direction, periodic along
first and second lattice directions. The layer index is folded into the orbital
index, so the output has no hopping along the finite direction.

The script reads the same basic layout used by Wannier90 seedname_tb.dat:
    header
    3 Bravais vectors
    number of Wannier orbitals
    number of real-space hopping blocks
    degeneracy list
    H(R) blocks
    position matrix blocks

Conventions:
    H[block, m, n] is copied without transposition.
    Indices in memory are zero-based.
    Indices written to disk are one-based.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


Translation3D = Tuple[int, int, int]
Translation2D = Tuple[int, int]
SlabHoppings = Dict[Translation2D, np.ndarray]


@dataclass(frozen=True)
class TBData:
    bravais: np.ndarray
    n_fock: int
    m_size: int
    degen: np.ndarray
    i_rn: np.ndarray
    hamiltonian: np.ndarray
    position: Optional[np.ndarray]
    gamma_index: int


@dataclass(frozen=True)
class SlabData:
    bravais: np.ndarray
    m_size: int
    i_rn: np.ndarray
    hamiltonian: np.ndarray
    degen: np.ndarray
    centers: Optional[np.ndarray]


def _read_nonempty_line(handle) -> str:
    """Read the next non-empty line."""
    for line in handle:
        stripped = line.strip()
        if stripped:
            return stripped
    raise EOFError("Unexpected end of file while reading seedname_tb.dat")


def _read_degeneracies(handle, n_fock: int) -> np.ndarray:
    """Read the Wannier90 degeneracy list, usually stored with 15 integers per line."""
    values: List[int] = []
    while len(values) < n_fock:
        line = _read_nonempty_line(handle)
        values.extend(int(x) for x in line.split())
    return np.asarray(values[:n_fock], dtype=int)


def parse_tb_dat(filename: str | Path) -> TBData:
    """
    Parse a Wannier90 seedname_tb.dat file.

    Returns the Bravais vectors, hopping translations, hopping matrices, optional
    position matrices, and the index of the (0, 0, 0) hopping block.
    """
    path = Path(filename)
    with path.open("r", encoding="utf-8") as handle:
        _ = next(handle)  # Header line.

        bravais = np.zeros((3, 3), dtype=float)
        for row in range(3):
            bravais[row] = np.asarray(_read_nonempty_line(handle).split(), dtype=float)

        m_size = int(_read_nonempty_line(handle))
        n_fock = int(_read_nonempty_line(handle))
        degen = _read_degeneracies(handle, n_fock)

        i_rn = np.zeros((n_fock, 3), dtype=int)
        hamiltonian = np.zeros((n_fock, m_size, m_size), dtype=complex)
        gamma_index: Optional[int] = None

        for block in range(n_fock):
            i_rn[block] = np.asarray(_read_nonempty_line(handle).split(), dtype=int)
            if np.array_equal(i_rn[block], np.array([0, 0, 0], dtype=int)):
                gamma_index = block

            entries_read = 0
            while entries_read < m_size * m_size:
                line = _read_nonempty_line(handle)
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed Hamiltonian line: {line}")
                i_index = int(float(parts[0])) - 1
                j_index = int(float(parts[1])) - 1
                real_part = float(parts[2])
                imag_part = float(parts[3])
                hamiltonian[block, i_index, j_index] = real_part + 1j * imag_part
                entries_read += 1

        if gamma_index is None:
            raise ValueError("Could not find the (0, 0, 0) hopping block.")

        position = _try_read_position_blocks(handle, n_fock, m_size)

    return TBData(
        bravais=bravais,
        n_fock=n_fock,
        m_size=m_size,
        degen=degen,
        i_rn=i_rn,
        hamiltonian=hamiltonian,
        position=position,
        gamma_index=gamma_index,
    )


def _try_read_position_blocks(handle, n_fock: int, m_size: int) -> Optional[np.ndarray]:
    """
    Read the second seedname_tb.dat block with position matrix elements.

    If the block is absent, return None. This makes the script usable even for
    reduced test files that contain only the Hamiltonian block.
    """
    position = np.zeros((n_fock, m_size, m_size, 3), dtype=complex)

    for block in range(n_fock):
        try:
            _ = _read_nonempty_line(handle)  # Translation label for position block.
        except EOFError:
            if block == 0:
                return None
            raise ValueError("Position block ended before all translations were read.")

        entries_read = 0
        while entries_read < m_size * m_size:
            line = _read_nonempty_line(handle)
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Malformed position line: {line}")
            i_index = int(float(parts[0])) - 1
            j_index = int(float(parts[1])) - 1
            position[block, i_index, j_index, 0] = float(parts[2]) + 1j * float(parts[3])
            position[block, i_index, j_index, 1] = float(parts[4]) + 1j * float(parts[5])
            position[block, i_index, j_index, 2] = float(parts[6]) + 1j * float(parts[7])
            entries_read += 1

    return position


def build_z_slab(i_rn: np.ndarray, hamiltonian: np.ndarray, degen: np.ndarray, m_size: int, n_layers: int) -> SlabHoppings:
    """
    Build slab hopping matrices H_slab(Rx, Ry) for a z-oriented slab.

    The finite layer index is internal to the enlarged orbital basis. Hoppings
    that would connect outside the finite slab are discarded.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")

    slab_size = m_size * n_layers
    slab_hoppings: SlabHoppings = {}

    for block, translation in enumerate(i_rn):
        rx, ry, rz = (int(value) for value in translation)
        key = (rx, ry)
        h_bulk = hamiltonian[block] / degen[block]

        for layer in range(n_layers):
            target_layer = layer + rz
            if target_layer < 0 or target_layer >= n_layers:
                continue

            if key not in slab_hoppings:
                slab_hoppings[key] = np.zeros((slab_size, slab_size), dtype=complex)

            row_start = layer * m_size
            col_start = target_layer * m_size
            slab_hoppings[key][row_start:row_start + m_size, col_start:col_start + m_size] += h_bulk

    return slab_hoppings


def build_slab_hoppings(
    bravais: np.ndarray,
    i_rn: np.ndarray,
    hamiltonian: np.ndarray,
    degen: np.ndarray,
    m_size: int,
    n_layers: int,
    direction: str = "z",
) -> Tuple[np.ndarray, SlabHoppings]:
    """
    Dispatch slab construction by finite direction.

    Only direction == "z" is implemented. The function is structured so that x
    and y slabs can later be added without changing the command-line interface.
    """
    direction = direction.lower()
    if direction != "z":
        raise NotImplementedError("Only slab_direction = 'z' is implemented in this first version.")

    bravais_slab = bravais.copy()
    bravais_slab[2] = n_layers * bravais[2]
    slab_hoppings = build_z_slab(i_rn, hamiltonian, degen, m_size, n_layers)
    return bravais_slab, slab_hoppings


def slab_dict_to_arrays(slab_hoppings: SlabHoppings) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a slab hopping dictionary into sorted translation and matrix arrays."""
    if not slab_hoppings:
        raise ValueError("No slab hoppings were generated.")

    keys = sorted(slab_hoppings.keys())
    slab_i_rn = np.zeros((len(keys), 3), dtype=int)
    slab_h = np.zeros((len(keys), next(iter(slab_hoppings.values())).shape[0], next(iter(slab_hoppings.values())).shape[1]), dtype=complex)

    for index, key in enumerate(keys):
        rx, ry = key
        slab_i_rn[index] = np.array([rx, ry, 0], dtype=int)
        slab_h[index] = slab_hoppings[key]

    return slab_i_rn, slab_h


def build_z_slab_position(i_rn: np.ndarray, position: Optional[np.ndarray], degen: np.ndarray, bravais: np.ndarray, gamma_index: int, m_size: int, n_layers: int) -> Optional[Dict[Translation2D, np.ndarray]]:
    """
    Build slab position matrices R_slab(Rx, Ry) for a z-oriented slab.

    The finite layer index is internal to the enlarged orbital basis. Position
    elements that would connect outside the finite slab are discarded.
    """
    if position is None:
        return None

    slab_size = m_size * n_layers
    slab_position: Dict[Translation2D, np.ndarray] = {}

    for block, translation in enumerate(i_rn):
        rx, ry, rz = (int(value) for value in translation)
        key = (rx, ry)
        r_bulk = position[block] / degen[block]

        for layer in range(n_layers):
            target_layer = layer + rz
            if target_layer < 0 or target_layer >= n_layers:
                continue

            if key not in slab_position:
                slab_position[key] = np.zeros((slab_size, slab_size, 3), dtype=complex)

            row_start = layer * m_size
            col_start = target_layer * m_size
            slab_position[key][row_start:row_start + m_size, col_start:col_start + m_size, :] += r_bulk

    # Correct the diagonal Wannier centers for the (0,0) block
    if (0, 0) in slab_position:
        bulk_centers = np.real(position[gamma_index, np.arange(m_size), np.arange(m_size), :])
        for layer in range(n_layers):
            for m in range(m_size):
                I = m + layer * m_size
                slab_center = bulk_centers[m] + layer * bravais[2]
                slab_position[(0, 0)][I, I, :] = slab_center

    return slab_position


def slab_position_dict_to_array(slab_i_rn: np.ndarray, slab_position_dict: Optional[Dict[Translation2D, np.ndarray]], slab_size: int) -> Optional[np.ndarray]:
    """Convert a slab position dictionary into an array aligned with slab_i_rn."""
    if slab_position_dict is None:
        return None

    n_fock = slab_i_rn.shape[0]
    position = np.zeros((n_fock, slab_size, slab_size, 3), dtype=complex)

    for block in range(n_fock):
        rx, ry, _ = slab_i_rn[block]
        key = (rx, ry)
        if key in slab_position_dict:
            position[block] = slab_position_dict[key]

    return position


def build_hk_2d(slab_hoppings: SlabHoppings, bravais: np.ndarray, k_cart: np.ndarray) -> np.ndarray:
    """Build H(kx, ky) using Cartesian k and Cartesian real-space lattice vectors."""
    first_block = next(iter(slab_hoppings.values()))
    h_k = np.zeros_like(first_block, dtype=complex)

    for (rx, ry), h_r in slab_hoppings.items():
        r_cart = rx * bravais[0] + ry * bravais[1]
        phase = np.exp(1j * np.dot(k_cart, r_cart))
        h_k += phase * h_r

    return h_k


def check_hermiticity(slab_hoppings: SlabHoppings, bravais: np.ndarray) -> float:
    """Check Hermiticity of the slab Bloch Hamiltonian at Gamma."""
    k_gamma = np.zeros(3, dtype=float)
    h_gamma = build_hk_2d(slab_hoppings, bravais, k_gamma)
    diff = h_gamma - h_gamma.conj().T
    max_error = float(np.max(np.abs(diff)))
    norm_error = float(np.linalg.norm(diff))
    print(f"Hermiticity check at Gamma: max_abs = {max_error:.6e}, frobenius = {norm_error:.6e}")
    return max_error


def check_nz_one_projection(
    i_rn: np.ndarray,
    hamiltonian: np.ndarray,
    degen: np.ndarray,
    m_size: int,
    slab_hoppings: SlabHoppings,
) -> float:
    """
    For Nz = 1, compare the slab with the Rz = 0 projected bulk model.

    This is exact only when the supplied slab_hoppings were generated with one
    layer. The returned value is the maximum absolute block difference.
    """
    projected: SlabHoppings = {}
    for block, translation in enumerate(i_rn):
        rx, ry, rz = (int(value) for value in translation)
        if rz != 0:
            continue
        key = (rx, ry)
        if key not in projected:
            projected[key] = np.zeros((m_size, m_size), dtype=complex)
        projected[key] += hamiltonian[block] / degen[block]

    all_keys = set(projected) | set(slab_hoppings)
    max_error = 0.0
    zero = np.zeros((m_size, m_size), dtype=complex)
    for key in all_keys:
        left = slab_hoppings.get(key, zero)
        right = projected.get(key, zero)
        max_error = max(max_error, float(np.max(np.abs(left - right))))

    print(f"Nz = 1 projection check: max_abs = {max_error:.6e}")
    return max_error


def write_tb_dat(
    filename: str | Path,
    bravais: np.ndarray,
    i_rn: np.ndarray,
    hamiltonian: np.ndarray,
    degen: Optional[np.ndarray] = None,
    position: Optional[np.ndarray] = None,
    header: str = "Slab tight-binding model generated from seedname_tb.dat",
) -> None:
    """Write a TB-style file following the seedname_tb.dat block structure."""
    path = Path(filename)
    n_fock = int(i_rn.shape[0])
    m_size = int(hamiltonian.shape[1])

    if degen is None:
        degen = np.ones(n_fock, dtype=int)
    if degen.shape[0] != n_fock:
        raise ValueError("Degeneracy list length does not match the number of hopping blocks.")

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        for vector in bravais:
            handle.write("{:20.12f} {:20.12f} {:20.12f}\n".format(*vector))
        handle.write(f"{m_size}\n")
        handle.write(f"{n_fock}\n")

        for start in range(0, n_fock, 15):
            values = degen[start:start + 15]
            handle.write("".join(f"{int(value):5d}" for value in values) + "\n")
        handle.write("\n")

        for block in range(n_fock):
            handle.write("{:5d} {:5d} {:5d}\n".format(*i_rn[block]))
            for i_index in range(m_size):
                for j_index in range(m_size):
                    value = hamiltonian[block, i_index, j_index]
                    handle.write(
                        "{:5d} {:5d} {:22.14e} {:22.14e}\n".format(
                            i_index + 1,
                            j_index + 1,
                            float(np.real(value)),
                            float(np.imag(value)),
                        )
                    )
            handle.write("\n")

        if position is not None:
            for block in range(n_fock):
                handle.write("{:5d} {:5d} {:5d}\n".format(*i_rn[block]))
                for i_index in range(m_size):
                    for j_index in range(m_size):
                        x_value = position[block, i_index, j_index, 0]
                        y_value = position[block, i_index, j_index, 1]
                        z_value = position[block, i_index, j_index, 2]
                        handle.write(
                            "{:5d} {:5d} {:22.14e} {:22.14e} {:22.14e} {:22.14e} {:22.14e} {:22.14e}\n".format(
                                i_index + 1,
                                j_index + 1,
                                float(np.real(x_value)),
                                float(np.imag(x_value)),
                                float(np.real(y_value)),
                                float(np.imag(y_value)),
                                float(np.real(z_value)),
                                float(np.imag(z_value)),
                            )
                        )
                if block + 1 < n_fock:
                    handle.write("\n")


def diagonalize_gamma(slab_hoppings: SlabHoppings, bravais: np.ndarray, n_print: int = 20) -> np.ndarray:
    """Diagonalize the slab Hamiltonian at Gamma and print the lowest eigenvalues."""
    h_gamma = build_hk_2d(slab_hoppings, bravais, np.zeros(3, dtype=float))
    eigenvalues = np.linalg.eigvalsh(h_gamma)
    n_show = min(n_print, eigenvalues.size)
    print(f"Lowest {n_show} Gamma eigenvalues:")
    for index, value in enumerate(eigenvalues[:n_show]):
        print(f"  {index:5d}  {value: .12f}")
    return eigenvalues


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a finite z-slab TB Hamiltonian from a Wannier90 seedname_tb.dat file."
    )
    parser.add_argument("input", help="Input seedname_tb.dat file.")
    parser.add_argument("n_layers", type=int, help="Number of finite slab layers.")
    parser.add_argument(
        "-d",
        "--direction",
        default="z",
        choices=("x", "y", "z"),
        help="Finite slab direction. Only z is implemented in this version.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output slab TB filename. Default: input stem plus _slab_tb.dat.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run construction and checks without writing the output file.",
    )
    parser.add_argument(
        "--diagonalize-gamma",
        action="store_true",
        help="Diagonalize the slab Hamiltonian at Gamma and print eigenvalues.",
    )
    parser.add_argument(
        "--no-position-block",
        action="store_true",
        help="Do not write the minimal slab position block.",
    )
    return parser


def default_output_name(input_file: str | Path) -> str:
    path = Path(input_file)
    name = path.name
    if name.endswith("_tb.dat"):
        return str(path.with_name(name.replace("_tb.dat", "_slab_tb.dat")))
    return str(path.with_name(path.stem + "_slab_tb.dat"))


def main() -> None:
    args = make_argument_parser().parse_args()
    output = args.output or default_output_name(args.input)

    tb_data = parse_tb_dat(args.input)
    print(f"Read {args.input}")
    print(f"  bulk orbitals: {tb_data.m_size}")
    print(f"  bulk hopping blocks: {tb_data.n_fock}")
    print(f"  gamma block index: {tb_data.gamma_index}")

    bravais_slab, slab_hoppings = build_slab_hoppings(
        tb_data.bravais,
        tb_data.i_rn,
        tb_data.hamiltonian,
        tb_data.degen,
        tb_data.m_size,
        args.n_layers,
        args.direction,
    )
    check_hermiticity(slab_hoppings, bravais_slab)

    if args.n_layers == 1:
        check_nz_one_projection(tb_data.i_rn, tb_data.hamiltonian, tb_data.degen, tb_data.m_size, slab_hoppings)

    if args.diagonalize_gamma:
        diagonalize_gamma(slab_hoppings, bravais_slab)

    slab_i_rn, slab_h = slab_dict_to_arrays(slab_hoppings)
    slab_size = tb_data.m_size * args.n_layers
    slab_degen = np.ones(slab_i_rn.shape[0], dtype=int)

    print("Bulk degeneracy factors have been absorbed into slab matrices; output degeneracies set to 1.")

    slab_position_dict = build_z_slab_position(
        tb_data.i_rn,
        tb_data.position,
        tb_data.degen,
        tb_data.bravais,
        tb_data.gamma_index,
        tb_data.m_size,
        args.n_layers,
    )
    position = None
    if not args.no_position_block:
        position = slab_position_dict_to_array(slab_i_rn, slab_position_dict, slab_size)
        print("Full slab position matrix block written, with diagonal centers corrected by layer offsets.")

    print(f"Built z slab with {args.n_layers} layers")
    print(f"  slab orbitals: {slab_size}")
    print(f"  slab hopping blocks: {slab_i_rn.shape[0]}")

    if args.no_write:
        print("Output writing skipped by --no-write.")
        return

    write_tb_dat(
        output,
        bravais_slab,
        slab_i_rn,
        slab_h,
        degen=slab_degen,
        position=position,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
