#!/usr/bin/env python3
"""
Build a finite slab tight-binding Hamiltonian from a Wannier90 seedname_tb.dat file.

Initial target:
    z-oriented slab, finite along a3, periodic along a1 and a2.

The slab Hamiltonian uses the enlarged basis:
    I = m + l * mSize

No external/static confinement potential is added here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TbData:
    bravais: np.ndarray
    m_size: int
    n_fock: int
    degen: np.ndarray
    i_rn: np.ndarray
    h_r: np.ndarray
    wf: np.ndarray | None
    diag_index: int | None


def parse_tb_dat(filename: str | Path) -> TbData:
    path = Path(filename)

    with path.open("r", encoding="utf-8") as handle:
        line_number = 0

        def next_nonempty_line(handle):
            nonlocal line_number

            while True:
                line = handle.readline()

                if line == "":
                    return None

                line_number += 1
                stripped = line.strip()

                if stripped:
                    return line_number, stripped

        def required_nonempty_line(description: str) -> tuple[int, str]:
            result = next_nonempty_line(handle)

            if result is None:
                raise ValueError(f"Unexpected end of file while reading {description}.")

            return result

        def parse_r_vector(
            text: str,
            line_number: int,
            description: str,
        ) -> tuple[int, int, int]:
            parts = text.split()

            if len(parts) < 3:
                raise ValueError(
                    f"Expected Rx Ry Rz for {description} at line {line_number}: {text}"
                )

            try:
                return int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Rx Ry Rz for {description} at line {line_number}: {text}"
                ) from exc

        def check_orbital_indices(i: int, j: int, line_number: int, text: str) -> None:
            if not (0 <= i < m_size and 0 <= j < m_size):
                raise ValueError(
                    f"Orbital index out of range at line {line_number}: {text}"
                )

        # Header
        header = handle.readline()
        if header == "":
            raise ValueError("Unexpected end of file while reading header.")
        line_number += 1

        bravais = np.zeros((3, 3), dtype=float)
        for vec_index in range(3):
            ln, text = required_nonempty_line(f"Bravais vector {vec_index + 1}")
            parts = text.split()

            if len(parts) < 3:
                raise ValueError(f"Expected 3 Bravais components at line {ln}: {text}")

            bravais[vec_index] = [float(x) for x in parts[:3]]

        _, text = required_nonempty_line("mSize")
        m_size = int(text.split()[0])

        _, text = required_nonempty_line("nFock")
        n_fock = int(text.split()[0])

        degen_values = []
        while len(degen_values) < n_fock:
            ln, text = required_nonempty_line("degeneracy list")
            parts = text.split()

            try:
                degen_values.extend(int(x) for x in parts)
            except ValueError as exc:
                raise ValueError(f"Invalid degeneracy value at line {ln}: {text}") from exc

        degen = np.array(degen_values[:n_fock], dtype=int)

        i_rn = np.zeros((n_fock, 3), dtype=int)
        h_r = np.zeros((n_fock, m_size, m_size), dtype=complex)

        for block in range(n_fock):
            ln, text = required_nonempty_line(f"H block {block} translation")
            i_rn[block] = parse_r_vector(text, ln, f"H block {block}")

            for entry in range(m_size * m_size):
                ln, text = required_nonempty_line(
                    f"H block {block} entry {entry + 1}"
                )
                parts = text.split()

                if len(parts) < 4:
                    raise ValueError(
                        f"Expected i j real imag for H block {block} at line {ln}: {text}"
                    )

                try:
                    i = int(parts[0]) - 1
                    j = int(parts[1]) - 1
                    value = float(parts[2]) + 1j * float(parts[3])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid H matrix entry for block {block} at line {ln}: {text}"
                    ) from exc

                check_orbital_indices(i, j, ln, text)
                h_r[block, i, j] = value

        wf = None
        first_wf_line = next_nonempty_line(handle)

        if first_wf_line is not None:
            wf = np.zeros((n_fock, m_size, m_size, 3), dtype=complex)

            for block in range(n_fock):
                if block == 0:
                    ln, text = first_wf_line
                else:
                    ln, text = required_nonempty_line(f"WF block {block} translation")

                parse_r_vector(text, ln, f"WF block {block}")

                for entry in range(m_size * m_size):
                    ln, text = required_nonempty_line(
                        f"WF block {block} entry {entry + 1}"
                    )
                    parts = text.split()

                    if len(parts) < 8:
                        raise ValueError(
                            f"Expected i j rx_real rx_imag ry_real ry_imag "
                            f"rz_real rz_imag for WF block {block} at line {ln}: {text}"
                        )

                    try:
                        i = int(parts[0]) - 1
                        j = int(parts[1]) - 1
                        wf[block, i, j, 0] = float(parts[2]) + 1j * float(parts[3])
                        wf[block, i, j, 1] = float(parts[4]) + 1j * float(parts[5])
                        wf[block, i, j, 2] = float(parts[6]) + 1j * float(parts[7])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid WF matrix entry for block {block} at line {ln}: {text}"
                        ) from exc

                    check_orbital_indices(i, j, ln, text)

    matches = np.where(np.all(i_rn == np.array([0, 0, 0]), axis=1))[0]
    diag_index = int(matches[0]) if len(matches) > 0 else None

    return TbData(
        bravais=bravais,
        m_size=m_size,
        n_fock=n_fock,
        degen=degen,
        i_rn=i_rn,
        h_r=h_r,
        wf=wf,
        diag_index=diag_index,
    )


def build_z_slab(
    i_rn: np.ndarray,
    h_r: np.ndarray,
    m_size: int,
    nz: int,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Build slab hopping matrices H_slab(Rx, Ry) for a slab finite along z.
    """
    if nz < 1:
        raise ValueError("Number of layers must be positive.")

    slab_size = m_size * nz
    slab_dict: dict[tuple[int, int], np.ndarray] = {}

    for block, r_vec in enumerate(i_rn):
        rx, ry, rz = (int(r_vec[0]), int(r_vec[1]), int(r_vec[2]))
        h_bulk = h_r[block]
        key = (rx, ry)

        for layer in range(nz):
            layer_p = layer + rz

            if layer_p < 0 or layer_p >= nz:
                continue

            if key not in slab_dict:
                slab_dict[key] = np.zeros((slab_size, slab_size), dtype=complex)

            row_start = layer * m_size
            col_start = layer_p * m_size

            # Keep Rx, Ry periodic while dropping z hoppings that leave the slab.
            slab_dict[key][
                row_start : row_start + m_size,
                col_start : col_start + m_size,
            ] += h_bulk

    return slab_dict


def build_slab_hoppings(
    bravais: np.ndarray,
    i_rn: np.ndarray,
    h_r: np.ndarray,
    m_size: int,
    n_layers: int,
    direction: str = "z",
    vacuum: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build slab hopping arrays.

    For now, direction == "z" means finite along the third input lattice vector a3.
    """
    if direction != "z":
        raise NotImplementedError(
            "Only direction='z', finite along bravais[2], is implemented."
        )

    slab_dict = build_z_slab(i_rn, h_r, m_size, n_layers)

    keys = sorted(slab_dict.keys())
    slab_i_rn = np.array([[rx, ry, 0] for rx, ry in keys], dtype=int)
    slab_h = np.array([slab_dict[key] for key in keys], dtype=complex)

    bravais_slab = bravais.copy()
    normal_norm = np.linalg.norm(bravais[2])
    if normal_norm == 0.0:
        raise ValueError("Cannot build slab lattice: input a3 vector has zero length.")

    normal = bravais[2] / normal_norm
    bravais_slab[2] = n_layers * bravais[2] + vacuum * normal

    return bravais_slab, slab_i_rn, slab_h


def build_hk_2d(
    slab_i_rn: np.ndarray,
    slab_h: np.ndarray,
    bravais: np.ndarray,
    k_cart: np.ndarray,
) -> np.ndarray:
    """
    Build H(kx, ky) using Cartesian k and Cartesian lattice vectors.
    """
    h_k = np.zeros_like(slab_h[0])

    for block, r_vec in enumerate(slab_i_rn):
        rx, ry = int(r_vec[0]), int(r_vec[1])
        r_cart = rx * bravais[0] + ry * bravais[1]
        phase = np.exp(1j * np.dot(k_cart, r_cart))
        h_k += phase * slab_h[block]

    return h_k


def check_hermiticity(
    slab_i_rn: np.ndarray,
    slab_h: np.ndarray,
    bravais: np.ndarray,
) -> float:
    """
    Check Hermiticity of H(k) at Gamma.
    """
    k_gamma = np.zeros(3, dtype=float)
    h_gamma = build_hk_2d(slab_i_rn, slab_h, bravais, k_gamma)
    err = np.max(np.abs(h_gamma - h_gamma.conj().T))
    return float(err)


def check_nz1_projection(tb: TbData, slab_i_rn: np.ndarray, slab_h: np.ndarray) -> float:
    """
    For Nz = 1, the slab must match the bulk model projected to Rz = 0.
    """
    projected: dict[tuple[int, int], np.ndarray] = {}

    for block, (rx, ry, rz) in enumerate(tb.i_rn):
        if rz != 0:
            continue

        key = (int(rx), int(ry))

        if key not in projected:
            projected[key] = np.zeros((tb.m_size, tb.m_size), dtype=complex)

        projected[key] += tb.h_r[block]

    slab_by_key = {
        (int(rx), int(ry)): slab_h[block]
        for block, (rx, ry, _rz) in enumerate(slab_i_rn)
    }

    all_keys = set(projected) | set(slab_by_key)
    max_error = 0.0

    for key in all_keys:
        expected = projected.get(key)
        actual = slab_by_key.get(key)

        if expected is None:
            expected = np.zeros_like(actual)
        if actual is None:
            actual = np.zeros_like(expected)

        max_error = max(max_error, float(np.max(np.abs(actual - expected))))

    return max_error


def extract_z_slab_centers(tb: TbData, nz: int) -> np.ndarray | None:
    """
    Build slab Wannier centers for a z-oriented slab.

    r_{m,l} = r_m + l * a3

    Returns:
        centers with shape (m_size * nz, 3), or None if WF is missing.
    """
    if tb.wf is None:
        print("Warning: WF block is absent; slab Wannier centers cannot be extracted.")
        return None

    if tb.diag_index is None:
        raise ValueError("Cannot extract slab centers: R=(0,0,0) block not found.")

    bulk_centers = np.zeros((tb.m_size, 3), dtype=float)

    for m in range(tb.m_size):
        center = tb.wf[tb.diag_index, m, m, :]
        bulk_centers[m] = np.real(center)

    centers = np.zeros((tb.m_size * nz, 3), dtype=float)

    for layer in range(nz):
        for m in range(tb.m_size):
            index = m + layer * tb.m_size
            centers[index] = bulk_centers[m] + layer * tb.bravais[2]

    return centers


def build_z_slab_wf_approx(
    i_rn: np.ndarray,
    wf: np.ndarray,
    m_size: int,
    n_layers: int,
    centers: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Build an approximate slab position matrix using the same layer map as H.

    This preserves the bulk in-plane position-matrix structure, but it is not a
    fully re-Wannierized slab PME and should not be trusted blindly for optical
    matrix elements involving surface states.
    """
    slab_size = m_size * n_layers
    wf_dict: dict[tuple[int, int], np.ndarray] = {}

    for block, r_vec in enumerate(i_rn):
        rx, ry, rz = map(int, r_vec)
        key = (rx, ry)

        for layer in range(n_layers):
            layer_p = layer + rz

            if layer_p < 0 or layer_p >= n_layers:
                continue

            if key not in wf_dict:
                wf_dict[key] = np.zeros((slab_size, slab_size, 3), dtype=complex)

            row_start = layer * m_size
            col_start = layer_p * m_size

            wf_dict[key][
                row_start : row_start + m_size,
                col_start : col_start + m_size,
                :,
            ] += wf[block]

    if (0, 0) in wf_dict:
        for i in range(slab_size):
            wf_dict[(0, 0)][i, i, :] = centers[i]

    return wf_dict


def write_tb_dat(
    filename: str | Path,
    bravais: np.ndarray,
    slab_i_rn: np.ndarray,
    slab_h: np.ndarray,
    write_empty_wf: bool = False,
    wf_centers: np.ndarray | None = None,
    slab_wf: np.ndarray | None = None,
) -> None:
    """
    Write a simple Wannier90-like _tb.dat file.

    Degeneracies are set to 1 for all slab hopping blocks.
    """
    path = Path(filename)
    n_fock = slab_h.shape[0]
    m_size = slab_h.shape[1]
    write_wf = write_empty_wf or wf_centers is not None or slab_wf is not None
    slab_diag_block = None

    if slab_wf is not None and slab_wf.shape != (n_fock, m_size, m_size, 3):
        raise ValueError(
            "Slab WF block must have shape "
            f"({n_fock}, {m_size}, {m_size}, 3), got {slab_wf.shape}."
        )

    if wf_centers is not None:
        if wf_centers.shape != (m_size, 3):
            raise ValueError(
                f"WF centers must have shape ({m_size}, 3), got {wf_centers.shape}."
            )

        matches = np.where(np.all(slab_i_rn == np.array([0, 0, 0]), axis=1))[0]
        if len(matches) == 0:
            raise ValueError("Cannot write WF centers: slab R=(0,0,0) block not found.")

        slab_diag_block = int(matches[0])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Slab tight-binding model generated from bulk seedname_tb.dat\n")

        for vec in bravais:
            handle.write(f"{vec[0]:22.14f} {vec[1]:22.14f} {vec[2]:22.14f}\n")

        handle.write(f"{m_size:d}\n")
        handle.write(f"{n_fock:d}\n")

        for start in range(0, n_fock, 15):
            values = ["1" for _ in range(start, min(start + 15, n_fock))]
            handle.write(" ".join(values) + "\n")

        handle.write("\n")

        for block in range(n_fock):
            rx, ry, rz = slab_i_rn[block]
            handle.write(f"{rx:5d} {ry:5d} {rz:5d}\n")

            for i in range(m_size):
                for j in range(m_size):
                    value = slab_h[block, i, j]
                    handle.write(
                        f"{i + 1:5d} {j + 1:5d} "
                        f"{value.real:22.14e} {value.imag:22.14e}\n"
                    )

            handle.write("\n")

        if write_wf:
            for block in range(n_fock):
                rx, ry, rz = slab_i_rn[block]
                handle.write(f"{rx:5d} {ry:5d} {rz:5d}\n")

                for i in range(m_size):
                    for j in range(m_size):
                        if slab_wf is not None:
                            value = slab_wf[block, i, j]
                        elif block == slab_diag_block and i == j:
                            center = wf_centers[i]
                            value = center.astype(complex)
                        else:
                            value = np.zeros(3, dtype=complex)

                        handle.write(
                            f"{i + 1:5d} {j + 1:5d} "
                            f"{value[0].real:22.14e} {value[0].imag:22.14e} "
                            f"{value[1].real:22.14e} {value[1].imag:22.14e} "
                            f"{value[2].real:22.14e} {value[2].imag:22.14e}\n"
                        )

                handle.write("\n")


def validate_xatu_format(filename: str | Path) -> bool:
    """
    Validate the strict block layout expected by XATU's Wannier90 parser.
    """
    path = Path(filename)
    lines = path.read_text(encoding="utf-8").splitlines()

    def text_at(index: int) -> str:
        if index >= len(lines):
            return "<EOF>"
        return lines[index]

    def fail(index: int, block: int | None, message: str) -> bool:
        line_number = index + 1 if index < len(lines) else len(lines) + 1
        block_text = "n/a" if block is None else str(block)
        print(
            f"XATU format validation failed at line {line_number}, "
            f"block {block_text}: {message}"
        )
        print(f"Offending text: {text_at(index)!r}")
        return False

    def require_line(index: int, block: int | None, description: str) -> bool:
        if index >= len(lines):
            return fail(index, block, f"expected {description}")
        return True

    cursor = 0

    if not require_line(cursor, None, "header"):
        return False
    cursor += 1

    for vec_index in range(3):
        if not require_line(cursor, None, f"Bravais vector {vec_index + 1}"):
            return False

        parts = lines[cursor].split()
        if len(parts) != 3:
            return fail(cursor, None, "Bravais vector must have exactly 3 fields")

        try:
            [float(part) for part in parts]
        except ValueError:
            return fail(cursor, None, "Bravais vector contains a non-float field")

        cursor += 1

    if not require_line(cursor, None, "mSize"):
        return False
    parts = lines[cursor].split()
    if len(parts) != 1:
        return fail(cursor, None, "mSize line must have exactly 1 field")
    try:
        m_size = int(parts[0])
    except ValueError:
        return fail(cursor, None, "mSize is not an integer")
    cursor += 1

    if not require_line(cursor, None, "nFock"):
        return False
    parts = lines[cursor].split()
    if len(parts) != 1:
        return fail(cursor, None, "nFock line must have exactly 1 field")
    try:
        n_fock = int(parts[0])
    except ValueError:
        return fail(cursor, None, "nFock is not an integer")
    cursor += 1

    degen_read = 0
    while degen_read < n_fock:
        if not require_line(cursor, None, "degeneracy line"):
            return False

        parts = lines[cursor].split()
        if not parts:
            return fail(cursor, None, "blank line before full degeneracy list")

        try:
            [int(part) for part in parts]
        except ValueError:
            return fail(cursor, None, "degeneracy line contains a non-integer field")

        degen_read += len(parts)
        cursor += 1

    if degen_read != n_fock:
        return fail(
            cursor - 1,
            None,
            f"degeneracy list has {degen_read} values, expected {n_fock}",
        )

    if not require_line(cursor, None, "blank line after degeneracy list"):
        return False
    if lines[cursor] != "":
        return fail(cursor, None, "expected exactly one blank line after degeneracy list")
    cursor += 1

    for block in range(n_fock):
        if not require_line(cursor, block, "H block translation"):
            return False

        parts = lines[cursor].split()
        if len(parts) != 3:
            return fail(cursor, block, "H block translation must have exactly 3 fields")

        try:
            [int(part) for part in parts]
        except ValueError:
            return fail(cursor, block, "H block translation contains a non-integer field")

        cursor += 1

        for _entry in range(m_size * m_size):
            if not require_line(cursor, block, "H matrix entry"):
                return False

            text = lines[cursor]
            parts = text.split()
            if len(parts) != 4:
                return fail(cursor, block, "H entry must have exactly 4 fields")

            try:
                i = int(parts[0])
                j = int(parts[1])
                float(parts[2])
                float(parts[3])
            except ValueError:
                return fail(cursor, block, "H entry has invalid numeric fields")

            if not (1 <= i <= m_size and 1 <= j <= m_size):
                return fail(cursor, block, "H entry indices are outside 1..mSize")

            cursor += 1

        if not require_line(cursor, block, "blank line after H block"):
            return False
        if lines[cursor] != "":
            return fail(cursor, block, "expected exactly one blank line after H block")
        cursor += 1

    for block in range(n_fock):
        if not require_line(cursor, block, "Rhop block translation"):
            return False

        parts = lines[cursor].split()
        if len(parts) != 3:
            return fail(cursor, block, "Rhop block translation must have exactly 3 fields")

        try:
            [int(part) for part in parts]
        except ValueError:
            return fail(cursor, block, "Rhop block translation contains a non-integer field")

        cursor += 1

        for _entry in range(m_size * m_size):
            if not require_line(cursor, block, "Rhop matrix entry"):
                return False

            parts = lines[cursor].split()
            if len(parts) != 8:
                return fail(cursor, block, "Rhop entry must have exactly 8 fields")

            try:
                i = int(parts[0])
                j = int(parts[1])
                [float(part) for part in parts[2:]]
            except ValueError:
                return fail(cursor, block, "Rhop entry has invalid numeric fields")

            if not (1 <= i <= m_size and 1 <= j <= m_size):
                return fail(cursor, block, "Rhop entry indices are outside 1..mSize")

            cursor += 1

        if not require_line(cursor, block, "blank line after Rhop block"):
            return False
        if lines[cursor] != "":
            return fail(cursor, block, "expected exactly one blank line after Rhop block")
        cursor += 1

    print(
        f"XATU format validation passed: read {n_fock} H blocks and "
        f"{n_fock} Rhop blocks."
    )
    return True


def print_tb_diagnostics(tb: TbData) -> None:
    print("Parsed bulk TB:")
    print(f"  m_size = {tb.m_size}")
    print(f"  n_fock = {tb.n_fock}")
    print(f"  has WF block = {tb.wf is not None}")
    print(f"  diag_index = {tb.diag_index}")
    print("  R range:")
    print(f"      Rx {np.min(tb.i_rn[:, 0])} / {np.max(tb.i_rn[:, 0])}")
    print(f"      Ry {np.min(tb.i_rn[:, 1])} / {np.max(tb.i_rn[:, 1])}")
    print(f"      Rz {np.min(tb.i_rn[:, 2])} / {np.max(tb.i_rn[:, 2])}")


def diagonalize_gamma(
    slab_i_rn: np.ndarray,
    slab_h: np.ndarray,
    bravais: np.ndarray,
) -> np.ndarray:
    h_gamma = build_hk_2d(slab_i_rn, slab_h, bravais, np.zeros(3, dtype=float))
    eigvals = np.linalg.eigvalsh(h_gamma)
    return eigvals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a finite slab TB Hamiltonian from a Wannier90 _tb.dat file."
    )

    parser.add_argument("input", help="Input seedname_tb.dat file.")
    parser.add_argument("-n", "--layers", type=int, required=True, help="Number of layers.")
    parser.add_argument(
        "-d",
        "--direction",
        default="z",
        choices=["z", "x", "y"],
        help="Finite slab direction. z means finite along input bravais vector a3.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output slab _tb.dat file.")
    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum length to add along the slab normal. Default: 0.0.",
    )
    parser.add_argument(
        "--miller",
        nargs=3,
        type=int,
        metavar=("H", "K", "L"),
        help="Placeholder for Miller-index slabs; not implemented yet.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run Hermiticity check and Gamma diagonalization.",
    )
    parser.add_argument(
        "--write-empty-wf",
        action="store_true",
        help="Write a zero position-matrix block after the Hamiltonian.",
    )
    parser.add_argument(
        "--write-centers-wf",
        action="store_true",
        help="Write a diagonal position block with slab centers.",
    )
    parser.add_argument(
        "--write-approx-full-wf",
        action="store_true",
        help="Write an approximate full slab position block mapped from the bulk PME.",
    )
    parser.add_argument(
        "--save-centers",
        default=None,
        help="Optional output text file for slab Wannier centers.",
    )

    args = parser.parse_args()

    if args.miller is not None:
        raise NotImplementedError(
            "Miller-index slabs require constructing a surface supercell before "
            "cutting. Use a pre-oriented conventional/surface cell for now."
        )

    tb = parse_tb_dat(args.input)
    print_tb_diagnostics(tb)
    print(
        "Slab direction z means finite along the third input lattice vector a3.\n"
        "This may not correspond to a crystallographic (001) surface unless the "
        "input cell is built that way."
    )

    bravais_slab, slab_i_rn, slab_h = build_slab_hoppings(
        bravais=tb.bravais,
        i_rn=tb.i_rn,
        h_r=tb.h_r,
        m_size=tb.m_size,
        n_layers=args.layers,
        direction=args.direction,
        vacuum=args.vacuum,
    )

    centers = None
    needs_centers = (
        args.write_centers_wf
        or args.write_approx_full_wf
        or args.save_centers is not None
    )

    if needs_centers:
        centers = extract_z_slab_centers(tb, args.layers)

    slab_wf = None

    if args.write_approx_full_wf:
        if tb.wf is None:
            raise ValueError("Cannot write approximate full WF: input WF block is absent.")
        if centers is None:
            raise ValueError("Cannot write approximate full WF: slab centers unavailable.")

        wf_dict = build_z_slab_wf_approx(
            i_rn=tb.i_rn,
            wf=tb.wf,
            m_size=tb.m_size,
            n_layers=args.layers,
            centers=centers,
        )
        slab_wf = np.zeros(
            (len(slab_i_rn), slab_h.shape[1], slab_h.shape[2], 3),
            dtype=complex,
        )

        for block, (rx, ry, _rz) in enumerate(slab_i_rn):
            slab_wf[block] = wf_dict.get((int(rx), int(ry)), slab_wf[block])

        print(
            "Warning: --write-approx-full-wf preserves the bulk PME layer mapping "
            "but is not a fully re-Wannierized slab position matrix."
        )
    elif args.write_centers_wf and centers is None:
        print("Warning: could not write centers WF because slab centers are unavailable.")

    write_tb_dat(
        filename=args.output,
        bravais=bravais_slab,
        slab_i_rn=slab_i_rn,
        slab_h=slab_h,
        write_empty_wf=args.write_empty_wf,
        wf_centers=centers if args.write_centers_wf else None,
        slab_wf=slab_wf,
    )

    print(f"Wrote slab TB file: {args.output}")
    print(f"Bulk orbitals per cell: {tb.m_size}")
    print(f"Slab layers: {args.layers}")
    print(f"Vacuum length along slab normal: {args.vacuum}")
    print(f"Slab orbitals: {tb.m_size * args.layers}")
    print(f"Slab hopping blocks: {len(slab_i_rn)}")
    print("GaAs note: for primitive fcc-like vectors, a3 may not be the conventional")
    print("cubic [001] direction. For a true (001) slab, use a pre-oriented cell.")

    if args.check:
        err = check_hermiticity(slab_i_rn, slab_h, bravais_slab)
        eigvals = diagonalize_gamma(slab_i_rn, slab_h, bravais_slab)

        print(f"Gamma Hermiticity max abs error: {err:.6e}")
        if args.layers == 1:
            projection_err = check_nz1_projection(tb, slab_i_rn, slab_h)
            print(f"Nz=1 projection max abs error: {projection_err:.6e}")
        print("Gamma eigenvalues:")
        for value in eigvals:
            print(f"{value:22.14e}")

    if args.save_centers is not None:
        if centers is None:
            print("Could not save centers: WF block or R=(0,0,0) block not found.")
        else:
            np.savetxt(args.save_centers, centers, fmt="%22.14e")
            print(f"Wrote slab centers: {args.save_centers}")


if __name__ == "__main__":
    main()
