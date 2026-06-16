#!/usr/bin/env python3
"""Postprocess Xatu .states plus Wannier90 seedname_tb.dat into .rswf blocks.

The script reconstructs single-particle eigenvectors by diagonalizing H(k).
Exact equality with Xatu is therefore not guaranteed when eigenvector phases or
degenerate subspace rotations differ from those used during the BSE run.

All exciton states found in .states are written by default. For the Wannier90
workflow, one Wannier function is one real-space plotting site.

Example:
  python postprocess_rswf.py \\
      --states calc.states \\
      --tb seedname_tb.dat \\
      -r 0,2,8 \\
      --ncells 30

Multiple fixed hole indices reuse the same parsed .states data and the same
rediagonalized H(k) data, then write one .rswf file per fixed hole.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


KKey = Tuple[float, float, float]


@dataclass
class StatesData:
    basis_dim: int
    kpoints: np.ndarray
    valence_bands: np.ndarray
    conduction_bands: np.ndarray
    states: np.ndarray


@dataclass
class WannierTBData:
    lattice_vectors: np.ndarray
    r_integer_vectors: np.ndarray
    degeneracies: np.ndarray
    hopping_blocks: np.ndarray
    position_blocks: Optional[np.ndarray]


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def k_key(kpoint: Sequence[float]) -> KKey:
    return tuple(np.round(np.asarray(kpoint, dtype=float), 10))  # type: ignore[return-value]


def next_nonempty_line(handle, context: str) -> str:
    for line in handle:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError(f"Unexpected end of file while reading {context}.")


def parse_states(path: str, band_index_base: int = 0) -> StatesData:
    if band_index_base not in (0, 1):
        raise ValueError("--band-index-base must be 0 or 1.")

    with open(path, "r", encoding="utf-8") as handle:
        lines = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not lines:
        raise ValueError(f"Empty .states file: {path}")

    try:
        basis_dim = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("First .states line must contain the exciton basis dimension.") from exc

    if len(lines) < 1 + basis_dim:
        raise ValueError(
            f".states file has {len(lines) - 1} basis lines, expected {basis_dim}."
        )

    kpoints = np.zeros((basis_dim, 3), dtype=float)
    valence = np.zeros(basis_dim, dtype=int)
    conduction = np.zeros(basis_dim, dtype=int)

    for i, line in enumerate(lines[1 : 1 + basis_dim]):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed basis line {i + 2}: expected kx ky kz v c.")
        try:
            kpoints[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            valence[i] = int(parts[3]) - band_index_base
            conduction[i] = int(parts[4]) - band_index_base
        except ValueError as exc:
            raise ValueError(f"Malformed numeric value in basis line {i + 2}.") from exc

    states: List[np.ndarray] = []
    expected_values = 2 * basis_dim
    for line_number, line in enumerate(lines[1 + basis_dim :], start=2 + basis_dim):
        parts = line.split()
        if len(parts) != expected_values:
            raise ValueError(
                f"Malformed state line {line_number}: expected {expected_values} "
                f"floating-point values, got {len(parts)}."
            )
        try:
            values = np.array([float(item) for item in parts], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Malformed numeric value in state line {line_number}.") from exc
        states.append(values[0::2] + 1j * values[1::2])

    if not states:
        raise ValueError(".states file does not contain any exciton-state lines.")

    return StatesData(
        basis_dim=basis_dim,
        kpoints=kpoints,
        valence_bands=valence,
        conduction_bands=conduction,
        states=np.asarray(states, dtype=np.complex128),
    )


def parse_index_list(text: str) -> List[int]:
    if text is None or text == "":
        raise ValueError("Hole index list must not be empty.")

    indices: List[int] = []
    seen = set()
    for item in text.split(","):
        if item == "":
            raise ValueError(f"Malformed hole index list: {text!r}.")

        if "-" in item:
            if item.count("-") != 1:
                raise ValueError(f"Malformed range in hole index list: {item!r}.")
            start_text, end_text = item.split("-", 1)
            if start_text == "" or end_text == "":
                raise ValueError(f"Malformed range in hole index list: {item!r}.")
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"Non-integer range in hole index list: {item!r}.") from exc
            if start < 0 or end < 0:
                raise ValueError("Hole indices must be non-negative.")
            if end < start:
                raise ValueError(f"Malformed descending range in hole index list: {item!r}.")
            expanded = range(start, end + 1)
        else:
            try:
                value = int(item)
            except ValueError as exc:
                raise ValueError(f"Non-integer hole index: {item!r}.") from exc
            if value < 0:
                raise ValueError("Hole indices must be non-negative.")
            expanded = (value,)

        for value in expanded:
            if value not in seen:
                indices.append(value)
                seen.add(value)

    if not indices:
        raise ValueError("Hole index list must contain at least one index.")
    return indices


# ----------------- wannier90 tb.dat parsing --------------------
def read_degeneracies(handle, n_fock: int) -> np.ndarray:
    values: List[int] = []
    while len(values) < n_fock:
        line = next_nonempty_line(handle, "degeneracy list")
        try:
            values.extend(int(item) for item in line.split())
        except ValueError as exc:
            raise ValueError("Malformed integer in Wannier90 degeneracy list.") from exc
    if len(values) > n_fock:
        values = values[:n_fock]
    return np.asarray(values, dtype=int)


def parse_wannier_tb(path: str) -> WannierTBData:
    with open(path, "r", encoding="utf-8") as handle:
        next(handle, None)  # header

        lattice = np.zeros((3, 3), dtype=float)
        for i in range(3):
            line = next_nonempty_line(handle, "Bravais lattice vectors")
            parts = line.split()
            if len(parts) < 3:
                raise ValueError("Each Bravais lattice row must contain 3 numbers.")
            lattice[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

        try:
            m_size = int(next_nonempty_line(handle, "number of Wannier orbitals").split()[0])
            n_fock = int(next_nonempty_line(handle, "number of hopping blocks").split()[0])
        except ValueError as exc:
            raise ValueError("Malformed m_size or n_fock in Wannier90 tb.dat.") from exc

        degeneracies = read_degeneracies(handle, n_fock)
        if np.any(degeneracies == 0):
            raise ValueError("Wannier90 degeneracy list contains zero.")

        r_integer_vectors = np.zeros((n_fock, 3), dtype=int)
        hopping_blocks = np.zeros((n_fock, m_size, m_size), dtype=np.complex128)

        for n in range(n_fock):
            line = next_nonempty_line(handle, f"hopping R-vector block {n}")
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed R-vector line for hopping block {n}.")
            r_integer_vectors[n] = [int(parts[0]), int(parts[1]), int(parts[2])]

            for _ in range(m_size * m_size):
                parts = next_nonempty_line(handle, f"hopping matrix block {n}").split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed hopping element in block {n}.")
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                if not (0 <= i < m_size and 0 <= j < m_size):
                    raise ValueError(f"Hopping index out of range in block {n}.")
                hopping_blocks[n, i, j] = complex(float(parts[2]), float(parts[3]))

        position_blocks = np.zeros((n_fock, m_size, m_size, 3), dtype=np.complex128)
        parsed_positions = False

        for n in range(n_fock):
            try:
                line = next_nonempty_line(handle, f"position-matrix R-vector block {n}")
            except ValueError:
                if n == 0:
                    position_blocks = None  # type: ignore[assignment]
                    break
                raise ValueError("Wannier90 position matrix block ended early.")

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed R-vector line for position block {n}.")

            for _ in range(m_size * m_size):
                parts = next_nonempty_line(handle, f"position matrix block {n}").split()
                if len(parts) < 8:
                    raise ValueError(f"Malformed position-matrix element in block {n}.")
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                if not (0 <= i < m_size and 0 <= j < m_size):
                    raise ValueError(f"Position-matrix index out of range in block {n}.")
                position_blocks[n, i, j, 0] = complex(float(parts[2]), float(parts[3]))
                position_blocks[n, i, j, 1] = complex(float(parts[4]), float(parts[5]))
                position_blocks[n, i, j, 2] = complex(float(parts[6]), float(parts[7]))
            parsed_positions = True

    if not parsed_positions:
        position_blocks = None

    return WannierTBData(
        lattice_vectors=lattice,
        r_integer_vectors=r_integer_vectors,
        degeneracies=degeneracies,
        hopping_blocks=hopping_blocks,
        position_blocks=position_blocks,
    )


def r_cartesian(tb_data: WannierTBData) -> np.ndarray:
    return tb_data.r_integer_vectors @ tb_data.lattice_vectors


def build_hk(tb_data: WannierTBData, kpoint: np.ndarray, k_units: str) -> np.ndarray:
    if k_units == "cartesian":
        phases = np.exp(1j * (r_cartesian(tb_data) @ kpoint))
    elif k_units == "fractional":
        phases = np.exp(1j * 2.0 * np.pi * (tb_data.r_integer_vectors @ kpoint))
    else:
        raise ValueError("--k-units must be 'cartesian' or 'fractional'.")

    weights = phases / tb_data.degeneracies.astype(float)
    return np.einsum("r,rij->ij", weights, tb_data.hopping_blocks, optimize=True)


def fix_eigenvector_phases(eigenvectors: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return eigenvectors
    if mode != "xatu":
        raise ValueError("--phase-fix must be 'xatu' or 'none'.")

    fixed = np.array(eigenvectors, dtype=np.complex128, copy=True)
    for band in range(fixed.shape[1]):
        column = fixed[:, band]
        reference = np.sum(column)
        if abs(reference) > 1e-14:
            column *= np.exp(-1j * np.angle(reference))
        fixed[:, band] = column
    return fixed


def diagonalize_all_unique_kpoints(
    tb_data: WannierTBData,
    kpoints: np.ndarray,
    k_units: str,
    phase_fix: str,
    hermitian_tolerance: float = 1e-8,
) -> Dict[KKey, np.ndarray]:
    eigvecs_by_k: Dict[KKey, np.ndarray] = {}
    for kpoint in kpoints:
        key = k_key(kpoint)
        if key in eigvecs_by_k:
            continue

        hk = build_hk(tb_data, np.asarray(kpoint, dtype=float), k_units)
        anti = hk - hk.conj().T
        anti_norm = np.linalg.norm(anti)
        if anti_norm > hermitian_tolerance:
            warn(
                f"H(k) at {key} has anti-Hermitian norm {anti_norm:.3e}; "
                "symmetrizing before diagonalization."
            )
        hk = 0.5 * (hk + hk.conj().T)

        _, eigvecs = np.linalg.eigh(hk)
        eigvecs_by_k[key] = fix_eigenvector_phases(eigvecs, phase_fix)

    return eigvecs_by_k


def unique_kpoint_count(kpoints: np.ndarray) -> int:
    return len({k_key(kpoint) for kpoint in kpoints})


def infer_k_grid_size_from_states(states_data: StatesData, cell_dims: int) -> int:
    nk = unique_kpoint_count(states_data.kpoints)
    if cell_dims == 1:
        return nk

    root = round(nk ** (1.0 / cell_dims))
    if root > 0 and root**cell_dims == nk:
        return root

    fallback = max(1, 2 * int(np.ceil(nk ** (1.0 / cell_dims))) + 1)
    warn(
        f"Could not infer Xatu ncell exactly from {nk} unique k-points and "
        f"cell_dims={cell_dims}; using fallback ncell={fallback} for row ordering."
    )
    return fallback


def translate_home(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Match Xatu System::translateHomeCell for row-vector Cartesian inputs."""
    frac = np.asarray(cart, dtype=float) @ np.linalg.inv(lattice)
    frac = frac - np.floor(frac)
    return frac @ lattice


def get_wannier_centers(tb_data: WannierTBData) -> np.ndarray:
    if tb_data.position_blocks is None:
        raise ValueError(
            "tb.dat does not contain position matrices; no fallback positions are available."
        )

    matches = np.where(np.all(tb_data.r_integer_vectors == 0, axis=1))[0]
    if matches.size == 0:
        raise ValueError("Could not find R=(0,0,0) block in Wannier90 tb.dat.")
    diag_block = int(matches[0])

    m_size = tb_data.hopping_blocks.shape[1]
    centers = np.zeros((m_size, 3), dtype=float)
    for i in range(m_size):
        centers[i] = np.real(tb_data.position_blocks[diag_block, i, i, :])
    return translate_home(centers, tb_data.lattice_vectors)


def generate_xatu_centered_cell_grid(nvalues: int, cell_dims: int) -> np.ndarray:
    if cell_dims not in (1, 2, 3):
        raise ValueError("--cell-dims must be 1, 2, or 3.")
    if nvalues <= 0:
        raise ValueError("Xatu ncell must be positive.")

    ncombinations = nvalues**cell_dims
    shift = nvalues // 2
    ones = np.ones(nvalues, dtype=int)
    combinations = np.zeros((ncombinations, 3), dtype=int)

    # Literal translation of Lattice::generateCombinations:
    #   values = regspace(0, nvalues - 1) - shift
    #   repeat kron(ones, values) for dimensions to the left
    #   repeat kron(values, ones) for dimensions to the right
    for dim in range(cell_dims):
        values = np.arange(nvalues, dtype=int) - shift
        for _ in range(cell_dims - dim - 1):
            values = np.kron(ones, values)
        for _ in range(dim):
            values = np.kron(values, ones)
        combinations[:, dim] = values

    return combinations


def get_xatu_rswf_cell_offsets(
    states_data: StatesData,
    lattice: np.ndarray,
    ncells: int,
    cell_dims: int,
) -> List[np.ndarray]:
    """Match ResultTB .rswf cell list: truncateSupercell(exciton->ncell, |a1| * ncells)."""
    if cell_dims not in (1, 2, 3):
        raise ValueError("--cell-dims must be 1, 2, or 3.")
    if ncells < 0:
        raise ValueError("--ncells must be non-negative.")

    # Xatu uses two different cell counts here:
    #   - the k-grid size comes from the unique k-points in .states
    #   - --ncells is the RSWF real-space radius parameter, not a box half-width
    xatu_ncell = infer_k_grid_size_from_states(states_data, cell_dims)
    radius = np.linalg.norm(lattice[0]) * ncells
    offsets: List[np.ndarray] = []
    for offset in generate_xatu_centered_cell_grid(xatu_ncell, cell_dims):
        lattice_vector = offset @ lattice
        if np.linalg.norm(lattice_vector) < radius + 1e-5:
            offsets.append(offset)
    return offsets


def validate_band_indices(states_data: StatesData, tb_data: WannierTBData) -> None:
    m_size = tb_data.hopping_blocks.shape[1]
    if np.any(states_data.valence_bands < 0) or np.any(states_data.conduction_bands < 0):
        raise ValueError("Negative band index found after --band-index-base conversion.")
    max_band = int(max(states_data.valence_bands.max(), states_data.conduction_bands.max()))
    if max_band >= m_size:
        raise ValueError(
            f"Band index {max_band} exceeds Hamiltonian dimension {m_size}."
        )


def validate_hole_indices(tb_data: WannierTBData, hole_indices: Sequence[int]) -> None:
    m_size = tb_data.hopping_blocks.shape[1]
    for fixed_hole_index in hole_indices:
        if not (0 <= fixed_hole_index < m_size):
            raise ValueError(f"hole_index {fixed_hole_index} is outside [0, {m_size}).")


def n_states_to_write(states_data: StatesData, n_states: Optional[int]) -> int:
    if n_states is None:
        return states_data.states.shape[0]
    if n_states <= 0:
        raise ValueError("-n/--n-states must be a positive integer.")
    if n_states > states_data.states.shape[0]:
        raise ValueError(
            f"Requested {n_states} states, but .states contains only "
            f"{states_data.states.shape[0]}."
        )
    return n_states


def phase_for_offsets(
    kpoints: np.ndarray,
    cell_delta: np.ndarray,
    lattice: np.ndarray,
    k_units: str,
) -> np.ndarray:
    if k_units == "cartesian":
        d_r = cell_delta @ lattice
        return np.exp(1j * (kpoints @ d_r))
    if k_units == "fractional":
        return np.exp(1j * 2.0 * np.pi * (kpoints @ cell_delta))
    raise ValueError("--k-units must be 'cartesian' or 'fractional'.")


def ordered_output_rows(
    states_data: StatesData,
    tb_data: WannierTBData,
    centers: np.ndarray,
    hole_cell: Sequence[int],
    ncells: int,
    cell_dims: int,
) -> List[Tuple[np.ndarray, int, np.ndarray]]:
    """Return Xatu's row order as (cell_offset, site_index, electron_position)."""
    del hole_cell  # Xatu writes electron positions from cellCombinations, not holeCell offsets.

    ordered_rows: List[Tuple[np.ndarray, int, np.ndarray]] = []
    for offset in get_xatu_rswf_cell_offsets(
        states_data, tb_data.lattice_vectors, ncells, cell_dims
    ):
        for site_index in range(tb_data.hopping_blocks.shape[1]):
            electron_position = offset @ tb_data.lattice_vectors + centers[site_index]
            ordered_rows.append((offset, site_index, electron_position))
    return ordered_rows


def compute_rswf_for_state(
    states_data: StatesData,
    tb_data: WannierTBData,
    state_index: int,
    fixed_hole_index: int,
    hole_cell: Sequence[int],
    k_units: str,
    wannier_centers: np.ndarray,
    eigvecs_for_basis: Sequence[np.ndarray],
    ordered_rows: Sequence[Tuple[np.ndarray, int, np.ndarray]],
) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, np.ndarray]]:
    if not (0 <= state_index < states_data.states.shape[0]):
        raise ValueError(
            f"state_index {state_index} is outside [0, {states_data.states.shape[0]})."
        )

    m_size = tb_data.hopping_blocks.shape[1]
    exciton_coefficients = states_data.states[state_index]

    prefactors = np.zeros((m_size, states_data.basis_dim), dtype=np.complex128)
    for j in range(states_data.basis_dim):
        c_band = states_data.conduction_bands[j]
        v_band = states_data.valence_bands[j]
        eigvecs = eigvecs_for_basis[j]
        hole_v = np.conj(eigvecs[fixed_hole_index, v_band])
        prefactors[:, j] = exciton_coefficients[j] * eigvecs[:, c_band] * hole_v

    hole_cell_arr = np.asarray(hole_cell, dtype=int)
    hole_position = hole_cell_arr @ tb_data.lattice_vectors + wannier_centers[fixed_hole_index]

    rows: List[Tuple[float, float, float, float]] = []
    current_offset: Optional[Tuple[int, int, int]] = None
    current_probabilities: Optional[np.ndarray] = None
    for offset, site_index, electron_position in ordered_rows:
        offset_key = tuple(int(item) for item in offset)
        if offset_key != current_offset:
            cell_delta = offset - hole_cell_arr
            phases = phase_for_offsets(
                states_data.kpoints, cell_delta, tb_data.lattice_vectors, k_units
            )
            amplitudes = prefactors @ phases
            current_probabilities = np.abs(amplitudes) ** 2
            current_offset = offset_key

        if current_probabilities is None:
            raise RuntimeError("Internal error: probabilities were not initialized.")
        probability = current_probabilities[site_index]
        rows.append(
            (
                float(electron_position[0]),
                float(electron_position[1]),
                float(electron_position[2]),
                float(probability),
            )
        )

    metadata = {
        "hole_position": hole_position.astype(float),
        "centers": wannier_centers,
        "state_index": np.asarray(state_index, dtype=int),
    }
    return rows, metadata


def compute_rswf_blocks_for_hole(
    states_data: StatesData,
    tb_data: WannierTBData,
    fixed_hole_index: int,
    hole_cell: Sequence[int],
    k_units: str,
    n_states: Optional[int] = None,
    wannier_centers: Optional[np.ndarray] = None,
    eigvecs_for_basis: Optional[Sequence[np.ndarray]] = None,
    ordered_rows: Optional[Sequence[Tuple[np.ndarray, int, np.ndarray]]] = None,
) -> List[Tuple[List[Tuple[float, float, float, float]], Dict[str, np.ndarray]]]:
    n_state_blocks = n_states_to_write(states_data, n_states)
    if wannier_centers is None:
        wannier_centers = get_wannier_centers(tb_data)
    if eigvecs_for_basis is None:
        raise ValueError("eigvecs_for_basis must be precomputed.")
    if ordered_rows is None:
        raise ValueError("ordered_rows must be precomputed.")

    blocks = []
    for state_index in range(n_state_blocks):
        blocks.append(
            compute_rswf_for_state(
                states_data=states_data,
                tb_data=tb_data,
                state_index=state_index,
                fixed_hole_index=fixed_hole_index,
                hole_cell=hole_cell,
                k_units=k_units,
                wannier_centers=wannier_centers,
                eigvecs_for_basis=eigvecs_for_basis,
                ordered_rows=ordered_rows,
            )
        )
    return blocks


def default_output_path(states_path: str, hole_index: int) -> str:
    path = Path(states_path)
    return str(path.with_name(f"{path.stem}_hole_{hole_index}.rswf"))


def output_path_for_hole(
    output_arg: Optional[str],
    states_path: str,
    fixed_hole_index: int,
    multiple_holes: bool,
) -> str:
    if output_arg is None:
        return default_output_path(states_path, fixed_hole_index)
    if not multiple_holes:
        return output_arg

    path = Path(output_arg)
    if path.suffix == ".rswf":
        return str(path.with_name(f"{path.stem}_hole_{fixed_hole_index}{path.suffix}"))
    return str(path.with_name(f"{path.name}_hole_{fixed_hole_index}.rswf"))


def write_rswf_blocks(
    path: str,
    blocks: Sequence[Tuple[Sequence[Tuple[float, float, float, float]], Dict[str, np.ndarray]]],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for rows, metadata in blocks:
            hole_position = metadata["hole_position"]
            handle.write(
                f"{hole_position[0]:11.8f}\t{hole_position[1]:11.8f}\t"
                f"{hole_position[2]:11.8f}\n"
            )
            for x, y, z, probability in rows:
                handle.write(
                    f"{x:11.8f}\t{y:11.8f}\t{z:11.8f}\t{probability:14.11f}\n"
                )
            handle.write("#\n")


def read_rswf_blocks(path: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    hole_position: Optional[np.ndarray] = None
    coords: List[List[float]] = []
    probabilities: List[float] = []

    def finish_block() -> None:
        nonlocal hole_position, coords, probabilities
        if hole_position is None and not coords and not probabilities:
            return
        if hole_position is None:
            raise ValueError(f"Found .rswf block without a hole row in {path}.")
        blocks.append(
            (
                hole_position.copy(),
                np.asarray(coords, dtype=float),
                np.asarray(probabilities, dtype=float),
            )
        )
        hole_position = None
        coords = []
        probabilities = []

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
                raise ValueError(f"Could not parse numeric row {line_number} in {path}.") from exc

            if hole_position is None:
                if len(values) < 3:
                    raise ValueError(f"Malformed hole row at {path}:{line_number}.")
                hole_position = np.asarray(values[:3], dtype=float)
                continue

            if len(values) < 4:
                raise ValueError(
                    f"Expected electron-density row with at least 4 columns at "
                    f"{path}:{line_number}."
                )
            coords.append(values[:3])
            probabilities.append(values[-1])

    finish_block()
    if not blocks:
        raise ValueError(f"No .rswf blocks found in file: {path}")
    return blocks


def lexsort_coords(coords: np.ndarray) -> np.ndarray:
    return np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))


def compare_rswf(generated_path: str, reference_path: str) -> None:
    coordinate_tolerance = 1e-7
    generated_blocks = read_rswf_blocks(generated_path)
    reference_blocks = read_rswf_blocks(reference_path)

    if len(generated_blocks) != len(reference_blocks):
        raise ValueError(
            "Cannot compare .rswf files with different numbers of state blocks "
            f"({len(generated_blocks)} vs {len(reference_blocks)}). The files "
            "were likely generated with different numbers of states."
        )

    row_order_matches = True
    coordinate_sets_match = True
    max_coord_rowwise = 0.0
    max_prob_rowwise = 0.0
    max_prob_sorted = 0.0

    for block_index, (generated_block, reference_block) in enumerate(
        zip(generated_blocks, reference_blocks)
    ):
        generated_hole, generated_coords, generated_prob = generated_block
        reference_hole, reference_coords, reference_prob = reference_block

        if generated_coords.shape[0] != reference_coords.shape[0]:
            raise ValueError(
                "Cannot compare .rswf files with different row counts in block "
                f"{block_index}: {generated_coords.shape[0]} vs "
                f"{reference_coords.shape[0]}. The files were likely generated "
                "with different ncells or grid settings."
            )

        hole_diff = float(np.max(np.abs(generated_hole - reference_hole)))
        max_coord_rowwise = max(max_coord_rowwise, hole_diff)

        coord_diff = generated_coords - reference_coords
        prob_diff = generated_prob - reference_prob
        if coord_diff.size:
            max_coord_rowwise = max(max_coord_rowwise, float(np.max(np.abs(coord_diff))))
        if prob_diff.size:
            max_prob_rowwise = max(max_prob_rowwise, float(np.max(np.abs(prob_diff))))

        block_row_order_matches = bool(
            np.allclose(
                generated_hole,
                reference_hole,
                rtol=0.0,
                atol=coordinate_tolerance,
            )
            and np.allclose(
                generated_coords,
                reference_coords,
                rtol=0.0,
                atol=coordinate_tolerance,
            )
        )
        row_order_matches = row_order_matches and block_row_order_matches

        generated_order = lexsort_coords(generated_coords)
        reference_order = lexsort_coords(reference_coords)
        generated_sorted_coords = generated_coords[generated_order]
        reference_sorted_coords = reference_coords[reference_order]
        block_coordinate_set_matches = bool(
            np.allclose(
                generated_sorted_coords,
                reference_sorted_coords,
                rtol=0.0,
                atol=coordinate_tolerance,
            )
        )
        coordinate_sets_match = coordinate_sets_match and block_coordinate_set_matches

        sorted_prob_diff = generated_prob[generated_order] - reference_prob[reference_order]
        if sorted_prob_diff.size:
            max_prob_sorted = max(max_prob_sorted, float(np.max(np.abs(sorted_prob_diff))))

    print(f"number of blocks compared: {len(generated_blocks)}")
    print(f"row_order_matches: {'yes' if row_order_matches else 'no'}")
    print(
        "coordinate_sets_match_after_sort: "
        f"{'yes' if coordinate_sets_match else 'no'}"
    )
    print(f"max coordinate difference rowwise: {max_coord_rowwise:.12e}")
    print(f"max probability difference rowwise: {max_prob_rowwise:.12e}")
    print(
        "max probability difference after coordinate sorting: "
        f"{max_prob_sorted:.12e}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Xatu-like real-space exciton density from .states and Wannier90 tb.dat."
    )
    parser.add_argument("--states", required=True, help="Path to the Xatu .states file.")
    parser.add_argument("--tb", required=True, help="Path to the Wannier90 seedname_tb.dat file.")
    parser.add_argument(
        "-r",
        "--hole-index",
        required=True,
        help="Fixed hole Wannier site/orbital index list, e.g. 0,2,8-10. Zero-based.",
    )
    parser.add_argument(
        "-n",
        "--n-states",
        type=int,
        default=None,
        help="Number of exciton states to write from the beginning. Default: all states.",
    )
    parser.add_argument(
        "--hole-cell",
        nargs=3,
        type=int,
        metavar=("I", "J", "K"),
        default=(0, 0, 0),
        help="Integer unit cell where the hole is fixed.",
    )
    parser.add_argument(
        "--ncells",
        type=int,
        default=5,
        help="Xatu-style real-space radius in units of |a1|.",
    )
    parser.add_argument(
        "--cell-dims",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Number of periodic directions to sample.",
    )
    parser.add_argument("--output", help="Output .rswf path.")
    parser.add_argument(
        "--k-units",
        choices=("cartesian", "fractional"),
        default="cartesian",
        help="Units of k-points stored in .states.",
    )
    parser.add_argument(
        "--band-index-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Band index base used by the .states basis block.",
    )
    parser.add_argument(
        "--phase-fix",
        choices=("xatu", "none"),
        default="xatu",
        help="Deterministic phase convention for rediagonalized eigenvectors.",
    )
    parser.add_argument(
        "--compare-rswf",
        help="Optional reference .rswf file to compare against after writing output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    states_data = parse_states(args.states, band_index_base=args.band_index_base)
    tb_data = parse_wannier_tb(args.tb)
    hole_indices = parse_index_list(args.hole_index)
    multiple_holes = len(hole_indices) > 1

    if multiple_holes and args.compare_rswf:
        raise ValueError("--compare-rswf can only be used with a single hole index.")

    validate_band_indices(states_data, tb_data)
    validate_hole_indices(tb_data, hole_indices)
    n_state_blocks = n_states_to_write(states_data, args.n_states)

    wannier_centers = get_wannier_centers(tb_data)
    eigvecs_by_k = diagonalize_all_unique_kpoints(
        tb_data,
        states_data.kpoints,
        args.k_units,
        args.phase_fix,
    )
    eigvecs_for_basis = [eigvecs_by_k[k_key(k)] for k in states_data.kpoints]
    output_rows = ordered_output_rows(
        states_data,
        tb_data,
        wannier_centers,
        args.hole_cell,
        args.ncells,
        args.cell_dims,
    )

    for fixed_hole_index in hole_indices:
        blocks = compute_rswf_blocks_for_hole(
            states_data=states_data,
            tb_data=tb_data,
            fixed_hole_index=fixed_hole_index,
            hole_cell=args.hole_cell,
            k_units=args.k_units,
            n_states=n_state_blocks,
            wannier_centers=wannier_centers,
            eigvecs_for_basis=eigvecs_for_basis,
            ordered_rows=output_rows,
        )

        output = output_path_for_hole(
            args.output,
            args.states,
            fixed_hole_index,
            multiple_holes,
        )
        write_rswf_blocks(output, blocks)

        rows_per_block = len(blocks[0][0]) if blocks else 0
        print(
            f"Wrote hole {fixed_hole_index}: {len(blocks)} state block(s), "
            f"{rows_per_block} electron-density rows per block, to {output}"
        )

    if args.compare_rswf:
        compare_rswf(output, args.compare_rswf)

    print(f"Processed {len(hole_indices)} hole index/indices.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"postprocess_rswf.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
