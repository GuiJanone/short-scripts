#!/usr/bin/env python3
"""
Build a finite slab tight-binding Hamiltonian from a Wannier90 seedname_tb.dat file.

Initial target:
    z-oriented slab, finite along a3, periodic along a1 and a2.

The slab Hamiltonian uses the enlarged basis:
    I = m + l * mSize

If the input has an overlap section (non-orthogonal basis), the slab overlap is
built with the same layer map and written out. Input matrices are divided by
their degeneracies on load, so the slab file is written with degeneracy 1.

No external/static confinement potential is added here.
"""

from __future__ import annotations

import argparse

import numpy as np

from w90tb import TbData, read_tb, write_tb


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


def print_tb_diagnostics(tb: TbData) -> None:
    print("Parsed bulk TB:")
    print(f"  m_size = {tb.m_size}")
    print(f"  n_fock = {tb.n_fock}")
    print(f"  has WF block = {tb.wf is not None}")
    print(f"  has overlap = {tb.overlap is not None}")
    print(f"  diag_index = {tb.diag_index}")
    print("  R range:")
    print(f"      Rx {np.min(tb.i_rn[:, 0])} / {np.max(tb.i_rn[:, 0])}")
    print(f"      Ry {np.min(tb.i_rn[:, 1])} / {np.max(tb.i_rn[:, 1])}")
    print(f"      Rz {np.min(tb.i_rn[:, 2])} / {np.max(tb.i_rn[:, 2])}")


def diagonalize_gamma(
    slab_i_rn: np.ndarray,
    slab_h: np.ndarray,
    bravais: np.ndarray,
    slab_s: np.ndarray | None = None,
) -> np.ndarray:
    """Gamma eigenvalues; solves H c = E S c via Cholesky when an overlap is given."""
    k0 = np.zeros(3, dtype=float)
    h_gamma = build_hk_2d(slab_i_rn, slab_h, bravais, k0)
    if slab_s is not None:
        s_gamma = build_hk_2d(slab_i_rn, slab_s, bravais, k0)
        chol = np.linalg.cholesky(s_gamma)
        y = np.linalg.solve(chol, h_gamma)
        h_gamma = np.linalg.solve(chol, y.conj().T).conj().T
    return np.linalg.eigvalsh(h_gamma)


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

    tb = read_tb(args.input).normalized()
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

    slab_s = None
    if tb.overlap is not None:
        _, _, slab_s = build_slab_hoppings(
            bravais=tb.bravais,
            i_rn=tb.i_rn,
            h_r=tb.overlap,
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

    if slab_wf is None and (args.write_empty_wf or args.write_centers_wf or slab_s is not None):
        # zero PME block, optionally with slab centers on the R=(0,0) diagonal
        slab_wf = np.zeros((len(slab_i_rn), slab_h.shape[1], slab_h.shape[2], 3), dtype=complex)
        if args.write_centers_wf and centers is not None:
            home = np.where(np.all(slab_i_rn == 0, axis=1))[0]
            if len(home) == 0:
                raise ValueError("Cannot write WF centers: slab R=(0,0,0) block not found.")
            slab_wf[home[0], np.arange(len(centers)), np.arange(len(centers))] = centers

    write_tb(
        args.output,
        TbData(
            header="Slab tight-binding model generated from bulk seedname_tb.dat",
            bravais=bravais_slab,
            degen=np.ones(len(slab_i_rn), dtype=int),
            i_rn=slab_i_rn,
            h_r=slab_h,
            wf=slab_wf,
            overlap=slab_s,
        ),
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
        eigvals = diagonalize_gamma(slab_i_rn, slab_h, bravais_slab, slab_s)

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
