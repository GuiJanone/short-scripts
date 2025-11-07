# kmesh.py
"""
K-space mesh utilities (Monkhorst-Pack, Gamma-centered, tetrahedra connectivity).

Works with direct (Bravais) lattice vectors in Cartesian coordinates (rows are a1,a2,a3).
Returns fractional k-points (in reciprocal-lattice units) and Cartesian k-points (1/length).
Weights are uniform by default (sum to 1).

If nkz == 1, you effectively get a 2D mesh; if nky == nkz == 1, it's 1D.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ---------- Linear algebra helpers ----------

def reciprocal_lattice(direct: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors (rows are b1,b2,b3) given direct lattice (rows a1,a2,a3).
    Conventions: b_i * a_j = 2pi delta_ij
    """
    a = np.asarray(direct, dtype=float)
    vol = np.dot(a[0], np.cross(a[1], a[2]))
    if abs(vol) < 1e-12:
        raise ValueError("Direct lattice volume is ~0; check bravais vectors.")
    b1 = 2.0 * np.pi * np.cross(a[1], a[2]) / vol
    b2 = 2.0 * np.pi * np.cross(a[2], a[0]) / vol
    b3 = 2.0 * np.pi * np.cross(a[0], a[1]) / vol
    return np.vstack([b1, b2, b3])


def bz_volume(recip: np.ndarray) -> float:
    """Volume of the first Brillouin zone (|det of reciprocal primitive cell|)."""
    return abs(np.dot(recip[0], np.cross(recip[1], recip[2])))


def frac_to_cart_k(frac_k: np.ndarray, recip: np.ndarray) -> np.ndarray:
    """
    Convert fractional k (in reciprocal-lattice units) to Cartesian (1/length):
    k_cart = k_frac_x * b1 + k_frac_y * b2 + k_frac_z * b3
    """
    return np.dot(frac_k, recip)  # (N,3) @ (3,3)


def cart_to_frac_k(cart_k: np.ndarray, recip: np.ndarray) -> np.ndarray:
    """Convert Cartesian k to fractional (reciprocal-lattice units)."""
    return np.dot(cart_k, np.linalg.inv(recip))


def wrap_frac_k(frac_k: np.ndarray) -> np.ndarray:
    """Map fractional k to the canonical cell [-0.5, 0.5) along each axis."""
    x = np.asarray(frac_k, dtype=float)
    return (x + 0.5) % 1.0 - 0.5


# ---------- Monkhorst-Pack ----------

def _mp_axis_points(n: int, gamma_centered: bool) -> np.ndarray:
    """
    1D Monkhorst-Pack points along one axis in fractional coords (in units of b_i).
    For Gamma-centered: symmetric around 0. For standard MP: shifted by 0.5/n.
    """
    if n <= 0:
        raise ValueError("Grid size must be >= 1")
    if n == 1:
        return np.array([0.0])
    if gamma_centered:
        # Gamma-centered: points at (i - (n-1)/2)/n, i=0..n-1  -> [-0.5, 0.5) symmetric
        return (np.arange(n) - 0.5 * (n - 1)) / n
    else:
        # Original MP: (2i - n - 1)/(2n), i=1..n
        i = np.arange(1, n + 1)
        return (2.0 * i - n - 1.0) / (2.0 * n)


@dataclass
class KMesh:
    """Container for a uniform k-mesh."""
    nk: Tuple[int, int, int]          # (nkx, nky, nkz)
    gamma_centered: bool
    shift: Tuple[float, float, float] # extra fractional shift to add (default (0,0,0))
    frac_kpts: np.ndarray             # (N,3) fractional coords in reciprocal units
    cart_kpts: np.ndarray             # (N,3) Cartesian coords (1/length)
    weights: np.ndarray               # (N,) normalized to sum to 1
    direct: np.ndarray                # (3,3) direct lattice (a1,a2,a3)
    recip: np.ndarray                 # (3,3) reciprocal lattice (b1,b2,b3)
    bz_vol: float                     # Brillouin-zone volume


def monkhorst_pack(direct: np.ndarray,
                   nk: Tuple[int, int, int],
                   gamma_centered: bool = True,
                   shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   wrap: bool = True) -> KMesh:
    """
    Build a uniform Monkhorst-Pack mesh.

    Parameters
    ----------
    direct : (3,3) array
        Direct (Bravais) lattice vectors as rows (Cartesian units).
    nk : tuple of ints
        (nkx, nky, nkz). Use 1 for effectively reduced dimensions.
    gamma_centered : bool
        If True, Gamma-centered mesh. If False, standard Monkhorst-Pack shift.
    shift : tuple of floats
        Extra fractional shift to add to all k-points BEFORE wrapping
        (useful to avoid Gamma when nk is even, etc.).
    wrap : bool
        If True, wrap fractional points into [-0.5, 0.5).

    Returns
    -------
    KMesh
    """
    a = np.asarray(direct, dtype=float)
    b = reciprocal_lattice(a)
    vol_bz = bz_volume(b)

    nkx, nky, nkz = map(int, nk)
    gx = _mp_axis_points(nkx, gamma_centered)
    gy = _mp_axis_points(nky, gamma_centered)
    gz = _mp_axis_points(nkz, gamma_centered)

    # Grid and optional extra shift
    kx, ky, kz = np.meshgrid(gx, gy, gz, indexing="ij")
    frac = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    if shift is not None:
        s = np.asarray(shift, dtype=float)
        if s.shape != (3,):
            raise ValueError("shift must be length-3 fractional vector")
        frac = frac + s

    if wrap:
        frac = wrap_frac_k(frac)

    cart = frac_to_cart_k(frac, b)

    N = frac.shape[0]
    weights = np.full(N, 1.0 / N, dtype=float)

    return KMesh(
        nk=(nkx, nky, nkz),
        gamma_centered=gamma_centered,
        shift=(float(shift[0]), float(shift[1]), float(shift[2])),
        frac_kpts=frac,
        cart_kpts=cart,
        weights=weights,
        direct=a,
        recip=b,
        bz_vol=vol_bz,
    )


# ---------- Tetrahedra connectivity (optional, 3D) ----------

def tetrahedra_connectivity(nk: Tuple[int, int, int]) -> np.ndarray:
    """
    Connectivity for a regular 3D grid (nkx,nky,nkz) into tetrahedra.
    Returns an array of shape (Nt, 4) with indices into the flattened k-grid
    (flattening with order='C' on ijk indexing used here).

    Scheme: each cube cell is split into 6 tetrahedra with vertices:
    (0,0,0),(1,0,0),(0,1,0),(0,0,1) and permutations shifted to the cube origin.
    Periodic wrap is NOT applied (use only for integration within the primitive cell).
    """
    nkx, nky, nkz = map(int, nk)
    if nkx < 2 or nky < 2 or nkz < 2:
        raise ValueError("Tetrahedra need nkx,nky,nkz >= 2")

    def idx(i, j, k):
        return i * (nky * nkz) + j * nkz + k

    tets = []
    for i in range(nkx - 1):
        for j in range(nky - 1):
            for k in range(nkz - 1):
                # Corner indices of the cell
                c000 = idx(i,   j,   k)
                c100 = idx(i+1, j,   k)
                c010 = idx(i,   j+1, k)
                c110 = idx(i+1, j+1, k)
                c001 = idx(i,   j,   k+1)
                c101 = idx(i+1, j,   k+1)
                c011 = idx(i,   j+1, k+1)
                c111 = idx(i+1, j+1, k+1)
                # Split the cube into 6 tetrahedra (one common pattern)
                tets.extend([
                    [c000, c100, c010, c001],
                    [c100, c110, c010, c111],
                    [c100, c010, c001, c111],
                    [c100, c001, c101, c111],
                    [c010, c001, c011, c111],
                    [c100, c010, c111, c110],
                ])
    return np.asarray(tets, dtype=np.int64)


# ---------- 2D triangle connectivity (optional) ----------

def triangle_connectivity(nk: Tuple[int, int, int]) -> np.ndarray:
    """
    Connectivity for a regular 2D grid (nkx,nky, nkz must be 1) into triangles.
    Returns an array of shape (Nt, 3) with flattened indices.
    """
    nkx, nky, nkz = map(int, nk)
    if nkz != 1:
        raise ValueError("For triangle connectivity, set nkz == 1 (2D mesh).")
    if nkx < 2 or nky < 2:
        raise ValueError("Triangles need nkx,nky >= 2")

    def idx(i, j):
        return i * nky + j  # flatten (i,j), k=0 omitted

    tris = []
    for i in range(nkx - 1):
        for j in range(nky - 1):
            c00 = idx(i,   j)
            c10 = idx(i+1, j)
            c01 = idx(i,   j+1)
            c11 = idx(i+1, j+1)
            # two triangles per cell
            tris.append([c00, c10, c11])
            tris.append([c00, c11, c01])
    return np.asarray(tris, dtype=np.int64)
