# hamiltonian.py
"""
Hamiltonian builder for Wannier tight-binding models.

Given a WannierTB object (from io_wannier.parse_tb_file) and
k-points (fractional or Cartesian), returns the Bloch Hamiltonian H(k).

Uses vectorized phase summation:
    H(k) = sum_R H(R) * exp(i k * R)
where R = iRn[alpha,0]*a1 + iRn[alpha,1]*a2 + iRn[alpha,2]*a3.

The user can request single k-point evaluation or batches.

Usage:
    from io_wannier import parse_tb_file
    from kmesh import monkhorst_pack
    from hamiltonian import bands

    tb = parse_tb_file('my_tb.dat')
    mesh = monkhorst_pack(tb.bravais_vectors, nk=(10,10,1))

    eigvals = bands(tb, mesh.frac_kpts)
    print("Valence top:", eigvals.max())

"""

from __future__ import annotations
import numpy as np
from typing import Literal
from io_wannier import WannierTB
from kmesh import reciprocal_lattice, frac_to_cart_k


# ------------------------------------------------------------
# Utility: precompute real-space lattice vectors R
# ------------------------------------------------------------

def realspace_R(tb: WannierTB) -> np.ndarray:
    """
    Compute real-space lattice vectors R (in Cartesian units)
    from iRn and bravais vectors.

    Returns
    -------
    R : (nFock,3) array
    """
    return np.dot(tb.iRn, tb.bravais_vectors)


# ------------------------------------------------------------
# Core: build H(k)
# ------------------------------------------------------------

def Hk(tb: WannierTB,
       k: np.ndarray,
       space: Literal["frac", "cart"] = "frac",
       cache_R: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the Bloch Hamiltonian H(k) for a given k-point.

    Parameters
    ----------
    tb : WannierTB
        Parsed tight-binding model.
    k : (3,) array
        k-vector (fractional or Cartesian depending on 'space').
    space : {"frac","cart"}
        If "frac", k is given in fractional reciprocal coordinates.
        If "cart", k is in Cartesian (1/length) units.
    cache_R : optional (nFock,3)
        Precomputed real-space lattice vectors (Cartesian) for speed.

    Returns
    -------
    Hk : (mSize,mSize) complex ndarray
        Hamiltonian matrix at k.
    """
    if cache_R is None:
        R = realspace_R(tb)
    else:
        R = cache_R

    # convert fractional to Cartesian if needed
    if space == "frac":
        b = reciprocal_lattice(tb.bravais_vectors)
        k_cart = np.dot(k, b)  # (3,)
    elif space == "cart":
        k_cart = np.asarray(k, dtype=float)
    else:
        raise ValueError("space must be 'frac' or 'cart'")

    # Phase factors e^{i k*R}
    phase = np.exp(1j * np.dot(R, k_cart))
    # Weighted sum over all real-space matrices
    # Broadcasting: (nFock,1,1) * (nFock,m,m)
    Hk = np.tensordot(phase, tb.H, axes=(0, 0))
    return Hk


# ------------------------------------------------------------
# Batch evaluation for a list of k-points
# ------------------------------------------------------------

def Hk_batch(tb: WannierTB,
             kpts: np.ndarray,
             space: Literal["frac","cart"] = "frac",
             dtype=np.complex128) -> np.ndarray:
    """
    Compute H(k) for many k-points.

    Parameters
    ----------
    kpts : (N,3) array
        List of k-points.
    space : {"frac","cart"}
        Coordinate convention.
    dtype : np.dtype
        Complex precision.

    Returns
    -------
    Hks : (N,mSize,mSize) complex array
    """
    R = realspace_R(tb)
    if space == "frac":
        b = reciprocal_lattice(tb.bravais_vectors)
        k_cart = np.dot(kpts, b)
    else:
        k_cart = kpts

    nFock, mSize = tb.nFock, tb.mSize
    Hks = np.zeros((len(kpts), mSize, mSize), dtype=dtype)

    # Vectorized accumulation: sum_R H(R) * exp(i k*R)
    # For memory reasons, do in blocks if nFock is large
    for n in range(nFock):
        phase = np.exp(1j * np.dot(k_cart, R[n]))  # (N,)
        Hks += phase[:, None, None] * tb.H[n]

    return Hks


# ------------------------------------------------------------
# Eigenvalues (band structure)
# ------------------------------------------------------------

def bands(tb: WannierTB,
          kpts: np.ndarray,
          space: Literal["frac","cart"] = "frac",
          eigvals_only: bool = True,
          sort: bool = True) -> np.ndarray:
    """
    Diagonalize H(k) at each k-point.

    Returns
    -------
    eigvals : (N, mSize)
    or
    eigvals, eigvecs : (N,mSize), (N,mSize,mSize)
    """
    Hks = Hk_batch(tb, kpts, space=space)
    N, m = len(kpts), tb.mSize
    eigvals = np.zeros((N, m), dtype=float)
    if not eigvals_only:
        eigvecs = np.zeros((N, m, m), dtype=np.complex128)

    for i in range(N):
        w, v = np.linalg.eigh(Hks[i])
        if sort:
            order = np.argsort(w)
            w = w[order]
            v = v[:, order]
        eigvals[i] = w
        if not eigvals_only:
            eigvecs[i] = v

    if eigvals_only:
        return eigvals
    return eigvals, eigvecs
