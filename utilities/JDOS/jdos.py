# jdos.py
"""
Joint Density of States (JDOS) calculator.

Implements the fundamental definition:
    J(omega) = (1/Nk) * sum_{v,c,k} delta_eta(Ec(k) - Ev(k) - hbar*omega)

The delta function is replaced by a Gaussian or Lorentzian broadening.

This module depends on:
    - io_wannier.WannierTB
    - kmesh.KMesh
    - hamiltonian.bands
"""

from __future__ import annotations
import numpy as np
from typing import Literal
from io_wannier import WannierTB
from kmesh import KMesh
from hamiltonian import bands


# ------------------------------------------------------------
# Delta functions with broadening
# ------------------------------------------------------------

def delta_gaussian(x: np.ndarray, eta: float) -> np.ndarray:
    """Normalized Gaussian broadening."""
    return np.exp(-(x / eta) ** 2) / (np.sqrt(np.pi) * eta)


def delta_lorentzian(x: np.ndarray, eta: float) -> np.ndarray:
    """Normalized Lorentzian broadening."""
    return (eta / np.pi) / (x ** 2 + eta ** 2)


# ------------------------------------------------------------
# JDOS computation
# ------------------------------------------------------------

def jdos(tb: WannierTB,
         mesh: KMesh,
         omega_grid: np.ndarray,
         eta: float,
         broadening: Literal["gaussian", "lorentzian"] = "gaussian",
         valence_bands: list[int] | None = None,
         conduction_bands: list[int] | None = None,
         eigvals: np.ndarray | None = None,
         verbose: bool = True) -> np.ndarray:
    """
    Compute the JDOS for a set of photon energies.

    Parameters
    ----------
    tb : WannierTB
        Tight-binding model.
    mesh : KMesh
        k-point mesh (from kmesh.monkhorst_pack).
    omega_grid : array
        Photon energies (same units as eigenvalues, e.g., eV).
    eta : float
        Broadening width (same energy units).
    broadening : {"gaussian","lorentzian"}
        Type of delta function broadening.
    valence_bands, conduction_bands : list of int, optional
        Indices of valence and conduction bands (0-based).
        If None, they are guessed by Fermi level (middle of gap).
    eigvals : optional (Nk, m)
        Precomputed eigenvalues. If None, they are computed here.
    verbose : bool
        Print progress info.

    Returns
    -------
    J : (N_omega,) array
        Joint density of states in arbitrary units.
    """

    Nk = len(mesh.frac_kpts)
    mSize = tb.mSize

    if eigvals is None:
        if verbose:
            print("Diagonalizing Hamiltonians...")
        eigvals = bands(tb, mesh.frac_kpts, space="frac")

    # Identify valence / conduction bands if not given
    if valence_bands is None or conduction_bands is None:
        # crude guess: split in half
        mid = mSize // 2
        if valence_bands is None:
            valence_bands = list(range(mid))
        if conduction_bands is None:
            conduction_bands = list(range(mid, mSize))

    if verbose:
        print(f"Using {len(valence_bands)} valence and {len(conduction_bands)} conduction bands")

    # Choose delta function
    if broadening.lower().startswith("g"):
        delta_fn = delta_gaussian
    elif broadening.lower().startswith("l"):
        delta_fn = delta_lorentzian
    else:
        raise ValueError("broadening must be 'gaussian' or 'lorentzian'")

    # Prepare energy grid
    omega_grid = np.asarray(omega_grid, dtype=float)
    J = np.zeros_like(omega_grid)

    # Weighted sum over k and band pairs
    weights = mesh.weights
    hbar_omega = omega_grid  # same units as eigvals

    if verbose:
        print("Accumulating JDOS histogram...")

    for ik in range(Nk):
        Ev = eigvals[ik, valence_bands]
        Ec = eigvals[ik, conduction_bands]
        # Build all energy differences Ec - Ev
        diff = Ec[:, None] - Ev[None, :]
        diff = diff.ravel()
        for dE in diff:
            J += weights[ik] * delta_fn(hbar_omega - dE, eta)

    # Normalize by number of k-points (weights already sum to 1)
    return J


# ------------------------------------------------------------
# Convenience wrapper with automatic omega grid
# ------------------------------------------------------------

def jdos_auto(tb: WannierTB,
              mesh: KMesh,
              eta: float = 0.01,
              n_omega: int = 1000,
              energy_range: tuple[float, float] | None = None,
              **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience version that builds an automatic omega grid.

    Returns
    -------
    omega_grid, J
    """
    if "eigvals" in kwargs and kwargs["eigvals"] is not None:
        eigvals = kwargs["eigvals"]
    else:
        eigvals = bands(tb, mesh.frac_kpts)

    E_min, E_max = eigvals.min(), eigvals.max()

    if energy_range is None:
        energy_range = (0.0, E_max - E_min)

    omega_grid = np.linspace(energy_range[0], energy_range[1], n_omega)
    J = jdos(tb, mesh, omega_grid, eta, eigvals=eigvals, **kwargs)
    return omega_grid, J
