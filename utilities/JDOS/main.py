#!/usr/bin/env python3
# main.py
"""
Main driver to compute the Joint Density of States (JDOS)
from a Wannier tight-binding model.

Usage (example):
    python main.py --tb my_tb.dat --nk 40 40 1 --eta 0.02 --n_omega 800

Output:
    - jdos.dat : two columns (omega  J(omega))
    - jdos.png : plot of JDOS
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from io_wannier import parse_tb_file
from kmesh import monkhorst_pack
from hamiltonian import bands
from jdos import jdos
from io_input import read_input, map_bands_relative

def main():
    # read JSON config
    cfg = read_input(sys.argv[1])

    tb = parse_tb_file(cfg.tb_file)
    mesh = monkhorst_pack(tb.bravais_vectors, nk=cfg.nk,
                          gamma_centered=cfg.gamma_centered,
                          shift=cfg.shift)

    # diagonalize once
    eigvals = bands(tb, mesh.frac_kpts, space="frac")

    # map user band selections
    v_idx, c_idx = map_bands_relative(cfg.n_occ, tb.mSize,
                                      cfg.valence_bands,
                                      cfg.conduction_bands)
    print("Valence bands (abs idx):", v_idx)
    print("Conduction bands (abs idx):", c_idx)

    # build omega grid
    if cfg.energy_range is None:
        e_min = eigvals.min()
        e_max = eigvals.max()
        energy_range = (0.0, e_max - e_min)
    else:
        energy_range = cfg.energy_range
    omega = np.linspace(energy_range[0], energy_range[1], cfg.n_omega)

    # compute JDOS using only requested bands
    J = jdos(tb, mesh, omega, cfg.eta,
             broadening=cfg.broadening,
             valence_bands=v_idx,
             conduction_bands=c_idx,
             eigvals=eigvals,
             verbose=True)

    np.savetxt("jdos.dat", np.column_stack([omega, J]),
               header="omega  JDOS")

    # Determine whether to draw the band gap line
    gap_energy = getattr(cfg, "gap_energy", None)
    if gap_energy is not None:
        print(f"Using user-defined band gap: {gap_energy:.3f} eV")

    plot_jdos(omega, J, gap_energy=gap_energy)


def plot_jdos(omega, J, gap_energy=None, save_name="jdos.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(omega, J, color="black")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("JDOS (arb. units)")

    ax.set_xlim(float(np.min(omega)), float(np.max(omega)))

    if gap_energy is not None:
        ax.axvline(gap_energy, color="gray", ls="--", lw=1.0, label=f"Gap = {gap_energy:.2f} eV")
        ax.legend(frameon=False)

    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_name, dpi=400)
    print(f"Plot saved to {save_name}")



if __name__ == "__main__":
    main()
