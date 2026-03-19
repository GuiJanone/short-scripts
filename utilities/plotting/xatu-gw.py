import numpy as np
import matplotlib.pyplot as plt

# Constants (SI)
EPS0 = 8.8541878128e-12       # F/m
HBAR_EVS = 6.582119569e-16    # eV*s (for omega = E/HBAR_EVS)
E_CHARGE = 1.602176634e-19    # C
HBAR_SI = 1.054571817e-34     # J*s

# Quantum of conductance in these units: e^2/hbar (Siemens)
E2_OVER_HBAR = (E_CHARGE ** 2) / HBAR_SI  # S

# --- inputs ---
eps_path = "/home/visitor/Projects/in2se3/GW/nb-4-1/absorption_computed_eh_broad_0.1.dat"
xatu_path = "/home/visitor/atomelix/projects/In2Se3/MLWF/sigma_first_ex_real_in2se3.dat"
# xatu_path = "/home/visitor/Projects/in2se3/chen2025/atomelix/CWF/in2se3_ex.dat"

# Supercell height used in the GW/BSE calculation (Angstrom)
Lz_ang = 50.0
Lz_m = Lz_ang * 1e-10

# --- load absorption spectrum ---
eps_arr = np.loadtxt(eps_path)
E_eV = eps_arr[:, 0]
eps2 = eps_arr[:, 1]  # Im eps_M

omega = E_eV / HBAR_EVS  # rad/s

# SI 3D conductivity (S/m)
sigma3D_S_per_m = EPS0 * omega * eps2

# Convert to 2D sheet conductance (S)
sigma2D_S = sigma3D_S_per_m * Lz_m

# Express in XATU units (e^2/hbar)
sigma_from_eps2_xatu_units = sigma2D_S / E2_OVER_HBAR  # dimensionless

# --- load xatu conductivity ---
arr = np.loadtxt(xatu_path)
E2_eV, sigma_xatu = arr[:, 0], arr[:, 1]  # assume already in e^2/hbar

plt.figure()
plt.plot(E2_eV+0.55, sigma_xatu, label="XATU (Re sigma) [e^2/hbar]")
plt.plot(E_eV, sigma_from_eps2_xatu_units, label="GW/BSE from eps2 [e^2/hbar]")
plt.xlabel("Energy (eV)")
plt.ylabel(r"$\mathrm{Re}\,\sigma^{2D}$ ($e^2/\hbar$)")
plt.legend()
plt.tight_layout()
plt.savefig("conductivity.png", dpi=300)
# plt.show()
