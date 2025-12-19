import numpy as np
import matplotlib.pyplot as plt

EPS0 = 8.8541878128e-12  # F/m
HBAR_EVS = 6.582119569e-16  # eV*s

def load_two_col(path):
    # expects: E_eV  value
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns")
    return arr[:, 0], arr[:, 1]

# --- inputs ---
abs_path = "/home/visitor/Projects/in2se3/GW/nb-4-1/absorption_computed_eh_broad_0.1.dat"
cond_path = "/home/visitor/Projects/in2se3/chen2025/atomelix/CWF/in2se3_ex.dat"  # <-- change to your file

# If your conductivity is 2D sheet conductivity, set Lz_m = supercell height in meters.
# If your conductivity is already 3D (S/m), set Lz_m = None.
Lz_m = 30e-10  # e.g. 60 Ang -> 60e-10 m

# --- load absorption spectrum ---
abs_arr = np.loadtxt(abs_path)
E_eV = abs_arr[:, 0]
eps2 = abs_arr[:, 1]  # use col2 for the conversion

omega = E_eV / HBAR_EVS  # rad/s

sigma_from_eps2 = EPS0 * omega * eps2  # S/m (3D)
if Lz_m is not None:
    sigma_from_eps2 = sigma_from_eps2 * Lz_m  # S (2D sheet)

# --- load your conductivity data ---
E2_eV, sigma_user = load_two_col(cond_path)

# --- plot ---
plt.figure()
plt.plot(E2_eV, sigma_user, label="Your conductivity (Re sigma)")
plt.plot(E_eV, sigma_from_eps2, label="From absorption: eps2 -> Re sigma")
plt.xlabel("Energy (eV)")
plt.ylabel("Re sigma (S/m) or sheet S (if multiplied by Lz)")
plt.legend()
plt.tight_layout()
plt.show()
