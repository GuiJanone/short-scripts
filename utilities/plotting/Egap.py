#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys


sp_file = "field_band.dat"
outfile = "bandgap_vs_field.dat"

if len(sys.argv) > 0:
    filling = int(sys.argv[1])
else:
    filling = int(input("Insert number of filled bands:"))

Egap = []
field = []

 # 1) Pick up folder names that parse as floats (including negatives)
folders = []
for f in os.listdir():
    if os.path.isdir(f):
        try:
            _ = float(f)
            folders.append(f)
        except ValueError:
            continue
folders = sorted(folders, key=lambda x: float(x))

for f in folders:
    path = os.path.join(f, sp_file)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            # Remove blank lines and strip whitespace
            lines = [line.strip() for line in lines if line.strip()]
            # Get 26th line (index 25)
            vb = float(lines[filling-1].split()[1])
            cb = float(lines[filling].split()[1])
            val = cb - vb
            Egap.append(val)
            field.append(float(f))
    except Exception as e:
        print(f"Error reading {path}: {e}")


print(f"Writing outfile '{outfile}'...")
with open(outfile, "w") as f:
    f.write('fieldIntensity     Egap\n')
    for i in range(len(Egap)):
        f.write(f' {field[i]:.5f}    {Egap[i]:.5f}\n')

print("DONE.")
print("Plotting...")
# Sort data by field value
field      = np.array(field)
Egap       = np.array(Egap)

sorted_idx = np.argsort(field)
field      = field[sorted_idx]
Egap       = Egap[sorted_idx]
size = len(field)

deriv = np.gradient(Egap)

slope, intercept, = linregress(field[int(size/2)+1:], Egap[int(size/2)+1:])[:2]
fit = slope*field[int(size/2)+1:] + intercept

print(f"Original Gap = {np.max(Egap)} eV")
print("---------LINEAR FIT---------")
print(f"Intercept = {intercept:.4f}")
print(f"Slope = {slope:.4f}")

# Plot
fig, ax = plt.subplots(2,1, height_ratios=[2, 1], sharex=True)
ax[0].plot(field*1000, Egap, marker="o", c='k')
# ax[0].plot(field[int(size/2)+1:]*1000, fit, c='red')

ax[1].plot(field*1000, deriv, c='red')
ax[1].set_xlabel(r"Field intensity (meV/$\AA$)")

ax[0].set_ylabel(r"Gap Energy (eV)")
ax[1].set_ylabel(r"$d E_{\text{gap}}/d E_{DC}$")

# ax[0].set_xlim(-15, 15)
# ax[1].set_xlim(-15, 15)
# ax[0].set_ylim(1.5, 1.75)
# ax[1].set_ylim(-0.01, 0.01)

ax[0].grid(True)
ax[1].grid(True)

plt.tight_layout()
plt.savefig("bandgap_vs_field.png", dpi=600)
plt.show()
