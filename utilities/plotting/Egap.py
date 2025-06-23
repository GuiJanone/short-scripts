#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


sp_file = "field_band.dat"

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

# Sort data by field value
field = np.array(field)
Egap = np.array(Egap)
sorted_idx = np.argsort(field)
field = field[sorted_idx]
Egap = Egap[sorted_idx]
print(f"Original Gap: {np.max(Egap)}")

# Plot
fig, ax = plt.subplots()
ax.plot(field*1000, Egap, marker="o", c='k')
ax.set_xlabel(r"Field intensity (meV/$\AA$)")
ax.set_ylabel(r"Gap Energy (eV)")
plt.grid(True)

plt.tight_layout()
plt.savefig("bandgap_vs_field.png", dpi=600)
plt.show()
