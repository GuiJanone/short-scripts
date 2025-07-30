import matplotlib.pyplot as plt
import numpy as np
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

#===============================================================================#
# zero vs field 
#===============================================================================#
fig, ax = plt.subplots(figsize=(8, 5))

data = np.loadtxt(file1)
energy = data[:, 0]
y_data1 = data[:, 1]

data = np.loadtxt(file2)
energy2 = data[:, 0]
y_data2 = data[:, 1]

ax.plot(energy, y_data1, label=f"No field", color='r')
ax.plot(energy2, y_data2, label=f"0.01", color='g')

ax.set_xlabel(r"$E_{photon}$ (eV)", fontsize=18)
ax.set_ylabel(r"$\sigma^{(2)}_{xxx}\ (\mu$A/V$^2 \cdot nm)$", fontsize=18)
# ax.set_xlim(1.5, 4.0)
# ax.set_ylim(-25, 25)
ax.set_xticks(np.arange(1.5, 4.1, 0.5))
ax.tick_params(axis='x', which='minor', length=7)
# ax.set_xticks(np.arange(1.5, 3.1, 0.25))

ax.legend(title=r"Field (eV/\AA)", fontsize=12)
plt.tight_layout()
plt.savefig("field_activation.png", dpi=400)