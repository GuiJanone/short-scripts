#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

data = np.loadtxt(sys.argv[1])

energy = data[:, 0]
s_xxx  = data[:, 1]
s_xyy  = data[:, 2]
s_yyy  = data[:, 3]
s_yxx  = data[:, 4]
# s_zxx  = data[:, 5]
# s_zyy  = data[:, 6]
# s_zzz  = data[:, 7]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(energy, s_xxx, label=f"xxx", alpha=0.7, lw=2.5)
ax.plot(energy, s_xyy, label=f"xyy", alpha=0.7, lw=2.5)
ax.plot(energy, s_yyy, label=f"yyy", alpha=0.7, lw=2.5)
ax.plot(energy, s_yxx, label=f"yxx", alpha=0.7, lw=2.5)
# ax.plot(energy, s_zxx, label=f"zxx", alpha=0.7, lw=2.5)
# ax.plot(energy, s_zyy, label=f"zyy", alpha=0.7, lw=2.5)
# ax.plot(energy, s_zzz, label=f"zzz", alpha=0.7, lw=2.5)

ax.set_xlabel("E (eV)", fontsize=18)
ax.set_ylabel(r"$\sigma^{(2)}\ (\mu$A/V$^2 \cdot nm)$", fontsize=18)
ax.set_xlim(energy[0], energy[-1])

ax.legend()
plt.tight_layout()
plt.savefig("shift_current.png", dpi=600)
plt.show()