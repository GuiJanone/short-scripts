'''
	As shown in Fu2018 Supp. Info.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import integrate
import scipy.interpolate as interp

def NormalizeData(data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))   

def findEfficiency(eta,gap):
	for i in range(len(eta)):
			ind = np.abs(gapE[i] - gap)
			if ind < 1e-2:
				eff = eta[i]
				print(f'Absolut efficiency for gap of {gapE[i]:.3f} = {100*eta[i]:.3f} %')
	return eff

h=4.13566743e-15     #in eV.s
c=299792458           #in m/s

filePath = './AM0AM1_5.txt'
wavelength, Et, Global, Direct = np.loadtxt(filePath, unpack=True, usecols=(0,1,2,3))

Solar = Direct

energy =(h*c)/(wavelength*1e-9)

gapE =(h*c)/(wavelength[::-1]*1e-9)
SolarR = Solar[::-1]

DeltaG = 1.23 # eV

gap = 1.47

#==================ABSOLUT ETA NORM========================#


totalPower = integrate.trapz(SolarR/gapE, x=gapE)
print(f'TotalPower: {totalPower}')

absorbtion = np.zeros(len(energy))

for i in range(len(energy)):
	# dE = (energy[-1] - energy[i])/len(energy[i:])
	absorbtion[i] = integrate.trapz(SolarR[i:]/gapE[i:], x=gapE[i:])

eta_abs = (absorbtion)/totalPower

for i in range(len(eta_abs)):
		ind = np.abs(gapE[i] - gap)
		if ind < 1e-2:
			eff = eta_abs[i]
			print(f'Absolut efficiency for gap of {gapE[i]:.3f} = {100*eta_abs[i]:.3f} %')


# plt.plot(energy, eta_abs)
# plt.plot(gapE, eta_abs)
# plt.scatter(gap, eff, c='red', zorder=1,label=f'Efficiency = {eff:.2f}')
# plt.title(r'$\eta_{Abs}$', fontsize=20)
# plt.xlabel('Gap Energy')
# plt.show()

#==================ETA CARRIER UTILZIATION========================#
over_eff = np.zeros(len(energy))
gap_eff = np.zeros(len(energy))
eta_cu = np.zeros(len(energy))

overpot = gap + 0.6 - 0.37 # gap + 0.6 - overport(O2)

for i in range(len(gapE)):
	overpot = gapE[i] + 0.6 - 0.37 # gap + 0.6 - overport(O2)
	for j in range(i,len(gapE)):
		indx =  overpot - gapE[j]

		if indx < 1e-2 and indx >= 0:
			# print(gapE[j],overpot)
			print(f'Im in with {gapE[i]} at  index {i} and overpot {gapE[j]} at {j}')
			overLim = j
			break
		else:
			overLim = -1

	over_eff[i] = integrate.trapz(SolarR[overLim:]/gapE[overLim:], x=gapE[overLim:])
	gap_eff[i] = integrate.trapz(SolarR[i:], x=gapE[i:])

	if over_eff[i] == 0:
		over_eff[i:] == 0
		break

eta_cu = (DeltaG * over_eff)/gap_eff

# plt.plot(gapE, eta_cu)
# plt.title('Carrier Utilization', fontsize=20)
# plt.xlabel('Gap Energy', fontsize=18)
# plt.ylabel(r'$\eta_{cu}$', fontsize=18)
# plt.show()

#==================ETA STH = abs x cu ========================#

eta_STH = eta_abs * eta_cu


aGe = findEfficiency(eta_STH, gap)

#================== WITHOUT OVERPOT ========================#
for i in range(len(gapE)):
	over_eff[i] = integrate.trapz(SolarR[i:]/gapE[i:], x=gapE[i:])
	gap_eff[i] = integrate.trapz(SolarR[i:], x=gapE[i:])

eta_cu = (DeltaG * over_eff)/gap_eff
eta_STH = eta_abs * eta_cu

In2Te3 	= 43.4
Ga2Se3 	= 28.4
Ga2S3 	= 6.9
Al2Te3 	= 28.4
Al2Se3 	= 9.1
Simono = 9.8

fig,ax = plt.subplots(figsize=(12,12))

ax.plot(gapE, eta_STH*100, c='black', zorder=0,lw=4)

#--------------------- POINTS ---------------------#

ax.scatter(gap, aGe*100, c='red', zorder=1,label=rf'$\alpha-Ge(111)$ = {aGe*100:.2f}%', lw=0, s=14**2)
offset = (gap-1, aGe*100 - 0.7)
ax.annotate(rf'$\alpha-Ge(111)$',(gap, aGe*100),xytext=offset,fontsize=30)

ax.scatter(1.30, In2Te3, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (1.30-0.5, In2Te3)
ax.annotate(rf'$In_2 Te_3$',(1.30, In2Te3),xytext=offset, fontsize=30)

ax.scatter(1.92, Ga2Se3, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (1.92+0.1, Ga2Se3 +1.8)
ax.annotate(rf'$Ga_2Se_3$',(1.92, Ga2Se3),xytext=offset, fontsize=30)

ax.scatter(2.77, Ga2S3, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (2.77+0.1, (Ga2S3)-0.8)
ax.annotate(rf'$Ga_2S_3$',(2.77, (Ga2S3)),xytext=offset, fontsize=30)

ax.scatter(1.91, Al2Te3, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (1.91+0.2, Al2Te3-1.8)
ax.annotate(rf'$Al_2Te_3$',(1.91, Al2Te3),xytext=offset, fontsize=30)

ax.scatter(2.60, Al2Se3, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (2.60+0.1, Al2Se3+0.5)
ax.annotate(rf'$Al_2Se_3$',(2.60, (Al2Se3)),xytext=offset, fontsize=30)

ax.scatter(3.264, Simono, c='black', zorder=1, marker='^', lw=0, s=18**2)
offset = (3.264, Simono+1.2)
ax.annotate('Si PV cell',(3.264, (Simono)),xytext=offset, fontsize=30)

#--------------------- ---- ---------------------#

ax.set_ylabel(r'$\eta_{STH}\ (\%)$ ', fontsize=45)
ax.set_xlabel('Gap Energy (eV)', fontsize=45)

ax.legend(fontsize=34)

ax.set_xlim(np.min(gapE),4)
ax.set_ylim(0,65)

ax.tick_params(axis='both',length = 12,width = 2)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.setp(ax.spines.values(), lw=2, color='black');

plt.grid(True,alpha=0.5)
plt.tight_layout()
plt.savefig('STH_classical.pdf', dpi=600)
plt.show()

#==================ETA STH 2.0 ========================#

abs_norm = np.zeros(len(wavelength))

totalPower = integrate.trapz(SolarR/gapE, x=gapE)

for i in range(len(wavelength)):
	abs_norm[i] = integrate.trapz(SolarR[i:]/gapE[i:], x=gapE[i:])

eta2 = eta_STH * abs_norm /totalPower  * 100

for i in range(len(eta2)):
		ind = np.abs(gapE[i] - gap)
		if ind < 1e-2:
			eff = eta2[i]
			print(f'Absolut efficiency for gap of {gapE[i]:.3f} = {100*eta2[i]:.3f} %')

print(f'Max efficiency = {np.max(eta2[:-1])}')

# plt.plot(gapE, eta2, zorder=0)
# plt.scatter(gap, eff, c='red', zorder=1,label=f'Efficiency = {eff:.2f}')
# plt.ylabel(r"""$\eta_{STH}'$ """, fontsize=18)
# plt.xlabel('Gap Energy', fontsize=18)
# plt.legend(fontsize=12)
# plt.show()
