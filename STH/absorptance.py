import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

#====================== Adsorption Data ======================#
hbar 	= 6.582e-16 		# eV s
h 		= 4.13566743e-15    #in eV.s
c 		= 2.998e8           #in m/s
d 		= 4.04e-10 			# thicknes of 4.04  Angs in meters

filePath = './absorption_noeh.dat'
omega, ep2, ep1 = np.loadtxt(filePath, unpack=True, usecols=(0,1,2))

# plt.plot(omega,ep2,label=r'$\varepsilon_2$')
# plt.plot(omega,ep1,label=r'$\varepsilon_1$')
# plt.legend()
# plt.show()

#====================== Adsorption Coefficient ======================#
'''
  As in Renan2024 - Fresnel Law for atom-thick films
'''
n1 = 1.000293 # air
n2 = 1.69     # flint glass
L = 8     # thickness in m
c = 299792458 # in m/s
c=3


epsilon = ep1 - 1j*ep2

sigma = ((1j*omega*L)/(4*np.pi)) * (epsilon-1)

T = n2 * np.abs(2/(n1+n2+((4*np.pi*sigma)/c)))**2
R = np.abs((n1-n2-((4*np.pi*sigma)/c))/(1+n2+((4*np.pi*sigma)/c)))**2
A = 1-R-T

# plt.plot(omega,epsilon)
# plt.plot(omega,sigma,label='sigma')
plt.plot(omega,T, label='T')
plt.plot(omega,R, label='R')
plt.plot(omega,A, label='A')
plt.legend()
# plt.yticks([0,0.5,1])
# plt.ylim(0.80,4.75)
plt.show()

# ============== SOLAR SPECTRUM ================== #
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))   

h=4.13566743e-15     #in eV.s
c=299792458           #in m/s

filePath = filePath = './AM0AM1_5.txt'
wavelength, Et, Global, Direct = np.loadtxt(filePath, unpack=True, usecols=(0,1,2,3))

Solar = Et

energy =(h*c)/(wavelength*1e-9)

gapE =(h*c)/(wavelength[::-1]*1e-9)
SolarR = Solar[::-1]

DeltaG = 1.23 # eV
gap = 1.49

#================= adjusting data to fit solar ==========================#
A_inter = interp.PchipInterpolator(np.arange(A.size),A)
A_stretch = A_inter(np.linspace(0,A.size-1,Solar.size))


omega_inter = interp.PchipInterpolator(np.arange(omega.size),omega)
omega_stretch = omega_inter(np.linspace(0,omega.size-1,Solar.size))

A_short = np.zeros(len(energy))
omega_short = np.zeros(len(energy))
i = 0
while omega_stretch[i] <= 4:
    A_short[i] = A_stretch[i]
    omega_short[i] = omega_stretch[i]
    i +=1

A_norm = NormalizeData(A_short)
Solar_norm = NormalizeData(Solar)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(omega_stretch, A_norm, c='red',label=r'Absorptance')
ax.plot(energy, Solar_norm, c='black', label='Solar Spectrum')
ax.set_xlim(0,4)

ax.set_xlabel('Energy (eV)',fontsize=12)
ax.set_xlabel('Energy (eV)', fontsize=16)
ax.set_ylabel('Intensity (%)', fontsize=16)
plt.legend()
plt.show()