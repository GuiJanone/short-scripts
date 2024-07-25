import numpy as np
import matplotlib.pyplot as plt

#===================== PARAMETERS =====================#

E = [1.23, 1.55, 1.63, 2.61]  # activation energies in eV
Particles = [1000000, 0, 0, 1000000] # number os particles available for each event
temp = np.arange(273.15, 973.15) # temperature in Kelvin
taxa = 60 # time in seconds

###########################################################

#===================== CONSTANTS =====================#
k = 8.617333262e-5 # Boltzman constant eV/K
g_0 = 10**13 # 1/s
massa_total = [[Particles[0]*3]]
#########################################################

#===================== METHODS ============================#

def getProbab(E, N, T):
  R_i = 0
  R = [0]
  for i in range(len(E)):
    gamma = g_0*(np.exp((-E[i]/(k*T)))) # função de partição
    R_i += gamma*N[i]
    R.append(R_i)
  R_tot = R[-1]
  return R, R_tot


def makeTransition(R, R_tot, N, M):
  for z in range(len(N)):
    if N[z] < 0:  # prevents negative number of particles
      N[z] = 0

  u = np.random.rand() # random number for monte-carlo

  massa = M[-1][-1]
  for i in range(1, len(R)):
    if R[i-1] < u*R_tot < R[i]:  # Monte-Carlo condition do happen
      if i == 1:
        # Defect
        N[0] -= (1 + 4)
        N[1] += 2
        N[2] += 2
        massa -= 0.5

      elif i == 2:
        # mono vacancy
        N[1] -= 2
        massa -= 1 # SO

      elif i == 3:
        # mono vacancy
        N[2] -= 2
        massa -= 1 # SO

      elif i == 4:
        # double vacancy
        N[3] -= (4 + 8)
        N[1] += 4
        N[2] += 4
        massa -= 2 # 2SO
  return massa

def tempo(R, t):
  u = np.random.rand()
  t_step =  t[-1] + np.log(1/u)/R # computes and add next time step
  t.append(t_step) # store time
  # print(f'R: {R}')

def massLoss(M):
  deltaM = []
  for i in range(len(M)):
    delta = (M[i][-1]/M[0][0]) * 100
    deltaM.append(delta)
  return deltaM

def executar(massa_total, particles, taxa):
  N = particles.copy()
  M = massa_total.copy()
  massa = 0

  for i in range(len(temp)-1):
    t=[0]

    while t[-1] < taxa: # rate of heating
      R, R_tot = getProbab(E, N, temp[i]) 

      if R_tot <= 0: #failsafe
          break;

      massa = makeTransition(R, R_tot, N, M)
      M[i].append(massa) # store new mass after event

      tempo(R_tot, t)

    M.append([massa]) # for every Temperature, a list of Masses

    deltaM = massLoss(M)
    # print(f'Checking process for temp {temp[i]}')
  return M, deltaM

######################################################################

#========================= Execution ========================#
massa_60, delta_60 = executar(massa_total, Particles, 60)
massa_300, delta_300 = executar(massa_total, Particles, 300)
massa_600, delta_600 = executar(massa_total, Particles, 600)

######################################################################

temp_celsius = np.subtract(temp, 273.15) # Kelvin to Celcius

#============================ Gráfico ============================#
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(temp_celsius, delta_60, label = r"$1^o$ /min", c = '#AA8CD4', linewidth=1.5)
ax.plot(temp_celsius, delta_300, label = r"$5^o$ /min", c = '#721EE8', linewidth=1.5)
ax.plot(temp_celsius, delta_600, label = r"$10^o$ /min", c = '#533182', linewidth=1.5)

ax.set_xlim(0, 700)
ax.tick_params(axis='both', which='major',
               direction ='in', labelsize=16,
               length=6, width=2,
               top = True, right = True)

ax.set_xlabel(r"Temperature ($^o$ C)", fontsize= 20)
ax.set_ylabel(r'$\Delta m$ (%)', fontsize= 20)
ax.set_title(r'Loss of mass for $Ti S_{3} O_{0.5}$', fontsize = 20)
plt.legend(frameon=False, prop={'size': 16})


plt.savefig('massaXtemp-defeito.png', bbox_inches='tight')
plt.show()
