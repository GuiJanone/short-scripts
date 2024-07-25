import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pauli Matrices
sig_x = np.array([[0+0j,1+0j],[1+0j,0+0j]])
sig_y = np.array([[0+0j,0-1j],[0+1j,0+0j]])
sig_z = np.array([[1+0j,0+0j],[0+0j,-1+0j]])

def mount_H(kx,ky):
  k = np.array([kx,ky])

  # Parameters of first Neighbor
  t1 = 1
  a1 = np.array([1,0])
  a2 = np.array([-1/2, (np.sqrt(3)/2)])   # remove this for armchair?
  a3 = np.array([-1/2, -(np.sqrt(3)/2)])  # remove this for zigzag?
  a = np.array([a1,a2,a3])


  first_neig = 0
  for i in range(len(a)):
    first_neig += (sig_x * np.cos(np.dot(k,a[i]))) - (sig_y * np.sin(np.dot(k,a[i])))
  first_neig = t1 * first_neig


  # Parameters of second Neighbor
  mass = 0.2
  M = mass*sig_z
  # t2 = mass/(3*np.sqrt(3))+0.05
  t2 = 0.03
  b1 = np.array([3/2, -np.sqrt(3)/2])
  b2 = np.array([-3/2, -np.sqrt(3)/2])
  b3 = np.array([0,-np.sqrt(3)])
  b = np.array([b1,b2,b3])

  second_neig = 0
  for i in range(len(b)):
    second_neig +=  sig_z * np.sin(np.dot(k,b[i]))
  second_neig = 2*t2 * second_neig


  H = first_neig #+ M + second_neig
  return H


def plot_dispersion3D(n,BZlim):
    # Generate a mesh
    kx_range = np.linspace(-BZlim, BZlim, num=n)
    ky_range = np.linspace(-BZlim, BZlim, num=n)

    # Get the number of levels with a dummy call (an NxN square matrix has N levels)
    num_levels = len(mount_H(0,0))
    # energies = np.zeros((m,n,num_levels)); # initialize
    Psi1 = np.zeros((n,n,num_levels), dtype=complex)
    Psi2 = np.zeros((n,n,num_levels), dtype=complex)
    energies = np.zeros((n,n,num_levels), dtype=complex)
    # Now iterate over discretized mesh, to consider each coordinate.
    for i in range(n):
      for j in range(n):
        # k = np.array([kx_range[i],ky_range[j]])
        H = mount_H(kx_range[i],ky_range[j]);
        evals, evec = LA.eig(H) # Numerically get eigenvalues and eigenvectors
        energies[i,j,:]=evals
        Psi1[i,j] = evec[0] # CBM
        Psi2[i,j] = evec[1] # VBM

    return energies,Psi1,Psi2,kx_range

BZrange = np.pi
res = 101
E, PsiC,PsiV, kx_range = plot_dispersion3D(res,BZrange)

X, Y = np.meshgrid(kx_range, kx_range) # Generate actual mesh for plotting.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=120, roll=0)

for i in range(2):
  ax.plot_surface(X, Y, E[:,:,i], alpha=0.9)

plt.show()