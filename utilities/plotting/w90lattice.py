#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

filename = sys.argv[1]
# bond_length = sys.argv[2]
def parse_file(filename):
    with open(filename, 'r') as file:
        # Skip the header
        for _ in range(1):
            next(file)

        # Read bravais vectors (a1, a2, a3)
        bravais_vectors = []
        for _ in range(3):
            line = next(file).strip()
            bravais_vectors.append([float(x) for x in line.split()])

        # Read nFock and mSize
        mSize = int(next(file).strip())
        nFock = int(next(file).strip())

        degen = np.zeros(nFock, dtype=int)
        degen_per_line = 15
        num_lines = nFock // degen_per_line  # Number of full lines
        remainder = nFock % degen_per_line  # Remaining integers

        # Read full lines of 15 integers
        for i in range(num_lines):
            line = next(file).strip()
            integers = list(map(int, line.split()))
            degen[i * degen_per_line : (i + 1) * degen_per_line] = integers

        # Read the remaining integers (if any)
        if remainder > 0:
            line = next(file).strip()
            integers = list(map(int, line.split()))
            degen[num_lines * degen_per_line : num_lines * degen_per_line + remainder] = integers[:remainder]

        next(file)
        # Initialize the first set of Fock matrices (nFock x mSize x mSize)
        H = np.zeros((nFock, mSize, mSize), dtype=complex)
        iRn = np.zeros((nFock, 3), dtype=int)

        # Parse the first set of nFock matrices
        for n in range(nFock):
            # Read the iRn indices (X, Y, Z)
            iRn[n] = [int(x) for x in next(file).strip().split()]
            if (iRn[n].any() == 0):
                diag = n
            # Read the mSize x mSize matrix
            for _ in range(mSize * mSize):
                line = next(file).strip()
                if line:  # Skip empty lines
                    i, j, real, imag = map(float, line.split())
                    H[n, int(i)-1, int(j)-1] = complex(real, imag)

            # Skip the blank line between matrices
            next(file)

        # Initialize the second set of Fock matrices for Rhop (nFock x mSize x mSize x 3)
        WF = np.zeros((nFock, mSize, mSize, 3), dtype=complex)

        # Parse the second set of nFock matrices (with X, Y, Z components)
        for n in range(nFock):
            line = next(file)
            iRn_tmp = line

            # Read the mSize x mSize matrix with X, Y, Z components -- ONLY DIAGONALS
            for _ in range(mSize * mSize):
                line = next(file).strip()
                if line:  # Skip empty lines
                    parts = list(map(float, line.split()))
                    i, j = int(parts[0]), int(parts[1])
                    if (i==j):          # get only diagonal
                        real_x, imag_x = parts[2], parts[3]
                        real_y, imag_y = parts[4], parts[5]
                        real_z, imag_z = parts[6], parts[7]
                        WF[n, i-1, j-1, 0] = complex(real_x, imag_x)  # X component
                        WF[n, i-1, j-1, 1] = complex(real_y, imag_y)  # Y component
                        WF[n, i-1, j-1, 2] = complex(real_z, imag_z)  # Z component

            # Skip the blank line between matrices
            if (n+1 == nFock):
                break;
            else:
                next(file)


    return np.array(bravais_vectors), nFock, mSize, iRn, H, WF, diag  # Second set of Fock matrices (nFock x mSize x mSize x 3)

#==============================================================#
# Create the plot
def map_to_real_space(iRn, bravais_vectors):
    """
    Map iRn indices to real-space positions using Bravais vectors.
    """
    return np.dot(iRn, bravais_vectors)

def plot_orbitals_Ncoded(iRn, bravais_vectors, WF):
    """
    Plot the localization of orbitals within each unit cell, coloring all elements in each n block with the same color.
    """
    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)

    # Extract orbital positions from WF
    orbital_positions = WF[:, :, :, :2]  # Use only X and Y components for 2D
    n_blocks = WF.shape[0]  # Number of 'n' blocks

    # Get a set of distinct colors for each block
    cmap = cm.get_cmap('prism', n_blocks)  # Use a qualitative colormap with n_blocks distinct colors

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Iterate over blocks and assign a single color to each block
    for i in range(n_blocks):
        block_color = cmap(i)  # Get a distinct color for this block
        points = []

        for j in range(orbital_positions.shape[1]):
            for k in range(orbital_positions.shape[2]):
                pos = real_space_positions[i, :2] + orbital_positions[i, j, k, :2]
                points.append(pos)

        points = np.array(points)

        # Scatter plot for the block
        ax.scatter(points[:, 0], points[:, 1], c=[block_color], s=4**2, alpha=0.6)
        ax.scatter(points[0, 0], points[0, 1], c='black', marker='x', s=2**2)

    ax.scatter(points[0, 0], points[0, 1], c='black', marker='x', s=2**2, label='r=0')
    ax.set_facecolor('gainsboro')
    # Labels and formatting
    ax.set_xlabel(f'X ($\\AA$)',fontsize=16)
    ax.set_ylabel(f'Y ($\\AA$)',fontsize=16)
    ax.set_aspect('equal')

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig("motif.png", dpi=300)
    plt.show()

def plot_orbitals_indexCoded(iRn, bravais_vectors, WF, n_index):
    """
    Plot a single n block with a gradient color mapping.
    
    Parameters:
        iRn: Real-space lattice positions.
        bravais_vectors: Bravais lattice vectors.
        WF: Wavefunction positions array (n, i, j, 3).
        n_index: Index of the n block to plot.
    """
    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)

    # Extract orbital positions from WF for the chosen block
    orbital_positions = WF[n_index, :, :, :2]  # Only X and Y components for 2D

    # Flatten the data for plotting
    points = []
    for j in range(orbital_positions.shape[0]):
        for k in range(orbital_positions.shape[1]):
            pos = real_space_positions[n_index, :2] + orbital_positions[-j, -k, :2]
            points.append(pos)

    points = np.array(points)
    
    # Generate gradient colors based on an index (e.g., sequential order)
    color_values = np.linspace(0, 1, points.shape[0])  # Normalize from 0 to 1
    cmap = cm.get_cmap('viridis')  # Choose a gradient colormap
    colors = cmap(color_values)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot with gradient colors
    sc = ax.scatter(points[:, 0], points[:, 1], c=color_values, cmap=cmap, s=8**2, alpha=0.6)
    ax.scatter(points[0, 0], points[0, 1], c='black', marker='x', s=10**2, label='r=0')

    # Colorbar to show the gradient meaning
    cbar = plt.colorbar(sc, fraction=0.040, pad=0.04, ax=ax)

    ax.set_facecolor('gainsboro')
    # Labels and formatting
    ax.set_xlabel(f'X ($\\AA$)',fontsize=16)
    ax.set_ylabel(f'Y ($\\AA$)',fontsize=16)
    ax.set_title(f'motif for $R = 0,0,0$ cell',fontsize=16)
    # ax.set_aspect('equal')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("motif_center.png", dpi=300)
    plt.show()


def plot_orbital_localizations_with_z(iRn, bravais_vectors, WF, n_index):
    """
    Plot a single n block of orbital localizations in 2D (X vs. Y) with the Z coordinate
    represented by a colormap.
    
    Parameters:
        iRn: Lattice translation indices.
        bravais_vectors: The Bravais lattice vectors.
        WF: Array with orbital localization information, shape (n, i, j, 3),
            where the last index holds [X, Y, Z] positions.
        n_index: Index of the n block to plot.
    """
    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)
    
    # Extract orbital positions from WF for the selected block (all X, Y, and Z)
    orbital_positions = WF[n_index, :, :, :]  # shape (n_orbitals_1, n_orbitals_2, 3)
    
    # Create lists to accumulate points and their corresponding Z values
    points = []
    z_values = []
    
    # For each orbital in the block, compute its full position in the unit cell.
    for j in range(orbital_positions.shape[0]):
        for k in range(orbital_positions.shape[1]):
            # Compute the absolute position by adding the real-space position offset (X,Y) of the block
            pos_xy = real_space_positions[n_index, :2] + orbital_positions[j, k, :2]
            points.append(pos_xy)
            # The Z coordinate is taken directly from the orbital data
            z_values.append(orbital_positions[j, k, 2])
    
    points = np.array(points)
    z_values = np.array(z_values)
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8,8))

    # ax.scatter(0, bond_length, marker='X', c='black') # PLOTTING THIS COMPLETELY BREAKS YTICKS??!!

    sc = ax.scatter(points[:, 0], points[:, 1], c=z_values, alpha=0.7,cmap='seismic', s=10**2)
    cbar = plt.colorbar(sc, fraction=0.040, pad=0.04, ax=ax)
    cbar.set_label(f"Z coordinate ($\\AA$)", fontsize=16)
    ax.set_facecolor('gainsboro')


    # Set labels, title, and formatting
    ax.set_xlabel(f"X ($\\AA$)", fontsize=16)
    ax.set_ylabel(f"Y ($\\AA$)", fontsize=16)
    ax.set_title(f'motif for $\\vec R = \\vec 0$ cell',fontsize=16)

    # ax.set_xlim(min(points[:,0])-0.5,max(points[:, 0])+0.5)
    # ax.set_ylim(min(points[:,1])-0.5,max(points[:, 1])+0.5)

    # yticks = np.arange(min(points[:,1].real)-0.5,max(points[:, 1].real)+0.5, 0.2)
    # ax.set_yticks(yticks)

    # ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("motif_center_z.png", dpi=300)
    plt.show()

#==============================================

# parser
bravais, nFock, mSize, iRn, H, WF, diag  = parse_file(filename)

# Access the parsed data - testing
# print("Bravais Vectors:\n", data['bravais_vectors'])
# print("nFock:", data['nFock'])
# print("mSize:", data['mSize'])
# print("First Fock Matrix (Real/Imag):\n", data['H'][0])
# print("First Fock Matrix (XYZ Components):\n", data['WF'][0])

# plot_orbitals_Ncoded(iRn, bravais, WF)
# plot_orbitals_indexCoded(iRn, bravais, WF, diag)
plot_orbital_localizations_with_z(iRn, bravais, WF, diag)