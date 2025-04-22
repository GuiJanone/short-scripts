#!/usr/bin/env python3

'''
    This script contains some options for plotting the position matrix elements (PME)
    using the seedname_tb.dat file of a wannierized system.

    Plot Options:
    1: plot x and y positions colored with z position
    2: plot xyz position in a 3D interactive plot using plotly -- needs atomic positions
    3: plots x and y positions colored by their index on the matrix 
    4: plots the decay of PME along Rx, Ry, Rz - exception, plots everything and not only R = 0 cell
    
    Every option plots the diagonal of the (0,0,0) cell! 
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

#===============================================================#
# PARSER
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
#==========================PLOTTING============================#
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
#============================================================================#

#============================================================================#
def plot_orbitals_indexCoded(iRn, bravais_vectors, WF, n_index):
    """
    Plot the diagonal elements of a single n block with a discrete color mapping.
    
    Parameters:
        iRn: Real-space lattice positions.
        bravais_vectors: Bravais lattice vectors.
        WF: Wavefunction positions array (n, i, j, 3).
        n_index: Index of the n block to plot.
    """
    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)

    # Extract only the diagonal elements from WF for the chosen block
    orbital_positions = np.array([WF[n_index, i, i, :2] for i in range(min(WF.shape[1], WF.shape[2]))])
    
    # Compute positions
    points = real_space_positions[n_index, :2] + orbital_positions
    
    # Create a discrete colormap with as many colors as points
    num_points = points.shape[0]
    cmap = cm.get_cmap('viridis', num_points)  # Discrete colormap
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_points+1)-0.5, ncolors=num_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot with discrete colors
    sc = ax.scatter(points[:, 0], points[:, 1], c=np.arange(num_points), cmap=cmap, norm=norm, s=8**2, alpha=0.6)
    ax.scatter(points[0, 0], points[0, 1], c='black', marker='x', s=10**2, label='r=0')

    # Colorbar to show the discrete mapping
    cbar = plt.colorbar(sc, fraction=0.040, pad=0.04, ax=ax, ticks=np.arange(num_points))
    cbar.set_label("Index")

    ax.set_facecolor('gainsboro')
    # Labels and formatting
    ax.set_xlabel(f'X ($\\AA$)', fontsize=16)
    ax.set_ylabel(f'Y ($\\AA$)', fontsize=16)
    ax.set_title(f'motif for $R = 0,0,0$ cell', fontsize=16)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("motif_index.png", dpi=300)
    plt.show()
#============================================================================#

#============================================================================#

def plot_WF_xy_zcolor(iRn, bravais_vectors, WF, n_index):
    """
    Plot the diagonal elements of a single n block of orbital localizations in 2D (X vs. Y) with the Z coordinate
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
    
    # Extract only the diagonal elements from WF for the chosen block
    orbital_positions = np.array([WF[n_index, i, i, :] for i in range(min(WF.shape[1], WF.shape[2]))])
    orbital_positions = np.real(orbital_positions)

    # Compute positions
    points = real_space_positions[n_index, :2] + orbital_positions[:, :2]
    z_values = orbital_positions[:, 2]
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(points[:, 0], points[:, 1], c=z_values, alpha=0.6, cmap='seismic', s=10**2)
    cbar = plt.colorbar(sc, fraction=0.040, pad=0.04, ax=ax)
    cbar.set_label(f'Z coordinate ($\\AA$)', fontsize=16)
    ax.set_facecolor('gainsboro')

    # Set labels, title, and formatting
    ax.set_xlabel(f"X ($\\AA$)", fontsize=16)
    ax.set_ylabel(f"Y ($\\AA$)", fontsize=16)
    ax.set_title(f'motif for $\\vec R = \\vec 0$ cell', fontsize=16)
    
    plt.tight_layout()
    plt.savefig("motif_center_xy_zcolor.png", dpi=300)
    plt.show()
#============================================================================#

#============================================================================#
import numpy as np
import plotly.graph_objects as go
import webbrowser

def plot_WF_3D_interactive(iRn, bravais_vectors, WF, n_index):
    """
    Interactive 3D plot of Wannier function centers with atomic positions.

    Parameters:
    - iRn: Integer lattice vectors defining unit cells.
    - bravais_vectors: The real-space lattice vectors.
    - WF: Wannier function positions stored in WF[n, i, j, 3] (last index for x, y, z).
    - index: Selected block (n) to plot.
    - atomic_positions: Array of atomic positions in Cartesian coordinates.
    """

    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)

    # Extract only the diagonal elements from WF for the chosen block
    orbital_positions = np.array([WF[n_index, i, i, :] for i in range(min(WF.shape[1], WF.shape[2]))])
    orbital_positions = np.real(orbital_positions)

    # Combine real-space positions with orbital positions
    points = real_space_positions[n_index, :] + orbital_positions[:, :]

    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Create a 3D scatter plot
    fig = go.Figure()

    # Add Wannier function centers (color-coded by Z height)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=z, colorscale='Bluered', opacity=0.8),
    ))


    # Plot atoms
#    for element, positions in atoms.items():
#        fig.add_trace(go.Scatter3d(
#            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
#            mode='markers', marker=dict(size=6**2, color=colors[element], opacity=0.8),
#            name=element
#        ))
    # Plot lattice vectors as lines (to preserve the cell perception)
    # a1 = bravais_vectors[0,:]
    # a2 = bravais_vectors[1,:]
    # a3 = np.array([0.00000,0.00000, 9.0000000])
    # lattice_points = np.array([[0, 0, 0], a1, a2, a3, (a1 + a2), (a1 + a3), (a2 + a3), (a1 + a2 + a3)])
    # fig.add_trace(go.Scatter3d(
    #     x=lattice_points[:, 0], y=lattice_points[:, 1], z=lattice_points[:, 2],
    #     mode='markers', marker=dict(size=3, color="black", symbol='cross'),
    #     name="Lattice"
    # ))

    # Customize layout
    fig.update_layout(
        title="Wannier Functions Localizations with Atomic Structure",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Save and open as an interactive HTML file
    filename = "Wannier3D.html"
    fig.write_html(filename)
    # webbrowser.open(filename)


#==============================================

def get_decay(iRn, bravais_vectors, WF, H, diag, decimals=3):
    """
    Compute and plot the decay of the diagonal PME components (x, y, z) and hopping amplitude 
    as functions of the distance |R| from the central cell (excluded, based on diag).
    
    For each valid Fock (n) block (excluding diag), we compute one value per block by taking the 
    mean of the absolute real parts of the diagonal elements of WF (for each component) and the 
    maximum of the diagonal of H (representing the hopping amplitude). Then, using the real-space 
    translation vector R (obtained via iRn and bravais_vectors) and its norm, we group data with 
    similar |R| values by rounding.
    
    Parameters:
        WF             : Complex array of PME data with shape (nFock, mSize, mSize, 3) — only the 
                         diagonal elements (i == j) are meaningful.
        iRn            : Integer array of lattice translation indices with shape (nFock, 3).
        bravais_vectors: 3x3 array of Bravais lattice vectors.
        H              : Complex Hamiltonian array with shape (nFock, mSize, mSize).
        diag           : Integer index identifying the central (R = 0) cell to exclude.
        decimals       : Number of decimals to round the distance |R| for grouping (default: 6).
    
    Returns:
        unique_R         : 1D array of unique |R| values (sorted).
        grouped_pme_x    : 1D array of binned average |PME_x| per unique R.
        grouped_pme_y    : 1D array of binned average |PME_y| per unique R.
        grouped_pme_z    : 1D array of binned average |PME_z| per unique R.
        grouped_hopps    : 1D array of binned average hopping amplitude per unique R.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Number of Fock blocks and the matrix size for WF.
    nFock, mSize = WF.shape[0], WF.shape[1]
    # Compute real-space positions from lattice translations; shape (nFock, 3)
    real_space_positions = np.dot(iRn, bravais_vectors)
    
    # Create a boolean mask to exclude the central (R=0) block:
    valid_mask = np.ones(nFock, dtype=bool)
    valid_mask[diag] = True
    valid_indices = np.nonzero(valid_mask)[0]
    n_valid = valid_indices.size

    # Preallocate arrays to store the decay properties for each valid block.
    pme_mean_x = np.empty(n_valid)
    pme_mean_y = np.empty(n_valid)
    pme_mean_z = np.empty(n_valid)
    hopps_max    = np.empty(n_valid)
    
    # We'll also compute |R| for each block (using the real-space vector)
    R_norm = np.empty(n_valid)
    
    # Loop over valid Fock blocks
    for idx, n in enumerate(valid_indices):
        R_vec = real_space_positions[n]    # [Rx, Ry, Rz] for this block
        R_norm[idx] = np.linalg.norm(R_vec)  # |R|
        
        # Extract diagonal elements from WF for this block: WF[n, i, i, :], for i in 0...mSize-1.
        diag_elements = np.array([WF[n, i, i, :] for i in range(mSize)])
        # Compute the average (mean) of the absolute real parts for each spatial component.
        pme_mean_x[idx] = np.mean(np.abs(diag_elements[:, 0].real))
        pme_mean_y[idx] = np.mean(np.abs(diag_elements[:, 1].real))
        pme_mean_z[idx] = np.mean(np.abs(diag_elements[:, 2].real))
        
        # Get the diagonal of the hopping matrix for this block and take its maximum as the representative value.
        diag_H = np.diagonal(H[n, :, :]).real
        hopps_max[idx] = np.mean(np.abs(diag_H))
    
    # Group data based on unique |R| values (after rounding)
    R_rounded = np.round(R_norm, decimals=decimals)
    unique_R, group_indices = np.unique(R_rounded, return_inverse=True)
    
    # For each unique |R|, average the PME components and hopping amplitudes.
    grouped_pme_x = np.array([pme_mean_x[group_indices == i].mean() for i in range(len(unique_R))])
    grouped_pme_y = np.array([pme_mean_y[group_indices == i].mean() for i in range(len(unique_R))])
    grouped_pme_z = np.array([pme_mean_z[group_indices == i].mean() for i in range(len(unique_R))])
    grouped_hopps = np.array([hopps_max[group_indices == i].mean() for i in range(len(unique_R))])
    
    # Plotting the decay curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot for PME_x decay vs. |R|
    axs[0, 0].scatter(unique_R, grouped_pme_x, color='r', s=50, alpha=0.7, label='PME$_x$')
    axs[0, 0].set_xlabel(f'$|R|$ ($\\AA$)', fontsize=14)
    axs[0, 0].set_ylabel(f'Mean |PME$_x$|', fontsize=14)
    axs[0, 0].legend()
    
    # Plot for PME_y decay vs. |R|
    axs[0, 1].scatter(unique_R, grouped_pme_y, color='g', s=50, alpha=0.7, label='PME$_y$')
    axs[0, 1].set_xlabel(f'$|R|$ ($\\AA$)', fontsize=14)
    axs[0, 1].set_ylabel(f'Mean |PME$_y$|', fontsize=14)
    axs[0, 1].legend()
    
    # Plot for PME_z decay vs. |R|
    axs[1, 0].scatter(unique_R, grouped_pme_z, color='b', s=50, alpha=0.7, label='PME$_z$')
    axs[1, 0].set_xlabel(f'$|R|$ ($\\AA$)', fontsize=14)
    axs[1, 0].set_ylabel(f'Mean |PME$_z$|', fontsize=14)
    axs[1, 0].legend()
    
    # Plot for hopping amplitude vs. |R|
    axs[1, 1].scatter(unique_R, grouped_hopps, color='k', s=50, alpha=0.7, label='Hopping Amplitude')
    axs[1, 1].set_xlabel(f'$|R|$ ($\\AA$)', fontsize=14)
    axs[1, 1].set_ylabel('Hopping Amplitude', fontsize=14)
    axs[1, 1].legend(fontsize=12)
    
    for ax in axs.flat:
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.set_yscale('log')
        # ax.set_ylim(1e-8, 15)

    # fig.suptitle('MLWF', fontsize=16)
    plt.tight_layout()
    plt.savefig("dipole_decay.png", dpi=400)
    plt.show()
    

#==============================================
#==============================================
#==============================================

# parser
bravais, nFock, mSize, iRn, H, WF, diag  = parse_file(filename)

plot_option = int(sys.argv[2])
if plot_option == 1:
    print("Plotting XY-Zcolor motif")
    plot_WF_xy_zcolor(iRn, bravais, WF, diag)
elif plot_option == 2:
    print("Plotting interactive 3D motif")
    plot_WF_3D_interactive(iRn, bravais, WF, diag)
elif plot_option == 3:
    print("Plotting index-coded motif")
    plot_orbitals_indexCoded(iRn, bravais, WF, diag)
elif plot_option == 4:
    get_decay(iRn, bravais, WF, H, diag)

# plot_orbitals_Ncoded(iRn, bravais, WF)
# plot_WF_xz(iRn, bravais, WF, diag)
