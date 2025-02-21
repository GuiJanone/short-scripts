import numpy as np
import matplotlib.pyplot as plt
filename = 'hBN-nonHSE_tb.dat'
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

            # Read the mSize x mSize matrix
            for _ in range(mSize * mSize):
                line = next(file).strip()
                if line:  # Skip empty lines
                    i, j, real, imag = map(float, line.split())
                    H[n, int(i)-1, int(j)-1] = complex(real, imag)  # Convert to 0-based indexing

            # Skip the blank line between matrices
            next(file)

        # Initialize the second set of Fock matrices (nFock x mSize x mSize x 3)
        H_xyz = np.zeros((nFock, mSize, mSize, 3), dtype=complex)

        # Parse the second set of nFock matrices (with X, Y, Z components)
        for n in range(nFock):
            line = next(file)
            iRn_tmp = line

            # Read the mSize x mSize matrix with X, Y, Z components
            for _ in range(mSize * mSize):
                line = next(file).strip()
                if line:  # Skip empty lines
                    parts = list(map(float, line.split()))
                    i, j = int(parts[0]), int(parts[1])
                    real_x, imag_x = parts[2], parts[3]
                    real_y, imag_y = parts[4], parts[5]
                    real_z, imag_z = parts[6], parts[7]
                    H_xyz[n, i-1, j-1, 0] = complex(real_x, imag_x)  # X component
                    H_xyz[n, i-1, j-1, 1] = complex(real_y, imag_y)  # Y component
                    H_xyz[n, i-1, j-1, 2] = complex(real_z, imag_z)  # Z component

            # Skip the blank line between matrices
            next(file)

    return np.array(bravais_vectors), nFock, mSize, iRn, H, H_xyz  # Second set of Fock matrices (nFock x mSize x mSize x 3)

# Example usage
bravais, nFock, mSize, iRn, H, H_xyz  = parse_file(filename)

# Access the parsed data
# print("Bravais Vectors:\n", data['bravais_vectors'])
# print("nFock:", data['nFock'])
# print("mSize:", data['mSize'])
# print("First Fock Matrix (Real/Imag):\n", data['H'][0])
# print("First Fock Matrix (XYZ Components):\n", data['H_xyz'][0])

#==============================================================#
# Create the plot
def map_to_real_space(iRn, bravais_vectors):
    """
    Map iRn indices to real-space positions using Bravais vectors.
    """
    return np.dot(iRn, bravais_vectors)

def plot_orbital_localizations(iRn, bravais_vectors, H_xyz):
    """
    Plot the localization of orbitals within each unit cell.
    """
    # Map iRn to real-space positions
    real_space_positions = map_to_real_space(iRn, bravais_vectors)

    # Extract orbital positions from H_xyz
    orbital_positions = H_xyz[:, :, :, :2]  # Use only X and Y components for 2D

    # Combine real-space positions with orbital positions
    combined_positions = []
    for i in range(real_space_positions.shape[0]):
        for j in range(orbital_positions.shape[1]):
            for k in range(orbital_positions.shape[2]):
                combined_positions.append(real_space_positions[i, :2] + orbital_positions[i, j, k, :2])

    combined_positions = np.array(combined_positions)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(combined_positions[:, 0], combined_positions[:, 1], c='blue', s=8**2)

    # Add labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title('Orbital Localizations in Real Space (2D)')

    # Add a legend
    ax.legend()

    # Set equal scaling for x and y axes
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("motif.png", dpi=300)
    # Show the plot
    plt.show()

# Example usage
# Assuming `data` is the dictionary returned by parse_file()
plot_orbital_localizations(iRn, bravais, H_xyz)
# Apply tight layout to prevent overlap