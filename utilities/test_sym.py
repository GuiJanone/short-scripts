import numpy as np
import tbmodels
import spglib
import sys

def parse_tb_dat(filename):
    with open(filename, 'r') as file:
        next(file)  # skip header

        # Lattice vectors
        lattice = np.array([list(map(float, next(file).split())) for _ in range(3)])

        # Number of R vectors and orbitals
        mSize = int(next(file).strip())
        nFock = int(next(file).strip())

        # Degeneracy line(s)
        n_degen = nFock
        degen_per_line = 15
        lines = n_degen // degen_per_line + int(n_degen % degen_per_line != 0)
        for _ in range(lines):
            next(file)

        next(file)  # Skip blank line

        # Hamiltonian terms
        hoppings = []
        r_vectors = []

        for _ in range(nFock):
            R = tuple(map(int, next(file).split()))
            r_vectors.append(R)
            for _ in range(mSize * mSize):
                line = next(file).strip()
                if not line:
                    continue
                i, j, re, im = line.split()
                i, j = int(i) - 1, int(j) - 1  # Fortran → Python index
                value = float(re) + 1j * float(im)
                hoppings.append((R, i, j, value))
            next(file)  # blank line

        # Orbital centers
        wf = np.zeros((mSize, 3))
        for _ in range(nFock):
            next(file)
            for _ in range(mSize * mSize):
                parts = next(file).strip().split()
                i, j = int(parts[0]), int(parts[1])
                if i == j:
                    # Only diagonal position matrix is used
                    for k in range(3):
                        re = float(parts[2 + k*2])
                        im = float(parts[3 + k*2])
                        wf[i - 1, k] = re  # neglecting imaginary part
            if _ != nFock - 1:
                next(file)

    return lattice, wf, hoppings


def build_tbmodels_model(lattice, positions, hoppings):
    model = tbmodels.Model(
        dim=3,
        occ=[],  # leave empty if you don't need this
        pos=positions,
        lattice=lattice
    )
    for R, i, j, val in hoppings:
        model.set_hop(i, j, val, R)
    return model


def check_spglib_symmetry(lattice, positions):
    """
    Convert TB model info to spglib format and check symmetry.
    """
    # Convert to spglib format
    lattice = np.array(lattice)
    positions = np.mod(positions, 1.0)  # ensure positions in [0, 1)
    numbers = [1] * len(positions)  # dummy atomic numbers

    cell = (lattice, positions, numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)

    print("\nSpglib detected space group:")
    print(f"Number: {dataset['number']}, Symbol: {dataset['international']}")

    print("\nSymmetry operations found:")
    for rot, trans in zip(dataset['rotations'], dataset['translations']):
        print("Rotation:\n", rot)
        print("Translation:", trans)
        print("─" * 30)


def main():
    filename = sys.argv[1]  # adapt this

    print("Parsing TB file...")
    lattice, positions, hoppings = parse_tb_dat(filename)

    print("Building TB model...")
    model = build_tbmodels_model(lattice, positions, hoppings)

    print("Checking symmetries...")
    check_spglib_symmetry(model._lattice, model._pos)

    # Optionally save the model
    model.to_file("model.h5", fmt="hdf5")

    print("Done.")


if __name__ == "__main__":
    main()