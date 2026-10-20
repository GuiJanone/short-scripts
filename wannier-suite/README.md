# wannier-suite

Tools for Wannier90 `seedname_tb.dat` files (also the XATU-extended format with an
optional overlap section appended by crystal2wannier).

- `w90tb.py` – shared fast reader/writer (`read_tb`, `write_tb`), overlap-aware.
- `w90field.py` – apply a uniform electric field: `H(R) -= E <a0|r_d|bR>`.
  Overlap `S(R)` is passed through unchanged (the field term is a position matrix
  element in the same non-orthogonal basis as `H`).

      python w90field.py seedname_tb.dat 0.01 -d z -o field_tb.dat

- `w90slab.py` – build finite z-slabs; builds and writes the slab overlap too, and `--check` solves the generalized problem `H c = E S c`. Input is divided by the degeneracies, so the slab file has degeneracy 1.
- `w90lattice.py` – plot position matrix elements / write Wannier centres.

All scripts share `w90tb.py` for I/O (run them from this directory).

`w90field.py` replaces the old Fortran `utilities/w90_field`.
