# wannier-suite

Tools for Wannier90 `seedname_tb.dat` files (also the XATU-extended format with an
optional overlap section appended by crystal2wannier).

- `w90tb.py` – shared fast reader/writer (`read_tb`, `write_tb`), overlap-aware.
- `w90field.py` – apply a uniform electric field: `H(R) -= E <a0|r_d|bR>`.
  Overlap `S(R)` is passed through unchanged (the field term is a position matrix
  element in the same non-orthogonal basis as `H`).

      python w90field.py seedname_tb.dat 0.01 -d z -o field_tb.dat

- `w90slab.py` – build finite slabs (has its own parser; does not yet carry the overlap).
- `w90lattice.py` – plot position matrix elements / write Wannier centres.

`w90field.py` replaces the old Fortran `utilities/w90_field`.
