# wannier-suite

Tools for Wannier90 `seedname_tb.dat` files (also the XATU-extended format with an
optional overlap section appended by crystal2wannier).

- `w90tb.py` – shared fast reader/writer (`read_tb`, `write_tb`), overlap-aware.
- `w90field.py` – apply a uniform electric field: `H(R) -= E <a0|r_d|bR>`.
  Overlap `S(R)` is passed through unchanged (the field term is a position matrix
  element in the same non-orthogonal basis as `H`). Warns if the field makes the
  `H(R)` blocks non-Hermitian (see below).

      python w90field.py seedname_tb.dat 0.01 -d z -o field_tb.dat

- `w90slab.py` – build finite z-slabs; builds and writes the slab overlap too, and `--check` solves the generalized problem `H c = E S c`. Input is divided by the degeneracies, so the slab file has degeneracy 1.
  With `--write-approx-full-wf` the slab position block includes the non-orthogonal
  shift `<la|r|l'b> = r_bulk + l*a3*S_bulk` (`r_ab(R)=<0a|r|Rb>` is measured from the
  cell of `a`). `--check` also reports block Hermiticity of H, S and r, the Nz=1
  projection, and the smallest eigenvalue of `S(k)`.
- `w90lattice.py` – plot position matrix elements / write Wannier centres.

All scripts share `w90tb.py` for I/O (run them from this directory).

`w90field.py` replaces the old Fortran `utilities/w90_field`.

## Non-orthogonal note
With an overlap, `r_ab(R)* = r_ba(-R) + R_cart S_ba(-R)`, so the position blocks are not
blockwise Hermitian by themselves. A uniform field along a periodic direction where
`S(R) != 0` therefore makes `H(R)` non-Hermitian (`w90field.py` warns); a z-field on a
slab (all `Rz = 0`) is fine.
