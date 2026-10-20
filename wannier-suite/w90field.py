#!/usr/bin/env python3
"""
Apply a static, uniform electric field to a Wannier90 seedname_tb.dat file:

    H(R)_{ab} -> H(R)_{ab} - E <a 0| r_d |b R>

using the position matrix elements stored in the file. Works for orthogonal
bases and for non-orthogonal ones (optional overlap section): the field term
is a position matrix element in the same basis as H, so the overlap S(R) is
unchanged and is carried through to the output.

Usage:  w90field.py seedname_tb.dat 0.01 [-d z] [-o field_tb.dat]
"""

import argparse

from w90tb import read_tb, write_tb

AXES = {"x": 0, "y": 1, "z": 2}


def apply_field(tb, amplitude: float, axis: int = 2):
    """Return tb with H -= amplitude * r_axis. Matrices are raw (same degeneracy factor)."""
    tb.h_r = tb.h_r - amplitude * tb.wf[..., axis]
    return tb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Input seedname_tb.dat")
    p.add_argument("amplitude", type=float, help="Field amplitude (energy/length units of the file)")
    p.add_argument("-d", "--direction", choices=AXES, default="z", help="Field direction (default z)")
    p.add_argument("-o", "--output", default="field_tb.dat", help="Output file (default field_tb.dat)")
    args = p.parse_args()

    tb = read_tb(args.input)
    kind = "non-orthogonal (overlap found)" if tb.overlap is not None else "orthogonal"
    print(f"{args.input}: {tb.m_size} orbitals, {tb.n_fock} R-blocks, {kind} basis")

    tb.header = f"Field applied = {args.amplitude} along {args.direction} | {tb.header.strip()}"
    write_tb(args.output, apply_field(tb, args.amplitude, AXES[args.direction]))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
