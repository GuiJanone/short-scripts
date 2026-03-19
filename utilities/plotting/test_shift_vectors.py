#!/usr/bin/env python3
"""
Test script to verify the shift vector reader implementation.
Creates a small test .omesp file with shift vector data and reads it.
"""

from __future__ import annotations
import tempfile
import os
import numpy as np

# Add the script directory to path
import sys
sys.path.insert(0, '/home/visitor/packages/short-scripts/utilities/plotting')

from vme_linear_plot import read_omesp_linear


def create_test_omesp_file(filepath: str, nk: int = 2, nb: int = 2) -> None:
    """Create a minimal .omesp test file."""
    with open(filepath, 'w') as f:
        # Header
        f.write("1\n")
        
        # For each k-point
        for ibz in range(nk):
            kx, ky, kz = ibz * 0.1, ibz * 0.2, 0.0
            # Eigenvalues
            eigenvalues = " ".join([str(i * 0.5) for i in range(nb)])
            f.write(f"{kx} {ky} {kz} {eigenvalues}\n")
            
            # For each band pair
            for i in range(nb):
                for j in range(nb):
                    # vme line (3 complex components)
                    f.write(f"{kx} {ky} {kz} 0.1 0.2 0.3 0.4 0.5 0.6\n")
                    
                    # berry_eigen line (3 complex components)
                    f.write(f"{kx} {ky} {kz} 0.01 0.02 0.03 0.04 0.05 0.06\n")
                    
                    # shift_vector lines (3 real components x 3 directions)
                    f.write(f"{kx} {ky} {kz} 1.1 1.2 1.3\n")  # direction 0
                    f.write(f"{kx} {ky} {kz} 2.1 2.2 2.3\n")  # direction 1
                    f.write(f"{kx} {ky} {kz} 3.1 3.2 3.3\n")  # direction 2
                    
                    # gen_der lines (3 complex components x 3 directions)
                    f.write(f"{kx} {ky} {kz} 0.001 0.002 0.003 0.004 0.005 0.006\n")
                    f.write(f"{kx} {ky} {kz} 0.001 0.002 0.003 0.004 0.005 0.006\n")
                    f.write(f"{kx} {ky} {kz} 0.001 0.002 0.003 0.004 0.005 0.006\n")


def test_read_omesp_linear():
    """Test the read_omesp_linear function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.omesp")
        create_test_omesp_file(filepath, nk=2, nb=2)
        
        # Read the file
        k, E, vme, berry, shift, gen_der = read_omesp_linear(filepath)
        
        # Verify shapes
        print("Testing read_omesp_linear()...")
        print(f"✓ k shape: {k.shape} (expected: (2, 3))")
        assert k.shape == (2, 3), f"k shape mismatch: {k.shape}"
        
        print(f"✓ E shape: {E.shape} (expected: (2, 2))")
        assert E.shape == (2, 2), f"E shape mismatch: {E.shape}"
        
        print(f"✓ vme shape: {vme.shape} (expected: (2, 2, 2, 3))")
        assert vme.shape == (2, 2, 2, 3), f"vme shape mismatch: {vme.shape}"
        assert vme.dtype == np.complex128, f"vme dtype mismatch: {vme.dtype}"
        
        print(f"✓ berry shape: {berry.shape} (expected: (2, 2, 2, 3))")
        assert berry.shape == (2, 2, 2, 3), f"berry shape mismatch: {berry.shape}"
        assert berry.dtype == np.complex128, f"berry dtype mismatch: {berry.dtype}"
        
        print(f"✓ shift shape: {shift.shape} (expected: (2, 2, 2, 3, 3))")
        assert shift.shape == (2, 2, 2, 3, 3), f"shift shape mismatch: {shift.shape}"
        assert shift.dtype == np.float64, f"shift dtype mismatch: {shift.dtype}"
        
        print(f"✓ gen_der shape: {gen_der.shape} (expected: (2, 2, 2, 3, 3))")
        assert gen_der.shape == (2, 2, 2, 3, 3), f"gen_der shape mismatch: {gen_der.shape}"
        assert gen_der.dtype == np.complex128, f"gen_der dtype mismatch: {gen_der.dtype}"
        
        # Verify some values
        print("\nVerifying shift vector values...")
        # First k-point, first band pair, direction 0
        expected_shift_d0 = np.array([1.1, 1.2, 1.3])
        actual_shift_d0 = shift[0, 0, 0, 0, :]
        print(f"✓ shift[0,0,0,0,:] = {actual_shift_d0} (expected: {expected_shift_d0})")
        assert np.allclose(actual_shift_d0, expected_shift_d0), "Shift values mismatch"
        
        # Direction 1
        expected_shift_d1 = np.array([2.1, 2.2, 2.3])
        actual_shift_d1 = shift[0, 0, 0, 1, :]
        print(f"✓ shift[0,0,0,1,:] = {actual_shift_d1} (expected: {expected_shift_d1})")
        assert np.allclose(actual_shift_d1, expected_shift_d1), "Shift values mismatch"
        
        # Direction 2
        expected_shift_d2 = np.array([3.1, 3.2, 3.3])
        actual_shift_d2 = shift[0, 0, 0, 2, :]
        print(f"✓ shift[0,0,0,2,:] = {actual_shift_d2} (expected: {expected_shift_d2})")
        assert np.allclose(actual_shift_d2, expected_shift_d2), "Shift values mismatch"
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_read_omesp_linear()
