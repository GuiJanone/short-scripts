#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
import argparse

def configure_matplotlib(fontsize=18):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams["axes.labelsize"] = fontsize+2
    plt.rcParams["axes.titlesize"] = fontsize+2
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams["legend.fontsize"] = fontsize+2
    # plt.rcParams["font.size"] = fontsize



# default Egap (can be overridden with --egap)
Egap = 0.0

def safe_make_spline(x, y, k_pref=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return None, None
    idx = np.argsort(x)
    x = x[idx]; y = y[idx]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.array([y[inv == i].mean() for i in range(len(ux))], dtype=float)
    if ux.size < 2:
        return None, None
    k = min(k_pref, int(ux.size) - 1)
    k = max(k, 1)
    try:
        spline = make_interp_spline(ux, uy, k=k)
    except Exception:
        return None, None
    return spline, ux

def plot_sigma_field(
    filename,
    centerV=False,
    compute_derivative=True,
    y_column=1,
    energy_min=Egap,
    energy_max=Egap + 1.0,
    field_max=1.000,
    n_field=1000,
    n_energies=10,
    cmap_name="turbo",
    label=None
):
    sp_file = filename

    # 1) collect numeric-named folders within range
    all_folders = []
    for f in os.listdir():
        if os.path.isdir(f):
            try:
                v = float(f)
                if abs(v) <= field_max:
                    all_folders.append((v, f))
            except ValueError:
                continue
    if not all_folders:
        print("No numeric-named folders found in range.")
        return

    all_folders.sort(key=lambda t: t[0])

    # 2) read spectra, keeping only successful ones and aligned fields
    F_used, spectra = [], []
    energy_ref = None
    for fval, folder in all_folders:
        path = os.path.join(folder, sp_file)
        if not os.path.exists(path):
            print(f"Warning: {path} missing; skipping.")
            continue
        try:
            data = np.loadtxt(path)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}; skipping.")
            continue
        if data.ndim != 2 or data.shape[1] <= y_column:
            print(f"Warning: {path} has unexpected columns; skipping.")
            continue

        if energy_ref is None:
            energy_ref = data[:, 0]
        else:
            if data.shape[0] != energy_ref.shape[0]:
                print(f"Warning: {path} has different number of points; skipping.")
                continue

        spectra.append(data[:, y_column])
        F_used.append(fval)

    if not spectra or energy_ref is None:
        print("No valid spectra found.")
        return

    F_used = np.asarray(F_used, dtype=float)
    S = np.vstack(spectra)
    energy = energy_ref

    # 3) select energy window
    mask_energy = (energy >= energy_min) & (energy <= energy_max)
    if not np.any(mask_energy):
        print("No energies within the requested window.")
        return
    Ewin = energy[mask_energy]
    Swin = S[:, mask_energy]

    # 4) choose representative energies
    n = min(n_energies, len(Ewin))
    idxs = np.linspace(0, len(Ewin) - 1, n).astype(int)
    Esel = Ewin[idxs]

    # 5) fine field axis using actual range
    Fmin, Fmax = float(np.nanmin(F_used)), float(np.nanmax(F_used))
    if not np.isfinite(Fmin) or not np.isfinite(Fmax) or Fmax == Fmin:
        Ffine = F_used.copy()
    else:
        Ffine = np.linspace(Fmin, Fmax, int(max(n_field, len(F_used))))

    # 6) figure layout
    if compute_derivative:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={"height_ratios": (2, 1)}
        )
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 4))
        ax1 = None

    # 7) colormap keyed to energy
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(Esel.min()), vmax=float(Esel.max()))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 8) per-energy curves vs field
    for omega, col_idx in zip(Esel, idxs):
        y_vs_F = Swin[:, col_idx]

        # optional centering at F=0
        if centerV:
            spline_c, ux = safe_make_spline(F_used, y_vs_F, k_pref=1)
            if spline_c is not None:
                sigma0 = float(spline_c(0.0))
            else:
                try:
                    ux2 = np.array(sorted(set(F_used)), dtype=float)
                    yx2 = np.array([y_vs_F[np.where(F_used == u)[0]].mean() for u in ux2])
                    sigma0 = float(np.interp(0.0, ux2, yx2))
                except Exception:
                    sigma0 = 0.0
            y_vals = y_vs_F - sigma0
        else:
            y_vals = y_vs_F

        # smooth along field
        spline, _ = safe_make_spline(F_used, y_vals, k_pref=3)
        col = sm.to_rgba(omega)

        if spline is not None and np.size(Ffine) > 1:
            sigma_smooth = spline(Ffine)
            mField = np.asarray(Ffine) * 1000.0
            ax0.plot(mField, sigma_smooth, color=col, linewidth=1.8, alpha=0.9)

            if compute_derivative and ax1 is not None:
                dsigma_dF = np.gradient(sigma_smooth, Ffine) * 10.0
                if centerV:
                    # subtract derivative value at F=0 to enforce centering
                    try:
                        ds0 = float(np.interp(0.0, Ffine, dsigma_dF))
                    except Exception:
                        ds0 = 0.0
                    dsigma_dF = dsigma_dF - ds0
                ax1.plot(mField, dsigma_dF, color=col, linestyle="--", alpha=0.9)




    # decorate axes
    if label is None:
        ax0.set_ylabel(r"$\sigma^{\(2\)}_{xxx}$ (nm$\cdot\mu$A/V$^2$)")
    else:
        ax0.set_ylabel(fr"$\sigma^{{(2)}}_{{{label}}}$  (nm$\cdot\mu$A/V$^2$)")
    ax0.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax0.tick_params(axis="x", which="minor", length=5)
    ax0.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax0.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(np.min(mField), np.max(mField))

    if compute_derivative and ax1 is not None:
        ax1.set_xlabel(r"Field $E_{\mathrm{DC}}$ (mV$/$\AA)")
        if label is None:
            ax1.set_ylabel(r"$d\sigma^{{(2)}}_{xxx}/dE$ (nm$^2\cdot\mu$A/V$^3$)")
        else:
            ax1.set_ylabel(rf"$d\sigma^{{(2)}}_{{{label}}}/dE$ (nm$^2\cdot\mu$A/V$^3$)")
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.tick_params(axis="x", which="minor", length=5)
        ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax1.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(np.min(mField), np.max(mField))
    else:
        ax0.set_xlabel(r"Field $E_{\mathrm{DC}}$ (mV$/$\AA)")


    # colorbar for energy
    if compute_derivative and ax1 is not None:
        cbar = fig.colorbar(sm, ax=(ax0, ax1), pad=0.02)
    else:
        cbar = fig.colorbar(sm, ax=ax0, pad=0.02)
    ticks = np.linspace(float(Esel.min()), float(Esel.max()), 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.set_label("Photon energy (eV)", rotation=270, labelpad=25)

    plt.savefig(f"linearity_{label}.png", dpi=800, bbox_inches="tight")
    # plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Plot sigma vs DC field, optionally with derivative.")
    p.add_argument("filename", help="file name inside each field folder to read")
    p.add_argument("--egap", type=float, default=Egap,
                   help=f"band gap energy in eV (default: {Egap})")
    p.add_argument("--center", action="store_true", help="center curves so sigma(F=0) = 0")
    p.add_argument("--no-deriv", action="store_true", help="disable derivative panel")
    p.add_argument("--ycol", type=int, default=1, help="column index for sigma (default: 1)")
    # make emin/emax optional; if omitted we derive them from egap in main()
    p.add_argument("--emin", type=float, default=None, help="min energy (default: egap)")
    p.add_argument("--emax", type=float, default=Egap+0.05, help="max energy (default: egap + 0.05)")
    p.add_argument("--fmax", type=float, default=1.000, help="max |field| to include (eV/Ang)")
    p.add_argument("--nfield", type=int, default=1000, help="number of field points in fine grid")
    p.add_argument("--nE", type=int, default=15, help="number of energies to plot")
    p.add_argument("--cmap", default="turbo", help="matplotlib colormap name")
    p.add_argument("--label", default=None, help="conductivity component (ijk)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("------------------------------------------------")
    print("            Welcome to LINEARITY.py!            ")
    print("------------------------------------------------")
    configure_matplotlib()

    # derive energy window from egap if not explicitly provided
    egap_val = float(args.egap)
    emin = args.emin if args.emin is not None else egap_val
    emax = args.emax if args.emax is not None else (egap_val + 0.05)

    plot_sigma_field(
        filename=args.filename,
        centerV=args.center,
        compute_derivative=not args.no-deriv if hasattr(args, "no-deriv") else not args.no_deriv,
        y_column=args.ycol,
        energy_min=emin,
        energy_max=emax,
        field_max=args.fmax,
        n_field=args.nfield,
        n_energies=args.nE,
        cmap_name=args.cmap,
        label=args.label
    )
    print(f"saved graph linearity_{args.label}.png")
