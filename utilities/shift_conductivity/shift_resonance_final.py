#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ArrayR = np.ndarray
ArrayC = np.ndarray
AXIS_MAP = {"x": 0, "y": 1, "z": 2}
HA_TO_EV = 27.211386245988

# -------------------------
# .omesp reader (unchanged layout)
# -------------------------
@dataclass
class OmespData:
    kpts: np.ndarray     # (Nk, 3)
    ek: np.ndarray       # (Nk, Nb)
    vme: np.ndarray      # (Nk, 3, Nb, Nb)          complex128
    berry: np.ndarray    # (Nk, 3, Nb, Nb)          complex128
    shift: np.ndarray    # (Nk, 3, 3, Nb, Nb)       float64
    gder: np.ndarray     # (Nk, 3, 3, Nb, Nb)       complex128

def _read_complex_triplet(line: str) -> np.ndarray:
    toks = line.strip().split()
    vals = toks[3:3+6]
    arr = np.zeros(3, dtype=np.complex128)
    for i in range(3):
        arr[i] = float(vals[2*i]) + 1j*float(vals[2*i+1])
    return arr

def _read_real_triplet(line: str) -> np.ndarray:
    toks = line.strip().split()
    vals = toks[3:3+3]
    return np.array([float(v) for v in vals], dtype=np.float64)

def parse_omesp(path: str) -> OmespData:
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Empty .omesp")

    i = 0
    _ = lines[i].strip(); i += 1
    toks = lines[i].strip().split()
    if len(toks) < 4: raise ValueError("Cannot infer nb.")
    nb = len(toks) - 3

    kpts, ek, vme, berry, shift, gder = [], [], [], [], [], []
    while i < len(lines):
        toks = lines[i].strip().split(); i += 1
        if len(toks) < 3+nb: break
        kx, ky, kz = map(float, toks[:3])
        e_arr = np.array([float(x) for x in toks[3:3+nb]], float)

        v_k = np.zeros((3, nb, nb), np.complex128)
        b_k = np.zeros((3, nb, nb), np.complex128)
        s_k = np.zeros((3, 3, nb, nb), np.float64)
        g_k = np.zeros((3, 3, nb, nb), np.complex128)

        for ii in range(nb):
            for jj in range(nb):
                v_arr = _read_complex_triplet(lines[i]); i += 1
                for a in range(3): v_k[a, ii, jj] = v_arr[a]
                b_arr = _read_complex_triplet(lines[i]); i += 1
                for a in range(3): b_k[a, ii, jj] = b_arr[a]
                for a in range(3):
                    s_arr = _read_real_triplet(lines[i]); i += 1
                    for b in range(3): s_k[a, b, ii, jj] = s_arr[b]
                for a in range(3):
                    gd_arr = _read_complex_triplet(lines[i]); i += 1
                    for b in range(3): g_k[a, b, ii, jj] = gd_arr[b]

        kpts.append([kx, ky, kz]); ek.append(e_arr)
        vme.append(v_k); berry.append(b_k); shift.append(s_k); gder.append(g_k)

    return OmespData(
        kpts=np.asarray(kpts, float),                    # (Nk,3)
        ek=np.asarray(ek, float),                        # (Nk,Nb)
        vme=np.asarray(vme, np.complex128),              # (Nk,3,Nb,Nb)
        berry=np.asarray(berry, np.complex128),
        shift=np.asarray(shift, float),                  # (Nk,3,3,Nb,Nb)
        gder=np.asarray(gder, np.complex128),
    )

def read_one_omesp(path: str):
    d = parse_omesp(path)
    vme  = np.transpose(d.vme,  (0,2,3,1))   # (Nk,Nb,Nb,3)
    berry= np.transpose(d.berry,(0,2,3,1))
    shift= np.transpose(d.shift,(0,3,4,1,2)) # (Nk,Nb,Nb,3,3)
    gder = np.transpose(d.gder, (0,3,4,1,2))
    return d.kpts, d.ek, vme, berry, shift, gder

# -------------------------
# Data / config
# -------------------------
@dataclass
class FieldSample:
    path: str
    E_field: float
    k: ArrayR              # (Nk,3) raw, untouched
    E: ArrayR              # (Nk,Nb) eV
    vme: ArrayC            # (Nk,Nb,Nb,3) eV*Å
    shift: ArrayR          # (Nk,Nb,Nb,3,3) Å

@dataclass
class ResonanceConfig:
    root_dir: str
    omesp_filename: str = "ome_nonlinear_sp_field.omesp"
    iv: int = 0
    ic: int = 1
    comp_a: str = "z"
    comp_b: str = "y"
    comp_c: str = "y"
    eta_eV: float = 0.02
    omega_mode: str = "fixed_zero_field"
    omega_value_eV: Optional[float] = None
    mask_nonresonant: bool = True
    draw_resonance_contour: bool = True
    contour_levels: int = 1

    label_G: Tuple[float, float] = (0.0, 0.0)
    label_K: Tuple[float, float] = (1.0 / 3.0, 1.0 / 3.0)
    label_M: Tuple[float, float] = (0.5, 0.0)

    # NEW — allow disabling specific markers
    show_label_G: bool = True
    show_label_K: bool = True
    show_label_M: bool = True


def lorentzian_delta(x: ArrayR, eta: float) -> ArrayR:
    return (eta/np.pi)/(x*x + eta*eta)

# -------------------------
# Discovery
# -------------------------
def _is_float_folder(name: str) -> bool:
    try: float(name); return True
    except: return False

def discover_field_files(cfg: ResonanceConfig) -> List[Tuple[float,str]]:
    out = []
    for entry in os.listdir(cfg.root_dir):
        p = os.path.join(cfg.root_dir, entry)
        if os.path.isdir(p) and _is_float_folder(entry):
            f = os.path.join(p, cfg.omesp_filename)
            if os.path.isfile(f):
                out.append((float(entry), f))
    out.sort(key=lambda t: t[0])
    if not out: raise FileNotFoundError("No .omesp files found.")
    return out

def build_field_sample(path: str, E_field: float) -> FieldSample:
    k, E, vme, _, shift, _ = read_one_omesp(path)
    # Force Hartree -> eV for energies and vme
    E   = E * HA_TO_EV
    vme = vme * HA_TO_EV
    return FieldSample(path=path, E_field=E_field, k=k, E=E, vme=vme, shift=shift)

# -------------------------
# Core compute
# -------------------------
@dataclass
class ResonanceResult:
    omega_eV: float
    E_field: float
    Ecv: ArrayR
    mask: ArrayR
    Rabs: ArrayR
    Aabs: ArrayR
    prod_no_delta: ArrayR   # |R|*|A|

def choose_omega(samples: List[FieldSample], cfg: ResonanceConfig) -> List[float]:
    if cfg.omega_mode == "manual":
        if cfg.omega_value_eV is None:
            raise ValueError("omega_mode=manual but omega_value_eV is None.")
        return [float(cfg.omega_value_eV)]*len(samples)
    if cfg.omega_mode == "per_field":
        return [float(np.min(s.E[:, cfg.ic]-s.E[:, cfg.iv])) for s in samples]
    # fixed_zero_field
    Ecv0 = samples[0].E[:, cfg.ic]-samples[0].E[:, cfg.iv]
    return [float(np.min(Ecv0))]*len(samples)

def compute_for_sample(s: FieldSample, cfg: ResonanceConfig, omega: float) -> ResonanceResult:
    a = AXIS_MAP[cfg.comp_a.lower()]
    b = AXIS_MAP[cfg.comp_b.lower()]
    c = AXIS_MAP[cfg.comp_c.lower()]

    Ecv = s.E[:, cfg.ic] - s.E[:, cfg.iv]
    x = Ecv - omega
    mask = (np.abs(x) <= cfg.eta_eV*2.0)  # window for visibility; separate from contour
    Rabs = np.abs(s.shift[:, cfg.iv, cfg.ic, a, b]).astype(float)
    Aabs = np.abs(s.vme[:,  cfg.iv, cfg.ic, b] * s.vme[:, cfg.ic, cfg.iv, c]).astype(float)
    prod = Rabs * Aabs
    return ResonanceResult(omega, s.E_field, Ecv, mask, Rabs, Aabs, prod)

# -------------------------
# Plot (no wrapping; raw kx,ky)
# -------------------------
def _add_labels(ax, cfg: ResonanceConfig):
    """Optionally draw G, K, M markers (raw coords)."""
    if cfg.show_label_G:
        x, y = cfg.label_G
        ax.scatter([x], [y], s=35, marker="x", c="k", zorder=8)
        ax.annotate("G", (x, y), textcoords="offset points", xytext=(6, 6), color="k")
    if cfg.show_label_K:
        x, y = cfg.label_K
        ax.scatter([x], [y], s=35, marker="x", c="tab:orange", zorder=8)
        ax.annotate("K", (x, y), textcoords="offset points", xytext=(6, 6), color="tab:orange")
    if cfg.show_label_M:
        x, y = cfg.label_M
        ax.scatter([x], [y], s=35, marker="x", c="tab:green", zorder=8)
        ax.annotate("M", (x, y), textcoords="offset points", xytext=(6, 6), color="tab:green")


def _auto_limits(kxy: np.ndarray, pad=0.03):
    xmin, ymin = np.min(kxy[:,0]), np.min(kxy[:,1])
    xmax, ymax = np.max(kxy[:,0]), np.max(kxy[:,1])
    dx = (xmax-xmin); dy=(ymax-ymin)
    return (xmin-pad*dx, xmax+pad*dx, ymin-pad*dy, ymax+pad*dy)

def panel_scatter_with_contour(
    kxy: np.ndarray, zvals: np.ndarray, title: str,
    Ecv: np.ndarray, omega: float,
    cfg: ResonanceConfig, mask: Optional[np.ndarray], cmap: str="viridis"
):
    # optional masking (only affects background points)
    if cfg.mask_nonresonant and (mask is not None):
        sel = mask.astype(bool)
        kx = kxy[sel,0]; ky = kxy[sel,1]; vals = zvals[sel]
    else:
        kx = kxy[:,0]; ky = kxy[:,1]; vals = zvals

    fig, ax = plt.subplots()
    sc = ax.scatter(kx, ky, c=vals, s=9, edgecolors="none", cmap=cmap)
    plt.colorbar(sc, ax=ax, label="value")

    # resonance contour from full (unmasked) triangulation
    tri = mtri.Triangulation(kxy[:,0], kxy[:,1])
    try:
        cs = ax.tricontour(tri, Ecv, levels=[omega], colors="w", linewidths=0.8)
        # optional: label the contour
        ax.clabel(cs, inline=True, fmt=lambda *_: "", fontsize=6)
    except Exception:
        # if triangulation fails, just skip the line
        pass

    ax.set_title(title)
    ax.set_xlabel("k_x (reduced)"); ax.set_ylabel("k_y (reduced)")
    x0,x1,y0,y1 = _auto_limits(kxy); ax.set_xlim(x0,x1); ax.set_ylim(y0,y1)
    ax.set_aspect("equal", adjustable="box")
    _add_labels(ax, cfg)
    fig.tight_layout()

# -------------------------
# Driver
# -------------------------
def load_config(path: str) -> ResonanceConfig:
    with open(path, "r") as f:
        d = json.load(f)
    return ResonanceConfig(
        root_dir=str(d["root_dir"]),
        omesp_filename=str(d.get("omesp_filename", "ome_nonlinear_sp_field.omesp")),
        iv=int(d.get("iv", 0)), ic=int(d.get("ic", 1)),
        comp_a=str(d.get("components", {}).get("a", "z")).lower(),
        comp_b=str(d.get("components", {}).get("b", "y")).lower(),
        comp_c=str(d.get("components", {}).get("c", "y")).lower(),
        eta_eV=float(d.get("eta_eV", 0.02)),
        omega_mode=str(d.get("omega_mode", "fixed_zero_field")),
        omega_value_eV=(float(d["omega_value_eV"]) if "omega_value_eV" in d else None),
        mask_nonresonant=bool(d.get("mask_nonresonant", True)),
        draw_resonance_contour=bool(d.get("draw_resonance_contour", True)),
        contour_levels=int(d.get("contour_levels", 1)),
        label_G=tuple(d.get("label_G", [0.0, 0.0])),
        label_K=tuple(d.get("label_K", [1.0 / 3.0, 1.0 / 3.0])),
        label_M=tuple(d.get("label_M", [0.5, 0.0])),
        show_label_G=bool(d.get("show_label_G", True)),
        show_label_K=bool(d.get("show_label_K", True)),
        show_label_M=bool(d.get("show_label_M", True)),
    )


def analyze_and_plot(cfg_path: str, outdir: str, show: bool=False):
    os.makedirs(outdir, exist_ok=True)
    cfg = load_config(cfg_path)

    found = discover_field_files(cfg)
    samples = [build_field_sample(p, E) for (E,p) in found]
    omegas  = choose_omega(samples, cfg)

    comp_tag = f"{cfg.comp_a};{cfg.comp_b}{cfg.comp_c}"

    for s, omega in zip(samples, omegas):
        print(f"[info] E={s.E_field:g}  file={s.path}  omega={omega:.6f} eV")
        res = compute_for_sample(s, cfg, omega)

        kxy = s.k[:, :2]  # RAW, untouched

        # |R|
        panel_scatter_with_contour(
            kxy, res.Rabs,
            title=f"|R_{comp_tag}|, omega={res.omega_eV:.3f} eV, E={s.E_field}",
            Ecv=res.Ecv, omega=res.omega_eV, cfg=cfg, mask=res.mask
        )
        # |A|
        panel_scatter_with_contour(
            kxy, res.Aabs,
            title=f"|A_{comp_tag}|, omega={res.omega_eV:.3f} eV, E={s.E_field}",
            Ecv=res.Ecv, omega=res.omega_eV, cfg=cfg, mask=res.mask
        )
        # |R|*|A| (no delta)
        panel_scatter_with_contour(
            kxy, res.prod_no_delta,
            title=f"|R|*|A| (no delta), omega={res.omega_eV:.3f} eV, E={s.E_field}",
            Ecv=res.Ecv, omega=res.omega_eV, cfg=cfg, mask=res.mask
        )

        # save the three most-recent figures
        figs = plt.get_fignums()[-3:]
        sub = os.path.join(outdir, f"E_{s.E_field:.5f}")
        os.makedirs(sub, exist_ok=True)
        for i, fid in enumerate(figs, start=1):
            plt.figure(fid).savefig(os.path.join(sub, f"panel_{i:02d}.png"),
                                    dpi=160, bbox_inches="tight")
        # close them to keep memory sane
        for fid in figs: plt.close(fid)

    if show: plt.show()
    print(f"[info] saved under: {outdir}")

# -------------------------
# CLI
# -------------------------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Minimal resonant-k analysis for shift conductivity.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--out", default="resonance_min_out", type=str)
    p.add_argument("--show", action="store_true")
    return p

def main():
    args = _build_arg_parser().parse_args()
    analyze_and_plot(args.config, args.out, show=args.show)

if __name__ == "__main__":
    main()
