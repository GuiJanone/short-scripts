#!/usr/bin/env python3
# shift_resonance_decomp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ArrayR = np.ndarray
ArrayC = np.ndarray

AXIS_MAP = {"x": 0, "y": 1, "z": 2}
# ==========================
# .omesp reader (robust; now also discards gen_der)
# ==========================
from dataclasses import dataclass

HA_TO_EV = 27.211386245988

@dataclass
class OmespData:
    kpts: np.ndarray     # (Nk, 3)
    ek: np.ndarray       # (Nk, Nb)
    vme: np.ndarray      # (Nk, 3, Nb, Nb)          complex128
    berry: np.ndarray    # (Nk, 3, Nb, Nb)          complex128
    shift: np.ndarray    # (Nk, 3, 3, Nb, Nb)       float64

def _clean_tokens(line: str) -> List[str]:
    raw = line.strip()
    for c in ("#", "!"):
        p = raw.find(c)
        if p != -1:
            raw = raw[:p]
    raw = raw.replace(",", " ")
    return raw.split()

def _to_float(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))

def _read_numbers_spanning(lines: List[str], i: int, need: int, drop_kxyz_first_line: bool) -> Tuple[List[float], int]:
    vals: List[float] = []
    first = True
    n = len(lines)
    while len(vals) < need and i < n:
        toks = _clean_tokens(lines[i]); i += 1
        if not toks:
            continue
        if first and drop_kxyz_first_line and len(toks) >= 3:
            # drop kx ky kz only on the first physical line of this record
            try:
                _ = _to_float(toks[0]); _ = _to_float(toks[1]); _ = _to_float(toks[2])
                toks = toks[3:]
            except Exception:
                pass
        first = False
        for t in toks:
            if len(vals) == need:
                break
            vals.append(_to_float(t))
    if len(vals) < need:
        raise ValueError(f"Needed {need} values, got {len(vals)}")
    return vals, i

def _read_eigen_line(lines: List[str], i: int) -> Tuple[float,float,float,List[float],int]:
    toks = _clean_tokens(lines[i]); i += 1
    if len(toks) < 4:
        raise ValueError("Eigen line too short or malformed.")
    kx, ky, kz = _to_float(toks[0]), _to_float(toks[1]), _to_float(toks[2])
    ener = [_to_float(t) for t in toks[3:]]
    return kx, ky, kz, ener, i

def _read_complex_triplet_from(lines: List[str], i: int) -> Tuple[np.ndarray, int]:
    nums, j = _read_numbers_spanning(lines, i, need=6, drop_kxyz_first_line=True)
    out = np.array([nums[0]+1j*nums[1], nums[2]+1j*nums[3], nums[4]+1j*nums[5]], dtype=np.complex128)
    return out, j

def _read_real_triplet_from(lines: List[str], i: int) -> Tuple[np.ndarray, int]:
    nums, j = _read_numbers_spanning(lines, i, need=3, drop_kxyz_first_line=True)
    return np.array(nums, dtype=np.float64), j

def parse_omesp(path: str) -> OmespData:
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Empty .omesp file")

    i = 0
    _ = lines[i].strip(); i += 1   # iflag_norder

    # infer nb from first eigen line
    kx0, ky0, kz0, ener0, i = _read_eigen_line(lines, i)
    nb = len(ener0)
    if nb <= 0:
        raise ValueError("Failed to infer number of bands (nb).")

    kpts, ek, vme, berry, shift = [], [], [], [], []

    def _read_one_k_block(kx, ky, kz, e_first, i0):
        v_k = np.zeros((3, nb, nb), dtype=np.complex128)
        b_k = np.zeros((3, nb, nb), dtype=np.complex128)
        s_k = np.zeros((3, 3, nb, nb), dtype=np.float64)
        i_loc = i0
        for p in range(nb):
            for q in range(nb):
                # vme (3 complex)
                v_arr, i_loc = _read_complex_triplet_from(lines, i_loc)
                for a in range(3): v_k[a, p, q] = v_arr[a]
                # berry (3 complex)
                b_arr, i_loc = _read_complex_triplet_from(lines, i_loc)
                for a in range(3): b_k[a, p, q] = b_arr[a]
                # shift: 3 real triplets (a = 0..2)
                for a in range(3):
                    s_arr, i_loc = _read_real_triplet_from(lines, i_loc)
                    for b in range(3): s_k[a, b, p, q] = s_arr[b]
                # gen_der: 3 complex triplets — we don't store, just consume
                for a in range(3):
                    _, i_loc = _read_complex_triplet_from(lines, i_loc)

        kpts.append([kx, ky, kz])
        ek.append(np.array(e_first, dtype=np.float64))
        vme.append(v_k); berry.append(b_k); shift.append(s_k)
        return i_loc

    i = _read_one_k_block(kx0, ky0, kz0, ener0, i)

    nlines = len(lines)
    while i < nlines:
        toks = _clean_tokens(lines[i])
        if not toks:
            i += 1
            continue
        kx, ky, kz, ener, i = _read_eigen_line(lines, i)
        if len(ener) < nb:
            need = nb - len(ener)
            more, i = _read_numbers_spanning(lines, i, need=need, drop_kxyz_first_line=False)
            ener.extend(more)
        i = _read_one_k_block(kx, ky, kz, ener[:nb], i)

    return OmespData(
        kpts=np.asarray(kpts, dtype=np.float64),
        ek=np.asarray(ek, dtype=np.float64),
        vme=np.asarray(vme, dtype=np.complex128),
        berry=np.asarray(berry, dtype=np.complex128),
        shift=np.asarray(shift, dtype=np.float64),
    )

def read_one_omesp(path: str):
    d = parse_omesp(path)
    # transpose to canonical shapes
    vme   = np.transpose(d.vme,   (0, 2, 3, 1))
    berry = np.transpose(d.berry, (0, 2, 3, 1))
    shift = np.transpose(d.shift, (0, 3, 4, 1, 2))
    # Hartree -> eV for energies and vme
    E_eV   = d.ek * HA_TO_EV
    vme_eV = vme * HA_TO_EV
    return d.kpts, E_eV, vme_eV, berry, shift


# ==========================
# Data / config
# ==========================

@dataclass
class FieldSample:
    path: str
    E_field: float
    k: ArrayR              # (Nk,3) raw, untouched
    E: ArrayR              # (Nk,Nb) eV
    vme: ArrayC            # (Nk,Nb,Nb,3) eV*Angstrom
    berry: ArrayC          # (Nk,Nb,Nb,3) Angstrom
    shift: ArrayR          # (Nk,Nb,Nb,3,3) Angstrom

@dataclass
class ResonanceConfig:
    root_dir: str
    omesp_filename: str = "ome_nonlinear_sp_field.omesp"

    # physics choices
    iv: int = 0
    ic: int = 1
    comp_a: str = "z"
    comp_b: str = "y"
    comp_c: str = "y"
    eta_eV: float = 0.02
    omega_mode: str = "fixed_zero_field"    # "fixed_zero_field"|"per_field"|"manual"
    omega_value_eV: Optional[float] = None

    # display choices
    mask_nonresonant: bool = True           # only show resonant points in scatter
    draw_resonance_contour: bool = True     # white iso-line from full triangulation

    # optional labels (all optional; you can reposition them)
    label_G: Tuple[float, float] = (0.0, 0.0)
    label_K: Tuple[float, float] = (1.0/3.0, 1.0/3.0)
    label_M: Tuple[float, float] = (0.5, 0.0)
    show_label_G: bool = True
    show_label_K: bool = True
    show_label_M: bool = True

# ==========================
# Discovery and loading
# ==========================

def _is_float_folder(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False

def discover_field_files(cfg: ResonanceConfig) -> List[Tuple[float, str]]:
    pairs: List[Tuple[float, str]] = []
    for entry in os.listdir(cfg.root_dir):
        sub = os.path.join(cfg.root_dir, entry)
        if not os.path.isdir(sub):
            continue
        if not _is_float_folder(entry):
            continue
        fval = float(entry)
        pth = os.path.join(sub, cfg.omesp_filename)
        if os.path.isfile(pth):
            pairs.append((fval, pth))
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        raise FileNotFoundError("No .omesp files found with the given pattern.")
    return pairs

def build_field_sample(path: str, E_field: float) -> FieldSample:
    k, E, vme, berry, shift = read_one_omesp(path)
    return FieldSample(path=path, E_field=E_field, k=k, E=E, vme=vme, berry=berry, shift=shift)

# ==========================
# Core compute + decomposition
# ==========================

@dataclass
class ResonanceResult:
    omega_eV: float
    E_field: float
    Ecv: ArrayR             # (Nk,)
    mask: ArrayR            # (Nk,) boolean window around resonance
    R_signed: ArrayR        # (Nk,) signed R^{a}_{vc;b}
    Rabs: ArrayR            # (Nk,)
    Aabs: ArrayR            # (Nk,)
    prod_no_delta: ArrayR   # (Nk,) = |R|*|A|
    xi_diff: ArrayR         # (Nk,) = Re[xi^b_vv - xi^b_cc]
    phase_rem: ArrayR       # (Nk,) = R_signed - xi_diff

def lorentzian_window(x: ArrayR, eta: float) -> np.ndarray:
    return (np.abs(x) <= 2.0*eta)

def choose_omega(samples: List[FieldSample], cfg: ResonanceConfig) -> List[float]:
    if cfg.omega_mode == "manual":
        if cfg.omega_value_eV is None:
            raise ValueError("omega_mode='manual' but omega_value_eV is None.")
        return [float(cfg.omega_value_eV)] * len(samples)
    if cfg.omega_mode == "per_field":
        return [float(np.min(s.E[:, cfg.ic] - s.E[:, cfg.iv])) for s in samples]
    # fixed_zero_field
    Ecv0 = samples[0].E[:, cfg.ic] - samples[0].E[:, cfg.iv]
    om0 = float(np.min(Ecv0))
    return [om0] * len(samples)

def compute_for_sample(s: FieldSample, cfg: ResonanceConfig, omega: float) -> ResonanceResult:
    a = AXIS_MAP[cfg.comp_a.lower()]
    b = AXIS_MAP[cfg.comp_b.lower()]
    c = AXIS_MAP[cfg.comp_c.lower()]

    Ecv = s.E[:, cfg.ic] - s.E[:, cfg.iv]
    x = Ecv - omega
    mask = lorentzian_window(x, cfg.eta_eV)

    # signed R^{a}_{vc;b} from file (real array)
    R_signed = s.shift[:, cfg.iv, cfg.ic, a, b].astype(float)
    Rabs = np.abs(R_signed)

    # |A| proxy from velocities
    Aabs = np.abs(s.vme[:, cfg.iv, cfg.ic, b] * s.vme[:, cfg.ic, cfg.iv, c]).astype(float)

    prod = Rabs * Aabs

    # diagonal Berry difference (real part)
    xi_vv = s.berry[:, cfg.iv, cfg.iv, b]
    xi_cc = s.berry[:, cfg.ic, cfg.ic, b]
    xi_diff = np.real(xi_vv - xi_cc)

    # phase-derivative remainder
    phase_rem = R_signed - xi_diff

    return ResonanceResult(
        omega_eV=omega, E_field=s.E_field, Ecv=Ecv, mask=mask,
        R_signed=R_signed, Rabs=Rabs, Aabs=Aabs, prod_no_delta=prod,
        xi_diff=xi_diff, phase_rem=phase_rem
    )

# ==========================
# Plotting (raw k; optional masking; white resonance contour)
# ==========================

def _add_labels(ax, cfg: ResonanceConfig):
    if cfg.show_label_G:
        x, y = cfg.label_G
        ax.scatter([x], [y], s=35, marker="x", c="k", zorder=6)
        ax.annotate("G", (x, y), textcoords="offset points", xytext=(6, 6), color="k")
    if cfg.show_label_K:
        x, y = cfg.label_K
        ax.scatter([x], [y], s=35, marker="x", c="tab:orange", zorder=6)
        ax.annotate("K", (x, y), textcoords="offset points", xytext=(6, 6), color="tab:orange")
    if cfg.show_label_M:
        x, y = cfg.label_M
        ax.scatter([x], [y], s=35, marker="x", c="tab:green", zorder=6)
        ax.annotate("M", (x, y), textcoords="offset points", xytext=(6, 6), color="tab:green")

def _auto_limits(kxy: np.ndarray, pad=0.03):
    xmin, ymin = float(np.min(kxy[:,0])), float(np.min(kxy[:,1]))
    xmax, ymax = float(np.max(kxy[:,0])), float(np.max(kxy[:,1]))
    dx = max(xmax-xmin, 1e-12); dy = max(ymax-ymin, 1e-12)
    return (xmin-pad*dx, xmax+pad*dx, ymin-pad*dy, ymax+pad*dy)

def panel_scatter_with_contour(
    kxy: np.ndarray, zvals: np.ndarray, title: str,
    Ecv: np.ndarray, omega: float,
    cfg: ResonanceConfig, mask: Optional[np.ndarray], cmap: str="viridis"
):
    if cfg.mask_nonresonant and (mask is not None):
        sel = mask.astype(bool)
        kx = kxy[sel,0]; ky = kxy[sel,1]; vals = zvals[sel]
    else:
        kx = kxy[:,0]; ky = kxy[:,1]; vals = zvals

    fig, ax = plt.subplots()
    sc = ax.scatter(kx, ky, c=vals, s=10, edgecolors="none", cmap=cmap)
    plt.colorbar(sc, ax=ax, label="value")

    if cfg.draw_resonance_contour:
        tri = mtri.Triangulation(kxy[:,0], kxy[:,1])
        try:
            cs = ax.tricontour(tri, Ecv, levels=[omega], colors="w", linewidths=0.9, zorder=5)
            # unlabeled white line (clean overlay)
            _ = cs
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel("k_x (reduced)"); ax.set_ylabel("k_y (reduced)")
    x0,x1,y0,y1 = _auto_limits(kxy); ax.set_xlim(x0,x1); ax.set_ylim(y0,y1)
    ax.set_aspect("equal", adjustable="box")
    _add_labels(ax, cfg)
    fig.tight_layout()

# ==========================
# Driver
# ==========================

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
        label_G=tuple(d.get("label_G", [0.0, 0.0])),
        label_K=tuple(d.get("label_K", [1.0/3.0, 1.0/3.0])),
        label_M=tuple(d.get("label_M", [0.5, 0.0])),
        show_label_G=bool(d.get("show_label_G", True)),
        show_label_K=bool(d.get("show_label_K", True)),
        show_label_M=bool(d.get("show_label_M", True)),
    )

def analyze_and_plot(cfg_path: str, outdir: str, show: bool=False):
    os.makedirs(outdir, exist_ok=True)
    cfg = load_config(cfg_path)
    found = discover_field_files(cfg)
    samples = [build_field_sample(p, E) for (E, p) in found]
    omegas  = choose_omega(samples, cfg)

    comp_tag = f"{cfg.comp_a};{cfg.comp_b}{cfg.comp_c}"

    for s, omega in zip(samples, omegas):
        print(f"[info] E={s.E_field:g}  file={s.path}  omega={omega:.6f} eV")
        res = compute_for_sample(s, cfg, omega)
        kxy = s.k[:, :2]  # RAW kx, ky

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
        # Decomposition (signed)
        panel_scatter_with_contour(
            kxy, res.xi_diff,
            title=f"Diagonal Berry (xi_vv - xi_cc) [{cfg.comp_b}], E={s.E_field}",
            Ecv=res.Ecv, omega=res.omega_eV, cfg=cfg, mask=res.mask, cmap="viridis"
        )
        panel_scatter_with_contour(
            kxy, res.phase_rem,
            title=f"R - (xi_vv - xi_cc) [{cfg.comp_b}], E={s.E_field}",
            Ecv=res.Ecv, omega=res.omega_eV, cfg=cfg, mask=res.mask, cmap="viridis"
        )

        # save last 5 figs
        figs = plt.get_fignums()[-5:]
        sub = os.path.join(outdir, f"E_{s.E_field:.5f}")
        os.makedirs(sub, exist_ok=True)
        for j, fid in enumerate(figs, start=1):
            plt.figure(fid).savefig(os.path.join(sub, f"panel_{j:02d}.png"),
                                    dpi=160, bbox_inches="tight")
        for fid in figs:
            plt.close(fid)

    if show:
        plt.show()
    print(f"[info] saved under: {outdir}")

# ==========================
# CLI
# ==========================

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Minimal resonant-k analysis with R decomposition (robust .omesp reader).")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--out", default="resonance_out", type=str)
    p.add_argument("--show", action="store_true")
    return p

def main():
    args = _build_arg_parser().parse_args()
    analyze_and_plot(args.config, args.out, show=args.show)

if __name__ == "__main__":
    main()
