# shift_resonance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ArrayR = np.ndarray
ArrayC = np.ndarray

AXIS_MAP = {"x": 0, "y": 1, "z": 2}

# ==========================
# Internal .omesp reader
# ==========================
from dataclasses import dataclass

@dataclass
class OmespData:
    kpts: np.ndarray     # (Nk, 3)
    ek: np.ndarray       # (Nk, Nb)
    vme: np.ndarray      # (Nk, 3, Nb, Nb)          complex128
    berry: np.ndarray    # (Nk, 3, Nb, Nb)          complex128
    shift: np.ndarray    # (Nk, 3, 3, Nb, Nb)       float64
    gder: np.ndarray     # (Nk, 3, 3, Nb, Nb)       complex128

def _read_floats(line: str) -> List[float]:
    return [float(x) for x in line.strip().split()]

def _read_complex_triplet(line: str) -> np.ndarray:
    toks = line.strip().split()
    if len(toks) < 3 + 2*3:
        raise ValueError("Line too short for 3 complex components.")
    vals = toks[3:3+6]
    arr = np.zeros(3, dtype=np.complex128)
    for i in range(3):
        re = float(vals[2*i])
        im = float(vals[2*i+1])
        arr[i] = re + 1j*im
    return arr

def _read_real_triplet(line: str) -> np.ndarray:
    toks = line.strip().split()
    if len(toks) < 3 + 3:
        raise ValueError("Line too short for 3 real components.")
    vals = toks[3:3+3]
    return np.array([float(v) for v in vals], dtype=np.float64)

def parse_omesp(path: str) -> OmespData:
    with open(path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        raise ValueError("Empty .omesp file")

    idx = 0
    _iflag_line = lines[idx].strip()
    idx += 1

    toks = lines[idx].strip().split()
    if len(toks) < 4:
        raise ValueError("Cannot infer nband_ex; eigenvalue line too short.")
    nb = len(toks) - 3

    kpts = []
    ek = []
    vme = []
    berry = []
    shift = []
    gder = []
    while idx < len(lines):
        toks = lines[idx].strip().split()
        idx += 1
        if len(toks) < 3 + nb:
            break
        kx, ky, kz = map(float, toks[:3])
        e_arr = np.array([float(x) for x in toks[3:3+nb]], dtype=np.float64)

        v_k = np.zeros((3, nb, nb), dtype=np.complex128)
        b_k = np.zeros((3, nb, nb), dtype=np.complex128)
        s_k = np.zeros((3, 3, nb, nb), dtype=np.float64)
        g_k = np.zeros((3, 3, nb, nb), dtype=np.complex128)

        for i in range(nb):
            for j in range(nb):
                v_arr = _read_complex_triplet(lines[idx]); idx += 1
                for nj in range(3):
                    v_k[nj, i, j] = v_arr[nj]

                b_arr = _read_complex_triplet(lines[idx]); idx += 1
                for nj in range(3):
                    b_k[nj, i, j] = b_arr[nj]

                for a in range(3):
                    s_arr = _read_real_triplet(lines[idx]); idx += 1
                    for nj in range(3):
                        s_k[a, nj, i, j] = s_arr[nj]

                for a in range(3):
                    gd_arr = _read_complex_triplet(lines[idx]); idx += 1
                    for nj in range(3):
                        g_k[a, nj, i, j] = gd_arr[nj]

        kpts.append([kx, ky, kz])
        ek.append(e_arr)
        vme.append(v_k)
        berry.append(b_k)
        shift.append(s_k)
        gder.append(g_k)

    kpts = np.asarray(kpts, dtype=np.float64)             # (Nk,3)
    ek = np.asarray(ek, dtype=np.float64)                 # (Nk,Nb)
    vme = np.asarray(vme, dtype=np.complex128)            # (Nk,3,Nb,Nb)
    berry = np.asarray(berry, dtype=np.complex128)        # (Nk,3,Nb,Nb)
    shift = np.asarray(shift, dtype=np.float64)           # (Nk,3,3,Nb,Nb)
    gder = np.asarray(gder, dtype=np.complex128)          # (Nk,3,3,Nb,Nb)
    return OmespData(kpts=kpts, ek=ek, vme=vme, berry=berry, shift=shift, gder=gder)

def read_one_omesp(path: str):
    """
    Canonical reader for this analysis.
    Returns:
      k:     (Nk, 3)
      E:     (Nk, Nb)
      vme:   (Nk, Nb, Nb, 3)
      berry: (Nk, Nb, Nb, 3)
      shift: (Nk, Nb, Nb, 3, 3)
      gder:  (Nk, Nb, Nb, 3, 3)
    """
    data = parse_omesp(path)
    # transpose axes to the canonical shapes expected elsewhere
    vme = np.transpose(data.vme, (0, 2, 3, 1))     # (Nk, Nb, Nb, 3)
    berry = np.transpose(data.berry, (0, 2, 3, 1)) # (Nk, Nb, Nb, 3)
    shift = np.transpose(data.shift, (0, 3, 4, 1, 2))  # (Nk, Nb, Nb, 3, 3)
    gder = np.transpose(data.gder, (0, 3, 4, 1, 2))    # (Nk, Nb, Nb, 3, 3)
    return data.kpts, data.ek, vme, berry, shift, gder


# -------------------------
# Data containers
# -------------------------

@dataclass
class FieldSample:
    path: str              # full path to the .omesp file
    E_field: float         # field value from folder name
    k: ArrayR              # (Nk, 3), reduced coords
    E: ArrayR              # (Nk, Nb), eV
    vme: ArrayC            # (Nk, Nb, Nb, 3), eV*Angstrom
    berry: ArrayC          # (Nk, Nb, Nb, 3), Angstrom
    shift: ArrayR          # (Nk, Nb, Nb, 3, 3), Angstrom
    gder: ArrayC           # (Nk, Nb, Nb, 3, 3), Angstrom

@dataclass
class ResonanceConfig:
    # Discovery
    root_dir: str                      # parent folder containing numeric subfolders
    omesp_filename: str                # same filename present in each numeric folder
    # Field selection/masking
    fields_include: Optional[List[float]] = None   # only include these exact field values
    fields_exclude: Optional[List[float]] = None   # exclude these
    field_min: Optional[float] = None              # numeric lower bound (inclusive)
    field_max: Optional[float] = None              # numeric upper bound (inclusive)
    # Physics choices
    iv: int = 0
    ic: int = 1
    comp_a: str = "z"
    comp_b: str = "y"
    comp_c: str = "y"
    eta_eV: float = 0.02
    omega_mode: str = "fixed_zero_field"           # "fixed_zero_field"|"per_field"|"manual"
    omega_value_eV: Optional[float] = None
    mask_window_mult: float = 2.0
    make_product_map: bool = True
    contour_mode: bool = False
    contour_grid_n: int = 200
    output_digits: int = 6              # zero-padding width for folder names
    output_scale: float = 10000.0       # scale factor to integerize E_field (e.g., 1e4 -> 4 decimals)
    series_weight: str = "mask"   # "mask" or "lorentzian"
    series_top_k: int = 8         # how many k-indices to highlight with line plots
    arc_mode: bool = False                  # turn on arclength parametrization
    arc_focus: str = "K"                    # "K","Kprime","G","M","custom"
    arc_custom: Optional[Tuple[float,float]] = None  # e.g., [0.333,0.333]
    arc_points: int = 200                   # number of samples along the contour
    arc_contour_levels: int = 1             # take first contour level only



@dataclass
class ResonanceResult:
    omega_eV: float
    E_field: float
    Ecv: ArrayR
    delta_w: ArrayR
    mask: ArrayR
    R_ab: ArrayR
    A_bc_abs: ArrayR
    S_local: Optional[ArrayR]

# -------------------------
# Discovery and loading
# -------------------------

def _is_float_folder(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False

def _passes_field_filters(val: float,
                          include: Optional[List[float]],
                          exclude: Optional[List[float]],
                          fmin: Optional[float],
                          fmax: Optional[float]) -> bool:
    if include is not None and len(include) > 0:
        if val not in include:
            return False
    if exclude is not None and len(exclude) > 0:
        if val in exclude:
            return False
    if fmin is not None and val < fmin:
        return False
    if fmax is not None and val > fmax:
        return False
    return True

def discover_field_files(cfg: ResonanceConfig) -> List[Tuple[float, str]]:
    pairs: List[Tuple[float, str]] = []
    for entry in os.listdir(cfg.root_dir):
        subdir = os.path.join(cfg.root_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if not _is_float_folder(entry):
            continue
        fval = float(entry)
        if not _passes_field_filters(
            fval, cfg.fields_include, cfg.fields_exclude, cfg.field_min, cfg.field_max
        ):
            continue
        omesp_path = os.path.join(subdir, cfg.omesp_filename)
        if os.path.isfile(omesp_path):
            pairs.append((fval, omesp_path))
    # sort by field value
    pairs.sort(key=lambda t: t[0])
    if len(pairs) == 0:
        raise FileNotFoundError("No .omesp files found matching the discovery rules.")
    return pairs

def build_field_sample(path: str, E_field: float, reader) -> FieldSample:
    k, E, vme, berry, shift, gder = reader(path)
    if k.ndim != 2 or k.shape[1] != 3:
        raise ValueError("k must have shape (Nk, 3).")
    return FieldSample(
        path=path, E_field=E_field, k=k, E=E, vme=vme, berry=berry, shift=shift, gder=gder
    )

# -------------------------
# Physics helpers
# -------------------------

HA_TO_EV = 27.211386245988

def _force_convert_energies_and_vme_to_eV(samples: List[FieldSample]) -> None:
    """
    Unconditionally convert energies and velocity matrix elements from Hartree
    to eV, assuming the .omesp producer used Hartree units. This scales:
      E  <- E * Ha_to_eV
      vme <- vme * Ha_to_eV
    Length-like quantities (shift, berry, gder) are left unchanged.
    """
    for smp in samples:
        smp.E *= HA_TO_EV
        smp.vme *= HA_TO_EV


def lorentzian_delta(x: ArrayR, eta: float) -> ArrayR:
    return (eta / np.pi) / (x * x + eta * eta)

def select_omegas(samples: List[FieldSample], cfg: ResonanceConfig) -> List[float]:
    if cfg.omega_mode == "manual":
        if cfg.omega_value_eV is None:
            raise ValueError("omega_mode is manual but omega_value_eV is None.")
        return [float(cfg.omega_value_eV)] * len(samples)
    if cfg.omega_mode == "per_field":
        out: List[float] = []
        for smp in samples:
            Ecv = smp.E[:, cfg.ic] - smp.E[:, cfg.iv]
            out.append(float(np.min(Ecv)))
        return out
    # fixed_zero_field: use the first (lowest E_field after sorting)
    Ecv0 = samples[0].E[:, cfg.ic] - samples[0].E[:, cfg.iv]
    omega0 = float(np.min(Ecv0))
    return [omega0] * len(samples)

def compute_resonant_maps(
    smp: FieldSample,
    cfg: ResonanceConfig,
    omega_eV: float
) -> ResonanceResult:
    a = AXIS_MAP[cfg.comp_a.lower()]
    b = AXIS_MAP[cfg.comp_b.lower()]
    c = AXIS_MAP[cfg.comp_c.lower()]
    Nk, Nb = smp.E.shape
    if cfg.iv < 0 or cfg.ic < 0 or cfg.iv >= Nb or cfg.ic >= Nb or cfg.iv == cfg.ic:
        raise ValueError("Invalid iv/ic indices for this file window.")
    Ecv = smp.E[:, cfg.ic] - smp.E[:, cfg.iv]
    x = Ecv - omega_eV
    delta_w = lorentzian_delta(x, cfg.eta_eV)
    mask = np.abs(x) <= (cfg.mask_window_mult * cfg.eta_eV)
    R_ab = smp.shift[:, cfg.iv, cfg.ic, a, b].astype(float)
    v_cv_b = smp.vme[:, cfg.iv, cfg.ic, b]
    v_vc_c = smp.vme[:, cfg.ic, cfg.iv, c]
    A_bc_abs = np.abs(v_cv_b * v_vc_c)
    S_local: Optional[ArrayR] = None
    if cfg.make_product_map:
        S_local = (R_ab * A_bc_abs * delta_w).astype(float)
    return ResonanceResult(
        omega_eV=float(omega_eV),
        E_field=float(smp.E_field),
        Ecv=Ecv,
        delta_w=delta_w,
        mask=mask,
        R_ab=R_ab,
        A_bc_abs=A_bc_abs,
        S_local=S_local,
    )

# -------------------------
# Plot helpers (optional)
# -------------------------
def _wrap_reduced_centered(k: np.ndarray) -> np.ndarray:
    """
    Wrap reduced k to the centered cell [-0.5, 0.5) x [-0.5, 0.5).
    Works on a copy; does not modify input.
    """
    kw = k.copy()
    kw[:, 0] = ((kw[:, 0] + 0.5) % 1.0) - 0.5
    kw[:, 1] = ((kw[:, 1] + 0.5) % 1.0) - 0.5
    return kw


def _wrap_point(x: float, y: float) -> Tuple[float, float]:
    xx = ((x + 0.5) % 1.0) - 0.5
    yy = ((y + 0.5) % 1.0) - 0.5
    return xx, yy

def _add_high_symmetry(ax):
    # Gamma
    G = [(0.0, 0.0)]
    # Hex M points: (1/2,0), (0,1/2), (1/2,1/2)
    Ms = [(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
    # Hex K/K' points (six corners)
    baseK = [(1.0/3.0, 1.0/3.0), (2.0/3.0, 1.0/3.0), (1.0/3.0, 2.0/3.0)]
    K_all = baseK + [(-x, -y) for (x, y) in baseK]

    def plot_one(pt, label, color):
        xx, yy = _wrap_point(*pt)
        ax.scatter([xx], [yy], s=35, marker="x", c=color)
        ax.annotate(label, (xx, yy), textcoords="offset points", xytext=(5, 5))

    plot_one(G[0], "G", "k")
    # place one M and one K label; the symmetry copies will be obvious from the grid
    plot_one(Ms[0], "M", "tab:green")     # (0.5, 0)
    plot_one(baseK[0], "K", "tab:orange") # (1/3, 1/3)


def scatter_map(k: ArrayR, values: ArrayR, mask: Optional[ArrayR],
                title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    import matplotlib.pyplot as plt
    k = _wrap_reduced_centered(k)
    kx = k[:, 0]; ky = k[:, 1]
    if mask is not None:
        sel = mask.astype(bool)
        kx = kx[sel]; ky = ky[sel]; values = values[sel]
    fig, ax = plt.subplots()
    h = ax.scatter(kx, ky, c=values, s=12, edgecolors="none")
    ax.set_xlabel("k_x (reduced)"); ax.set_ylabel("k_y (reduced)"); ax.set_title(title)
    if vmin is not None or vmax is not None:
        h.set_clim(vmin, vmax)
    plt.colorbar(h, ax=ax, label="value")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    _add_high_symmetry(ax)
    fig.tight_layout()

def contour_map(k: ArrayR, values: ArrayR, mask: Optional[ArrayR],
                title: str, levels: int = 20) -> None:
    
    k = _wrap_reduced_centered(k)
    kx = k[:, 0]; ky = k[:, 1]
    if mask is not None:
        sel = mask.astype(bool)
        kx = kx[sel]; ky = ky[sel]; values = values[sel]
    tri = mtri.Triangulation(kx, ky)
    fig, ax = plt.subplots()
    cf = ax.tricontourf(tri, values, levels=levels)
    ax.set_xlabel("k_x (reduced)"); ax.set_ylabel("k_y (reduced)"); ax.set_title(title)
    plt.colorbar(cf, ax=ax, label="value")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    _add_high_symmetry(ax)
    fig.tight_layout()


def plot_resonant_panels(smp: FieldSample, res: ResonanceResult, comp_label: str, use_contour: bool) -> None:
    Rabs = np.abs(res.R_ab)
    Aabs = res.A_bc_abs
    mask = res.mask
    ttl0 = f"|R_{comp_label}|, omega={res.omega_eV:.3f} eV, E={res.E_field}"
    ttl1 = f"|A_{comp_label}|, omega={res.omega_eV:.3f} eV, E={res.E_field}"
    ttl2 = f"|R|*|A|*delta, omega={res.omega_eV:.3f} eV, E={res.E_field}"
    if use_contour:
        contour_map(smp.k, Rabs, mask, ttl0)
        contour_map(smp.k, Aabs, mask, ttl1)
        if res.S_local is not None:
            contour_map(smp.k, res.S_local, mask, ttl2)
    else:
        scatter_map(smp.k, Rabs, mask, ttl0)
        scatter_map(smp.k, Aabs, mask, ttl1)
        if res.S_local is not None:
            scatter_map(smp.k, res.S_local, mask, ttl2)

def _nearest_index_to_point(k: np.ndarray, tgt: Tuple[float, float]) -> int:
    k = _wrap_reduced_centered(k)
    d = np.linalg.norm(k[:, :2] - np.array(tgt, dtype=float).reshape(1, 2), axis=1)
    return int(np.argmin(d))

def build_series(samples: List[FieldSample],
                 results: Dict[str, ResonanceResult],
                 series_weight: str = "mask") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (fields, R_series, A_series, S_series)
    shapes: fields -> (Nf,), each series -> (Nf, Nk)
    - R_series holds |R_ab|, zeroed (mask) or weighted (lorentzian) off resonance
    - A_series holds |A_bc|, idem
    - S_series holds |R|*|A|*delta (already delta-weighted); we still zero outside mask for clarity
    """
    # ensure same sample order as in analyze loop (sorted by field)
    ordered = sorted(((s.E_field, s, results[s.path]) for s in samples), key=lambda t: t[0])
    fields = np.array([t[0] for t in ordered], dtype=float)
    Nf = len(ordered)
    Nk = ordered[0][1].E.shape[0]

    R_ser = np.zeros((Nf, Nk), dtype=float)
    A_ser = np.zeros((Nf, Nk), dtype=float)
    S_ser = np.zeros((Nf, Nk), dtype=float)

    for i, (_, smp, res) in enumerate(ordered):
        mask = res.mask.astype(bool)
        if series_weight.lower() == "lorentzian":
            w = res.delta_w
        else:  # mask
            w = mask.astype(float)

        Rabs = np.abs(res.R_ab)
        Aabs = res.A_bc_abs
        # product proxy already includes delta, but we will still zero outside mask for cleaner visuals
        Svals = res.S_local if res.S_local is not None else (Rabs * Aabs * res.delta_w)

        R_ser[i, :] = Rabs * w
        A_ser[i, :] = Aabs * w
        S_tmp = np.array(Svals, dtype=float)
        S_tmp[~mask] = 0.0
        S_ser[i, :] = S_tmp

    return fields, R_ser, A_ser, S_ser

def plot_series_heatmaps(outdir: str,
                         fields: np.ndarray,
                         R_ser: np.ndarray,
                         A_ser: np.ndarray,
                         S_ser: np.ndarray,
                         kref: np.ndarray) -> None:
    """
    Save three heatmaps: R, A, product. Y: field, X: k-index.
    Marks nearest indices to G, K, M.
    """
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    # label rows by field
    y = fields
    extent = [0, R_ser.shape[1], float(y.min()), float(y.max())]

    def _hm(arr: np.ndarray, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(arr, aspect="auto", origin="lower", extent=extent)
        ax.set_xlabel("k index")
        ax.set_ylabel("E_field")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="value")
        # markers: vertical lines at special k indices
        iG = _nearest_index_to_point(kref, (0.0, 0.0))
        iK = _nearest_index_to_point(kref, (1.0/3.0, 1.0/3.0))
        iM = _nearest_index_to_point(kref, (0.5, 0.0))
        for ix, lab, col in [(iG, "G", "k"), (iK, "K", "tab:orange"), (iM, "M", "tab:green")]:
            ax.axvline(ix, color=col, lw=1.0, ls="--")
            ax.text(ix + 2, y.max(), lab, color=col, va="top")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)

    _hm(R_ser, "|R| vs field per k index", "series_R_heatmap.png")
    _hm(A_ser, "|A| vs field per k index", "series_A_heatmap.png")
    _hm(S_ser, "|R|*|A|*delta vs field per k index", "series_S_heatmap.png")

def plot_series_top_traces(outdir: str,
                           fields: np.ndarray,
                           R_ser: np.ndarray,
                           A_ser: np.ndarray,
                           S_ser: np.ndarray,
                           kref: np.ndarray,
                           top_k: int = 8) -> None:
    """
    Pick k-indices with largest total S signal across fields and plot line traces vs field.
    """
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    score = np.sum(S_ser, axis=0)  # importance per k-index
    order = np.argsort(-score)[:max(1, int(top_k))]
    labels = []
    # nearest special points for reference
    iG = _nearest_index_to_point(kref, (0.0, 0.0))
    iK = _nearest_index_to_point(kref, (1.0/3.0, 1.0/3.0))
    iM = _nearest_index_to_point(kref, (0.5, 0.0))

    def _one(arr: np.ndarray, title: str, fname: str):
        fig, ax = plt.subplots()
        for idx in order:
            ax.plot(fields, arr[:, idx], "-o", ms=3, alpha=0.9, label=f"k{idx}")
        ax.set_xlabel("E_field")
        ax.set_title(title)
        # annotate special k indices
        ax.axvline(fields[0], color="0.8", lw=0.5)  # just to keep x-limits sane
        ax.legend(ncol=4, fontsize=8, frameon=False)
        # add text about special k indices
        ax.text(0.02, 0.98, f"G={iG}, K={iK}, M={iM}", transform=ax.transAxes, ha="left", va="top")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)

    _one(R_ser, "Top |R| traces vs field", "series_R_traces.png")
    _one(A_ser, "Top |A| traces vs field", "series_A_traces.png")
    _one(S_ser, "Top |R|*|A|*delta traces vs field", "series_S_traces.png")


# -------------------------
# JSON driver
# -------------------------

def load_config(path: str) -> ResonanceConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return ResonanceConfig(
        root_dir=str(data["root_dir"]),
        omesp_filename=str(data.get("omesp_filename", "ome_nonlinear_sp_field.omesp")),
        fields_include=(list(data["fields_include"]) if "fields_include" in data else None),
        fields_exclude=(list(data["fields_exclude"]) if "fields_exclude" in data else None),
        field_min=(float(data["field_min"]) if "field_min" in data else None),
        field_max=(float(data["field_max"]) if "field_max" in data else None),
        iv=int(data.get("iv", 0)),
        ic=int(data.get("ic", 1)),
        comp_a=str(data.get("components", {}).get("a", "z")).lower(),
        comp_b=str(data.get("components", {}).get("b", "y")).lower(),
        comp_c=str(data.get("components", {}).get("c", "y")).lower(),
        eta_eV=float(data.get("eta_eV", 0.02)),
        omega_mode=str(data.get("omega_mode", "fixed_zero_field")),
        omega_value_eV=(float(data["omega_value_eV"]) if "omega_value_eV" in data else None),
        mask_window_mult=float(data.get("mask_window_mult", 2.0)),
        make_product_map=bool(data.get("make_product_map", True)),
        contour_mode=bool(data.get("contour_mode", False)),
        contour_grid_n=int(data.get("contour_grid_n", 200)),
        output_digits=int(data.get("output_digits", 6)),
        output_scale=float(data.get("output_scale", 10000.0)),
        series_weight=str(data.get("series_weight", "mask")),
        series_top_k=int(data.get("series_top_k", 8)),
        arc_mode=bool(data.get("arc_mode", False)),
        arc_focus=str(data.get("arc_focus", "K")),
        arc_custom=(tuple(data["arc_custom"]) if "arc_custom" in data else None),
        arc_points=int(data.get("arc_points", 200)),
        arc_contour_levels=int(data.get("arc_contour_levels", 1)),
    )

def analyze_from_config(cfg_path: str, reader) -> Dict[str, ResonanceResult]:
    cfg = load_config(cfg_path)
    discovered = discover_field_files(cfg)  # List[(E_field, path)]
    samples: List[FieldSample] = [build_field_sample(p, E, reader) for (E, p) in discovered]
    omegas = select_omegas(samples, cfg)
    comp_label = f"{cfg.comp_a};{cfg.comp_b}{cfg.comp_c}"
    results: Dict[str, ResonanceResult] = {}
    for smp, omega in zip(samples, omegas):
        res = compute_resonant_maps(smp, cfg, omega)
        results[smp.path] = res
        plot_resonant_panels(smp, res, comp_label)
    return results

def _focus_point(name: str, custom: Optional[Tuple[float,float]]) -> Tuple[float,float]:
    name = name.lower()
    if name == "k":       return (1.0/3.0, 1.0/3.0)
    if name == "kprime":  return (-1.0/3.0, -1.0/3.0)
    if name == "g":       return (0.0, 0.0)
    if name == "m":       return (0.5, 0.0)
    if name == "custom" and custom is not None: return (float(custom[0]), float(custom[1]))
    return (1.0/3.0, 1.0/3.0)

def _wrap_center_on_focus(k: np.ndarray, focus: Tuple[float,float]) -> np.ndarray:
    """
    Shift and wrap reduced coords so the chosen focus is at the center.
    This minimizes contour cuts by the unit-cell seam.
    """
    kxy = k[:, :2].copy()
    kxy[:, 0] = ((kxy[:, 0] - focus[0] + 0.5) % 1.0) - 0.5
    kxy[:, 1] = ((kxy[:, 1] - focus[1] + 0.5) % 1.0) - 0.5
    return kxy

def _extract_isocontour_path(kxy: np.ndarray, z: np.ndarray, level: float) -> Optional[np.ndarray]:
    """
    Extract a single closed iso-contour of z(kx,ky)=level on a triangulation.
    Returns Nx2 array of points ordered along the contour, or None if not found.
    """
    tri = mtri.Triangulation(kxy[:,0], kxy[:,1])
    cs = plt.tricontour(tri, z, levels=[level])
    plt.clf()
    if len(cs.allsegs) == 0 or len(cs.allsegs[0]) == 0:
        return None
    # take the longest segment as the main loop
    segs = cs.allsegs[0]
    path = max(segs, key=lambda arr: arr.shape[0])
    return path

def _parametrize_and_resample(path: np.ndarray, n_points: int) -> Tuple[np.ndarray,np.ndarray]:
    """
    Given path Nx2, compute cumulative arclength s in [0,1] and resample to n_points evenly in s.
    Returns (s_grid, path_resampled).
    """
    dif = np.diff(path, axis=0)
    ds = np.hypot(dif[:,0], dif[:,1])
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if s[-1] <= 0.0:
        s[-1] = 1.0
    s /= s[-1]
    s_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    x_new = np.interp(s_grid, s, path[:,0])
    y_new = np.interp(s_grid, s, path[:,1])
    return s_grid, np.column_stack([x_new, y_new])

def _orient_and_register(path: np.ndarray, focus: Tuple[float,float]) -> np.ndarray:
    """
    Make contour orientation consistent and register start: pick the point with
    maximum projection along +x (away from focus), then roll so it is index 0.
    """
    # orientation: enforce CCW by signed area
    x, y = path[:,0], path[:,1]
    area = 0.5 * np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))
    if area < 0:  # clockwise -> reverse
        path = path[::-1].copy()
    # register start
    proj = path[:,0]  # since focus-centered coords
    i0 = int(np.argmax(proj))
    return np.roll(path, -i0, axis=0)

def extract_arc_series_for_field(
    smp: FieldSample,
    res: ResonanceResult,
    cfg: ResonanceConfig
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    For one field: get Ecv-omega contour near the focus, resample by arclength,
    and interpolate arrays for |R|, |A|, and product.
    Returns (s_grid, kxy_centered, R_s, A_s, S_s) or None if contour not found.
    """
    focus = _focus_point(cfg.arc_focus, cfg.arc_custom)

    # center the k-grid at focus
    kxy = _wrap_center_on_focus(smp.k, focus)
    Ecv = res.Ecv
    path = _extract_isocontour_path(kxy, Ecv, res.omega_eV)
    if path is None or path.shape[0] < 5:
        return None

    # consistent orientation and start
    path = _orient_and_register(path, focus)

    # resample
    s_grid, path_s = _parametrize_and_resample(path, cfg.arc_points)

    # interpolate values along the path
    Rabs = np.abs(res.R_ab)
    Aabs = res.A_bc_abs
    # On the exact contour, delta is not needed; show product magnitude
    Sabs = Rabs * Aabs

    R_s = griddata(kxy, Rabs, (path_s[:,0], path_s[:,1]), method="linear")
    A_s = griddata(kxy, Aabs, (path_s[:,0], path_s[:,1]), method="linear")
    S_s = griddata(kxy, Sabs, (path_s[:,0], path_s[:,1]), method="linear")

    return s_grid, path_s, R_s, A_s, S_s

def build_arc_series(samples: List[FieldSample],
                     results: Dict[str, ResonanceResult],
                     cfg: ResonanceConfig):
    ordered = sorted(((s.E_field, s, results[s.path]) for s in samples), key=lambda t: t[0])
    fields = np.array([t[0] for t in ordered], dtype=float)
    Nf = len(ordered); Ns = cfg.arc_points
    R2 = np.full((Nf, Ns), np.nan, dtype=float)
    A2 = np.full((Nf, Ns), np.nan, dtype=float)
    S2 = np.full((Nf, Ns), np.nan, dtype=float)
    for i, (_, smp, res) in enumerate(ordered):
        out = extract_arc_series_for_field(smp, res, cfg)
        if out is None:
            continue
        s_grid, path_s, R_s, A_s, S_s = out
        R2[i,:] = R_s
        A2[i,:] = A_s
        S2[i,:] = S_s
    return fields, np.linspace(0.0,1.0,Ns,endpoint=False), R2, A2, S2

def plot_arc_heatmaps(outdir: str, fields, sgrid, R2, A2, S2):
    os.makedirs(outdir, exist_ok=True)
    extent = [float(sgrid.min()), float(sgrid.max()),
              float(np.nanmin(fields)), float(np.nanmax(fields))]
    def hm(arr, title, name):
        fig, ax = plt.subplots(figsize=(10,4))
        im = ax.imshow(arr, aspect="auto", origin="lower", extent=extent)
        ax.set_xlabel("contour arclength s")
        ax.set_ylabel("E_field")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="value")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=160, bbox_inches="tight")
        plt.close(fig)
    hm(R2, "|R| on Ecv=omega contour (arclength)", "arc_R_heatmap.png")
    hm(A2, "|A| on Ecv=omega contour (arclength)", "arc_A_heatmap.png")
    hm(S2, "|R|*|A| on Ecv=omega contour (arclength)", "arc_S_heatmap.png")


# -------------------------
# Execution helpers: saving and CLI
# -------------------------

def _safe_dirname_from_field(E_field: float, digits: int, scale: float) -> str:
    # Convert field to signed, zero-padded integer using the given scale.
    # Example: E_field=0.02, scale=1e4 -> 200, digits=6 -> "000200"
    val = int(round(abs(E_field) * scale))
    sign = "m" if E_field < 0 else "p"
    return f"E_{sign}{val:0{digits}d}"


def save_results(output_dir: str,
                 smp: FieldSample,
                 res: ResonanceResult,
                 comp_label: str,
                 cfg: ResonanceConfig,
                 save_arrays: bool = True,
                 save_figs: bool = True) -> None:
    import os
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    sub = os.path.join(output_dir, _safe_dirname_from_field(smp.E_field, cfg.output_digits, cfg.output_scale))
    os.makedirs(sub, exist_ok=True)

    # save arrays
    if save_arrays:
        np.savez(
            os.path.join(sub, "resonant.npz"),
            k=smp.k,
            Ecv=res.Ecv,
            delta_w=res.delta_w,
            mask=res.mask,
            R_ab=res.R_ab,
            A_bc_abs=res.A_bc_abs,
            S_local=(res.S_local if res.S_local is not None else np.array([])),
            omega_eV=res.omega_eV,
            E_field=smp.E_field,
            components=np.array([comp_label]),
        )

    # save figures generated by plot_resonant_panels
    if save_figs:
        fig_nums = plt.get_fignums()
        # save only the last three figures which belong to this sample
        # because we create three per sample in plot_resonant_panels
        to_save = fig_nums[-3:] if len(fig_nums) >= 3 else fig_nums
        for i, fid in enumerate(to_save, start=1):
            fig = plt.figure(fid)
            fig.savefig(os.path.join(sub, f"panel_{i:02d}.png"),
                        dpi=160, bbox_inches="tight")
        # close saved figs to keep memory in check
        for fid in to_save:
            plt.close(fid)

def analyze_and_save(cfg_path: str,
                     reader,
                     output_dir: str,
                     save_plots: bool = True,
                     show_plots: bool = False) -> Dict[str, ResonanceResult]:
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    # If running headless and only saving, use a non-GUI backend
    if save_plots and not show_plots:
        try:
            matplotlib.use("Agg")  # safe in headless environments
        except Exception:
            pass

    cfg = load_config(cfg_path)
    print(f"[info] scanning root_dir={cfg.root_dir} for '{cfg.omesp_filename}'")
    discovered = discover_field_files(cfg)
    print(f"[info] discovered {len(discovered)} field folders")

    samples: List[FieldSample] = [build_field_sample(p, E, read_one_omesp) for (E, p) in discovered]

    # FORCE unit conversion: Hartree -> eV for E and vme
    _force_convert_energies_and_vme_to_eV(samples)

    omegas = select_omegas(samples, cfg)

    comp_label = f"{cfg.comp_a};{cfg.comp_b}{cfg.comp_c}"

    results: Dict[str, ResonanceResult] = {}
    os.makedirs(output_dir, exist_ok=True)

    # analysis loop
    for smp, omega in zip(samples, omegas):
        print(f"[info] analyzing E_field={smp.E_field:.6g}  file={smp.path}  omega={omega:.6f} eV")
        res = compute_resonant_maps(smp, cfg, omega)
        results[smp.path] = res

        # inside analyze_and_save, after computing res:
        plot_resonant_panels(smp, res, comp_label, use_contour=cfg.contour_mode)

        # when saving:
        save_results(output_dir, smp, res, comp_label, cfg, save_arrays=True, save_figs=save_plots)


    # summary: resonant count vs field
    fields = np.array([s.E_field for s in samples], dtype=float)
    counts = np.array([int(results[s.path].mask.sum()) for s in samples], dtype=int)
    plt.figure()
    plt.plot(fields, counts, marker="o")
    plt.xlabel("E_field (from folder name)")
    plt.ylabel("resonant k count (mask true)")
    plt.title("Resonant set size vs field")
    if save_plots:
        plt.savefig(os.path.join(output_dir, "resonant_count_vs_field.png"),
                    dpi=160, bbox_inches="tight")
        plt.close()

    if show_plots:
        plt.show()

    # also dump a small machine-readable summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"omega_mode={cfg.omega_mode}\n")
        if cfg.omega_value_eV is not None:
            f.write(f"omega_value_eV={cfg.omega_value_eV}\n")
        f.write(f"eta_eV={cfg.eta_eV}\n")
        f.write(f"components={comp_label}\n")
        f.write("field  resonant_count\n")
        for s in samples:
            f.write(f"{s.E_field:.10g}  {int(results[s.path].mask.sum())}\n")

    print(f"[info] results saved under: {output_dir}")

    # build series across fields (no integration)
    series_dir = os.path.join(output_dir, "series")
    fields, R_ser, A_ser, S_ser = build_series(samples, results, series_weight=cfg.series_weight)

    # use the first sample's k grid for k-index references
    kref = samples[0].k

    # heatmaps
    plot_series_heatmaps(series_dir, fields, R_ser, A_ser, S_ser, kref)

    # line traces for the most contributing k-indices
    plot_series_top_traces(series_dir, fields, R_ser, A_ser, S_ser, kref, top_k=cfg.series_top_k)

    # save arrays for reuse
    np.savez(os.path.join(series_dir, "series_data.npz"),
             fields=fields, R_series=R_ser, A_series=A_ser, S_series=S_ser)

    tracked_dir = os.path.join(output_dir, "series_tracked")
    fields_tr, anchors, R_tr, A_tr, S_tr = build_tracked_series(samples, results, ref_field_index=0)
    plot_tracked_series(tracked_dir, fields_tr, anchors, R_tr, A_tr, S_tr)
    np.savez(os.path.join(tracked_dir, "tracked_series.npz"),
             fields=fields_tr, anchors=anchors, R_series=R_tr, A_series=A_tr, S_series=S_tr)

    if cfg.arc_mode:
        arc_dir = os.path.join(output_dir, "arc_length")
        fields_arc, sgrid, R2, A2, S2 = build_arc_series(samples, results, cfg)
        plot_arc_heatmaps(arc_dir, fields_arc, sgrid, R2, A2, S2)
        np.savez(os.path.join(arc_dir, "arc_series.npz"),
                 fields=fields_arc, sgrid=sgrid, R_series=R2, A_series=A2, S_series=S2)


    return results


def _torus_distance(p: np.ndarray, q: np.ndarray) -> float:
    # p, q: shape (2,), reduced coords; torus metric in [-0.5,0.5)
    d = np.abs(p - q)
    d = np.minimum(d, 1.0 - d)
    return float(np.hypot(d[0], d[1]))

def build_tracked_series(samples: List[FieldSample],
                         results: Dict[str, ResonanceResult],
                         ref_field_index: int = 0,
                         max_neighbors: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Track resonant k across fields.
    - Take resonant mask at 'ref_field_index' as anchors.
    - For each other field, for each anchor, choose the nearest k among that field's resonant mask.
    Returns:
      fields  (Nf,)
      k_anchor_idx (Na,) indices (in the ref field)
      R_ser, A_ser, S_ser  each (Nf, Na)
    """
    ordered = sorted(((s.E_field, s, results[s.path]) for s in samples), key=lambda t: t[0])
    fields = np.array([t[0] for t in ordered], dtype=float)
    Nf = len(ordered)
    # reference
    smp0 = ordered[ref_field_index][1]
    res0 = ordered[ref_field_index][2]
    k0 = _wrap_reduced_centered(smp0.k)[:, :2]
    mask0 = res0.mask.astype(bool)
    anchor_idx = np.flatnonzero(mask0)
    Na = anchor_idx.size
    if Na == 0:
        raise RuntimeError("Reference field has empty resonant set; cannot track.")

    R_ser = np.zeros((Nf, Na), dtype=float)
    A_ser = np.zeros((Nf, Na), dtype=float)
    S_ser = np.zeros((Nf, Na), dtype=float)

    # fill reference values
    R_ser[ref_field_index, :] = np.abs(res0.R_ab[anchor_idx])
    A_ser[ref_field_index, :] = res0.A_bc_abs[anchor_idx]
    S0 = res0.S_local if res0.S_local is not None else (np.abs(res0.R_ab) * res0.A_bc_abs * res0.delta_w)
    S_ser[ref_field_index, :] = S0[anchor_idx]

    # track for other fields
    for i, (_, smp, res) in enumerate(ordered):
        if i == ref_field_index:
            continue
        k = _wrap_reduced_centered(smp.k)[:, :2]
        m = res.mask.astype(bool)
        candidates = np.flatnonzero(m)
        if candidates.size == 0:
            # nothing resonant: leave zeros
            continue
        # build KD by brute force (Nk is manageable)
        for j, idx0 in enumerate(anchor_idx):
            p = k0[idx0]
            # nearest among resonant mask at this field
            dmin = 1e9
            best = candidates[0]
            for c in candidates:
                d = _torus_distance(p, k[c])
                if d < dmin:
                    dmin = d
                    best = c
            R_ser[i, j] = np.abs(res.R_ab[best])
            A_ser[i, j] = res.A_bc_abs[best]
            Sval = res.S_local[best] if res.S_local is not None else (np.abs(res.R_ab[best]) * res.A_bc_abs[best] * res.delta_w[best])
            S_ser[i, j] = Sval

    return fields, anchor_idx, R_ser, A_ser, S_ser

def plot_tracked_series(outdir: str,
                        fields: np.ndarray,
                        anchor_idx: np.ndarray,
                        R_ser: np.ndarray,
                        A_ser: np.ndarray,
                        S_ser: np.ndarray) -> None:
    import matplotlib.pyplot as plt, os
    os.makedirs(outdir, exist_ok=True)

    # summarize with heatmaps (fields x anchors)
    def hm(arr, title, fname):
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(arr, aspect="auto", origin="lower",
                       extent=[0, arr.shape[1], float(fields.min()), float(fields.max())])
        ax.set_xlabel("anchor k index (E=ref)")
        ax.set_ylabel("E_field")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="value")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)

    hm(R_ser, "|R| tracked (ref anchors)", "tracked_R_heatmap.png")
    hm(A_ser, "|A| tracked (ref anchors)", "tracked_A_heatmap.png")
    hm(S_ser, "|R|*|A|*delta tracked (ref anchors)", "tracked_S_heatmap.png")

# -------------------------
# CLI entry point
# -------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Resonant k analysis for shift-conductivity ingredients."
    )
    p.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    p.add_argument("--out", type=str, default="resonance_out", help="Output directory.")
    p.add_argument("--show", action="store_true", help="Show plots.")
    p.add_argument("--no-save", action="store_true", help="Do not save plots (arrays still saved).")
    return p

def main():
    import matplotlib
    args = _build_arg_parser().parse_args()
    # headless-friendly
    if not args.show:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

    # Use the internal reader defined above
    results = analyze_and_save(
        cfg_path=args.config,
        reader=read_one_omesp,
        output_dir=args.out,
        save_plots=(not args.no_save),
        show_plots=args.show,
    )

if __name__ == "__main__":
    main()
