#!/usr/bin/env python3
# vme_linear_plot.py
#
# Reads:
#   1) ome_linear_sp_*.omesp  (k-resolved SP vme matrix elements for band pairs)
#   2) ome_linear_ex_k_*.omeexk (k-resolved excitonic contributions V_{0N}(k))
#
# Plots:
#   - SP band-pair-resolved vme (path/contour)
#   - excitonic k-resolved V0N(k) for chosen exciton indices (path/contour)
#
# New features:
#   - Contour colormap:
#       value=abs  -> Reds
#       value=real/imag -> seismic
#   - Path option to plot real and imag together on the same axes:
#       config: "path_plot_real_imag": true
#       cli: --plot-real-imag
#
# ASCII-only.

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


AXIS_MAP = {"x": 0, "y": 1, "z": 2}


@dataclass
class Config:
    omesp_file: Optional[str] = None
    omeexk_file: Optional[str] = None

    out_dir: str = "vme_plots"
    mode: str = "path"          # path or contour
    component: str = "x"        # x,y,z
    value: str = "abs"          # abs, real, imag

    pairs: Optional[List[Tuple[int, int]]] = None
    excitons: Optional[List[int]] = None  # 1-based exciton indices

    order: str = "snake"  # snake or row
    kpath_file: Optional[str] = None
    closed: bool = False

    use_kdist: bool = True
    reduce: str = "mean"
    reduce_tol: float = 1e-10

    interp_nx: int = 300
    interp_ny: int = 300

    max_pairs: int = 64

    match_kpoints: bool = True
    match_tol: float = 1e-10

    # NEW: for mode=path, plot both real and imag on the same axes
    path_plot_real_imag: bool = False


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = json.load(f)

    pairs_raw = d.get("pairs", None)
    pairs: Optional[List[Tuple[int, int]]]
    if pairs_raw is None:
        pairs = None
    else:
        pairs = [tuple(map(int, p)) for p in pairs_raw]

    excitons_raw = d.get("excitons", None)
    excitons: Optional[List[int]]
    if excitons_raw is None:
        excitons = None
    else:
        excitons = [int(x) for x in excitons_raw]

    return Config(
        omesp_file=d.get("omesp_file", d.get("omesp_filename", None)),
        omeexk_file=d.get("omeexk_file", d.get("omeexk_filename", None)),
        out_dir=str(d.get("out_dir", d.get("out_prefix", "vme_plots"))),
        mode=str(d.get("mode", "path")),
        component=str(d.get("component", d.get("component_axis", "x"))),
        value=str(d.get("value", "abs")),
        pairs=pairs,
        excitons=excitons,
        order=str(d.get("order", "snake")),
        kpath_file=d.get("kpath_file", d.get("kpath", None)),
        closed=bool(d.get("closed", False)),
        use_kdist=bool(d.get("use_kdist", True)),
        reduce=str(d.get("reduce", "mean")),
        reduce_tol=float(d.get("reduce_tol", 1e-10)),
        interp_nx=int(d.get("interp_nx", d.get("interp_n", 300))),
        interp_ny=int(d.get("interp_ny", d.get("interp_n", 300))),
        max_pairs=int(d.get("max_pairs", 64)),
        match_kpoints=bool(d.get("match_kpoints", True)),
        match_tol=float(d.get("match_tol", 1e-10)),
        path_plot_real_imag=bool(d.get("path_plot_real_imag", False)),
    )


def read_omesp_linear(path: str):
    k_list = []
    E_list = []
    vme_list = []

    with open(path, "r") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"Empty file: {path}")

        while True:
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            Es = np.array([float(x) for x in parts[3:]], dtype=np.float64)
            nb = int(Es.size)

            vme_k = np.zeros((nb, nb, 3), dtype=np.complex128)
            for i in range(nb):
                for j in range(nb):
                    vline = f.readline()
                    if not vline:
                        raise ValueError("Unexpected EOF while reading vme block.")
                    vparts = vline.strip().split()
                    if len(vparts) < 9:
                        raise ValueError(f"Bad vme line: '{vline.strip()}'")
                    vals = [float(x) for x in vparts[3:9]]
                    vx = vals[0] + 1j * vals[1]
                    vy = vals[2] + 1j * vals[3]
                    vz = vals[4] + 1j * vals[5]
                    vme_k[i, j, :] = (vx, vy, vz)

            k_list.append([kx, ky, kz])
            E_list.append(Es)
            vme_list.append(vme_k)

    k = np.array(k_list, dtype=np.float64)
    E = np.array(E_list, dtype=np.float64)
    vme = np.array(vme_list, dtype=np.complex128)
    return k, E, vme


def read_omeexk(path: str):
    with open(path, "r") as f:
        line1 = f.readline()
        if not line1:
            raise ValueError(f"Empty file: {path}")
        line2 = f.readline()
        if not line2:
            raise ValueError(f"Missing norb_ex_cut line in: {path}")
        try:
            nexc = int(line2.strip().split()[0])
        except Exception as e:
            raise ValueError(f"Could not parse norb_ex_cut from '{line2.strip()}' in {path}") from e

        k_list: List[List[float]] = []
        v_list: List[np.ndarray] = []

        while True:
            kline = f.readline()
            if not kline:
                break
            parts = kline.strip().split()
            if len(parts) < 3:
                continue
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])

            vme_k = np.zeros((nexc, 3), dtype=np.complex128)
            for _ in range(nexc):
                dline = f.readline()
                if not dline:
                    raise ValueError("Unexpected EOF while reading omeexk block.")
                dp = dline.strip().split()
                if len(dp) < 7:
                    raise ValueError(f"Bad omeexk line: '{dline.strip()}'")
                nn = int(dp[0])
                vals = [float(x) for x in dp[1:7]]
                vx = vals[0] + 1j * vals[1]
                vy = vals[2] + 1j * vals[3]
                vz = vals[4] + 1j * vals[5]
                if nn < 1 or nn > nexc:
                    raise ValueError(f"Exciton index out of range in omeexk: nn={nn}, nexc={nexc}")
                vme_k[nn - 1, :] = (vx, vy, vz)

            k_list.append([kx, ky, kz])
            v_list.append(vme_k)

    k_ex = np.array(k_list, dtype=np.float64)
    v_ex_k = np.array(v_list, dtype=np.complex128)  # (Nk, nexc, 3)
    return k_ex, v_ex_k


def _k_key(k: np.ndarray, tol: float) -> Tuple[int, int, int]:
    return (
        int(round(float(k[0]) / tol)),
        int(round(float(k[1]) / tol)),
        int(round(float(k[2]) / tol)),
    )


def match_k_order(k_ref: np.ndarray, k_src: np.ndarray, tol: float) -> np.ndarray:
    src_map: Dict[Tuple[int, int, int], int] = {}
    for i in range(k_src.shape[0]):
        key = _k_key(k_src[i, :], tol)
        if key in src_map:
            raise ValueError("Duplicate k-point key in source set; decrease match_tol or fix input.")
        src_map[key] = i

    idx = np.zeros(k_ref.shape[0], dtype=int)
    for i in range(k_ref.shape[0]):
        key = _k_key(k_ref[i, :], tol)
        if key not in src_map:
            raise ValueError("Could not match k-point between files. Missing key in source set.")
        idx[i] = src_map[key]
    return idx


def infer_rect_grid(kx: np.ndarray, ky: np.ndarray, tol: float = 1e-10):
    kx_r = np.round(kx / tol) * tol
    ky_r = np.round(ky / tol) * tol
    ux = np.unique(kx_r)
    uy = np.unique(ky_r)
    nx = int(ux.size)
    ny = int(uy.size)
    if nx * ny != kx.size:
        return None

    x_to_ix = {val: i for i, val in enumerate(ux)}
    y_to_iy = {val: i for i, val in enumerate(uy)}
    ix = np.array([x_to_ix[val] for val in kx_r], dtype=int)
    iy = np.array([y_to_iy[val] for val in ky_r], dtype=int)

    order_row = np.lexsort((ix, iy))

    order_snake_rows = []
    for j in range(ny):
        row_mask = (iy == j)
        row_ids = np.where(row_mask)[0]
        row_ix = ix[row_mask]
        row_sorted = row_ids[np.argsort(row_ix)]
        if j % 2 == 1:
            row_sorted = row_sorted[::-1]
        order_snake_rows.append(row_sorted)
    order_snake = np.concatenate(order_snake_rows)
    return nx, ny, ix, iy, order_row, order_snake


def read_kpath_waypoints(path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("kpath_file must contain at least 2 waypoints (kx ky).")
    return pts


def build_kpath_nearest(kx: np.ndarray, ky: np.ndarray, waypoints: List[Tuple[float, float]], closed: bool) -> np.ndarray:
    pts = np.vstack([kx, ky]).T
    if pts.shape[0] == 0:
        return np.array([], dtype=int)

    sample = pts[::max(1, pts.shape[0] // 300)]
    dmin = []
    for p in sample:
        d2 = np.sum((pts - p) ** 2, axis=1)
        d2.sort()
        if d2.size > 1:
            dmin.append(math.sqrt(float(d2[1])))
    avg_nn = float(np.median(dmin)) if dmin else 1.0
    samples_per_unit = max(5, int(1.0 / (avg_nn + 1e-12)))

    segments = list(zip(waypoints[:-1], waypoints[1:]))
    if closed:
        segments.append((waypoints[-1], waypoints[0]))

    idx_all: List[int] = []
    for (x0, y0), (x1, y1) in segments:
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L * samples_per_unit))
        ts = np.linspace(0.0, 1.0, n, endpoint=True)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        for x, y in zip(xs, ys):
            d2 = (kx - x) ** 2 + (ky - y) ** 2
            idx_all.append(int(np.argmin(d2)))

    seen = set()
    order = []
    for i in idx_all:
        if i not in seen:
            order.append(i)
            seen.add(i)
    return np.array(order, dtype=int)


def select_value(z: np.ndarray, value: str) -> np.ndarray:
    v = value.lower().strip()
    if v == "abs":
        return np.abs(z)
    if v == "real":
        return np.real(z)
    if v == "imag":
        return np.imag(z)
    raise ValueError("value must be one of: abs, real, imag")


def choose_contour_cmap(value: str) -> str:
    v = value.lower().strip()
    if v == "abs":
        return "Reds"
    if v == "real" or v == "imag":
        return "seismic"
    return "viridis"


def compute_kdist_x(k: np.ndarray) -> np.ndarray:
    x = np.zeros(k.shape[0], dtype=np.float64)
    for i in range(1, k.shape[0]):
        dk = k[i, :] - k[i - 1, :]
        x[i] = x[i - 1] + float(np.linalg.norm(dk))
    return x


def reduce_by_xbin(x: np.ndarray, y: np.ndarray, tol: float, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, y

    xr = np.round(x / tol) * tol
    order = np.argsort(xr)
    xr = xr[order]
    y = y[order]

    method_l = method.lower().strip()
    if method_l not in ("mean", "max", "min"):
        raise ValueError("reduce must be one of: mean, max, min")

    out_x: List[float] = []
    out_y: List[float] = []

    i0 = 0
    while i0 < xr.size:
        i1 = i0 + 1
        while i1 < xr.size and abs(xr[i1] - xr[i0]) <= tol:
            i1 += 1
        chunk = y[i0:i1]
        if method_l == "mean":
            yy = float(np.mean(chunk))
        elif method_l == "max":
            yy = float(np.max(chunk))
        else:
            yy = float(np.min(chunk))
        out_x.append(float(xr[i0]))
        out_y.append(yy)
        i0 = i1

    return np.array(out_x, dtype=np.float64), np.array(out_y, dtype=np.float64)


def plot_path(x: np.ndarray, y: np.ndarray, xlabel: str, title: str, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(x, y, lw=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_path_real_imag(x: np.ndarray, z: np.ndarray, xlabel: str, title: str, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(x, np.real(z), lw=1.6, label="real")
    ax.plot(x, np.imag(z), lw=1.6, label="imag")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_contour_interp(kx: np.ndarray, ky: np.ndarray, z: np.ndarray, nx: int, ny: int, title: str, out_png: str, cmap: str) -> None:
    if nx < 2 or ny < 2:
        raise ValueError("interp_nx and interp_ny must be >= 2")

    pts = np.vstack([kx, ky]).T
    rpts = np.round(pts, decimals=12)
    _, uniq_idx = np.unique(rpts, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)

    tx = kx[uniq_idx]
    ty = ky[uniq_idx]
    tz = z[uniq_idx]

    tri = mtri.Triangulation(tx, ty)
    interp = mtri.LinearTriInterpolator(tri, tz)

    gx = np.linspace(float(np.min(tx)), float(np.max(tx)), int(nx))
    gy = np.linspace(float(np.min(ty)), float(np.max(ty)), int(ny))
    GX, GY = np.meshgrid(gx, gy)
    GZ = interp(GX, GY)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    pcm = ax.pcolormesh(GX, GY, GZ, shading="auto", cmap=cmap)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def determine_pairs(nb: int, pairs: Optional[List[Tuple[int, int]]], max_pairs: int) -> List[Tuple[int, int]]:
    if pairs is None:
        return [(i, i) for i in range(nb)]
    if len(pairs) > 0:
        return pairs
    out: List[Tuple[int, int]] = []
    for i in range(nb):
        for j in range(nb):
            out.append((i, j))
            if len(out) >= max_pairs:
                return out
    return out


def parse_pairs_cli(pairs_str: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if pairs_str is None:
        return None
    s = pairs_str.strip().lower()
    if s == "all":
        return []
    if not s:
        return None
    out: List[Tuple[int, int]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        ij = chunk.split(",")
        if len(ij) != 2:
            raise ValueError("Use --pairs 'all' or 'i,j;i,j' (0-based).")
        out.append((int(ij[0].strip()), int(ij[1].strip())))
    return out


def parse_excitons_cli(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: List[int] = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot SP vme (omesp) and/or k-resolved excitonic V0N(k) (omeexk).")
    ap.add_argument("--config", type=str, default=None, help="JSON config file")

    ap.add_argument("--omesp", type=str, default=None, help="Path to ome_linear_sp_*.omesp")
    ap.add_argument("--omeexk", type=str, default=None, help="Path to ome_linear_ex_k_*.omeexk")

    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--mode", choices=["path", "contour"], default=None)

    ap.add_argument("--component", choices=["x", "y", "z"], default=None)
    ap.add_argument("--value", choices=["abs", "real", "imag"], default=None)

    ap.add_argument("--pairs", type=str, default=None, help="SP band pairs: 'all' or 'i,j;i,j' (0-based)")
    ap.add_argument("--max-pairs", type=int, default=None)

    ap.add_argument("--excitons", type=str, default=None, help="Excitons to plot from omeexk (1-based), e.g. '1,2,5'")

    ap.add_argument("--order", choices=["snake", "row"], default=None)
    ap.add_argument("--kpath", type=str, default=None)
    ap.add_argument("--closed", action="store_true")

    ap.add_argument("--use-kdist", action="store_true")
    ap.add_argument("--use-kx", action="store_true")

    ap.add_argument("--reduce", choices=["mean", "max", "min"], default=None)
    ap.add_argument("--reduce-tol", type=float, default=None)

    ap.add_argument("--interp-nx", type=int, default=None)
    ap.add_argument("--interp-ny", type=int, default=None)

    ap.add_argument("--no-match-kpoints", action="store_true", help="Do not match ordering by k keys")
    ap.add_argument("--match-tol", type=float, default=None)

    # NEW
    ap.add_argument("--plot-real-imag", action="store_true", help="(path mode) Plot real+imag together on same axes")

    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else None

    omesp_file = args.omesp or (cfg.omesp_file if cfg else None)
    omeexk_file = args.omeexk or (cfg.omeexk_file if cfg else None)

    if omesp_file is None and omeexk_file is None:
        raise SystemExit("Provide at least one of --omesp or --omeexk (or via --config).")

    out_dir = args.out_dir or (cfg.out_dir if cfg else "vme_plots")
    mode = args.mode or (cfg.mode if cfg else "path")
    component = (args.component or (cfg.component if cfg else "x")).lower()
    value = (args.value or (cfg.value if cfg else "abs")).lower()

    order_mode = args.order or (cfg.order if cfg else "snake")
    kpath_file = args.kpath or (cfg.kpath_file if cfg else None)
    closed = bool(args.closed or (cfg.closed if cfg else False))

    use_kdist = bool(cfg.use_kdist if cfg else True)
    if args.use_kdist:
        use_kdist = True
    if args.use_kx:
        use_kdist = False

    reduce_method = (args.reduce or (cfg.reduce if cfg else "mean")).lower()
    reduce_tol = float(args.reduce_tol or (cfg.reduce_tol if cfg else 1e-10))

    interp_nx = int(args.interp_nx or (cfg.interp_nx if cfg else 300))
    interp_ny = int(args.interp_ny or (cfg.interp_ny if cfg else 300))

    max_pairs = int(args.max_pairs or (cfg.max_pairs if cfg else 64))

    pairs_cli = parse_pairs_cli(args.pairs)
    pairs_cfg = cfg.pairs if cfg else None
    pairs = pairs_cli if pairs_cli is not None else pairs_cfg

    excitons_cli = parse_excitons_cli(args.excitons)
    excitons_cfg = cfg.excitons if cfg else None
    excitons = excitons_cli if excitons_cli is not None else excitons_cfg

    match_kpoints = bool(cfg.match_kpoints if cfg else True)
    if args.no_match_kpoints:
        match_kpoints = False
    match_tol = float(args.match_tol or (cfg.match_tol if cfg else 1e-10))

    # NEW
    path_plot_real_imag = bool(cfg.path_plot_real_imag if cfg else False)
    if args.plot_real_imag:
        path_plot_real_imag = True

    if component not in AXIS_MAP:
        raise SystemExit("component must be x, y, or z")

    _safe_mkdir(out_dir)
    ax = AXIS_MAP[component]

    # Load SP
    k_sp = None
    E_sp = None
    vme_sp = None
    nb = None
    if omesp_file is not None:
        k_sp, E_sp, vme_sp = read_omesp_linear(omesp_file)
        nb = int(E_sp.shape[1])

    # Load EX
    k_ex = None
    v_ex_k = None
    nexc = None
    if omeexk_file is not None:
        k_ex, v_ex_k = read_omeexk(omeexk_file)
        nexc = int(v_ex_k.shape[1])

    # Choose reference k set for ordering/plotting
    if k_sp is not None:
        k_ref = k_sp
    else:
        k_ref = k_ex

    kx_ref, ky_ref = k_ref[:, 0], k_ref[:, 1]
    grid_info = infer_rect_grid(kx_ref, ky_ref, tol=1e-10)

    # Build ordering for path mode
    if mode == "path":
        if kpath_file:
            waypoints = read_kpath_waypoints(kpath_file)
            order = build_kpath_nearest(kx_ref, ky_ref, waypoints, closed=closed)
        else:
            if grid_info is None:
                order = np.lexsort((kx_ref, ky_ref))
            else:
                _, _, _, _, order_row, order_snake = grid_info
                order = order_snake if order_mode == "snake" else order_row
    else:
        order = None

    # Match ordering by k keys if needed
    if match_kpoints:
        if k_sp is not None and not np.shares_memory(k_sp, k_ref):
            idx = match_k_order(k_ref, k_sp, tol=match_tol)
            k_sp = k_sp[idx, :]
            E_sp = E_sp[idx, :]
            vme_sp = vme_sp[idx, :, :, :]
        if k_ex is not None and not np.shares_memory(k_ex, k_ref):
            idx = match_k_order(k_ref, k_ex, tol=match_tol)
            k_ex = k_ex[idx, :]
            v_ex_k = v_ex_k[idx, :, :]

    def get_x_for_path(k_ordered: np.ndarray) -> Tuple[np.ndarray, str]:
        if use_kdist:
            return compute_kdist_x(k_ordered), "k distance"
        return k_ordered[:, 0].astype(np.float64), "kx"

    # Colormap rule for contour
    cmap_contour = choose_contour_cmap(value)

    # -------------------------
    # Plot SP vme
    # -------------------------
    if vme_sp is not None:
        pairs_final = determine_pairs(nb, pairs, max_pairs)

        if mode == "path":
            ko = k_ref[order, :]
            x_raw, xlabel = get_x_for_path(ko)

            for (i, j) in pairs_final:
                if i < 0 or i >= nb or j < 0 or j >= nb:
                    continue

                z = vme_sp[order, i, j, ax]

                if path_plot_real_imag:
                    if use_kdist:
                        x_plot = x_raw
                        z_plot = z
                    else:
                        # For real/imag together, reduce each separately then plot on same x.
                        xr, yr = reduce_by_xbin(x_raw, np.real(z), tol=reduce_tol, method=reduce_method)
                        xi, yi = reduce_by_xbin(x_raw, np.imag(z), tol=reduce_tol, method=reduce_method)
                        # assume same bins
                        x_plot = xr
                        z_plot = yr + 1j * yi

                    title = f"sp_vme_{component} real+imag bands (i,j)=({i},{j})"
                    out_png = os.path.join(out_dir, f"sp_vme_{component}_real_imag_i{i}_j{j}_path.png")
                    plot_path_real_imag(x_plot, z_plot, xlabel, title, out_png)
                    continue

                y = select_value(z, value=value)

                if not use_kdist:
                    x_plot, y_plot = reduce_by_xbin(x_raw, y.astype(np.float64), tol=reduce_tol, method=reduce_method)
                else:
                    x_plot, y_plot = x_raw, y

                title = f"sp_vme_{component} {value} bands (i,j)=({i},{j})"
                out_png = os.path.join(out_dir, f"sp_vme_{component}_{value}_i{i}_j{j}_path.png")
                plot_path(x_plot, y_plot, xlabel, title, out_png)

            print(f"[OK] wrote SP path plots to {out_dir}")

        elif mode == "contour":
            kx, ky = k_ref[:, 0], k_ref[:, 1]
            for (i, j) in pairs_final:
                if i < 0 or i >= nb or j < 0 or j >= nb:
                    continue
                zc = select_value(vme_sp[:, i, j, ax], value=value)
                title = f"sp_vme_{component} {value} bands (i,j)=({i},{j})"
                out_png = os.path.join(out_dir, f"sp_vme_{component}_{value}_i{i}_j{j}_contour.png")
                plot_contour_interp(kx, ky, zc, interp_nx, interp_ny, title, out_png, cmap=cmap_contour)

            print(f"[OK] wrote SP contour plots to {out_dir}")

    # -------------------------
    # Plot EX V0N(k)
    # -------------------------
    if v_ex_k is not None and excitons is not None and len(excitons) > 0:
        ex_idx = [n - 1 for n in excitons]
        for n0, n1 in zip(excitons, ex_idx):
            if n1 < 0 or n1 >= nexc:
                raise ValueError(f"Requested exciton {n0} out of range (1..{nexc}).")

        if mode == "path":
            ko = k_ref[order, :]
            x_raw, xlabel = get_x_for_path(ko)

            for n0, n1 in zip(excitons, ex_idx):
                z = v_ex_k[order, n1, ax]

                if path_plot_real_imag:
                    if use_kdist:
                        x_plot = x_raw
                        z_plot = z
                    else:
                        xr, yr = reduce_by_xbin(x_raw, np.real(z), tol=reduce_tol, method=reduce_method)
                        xi, yi = reduce_by_xbin(x_raw, np.imag(z), tol=reduce_tol, method=reduce_method)
                        x_plot = xr
                        z_plot = yr + 1j * yi

                    title = f"ex_V0N(k)_{component} real+imag exciton N={n0}"
                    out_png = os.path.join(out_dir, f"exk_V0N_{component}_real_imag_N{n0}_path.png")
                    plot_path_real_imag(x_plot, z_plot, xlabel, title, out_png)
                    continue

                y = select_value(z, value=value)

                if not use_kdist:
                    x_plot, y_plot = reduce_by_xbin(x_raw, y.astype(np.float64), tol=reduce_tol, method=reduce_method)
                else:
                    x_plot, y_plot = x_raw, y

                title = f"ex_V0N(k)_{component} {value} exciton N={n0}"
                out_png = os.path.join(out_dir, f"exk_V0N_{component}_{value}_N{n0}_path.png")
                plot_path(x_plot, y_plot, xlabel, title, out_png)

            print(f"[OK] wrote EX path plots to {out_dir}")

        elif mode == "contour":
            kx, ky = k_ref[:, 0], k_ref[:, 1]
            for n0, n1 in zip(excitons, ex_idx):
                zc = select_value(v_ex_k[:, n1, ax], value=value)
                title = f"ex_V0N(k)_{component} {value} exciton N={n0}"
                out_png = os.path.join(out_dir, f"exk_V0N_{component}_{value}_N{n0}_contour.png")
                plot_contour_interp(kx, ky, zc, interp_nx, interp_ny, title, out_png, cmap=cmap_contour)

            print(f"[OK] wrote EX contour plots to {out_dir}")


if __name__ == "__main__":
    main()
