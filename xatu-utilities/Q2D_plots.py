#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import argparse
import glob

import numpy as np


def heatmap_cmap():
    """Return the common diverging colormap used by every heatmap.

    Importing colorcet registers ``cet_bkr`` with Matplotlib.  Seismic is
    retained as a fallback so the script remains usable without colorcet.
    """
    from matplotlib import colormaps

    try:
        import colorcet as cc  # noqa: F401  (registers Colorcet colormaps)

        return colormaps.get_cmap("cet_bkr")
    except (ImportError, ValueError):
        return colormaps.get_cmap("seismic")


@dataclass
class Q2DQBlock:
    q_index: int
    q: np.ndarray
    matrix: np.ndarray


@dataclass
class Q2DChebFile:
    path: Path
    object_name: str
    screening_mode: str
    representation: str
    stored_matrix: str
    nGs: int
    Nz: int
    matrix_rows: int
    matrix_cols: int
    zmin: float
    zmax: float
    x_nodes: np.ndarray
    z_nodes: np.ndarray
    weights: np.ndarray
    g_vectors: np.ndarray
    blocks: list


def _parse_header_line(line, metadata):
    text = line[1:].strip()
    if not text:
        return None

    if text.startswith("G_index"):
        parts = text.split()
        if len(parts) > 1 and parts[1] == "=":
            parts.pop(1)
        if len(parts) < 5:
            raise ValueError(f"Malformed G_index header line: {line.rstrip()}")
        g_index = int(parts[1])
        g_vector = [float(parts[2]), float(parts[3]), float(parts[4])]
        metadata.setdefault("g_vectors", {})[g_index] = g_vector
        return None

    if text.startswith("q_index"):
        parts = text.split()
        if len(parts) > 1 and parts[1] == "=":
            parts.pop(1)
        if len(parts) < 2:
            raise ValueError(f"Malformed q_index header line: {line.rstrip()}")

        q_index = int(parts[1])
        q = None
        if "qx" in parts:
            q_values = []
            for key in ("qx", "qy", "qz"):
                try:
                    q_values.append(float(parts[parts.index(key) + 1]))
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Malformed q_index header line: {line.rstrip()}") from error
            q = np.array(q_values, dtype=float)
        return ("q_index", q_index, q)

    if text.startswith("q"):
        parts = text.split()
        if parts[0] != "q":
            return None
        if len(parts) > 1 and parts[1] == "=":
            parts.pop(1)
        if len(parts) < 4:
            raise ValueError(f"Malformed q header line: {line.rstrip()}")
        return ("q", np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))

    for key in ("x_nodes", "z_nodes_A", "weights"):
        if text.startswith(key):
            metadata[key] = np.array(parse_numeric_list_after_key(text, key), dtype=float)
            return None

    if "=" in text:
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"nGs", "Nz", "matrix_rows", "matrix_cols", "n_qpoints"}:
            metadata[key] = int(value.split()[0])
        elif key in {"zmin", "zmax"}:
            metadata[key] = float(value.split()[0])
        elif key == "screening.mode":
            metadata["screening_mode"] = value
        elif key == "object":
            metadata["object_name"] = value
        elif key == "representation":
            metadata["representation"] = value
        elif key == "stored_matrix":
            metadata["stored_matrix"] = value
        return None

    parts = text.split()
    if len(parts) >= 2:
        key = parts[0]
        if key in {"nGs", "Nz", "matrix_rows", "matrix_cols", "n_qpoints"}:
            metadata[key] = int(parts[1])
        elif key in {"zmin", "zmax"}:
            metadata[key] = float(parts[1])
        elif key == "screening.mode":
            metadata["screening_mode"] = " ".join(parts[1:])
        elif key == "object":
            metadata["object_name"] = " ".join(parts[1:])
        elif key == "representation":
            metadata["representation"] = " ".join(parts[1:])
        elif key == "stored_matrix":
            metadata["stored_matrix"] = " ".join(parts[1:])
    return None


def parse_numeric_list_after_key(text, key):
    parts = text.split()
    if not parts or parts[0] != key:
        raise ValueError(f"Expected numeric list headed by {key}: {text}")
    values = parts[1:]
    if values and values[0] == "=":
        values = values[1:]
    try:
        return [float(value) for value in values]
    except ValueError as error:
        raise ValueError(f"Malformed numeric list for {key}: {text}") from error


def _metadata_value(metadata, key, path):
    if key not in metadata:
        raise ValueError(f"{path}: missing required header metadata: {key}")
    return metadata[key]


def _read_matrix_rows(input_file, path, q_index, matrix_rows, matrix_cols, start_line_number):
    rows = []
    line_number = start_line_number
    while len(rows) < matrix_rows:
        line = input_file.readline()
        if line == "":
            raise ValueError(
                f"{path}: q_index {q_index} has {len(rows)} numeric matrix rows "
                f"but expected {matrix_rows}"
            )
        line_number += 1
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            raise ValueError(
                f"{path}: q_index {q_index} has {len(rows)} numeric matrix rows before "
                f"header at line {line_number}, expected {matrix_rows}"
            )

        values = np.fromstring(stripped, sep=" ")
        expected_values = 2 * matrix_cols
        if values.size != expected_values:
            raise ValueError(
                f"{path}: q_index {q_index} numeric row {len(rows) + 1} at line {line_number} "
                f"has {values.size} values but expected 2*matrix_cols={expected_values}"
            )
        rows.append(values[0::2] + 1j * values[1::2])
    return np.vstack(rows), line_number


def read_q2d_cheb_file(path: Path) -> Q2DChebFile:
    path = Path(path)
    metadata = {}
    blocks = []
    line_number = 0

    with path.open("r", encoding="utf-8") as input_file:
        while True:
            line = input_file.readline()
            if line == "":
                break
            line_number += 1
            stripped = line.strip()
            if not stripped:
                continue

            if not stripped.startswith("#"):
                raise ValueError(f"{path}: Numeric matrix row found before q_index.")

            parsed = _parse_header_line(stripped, metadata)
            if parsed is None:
                continue
            if parsed[0] != "q_index":
                raise ValueError(f"{path}: q vector found before q_index at line {line_number}")

            q_index = parsed[1]
            q = parsed[2]

            n_gs = _metadata_value(metadata, "nGs", path)
            nz = _metadata_value(metadata, "Nz", path)
            matrix_rows = _metadata_value(metadata, "matrix_rows", path)
            expected_dim = n_gs * nz
            matrix_cols = metadata.get("matrix_cols", matrix_rows)
            metadata["matrix_cols"] = matrix_cols
            if matrix_rows != expected_dim:
                raise ValueError(
                    f"{path}: matrix_rows={matrix_rows} but expected nGs*Nz={expected_dim} "
                    f"(nGs={n_gs}, Nz={nz})"
                )
            if matrix_cols != expected_dim:
                raise ValueError(
                    f"{path}: matrix_cols={matrix_cols} but expected nGs*Nz={expected_dim} "
                    f"(nGs={n_gs}, Nz={nz})"
                )

            if q is None:
                while True:
                    q_line = input_file.readline()
                    if q_line == "":
                        raise ValueError(f"{path}: q_index {q_index} is missing following q header")
                    line_number += 1
                    q_stripped = q_line.strip()
                    if not q_stripped:
                        continue
                    if not q_stripped.startswith("#"):
                        raise ValueError(f"{path}: q_index {q_index} is missing following q header")
                    q_parsed = _parse_header_line(q_stripped, metadata)
                    if q_parsed is None:
                        continue
                    if q_parsed[0] != "q":
                        raise ValueError(f"{path}: expected q header after q_index {q_index}")
                    q = q_parsed[1]
                    break

            matrix, line_number = _read_matrix_rows(
                input_file,
                path,
                q_index,
                matrix_rows,
                matrix_cols,
                line_number,
            )
            if matrix.shape != (matrix_rows, matrix_cols):
                raise ValueError(
                    f"{path}: q_index {q_index} matrix shape is {matrix.shape}, "
                    f"expected ({matrix_rows}, {matrix_cols})"
                )
            blocks.append(Q2DQBlock(q_index=q_index, q=q, matrix=matrix))

    n_gs = _metadata_value(metadata, "nGs", path)
    nz = _metadata_value(metadata, "Nz", path)
    matrix_rows = _metadata_value(metadata, "matrix_rows", path)
    matrix_cols = metadata.get("matrix_cols", matrix_rows)
    z_nodes = _metadata_value(metadata, "z_nodes_A", path)
    x_nodes = metadata.get("x_nodes", np.empty(0, dtype=float))
    weights = metadata.get("weights", np.empty(0, dtype=float))
    zmin = metadata.get("zmin", float(np.min(z_nodes)))
    zmax = metadata.get("zmax", float(np.max(z_nodes)))

    expected_dim = n_gs * nz
    if matrix_rows != expected_dim:
        raise ValueError(
            f"{path}: matrix_rows={matrix_rows} but expected nGs*Nz={expected_dim} "
            f"(nGs={n_gs}, Nz={nz})"
        )
    if matrix_cols != expected_dim:
        raise ValueError(
            f"{path}: matrix_cols={matrix_cols} but expected nGs*Nz={expected_dim} "
            f"(nGs={n_gs}, Nz={nz})"
        )
    if len(z_nodes) != nz:
        raise ValueError(f"{path}: z_nodes_A has {len(z_nodes)} entries but Nz={nz}")
    if x_nodes.size not in {0, nz}:
        raise ValueError(f"{path}: x_nodes has {x_nodes.size} entries but Nz={nz}")
    if weights.size not in {0, nz}:
        raise ValueError(f"{path}: weights has {weights.size} entries but Nz={nz}")
    if not blocks:
        raise ValueError(f"{path}: no q blocks found")

    g_metadata = metadata.get("g_vectors", {})
    if len(g_metadata) == n_gs:
        g_vectors = np.array([g_metadata[index] for index in range(n_gs)], dtype=float)
    else:
        g_vectors = np.empty((0, 3), dtype=float)

    return Q2DChebFile(
        path=path,
        object_name=metadata.get("object_name", "unknown"),
        screening_mode=metadata.get("screening_mode", ""),
        representation=metadata.get("representation", ""),
        stored_matrix=metadata.get("stored_matrix", ""),
        nGs=n_gs,
        Nz=nz,
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        zmin=zmin,
        zmax=zmax,
        x_nodes=x_nodes,
        z_nodes=z_nodes,
        weights=weights,
        g_vectors=g_vectors,
        blocks=blocks,
    )


def find_g_index(g_vectors, target, tol=1e-8):
    target = np.asarray(target, dtype=float)
    if target.shape != (3,):
        raise ValueError("G vector target must have exactly three components")

    differences = np.linalg.norm(g_vectors - target[None, :], axis=1)
    matches = np.where(differences <= tol)[0]
    if len(matches) == 0:
        raise ValueError(f"G vector {target.tolist()} not found within tolerance {tol}")
    return int(matches[0])


def _parse_g_selector(selector, option_name):
    if selector is None:
        return None
    if len(selector) == 1:
        value = selector[0]
        index = int(value)
        if not np.isclose(value, index):
            raise ValueError(f"{option_name} raw index must be an integer")
        return ("index", index)
    if len(selector) == 3:
        return ("vector", selector)
    raise ValueError(f"{option_name} expects either one raw index or three vector components")


def _raw_index_from_selector(selector, option_name):
    value = selector[0]
    index = int(value)
    if not np.isclose(value, index):
        raise ValueError(
            f"{option_name} cannot be matched as a vector because G vectors are absent, "
            "and its first value is not an integer raw index"
        )
    return index


def resolve_g_indices(op, args):
    if args.g_index is not None:
        g_index = args.g_index
    elif args.g is not None:
        selector_kind, selector_value = _parse_g_selector(args.g, "--g")
        if selector_kind == "index":
            g_index = selector_value
        elif op.g_vectors.size == 0:
            raise ValueError("G vectors are not available in this file. Use --g-index and --gp-index.")
        else:
            g_index = find_g_index(op.g_vectors, selector_value, args.g_tol)
    else:
        g_index = 0

    if args.gp_index is not None:
        gp_index = args.gp_index
    elif args.gp is not None:
        selector_kind, selector_value = _parse_g_selector(args.gp, "--gp")
        if selector_kind == "index":
            gp_index = selector_value
        elif op.g_vectors.size == 0:
            raise ValueError("G vectors are not available in this file. Use --g-index and --gp-index.")
        else:
            gp_index = find_g_index(op.g_vectors, selector_value, args.g_tol)
    else:
        gp_index = 0

    if op.g_vectors.size == 0 and (args.g is not None or args.gp is not None):
        print("Warning: G vectors not found in header. Interpreting --g and --gp as raw G indices.")

    if g_index < 0 or g_index >= op.nGs:
        raise ValueError(f"g_index={g_index} is out of bounds for nGs={op.nGs}")
    if gp_index < 0 or gp_index >= op.nGs:
        raise ValueError(f"gp_index={gp_index} is out of bounds for nGs={op.nGs}")
    return g_index, gp_index


def extract_block(op, q_block, g_index, gp_index):
    n = op.Nz
    i0 = g_index * n
    j0 = gp_index * n
    return q_block.matrix[i0:i0 + n, j0:j0 + n]


def select_q_block(q2d_file, iq):
    if iq is not None:
        for block in q2d_file.blocks:
            if block.q_index == iq:
                return block
        available = ", ".join(str(block.q_index) for block in q2d_file.blocks)
        raise ValueError(f"{q2d_file.path}: q_index {iq} not found. Available q_index values: {available}")
    if len(q2d_file.blocks) > 1:
        first = q2d_file.blocks[0]
        print(
            "Warning: file contains multiple q blocks; using "
            f"q_index = {first.q_index}. Use --iq to select another one."
        )
    return q2d_file.blocks[0]


def convert_view(block, weights, view):
    if view == "stored":
        return block
    if view == "kernel":
        if weights is None or len(weights) == 0:
            raise ValueError("--view kernel requires weights in the file header.")
        return block / weights[None, :]
    raise ValueError(f"Unknown view: {view}")


def view_label(view):
    if view == "stored":
        return "stored operator"
    if view == "kernel":
        return "unweighted kernel estimate"
    raise ValueError(f"Unknown view: {view}")


def physical_to_cheb_x(z, zmin, zmax):
    if zmax == zmin:
        raise ValueError("zmax and zmin must differ")
    return (2.0 * z - (zmax + zmin)) / (zmax - zmin)


def cheb_vandermonde_from_z(z_values, zmin, zmax, Nz):
    z_values = np.asarray(z_values, dtype=float)
    x = physical_to_cheb_x(z_values, zmin, zmax)
    x = np.clip(x, -1.0, 1.0)
    theta = np.arccos(x)
    return np.cos(np.outer(theta, np.arange(Nz)))


def cheb_vandermonde_from_x(x_values, Nz):
    x_values = np.asarray(x_values, dtype=float)
    x_values = np.clip(x_values, -1.0, 1.0)
    theta = np.arccos(x_values)
    return np.cos(np.outer(theta, np.arange(Nz)))


def default_dense_nz(op, dense_nz):
    if dense_nz is not None:
        return dense_nz
    return max(200, 5 * op.Nz)


def reconstruct_block(op, block, dense_nz):
    zmin = float(op.zmin)
    zmax = float(op.zmax)
    z_dense = np.linspace(zmin, zmax, dense_nz)
    if op.x_nodes.size == op.Nz:
        c_nodes = cheb_vandermonde_from_x(op.x_nodes, op.Nz)
    else:
        c_nodes = cheb_vandermonde_from_z(op.z_nodes, zmin, zmax, op.Nz)
    c_inv = np.linalg.inv(c_nodes)
    t_dense = cheb_vandermonde_from_z(z_dense, zmin, zmax, op.Nz)
    coeff = c_inv @ block @ c_inv.T
    dense = t_dense @ coeff @ t_dense.T
    return z_dense, dense, coeff, c_nodes


def reconstruction_relative_error(block, coeff, c_nodes):
    reconstructed_at_nodes = c_nodes @ coeff @ c_nodes.T
    norm = np.linalg.norm(block)
    if norm == 0.0:
        return np.linalg.norm(reconstructed_at_nodes - block)
    return np.linalg.norm(reconstructed_at_nodes - block) / norm


def component_values(matrix, component):
    if component == "real":
        return matrix.real
    if component == "imag":
        return matrix.imag
    if component == "abs":
        return np.abs(matrix)
    if component == "phase":
        return np.angle(matrix)
    raise ValueError(f"Unknown component: {component}")


def component_label(component):
    return component_object_label(component, "unknown", "stored")


def object_symbol(object_name):
    if object_name == "inverse_epsilon_operator":
        return r"\varepsilon^{-1}"
    if object_name == "epsilon_operator":
        return r"\varepsilon"
    if object_name == "v_grid":
        return "v"
    return "O"


def component_object_label(component, object_name, view):
    symbol = object_symbol(object_name)
    descriptor = view_label(view)
    if component == "real":
        return rf"Re {descriptor} ${symbol}_{{GG'}}(q; z,z')$"
    if component == "imag":
        return rf"Im {descriptor} ${symbol}_{{GG'}}(q; z,z')$"
    if component == "abs":
        return rf"{descriptor} $|{symbol}_{{GG'}}(q; z,z')|$"
    if component == "phase":
        return rf"phase {descriptor} ${symbol}_{{GG'}}(q; z,z')$"
    raise ValueError(f"Unknown component: {component}")


def output_prefix(path):
    return Path(path).with_suffix("").name


def output_from_alias(args):
    if getattr(args, "output", None) is not None:
        return args.output
    if getattr(args, "plot", None) is not None:
        return args.plot
    return None


def z_selector_label(value):
    text = str(value).strip().lower()
    if text in {"bottom", "center", "top"}:
        return text
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def q_abs(q_block):
    return float(np.linalg.norm(q_block.q))


def component_prefix(component):
    if component == "real":
        return "Re"
    if component == "imag":
        return "Im"
    if component == "abs":
        return "Abs"
    if component == "phase":
        return "Phase"
    raise ValueError(f"Unknown component: {component}")


def load_operator_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched pattern: {pattern}")
    return [Path(item) for item in files]


def save_dense_matrix(path, z_values, dense, header="z zprime Re Im Abs"):
    rows = []
    for i, z in enumerate(z_values):
        for j, zprime in enumerate(z_values):
            value = dense[i, j]
            rows.append((z, zprime, value.real, value.imag, abs(value)))
    np.savetxt(path, np.array(rows, dtype=float), header=header)


def finite_plot_values(values):
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Cannot plot heatmap: no finite values found.")
    return finite


def print_normalization_info(mode, finite, extra=None):
    print(f"Normalization: {mode}")
    print(f"min value: {np.min(finite):.16g}")
    print(f"max value: {np.max(finite):.16g}")
    print(f"1st percentile: {np.percentile(finite, 1):.16g}")
    print(f"99th percentile: {np.percentile(finite, 99):.16g}")
    if extra:
        for key, value in extra.items():
            print(f"{key}: {value:.16g}")


def heatmap_norm_and_cmap(values, norm_mode):
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    finite = finite_plot_values(values)

    if norm_mode == "linear":
        print_normalization_info("linear", finite)
        return None

    if norm_mode == "log":
        if np.any(finite <= 0.0):
            raise ValueError("--norm log requires strictly positive plotted values.")
        vmin = np.min(finite[finite > 0.0])
        vmax = np.max(finite)
        if vmin == vmax:
            vmax = vmin * (1.0 + 1e-12)
        print_normalization_info("log", finite, {"vmin": vmin, "vmax": vmax})
        return LogNorm(vmin=vmin, vmax=vmax)

    if norm_mode == "symlog":
        vmax = np.max(np.abs(finite))
        if vmax == 0.0:
            vmax = 1.0
        abs_finite = np.abs(finite)
        linthresh = max(vmax * 1e-3, np.percentile(abs_finite, 5))
        if linthresh == 0.0:
            linthresh = vmax * 1e-3
        print_normalization_info("symlog", finite, {"linthresh": linthresh})
        return SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax)

    if norm_mode == "percentile":
        low = np.percentile(finite, 2)
        high = np.percentile(finite, 98)
        if low == high:
            spread = max(abs(low) * 1e-6, 1e-12)
            low -= spread
            high += spread
        print_normalization_info("percentile", finite, {"low": low, "high": high})
        return Normalize(vmin=low, vmax=high)

    raise ValueError(f"Unknown normalization mode: {norm_mode}")


def plot_heatmap(path, op, q_block, z_dense, dense, component, view, norm_mode, g_index, gp_index):
    import matplotlib.pyplot as plt

    values = component_values(dense, component)
    norm = heatmap_norm_and_cmap(values, norm_mode)
    cmap = heatmap_cmap()
    zmin = float(z_dense[0])
    zmax = float(z_dense[-1])

    plt.figure(figsize=(6, 5))
    plt.imshow(
        values,
        origin="lower",
        extent=[zmin, zmax, zmin, zmax],
        aspect="equal",
        norm=norm,
        cmap=cmap,
    )
    plt.colorbar(label=component_object_label(component, op.object_name, view))
    plt.xlabel("z' (Å)")
    plt.ylabel("z (Å)")
    plt.title(
        f"{op.object_name}\n"
        f"{view_label(view)}, q_index = {q_block.q_index}, "
        f"q = ({q_block.q[0]:.6g}, {q_block.q[1]:.6g}, {q_block.q[2]:.6g}), "
        f"G index = {g_index}, G' index = {gp_index}, {component}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_line(path, x, y, xlabel, ylabel, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_qz_heatmap(path, q_values, z_values, values_qz, colorbar_label, title, norm_mode, ylabel="z (Å)"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.8))
    q_edges = centers_to_edges(q_values)
    z_edges = centers_to_edges(z_values)
    norm = heatmap_norm_and_cmap(values_qz, norm_mode)
    cmap = heatmap_cmap()
    mesh = plt.pcolormesh(q_edges, z_edges, values_qz.T, shading="auto", norm=norm, cmap=cmap)
    plt.colorbar(mesh, label=colorbar_label)
    plt.xlabel(r"|q| ($\mathrm{\AA}^{-1}$)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def centers_to_edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot build edges from an empty coordinate array")
    if values.size == 1:
        width = max(abs(values[0]) * 0.1, 0.5)
        return np.array([values[0] - width, values[0] + width], dtype=float)
    adjusted = values.copy()
    raw_diffs = np.diff(values)
    positive_diffs = raw_diffs[raw_diffs > 0.0]
    fallback = positive_diffs.min() if positive_diffs.size else max(abs(values[0]) * 0.1, 0.5)
    min_step = fallback * 1e-6
    for index in range(1, adjusted.size):
        if adjusted[index] <= adjusted[index - 1]:
            adjusted[index] = adjusted[index - 1] + min_step
    diffs = np.diff(adjusted)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (adjusted[:-1] + adjusted[1:])
    edges[0] = adjusted[0] - 0.5 * diffs[0]
    edges[-1] = adjusted[-1] + 0.5 * diffs[-1]
    return edges


def parse_z_value(value, zmin, zmax):
    text = str(value).strip().lower()
    if text == "bottom":
        return zmin
    if text == "center":
        return 0.0
    if text == "top":
        return zmax
    return float(value)


def print_inspect(op):
    print(f"file: {op.path}")
    print(f"screening.mode: {op.screening_mode}")
    print(f"object: {op.object_name}")
    print(f"representation: {op.representation}")
    print(f"stored_matrix: {op.stored_matrix}")
    print(f"q blocks: {len(op.blocks)}")
    for block in op.blocks:
        print(f"  q_index {block.q_index}: {block.q[0]:.16g} {block.q[1]:.16g} {block.q[2]:.16g}")
    print(f"nGs: {op.nGs}")
    print(f"Nz: {op.Nz}")
    print(f"matrix shape per q block: ({op.matrix_rows}, {op.matrix_cols})")
    print(f"z range: {op.zmin:.16g} {op.zmax:.16g}")
    print("z nodes:")
    for index, z_value in enumerate(op.z_nodes):
        print(f"  {index}: {z_value:.16g}")
    if op.g_vectors.size == 0:
        print("available G vectors: not found in header")
    else:
        print("available G vectors:")
        for index, vector in enumerate(op.g_vectors):
            print(f"  {index}: {vector[0]:.16g} {vector[1]:.16g} {vector[2]:.16g}")


def add_g_selection_arguments(parser):
    parser.add_argument("--g-index", type=int, default=None, help="Raw G index.")
    parser.add_argument("--gp-index", type=int, default=None, help="Raw G' index.")
    parser.add_argument(
        "--g",
        nargs="+",
        type=float,
        default=None,
        metavar="G",
        help="G selector: one raw index, or three vector components.",
    )
    parser.add_argument(
        "--gp",
        nargs="+",
        type=float,
        default=None,
        metavar="GP",
        help="G' selector: one raw index, or three vector components.",
    )
    parser.add_argument("--g-tol", type=float, default=1e-8, help="Tolerance for G-vector matching.")


def add_common_operator_arguments(parser):
    add_g_selection_arguments(parser)
    parser.add_argument(
        "--component",
        choices=["real", "imag", "abs", "phase"],
        default="real",
        help="Matrix component to output or plot.",
    )
    parser.add_argument("--view", choices=["stored", "kernel"], default="stored", help="Matrix view to plot.")
    parser.add_argument("--dense-nz", type=int, default=None, help="Number of dense z-grid points.")
    parser.add_argument(
        "--check-reconstruction",
        action="store_true",
        help="Print the relative reconstruction error at the Chebyshev nodes.",
    )


def add_plot_alias(parser):
    parser.add_argument(
        "--plot",
        dest="plot",
        default=None,
        help=argparse.SUPPRESS,
    )


def add_heatmap_norm_argument(parser):
    parser.add_argument(
        "--norm",
        choices=["linear", "log", "symlog", "percentile"],
        default="symlog",
        help="Color normalization for heatmap plots.",
    )


def reconstruct_selected_block(op, q_block, args):
    g_index, gp_index = resolve_g_indices(op, args)
    block = extract_block(op, q_block, g_index, gp_index)
    block = convert_view(block, op.weights, args.view)
    dense_nz = default_dense_nz(op, args.dense_nz)
    if dense_nz < 2:
        raise ValueError("--dense-nz must be at least 2")
    z_dense, dense, coeff, c_nodes = reconstruct_block(op, block, dense_nz)
    if args.check_reconstruction:
        error = reconstruction_relative_error(block, coeff, c_nodes)
        print(f"reconstruction_relative_error: {error:.16e}")
    return op, q_block, g_index, gp_index, z_dense, dense


def load_block_and_reconstruct(args):
    op = read_q2d_cheb_file(args.input)
    q_block = select_q_block(op, args.iq)
    return reconstruct_selected_block(op, q_block, args)


def command_inspect(args):
    op = read_q2d_cheb_file(args.input)
    print_inspect(op)


def command_heatmap(args):
    op, q_block, g_index, gp_index, z_dense, dense = load_block_and_reconstruct(args)

    output = output_from_alias(args)
    if output is None:
        output = (
            f"{output_prefix(op.path)}_q{q_block.q_index}_G{g_index}_Gp{gp_index}_"
            f"{args.view}_{args.component}_heatmap.png"
        )
    plot_heatmap(output, op, q_block, z_dense, dense, args.component, args.view, args.norm, g_index, gp_index)
    print(f"Wrote {output}")

    save_data = args.save_data
    if save_data is not None:
        save_dense_matrix(save_data, z_dense, dense, header=f"view={args.view} {view_label(args.view)}")
        print(f"Wrote {save_data}")


def command_cut(args):
    op, q_block, g_index, gp_index, z_dense, dense = load_block_and_reconstruct(args)
    zmin = float(z_dense[0])
    zmax = float(z_dense[-1])
    requested_z = parse_z_value(args.z, zmin, zmax)
    index = nearest_index(z_dense, requested_z)
    fixed_z = z_dense[index]

    if args.vary == "zprime":
        x_values = z_dense
        values = dense[index, :]
        header = f"view={args.view} q_index={q_block.q_index} zprime {args.component} Re Im Abs Phase fixed_z={fixed_z:.16g}"
        xlabel = "z' (Å)"
        title_fixed = f"fixed z = {fixed_z:.6g} Å"
    else:
        x_values = z_dense
        values = dense[:, index]
        header = f"view={args.view} q_index={q_block.q_index} z {args.component} Re Im Abs Phase fixed_zprime={fixed_z:.16g}"
        xlabel = "z (Å)"
        title_fixed = f"fixed z' = {fixed_z:.6g} Å"

    selected = component_values(values, args.component)
    data = np.column_stack([x_values, selected, values.real, values.imag, np.abs(values), np.angle(values)])
    save_data = args.save_data
    if save_data is not None:
        np.savetxt(save_data, data, header=header)
        print(f"Wrote {save_data}")

    output = output_from_alias(args)
    if output is None:
        output = (
            f"{output_prefix(op.path)}_G{g_index}_Gp{gp_index}_"
            f"q{q_block.q_index}_{args.view}_cut_z_{z_selector_label(args.z)}_vary_{args.vary}_{args.component}.png"
        )
    plot_line(
        output,
        x_values,
        selected,
        xlabel,
        component_object_label(args.component, op.object_name, args.view),
        (
            f"{op.object_name}\n"
            f"{view_label(args.view)}, q_index = {q_block.q_index}, "
            f"q = ({q_block.q[0]:.6g}, {q_block.q[1]:.6g}, {q_block.q[2]:.6g}), "
            f"G index = {g_index}, G' index = {gp_index}, {title_fixed}"
        ),
    )
    print(f"Wrote {output}")


def command_diag(args):
    op, q_block, g_index, gp_index, z_dense, dense = load_block_and_reconstruct(args)
    values = np.diag(dense)
    selected = component_values(values, args.component)
    data = np.column_stack([z_dense, selected, values.real, values.imag, np.abs(values), np.angle(values)])
    save_data = args.save_data
    if save_data is not None:
        np.savetxt(
            save_data,
            data,
            header=f"view={args.view} q_index={q_block.q_index} z {args.component} Re Im Abs Phase",
        )
        print(f"Wrote {save_data}")

    output = output_from_alias(args)
    if output is None:
        output = f"{output_prefix(op.path)}_q{q_block.q_index}_G{g_index}_Gp{gp_index}_{args.view}_diag_{args.component}.png"
    plot_line(
        output,
        z_dense,
        selected,
        "z (Å)",
        component_object_label(args.component, op.object_name, args.view),
        (
            f"{op.object_name}\n"
            f"{view_label(args.view)}, q_index = {q_block.q_index}, "
            f"q = ({q_block.q[0]:.6g}, {q_block.q[1]:.6g}, {q_block.q[2]:.6g}), "
            f"G index = {g_index}, G' index = {gp_index}"
        ),
    )
    print(f"Wrote {output}")


def collect_q_blocks(pattern):
    paths = load_operator_files(pattern)
    for path in paths:
        parsed = read_q2d_cheb_file(path)
        for block in parsed.blocks:
            yield parsed, block


def _sorted_batch_records(pattern, qmax):
    records = sorted(
        ((q_abs(q_block), op, q_block) for op, q_block in collect_q_blocks(pattern)),
        key=lambda record: record[0],
    )
    if qmax is not None:
        records = [record for record in records if record[0] <= qmax]
    if not records:
        raise RuntimeError(f"No files remained after applying --qmax {qmax}")
    return records


def command_batch_line(args):
    records = _sorted_batch_records(args.pattern, args.qmax)
    rows = []
    q_values = []
    selected_values = []
    object_name = records[0][1].object_name
    g_index = gp_index = None
    fixed_z = fixed_zprime = None

    for q_value, op, q_block in records:
        op, q_block, g_index, gp_index, z_dense, dense = reconstruct_selected_block(op, q_block, args)
        zmin = float(z_dense[0])
        zmax = float(z_dense[-1])
        requested_z = parse_z_value(args.z, zmin, zmax)
        requested_zprime = parse_z_value(args.zprime, zmin, zmax)
        i = nearest_index(z_dense, requested_z)
        j = nearest_index(z_dense, requested_zprime)
        fixed_z = z_dense[i]
        fixed_zprime = z_dense[j]
        value = dense[i, j]
        selected = float(component_values(np.asarray([value]), args.component)[0])
        rows.append(
            (
                q_value,
                q_block.q_index,
                q_block.q[0],
                q_block.q[1],
                q_block.q[2],
                fixed_z,
                fixed_zprime,
                value.real,
                value.imag,
                abs(value),
                np.angle(value),
                selected,
            )
        )
        q_values.append(q_value)
        selected_values.append(selected)

    save_data = args.save_data
    if save_data is not None:
        np.savetxt(
            save_data,
            np.array(rows, dtype=float),
            header="view={} q_abs q_index qx qy qz z zprime Re Im Abs Phase selected_component".format(args.view),
        )
        print(f"Wrote {save_data}")

    output = output_from_alias(args)
    if output is None:
        output = (
            f"batch_line_G{g_index}_Gp{gp_index}_{args.view}_z_{z_selector_label(args.z)}_"
            f"zp_{z_selector_label(args.zprime)}_{args.component}.png"
        )
    plot_line(
        output,
        np.asarray(q_values, dtype=float),
        np.asarray(selected_values, dtype=float),
        r"|q| ($\mathrm{\AA}^{-1}$)",
        component_object_label(args.component, object_name, args.view),
        (
            f"{object_name}\n"
            f"{view_label(args.view)}, G index = {g_index}, G' index = {gp_index}, "
            f"z = {fixed_z:.6g} Å, z' = {fixed_zprime:.6g} Å"
        ),
    )
    print(f"Wrote {output}")


def command_batch_diagmap(args):
    records = _sorted_batch_records(args.pattern, args.qmax)
    first_op = records[0][1]
    first_range = (float(first_op.zmin), float(first_op.zmax))
    rows = []
    q_values = []
    values_qz = []
    z_reference = None
    object_name = first_op.object_name
    g_index = gp_index = None

    for q_value, op, q_block in records:
        z_range = (float(op.zmin), float(op.zmax))
        if not np.allclose(z_range, first_range, rtol=1e-10, atol=1e-10):
            raise RuntimeError("All files in batch-diagmap must have the same z range.")

        op, q_block, g_index, gp_index, z_dense, dense = reconstruct_selected_block(op, q_block, args)
        if z_reference is None:
            z_reference = z_dense
        elif not np.allclose(z_dense, z_reference, rtol=1e-10, atol=1e-10):
            raise RuntimeError("All files in batch-diagmap must have the same z range.")

        diag_values = np.diag(dense)
        selected = component_values(diag_values, args.component)
        values_qz.append(selected)
        q_values.append(q_value)
        for z_value, value, selected_value in zip(z_dense, diag_values, selected):
            rows.append(
                (
                    q_value,
                    q_block.q_index,
                    q_block.q[0],
                    q_block.q[1],
                    q_block.q[2],
                    z_value,
                    value.real,
                    value.imag,
                    abs(value),
                    np.angle(value),
                    selected_value,
                )
            )

    save_data = args.save_data
    if save_data is not None:
        np.savetxt(
            save_data,
            np.array(rows, dtype=float),
            header="view={} q_abs q_index qx qy qz z Re Im Abs Phase selected_component".format(args.view),
        )
        print(f"Wrote {save_data}")

    output = output_from_alias(args)
    if output is None:
        output = f"batch_diagmap_G{g_index}_Gp{gp_index}_{args.view}_{args.component}.png"
    plot_qz_heatmap(
        output,
        np.asarray(q_values, dtype=float),
        z_reference,
        np.asarray(values_qz, dtype=float),
        component_object_label(args.component, object_name, args.view),
        f"{object_name}\n{view_label(args.view)}, G index = {g_index}, G' index = {gp_index}",
        args.norm,
    )
    print(f"Wrote {output}")


def command_batch_cutmap(args):
    records = _sorted_batch_records(args.pattern, args.qmax)
    rows = []
    q_values = []
    values_qz = []
    z_reference = None
    fixed_value_actual = None
    object_name = records[0][1].object_name
    g_index = gp_index = None

    for q_value, op, q_block in records:
        op, q_block, g_index, gp_index, z_dense, dense = reconstruct_selected_block(op, q_block, args)
        if z_reference is None:
            z_reference = z_dense
        elif not np.allclose(z_dense, z_reference, rtol=1e-10, atol=1e-10):
            raise RuntimeError("All files in batch-cutmap must have the same dense z grid.")

        zmin = float(z_dense[0])
        zmax = float(z_dense[-1])

        if args.fixed == "z":
            requested = parse_z_value(args.z, zmin, zmax)
            i = nearest_index(z_dense, requested)
            fixed_value_actual = z_dense[i]
            cut_values = dense[i, :]
            variable_values = z_dense
        else:
            requested = parse_z_value(args.zprime, zmin, zmax)
            j = nearest_index(z_dense, requested)
            fixed_value_actual = z_dense[j]
            cut_values = dense[:, j]
            variable_values = z_dense

        selected = component_values(cut_values, args.component)
        values_qz.append(selected)
        q_values.append(q_value)

        for variable, value, selected_value in zip(variable_values, cut_values, selected):
            rows.append(
                (
                    q_value,
                    q_block.q_index,
                    q_block.q[0],
                    q_block.q[1],
                    q_block.q[2],
                    fixed_value_actual,
                    variable,
                    value.real,
                    value.imag,
                    abs(value),
                    np.angle(value),
                    selected_value,
                )
            )

    save_data = args.save_data
    if save_data is not None:
        np.savetxt(
            save_data,
            np.array(rows, dtype=float),
            header=(
                f"view={args.view} fixed={args.fixed} "
                "q_abs q_index qx qy qz fixed_coordinate variable_coordinate "
                "Re Im Abs Phase selected_component"
            ),
        )
        print(f"Wrote {save_data}")

    output = output_from_alias(args)
    if output is None:
        if args.fixed == "z":
            output = (
                f"batch_cutmap_fixed_z_{z_selector_label(args.z)}_G{g_index}_Gp{gp_index}_"
                f"{args.view}_{args.component}.png"
            )
            title_fixed = f"fixed z = {fixed_value_actual:.6g} Å"
            ylabel = "z' (Å)"
        else:
            output = (
                f"batch_cutmap_fixed_zprime_{z_selector_label(args.zprime)}_G{g_index}_Gp{gp_index}_"
                f"{args.view}_{args.component}.png"
            )
            title_fixed = f"fixed z' = {fixed_value_actual:.6g} Å"
            ylabel = "z (Å)"
    elif args.fixed == "z":
        title_fixed = f"fixed z = {fixed_value_actual:.6g} Å"
        ylabel = "z' (Å)"
    else:
        title_fixed = f"fixed z' = {fixed_value_actual:.6g} Å"
        ylabel = "z (Å)"

    plot_qz_heatmap(
        output,
        np.asarray(q_values, dtype=float),
        z_reference,
        np.asarray(values_qz, dtype=float),
        component_object_label(args.component, object_name, args.view),
        f"{object_name}\n{view_label(args.view)}, G index = {g_index}, G' index = {gp_index}, {title_fixed}",
        args.norm,
        ylabel=ylabel,
    )
    print(f"Wrote {output}")




def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot and inspect XATU q2d_chebyshev epsilon, inverse-epsilon, "
            "or generic Chebyshev operator files."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print q2d Chebyshev file metadata.")
    inspect_parser.add_argument("input", type=Path, help="XATU q2d_cheb operator file.")
    inspect_parser.set_defaults(func=command_inspect)

    heatmap_parser = subparsers.add_parser(
        "heatmap",
        help="Diagnostic fixed-q heatmap of an operator block O(q;z,z').",
    )
    heatmap_parser.add_argument("input", type=Path, help="XATU q2d_cheb operator file.")
    add_common_operator_arguments(heatmap_parser)
    add_heatmap_norm_argument(heatmap_parser)
    heatmap_parser.add_argument("--iq", type=int, default=None, help="q_index to select from a multi-q file.")
    heatmap_parser.add_argument("--output", default=None, help="Output heatmap image.")
    heatmap_parser.add_argument("--save-data", default=None, help="Optional dense matrix data file.")
    add_plot_alias(heatmap_parser)
    heatmap_parser.set_defaults(func=command_heatmap)

    cut_parser = subparsers.add_parser(
        "cut",
        help="Diagnostic fixed-q line cut of an operator block O(q;z,z').",
    )
    cut_parser.add_argument("input", type=Path, help="XATU q2d_cheb operator file.")
    add_common_operator_arguments(cut_parser)
    cut_parser.add_argument("--iq", type=int, default=None, help="q_index to select from a multi-q file.")
    cut_parser.add_argument("--z", required=True, help="Fixed z value, or bottom/center/top.")
    cut_parser.add_argument("--vary", choices=["z", "zprime"], default="zprime", help="Coordinate to vary.")
    cut_parser.add_argument("--output", default=None, help="Output cut plot image.")
    cut_parser.add_argument("--save-data", default=None, help="Optional cut data file.")
    add_plot_alias(cut_parser)
    cut_parser.set_defaults(func=command_cut)

    diag_parser = subparsers.add_parser(
        "diag",
        help="Diagnostic fixed-q plot of the dense-grid diagonal operator block O(q;z,z).",
    )
    diag_parser.add_argument("input", type=Path, help="XATU q2d_cheb operator file.")
    add_common_operator_arguments(diag_parser)
    diag_parser.add_argument("--iq", type=int, default=None, help="q_index to select from a multi-q file.")
    diag_parser.add_argument("--output", default=None, help="Output diagonal plot image.")
    diag_parser.add_argument("--save-data", default=None, help="Optional diagonal data file.")
    add_plot_alias(diag_parser)
    diag_parser.set_defaults(func=command_diag)

    batch_line_parser = subparsers.add_parser(
        "batch-line",
        help="Plot a selected operator value O(q;z,z') versus |q|.",
    )
    batch_line_parser.add_argument("--pattern", required=True, help="Glob pattern for q2d_cheb operator files.")
    add_common_operator_arguments(batch_line_parser)
    batch_line_parser.add_argument("--z", required=True, help="z value, or bottom/center/top.")
    batch_line_parser.add_argument("--zprime", required=True, help="z' value, or bottom/center/top.")
    batch_line_parser.add_argument("--qmax", type=float, default=None, help="Maximum |q| to include.")
    batch_line_parser.add_argument("--output", default=None, help="Output line plot image.")
    batch_line_parser.add_argument("--save-data", default=None, help="Optional line data file.")
    add_plot_alias(batch_line_parser)
    batch_line_parser.set_defaults(func=command_batch_line)

    batch_diagmap_parser = subparsers.add_parser(
        "batch-diagmap",
        help="Plot a q,z heatmap of the diagonal operator block O(q;z,z).",
    )
    batch_diagmap_parser.add_argument("--pattern", required=True, help="Glob pattern for q2d_cheb operator files.")
    add_common_operator_arguments(batch_diagmap_parser)
    add_heatmap_norm_argument(batch_diagmap_parser)
    batch_diagmap_parser.add_argument("--qmax", type=float, default=None, help="Maximum |q| to include.")
    batch_diagmap_parser.add_argument("--output", default=None, help="Output q,z diagonal heatmap image.")
    batch_diagmap_parser.add_argument("--save-data", default=None, help="Optional q,z diagonal data file.")
    add_plot_alias(batch_diagmap_parser)
    batch_diagmap_parser.set_defaults(func=command_batch_diagmap)

    batch_cutmap_parser = subparsers.add_parser(
        "batch-cutmap",
        help="Plot a q-z or q-z' heatmap of a fixed-coordinate cut O(q;z_fixed,z') or O(q;z,z'_fixed).",
    )
    batch_cutmap_parser.add_argument("--pattern", required=True, help="Glob pattern for q2d_cheb operator files.")
    add_common_operator_arguments(batch_cutmap_parser)
    add_heatmap_norm_argument(batch_cutmap_parser)
    batch_cutmap_parser.add_argument("--qmax", type=float, default=None, help="Maximum |q| to include.")
    batch_cutmap_parser.add_argument(
        "--fixed",
        choices=["z", "zprime"],
        required=True,
        help="Coordinate to hold fixed in the nonlocal operator block.",
    )
    batch_cutmap_parser.add_argument("--z", default="center", help="Fixed z value, or bottom/center/top.")
    batch_cutmap_parser.add_argument("--zprime", default="center", help="Fixed z' value, or bottom/center/top.")
    batch_cutmap_parser.add_argument("--output", default=None, help="Output q-coordinate cut heatmap image.")
    batch_cutmap_parser.add_argument("--save-data", default=None, help="Optional q-coordinate cut data file.")
    add_plot_alias(batch_cutmap_parser)
    batch_cutmap_parser.set_defaults(func=command_batch_cutmap)


    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
