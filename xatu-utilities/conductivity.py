#!/usr/bin/env python3
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt


COMPONENTS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
DEFAULT_COMPONENTS = ["xx", "yy", "zz"]
COMPONENT_COLORS = {
    "xx": "tab:red",
    "xy": "tab:purple",
    "xz": "tab:brown",
    "yx": "tab:pink",
    "yy": "tab:blue",
    "yz": "tab:cyan",
    "zx": "tab:gray",
    "zy": "tab:olive",
    "zz": "tab:green",
}
OSC_COMPONENTS = {
    "xx": ("x", 0),
    "yy": ("y", 1),
    "zz": ("z", 2),
}


def configure_matplotlib(fontsize=15, use_latex=True):
    """Set global matplotlib rcParams."""
    plt.rcParams.update({
        "text.usetex": bool(use_latex),
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "font.size": fontsize,
    })
    if use_latex:
        plt.rcParams["font.family"] = "serif"


def read_tensor_file(path):
    """
    Read a conductivity/tensor file where each row contains
    omega xx xy xz yx yy yz zx zy zz.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < len(COMPONENTS) + 1:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected at least "
            f"{len(COMPONENTS) + 1} columns: omega {' '.join(COMPONENTS)}"
        )

    energy = data[:, 0]
    comps = {comp: data[:, idx + 1] for idx, comp in enumerate(COMPONENTS)}
    return energy, comps


def read_exciton_file(path):
    """Backward-compatible helper returning only diagonal components."""
    energy, data = read_tensor_file(path)
    return energy, data["xx"], data["yy"], data["zz"]


def read_sp_file(path):
    """Backward-compatible helper returning only diagonal components."""
    _, data = read_tensor_file(path)
    return data["xx"], data["yy"], data["zz"]


def read_oscillator_file(path):
    """
    Read oscillator strengths.

    Expected columns:
      E_exc Re(Vx) Im(Vx) Re(Vy) Im(Vy) Re(Vz) Im(Vz)
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected at least 7 columns"
        )

    energy = data[:, 0]
    strengths = {
        "xx": np.abs(data[:, 1] + 1j * data[:, 2]) ** 2,
        "yy": np.abs(data[:, 3] + 1j * data[:, 4]) ** 2,
        "zz": np.abs(data[:, 5] + 1j * data[:, 6]) ** 2,
    }
    return energy, strengths


def parse_components(value):
    if value is None:
        return list(DEFAULT_COMPONENTS)

    components = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not components:
        raise argparse.ArgumentTypeError("component list cannot be empty")

    bad = [comp for comp in components if comp not in COMPONENTS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown component(s): {', '.join(bad)}. Allowed: {', '.join(COMPONENTS)}"
        )
    return components


def parse_limits(values, parser, option):
    if values is None:
        return None
    lo, hi = values
    if lo >= hi:
        parser.error(f"{option} limits must be in increasing order")
    return (lo, hi)


def print_header(mode, output_file, components):
    print("===================================")
    print("       PLOTTING CONDUCTIVITY       ")
    print("===================================")
    print(f"Mode: {mode}")
    print(f"Components: {', '.join(components)}")
    print(f"Output file: {output_file}")
    print()


def default_xlim(datasets, oscillator):
    ranges = []
    for dataset in datasets:
        if dataset is not None:
            energy = dataset["energy"]
            ranges.append((float(np.min(energy)), float(np.max(energy))))
    if oscillator is not None:
        energy = oscillator["energy"]
        ranges.append((float(np.min(energy)), float(np.max(energy))))
    if not ranges:
        return None
    return (min(item[0] for item in ranges), max(item[1] for item in ranges))


def plot_conductivity(
    exciton=None,
    ipa=None,
    oscillator=None,
    components=None,
    output="conductivity.png",
    xlim=None,
    ylim=None,
    osc_ylim=None,
    figsize=(6.4, 4.8),
    dpi=600,
    title=None,
    xlabel="E (eV)",
    ylabel=r"$\sigma$ ($e^2/\hbar$)",
    osc_threshold=0.1,
    osc_width=0.03,
    show=False,
):
    components = components or list(DEFAULT_COMPONENTS)
    datasets = [item for item in (exciton, ipa) if item is not None]
    if not datasets and oscillator is None:
        raise ValueError("nothing to plot: provide at least one of --ex, --ipa, or --osc")

    if oscillator is None:
        fig, axes = plt.subplots(figsize=figsize)
        ax_sigma = axes
        ax_osc = None
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_sigma, ax_osc = axes

    has_both_sigma = exciton is not None and ipa is not None
    for dataset in (exciton, ipa):
        if dataset is None:
            continue
        linestyle = "--" if dataset["kind"] == "IPA" and has_both_sigma else "-"
        alpha = 0.95 if dataset["kind"] == "IPA" else 0.8
        for comp in components:
            ax_sigma.plot(
                dataset["energy"],
                dataset["data"][comp],
                color=COMPONENT_COLORS[comp],
                linestyle=linestyle,
                linewidth=2.0,
                alpha=alpha,
                label=f"{dataset['label']}$_{{ {comp} }}$",
            )

    if ax_osc is not None:
        plotted = False
        for comp in components:
            if comp not in OSC_COMPONENTS:
                continue
            axis_label, _ = OSC_COMPONENTS[comp]
            strength = oscillator["strengths"][comp]
            values = oscillator["energy"] * strength
            mask = strength > osc_threshold
            if not np.any(mask):
                continue
            plotted = True
            ax_osc.bar(
                oscillator["energy"][mask],
                values[mask],
                width=osc_width,
                color=COMPONENT_COLORS[comp],
                alpha=0.35,
                label=f"V{axis_label}",
            )

        ax_osc.set_ylabel(r"$E|V^\alpha|^2$")
        ax_osc.set_xlabel(xlabel)
        if osc_ylim is not None:
            ax_osc.set_ylim(osc_ylim)
        if plotted:
            ax_osc.legend()

    ax_sigma.set_ylabel(ylabel)
    if ax_osc is None:
        ax_sigma.set_xlabel(xlabel)
    if title:
        ax_sigma.set_title(title)
    if ylim is not None:
        ax_sigma.set_ylim(ylim)

    resolved_xlim = xlim if xlim is not None else default_xlim(datasets, oscillator)
    if resolved_xlim is not None:
        ax_sigma.set_xlim(resolved_xlim)
        if ax_osc is not None:
            ax_osc.set_xlim(resolved_xlim)

    if datasets:
        ax_sigma.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_multi(energies, datas, labels, components,
               output="conductivity.png", xlim=None, ylim=None):
    """Backward-compatible multi-dataset plot helper."""
    datasets = [
        {"energy": energy, "data": data, "label": label, "kind": label}
        for energy, data, label in zip(energies, datas, labels)
    ]
    return _plot_arbitrary_datasets(datasets, components, output, xlim, ylim)


def _plot_arbitrary_datasets(datasets, components, output, xlim, ylim):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for dataset in datasets:
        for comp in components:
            ax.plot(
                dataset["energy"],
                dataset["data"][comp],
                color=COMPONENT_COLORS.get(comp),
                label=f"{dataset['label']}_{comp}",
            )
    ax.set_xlabel("E (eV)")
    ax.set_ylabel(r"$\sigma$ ($e^2/\hbar$)")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=600)
    return fig


def generate_example_json(path="example.json"):
    """Write a skeleton configuration file to *path*."""
    example = {
        "ex": "excitonic.dat",
        "ipa": "ipa.dat",
        "osc": "oscillator.dat",
        "components": ["xx", "yy", "zz"],
        "xlim": [0, 10],
        "ylim": [0, 5],
        "output": "conductivity.png",
        "show": False,
        "oscillator_threshold": 0.1,
    }
    with open(path, "w") as f:
        json.dump(example, f, indent=2)
    print(f"Example JSON written to {path}")


def run_from_json(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if "datasets" in cfg:
        return run_legacy_json(cfg)

    components = cfg.get("components", DEFAULT_COMPONENTS)
    components = parse_components(",".join(components) if isinstance(components, list) else components)
    output = cfg.get("output", "conductivity.png")
    exciton = load_sigma_dataset(cfg.get("ex"), "BSE", "BSE", cfg.get("ex_offset", 0.0))
    ipa = load_sigma_dataset(cfg.get("ipa"), "IPA", "IPA", cfg.get("ipa_offset", 0.0))
    oscillator = load_oscillator_dataset(cfg.get("osc"))

    print_header("json", output, components)
    configure_matplotlib(
        fontsize=int(cfg.get("font_size", 15)),
        use_latex=not bool(cfg.get("no_latex", False)),
    )
    plot_conductivity(
        exciton=exciton,
        ipa=ipa,
        oscillator=oscillator,
        components=components,
        output=output,
        xlim=tuple(cfg["xlim"]) if cfg.get("xlim") is not None else None,
        ylim=tuple(cfg["ylim"]) if cfg.get("ylim") is not None else None,
        osc_ylim=tuple(cfg["osc_ylim"]) if cfg.get("osc_ylim") is not None else None,
        figsize=tuple(cfg.get("figsize", [6.4, 4.8])),
        dpi=int(cfg.get("dpi", 600)),
        title=cfg.get("title"),
        xlabel=cfg.get("xlabel", "E (eV)"),
        ylabel=cfg.get("ylabel", r"$\sigma$ ($e^2/\hbar$)"),
        osc_threshold=float(cfg.get("oscillator_threshold", 0.1)),
        osc_width=float(cfg.get("oscillator_width", 0.03)),
        show=bool(cfg.get("show", False)),
    )


def run_legacy_json(cfg):
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError('configuration must contain a non-empty "datasets" list')
    components = cfg.get("components", DEFAULT_COMPONENTS)
    components = parse_components(",".join(components) if isinstance(components, list) else components)
    xlim = cfg.get("xlim", None)
    ylim = cfg.get("ylim", None)
    output = cfg.get("output", "conductivity.png")

    energies = []
    datas = []
    labels = []
    for ds in datasets:
        energy, data = read_tensor_file(ds["path"])
        energies.append(energy + float(ds.get("offset", 0.0)))
        datas.append(data)
        labels.append(ds.get("label", ds["path"]))

    print_header("legacy-json", output, components)
    configure_matplotlib()
    plot_multi(energies, datas, labels, components,
               output=output, xlim=xlim, ylim=ylim)


def load_sigma_dataset(path, label, kind, offset=0.0):
    if path is None:
        return None
    energy, data = read_tensor_file(path)
    return {
        "energy": energy + float(offset),
        "data": data,
        "label": label,
        "kind": kind,
    }


def load_oscillator_dataset(path):
    if path is None:
        return None
    energy, strengths = read_oscillator_file(path)
    return {"energy": energy, "strengths": strengths}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot excitonic, IPA, and oscillator conductivity data."
    )
    parser.add_argument(
        "legacy_files",
        nargs="*",
        help="Legacy positional mode: <excitonic_file> <ipa_file> [oscillator_file].",
    )
    parser.add_argument("--ex", help="Excitonic/BSE conductivity tensor file.")
    parser.add_argument("--ipa", help="Independent-particle conductivity tensor file.")
    parser.add_argument("--osc", help="Oscillator-strength file.")
    parser.add_argument(
        "--ij",
        type=parse_components,
        default=list(DEFAULT_COMPONENTS),
        help="Comma-separated tensor components to plot. Default: xx,yy,zz.",
    )
    parser.add_argument("--config", help="JSON configuration file.")
    parser.add_argument("--example", "--generate-json", nargs="?", const="example.json",
                        help="Write an example JSON config and exit.")

    parser.add_argument("-o", "--out", default=None, help="Output image path.")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Energy-axis limits.")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Conductivity-axis limits.")
    parser.add_argument("--osc-ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Oscillator subplot y limits.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--xlabel", default="E (eV)", help="X-axis label.")
    parser.add_argument("--ylabel", default=r"$\sigma$ ($e^2/\hbar$)",
                        help="Conductivity-axis label.")
    parser.add_argument("--figsize", nargs=2, type=float, default=(6.4, 4.8),
                        metavar=("W", "H"), help="Figure size in inches.")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI.")
    parser.add_argument("--font-size", type=int, default=15, help="Base font size.")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX text rendering.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    parser.add_argument("--osc-threshold", type=float, default=0.1,
                        help="Minimum oscillator strength to draw. Default: 0.1.")
    parser.add_argument("--osc-width", type=float, default=0.03,
                        help="Oscillator bar width in eV. Default: 0.03.")
    return parser


def resolve_args(args, parser):
    if args.example:
        generate_example_json(args.example)
        return None

    if args.config is None and len(args.legacy_files) == 1 and args.legacy_files[0].lower().endswith(".json"):
        args.config = args.legacy_files[0]
        args.legacy_files = []

    if args.config:
        if args.legacy_files or args.ex or args.ipa or args.osc:
            parser.error("--config cannot be combined with data files")
        run_from_json(args.config)
        return None

    if args.legacy_files:
        if args.ex or args.ipa or args.osc:
            parser.error("do not combine legacy positional files with --ex/--ipa/--osc")
        if len(args.legacy_files) not in (2, 3):
            parser.error("legacy positional mode expects <excitonic_file> <ipa_file> [oscillator_file]")
        args.ex = args.legacy_files[0]
        args.ipa = args.legacy_files[1]
        if len(args.legacy_files) == 3:
            args.osc = args.legacy_files[2]

    if not (args.ex or args.ipa or args.osc):
        parser.error("provide at least one of --ex, --ipa, --osc, or a JSON config")

    args.xlim = parse_limits(args.xlim, parser, "--xlim")
    args.ylim = parse_limits(args.ylim, parser, "--ylim")
    args.osc_ylim = parse_limits(args.osc_ylim, parser, "--osc-ylim")
    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args = resolve_args(args, parser)
    if args is None:
        return 0

    output = args.out
    if output is None:
        output = "conductivity_oscillators.png" if args.osc else "conductivity.png"

    configure_matplotlib(fontsize=args.font_size, use_latex=not args.no_latex)
    exciton = load_sigma_dataset(args.ex, "BSE", "BSE")
    ipa = load_sigma_dataset(args.ipa, "IPA", "IPA")
    oscillator = load_oscillator_dataset(args.osc)

    mode = "+".join(
        label for label, present in (
            ("BSE", exciton is not None),
            ("IPA", ipa is not None),
            ("osc", oscillator is not None),
        ) if present
    )
    print_header(mode, output, args.ij)
    plot_conductivity(
        exciton=exciton,
        ipa=ipa,
        oscillator=oscillator,
        components=args.ij,
        output=output,
        xlim=args.xlim,
        ylim=args.ylim,
        osc_ylim=args.osc_ylim,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        osc_threshold=args.osc_threshold,
        osc_width=args.osc_width,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
