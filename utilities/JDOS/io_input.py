# io_input.py
"""
Input file parser for the JDOS tool.

Format: JSON

Example minimal config:
{
  "tb_file": "my_tb.dat",
  "nk": [40, 40, 1],
  "eta": 0.02,
  "n_omega": 800,
  "n_occ": 24,
  "valence_bands": [0, -1],
  "conduction_bands": [1, 2]
}

Optional fields:
  "energy_range": [0.0, 5.0]
  "broadening": "gaussian" or "lorentzian"
  "gamma_centered": true/false
  "shift": [0.0, 0.0, 0.0]
  "no_plot": true/false
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union


@dataclass
class InputConfig:
    tb_file: str
    nk: Tuple[int, int, int]
    eta: float
    n_omega: int
    n_occ: int
    valence_bands: Union[int, list]
    conduction_bands: Union[int, list]
    energy_range: Optional[Tuple[float, float]] = None
    broadening: str = "gaussian"
    gamma_centered: bool = True
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    no_plot: bool = False
    gap_energy: float = None


def _as_tuple3_int(x, name):
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError(f"{name} must be a list of 3 integers")
    return int(x[0]), int(x[1]), int(x[2])


def _as_tuple2_float(x, name):
    if x is None:
        return None
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be a list of 2 floats")
    return float(x[0]), float(x[1])


def read_input(path: str | Path) -> InputConfig:
    path = Path(path)
    with open(path, "r") as f:
        cfg = json.load(f)

    required = ["tb_file", "nk", "eta", "n_omega", "n_occ",
                "valence_bands", "conduction_bands"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required field: {key}")

    tb_file = str(cfg["tb_file"])
    nk = _as_tuple3_int(cfg["nk"], "nk")
    eta = float(cfg["eta"])
    n_omega = int(cfg["n_omega"])
    n_occ = int(cfg["n_occ"])
    valence_bands = cfg["valence_bands"]
    conduction_bands = cfg["conduction_bands"]

    # Optionals
    energy_range = _as_tuple2_float(cfg.get("energy_range"), "energy_range")
    broadening = str(cfg.get("broadening", "gaussian")).lower()
    gamma_centered = bool(cfg.get("gamma_centered", True))
    shift = _as_tuple3_int(cfg.get("shift", [0, 0, 0]), "shift")
    shift = (float(shift[0]), float(shift[1]), float(shift[2]))
    no_plot = bool(cfg.get("no_plot", False))
    gap_energy = float(cfg["gap_energy"])

    return InputConfig(
        tb_file=tb_file,
        nk=nk,
        eta=eta,
        n_omega=n_omega,
        n_occ=n_occ,
        valence_bands=valence_bands,
        conduction_bands=conduction_bands,
        energy_range=energy_range,
        broadening=broadening,
        gamma_centered=gamma_centered,
        shift=shift,
        no_plot=no_plot,
        gap_energy=gap_energy,
    )


def map_bands_relative(n_occ: int,
                       m_size: int,
                       valence_sel: Union[int, list],
                       conduction_sel: Union[int, list]) -> Tuple[List[int], List[int]]:
    """
    Map user band selectors to absolute 0-based band indices.

    Rules:
      - n_occ is the number of occupied bands.
      - If valence_sel is an int N, take the top N filled bands:
          indices: n_occ-N ... n_occ-1
      - If valence_sel is a list, each entry must be 0 or negative:
          0 -> n_occ-1 (last filled)
         -1 -> n_occ-2 (second to last), etc.
      - If conduction_sel is an int N, take the lowest N empty bands:
          indices: n_occ ... n_occ+N-1
      - If conduction_sel is a list, each entry must be positive:
          1 -> n_occ (first CB), 2 -> n_occ+1, etc.

    Returns:
      (valence_indices, conduction_indices), each a sorted unique list.
    """
    if not (0 < n_occ <= m_size):
        raise ValueError("n_occ must be in 1..m_size")

    # Valence
    if isinstance(valence_sel, int):
        if valence_sel < 1:
            raise ValueError("valence_bands integer must be >= 1")
        v = list(range(n_occ - valence_sel, n_occ))
    elif isinstance(valence_sel, list):
        v = []
        for item in valence_sel:
            if not isinstance(item, int):
                raise ValueError("valence_bands list must contain integers")
            if item > 0:
                raise ValueError("valence_bands list uses 0,-1,-2,..., not positive numbers")
            idx = n_occ - 1 + item  # item=0 -> n_occ-1; item=-1 -> n_occ-2
            if idx < 0 or idx >= n_occ:
                raise ValueError(f"valence band selector {item} maps out of range")
            v.append(idx)
    else:
        raise TypeError("valence_bands must be int or list")

    # Conduction
    if isinstance(conduction_sel, int):
        if conduction_sel < 1:
            raise ValueError("conduction_bands integer must be >= 1")
        c = list(range(n_occ, min(n_occ + conduction_sel, m_size)))
    elif isinstance(conduction_sel, list):
        c = []
        for item in conduction_sel:
            if not isinstance(item, int):
                raise ValueError("conduction_bands list must contain integers")
            if item < 1:
                raise ValueError("conduction_bands list uses 1,2,3,...")
            idx = n_occ + (item - 1)  # item=1 -> n_occ; item=2 -> n_occ+1
            if idx < n_occ or idx >= m_size:
                raise ValueError(f"conduction band selector {item} maps out of range")
            c.append(idx)
    else:
        raise TypeError("conduction_bands must be int or list")

    # Deduplicate and sort for consistency
    v = sorted(set(v))
    c = sorted(set(c))
    return v, c
