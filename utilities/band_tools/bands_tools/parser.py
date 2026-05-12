import json
from typing import Any

import numpy as np


def load_band_blocks(filename: str) -> list[np.ndarray]:
    with open(filename, "r") as f:
        lines = [line.rstrip() for line in f]

    blocks = []
    current = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(np.loadtxt(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(np.loadtxt(current))

    if not blocks:
        raise ValueError(f"bands_file '{filename}' contains no data blocks.")

    for i, block in enumerate(blocks, start=1):
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(
                f"Each block in '{filename}' must have exactly two columns (k and energy). "
                f"Block {i} has shape {block.shape}."
            )

    return blocks


def _coerce_band_indices(raw_indices: Any, field_name: str) -> list[int]:
    if raw_indices is None:
        return []
    if isinstance(raw_indices, (int, np.integer)):
        raw_indices = [raw_indices]
    if not isinstance(raw_indices, (list, tuple, np.ndarray)):
        raise ValueError(f"{field_name} must be a list of band indices")

    indices = []
    for value in raw_indices:
        if isinstance(value, str):
            if not value.isdigit():
                raise ValueError(
                    f"{field_name} values must be integer strings or integers, got {value!r}"
                )
            value = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{field_name} values must be integer-like, got {value}")
            value = int(value)
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        else:
            raise ValueError(f"{field_name} values must be integers, got {type(value).__name__}")
        indices.append(value)

    return indices


def parse_red_bands(raw_red_bands: Any, n_bands: int) -> list[int]:
    return parse_band_indices(raw_red_bands, n_bands, "red_bands")


def parse_band_indices(raw_indices: Any, n_bands: int, field_name: str = "band indices") -> list[int]:
    indices = _coerce_band_indices(raw_indices, field_name)
    if not indices:
        return []

    zero_based = all(0 <= i < n_bands for i in indices)
    one_based = all(1 <= i <= n_bands for i in indices)

    if zero_based and not one_based:
        return indices
    if one_based and not zero_based:
        return [i - 1 for i in indices]
    if zero_based and one_based:
        return indices

    raise ValueError(
        f"{field_name} values must be within 0..{n_bands - 1} or 1..{n_bands} consistently. "
        f"Got {indices}"
    )


def load_json_config(filename: str) -> dict:
    with open(filename) as f:
        return json.load(f)


def write_example_json(filename: str = "example_bands.json") -> None:
    example = {
        "bands_file": "bands.dat",
        "Efermi": 0.0,
        "xlim": [0, 10],
        "ylim": [-5, 5],
        "xticks": {
            "locations": [0, 5, 10],
            "labels": ["$\\Gamma$", "X", "M"],
        },
        "red_bands": [9, 10],
        "effective_mass": {
            "enabled": False,
            "center": 0.0,
            "bands": [9, 10],
            "fit_window": 0.05,
            "carrier_type": "auto",
        },
    }
    with open(filename, "w") as f:
        json.dump(example, f, indent=4)
    print(f"Generated {filename}")
