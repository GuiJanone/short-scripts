from typing import Any

import numpy as np

from .parser import parse_band_indices


HBAR_J_S = 1.054571817e-34
ELECTRON_MASS_KG = 9.1093837015e-31
JOULE_PER_EV = 1.602176634e-19
METER_PER_ANGSTROM = 1.0e-10
VALID_CARRIER_TYPES = {"electron", "hole", "auto"}
CURVATURE_TOLERANCE = 1.0e-12


def _mass_prefactor() -> float:
    curvature_si_for_one_ev_a2 = JOULE_PER_EV * METER_PER_ANGSTROM**2
    return HBAR_J_S**2 / (curvature_si_for_one_ev_a2 * ELECTRON_MASS_KG)


def _require_number(config: dict, key: str) -> float:
    if key not in config:
        raise ValueError(f"effective_mass.{key} is required")
    value = config[key]
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"effective_mass.{key} must be a number")
    return float(value)


def compute_effective_masses(blocks: list[np.ndarray], config: dict, efermi: float = 0.0) -> list[dict]:
    if not isinstance(config, dict):
        raise ValueError("effective_mass must be a JSON object")

    center = _require_number(config, "center")
    fit_window = _require_number(config, "fit_window")
    if fit_window < 0:
        raise ValueError("effective_mass.fit_window must be non-negative")

    carrier_type = config.get("carrier_type", "auto")
    if carrier_type not in VALID_CARRIER_TYPES:
        raise ValueError(
            "effective_mass.carrier_type must be one of "
            f"{sorted(VALID_CARRIER_TYPES)}, got {carrier_type!r}"
        )

    band_indices = parse_band_indices(config.get("bands"), len(blocks), "effective_mass.bands")
    if not band_indices:
        raise ValueError("effective_mass.bands must contain at least one band index")

    results = []
    prefactor = _mass_prefactor()
    for band_index in band_indices:
        block = blocks[band_index]
        k_points = block[:, 0]
        shifted_energies = block[:, 1] - efermi
        mask = np.abs(k_points - center) <= fit_window
        if int(np.count_nonzero(mask)) < 3:
            raise ValueError(
                f"effective_mass band {band_index} has fewer than 3 points within "
                f"fit_window {fit_window} around center {center}"
            )

        selected_k = k_points[mask]
        k_min = float(np.min(selected_k))
        k_max = float(np.max(selected_k))

        x = selected_k - center
        y = shifted_energies[mask]
        c2, c1, c0 = np.polyfit(x, y, 2)
        curvature = float(2.0 * c2)

        resolved_type, mass = _classify_and_compute_mass(carrier_type, curvature, prefactor)
        results.append(
            {
                "band_index": band_index,
                "carrier_type": resolved_type,
                "mass": mass,
                "curvature": curvature,
                "coefficients": {
                    "c0": float(c0),
                    "c1": float(c1),
                    "c2": float(c2),
                },
                "n_points": int(np.count_nonzero(mask)),
                "k_min": k_min,
                "k_max": k_max,
                "center": center,
                "fit_window": fit_window,
            }
        )

    return results


def _classify_and_compute_mass(
    requested_type: str, curvature: float, prefactor: float
    ) -> tuple[str, float | None]:
    if abs(curvature) <= CURVATURE_TOLERANCE:
        if requested_type == "auto":
            return "flat_or_invalid", None
        raise ValueError(f"Cannot compute {requested_type} mass from near-zero curvature {curvature}")

    if requested_type == "auto":
        requested_type = "electron" if curvature > 0 else "hole"

    if requested_type == "electron":
        if curvature <= 0:
            raise ValueError(f"Electron effective mass requires positive curvature, got {curvature}")
        return requested_type, prefactor / curvature

    if requested_type == "hole":
        if curvature >= 0:
            raise ValueError(f"Hole effective mass requires negative curvature, got {curvature}")
        return requested_type, -prefactor / curvature

    raise ValueError(f"Invalid carrier_type {requested_type!r}")


def print_effective_mass_results(results: list[dict[str, Any]]) -> None:
    print("Effective mass results:")
    print("  Note: masses are fitted along the 1D plotted k-path, not a full tensor.")
    for result in results:
        mass = result["mass"]
        mass_text = "None" if mass is None else f"{mass:.6g}"
        print(
            f"  band {result['band_index']}: {result['carrier_type']}, "
            f"m*/m_e = {mass_text}, curvature = {result['curvature']:.6g}, "
            f"points = {result['n_points']}, "
            f"k_range = [{result['k_min']:.6g}, {result['k_max']:.6g}]"
        )
